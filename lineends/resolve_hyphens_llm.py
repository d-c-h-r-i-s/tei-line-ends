#!/usr/bin/env python3
"""
Hyphen Review Resolution via a local LLM
========================================

Settles the line-break word pairs that `validate_xml_hyphens.py` could not
decide from corpus statistics alone, and writes the answers into the decision
store (`data/csv/hyphen_decisions.csv`).

Only genuinely contested pairs reach this script — roughly 1,500 of them for
735 issues — and each distinct pair is asked exactly once, no matter how often
it occurs. Pairs already present in the store are skipped before any request is
made, so growing the corpus from 735 to ~2,700 issues only ever costs new
questions. Run it repeatedly and the review queue drains towards zero.

The model chooses between five readings of "LEFT- | RIGHT":

    join     the word simply continues        "Prin"  + "zessin" -> Prinzessin
    hyphen   a real compound hyphen           "Windisch-Graetz"
    period   the mark is a misread full stop  "nicht aus. Karl Schäfer wurde"
    comma    the mark is a misread comma      "fand statt, wo die Gastgeber"
    split    two separate words               "die Leiden der Bedauernswerten"
    skip     genuinely undecidable            left untouched everywhere

`period` matters more than its frequency suggests: it is how the abbreviated
titles this corpus is full of get restored — "Geh. Rat", "Se. Majestät",
"Prof. Dr.", "geb. Gräfin" — all of which feed the NER downstream.

The break mark itself carries no information and the prompt says so: the
Antiqua issues write `¬` for a continued word and `-` for a compound, the older
Fraktur ones write `=` for both, and the contexts the model is shown contain
whichever the file uses. Only the reading is asked for, so a decision holds for
either typesetting and the store stays free of notation.

Answers arrive as one JSON object per pair. Anything unparseable, or any form
outside the six above, is recorded as `skip` with the raw reply kept in the
note column, so a bad batch never corrupts the store.

A failure of the *server* is treated quite differently from a bad answer. The
model being unreachable, or gone, says nothing about the word pair, and writing
`skip` for it would permanently mark pairs nobody ever looked at as
unresolvable. Such a run stops instead, keeping the answers it did get, and can
simply be started again — the decision store makes it resume where it ended.
The model is checked to exist before the first question is asked.

Backend:
    Either Ollama or LM Studio, picked with `--backend` (default: ollama, same
    convention as the other local-model scripts in the project it came from). `--model` and `--url`
    fall back to a sensible default for whichever backend is chosen
    (`[llm.hyphen_resolver]` in config.toml). `--dry-run` prints the prompts
    without calling anything.

    The wire protocol, the model check and the retry policy live in
    `llm_backend.py`, shared with `extract_ocr_corrections.py` and
    `resolve_line_end_llm.py`; only the prompt and the form parsing are here.

Prompt:
    `data/prompts/prompt_hyphen_correction_v3.txt`, not a string literal in this
    file — a prompt is content, and revising one should show up as a diff of
    the text rather than of the code that sends it (every prompt in the project it
    came from lives there for the same reason). The file holds three `### <name>`
    sections: `system`, `user`, and `user_blind` for `--blind`; the two `user`
    ones are `str.format` templates and their placeholders are listed in the
    comment block at the head of the file. `--prompt other.txt` reads another
    one from `data/prompts/` (or from anywhere, given a path with a directory
    in it), which is how a reworded prompt is scored against this one over
    `--benchmark`. It is read at startup, so a missing section costs a second
    rather than the first question of a run that takes hours.

A run of well over a thousand questions takes hours, so its end — finished,
or given up — is pushed as a notification through `utils/notify.py` and nobody
has to sit and watch it. That needs `NTFY_CHANNEL` in `.secret.env`; without
it the run is simply silent.

Usage:
    python validate_xml_hyphens.py --export-review review.jsonl
    python resolve_hyphens_llm.py review.jsonl --model qwen2.5:14b
    python check_hyphen_consistency.py --patch hyphen_patch.csv
    python correct_xml_hyphens.py --dry-run

Flags:
    --backend NAME     'ollama' or 'lmstudio' (default: ollama).
    --model NAME       Model to query (default depends on --backend).
    --url URL          Server base URL (default depends on --backend).
    --decisions P      Decision store to update (default:
                       PATHS['csv']/hyphen_decisions.csv).
    --prompt NAME      Prompt file; a bare name is read from data/prompts/
                       (default: prompt_hyphen_correction_v3.txt).
    --limit N          Stop after N pairs (useful for a first sanity batch).
    --min-count N      Only resolve pairs with at least N occurrences.
    --temperature T    Sampling temperature (default: [llm].temperature in
                       config.toml).
    --benchmark        Score the model against a labelled set from
                       validate_xml_hyphens.py --export-benchmark instead of
                       resolving anything. Appends the scorecard to
                       output/hyphen_correction_eval.txt. Note that entries
                       recorded before 2026-07-29 were scored with a prompt
                       that only knew the Antiqua `¬`, so they compare with
                       later ones only loosely. Entries recorded before
                       2026-08-15 were drawn from a corpus that still
                       included the advertisements, whose OCR is bad enough
                       to distort every form, and do not compare with later
                       ones at all.
    --benchmark-out P  Where a --benchmark run writes its per-pair answers
                       (default: output/hyphen_correction_items_<date>_
                       <model>.jsonl). --no-benchmark-out suppresses it.
    --blind            Withhold the corpus frequencies, so the benchmark
                       measures German reading rather than table reading.
    --dry-run          Show the prompts, ask nothing, write nothing.
    --quiet            Suppress per-pair output.
    --no-notify        Do not push a notification when the run ends.

Author: Christian Lendl
Created: 2026-07-22
Last Modified: 2026-08-24
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS, LLM_TEMPERATURE
from hyphen_decisions import FORMS, Decision, load as load_decisions, merge
# The wire protocol, the model check and the retry policy are shared with
# extract_ocr_corrections.py and resolve_line_end_llm.py; only the prompt and
# the answer parsing below are specific to hyphen forms.
from llm_backend import (
    DEFAULT_BACKEND,
    BackendError,
    backend_defaults,
    call_llm,
    ensure_model,
    format_duration,
    load_prompt,
    parse_json_reply,
)
from utils.notify import notify

# Per-backend server URL and default model, the latter from config.toml
# ([llm.hyphen_resolver]); `--backend` picks which of them applies.
BACKEND_DEFAULTS = backend_defaults('hyphen_resolver')

# Names this script in the push notifications it sends
NOTIFY_TITLE = 'Salonblatt | Resolve Hyphens LLM'

# The prompts live in data/prompts/ rather than in this file, so they can be
# revised and diffed without touching the code that sends them - the same place
# and the same reason as every other prompt in the project it came from. The file is cut
# into `### <name>` sections; `--prompt` swaps in another one, which is how a
# reworded prompt is compared against this one over the benchmark.
PROMPT_FILE = 'prompt_hyphen_correction_v3.txt'
PROMPT_SECTIONS = ('system', 'user', 'user_blind')


def canonical_form(forms: Sequence[str]) -> str:
    """
    The single form to offer, and to record, when several spell alike.

    Only `split` can ever share a spelling with another form: `join` and
    `hyphen` differ by the hyphen, `period` and `comma` by the character. Of
    the two it is the one that writes nothing (`hyphen_decisions.LINE_ENDS`),
    which is what such a break needs — the character is in the text already —
    and the only safe one, `correct_xml_hyphens.py` guarding against a doubled
    full stop but not a doubled comma. Same rule and same reason as
    `resolve_break_context.canonical_form`.
    """
    return 'split' if 'split' in forms else forms[0]


def render_candidates(candidates: Dict[str, str]) -> str:
    """
    List the readings, one line per *distinct* spelling, one keyword each.

    When the left word already carries its full stop, "period" and "split"
    produce the same string; offering it twice invites a coin flip, so it is
    offered once, under `canonical_form`'s keyword.

    Naming both — `period/split`, as this did until 2026-09-01 — reads to the
    model as the keyword it was told to answer with, and `parse_answer` then
    discards the answer as an unknown form. That is what it cost the
    2026-08-30 `resolve_break_context.py` run 145 of 2,842 breaks.
    """
    by_spelling: Dict[str, List[str]] = {}
    for form in ('join', 'hyphen', 'period', 'comma', 'split'):
        spelling = candidates.get(form)
        if spelling:
            by_spelling.setdefault(spelling, []).append(form)
    return '\n'.join(f"  {canonical_form(forms):14s} -> {spelling}"
                     for spelling, forms in by_spelling.items())


def build_prompt(record: dict, prompts: Dict[str, str],
                 blind: bool = False) -> str:
    """
    Render the user prompt for one pair from the `user` section of `prompts`.

    With `blind` the corpus frequencies are withheld and the model has to
    decide from the sentences alone. That matters when scoring models against
    the benchmark: the frequencies are exactly what the statistics used to
    derive the expected answer, so leaving them in measures whether a model can
    read a table, while removing them measures whether it can read German.
    """
    candidates = record['candidates']
    evidence = record['evidence']
    contexts = '\n'.join(f'  {i}. {c}'
                         for i, c in enumerate(record['contexts'], 1))
    if blind:
        return prompts['user_blind'].format(
            left=record['left'],
            right=record['right'],
            candidates=render_candidates(candidates),
            contexts=contexts,
        )
    return prompts['user'].format(
        left=record['left'],
        right=record['right'],
        candidates=render_candidates(candidates),
        joined=candidates['join'],
        compound_freq=evidence['compound_freq'],
        bigram_freq=evidence['bigram_freq'],
        dashed_freq=evidence['dashed_freq'],
        period_freq=evidence.get('period_freq', 0),
        comma_freq=evidence.get('comma_freq', 0),
        contexts=contexts,
    )


def parse_answer(reply: str, candidates: Optional[Dict[str, str]] = None
                 ) -> tuple[str, str]:
    """
    Pull (form, reason) out of the model's reply.

    Anything that is not one of the known forms becomes 'skip', so a malformed
    answer leaves the source text untouched instead of corrupting it. What
    counts as unusable is `llm_backend.parse_json_reply`'s business (it also
    strips the `<think>` block reasoning models preface their answer with);
    what an unusable answer *means* is this store's, and here it means "leave
    this pair alone".

    A slash-joined label ("period/split") is the exception, and not a
    coercion: it is taken only when every form it names is real *and* all of
    them spell this pair the same way, which makes it one answer under two
    names. `candidates` is what settles that, so without it the label is a
    skip like any other unreadable reply.
    """
    data, error = parse_json_reply(reply)
    if data is None:
        return 'skip', error

    form = str(data.get('form', '')).strip().lower()
    reason = str(data.get('reason', '')).strip()[:200]
    # `context` is in FORMS but is not on offer: it says no corpus-wide answer
    # exists, which is a judgement about the *pair* that a model shown three
    # sentences cannot make and was not asked to. A model that produces it
    # anyway has answered a different question, and that is a skip like any
    # other unusable reply.
    if form in FORMS and form != 'context':
        return form, reason

    named = [part.strip() for part in form.split('/') if part.strip()]
    if (len(named) > 1 and candidates
            and all(part in FORMS and part != 'context' for part in named)):
        spellings = {candidates.get(part) for part in named}
        if len(spellings) == 1 and None not in spellings:
            return canonical_form(named), reason

    return 'skip', f'unknown form {form!r}: {reason}'


def load_queue(path: Path, decisions, min_count: int,
               limit: Optional[int]) -> List[dict]:
    """Read the review file and drop everything already settled."""
    queue: List[dict] = []
    skipped = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('affected', 1) < min_count:
                continue
            if (record['left'], record['right']) in decisions:
                skipped += 1
                continue
            queue.append(record)
    if skipped:
        print(f"Already settled, not asking again: {skipped:,} pairs")
    if limit:
        queue = queue[:limit]
    return queue


def run_benchmark(path: Path, args, prompts: Dict[str, str]) -> int:
    """
    Score a model against the labelled set from
    `validate_xml_hyphens.py --export-benchmark`.

    The scorecard is printed to the console and appended to
    `output/hyphen_correction_eval.txt`, so a run's result survives after the
    terminal is gone.

    Every answer is also written to a JSONL beside it, one line per pair, with
    the evidence the model saw and the reason it gave. The scorecard says how
    many `comma` pairs were missed; only this file says *which* ones, and a
    summary cannot be queried after the fact - the misses of a run whose
    terminal is closed used to be gone for good. `--benchmark-out` picks
    another path, `--no-benchmark-out` suppresses it.
    """
    records = [json.loads(line) for line in open(path, encoding='utf-8')
               if line.strip()]
    if args.limit:
        records = records[:args.limit]

    print(f"Benchmarking {args.model} on {len(records):,} labelled pairs"
          f"{' (blind - no frequency evidence)' if args.blind else ''}\n")

    started = time.time()
    hits = 0
    per_form: Dict[str, List[int]] = {}
    confusion: Dict[Tuple[str, str], int] = {}
    items: List[dict] = []
    for n, record in enumerate(records, 1):
        expected = record['expected']
        # A backend failure is not a wrong answer - let it stop the run rather
        # than score the model on questions it never received.
        reply = call_llm(prompts['system'],
                         build_prompt(record, prompts, args.blind),
                         args.model, args.url, args.temperature, args.backend)
        got, reason = parse_answer(reply, record['candidates'])

        ok = got == expected
        hits += ok
        # Kept whole rather than counted: the evidence and the model's own
        # reason are what an error analysis needs, and neither survives in
        # the scorecard.
        items.append({
            'left': record['left'], 'right': record['right'],
            'expected': expected, 'got': got, 'ok': ok,
            'reason': reason, 'expected_reason': record.get('expected_reason'),
            'evidence': record.get('evidence', {}),
        })
        per_form.setdefault(expected, [0, 0])
        per_form[expected][0] += ok
        per_form[expected][1] += 1
        confusion[(expected, got)] = confusion.get((expected, got), 0) + 1
        if not args.quiet:
            flag = '  ok' if ok else 'MISS'
            print(f"  [{n}/{len(records)}] {flag}  {record['left']} + "
                  f"{record['right']}: expected {expected}, got {got}")
            # Its own line, untruncated: a clipped reason once made a 404 look
            # like a malformed URL.
            if not ok and reason:
                print(f"           {reason}")

    elapsed = time.time() - started
    header = (f"{args.model}   accuracy {hits}/{len(records)} "
              f"({hits / max(len(records), 1):.1%})   "
              f"{format_duration(elapsed)}   {datetime.date.today()}")
    lines = ['=' * len(header), header, '=' * len(header)]
    # Which set, which prompt, sighted or blind: two runs of the same model
    # differ by these and by nothing the header shows, and a scorecard that
    # omits them cannot be read back a month later.
    lines.append(f"  benchmark {path.name}   prompt {args.prompt}   "
                 f"blind {'yes' if args.blind else 'no'}   "
                 f"temperature {args.temperature}")
    lines.append('')
    for form, (right, total) in sorted(per_form.items()):
        lines.append(f"  {form:8s} {right:3d}/{total:3d}  ({right / total:5.1%})")
    lines.append('')
    lines.append('  most common confusions (expected -> got):')
    wrong = sorted(((c, e, g) for (e, g), c in confusion.items() if e != g),
                   reverse=True)
    for count, exp, got in wrong[:6]:
        lines.append(f"    {count:3d}x  {exp} -> {got}")
    items_path = None
    if not args.no_benchmark_out:
        items_path = args.benchmark_out or (
            PATHS['output'] /
            (f"hyphen_correction_items_{datetime.date.today():%Y%m%d}_"
             f"{args.model.replace('/', '-').replace(':', '-')}"
             f"{'_blind' if args.blind else ''}.jsonl"))
        items_path.parent.mkdir(parents=True, exist_ok=True)
        with open(items_path, 'w', encoding='utf-8') as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + '\n')
        lines.append('')
        lines.append(f"  per-item answers: {items_path.name}")

    report = '\n'.join(lines)

    print('\n' + report)

    # Appended, not overwritten: a benchmark file is a running scorecard
    # across models and runs, not a single latest result.
    eval_path = PATHS['output'] / 'hyphen_correction_eval.txt'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    prior = eval_path.exists() and eval_path.stat().st_size > 0
    with open(eval_path, 'a', encoding='utf-8') as fh:
        if prior:
            fh.write('\n')
        fh.write(report + '\n')

    if not args.no_notify:
        notify(f"{args.model}: accuracy {hits}/{len(records)} "
               f"({hits / max(len(records), 1):.1%}) on the hyphen benchmark"
               f"{' (blind)' if args.blind else ''}, "
               f"{format_duration(elapsed)}.", NOTIFY_TITLE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Resolve contested line-break hyphens with a local LLM.')
    parser.add_argument('review', type=Path,
                        help='JSONL written by validate_xml_hyphens.py '
                             '--export-review (or --export-benchmark when '
                             '--benchmark is given)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Score against a labelled set instead of '
                             'resolving; writes nothing to the store')
    parser.add_argument('--blind', action='store_true',
                        help='Withhold the corpus frequencies so the model has '
                             'to decide from the sentences alone')
    parser.add_argument('--backend', choices=sorted(BACKEND_DEFAULTS),
                        default=DEFAULT_BACKEND,
                        help=f'Which server to talk to (default: '
                             f'{DEFAULT_BACKEND})')
    parser.add_argument('--model',
                        help='Model to query (default depends on --backend)')
    parser.add_argument('--url',
                        help='Server base URL (default depends on --backend)')
    parser.add_argument('--decisions', type=Path,
                        default=PATHS['csv'] / 'hyphen_decisions.csv')
    parser.add_argument('--prompt', default=PROMPT_FILE,
                        help=f'Prompt file, a bare name being read from '
                             f"{PATHS['prompts']} (default: {PROMPT_FILE})")
    parser.add_argument('--benchmark-out', type=Path,
                        help='Where --benchmark writes its per-pair answers '
                             '(default: output/hyphen_correction_items_'
                             '<date>_<model>.jsonl)')
    parser.add_argument('--no-benchmark-out', action='store_true',
                        help='Do not write the per-pair answers of a '
                             '--benchmark run')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=LLM_TEMPERATURE)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-notify', action='store_true',
                        help='Do not push a notification when the run ends')
    args = parser.parse_args()

    # Read before the model is checked: a mistyped section name should cost a
    # second, not the first question of a run that takes hours.
    prompts = load_prompt(args.prompt, PROMPT_SECTIONS)

    defaults = BACKEND_DEFAULTS[args.backend]
    args.model = args.model or defaults['model']
    args.url = args.url or defaults['url']

    # Fail before any work when the server is down or the model is absent.
    # Still worth a notification: this is the way a run started before leaving
    # the house turns out to have done nothing at all.
    if not args.dry_run:
        try:
            ensure_model(args.model, args.url, args.backend)
        except BackendError as exc:
            print(f"\n{exc}", file=sys.stderr)
            if not args.no_notify:
                notify(f"Did not start: {str(exc).splitlines()[0]}",
                       NOTIFY_TITLE)
            return 1

    if args.benchmark:
        try:
            return run_benchmark(args.review, args, prompts)
        except BackendError as exc:
            print(f"\nAborted: {exc}", file=sys.stderr)
            if not args.no_notify:
                notify(f"Benchmark of {args.model} aborted: "
                       f"{str(exc).splitlines()[0]}", NOTIFY_TITLE)
            return 1

    decisions = load_decisions(args.decisions)
    print(f"Decision store: {len(decisions):,} pairs already settled")

    queue = load_queue(args.review, decisions, args.min_count, args.limit)
    print(f"To resolve: {len(queue):,} pairs\n")
    if not queue:
        print("Nothing left to ask.")
        return 0

    if args.dry_run:
        for record in queue[:3]:
            print('=' * 62)
            print(build_prompt(record, prompts, args.blind))
        print('=' * 62)
        print(f"\nDRY RUN - {len(queue):,} pairs would be sent to "
              f"{args.model}, nothing written.")
        return 0

    started = time.time()
    resolved: List[Decision] = []
    counts: Dict[str, int] = {}
    answered = 0
    for n, record in enumerate(queue, 1):
        prompt = build_prompt(record, prompts)
        try:
            reply = call_llm(prompts['system'], prompt, args.model,
                             args.url, args.temperature, args.backend)
        except BackendError as exc:
            # The server, not the model. Recording these as `skip` would write
            # a permanent "leave this pair alone" for questions that were never
            # answered, so keep what was really decided and stop.
            merge(args.decisions, resolved, quiet=True)
            print(f"\nAborted at pair {n} of {len(queue):,}: {exc}",
                  file=sys.stderr)
            print(f"{answered:,} answered pairs were saved to "
                  f"{args.decisions}; rerun to continue where this left off.",
                  file=sys.stderr)
            if not args.no_notify:
                notify(f"Aborted at pair {n:,} of {len(queue):,} after "
                       f"{format_duration(time.time() - started)}: "
                       f"{str(exc).splitlines()[0]}\n"
                       f"{answered:,} answered pairs saved - rerun to "
                       f"continue.", NOTIFY_TITLE)
            return 1

        form, reason = parse_answer(reply, record['candidates'])
        answered += 1
        counts[form] = counts.get(form, 0) + 1
        resolved.append(Decision(
            left=record['left'], right=record['right'], form=form,
            source='llm', model=args.model, note=reason,
        ))
        if not args.quiet:
            print(f"  [{n}/{len(queue)}] {record['left']} + "
                  f"{record['right']} -> {form}  ({reason[:60]})")

        # Persist as we go: a long run that dies halfway keeps its answers.
        if n % 50 == 0:
            merge(args.decisions, resolved, quiet=True)
            resolved = []

    if resolved:
        merge(args.decisions, resolved, quiet=True)

    elapsed = format_duration(time.time() - started)
    settled = len(load_decisions(args.decisions))
    print("\n" + "=" * 62)
    print(f"Resolved {len(queue):,} pairs with {args.model} in {elapsed}")
    for form, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {form:8s} {count:6,}")
    print(f"\nDecision store now at {settled:,} pairs -> {args.decisions}")
    # Consistency first: this script asks about each pair once and in
    # isolation, so pairs that are really one decision can come back settled
    # two ways, and fixing that after the XML is written means correcting
    # files that have already been changed.
    print("Next: python check_hyphen_consistency.py --patch hyphen_patch.csv,"
          " then correct_xml_hyphens.py --dry-run")

    if not args.no_notify:
        breakdown = ', '.join(
            f'{form} {count:,}'
            for form, count in sorted(counts.items(), key=lambda kv: -kv[1]))
        notify(f"Resolved {len(queue):,} pairs with {args.model} in "
               f"{elapsed}.\n{breakdown}\nStore now at {settled:,} pairs.",
               NOTIFY_TITLE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
