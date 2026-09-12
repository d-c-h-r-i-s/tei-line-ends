#!/usr/bin/env python3
"""
Line-End Character Resolution via a local LLM
=============================================

Settles the line-final tokens that `validate_line_end_chars.py` exported, and
writes the answers into the OCR correction store (`data/csv/ocr_corrections.csv`)
as wildcard rows, ready for `correct_xml_ocr.py`.

Two jobs, one question
----------------------
The export carries the statistical verdict with each token, and the model is
asked the same thing either way — "which spelling does this sentence want?" —
but the verdict decides what its answer is worth:

    REVIEW    the corpus could not settle it. Frequencies are the wrong tool
              here: "später" outnumbers "späten" many times over inside a line,
              yet "am späten Abend" is simply correct German. Only the syntax of
              the sentence decides, which is what the model is for.

    MISREAD   the corpus settled it, and the model is checking the answer. On
              this corpus that check is not a formality. The statistics propose
              `Reiss` -> `Reise`, `Zeiß` -> `Zeit` and `Mare` -> `Mary`, all of
              which are surnames or place names in a newspaper largely made of
              them; and they propose `Scht` -> `Schw`, where the token is
              *truncated* rather than misread and no substitution can repair it.
              Both kinds are obvious to a reader of the sentence.

An LLM verdict is written as `source=llm`, which outranks the `source=rule` row
the statistics wrote, so a confirmation leaves the store as it was and a
rejection overturns it. Nothing here can override a hand-made `manual` row.

Why the model is not simply asked to spell-check
------------------------------------------------
It is shown the candidate spellings and told to choose among them, never to
propose its own — and the transcribed spelling is always offered first, so
"leave it alone" is as easy to choose as any correction. The candidates come from
a single character substitution at the end of an actually observed token, which
keeps the model from modernising historical orthography, the one failure that
would quietly rewrite the source. An answer outside the candidate set is recorded
as "leave it alone".

Prompt
------
`data/prompts/prompt_ocr_correction.txt`, not a string literal in this file — a
prompt is content, and revising one should show up as a diff of the text rather
than of the code that sends it (every prompt in the project it came
from lives there for the same reason). The file holds a `### system` and a `### user` section, the latter
a `str.format` template whose placeholders are listed in the comment block at the
head of the file. The rules that keep proper names and historical spellings out
of the corrections live in that text, which is where they are worth arguing
about. `--prompt other.txt` reads another one from `data/prompts/` (or from
anywhere, given a path with a directory in it). It is read at startup, so a
missing section costs a second rather than the first question of a long run.

Resuming, sharding, failure
---------------------------
Tokens already settled by an `llm` or `manual` row are skipped before any request
is made, so a run can be stopped and restarted freely; a `rule` row is *not* a
reason to skip, being exactly what this script is here to confirm or overturn.
`--shard k/n` splits the queue over several machines. A server failure stops the
run rather than recording a verdict nobody gave, and the end of a run is pushed
through `utils/notify.py` so it does not have to be watched.

Usage:
    python validate_line_end_chars.py --export-review linechar_review.jsonl \
        --review-misread
    python resolve_line_end_llm.py output/$(date +%Y%m%d)_linechar_review.jsonl
    python correct_xml_ocr.py --dry-run

Flags:
    --corrections P    Correction store to update (default:
                       PATHS['csv']/ocr_corrections.csv).
    --prompt NAME      Prompt file; a bare name is read from data/prompts/
                       (default: prompt_ocr_correction.txt).
    --shard K/N        Take every Nth token starting at K, for running several
                       machines in parallel. Give each its own --corrections.
    --backend NAME     'ollama' or 'lmstudio' (default: ollama).
    --model NAME       Model to query (default: [llm.line_end_resolver] in
                       config.toml, per backend).
    --url URL          Server base URL (default depends on --backend).
    --limit N          Stop after N tokens (useful for a first sanity batch).
    --min-count N      Only resolve tokens seen at least N times at a line end.
    --temperature T    Sampling temperature (default: [llm].temperature).
    --dry-run          Show the prompts, ask nothing, write nothing.
    --quiet            Suppress per-token output.
    --no-notify        Do not push a notification when the run ends.

Author: Christian Lendl
Created: 2026-08-14
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS, LLM_TEMPERATURE
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
from ocr_corrections import (
    CONFIDENCE_LEVELS,
    WILDCARD,
    Correction,
    load as load_corrections,
    merge,
)
from utils.notify import notify

DEFAULT_CORRECTIONS = PATHS['csv'] / 'ocr_corrections.csv'

BACKEND_DEFAULTS = backend_defaults('line_end_resolver')

NOTIFY_TITLE = 'Salonblatt | Resolve Line-End Characters'

# The prompts live in data/prompts/ rather than in this file, so they can be
# revised and diffed without touching the code that sends them - the same place
# and the same reason as every other prompt in the project it came from. The file is cut
# into `### <name>` sections, and `--prompt` swaps in another one.
PROMPT_FILE = 'prompt_ocr_correction.txt'
PROMPT_SECTIONS = ('system', 'user')


def build_prompt(record: dict, prompts: Dict[str, str]) -> str:
    """Render the user prompt for one token from the `user` section."""
    observed = record['observed']
    # The observed spelling is always offered first: "leave it alone" has to be
    # as easy to choose as any correction, or the model invents errors.
    options = [observed] + [c for c in record['candidates'] if c != observed]
    # The export keys the transcribed form's own frequency as `observed_freq`
    # and every candidate under its own spelling, so the two are looked up
    # differently; showing them in one table is what makes the comparison
    # legible to the model.
    evidence = record.get('evidence', {})
    counts = {observed: evidence.get('observed_freq', 0)}
    counts.update({w: evidence.get(w, 0) for w in options if w != observed})
    return prompts['user'].format(
        observed=observed,
        candidates='\n'.join(f'  {w}' for w in options),
        evidence='\n'.join(f'  "{w}": {counts[w]}' for w in options),
        contexts='\n'.join(f'  {i}. ... {c}'
                           for i, c in enumerate(record['contexts'], 1)),
    )


def parse_answer(reply: str, record: dict) -> Tuple[str, str, str]:
    """
    Pull (word, confidence, reason) from the reply.

    A word outside the offered set is discarded in favour of the observed
    spelling — the model has either invented one or modernised the orthography,
    and both mean "leave the source alone".
    """
    observed = record['observed']
    data, error = parse_json_reply(reply)
    if data is None:
        return observed, 'low', error

    word = str(data.get('word', '')).strip()
    allowed = {observed, *record['candidates']}
    confidence = str(data.get('confidence', '')).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = 'low'
    reason = str(data.get('reason', '')).strip()[:200]
    if word not in allowed:
        return observed, 'low', f'answer {word!r} not among the candidates'
    return word, confidence, reason


def parse_shard(value: str) -> Tuple[int, int]:
    """Parse a '1/4' shard argument into (k, n), validating the range."""
    match = re.fullmatch(r'(\d+)\s*/\s*(\d+)', value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"expected K/N (e.g. 1/4), got {value!r}")
    k, n = int(match.group(1)), int(match.group(2))
    if n < 1 or not 1 <= k <= n:
        raise argparse.ArgumentTypeError(
            f"shard {value!r} out of range: need 1 <= K <= N")
    return k, n


def load_queue(path: Path, corrections, min_count: int,
               shard: Optional[Tuple[int, int]], limit: Optional[int]
               ) -> List[dict]:
    """
    Read the review file, dropping tokens an LLM or a human already settled.

    A `rule` row is *not* a reason to skip: those are exactly the statistical
    verdicts this script is here to confirm or overturn.
    """
    queue: List[dict] = []
    settled = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('seen_at_line_end', 1) < min_count:
                continue
            existing = corrections.get((record['observed'], WILDCARD))
            if existing is not None and existing.source in ('llm', 'manual'):
                settled += 1
                continue
            queue.append(record)
    if settled:
        print(f"Already settled, not asking again: {settled:,} tokens")
    if shard:
        k, n = shard
        queue = [record for i, record in enumerate(queue) if i % n == k - 1]
    if limit:
        queue = queue[:limit]
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Decide contested line-final spellings with a local LLM.')
    parser.add_argument('review', type=Path,
                        help='JSONL written by validate_line_end_chars.py '
                             '--export-review')
    parser.add_argument('--corrections', type=Path,
                        default=DEFAULT_CORRECTIONS)
    parser.add_argument('--prompt', default=PROMPT_FILE,
                        help=f'Prompt file, a bare name being read from '
                             f"{PATHS['prompts']} (default: {PROMPT_FILE})")
    parser.add_argument('--shard', type=parse_shard, metavar='K/N',
                        help='Take every Nth token starting at K, for running '
                             'several machines in parallel')
    parser.add_argument('--backend', choices=sorted(BACKEND_DEFAULTS),
                        default=DEFAULT_BACKEND)
    parser.add_argument('--model')
    parser.add_argument('--url')
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

    corrections = load_corrections(args.corrections)
    print(f"Correction store: {len(corrections):,} rows ({args.corrections})")

    queue = load_queue(args.review, corrections, args.min_count, args.shard,
                       args.limit)
    shard_note = (f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard
                  else '')
    print(f"To resolve: {len(queue):,} tokens{shard_note}\n")
    if not queue:
        print("Nothing left to ask.")
        return 0

    if args.dry_run:
        for record in queue[:3]:
            print('=' * 62)
            print(build_prompt(record, prompts))
        print('=' * 62)
        print(f"\nDRY RUN - {len(queue):,} tokens would be sent to "
              f"{args.model}, nothing written.")
        return 0

    # Fail before any work when the server is down or the model is absent.
    try:
        ensure_model(args.model, args.url, args.backend)
    except BackendError as exc:
        print(f"\n{exc}", file=sys.stderr)
        if not args.no_notify:
            notify(f"Did not start: {str(exc).splitlines()[0]}", NOTIFY_TITLE)
        return 1

    started = time.time()
    resolved: List[Correction] = []
    counts: Dict[str, int] = {'corrected': 0, 'kept': 0, 'overturned': 0}
    for n, record in enumerate(queue, 1):
        try:
            reply = call_llm(prompts['system'],
                             build_prompt(record, prompts), args.model,
                             args.url, args.temperature, args.backend)
        except BackendError as exc:
            # The server, not the model: keep what was really decided and stop,
            # rather than record answers nobody gave.
            merge(args.corrections, resolved, quiet=True)
            print(f"\nAborted at token {n} of {len(queue):,}: {exc}",
                  file=sys.stderr)
            print(f"{len(resolved):,} answers were saved to "
                  f"{args.corrections}; rerun to continue where this left "
                  f"off.", file=sys.stderr)
            if not args.no_notify:
                notify(f"Aborted at token {n:,} of {len(queue):,} after "
                       f"{format_duration(time.time() - started)}: "
                       f"{str(exc).splitlines()[0]}", NOTIFY_TITLE)
            return 1

        observed = record['observed']
        word, confidence, reason = parse_answer(reply, record)
        statistical = record.get('statistical_verdict', '')
        suggested = record.get('suggested', '')

        if word == observed:
            counts['kept'] += 1
            # Overturning a statistical MISREAD is the point of the check, so it
            # is written down rather than silently dropped: an `llm` row saying
            # "no change" outranks the `rule` row that said otherwise.
            if statistical == 'MISREAD':
                counts['overturned'] += 1
        else:
            counts['corrected'] += 1

        resolved.append(Correction(
            left=observed, right=WILDCARD,
            left_correct=word, right_correct=WILDCARD,
            confidence=confidence, source='llm', model=args.model,
            note=(f"line-end {statistical.lower() or 'review'}: {reason}"
                  if statistical != 'MISREAD' or word == observed
                  else f"confirms {suggested}: {reason}"),
        ))
        if not args.quiet:
            verb = ('keeps' if word == observed else f'-> {word}')
            flag = '  OVERTURNS' if (statistical == 'MISREAD'
                                     and word == observed) else ''
            print(f"  [{n}/{len(queue)}] {observed} {verb} "
                  f"[{confidence}]{flag}  ({reason[:45]})")

        # Persist as we go: a long run that dies halfway keeps its answers.
        if n % 50 == 0:
            merge(args.corrections, resolved, quiet=True)
            resolved = []

    if resolved:
        merge(args.corrections, resolved, quiet=True)

    elapsed = format_duration(time.time() - started)
    stored = load_corrections(args.corrections)
    print("\n" + "=" * 62)
    print(f"Resolved {len(queue):,} tokens with {args.model} in {elapsed}")
    print(f"  corrected            {counts['corrected']:,}")
    print(f"  left as transcribed  {counts['kept']:,}")
    print(f"    of which overturn a statistical MISREAD "
          f"{counts['overturned']:,}")
    print(f"\nCorrection store now at {len(stored):,} rows -> "
          f"{args.corrections}")
    print("Next: python correct_xml_ocr.py --dry-run")

    if not args.no_notify:
        notify(f"Resolved {len(queue):,} line-end tokens with {args.model} in "
               f"{elapsed}.\n{counts['corrected']:,} corrected, "
               f"{counts['overturned']:,} statistical verdicts overturned.",
               NOTIFY_TITLE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
