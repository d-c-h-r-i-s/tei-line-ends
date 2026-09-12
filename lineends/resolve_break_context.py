#!/usr/bin/env python3
"""
Per-Occurrence Break Disambiguation
===================================

Settles the line breaks whose word pair has no answer that holds for the whole
corpus — "kaiser|in" is "Kaiserin Zita" in one column and "der Kaiser in Wien"
in the next — one occurrence at a time, and writes the verdicts to
`data/csv/hyphen_occurrences.csv`.

Everything upstream of this decides per *pair*. `validate_xml_hyphens.py`
already notices when a pair cannot be decided that way:

    if ev.separation >= 2 and not settled:
        ev.reason = 'contested-both-readings'
        return 'REVIEW'

and sends it to `resolve_hyphens_llm.py`, which answers it — globally, from
three sample contexts — and writes the answer into the pair store. `and not
settled` then makes the guard stop firing. So the pipeline detects the problem
correctly and then loses the finding: the review queue is the funnel that turns
"this pair is contested" into a corpus-wide verdict. This script is where that
finding goes instead.

Which pairs (`--promote`)
-------------------------
A pair is contested when the *minority* reading is attested often enough in
line-interior text — a position no break can have damaged — to be a live
reading. That is an absolute floor on the rarer reading, `--min-minority`, and
deliberately not a ratio:

    wie|der    wieder 5,347  /  wie der 131      ratio 41:1, contested
    an|deren   anderen 834   /  an deren  36     ratio 23:1, contested
    mit|der    mitder      6 /  mit der 3,215    ratio  1:536, noise

A ratio gate would drop the first two, and they are the whole problem: a
lopsided prior is what lets a global verdict steamroller the rare reading, not
a reason to think the rare reading is not there. The floor is what keeps OCR
noise out — "mitder" occurs 6 times in four million tokens and is not a
reading of anything.

At `--min-minority 10` this is 61 pairs and 1,978 breaks over the 787-issue
corpus, 0.24% of all breaks and 2.5 per issue. `--promote` drafts them as
`form=context` rows for `hyphen_decisions.csv`, which is that store saying it
declines to answer and the answer is here.

Which reading (`--score`, `--resolve`)
--------------------------------------
`--score` scores both readings of every contested break under a bigram model
built from the same line-interior lexicon, and reports how they fall. The
margin is evidence and never a verdict, because it fails in two ways no
threshold catches:

  * the corpus prior swamps a one-token window. On `wie|der` it answers "join"
    162 times out of 163 and is confidently wrong on "wie der vorangegangenen
    hl. Messe";
  * syntax the window cannot reach. On `Botschafter|in` — all 39 of them
    "Botschafter in Wien", "in Paris" — it answers "join" 15 times, because
    with `prev = k.` and `next = St.` there is no signal at all. That is the
    `kaiser|in` class, which is to say the class this stage exists for.

So nothing is gated on it. At 1,978 breaks the whole queue fits in one evening
of the local model, which is cheaper than the two silent error classes a gate
would buy. `--resolve` asks the model about one break with its own sentence in
front of it, carrying the margin as evidence rather than as an instruction.

Is it working (`--evaluate`)
----------------------------
The corpus supplies its own gold standard. Every line-interior "sowie" is a
labelled `join` example with context and every "so wie" a labelled `split`, in
the same magazine, register and orthography, at no annotation cost.
`--evaluate` holds them out and scores against them.

It reports **balanced accuracy and per-class recall, never raw accuracy.**
These classes run 40:1 and worse, so a majority-class predictor scores 95% on
`wie|der` while getting every interesting case wrong — raw accuracy would
certify precisely the failure mode above.

Usage:
    python resolve_break_context.py --promote
    python resolve_break_context.py --promote --apply
    python resolve_break_context.py --score
    python resolve_break_context.py --evaluate
    python resolve_break_context.py --export-review break_context.jsonl
    python resolve_break_context.py --resolve break_context.jsonl

Author: Christian Lendl
Created: 2026-08-29
"""

import argparse
import csv
import datetime
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LLM_TEMPERATURE, PATHS
import break_occurrences
import hyphen_decisions
from context_model import BIGRAM_WEIGHT, ContextModel  # noqa: F401
import validate_xml_hyphens as hyphens

# How many times the rarer reading must appear in line-interior text before the
# pair is treated as contested. See the module docstring for why this is a
# floor on the minority and not a ratio between the two.
DEFAULT_MIN_MINORITY = 10

# Tokens of context each side. One for the score, and that is not a budget but
# an arithmetic fact: in a bigram model every term beyond the token adjacent to
# the break is identical under both readings and cancels out of the margin.
# Measured over the free gold standard, widening this to 2 or 3 changes the
# balanced accuracy by nothing at all - not by a little, by 0.000. Using more
# context needs trigrams, which is a different lexicon and a separate question.
# The model gets the wider window, where it can actually read.
SCORE_WINDOW = 1
PROMPT_WINDOW = 3


@dataclass
class Break:
    """One contested line break, with everything needed to judge it."""
    file: str
    line_facs: str
    page: int
    left: str
    right: str
    raw_left: str
    raw_right: str
    hyphenated: bool
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.left, self.right)

    def context(self, window: int) -> str:
        """The sentence around the break, with the break itself marked."""
        before = ' '.join(self.before[-window:])
        after = ' '.join(self.after[:window])
        return f"{before} [{self.raw_left} | {self.raw_right}] {after}".strip()

    def candidates(self) -> Dict[str, str]:
        """The readings, spelled as they would appear in the joined text."""
        left, right = self.raw_left, self.raw_right
        bare = left.rstrip('.,')
        return {'join': f"{left}{right}", 'hyphen': f"{left}-{right}",
                'period': f"{bare}. {right}", 'comma': f"{bare}, {right}",
                'split': f"{left} {right}"}

def read_corpus(xml_folder: Path, max_files: Optional[int], quiet: bool,
                skip_subtypes=frozenset(hyphens.SKIPPED_SUBTYPES)):
    """
    Every paragraph in the corpus, with each line's `facs` id kept.

    The scope is `validate_xml_hyphens.SKIPPED_SUBTYPES`, shared with every
    other script that reads or writes the transcription. Any subtype the scope
    does not name is reported rather than quietly included: this pass builds
    the lexicon *and* the queue from the same read, so a region class nobody
    has classified would become evidence and a correction target in one step.
    """
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No XML files found in {xml_folder}")
    if not quiet:
        print(f"Reading {len(files)} XML files from {xml_folder} ...")

    paragraphs = []
    seen_subtypes = set()
    skip = frozenset(skip_subtypes)
    for number, path in enumerate(files, 1):
        text = path.read_text(encoding='utf-8')
        for _, page, subtype, lines in hyphens.read_paragraph_lines(
                text, skip_subtypes=skip):
            paragraphs.append((path.name, page, lines))
            seen_subtypes.add(subtype)
        if not quiet and number % 200 == 0:
            print(f"  {number}/{len(files)} files")

    unknown = hyphens.unknown_subtypes(seen_subtypes)
    if unknown:
        print(f"  WARNING: {', '.join(sorted(unknown))} is in scope but named "
              f"in neither CORRECTED_SUBTYPES nor SKIPPED_SUBTYPES "
              f"(validate_xml_hyphens.py) - classify it before trusting this "
              f"run")
    if not paragraphs:
        raise SystemExit(
            f"Read {len(files)} files from {xml_folder} and found no "
            f"paragraphs. Either --skip-subtypes removed everything, or "
            f"PARA_RE/LINE_RE in validate_xml_hyphens.py no longer match "
            f"what `<p>`/`<l>` look like.")
    return paragraphs


def contested_pairs(evidence: Dict[Tuple[str, str], object],
                    min_minority: int) -> Dict[Tuple[str, str], object]:
    """
    The pairs whose rarer reading is attested often enough to be real.

    `separation` counts a comma and a full stop alongside a plain space: all
    three part the two words, and which of them stands between is a separate
    question `separator_form` answers later.
    """
    return {key: ev for key, ev in evidence.items()
            if min(ev.join_count + ev.dash_count, ev.separation)
            >= min_minority}


def collect_breaks(paragraphs, wanted: set) -> List[Break]:
    """Every break in the corpus whose pair is in `wanted`."""
    breaks: List[Break] = []
    for filename, page, lines in paragraphs:
        for index in range(len(lines) - 1):
            (line_facs, current), (_, following) = lines[index], lines[index + 1]
            pair = hyphens.break_pair(current, following)
            if pair is None:
                continue
            left, right, raw_left, raw_right = pair
            if (left, right) not in wanted:
                continue
            head = current.split()
            tail = following.split()
            breaks.append(Break(
                file=filename, line_facs=line_facs, page=page,
                left=left, right=right,
                raw_left=raw_left, raw_right=raw_right,
                hyphenated=current.endswith(hyphens.HYPHEN_CHARS),
                before=[hyphens.strip_token(t) for t in head[:-1]],
                after=[hyphens.strip_token(t) for t in tail[1:]],
            ))
    return breaks


# ---------------------------------------------------------------------------
# --promote
# ---------------------------------------------------------------------------

def write_promotion(evidence, breaks: List[Break], path: Path,
                    decisions: Dict[Tuple[str, str], object]) -> List[list]:
    """
    Draft `form=context` rows for `hyphen_decisions.csv`.

    Drafted rather than written, by default, for the reason
    `check_hyphen_consistency.py --patch` drafts too: this changes what a
    settled pair means, and the store is a spreadsheet a person owns. It also
    sidesteps a rank problem that would otherwise be silent — `merge()`
    replaces only on strictly *higher* source rank, so a `manual` promotion of
    a pair a human had already settled by hand would be dropped without a
    word, and 22 of the 61 contested pairs are exactly that. Appending, which
    is what `--apply` does, wins the tie in `load()` instead.
    """
    counts = Counter(b.pair for b in breaks)
    today = datetime.date.today().isoformat()
    rows = []
    for key in sorted(counts, key=lambda k: -counts[k]):
        ev = evidence[key]
        settled = decisions.get(key)
        if settled is not None and settled.form == 'context':
            continue
        together = ev.join_count + ev.dash_count
        note = (f"both readings live in the corpus: joined {together}x, "
                f"apart {ev.separation}x over {counts[key]} breaks"
                + (f"; was {settled.form} ({settled.source})"
                   if settled else "; previously undecided"))
        rows.append([key[0], key[1], 'context', 'manual', '', today, note])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(hyphen_decisions.FIELDNAMES)
        writer.writerows(rows)
    print(f"\nPromotion drafted to: {path}")
    print(f"  {len(rows)} pairs, {sum(counts.values()):,} breaks between them")
    return rows


def apply_promotion(rows: List[list], store: Path) -> None:
    """
    Append the drafted rows to the decision store.

    Appended, never rewritten: the model's original verdict stays in the file
    as a record of what it said, and `load()` resolves the duplicate in favour
    of the later row at equal rank. The trailing newline is checked because
    the store has been written without one, and appending to that file joins
    two rows into one unparseable line.
    """
    text = store.read_text(encoding='utf-8')
    if not text.endswith('\n'):
        store.write_text(text + '\n', encoding='utf-8')
    with open(store, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh, lineterminator='\n').writerows(rows)
    print(f"  appended {len(rows)} rows to {store}")


# ---------------------------------------------------------------------------
# --score
# ---------------------------------------------------------------------------

def report_scores(breaks: List[Break], model: ContextModel,
                  decisions: Dict[Tuple[str, str], object]) -> None:
    """
    How the context model falls over the contested breaks.

    A diagnostic, not a verdict: nothing here is written to the occurrence
    store. What it is for is watching the two numbers that decide whether the
    stage is worth its cost — how much of the queue the model is confident
    about, and how often it disagrees with the pair verdict now in force.
    """
    for item in breaks:
        item.score = model.margin(item.before, item.left, item.right,
                                  item.after, SCORE_WINDOW)

    total = len(breaks)
    print(f"\n{'=' * 62}\nContext model over {total:,} contested breaks\n"
          f"{'=' * 62}")
    for gate in (1, 2, 3, 4, 5):
        confident = sum(1 for b in breaks if abs(b.score) >= gate)
        print(f"  |margin| >= {gate}   {confident:6,} "
              f"({confident / max(total, 1):5.1%}) confident, "
              f"{total - confident:6,} not")

    disagree = 0
    comparable = 0
    for item in breaks:
        settled = decisions.get(item.pair)
        if settled is None or settled.form in (None, 'skip', 'context'):
            continue
        comparable += 1
        stored_join = settled.form in ('join', 'hyphen')
        if (item.score > 0) != stored_join:
            disagree += 1
    if comparable:
        print(f"\n  disagrees with the pair verdict still in force on "
              f"{disagree:,} of {comparable:,} ({disagree / comparable:.0%})")
        print("  An upper bound on the disputed set, not a count of errors:")
        print("  the model over-joins where the corpus prior is lopsided.")

    print(f"\n{'Pair':>26} {'breaks':>7} {'joined':>7} {'apart':>7}")
    by_pair: Dict[Tuple[str, str], List[Break]] = defaultdict(list)
    for item in breaks:
        by_pair[item.pair].append(item)
    for key in sorted(by_pair, key=lambda k: -len(by_pair[k]))[:20]:
        group = by_pair[key]
        joined = sum(1 for b in group if b.score > 0)
        print(f"{key[0] + '|' + key[1]:>26} {len(group):7,} {joined:7,} "
              f"{len(group) - joined:7,}")


# ---------------------------------------------------------------------------
# --evaluate
# ---------------------------------------------------------------------------

def gold_from_interior(paragraphs, wanted: set, sample: int,
                       seed: int = 0) -> List[Tuple[Tuple[str, str], str,
                                                    List[str], List[str]]]:
    """
    A labelled test set, free, from the corpus itself.

    For each contested pair, every line-interior occurrence of the joined
    spelling is a `join` example and every interior occurrence of the two
    words adjacent is a `split` example — same magazine, same register, same
    historical orthography, and undamaged by construction, since a line break
    cannot reach a token that is neither first nor last on its line.

    The context is taken exactly as `collect_breaks` takes it at a real break,
    so the model is asked the same question in the same shape. What is missing
    is the break mark itself, which is the one signal this evidence cannot
    supply and therefore the one the scorer must not learn to want.
    """
    joined_form = {left + right: (left, right) for left, right in wanted}
    examples: Dict[Tuple[Tuple[str, str], str], list] = defaultdict(list)

    for _, _, lines in paragraphs:
        for _, line in lines:
            interior = [hyphens.strip_token(t) for t in line.split()[1:-1]]
            for index, token in enumerate(interior):
                key = joined_form.get(token)
                if key is not None:
                    examples[(key, 'join')].append(
                        (interior[:index], interior[index + 1:]))
            for index in range(len(interior) - 1):
                key = (interior[index], interior[index + 1])
                if key in wanted:
                    examples[(key, 'split')].append(
                        (interior[:index], interior[index + 2:]))

    rng = random.Random(seed)
    gold = []
    for (key, label), rows in examples.items():
        rng.shuffle(rows)
        for before, after in rows[:sample]:
            gold.append((key, label, before, after))
    return gold


def evaluate(gold, model: ContextModel) -> None:
    """
    Score the context model against the free gold standard.

    Balanced accuracy and per-class recall, never raw accuracy: these classes
    run 40:1 and worse, and on `wie|der` a predictor that answers "join" every
    time scores 97% while being wrong about every occurrence anyone cares
    about. Raw accuracy would certify the failure mode.
    """
    per_pair: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for key, label, before, after in gold:
        margin = model.margin(before, key[0], key[1], after, SCORE_WINDOW)
        predicted = 'join' if margin > 0 else 'split'
        per_pair[key][(label, predicted)] += 1

    print(f"\n{'=' * 78}\nContext model against held-out line-interior "
          f"occurrences ({len(gold):,})\n{'=' * 78}")
    print(f"{'pair':>24} {'join n':>7} {'rec':>6} {'split n':>8} {'rec':>6} "
          f"{'balanced':>9} {'majority':>9}")

    recalls: List[float] = []
    for key in sorted(per_pair, key=lambda k: -sum(per_pair[k].values())):
        counts = per_pair[key]
        n_join = counts[('join', 'join')] + counts[('join', 'split')]
        n_split = counts[('split', 'split')] + counts[('split', 'join')]
        if not (n_join and n_split):
            # One class absent: the pair supplies no evidence about telling
            # them apart, and a recall computed over the other alone would
            # read as a score.
            continue
        join_recall = counts[('join', 'join')] / n_join
        split_recall = counts[('split', 'split')] / n_split
        balanced = (join_recall + split_recall) / 2
        majority = max(n_join, n_split) / (n_join + n_split)
        recalls.append(balanced)
        print(f"{key[0] + '|' + key[1]:>24} {n_join:7,} {join_recall:6.2f} "
              f"{n_split:8,} {split_recall:6.2f} {balanced:9.2f} "
              f"{majority:9.2f}")

    if recalls:
        print(f"\n  {len(recalls)} pairs with both classes present")
        print(f"  mean balanced accuracy {sum(recalls) / len(recalls):.3f}")
        print("  'majority' is what answering the commoner reading every time "
              "would score.\n  A balanced accuracy near it is a model that "
              "has learned the prior, not the sentence.")


# ---------------------------------------------------------------------------
# --export-review / --resolve
# ---------------------------------------------------------------------------

def question_record(item: Break) -> dict:
    """One break as a question: the readings, the sentence, the evidence."""
    return {
        'file': item.file,
        'line_facs': item.line_facs,
        'page': item.page,
        'left': item.left,
        'right': item.right,
        'hyphenated': item.hyphenated,
        'candidates': item.candidates(),
        'context': item.context(PROMPT_WINDOW),
        'score': round(item.score, 2),
    }


def export_review(breaks: List[Break], path: Path) -> None:
    """
    Write the contested breaks as JSONL, one record per *occurrence*.

    The opposite of `validate_xml_hyphens.export_review_jsonl`, which writes
    one record per pair so the same question is never asked twice. Here the
    same question genuinely has different answers, and asking it once per
    occurrence is the whole point of the stage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for item in sorted(breaks, key=lambda b: (b.file, b.line_facs)):
            fh.write(json.dumps(question_record(item),
                                ensure_ascii=False) + '\n')
    print(f"Review set saved to: {path}  ({len(breaks):,} breaks)")


# Appended to the model's own reason when its answer named several forms that
# spell the break identically. The store has one `form` column and no room for
# "these two are the same answer here", so the note is where that is recorded.
MERGED_NOTE = ("[Antwort '{label}': identische Schreibweise, "
               "als {form} verbucht]")


def canonical_form(forms: Sequence[str]) -> str:
    """
    The single form to offer, and to record, when several spell alike.

    Only `split` can ever share a spelling with another form: `join` and
    `hyphen` differ by the hyphen, `period` and `comma` by the character. So
    the choice is always between `split` and the form naming a character the
    left word already carries, and `split` is the one that writes nothing
    (`hyphen_decisions.LINE_ENDS`) — which is what these breaks need, the
    character being in the text already.

    It is also the only safe choice. `correct_xml_hyphens.py` guards against
    doubling a full stop the line already ends with, but has no such guard for
    a comma, so recording `comma` on a hyphenated line that already ends in one
    would write `Trousseaux,,`.
    """
    return 'split' if 'split' in forms else forms[0]


def render_candidates(candidates: Dict[str, str]) -> str:
    """
    List the readings, one line per *distinct* spelling, one keyword each.

    Two forms can render identically — "period" and "split" when the left word
    already carries its full stop. Offering the spelling twice would invite a
    coin flip between two names for one answer, so it is offered once, under
    `canonical_form`'s keyword.

    Naming both, as this did until 2026-09-01, is worse than either: the prompt
    tells the model to answer with the keyword standing left of the arrow, so
    `period/split` *is* the keyword it is being shown, and `parse_form` then
    threw the answer away as an unknown form. That cost 145 of 2,842 breaks in
    the 2026-08-30 run. `parse_form` still accepts the joined label, for a
    model that produces one unprompted.
    """
    by_spelling: Dict[str, List[str]] = {}
    for form in break_occurrences.FORMS:
        spelling = candidates.get(form)
        if spelling:
            by_spelling.setdefault(spelling, []).append(form)
    return '\n'.join(f"  {canonical_form(forms):18s} -> {spelling}"
                     for spelling, forms in by_spelling.items())


def parse_form(reply: str, candidates: Optional[Dict[str, str]] = None
               ) -> Tuple[Optional[str], str]:
    """
    The form the model chose, or None with the reason it could not be read.

    A form outside the offered set is no answer. It is returned as None rather
    than coerced to anything, because the store's own convention is that a
    break nobody settled has no row, and inventing one here would put an
    unmade decision on the same footing as a made one.

    A slash-joined label ("period/split") is the one exception, and it is not a
    coercion: it is accepted only when every form it names is a real form *and*
    all of them spell this break the same way, which makes it one answer under
    two names rather than two answers. `candidates` is what settles that, so
    without it the label is unreadable like any other. What is recorded is
    `canonical_form`'s pick, with the label kept in the note.
    """
    from llm_backend import parse_json_reply

    data, error = parse_json_reply(reply)
    if data is None:
        return None, error
    form = str(data.get('form', '')).strip().lower()
    reason = str(data.get('reason', '')).strip()
    if form in break_occurrences.FORMS:
        return form, reason

    named = [part.strip() for part in form.split('/') if part.strip()]
    if (len(named) > 1 and candidates
            and all(part in break_occurrences.FORMS for part in named)):
        spellings = {candidates.get(part) for part in named}
        if len(spellings) == 1 and None not in spellings:
            chosen = canonical_form(named)
            note = MERGED_NOTE.format(label=form, form=chosen)
            return chosen, f"{reason} {note}".strip()

    return None, f'unknown form {form!r}: {reason}'


def resolve(queue_path: Path, store: Path, backend: str, model: str,
            base_url: str, temperature: float, prompt_file: str,
            shard: Optional[str], dry_run: bool,
            flush_every: int = 50) -> int:
    """
    Ask the local model about each break and record the answers.

    Breaks already settled by an `llm` or `manual` row are skipped before any
    request, so a rerun costs nothing and an interrupted run resumes where it
    stopped. A failure of the *server* stops the run rather than being written
    down as a verdict: an unreachable server says nothing about the break, and
    recording it as an answer would be the one error this store cannot detect
    later.
    """
    from llm_backend import (BackendError, call_llm, ensure_model,
                             format_duration, load_prompt)
    import time

    sections = load_prompt(prompt_file, ('system', 'user'))
    questions = [json.loads(line) for line in
                 queue_path.read_text(encoding='utf-8').splitlines() if line]

    if shard:
        index, count = (int(part) for part in shard.split('/'))
        questions = [q for n, q in enumerate(questions) if n % count == index - 1]
        print(f"Shard {shard}: {len(questions):,} of the queue")

    settled = break_occurrences.load(store)
    pending = [q for q in questions
               if settled.get((q['file'], q['line_facs'])) is None
               or settled[(q['file'], q['line_facs'])].source == 'rule']
    print(f"{len(questions):,} breaks in the queue, {len(pending):,} still "
          f"unanswered", flush=True)
    if not pending:
        return 0

    if not dry_run:
        ensure_model(model, base_url, backend)

    today = datetime.date.today().isoformat()
    answers: List[break_occurrences.Occurrence] = []
    started = time.time()
    for number, question in enumerate(pending, 1):
        user = sections['user'].format(
            left=question['left'], right=question['right'],
            candidates=render_candidates(question['candidates']),
            context=question['context'],
            score=f"{question['score']:+.2f}",
            hyphen='ja' if question['hyphenated'] else 'nein')
        if dry_run:
            print(f"\n--- {question['file']} {question['line_facs']} ---")
            print(user)
            continue
        try:
            reply = call_llm(sections['system'], user, model, base_url,
                             temperature, backend)
        except BackendError as error:
            print(f"\nServer failure after {number - 1} answers: {error}",
                  file=sys.stderr)
            break
        form, reason = parse_form(reply, question['candidates'])
        if form is None:
            print(f"  unusable reply for {question['line_facs']}: {reason}",
                  flush=True)
            continue
        answers.append(break_occurrences.Occurrence(
            file=question['file'], line_facs=question['line_facs'],
            left=question['left'], right=question['right'], form=form,
            score=f"{question['score']:+.2f}", source='llm', model=model,
            decided_on=today, note=reason))
        if len(answers) >= flush_every:
            break_occurrences.merge(store, answers, quiet=True)
            answers = []
        if number % 25 == 0:
            print(f"  {number}/{len(pending)}  "
                  f"({format_duration(time.time() - started)})", flush=True)

    if answers:
        break_occurrences.merge(store, answers)
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Settle contested line breaks one occurrence at a time.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--skip-subtypes',
                        default=','.join(hyphens.SKIPPED_SUBTYPES),
                        help='Comma-separated region subtypes to leave out of '
                             'the lexicon and the queue alike (default: '
                             f"{','.join(hyphens.SKIPPED_SUBTYPES)})")
    parser.add_argument('--decisions', type=Path,
                        default=PATHS['csv'] / 'hyphen_decisions.csv')
    parser.add_argument('--store', type=Path,
                        default=PATHS['csv'] / 'hyphen_occurrences.csv')
    parser.add_argument('--min-minority', type=int,
                        default=DEFAULT_MIN_MINORITY,
                        help='Occurrences the rarer reading needs in '
                             'line-interior text before the pair counts as '
                             f'contested (default: {DEFAULT_MIN_MINORITY})')
    parser.add_argument('--promote', action='store_true',
                        help='Draft `form=context` rows for the decision store')
    parser.add_argument('--apply', action='store_true',
                        help='Append the drafted rows instead of only writing '
                             'the draft')
    parser.add_argument('--score', action='store_true',
                        help='Report how the context model falls over the '
                             'contested breaks (a diagnostic; writes nothing)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Score the context model against held-out '
                             'line-interior occurrences')
    parser.add_argument('--eval-sample', type=int, default=200,
                        help='Held-out examples per pair and class '
                             '(default: 200)')
    parser.add_argument('--export-review',
                        help='Write the contested breaks as JSONL; a bare '
                             f"filename lands in {PATHS['output']}")
    parser.add_argument('--resolve', type=Path,
                        help='Run the local model over a queue written by '
                             '--export-review')
    parser.add_argument('--backend', default=None,
                        help='ollama or lmstudio')
    parser.add_argument('--model', default=None)
    parser.add_argument('--temperature', type=float, default=LLM_TEMPERATURE)
    parser.add_argument('--prompt', default='prompt_break_context.txt')
    parser.add_argument('--shard', help='K/N, to split a run over machines')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the prompts instead of sending them')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if args.resolve:
        from llm_backend import DEFAULT_BACKEND, backend_defaults
        defaults = backend_defaults('break_context')
        backend = args.backend or DEFAULT_BACKEND
        settings = defaults.get(backend, {})
        return resolve(args.resolve, args.store, backend,
                       args.model or settings.get('model', ''),
                       settings.get('url', ''), args.temperature,
                       args.prompt, args.shard, args.dry_run)

    if not (args.promote or args.score or args.evaluate
            or args.export_review):
        parser.error('nothing to do: pass --promote, --score, --evaluate, '
                     '--export-review or --resolve')

    skip_subtypes = frozenset(
        part.strip() for part in args.skip_subtypes.split(',') if part.strip())
    paragraphs = read_corpus(args.xml_folder, args.max_files, args.quiet,
                             skip_subtypes)
    if not args.quiet:
        print("Building lexicon from line-interior tokens ...")
    unigrams, bigrams, dashed, periods, commas = hyphens.build_lexicon(
        [(f, p, [text for _, text in lines]) for f, p, lines in paragraphs])
    model = ContextModel(unigrams, bigrams)

    # The pair table, built the way the analysis builds it, so `join_count`
    # and `separation` mean here exactly what they mean there.
    evidence = hyphens.collect_breaks(
        [(f, p, [text for _, text in lines]) for f, p, lines in paragraphs])
    for ev in evidence.values():
        ev.join_count = unigrams[ev.left + ev.right]
        ev.split_count = bigrams[(ev.left, ev.right)]
        ev.dash_count = dashed[(ev.left, ev.right)]
        ev.period_count = periods[(ev.left, ev.right)]
        ev.comma_count = commas[(ev.left, ev.right)]

    contested = contested_pairs(evidence, args.min_minority)
    wanted = set(contested)
    breaks = collect_breaks(paragraphs, wanted)
    # Per issue over the files actually read, not over the folder: with
    # --max-files the two differ by more than an order of magnitude and the
    # figure is there to be compared against a full run.
    issues = len({filename for filename, _, _ in paragraphs})
    print(f"\nContested pairs (minority >= {args.min_minority}): "
          f"{len(contested):,} pairs, {len(breaks):,} breaks over "
          f"{issues} issues ({len(breaks) / max(issues, 1):.1f} per issue)")

    decisions = hyphen_decisions.load(args.decisions)
    date_prefix = datetime.date.today().strftime('%Y%m%d')

    if args.promote:
        rows = write_promotion(
            evidence, breaks,
            PATHS['output'] / f'{date_prefix}_context_promotion.csv',
            decisions)
        if args.apply:
            apply_promotion(rows, args.decisions)
        else:
            print(f"  Review it, then re-run with --apply, or: "
                  f"tail -n +2 <file> >> {args.decisions}")

    if args.score:
        report_scores(breaks, model, decisions)

    if args.evaluate:
        gold = gold_from_interior(paragraphs, wanted, args.eval_sample)
        evaluate(gold, model)

    if args.export_review:
        for item in breaks:
            item.score = model.margin(item.before, item.left, item.right,
                                      item.after, SCORE_WINDOW)
        out = Path(args.export_review)
        export_review(breaks, out if out.parent != Path('.')
                      else PATHS['output'] / f'{date_prefix}_{out.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
