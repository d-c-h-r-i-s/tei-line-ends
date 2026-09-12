#!/usr/bin/env python3
"""
Per-Occurrence Line-End Character Correction
============================================

The class of line-end OCR error no frequency table can see: the misread
spelling is *itself* a common word. `validate_line_end_chars.py` says so in its
own docstring —

    "'den' and 'dem' misread as 'der' is exactly this: `der` is one of the
     commonest words in German, so a line ending in `der` looks perfectly
     normal however it got there. Only the syntax of the sentence settles it."

— and then, correctly, refuses to guess. This is the script that asks the
sentence.

See `docs/line-end-context-correction.md`. Read the next two sections before
using any of it, because the second one is the whole design.

Two switches, not one
---------------------
That class is behind `--ambiguous` in the validator, which turns 594 review
items into 82,836. It is also behind a second default nobody thinks of:

    if len(ev.word) < args.min_length:      # default 4
        ev.verdict, ev.reason = 'OK', 'too-short'

so `der`, `den` and `dem` — the docstring's own example — are removed before
`--ambiguous` is ever consulted. Splitting the class at that boundary:

    long   (len >= 4, judged today)      81,098   28.6%
    short  (len <  4, excluded)         202,707   71.4%
                                        -------
                                        283,805

The exclusion's reason is sound on its own terms ("short words are mostly
function words whose one-letter variants are other real words; judging them by
frequency invites false positives") and is exactly the objection a context
model answers: the short band is where the model is *strongest*, because a
function word's case is fixed by its neighbours and by nothing else.

Why nothing here ever corrects anything
---------------------------------------
Being right 89% of the time is nowhere near good enough when 99.5% of the
input is already correct. Simulated at a 0.5% error rate, plain argmax over
this model **breaks 4,705 correct tokens to fix 258** — 5% precision. That is a
base-rate effect and no better model fixes it; only a gate does, and a gate
trades away most of the recall.

So the roles here are the reverse of `resolve_break_context.py`. There the LLM
decides every case and the score is evidence, because there are only 2,842 of
them. Here the score decides *what gets asked* and the LLM decides the answer,
because 283,805 is not a queue anyone can afford. **Nothing in this script
writes a correction.** `--score` reports, `--evaluate` measures, `--calibrate`
fits, `--export-sample` and `--export-review` hand work to a person or a model.

What the hand check settled (271 rows, 2026-09-04)
--------------------------------------------------
The stratified sample came back 45 y / 226 n, and the errors are not spread
evenly - they are three populations the margin cannot separate, which is what
`gate_reason` now holds back:

  * **paragraph-final: 102 checked, 0 right.** Not "weaker" as the plan had it
    (0.818 against 0.891) - worthless. The clearest class is `ab -> am`, 25 of
    the 226 and never right: "stiegen im »Hotel Panhans« ab" is a separable
    verb whose stem is three to six tokens back and usually on an earlier line.
  * **no attested bigram on the right: 127 checked, 1 right (0.8%).** The
    substitution is on the token's *final letter*, and in German the final
    letter is what agrees with what follows - adjective to noun, determiner to
    noun. The left neighbour cannot discriminate an inflection, and the score
    is then a scaled unigram prior, which is the base-rate failure above
    wearing a margin.
  * **capitalised on both sides: 26 wrong against 4 right.** "Frederik" for
    "Frederic", "Marchese" for "Marchesa". Whether the scan says one or the
    other is a question about the scan; no amount of sentence settles it.

Together: 264 usable rows drop to 111, precision 17.0% -> 36.0%, and 40 of the
45 confirmed corrections survive. Over the corpus that is 3,598 proposals at
margin >= 6 down to 2,048.

Why the answer is not "more context"
------------------------------------
The obvious reading of the remaining 69 errors is that the window is too
narrow - "eine Abordnung ungarischer Pilger" needs the quantifier three tokens
back. It is the right diagnosis and the wrong fix, because `ContextModel` is a
*bigram* model: both readings occupy the same single slot, so every term beyond
the adjacent token is identical under both and cancels out of the margin. Re-
scoring all 264 checked rows at windows of 2, 3 and 5 changes 0 margins and 0
proposals. See `ContextModel.margin`, which says the same thing about the
sibling stage. More context means a higher-order model or the LLM, never a
window parameter - which is what the division of labour above already assumes.

Usage:
    python resolve_line_end_context.py --score
    python resolve_line_end_context.py --evaluate
    python resolve_line_end_context.py --calibrate
    python resolve_line_end_context.py --export-sample line_end_sample.csv

Author: Christian Lendl
Created: 2026-08-30
"""

import argparse
import csv
import datetime
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS
import validate_xml_hyphens as hyphens
from context_model import ContextModel
from validate_line_end_chars import ALPHABET

try:
    # Same optional import as the sibling exports: a deep link needs
    # data/csv/transkribus_filenames.csv, and an issue missing from it is a
    # gap in that table rather than a reason for the run to fail.
    from utils.get_transkribus_link import get_transkribus_link
except ImportError:  # pragma: no cover
    get_transkribus_link = None

# A token this common in line-interior position is a word in its own right, so
# a substitution cannot be told from a correct reading by frequency alone.
# Same constant and same value as `validate_line_end_chars.AMBIGUOUS_AT`; this
# is the class that script hands over rather than judging.
AMBIGUOUS_AT = 20

# How often a candidate spelling must occur inside a line before it counts as a
# reading at all. Below this it is a typo in the reference, not an alternative.
MIN_CANDIDATE = 20

# Shortest line-final token to consider. Two, not the validator's four - the
# whole point of this script is the band the four excludes (see the docstring).
# One is excluded rather than judged, and that is not the same argument: a lone
# letter at a line end is a middle initial ("Cary T. | Grayson"), and every
# letter of the alphabet is an attested "candidate" for it.
MIN_LENGTH = 2

# The boundary the validator draws, kept here only to report on both sides of
# it. Nothing in this script behaves differently across it.
VALIDATOR_MIN_LENGTH = 4


@dataclass
class LineEnd:
    """One line-final token, with everything needed to judge it."""
    file: str
    line_facs: str
    page: int
    subtype: str
    observed: str
    previous: Optional[str]
    following: Optional[str]
    line: str
    candidates: List[str] = field(default_factory=list)
    best: str = ''
    margin: float = 0.0
    # Whether the corpus has ever seen this token beside that neighbour, under
    # either reading. Set by `score_all`; read by `gate_reason`, where the
    # right-hand one is the single strongest signal in the whole stage.
    left_evidence: bool = False
    right_evidence: bool = False

    @property
    def band(self) -> str:
        return 'short' if len(self.observed) < VALIDATOR_MIN_LENGTH else 'long'

    @property
    def has_next(self) -> bool:
        """
        Whether the following line's first token was available.

        False on a paragraph's last line, where the continuation is in the next
        column or on the next page and no script in `lineends` can reach it yet -
        see "The paragraph-final gap" in the plan. Those are scored on
        `previous` alone, which is measurably weaker (0.818 against 0.891), so
        they are reported as their own band rather than mixed in.
        """
        return self.following is not None


def is_initial(word: str) -> bool:
    """A lone capital letter, i.e. somebody's middle initial."""
    return len(word) == 1 and word.isupper()


def candidates_for(word: str, interior: Counter,
                   min_candidate: int = MIN_CANDIDATE) -> List[str]:
    """
    The readings of this line-final token: itself, plus every attested
    one-character substitution of its final letter.

    `--confusions all` is the validator's default and is what is used here: the
    signature error is a final `n` read as `r`, but the same test over every
    final letter finds `l->n`, `s->e`, `e->t` and more behaving identically,
    and restricting the set by hand is how the first measurement of this model
    came to flatter it.
    """
    final = word[-1].lower()
    found = [word]
    for char in ALPHABET:
        if char == final:
            continue
        variant = word[:-1] + char
        if interior[variant] >= min_candidate:
            found.append(variant)
    return found


def collect(xml_folder: Path, max_files: Optional[int], skip_subtypes,
            quiet: bool) -> Tuple[Counter, Counter, List[LineEnd]]:
    """
    Read the corpus once: the interior lexicon, its bigrams, and every
    judgeable line end.

    The lexicon is delegated to `validate_xml_hyphens.build_lexicon`, on the
    paragraphs collected here, and that is deliberate. Counting the tokens in
    this function would have been three lines shorter and would have produced a
    *different* table: `build_lexicon` sorts an adjacent pair into `bigrams`,
    `periods` or `commas` by the punctuation between them, so a plain inline
    count silently folds "aus. Karl" in with "aus Karl". `resolve_break_context.py`
    scores against `build_lexicon`'s definition, and two scripts claiming the
    same reference while counting differently is the failure this codebase has
    already been bitten by once (see `test_and_debug/test_xml_readers.py`).
    One definition, one call.

    A line ending in a break mark is skipped as a *line end* - it holds a
    fragment of a word continuing on the next line, whose final character
    legitimately belongs mid-word - but its interior tokens still reach the
    lexicon like any other line's.

    """
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No XML files found in {xml_folder}")
    if not quiet:
        print(f"Reading {len(files)} XML files from {xml_folder} ...")

    paragraphs: List[Tuple[str, int, List[str]]] = []
    raw: List[LineEnd] = []
    seen_subtypes = set()

    for number, path in enumerate(files, 1):
        text = path.read_text(encoding='utf-8')
        for _, page, subtype, lines in hyphens.read_paragraph_lines(
                text, min_lines=1, skip_subtypes=skip_subtypes):
            seen_subtypes.add(subtype)
            paragraphs.append((path.name, page, [text for _, text in lines]))
            for index, (line_facs, line) in enumerate(lines):
                tokens = line.split()
                if not tokens:
                    continue
                if line.endswith(hyphens.HYPHEN_CHARS):
                    continue
                word = hyphens.strip_token(tokens[-1])
                if not hyphens.WORD_RE.fullmatch(word):
                    continue
                previous = (hyphens.strip_token(tokens[-2])
                            if len(tokens) >= 2 else None)
                following = None
                if index + 1 < len(lines):
                    tail = lines[index + 1][1].split()
                    if tail:
                        following = hyphens.strip_token(tail[0])
                raw.append(LineEnd(file=path.name, line_facs=line_facs,
                                   page=page, subtype=subtype, observed=word,
                                   previous=previous, following=following,
                                   line=line))
        if not quiet and number % 200 == 0:
            print(f"  {number}/{len(files)} files")

    interior, bigrams, _, _, _ = hyphens.build_lexicon(paragraphs)

    unknown = hyphens.unknown_subtypes(seen_subtypes)
    if unknown:
        print(f"  WARNING: {', '.join(sorted(unknown))} is in scope but named "
              f"in neither CORRECTED_SUBTYPES nor SKIPPED_SUBTYPES - classify "
              f"it before trusting this run")
    return interior, bigrams, raw


def ambiguous(raw: List[LineEnd], interior: Counter, min_length: int,
              min_candidate: int) -> List[LineEnd]:
    """
    Keep the line ends this script is for: the observed spelling is itself a
    word, and at least one substitution of its final letter is also a word.

    Everything else is either `validate_line_end_chars.py`'s job (the observed
    spelling is unattested, so frequency settles it) or nobody's.
    """
    kept = []
    for item in raw:
        if len(item.observed) < min_length or is_initial(item.observed):
            continue
        if interior[item.observed] < AMBIGUOUS_AT:
            continue
        found = candidates_for(item.observed, interior, min_candidate)
        if len(found) < 2:
            continue
        item.candidates = found
        kept.append(item)
    return kept


def score_all(items: List[LineEnd], model: ContextModel) -> None:
    """
    Attach the best-scoring reading, its margin, and where the evidence is.

    Both readings are one token, so unlike the join/split question there is no
    length term to correct for - the sequences being compared are the same
    shape and differ in one slot. That is also why the window is one token and
    widening it is a no-op: see "Why the answer is not more context" above.

    The two evidence flags are not part of the score. They record whether the
    bigram table has anything to say about this token beside that neighbour
    *under either reading* - because when it does not, the term degenerates to
    a scaled unigram and the margin is reporting the corpus prior rather than
    the sentence. `gate_reason` is where that matters.
    """
    for item in items:
        context = ([item.previous] if item.previous else [],
                   [item.following] if item.following else [])
        def total(word: str) -> float:
            return model.score(context[0] + [word] + context[1])
        observed = total(item.observed)
        item.best = max(item.candidates, key=total)
        item.margin = total(item.best) - observed
        item.left_evidence = item.previous is not None and bool(
            model.bigrams[(item.previous, item.observed)]
            + model.bigrams[(item.previous, item.best)])
        item.right_evidence = item.following is not None and bool(
            model.bigrams[(item.observed, item.following)]
            + model.bigrams[(item.best, item.following)])


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------

# In the order `gate_reason` tests them, which is deliberate: each implies the
# next is untestable rather than passed, so a proposal is reported under the
# first thing wrong with it and the counts add up to the population.
GATE_REASONS = ('paragraph-final', 'no-right-bigram', 'capitalised-pair')


def gate_reason(item: LineEnd) -> Optional[str]:
    """
    Why this proposal is not fit to hand to a person or a model, or None.

    Three populations the 2026-09-04 hand check found the margin cannot
    separate - see the module docstring for the counts. None of them is a
    threshold: they are questions this stage is not equipped to answer, and a
    higher margin makes a wrong answer to them more confident, not less.

    `paragraph-final` is the one that may lift on its own. The continuation is
    in the next column or on the next page, and once `merge_factoids.py` has
    written the `@next` chain the following token becomes reachable; these
    items would then be scored like any other rather than on `previous` alone.
    """
    if not item.has_next:
        return 'paragraph-final'
    if not item.right_evidence:
        return 'no-right-bigram'
    if item.observed[:1].isupper() and item.best[:1].isupper():
        return 'capitalised-pair'
    return None


def proposals(items: List[LineEnd], gate: float,
              ungated: bool = False) -> List[LineEnd]:
    """Every scored item this stage is willing to put in front of somebody."""
    changed = [i for i in items if i.best != i.observed and i.margin >= gate]
    return changed if ungated else [i for i in changed if gate_reason(i) is None]


# ---------------------------------------------------------------------------
# --score
# ---------------------------------------------------------------------------

GATES = (2, 4, 6, 8, 10, 12, 16)


def report_scores(items: List[LineEnd]) -> None:
    """
    How the context model falls over the ambiguous line ends.

    A diagnostic. Nothing here is written anywhere, and the margin is not a
    verdict at any value - see the module docstring. What this is for is the
    only two numbers that decide whether the stage is worth building: how big
    the queue is at each gate, and how it splits across the bands where the
    model is known to behave differently.
    """
    total = len(items)
    print(f"\n{'=' * 70}\nContext model over {total:,} ambiguous line ends\n"
          f"{'=' * 70}")

    bands = [('long', lambda i: i.band == 'long'),
             ('short', lambda i: i.band == 'short'),
             ('no next', lambda i: not i.has_next)]
    counts = {name: sum(1 for i in items if test(i)) for name, test in bands}
    print("  population:  " + "   ".join(
        f"{name} {counts[name]:,} ({counts[name] / max(total, 1):.1%})"
        for name, _ in bands))
    print("  ('no next' is a paragraph's last line and overlaps the other two:"
          "\n   it is scored on `previous` alone, which is weaker - see the "
          "plan)")

    changed = [i for i in items if i.best != i.observed]
    print(f"\n{'margin':>7} {'flagged':>9} {'share':>8}   "
          + "  ".join(f"{name:>9}" for name, _ in bands))
    for gate in GATES:
        over = [i for i in changed if i.margin >= gate]
        row = "  ".join(f"{sum(1 for i in over if test(i)):>9,}"
                        for _, test in bands)
        print(f"{gate:7} {len(over):9,} {len(over) / max(total, 1):8.2%}   "
              f"{row}")

    print(f"\n  top substitutions proposed at margin >= 8 "
          f"(observed -> best, by count):")
    subs = Counter((i.observed[-1], i.best[-1])
                   for i in changed if i.margin >= 8
                   and len(i.observed) == len(i.best))
    for (a, b), n in subs.most_common(10):
        print(f"    {a} -> {b}   {n:,}")

    report_gates(items)


def report_gates(items: List[LineEnd]) -> None:
    """
    What `gate_reason` holds back, and what the hand check priced it at.

    The precisions are quoted rather than computed because they belong to one
    labelled sample and not to this run: recomputing them would mean carrying
    the verdict CSV around, and reprinting them beside live counts is what
    makes the counts mean anything.
    """
    priced = {'paragraph-final': '102 checked, 0 right',
              'no-right-bigram': '127 checked, 1 right (0.8%)',
              'capitalised-pair': '30 checked, 4 right (13%)'}
    print(f"\n  held back by the gates (of the {len(proposals(items, 6, True)):,} "
          f"proposals at margin >= 6):")
    held = Counter(gate_reason(i) for i in proposals(items, 6, True))
    for reason in GATE_REASONS:
        print(f"    {reason:18s} {held[reason]:6,}   ({priced[reason]})")
    kept = proposals(items, 6)
    print(f"    {'kept':18s} {len(kept):6,}   "
          f"(111 checked, 40 right (36.0%), against 17.0% ungated)")

    # Not a gate, and reported so the choice stays visible: adding it takes the
    # checked sample from 111 rows at 36.0% to 86 at 44.2%, for two of the 40
    # confirmed corrections. Whether that trade is worth making is a decision
    # about the queue's size, which this script does not get to make.
    weak = sum(1 for i in kept if not i.left_evidence)
    print(f"\n  of those kept, {weak:,} have no attested bigram on the *left* "
          f"either\n    (36 checked, 3 right (8.3%) - a fourth gate, not "
          f"applied)")


# ---------------------------------------------------------------------------
# --evaluate
# ---------------------------------------------------------------------------

def gold_from_interior(items: List[LineEnd], interior: Counter,
                       xml_folder: Path, max_files: Optional[int],
                       skip_subtypes, sample: int, seed: int = 0):
    """
    A labelled test set, free, from the corpus itself.

    Every line-*interior* occurrence of an ambiguous token is an example whose
    answer is known: the token is what it is, in a position no line break can
    have damaged. Take its neighbours exactly as `collect` takes them at a real
    line end and the model is asked the same question in the same shape.

    What this cannot measure is the other half of the problem. An interior
    token carries no OCR error, so this scores `P(context | candidate)` and
    says nothing about `P(observed | candidate)` - the channel. It also gives
    the model an undamaged `next`, where a real line end's `next` is itself a
    line-initial token. Both make the figure optimistic; only a hand-checked
    sample can price them.
    """
    wanted = {item.observed for item in items}
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]

    pool = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        for _, _, _, lines in hyphens.read_paragraph_lines(
                text, min_lines=1, skip_subtypes=skip_subtypes):
            for _, line in lines:
                tokens = [hyphens.strip_token(t) for t in line.split()]
                for index in range(1, len(tokens) - 1):
                    if tokens[index] in wanted:
                        pool.append((tokens[index], tokens[index - 1],
                                     tokens[index + 1]))
    random.Random(seed).shuffle(pool)
    return pool[:sample]


def evaluate(gold, interior: Counter, model: ContextModel,
             min_candidate: int) -> None:
    """
    Score the model against the free gold standard.

    Two numbers, and the second is the one that matters. Raw accuracy at the
    corpus's natural frequencies is what the system would actually deliver;
    the per-set balanced view underneath shows whether it is reading the
    sentence or the prior, which raw accuracy on a 40:1 class cannot.
    """
    correct = 0
    baseline = 0
    per_set: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for truth, previous, following in gold:
        found = candidates_for(truth, interior, min_candidate)

        def total(word: str) -> float:
            return model.score([previous, word, following])
        predicted = max(found, key=total)
        commonest = max(found, key=lambda c: interior[c])
        correct += predicted == truth
        baseline += commonest == truth
        per_set[tuple(sorted(found))][(truth, predicted)] += 1

    n = max(len(gold), 1)
    print(f"\n{'=' * 70}\nHeld-out line-interior occurrences ({len(gold):,})\n"
          f"{'=' * 70}")
    print(f"  context model                     {correct / n:.3f}")
    print(f"  pick the commonest candidate      {baseline / n:.3f}")

    print(f"\n{'candidates':>34} {'n':>7} {'balanced':>9} {'majority':>9}")
    rows = []
    for members, counts in per_set.items():
        seen = Counter(truth for truth, _ in counts.elements())
        present = [m for m in members if seen[m]]
        if len(present) < 2:
            # One class only: the set supplies no evidence about telling its
            # members apart, and a recall over the single class present would
            # read as a score.
            continue
        recalls = [counts[(m, m)] / seen[m] for m in present]
        n_set = sum(seen.values())
        rows.append((n_set, sum(recalls) / len(recalls),
                     max(seen.values()) / n_set, '/'.join(present)))
    rows.sort(key=lambda r: -r[0])
    for n_set, balanced, majority, names in rows[:20]:
        flag = '' if balanced > majority else '   <- no better than the prior'
        print(f"{names[:34]:>34} {n_set:7,} {balanced:9.3f} {majority:9.3f}"
              f"{flag}")
    if rows:
        weight = sum(r[0] for r in rows)
        print(f"\n  {len(rows)} candidate sets with more than one class present")
        print(f"  weighted balanced accuracy {sum(r[0] * r[1] for r in rows) / weight:.3f}"
              f"   against a {sum(r[0] * r[2] for r in rows) / weight:.3f} majority baseline")


# ---------------------------------------------------------------------------
# --calibrate
# ---------------------------------------------------------------------------

def load_channel(path: Path) -> Dict[Tuple[str, str], int]:
    """
    How often the OCR turns one final letter into another, from the confident
    `MISREAD` verdicts `validate_line_end_chars.py` already produces.

    Its `reason` column reads `r->n`: observed `r`, correct `n`. So the counts
    are `P(observed | intended)` - the channel - and they are strongly
    asymmetric, which is the physical fact the whole detector rests on: a final
    `n` clipped by the margin comes out as `r` fourteen times more often than
    the reverse.

    A caveat that belongs on the number and not in a footnote: these labels are
    the *statistical* detector's, and its known failure modes (surnames, place
    names, truncations) are in them. For estimating a character-level rate the
    aggregate is probably tolerable; for anything finer it is not evidence.
    """
    channel: Dict[Tuple[str, str], int] = Counter()
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row.get('verdict') != 'MISREAD':
                continue
            reason = (row.get('reason') or '').strip()
            if '->' not in reason:
                continue
            observed, _, intended = reason.partition('->')
            if len(observed) == 1 and len(intended) == 1:
                channel[(observed, intended)] += int(
                    row.get('seen_at_line_end') or 0)
    return channel


def report_channel(channel: Dict[Tuple[str, str], int]) -> None:
    """Print the channel and, more usefully, how lopsided it is."""
    print(f"\n{'=' * 70}\nOCR channel, from the confident MISREAD verdicts\n"
          f"{'=' * 70}")
    if not channel:
        print("  empty - run validate_line_end_chars.py first and pass its CSV")
        return
    print(f"  {'observed -> intended':>24} {'n':>7}   {'reverse':>8} "
          f"{'ratio':>8}")
    for (observed, intended), n in sorted(channel.items(),
                                          key=lambda kv: -kv[1])[:14]:
        back = channel.get((intended, observed), 0)
        print(f"  {observed + ' -> ' + intended:>24} {n:7,}   {back:8,} "
              f"{n / max(back, 1):8.1f}")
    print("\n  A flat margin treats all of these alike and should not: the "
          "evidence\n  needed to overturn the transcription is not the same "
          "for `r -> n` as\n  for `n -> r`. Wiring this into the gate is "
          "phase 3 of the plan, and it\n  must be scored against the "
          "hand-checked sample before it is trusted.")


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def context_string(item: LineEnd) -> str:
    """The line with the judged token marked, plus the next line's first word."""
    body = item.line.rsplit(None, 1)[0] if len(item.line.split()) > 1 else ''
    tail = f" {item.following}" if item.following else "  [paragraph ends]"
    return f"{body} [{item.observed}]{tail}"


def export_sample(items: List[LineEnd], path: Path, size: int,
                  seed: int = 0, ungated: bool = False) -> None:
    """
    The stratified hand-check sample: phase 1, and the phase everything else
    waits on.

    Stratified over three axes at once, because each is a place the plan is
    currently guessing and a sample drawn from one of them would settle none of
    the others: margin band, token band (short against long), and whether the
    line end has a following token at all. `verdict` is left blank for a person
    to fill with y/n.

    Since 2026-09-04 the rows come from `proposals`, so the has-next axis is
    constant and two of the three strata that filled the first sample are gone.
    That is the point: the first sample existed to price them, it did, and
    re-checking a band that came back 102/102 wrong buys nothing. `--ungated`
    draws the old way, which is how the gates get re-measured rather than
    trusted.

    Each row carries a Transkribus deep link to its page, because a fair share
    of these cannot be settled from the transcription at all: whether
    "Frederik" or "Frederic" stands there is a question about the scan, and the
    reviewer should not have to go looking for the page themselves.
    """
    strata: Dict[Tuple, List[LineEnd]] = defaultdict(list)
    changed = proposals(items, 6, ungated)
    for item in changed:
        band = next(g for g in (16, 12, 10, 8, 6) if item.margin >= g)
        strata[(band, item.band, item.has_next)].append(item)

    rng = random.Random(seed)
    per = max(size // max(len(strata), 1), 1)
    picked = []
    for key in sorted(strata, key=lambda k: (-k[0], k[1], k[2])):
        rows = strata[key]
        rng.shuffle(rows)
        picked.extend((key, row) for row in rows[:per])

    path.parent.mkdir(parents=True, exist_ok=True)
    missing_links = 0
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        # The link goes last, not next to `context`: the reviewer's eye runs
        # along the short columns and only reaches for the scan when the text
        # alone will not settle it, and a 90-character URL in the middle of the
        # row pushes everything worth reading off the screen.
        writer.writerow(['verdict', 'margin_band', 'token_band', 'has_next',
                         'observed', 'proposed', 'margin', 'context',
                         'file', 'page', 'line_facs', 'subtype',
                         'transkribus_link'])
        for (band, token_band, has_next), item in picked:
            link = ''
            if get_transkribus_link:
                try:
                    link = get_transkribus_link(item.file, page=item.page)
                except Exception:
                    missing_links += 1
            writer.writerow(['', f'>={band}', token_band, int(has_next),
                             item.observed, item.best, f'{item.margin:.1f}',
                             context_string(item), item.file, item.page,
                             item.line_facs, item.subtype, link])
    print(f"\nHand-check sample saved to: {path}")
    print(f"  {len(picked):,} rows over {len(strata)} strata "
          f"(margin band x token band x has-next)")
    print(f"  drawn from {len(changed):,} proposals"
          + ("  (--ungated: no gates applied)" if ungated else
             f", {sum(1 for i in items if i.best != i.observed and i.margin >= 6 and gate_reason(i)):,} held back by the gates"))
    if missing_links:
        print(f"  {missing_links} rows have no Transkribus link - their issue "
              f"is not in\n  data/csv/transkribus_filenames.csv")
    print("  Fill `verdict` with y (the proposal is right) or n (it is not).")
    print("  That column is the only thing standing between this plan and a "
          "guess.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Judge ambiguous line-end tokens from their context.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--skip-subtypes',
                        default=','.join(hyphens.SKIPPED_SUBTYPES),
                        help='Comma-separated region subtypes to leave out of '
                             'the lexicon and the scan alike')
    parser.add_argument('--min-length', type=int, default=MIN_LENGTH,
                        help=f'Shortest line-final token to judge (default: '
                             f'{MIN_LENGTH}; the validator uses '
                             f'{VALIDATOR_MIN_LENGTH}, which is what hides '
                             f'71%% of this class)')
    parser.add_argument('--min-candidate', type=int, default=MIN_CANDIDATE,
                        help='Interior occurrences a substitution needs before '
                             f'it counts as a reading (default: {MIN_CANDIDATE})')
    parser.add_argument('--score', action='store_true',
                        help='Report the margin distribution (writes nothing)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Score against held-out line-interior occurrences')
    parser.add_argument('--eval-sample', type=int, default=40000)
    parser.add_argument('--calibrate', nargs='?', const='auto',
                        help='Report the OCR channel from a '
                             'validate_line_end_chars.py verdict CSV '
                             '(default: the newest in output/)')
    parser.add_argument('--export-sample',
                        help='Write the stratified hand-check sample; a bare '
                             f"filename lands in {PATHS['output']}")
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--ungated', action='store_true',
                        help='Draw the sample without `gate_reason`, the way '
                             'the 2026-08-30 one was drawn, to re-measure the '
                             'gates instead of trusting them')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if not (args.score or args.evaluate or args.calibrate
            or args.export_sample):
        parser.error('nothing to do: pass --score, --evaluate, --calibrate '
                     'or --export-sample')

    if args.calibrate:
        path = Path(args.calibrate) if args.calibrate != 'auto' else None
        if path is None:
            found = sorted(PATHS['output'].glob('*_line_end_chars.csv'))
            if not found:
                raise SystemExit(
                    "No *_line_end_chars.csv in output/. Run "
                    "validate_line_end_chars.py first, or pass a path.")
            path = found[-1]
        print(f"Channel read from: {path}")
        report_channel(load_channel(path))
        if not (args.score or args.evaluate or args.export_sample):
            return 0

    skip = frozenset(p.strip() for p in args.skip_subtypes.split(',')
                     if p.strip())
    interior, bigrams, raw = collect(args.xml_folder, args.max_files, skip,
                                     args.quiet)
    if not args.quiet:
        print(f"  {len(interior):,} interior word types / "
              f"{sum(interior.values()):,} tokens")
    items = ambiguous(raw, interior, args.min_length, args.min_candidate)
    print(f"\n{len(raw):,} judgeable line ends, {len(items):,} ambiguous "
          f"(observed spelling is itself a word and has an attested variant)")

    model = ContextModel(interior, bigrams)

    if args.score:
        score_all(items, model)
        report_scores(items)

    if args.evaluate:
        gold = gold_from_interior(items, interior, args.xml_folder,
                                  args.max_files, skip, args.eval_sample)
        evaluate(gold, interior, model, args.min_candidate)

    if args.export_sample:
        if not args.score:
            score_all(items, model)
        out = Path(args.export_sample)
        prefix = datetime.date.today().strftime('%Y%m%d')
        export_sample(items, out if out.parent != Path('.')
                      else PATHS['output'] / f'{prefix}_{out.name}',
                      args.sample_size, ungated=args.ungated)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
