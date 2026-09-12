#!/usr/bin/env python3
"""
OCR Word Correction Store
=========================

Word-level OCR fixes at a line break, keyed by the same `(left, right)` pair as
`hyphen_decisions.py` and kept in the same shape: a plain CSV in `data/csv/`
that can be edited in a spreadsheet, committed, and grown over time.

This is the *positional* half of the OCR correction in `lineends`. The other
half is `data/csv/replacement.csv`, a global search-and-replace list applied by
`correct_xml_ocr.py` to the text of every paragraph. The two answer different
questions and must not be merged:

    replacement.csv   "this string is always wrong, wherever it occurs"
                      " Selte " -> " Seite ", ' österreich' -> ' Österreich'

    this store        "this word is wrong *here*, at a line break"
                      "Antra" is a misreading of "Antrag" before "der";
                      "Wier" is a misreading of "Wien" at the end of a line.
                      Neither is wrong in the middle of a line, where the
                      letters were never clipped by the margin.

Writing a positional error into `replacement.csv` would apply it corpus-wide and
corrupt every innocent occurrence. The position is half the warrant, which is
why these live apart.

Where the rows come from, and why they are held to a higher bar
--------------------------------------------------------------
Two sources, neither of which was asked the question it ends up answering:

    extract_ocr_corrections.py   While judging a hyphen, the model often
                                 remarked in passing that one of the words is
                                 itself misread. That is an observation nobody
                                 asked for, made from surrounding text alone
                                 and without ever seeing the scan.

    validate_line_end_chars.py   Corpus statistics: a line-final spelling that
                                 is unattested inside a line while a
                                 one-character variant is common. Measured on
                                 this corpus that is right far more often than
                                 not, but it also proposes `dorf` -> `dort`
                                 and `Reiss` -> `Reise`, which are surnames
                                 and place names, not misreadings.

So unlike `hyphen_decisions.py`, a row here does **not** apply just by existing:

    confidence  high | medium | low   what the source claimed about itself
    approved    yes | no | (empty)    what a human decided, and it always wins

`correct_xml_ocr.py` applies rows with `approved=yes`, plus rows whose
confidence clears `--min-confidence` and that are not `approved=no`. Setting
`approved=no` is how a wrong suggestion is retired for good without deleting the
row, so a later re-run does not propose it again.

Columns:
    left           left-hand word at the line break, punctuation stripped
    right          right-hand word at the line break, punctuation stripped,
                   or `*` for a row that holds at any line end
    left_correct   what the left word should read; equal to `left` if fine
    right_correct  what the right word should read; equal to `right` if fine
    confidence     high | medium | low
    approved       yes | no | empty (undecided - confidence decides)
    source         llm | manual | rule
    model          model name for source=llm, otherwise empty
    decided_on     ISO date
    note           the model's one-line justification, or a hand-written one

Both sides carry their own corrected form because a break can need fixing on
either — or on both — and because a word split across the break has to stay
splittable:

    Antra  + der              -> Antrag  + der          (left only)
    August + Verlin           -> August  + Berlin       (right only)
    Abge   + geordnetenhause  -> Abge    + ordnetenhause
                                 ("Abgeordnetenhause", still joinable)

A row where neither side changes is a no-op. It is kept all the same, so the
pair counts as settled and is never asked about again.

Line-final rows: `right = *`
----------------------------
`validate_line_end_chars.py` finds a different kind of error — the last
character of a line misread, "Wien" coming out as "Wier" — which is a property
of the word alone, not of the pair it happens to stand next to. Writing one row
per following word would repeat the same decision hundreds of times, so such a
row uses `*` for `right`:

    Wier , * , Wien , , high , , rule , ...

A wildcard row means "this token, wherever it ends a line". It differs from an
ordinary row in three ways, all enforced by `correct_xml_ocr.py`:

  * it applies to the last token of *any* line, including the last line of a
    paragraph, which no pair row can reach;
  * it never applies to a line that ends in a hyphen — there the token is a
    fragment of a word continuing on the next line, not the word `left` claims
    to be;
  * it can only correct the left side, having no knowledge of the right.

An exact pair row always wins over a wildcard for the same token, which is how
one awkward context gets an exception without disturbing the general rule.

Author: Christian Lendl
Created: 2026-08-14
"""

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Ordered weakest to strongest, so `--min-confidence` can compare by index
CONFIDENCE_LEVELS = ('low', 'medium', 'high')

APPROVALS = ('yes', 'no', '')

# `right` value marking a row that applies to a token at any line end
WILDCARD = '*'

# A human outranks the model, which outranks the corpus statistics; nothing
# outranks a human. `rule` is what validate_line_end_chars.py writes, `llm` what
# extract_ocr_corrections.py and resolve_line_end_llm.py write. The ranking is
# what lets the LLM overturn a statistical verdict it disagrees with, which is
# the whole point of running the resolver over the MISREAD rows.
SOURCE_RANK = {'manual': 3, 'llm': 2, 'rule': 1}

FIELDNAMES = ['left', 'right', 'left_correct', 'right_correct',
              'confidence', 'approved', 'source', 'model', 'decided_on',
              'note']


@dataclass
class Correction:
    """One settled word pair, with whatever word fix it needs."""
    left: str
    right: str
    left_correct: str = ''
    right_correct: str = ''
    confidence: str = 'low'
    approved: str = ''
    source: str = 'llm'
    model: str = ''
    decided_on: str = ''
    note: str = ''

    def __post_init__(self):
        # An omitted corrected form means "this side is fine as it is", which
        # keeps every row self-describing: `left_correct` is always the word
        # that should end up in the XML.
        self.left_correct = self.left_correct or self.left
        self.right_correct = self.right_correct or self.right

    @property
    def key(self) -> Tuple[str, str]:
        return (self.left, self.right)

    @property
    def rank(self) -> int:
        return SOURCE_RANK.get(self.source, 0)

    @property
    def changes_left(self) -> bool:
        return self.left_correct != self.left

    @property
    def changes_right(self) -> bool:
        # A wildcard row stands for many different right-hand words and knows
        # none of them, so it can never claim to correct one.
        return not self.is_wildcard and self.right_correct != self.right

    @property
    def is_wildcard(self) -> bool:
        """True for a row that applies to this token at any line end."""
        return self.right == WILDCARD

    @property
    def is_noop(self) -> bool:
        """True when the row asserts no word error at all."""
        return not (self.changes_left or self.changes_right)

    def applies(self, min_confidence: str = 'high') -> bool:
        """
        Whether this row may rewrite the XML.

        A human verdict is absolute in both directions; without one the source's
        own confidence has to clear the bar. A no-op never applies, whatever it
        says about itself.
        """
        if self.is_noop or self.approved == 'no':
            return False
        if self.approved == 'yes':
            return True
        return (_confidence_rank(self.confidence)
                >= _confidence_rank(min_confidence))


def _confidence_rank(level: str) -> int:
    """Position in CONFIDENCE_LEVELS; an unknown level ranks lowest."""
    try:
        return CONFIDENCE_LEVELS.index(level)
    except ValueError:
        return -1


def load(path: Path) -> Dict[Tuple[str, str], Correction]:
    """
    Read the correction store.

    Returns an empty mapping when the file does not exist yet, so a first run
    needs no setup. Duplicate pairs are resolved by source rank, which lets a
    hand-corrected row override an earlier model verdict without deleting it.
    """
    corrections: Dict[Tuple[str, str], Correction] = {}
    if not path.exists():
        return corrections

    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            left = (row.get('left') or '').strip()
            right = (row.get('right') or '').strip()
            if not left or not right:
                continue
            approved = (row.get('approved') or '').strip().lower()
            correction = Correction(
                left=left,
                right=right,
                left_correct=(row.get('left_correct') or '').strip(),
                right_correct=(row.get('right_correct') or '').strip(),
                confidence=(row.get('confidence') or 'low').strip().lower(),
                approved=approved if approved in APPROVALS else '',
                source=(row.get('source') or 'llm').strip(),
                model=(row.get('model') or '').strip(),
                decided_on=(row.get('decided_on') or '').strip(),
                note=(row.get('note') or '').strip(),
            )
            previous = corrections.get(correction.key)
            if previous is None or correction.rank >= previous.rank:
                corrections[correction.key] = correction
    return corrections


def merge(path: Path, new_corrections: Iterable[Correction],
          quiet: bool = False) -> Tuple[int, int]:
    """
    Add corrections to the store and write it back.

    An incoming row replaces an existing one only when it ranks strictly
    higher, so a hand-made decision can never be undone by re-running the
    extraction, and re-running it changes nothing at all.

    To genuinely revisit a pair, delete its row (or edit it in place and set
    `source` to `manual`).

    Returns:
        (added, updated)
    """
    corrections = load(path)
    added = updated = 0

    for correction in new_corrections:
        if not correction.decided_on:
            correction.decided_on = datetime.date.today().isoformat()
        previous = corrections.get(correction.key)
        if previous is None:
            corrections[correction.key] = correction
            added += 1
        elif correction.rank > previous.rank:
            corrections[correction.key] = correction
            updated += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for key in sorted(corrections):
            c = corrections[key]
            writer.writerow({
                'left': c.left, 'right': c.right,
                'left_correct': c.left_correct,
                'right_correct': c.right_correct,
                'confidence': c.confidence, 'approved': c.approved,
                'source': c.source, 'model': c.model,
                'decided_on': c.decided_on, 'note': c.note,
            })

    if not quiet:
        real = sum(1 for c in corrections.values() if not c.is_noop)
        print(f"Correction store: {len(corrections):,} pairs, {real:,} with a "
              f"word fix ({added:,} added, {updated:,} updated) -> {path}")
    return added, updated
