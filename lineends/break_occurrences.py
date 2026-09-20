#!/usr/bin/env python3
"""
Break Occurrence Store
======================

How *one* line break should be written, when its word pair has no answer that
holds for the whole corpus.

`hyphen_decisions.py` keys a verdict by `(left, right)` and that is right for
almost every pair: `Prin`+`zessin` is a broken word wherever it occurs, and
settling it once is what makes a rerun over a growing corpus cheap. It is
wrong for the pairs where both readings are real German and only the sentence
says which is meant:

    kaiser|in    "Kaiserin Zita"        vs  "der Kaiser in Wien"
    wie|der      "wieder in der Stadt"  vs  "wie der Bruder"
    nach|dem     "nachdem Er verfügt"   vs  "Witwe nach dem Grafen"

For those the pair store holds `form=context`, which says only that it will
not answer, and the answer lives here — one row per break, not per pair.

The occurrence key: `(file, line_facs)`
---------------------------------------
A break belongs to the line it *ends*, and every `<l>` in the corpus carries a
`facs` id unique within its file:

    ONB_wsb_19140104.xml , facs_1_tr_1784278140_tl_3

No new machinery was needed for this. `mark_xml_word_breaks.py` already writes
per-line attributes keyed the same way, `correct_xml_hyphens.py` already walks
the lines with lxml and has the element in hand, and `LINE_RE` always captured
the id — only `read_paragraphs` threw it away, which is why
`validate_xml_hyphens.read_paragraph_lines` exists.

A line's position in its paragraph is deliberately *not* the key.
`correct_xml_hyphens.py` iterates the markup-free lines only, so the same break
has different indices in different scripts, and a key that means different
things to its two users is worse than no key.

`left` and `right` are stored too, redundantly. They are what makes the file
readable in a spreadsheet, and they are how a stale key is caught: the `tr_…`
component is a Transkribus region id and need not survive a re-export, so
every consumer checks that the pair at that line is still the pair the row was
written about, and reports rather than guesses when it is not.

Why the forms are narrower than the pair store's
------------------------------------------------
`join | hyphen | period | comma | split` — the five that name a character.
There is no `skip` and no `context`:

    skip     means "this pair is unresolvable". A question about one
             occurrence with its own sentence in front of it does not get to
             be unresolvable; if the sentence cannot settle it, the honest
             record is no row at all, and a missing row already means "left
             alone".
    context  means "no corpus-wide answer exists". That is the claim that
             sent the break here. Repeating it here would be a loop.

Every row also carries `score`, the context-model margin from
`resolve_break_context.py` — positive for the joined reading, negative for the
separated one. It is recorded whatever decided the row, including rows a human
wrote, because a verdict in this project carries its evidence and because the
disagreements between the score and the answer are the stage's own error
signal.

Author: Christian Lendl
Created: 2026-08-29
"""

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# The five forms that name a character. See the module docstring for why
# `skip` and `context` are not among them.
FORMS = ('join', 'hyphen', 'period', 'comma', 'split')

# A human outranks the model, which outranks the context score. Identical to
# the two sibling stores, and load-bearing for the same reason: it is what
# lets a rerun be idempotent and a hand correction survive one.
SOURCE_RANK = {'manual': 3, 'llm': 2, 'rule': 1}

FIELDNAMES = ['file', 'line_facs', 'left', 'right', 'form', 'score',
              'source', 'model', 'decided_on', 'note']


@dataclass
class Occurrence:
    """One settled line break."""
    file: str
    line_facs: str
    left: str
    right: str
    form: str
    score: str = ''
    source: str = 'llm'
    model: str = ''
    decided_on: str = ''
    note: str = ''

    @property
    def key(self) -> Tuple[str, str]:
        return (self.file, self.line_facs)

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.left, self.right)

    @property
    def rank(self) -> int:
        return SOURCE_RANK.get(self.source, 0)

    def matches(self, left: str, right: str) -> bool:
        """
        Is this still the break the row was written about?

        Checked by every consumer before the form is used. A `facs` id that
        has been reused by a re-export points at a different break, and
        applying a verdict to it would be a silent corruption of exactly the
        kind this store exists to prevent.
        """
        return (self.left, self.right) == (left, right)


def load(path: Path) -> Dict[Tuple[str, str], Occurrence]:
    """
    Read the occurrence store.

    Returns an empty mapping when the file does not exist yet, so a first run
    needs no setup. Duplicates resolve by source rank, the later row winning a
    tie, which is what lets a hand-edited row be appended rather than typed
    over the machine's.
    """
    occurrences: Dict[Tuple[str, str], Occurrence] = {}
    if not path.exists():
        return occurrences

    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            file = (row.get('file') or '').strip()
            facs = (row.get('line_facs') or '').strip()
            form = (row.get('form') or '').strip()
            if not file or not facs or form not in FORMS:
                continue
            occurrence = Occurrence(
                file=file,
                line_facs=facs,
                left=(row.get('left') or '').strip(),
                right=(row.get('right') or '').strip(),
                form=form,
                score=(row.get('score') or '').strip(),
                source=(row.get('source') or 'llm').strip(),
                model=(row.get('model') or '').strip(),
                decided_on=(row.get('decided_on') or '').strip(),
                note=(row.get('note') or '').strip(),
            )
            previous = occurrences.get(occurrence.key)
            if previous is None or occurrence.rank >= previous.rank:
                occurrences[occurrence.key] = occurrence
    return occurrences


def by_file(occurrences: Dict[Tuple[str, str], Occurrence]
            ) -> Dict[str, Dict[str, Occurrence]]:
    """
    Regroup as `{filename: {line_facs: Occurrence}}`.

    The correcting scripts open one file at a time and look up many lines in
    it; handing them the whole store to index per line would make every
    correction pass quadratic in a store that only grows.
    """
    grouped: Dict[str, Dict[str, Occurrence]] = {}
    for occurrence in occurrences.values():
        grouped.setdefault(occurrence.file, {})[occurrence.line_facs] = \
            occurrence
    return grouped


def merge(path: Path, new_occurrences: Iterable[Occurrence],
          quiet: bool = False) -> Tuple[int, int]:
    """
    Add occurrences to the store and write it back.

    An incoming row replaces an existing one only when it ranks strictly
    higher, so re-running the scorer over a store the model has already
    answered changes nothing, and a hand-made decision is never undone by a
    rerun. To revisit a break, delete its row.

    The exception is a row about a *different* break. A re-export can hand a
    `facs` id to a line whose word pair has changed - an OCR correction, an
    edit in Transkribus - and a verdict on the old pair answers nothing about
    the new one, whatever its rank. So an incoming row for another pair
    replaces it; 188 of 2,842 rows were in that state after the re-export of
    2026-09-18.

    Returns:
        (added, updated) - `updated` including the stale rows replaced
    """
    occurrences = load(path)
    added = updated = replaced = 0

    for occurrence in new_occurrences:
        if not occurrence.decided_on:
            occurrence.decided_on = datetime.date.today().isoformat()
        previous = occurrences.get(occurrence.key)
        if previous is None:
            occurrences[occurrence.key] = occurrence
            added += 1
        elif not previous.matches(occurrence.left, occurrence.right):
            occurrences[occurrence.key] = occurrence
            replaced += 1
        elif occurrence.rank > previous.rank:
            occurrences[occurrence.key] = occurrence
            updated += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES,
                                lineterminator='\n')
        writer.writeheader()
        for key in sorted(occurrences):
            o = occurrences[key]
            writer.writerow({
                'file': o.file, 'line_facs': o.line_facs,
                'left': o.left, 'right': o.right, 'form': o.form,
                'score': o.score, 'source': o.source, 'model': o.model,
                'decided_on': o.decided_on, 'note': o.note,
            })

    if not quiet:
        pairs = len({o.pair for o in occurrences.values()})
        print(f"Occurrence store: {len(occurrences):,} breaks over {pairs:,} "
              f"pairs ({added:,} added, {updated:,} updated"
              + (f", {replaced:,} stale replaced" if replaced else '')
              + f") -> {path}")
    return added, updated + replaced
