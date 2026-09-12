#!/usr/bin/env python3
"""
Hyphen Decision Store
=====================

A growing record of how each ambiguous line-break word pair should be written.
Every pair that has been settled once - by the LLM or by hand - is kept here
and is never asked again, so a rerun over a larger corpus only ever queries the
pairs that are genuinely new.

The store is a plain CSV so it can be edited in a spreadsheet; a hand-made row
always outranks a machine-made one for the same pair.

Columns:
    left        left-hand word at the line break, punctuation stripped
    right       right-hand word at the line break, punctuation stripped
    form        join | hyphen | period | comma | split | skip | context
    source      llm | manual | rule
    model       model name for source=llm, otherwise empty
    decided_on  ISO date
    note        free text, e.g. the model's one-line justification

`form` values:
    join    the word continues       "Prin"  + "zessin" -> "Prinzessin"
    hyphen  real compound hyphen     "Windisch-Graetz"
    period  the `¬` is a full stop   "nicht aus. Karl Schäfer"
    comma   the `¬` is a comma       "fand ... statt, wo die Gastgeber"
    split   two separate words       "die Leiden der Bedauernswerten"
    skip    unresolvable - leave the source untouched
    context no corpus-wide answer exists - decided per occurrence

`context` is the one form that does not describe a break. It describes the
*pair*: both readings are real German and which one is meant depends on the
sentence, so there is nothing to write here that would not be wrong somewhere.
"Kaiserin Zita" and "der Kaiser in Wien" are both this corpus, and so are
"wieder in der Stadt" and "wie der Bruder". A pair marked `context` is answered
one occurrence at a time in `data/csv/hyphen_occurrences.csv`, and every
consumer of this store treats it as "not mine to decide".

It is deliberately unlike `skip`, which the two resemble in effect until the
occurrence store is consulted: `skip` means *leave every occurrence alone*,
`context` means *every occurrence gets its own verdict*. Only a human writes
it - the resolver is never offered it as an answer, because a model shown three
contexts is in no position to say that no answer exists.

A form is a *decision*, not a character. Which character it puts at the end of
a line depends on how the issue was set: the Antiqua issues mark a continued
word with `¬` and a compound with an ordinary `-`, whereas the older Fraktur
ones write both with the Doppelbindestrich `=`. The store therefore carries no
notation of its own - a pair settled once holds for either typesetting, and
`line_end_for` picks the character when the correction is written.

Author: Christian Lendl
Created: 2026-07-22
Last Modified: 2026-07-29
"""

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Every form the store understands. `skip` deliberately makes no change.
FORMS = ('join', 'hyphen', 'period', 'comma', 'split', 'skip', 'context')

# What each form puts at the end of the line in the TEI, replacing the break
# mark the OCR produced - one table per line-break notation.
#
# An issue set in Antiqua distinguishes the two: `¬` says the word continues,
# a plain `-` is a real compound hyphen carried over the break. Fraktur has no
# such distinction - the Doppelbindestrich `=` does both jobs - so both forms
# write `=` there and the join/hyphen decision survives only in this store,
# which is where `hyphen_utils.py` needs it anyway.
LINE_ENDS = {
    '¬': {'join': '¬', 'hyphen': '-',
          'period': '.', 'comma': ',', 'split': ''},
    '=': {'join': '=', 'hyphen': '=',
          'period': '.', 'comma': ',', 'split': ''},
}

# The mark a continued line ends with, per notation. These two characters are
# never anything but a line-break notation, which is what makes them usable to
# tell the typesettings apart; `-` is deliberately absent, being the ordinary
# compound hyphen in either style.
NOTATIONS = tuple(LINE_ENDS)

# Antiqua covers the whole corpus as it stands, so it is what an undecidable
# file falls back to.
DEFAULT_NOTATION = '¬'

# Spellable aliases, so a notation can be forced from a command line without
# having to type `¬`
NOTATION_NAMES = {'antiqua': '¬', 'fraktur': '='}

# Kept for callers that only ever deal with the Antiqua notation
FORM_TO_LINE_END = LINE_ENDS[DEFAULT_NOTATION]

# A manual verdict wins over a model verdict, which wins over a rule
SOURCE_RANK = {'manual': 3, 'llm': 2, 'rule': 1}

FIELDNAMES = ['left', 'right', 'form', 'source', 'model', 'decided_on', 'note']


@dataclass
class Decision:
    """One settled word pair."""
    left: str
    right: str
    form: str
    source: str = 'llm'
    model: str = ''
    decided_on: str = ''
    note: str = ''

    @property
    def key(self) -> Tuple[str, str]:
        return (self.left, self.right)

    @property
    def rank(self) -> int:
        return SOURCE_RANK.get(self.source, 0)


def load(path: Path) -> Dict[Tuple[str, str], Decision]:
    """
    Read the decision store.

    Returns an empty mapping when the file does not exist yet, so a first run
    needs no setup. Duplicate pairs are resolved by source rank, which lets a
    hand-corrected row override an earlier model verdict without deleting it.
    """
    decisions: Dict[Tuple[str, str], Decision] = {}
    if not path.exists():
        return decisions

    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            form = (row.get('form') or '').strip()
            left = (row.get('left') or '').strip()
            right = (row.get('right') or '').strip()
            if not left or not right or form not in FORMS:
                continue
            decision = Decision(
                left=left,
                right=right,
                form=form,
                source=(row.get('source') or 'llm').strip(),
                model=(row.get('model') or '').strip(),
                decided_on=(row.get('decided_on') or '').strip(),
                note=(row.get('note') or '').strip(),
            )
            previous = decisions.get(decision.key)
            if previous is None or decision.rank >= previous.rank:
                decisions[decision.key] = decision
    return decisions


def merge(path: Path, new_decisions: Iterable[Decision],
          quiet: bool = False) -> Tuple[int, int]:
    """
    Add decisions to the store and write it back.

    An incoming decision replaces an existing one only when it ranks strictly
    higher: a human can overrule the model, the model can overrule a rule, and
    nothing overrules a human. Two verdicts of the same rank leave the first
    one standing, which makes re-running the resolver idempotent.

    To genuinely revisit a pair, delete its row (or edit `form` in place and
    set `source` to `manual`).

    Returns:
        (added, updated)
    """
    decisions = load(path)
    added = updated = 0

    for decision in new_decisions:
        if decision.form not in FORMS:
            continue
        if not decision.decided_on:
            decision.decided_on = datetime.date.today().isoformat()
        previous = decisions.get(decision.key)
        if previous is None:
            decisions[decision.key] = decision
            added += 1
        elif decision.rank > previous.rank and decision.form != previous.form:
            decisions[decision.key] = decision
            updated += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for key in sorted(decisions):
            d = decisions[key]
            writer.writerow({
                'left': d.left, 'right': d.right, 'form': d.form,
                'source': d.source, 'model': d.model,
                'decided_on': d.decided_on, 'note': d.note,
            })

    if not quiet:
        print(f"Decision store: {len(decisions):,} pairs "
              f"({added:,} added, {updated:,} updated) -> {path}")
    return added, updated


def line_end_for(form: str, notation: str = DEFAULT_NOTATION) -> Optional[str]:
    """
    The character a line should end with for this form.

    None for `skip` and for `context`, neither of which names a character:
    `skip` has no answer and `context` has no *corpus-wide* answer, so both
    leave the line as it is and the caller looks elsewhere - for `context`,
    in the occurrence store.

    `notation` is the break mark the file being corrected is transcribed with
    (see `LINE_ENDS`), so a Fraktur issue keeps its `=` instead of acquiring a
    `¬` it never used. An unknown notation falls back to Antiqua.
    """
    return LINE_ENDS.get(notation, LINE_ENDS[DEFAULT_NOTATION]).get(form)
