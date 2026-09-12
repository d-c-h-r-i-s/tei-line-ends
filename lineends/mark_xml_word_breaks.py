#!/usr/bin/env python3
"""
TEI Word-Break Marking
======================

Records in the TEI files themselves what a mark at the end of a line *means*:
whether the two halves are one word broken across the break, or a compound
whose hyphen belongs to the word. It is the last step of the hyphen loop, run
after `validate_xml_hyphens.py` has judged the breaks and
`correct_xml_hyphens.py` has fixed the wrong ones — the marks that remain are
the right ones, and this pass says which kind each of them is.

    <l facs='#…_tl_3' wsb:break='word'>… Frau Erzherzogin Auguste, Ge¬</l>
    <l facs='#…_tl_4'>mahlin des Herrn Erzherzogs Josef, dem sie am 15. Nov.</l>

    <l facs='#…_tl_13' wsb:break='compound'>… Ehrendame des bayer. Theresien-</l>
    <l facs='#…_tl_14'>Ordens und Dame des bayer. St. Elisabethen-Ordens ist</l>

Why the character is not enough
-------------------------------
In an Antiqua issue the transcription already distinguishes the two: `¬` says
the word continues, a plain `-` is a real compound hyphen. In the older Fraktur
issues it does not — the Doppelbindestrich `=` does both jobs, and there is no
character left to tell them apart. The distinction survives only in
`data/csv/hyphen_decisions.csv` and in the frequency evidence, neither of which
travels with the corpus. Marking it onto the line is what makes the answer
readable from the file alone, in either typesetting, by anyone who has never
heard of `¬`.

That is also what `hyphen_utils.py` currently has to *guess*: it keeps
a line-end hyphen before a capitalised word and drops it before a lowercase one,
which is a decent rule and still a rule. Once the marks are in place the import
can read the answer instead of re-deriving it, and every consumer of the corpus
gets the same text the database does.

What gets marked
----------------
Only a line that ends in a break mark (`¬`, `-`, `=`) is touched — 21% of lines,
the ones where a question actually arises. A line whose token simply ends gets
no attribute, and *that is a statement*: the `<application>` marker this script
writes into `<teiHeader>` is what says the pass ran, so absence means "no
continuation here" rather than "nobody looked".

    wsb:break='word'      one word broken across the break; the mark is not
                          part of the word and is dropped when joining
    wsb:break='compound'  a compound whose hyphen belongs to the word; the
                          hyphen is kept when joining
    wsb:break='unknown'   a mark whose kind cannot be established here — in
                          practice a paragraph's last line, whose continuation
                          is not in this paragraph (see below)

**The attribute is in the project's namespace, not TEI's.** TEI does have a
`@break` — `att.breaking`, on the milestone-like elements `<lb>`, `<pb>`,
`<cb>`, `<gb>`, `<milestone>` — and `<l>` is not among them, so a bare
`break='word'` on an `<l>` is rejected outright by `tei_all` (checked against
the schema, not assumed). Its `no`/`yes`/`maybe` vocabulary also has no value
meaning "continues, but keep the hyphen". Both point the same way: this is the
project's own attribute, it goes in the namespace `config.toml` names
(`[corpus] project_namespace`), where your own ODD can declare it. `<l>` keeps its plain-text content:
writing `<lb break='no'/>` into it instead would be conformant and would also
make every marked line invisible to `correct_xml_hyphens.py` and
`correct_xml_ocr.py`, both of which skip a line carrying markup.

Certainty
---------
A mark backed by the corpus evidence or by the decision store carries no `@cert`.
Where neither has an answer — 54k of the 261k marked line ends, pairs the
statistics never had to decide because nothing contradicted the `¬` — the break
type comes from the rule the import has always used
(`hyphen_utils.determine_hyphen_action`: a capitalised next word, `und` or a
Hungarian street abbreviation keeps the hyphen, anything lowercase drops it), and
`cert='low'` says so:

    <l facs='#…_tl_19' wsb:break='word' cert='low'>… der Gräfin Elisa¬</l>

`@cert` is TEI's own (att.global.responsibility), so this much is conformant.
Falling back to *that* rule rather than to the break character is deliberate:
it means a low-certainty mark states what the database text already says instead
of introducing a third opinion, so the marks change nothing except where there
is evidence to change it. It is also the better rule. The character looks like
the obvious signal and is the weaker one — a line-final `-` before a lowercase
word is a continuation the OCR read as a hyphen ("Kunst-" + "gewerbemuseum" is
one word), and `¬` before a capitalised word is just as often a compound whose
own hyphen was read as a continuation mark ("Arcièren¬" + "Leibgarde"). Reading
the character instead would have contradicted the import on 15,556 breaks,
mostly wrongly. It is precisely the class `decide_form` refuses to judge
('capitalised-right'), which is why it dominates this bucket.

Draining it is what `resolve_hyphens_llm.py` does — every pair it settles turns a
`cert='low'` mark into a plain one on the next run.

A mark that should not be there
-------------------------------
Two kinds of marked line end get no break type at all, and the difference
matters:

- The evidence reads the pair as two separate words, a full stop or a comma. The
  mark is an *error*, not a break type, so the line is left unmarked and
  reported — encoding it as `unknown` would hide something that is known. A
  non-zero count means `correct_xml_hyphens.py` has not run over these files yet
  (or ran with `--skip-subtypes`, or with a `--min-count` that held the pair
  back); re-running this pass afterwards drives the count to its irreducible
  remainder.
- The next word cannot be a word-part at all — one of
  `hyphen_utils.OCR_SPACE_PATTERNS` (`geb.`, `v.`, `gew.`, `z.`, `(`), as in
  "Justizmin. Eugen¬ | v. Balogh". The mark is OCR noise and the line genuinely
  ends its token, so leaving it unmarked is the answer rather than a gap. This
  one is a *guard* and runs before the evidence, because the frequency tables
  cannot see it and do get it wrong: "Schaumburg-Lippe¬ | geb. Prinzessin" is
  judged `join` (the pair is only ever seen hyphenated, the bigram lexicon being
  built from line-interior tokens where the phrase never lands), and marking that
  would join it into "Schaumburg-Lippegeb".

Paragraph-final marks
---------------------
A mark on the *last* line of a paragraph has no following line in this file, so
neither the evidence nor the rule has anything to read — the paragraph continues
in the next column or on the next page and is joined by
the paragraph-merge step (downstream, not shipped), which is the step that can see the next word.
There are 4,136 of them and they get `wsb:break='unknown'`, the one place that value
is used.

Re-running
----------
Idempotent by construction. Every `<l>` in scope has its `@wsb:break` (and the
`@cert` that came with it) cleared before the pass recomputes it, so a mark whose
pair has since been settled loses its `cert='low'`, and a line that has since
been corrected loses its mark altogether. A file already carrying this version
of the marking is skipped unless `--force` is given; a region held back by
`--skip-subtypes` keeps whatever marks it already has, since this run is not
recomputing them.

Usage:
    python mark_xml_word_breaks.py --dry-run
    python mark_xml_word_breaks.py --backup
    python mark_xml_word_breaks.py --force --report-all

Flags:
    --xml-folder P    Path to XML files folder (default: PATHS['xml']).
    --decisions P     Decision store to consult (default:
                      PATHS['csv']/hyphen_decisions.csv).
    --max-files N     Maximum number of files to process.
    --ratio R         Evidence ratio required for a verdict (default: 3.0),
                      passed to the analysis so it matches the correction run.
    --skip-subtypes S Comma-separated TextRegion subtypes to leave untouched,
                      e.g. 'ad,ad-frame'. Nothing is skipped by default: a mark
                      states what the text already shows and costs nothing even
                      in the noisiest region.
    --force           Re-mark files already carrying this version.
    --backup          Write <name>.xml.bak before modifying a file.
    --dry-run         Report what would be marked without writing files.
    --report NAME     Write a per-mark CSV to output/ (default:
                      word_break_marks.csv). Only the marks needing attention
                      go in it - the low-certainty ones and the errors found.
    --report-all      Put every mark in the report, confident ones included.
    --quiet           Suppress progress messages.

Author: Christian Lendl
Created: 2026-08-14
Last Modified: 2026-08-14
"""

import argparse
import csv
import datetime
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS
from hyphen_decisions import load as load_decisions
from hyphen_utils import (
    OCR_SPACE_PATTERNS,
    determine_hyphen_action,
    get_next_word,
)
from tei_text import (
    BREAK_ATTR,
    BREAK_COMPOUND,
    BREAK_UNKNOWN,
    BREAK_WORD,
    CERT_ATTR,
    WORDBREAK_IDENT,
)
from validate_xml_hyphens import (
    HYPHEN_CHARS,
    SKIPPED_SUBTYPES,
    analyse,
    break_pair,
    detect_notation,
)
from xml_io import (
    NS,
    XML_ID,
    ensure_wsb_namespace,
    is_up_to_date,
    parse_tei,
    set_marker,
    tei,
    write_with_single_quotes,
)

try:
    from utils.get_transkribus_link import get_transkribus_link
except ImportError:  # pragma: no cover - link is a convenience only
    get_transkribus_link = None

# ============================================================================
# Configuration
# ============================================================================

# Bumping this re-marks the whole corpus on the next run: a file is skipped only
# when the version recorded in its header is >= this one. Bump it when the
# vocabulary below changes, never for a change in the evidence — a pair the LLM
# has newly settled is picked up with `--force`, not with a new version.
VERSION = '2.0'
APP_IDENT = WORDBREAK_IDENT

# The attribute this script owns, the one it uses to qualify it, and the three
# values it may write, all imported from `tei_text` — the module that reads
# them back. Two copies of a vocabulary is one copy too many: a value added
# here and not there would be written into the corpus and silently ignored by
# everything that consumes it. Both attributes are cleared and rewritten on
# every run, so nothing else may write them onto an `<l>`.

# The evidence speaks in forms (see `hyphen_decisions.py`); only two of them
# leave a mark on the line, and these are what those two mean.
FORM_TO_BREAK = {'join': BREAK_WORD, 'hyphen': BREAK_COMPOUND}

# Forms that say the two words stand apart. A line still carrying a mark under
# one of these is an uncorrected error, not a break type.
SEPARATING_FORMS = ('split', 'period', 'comma')

# What the import's own rule returns, translated into this vocabulary. Its
# 'space' is not a break type at all - it is the claim that the mark is an OCR
# artifact and the two words are separate - so it leaves the line unmarked. In
# practice `break_type`'s guard has already caught every case that produces it,
# both being driven by `OCR_SPACE_PATTERNS`; it is mapped anyway so that
# extending that rule cannot turn into a KeyError here.
ACTION_TO_BREAK = {'remove': BREAK_WORD, 'hyphen': BREAK_COMPOUND,
                   'space': None}

# Where a mark's break type came from, for the report's `source` column
SRC_EVIDENCE = 'evidence'      # the frequency analysis decided it
SRC_DECISION = 'decision'      # the decision store decided it
SRC_RULE = 'rule'              # hyphen_utils.determine_hyphen_action
SRC_GUARD = 'rule-guard'       # the right side cannot be a word-part at all
SRC_PARA_FINAL = 'paragraph-final'   # no following line in this file at all

# A mark from one of these carries cert='low'. `unknown` needs no `@cert`: the
# value already says there is no answer, whereas `@cert` says "this value is a
# rule's guess rather than evidence".
LOW_CERTAINTY_SOURCES = (SRC_RULE,)

MARKER_LABEL = (
    "@wsb:break on <l> records what the break mark at the end of the line "
    "means: "
    "'word' - one word broken across the line break, the mark is not part of "
    "the word and is dropped when the lines are joined; 'compound' - a compound "
    "whose hyphen belongs to the word and is kept; 'unknown' - a mark whose kind "
    "could not be established, the paragraph continuing elsewhere. A line "
    "without @break ends its token. @cert='low' marks a break type derived from "
    "an orthographic rule rather than from corpus evidence."
)


# ============================================================================
# Deciding one break
# ============================================================================

def break_type(char: str, form: Optional[str], verdict: str, decided_by: str,
               next_line: Optional[str]) -> Tuple[Optional[str], str]:
    """
    What to write for one marked line end.

    Returns `(break value, source)`, with a value of None when the line must be
    left unmarked — either because the mark itself is an error the correction
    step has not applied yet, or because the fallback rule reads it as OCR
    noise between two separate words.

    `next_line` is None for a mark on a paragraph's last line: there is no
    following line in this file, the evidence never judged it and the rule has
    nothing to read, so `unknown` is the only honest answer. What continues the
    paragraph lives in the next column or on the next page and is joined by
    the paragraph-merge step (downstream, not shipped).

    A REVIEW verdict counts as no answer even though a form is attached to it:
    the form lost to a competing reading there ("nachdem" against "nach dem"),
    and writing it would present a coin flip as a decision. Only a form from the
    store — which is never REVIEW — or an uncontested one counts.

    `context` is no answer either, for the opposite reason: the store has
    looked at the pair and concluded that no corpus-wide answer exists. It is
    not in `FORM_TO_BREAK` or `SEPARATING_FORMS` and so falls through to the
    rule below, which is the right fallback and not an accident — until the
    occurrence store is consulted here, a `cert='low'` mark stating what the
    import already does is exactly what an unanswered break should carry.
    """
    if next_line is None:
        return BREAK_UNKNOWN, SRC_PARA_FINAL

    next_word = get_next_word(next_line)
    # A guard, not an opinion, which is why it runs before the evidence: `geb.`,
    # `v.`, `gew.`, `z.` and `(` cannot continue a word or be the second half of
    # a compound under any reading, so the mark in front of one of them is OCR
    # noise and the line ends its token. The frequency tables have no way of
    # knowing this and do get it wrong — "Schaumburg-Lippe¬ | geb. Prinzessin"
    # is judged `join` because the pair is only ever *seen* hyphenated (the
    # bigram lexicon is built from line-interior tokens, where this phrase never
    # lands), and marking that would join it to "Schaumburg-Lippegeb".
    if next_word in OCR_SPACE_PATTERNS:
        return None, SRC_GUARD

    if form is not None and verdict != 'REVIEW':
        if form in FORM_TO_BREAK:
            return (FORM_TO_BREAK[form],
                    SRC_DECISION if decided_by else SRC_EVIDENCE)
        if form in SEPARATING_FORMS:
            # The mark should not be here at all. Reported, never encoded.
            return None, form

    # No answer: an undecided pair, a contested one, one explicitly declared
    # unresolvable, or a break too short to judge. What is left is the rule the
    # import has always used, so a `cert='low'` mark states what the database
    # text already says rather than a third opinion nothing else shares.
    #
    # The break character is deliberately not consulted, because it is the
    # weaker signal of the two: a line-final `-` before a lowercase word is a
    # continuation the OCR read as a hyphen ("Kunst-" + "gewerbemuseum" is one
    # word, not a compound), and `¬` before a capitalised one is just as often a
    # compound whose own hyphen the OCR read as a continuation mark
    # ("Arcièren¬" + "Leibgarde"). Which is why the analysis refuses to decide
    # that class at all (`decide_form`'s 'capitalised-right') and why it is by
    # far the largest part of this bucket.
    action = determine_hyphen_action(char, next_word)
    return ACTION_TO_BREAK[action], SRC_RULE


# ============================================================================
# Marking one file
# ============================================================================

def clear_marks(paragraph: etree._Element) -> None:
    """
    Drop this script's attributes from every line of a paragraph.

    Recomputing from a clean slate is what makes a re-run converge: a line whose
    pair has since been settled must lose its `cert='low'`, and one that has
    since been corrected must lose its mark entirely. `@cert` is only removed
    where a `@break` stood, so nothing else that might come to use it is
    disturbed.
    """
    for line in paragraph.iterfind('.//' + tei('l')):
        if line.get(BREAK_ATTR) is not None:
            del line.attrib[BREAK_ATTR]
            if line.get(CERT_ATTR) is not None:
                del line.attrib[CERT_ATTR]


def mark_tree(tree: etree._ElementTree, filename: str, pairs,
              skip_subtypes: Set[str] = frozenset(),
              report_all: bool = False,
              ) -> Tuple[Counter, List[dict], str]:
    """
    Mark every break-marked line end in one parsed file.

    Returns `(counters, rows for the report, notation detected)`. The tree is
    modified in place; the header marker and the write are the caller's
    business, which is what lets `--dry-run` do everything but the last step.

    The notation decides nothing here — no break type is read off the break
    character (see `break_type`) — but it goes in the report, because whether
    the `¬` in front of a reviewer is meaningful notation or a Fraktur `=` that
    says nothing about the two readings is the first thing they need to know.
    """
    root = tree.getroot()

    # Read off every line, not just the ones carrying a mark: the majority over
    # the whole file is what makes the verdict stable.
    notation = detect_notation(
        ''.join(el.itertext()) for el in root.iterfind('.//' + tei('l')))

    # Region id -> subtype, so a whole layout class can be held back and so the
    # report can say which kind of region a mark sits in.
    subtypes: Dict[str, str] = {}
    for zone in root.xpath("//tei:zone[@rendition='TextRegion']", namespaces=NS):
        zone_id = zone.get(XML_ID)
        if zone_id:
            subtypes[zone_id] = zone.get('subtype', '')

    counts: Counter = Counter()
    rows: List[dict] = []
    page = 0

    # `<pb/>` and `<p>` interleave in document order, so one walk gives every
    # paragraph the page it sits on.
    for element in root.iter(tei('pb'), tei('p')):
        if element.tag == tei('pb'):
            facs = (element.get('facs') or '').lstrip('#')
            if facs.startswith('facs_') and facs[5:].isdigit():
                page = int(facs[5:])
            continue

        region = (element.get('facs') or '').lstrip('#')
        subtype = subtypes.get(region, '')
        if subtype in skip_subtypes:
            counts['paragraphs-skipped'] += 1
            continue

        clear_marks(element)

        # Faithful to `read_paragraphs`, which is what the pair table was built
        # from: a line's text is all of its text nodes, and an empty line is not
        # a line. Unlike the two correction scripts this one has no reason to
        # skip a line carrying markup — it only reads the text and writes an
        # attribute, so a `<hi>` inside an `<l>` costs nothing here.
        lines = [(el, text) for el, text in
                 ((el, ''.join(el.itertext()).strip())
                  for el in element.iterfind('.//' + tei('l')))
                 if text]

        for i, (line, text) in enumerate(lines):
            if not text.endswith(HYPHEN_CHARS):
                continue
            char = text[-1]
            paragraph_final = (i == len(lines) - 1)

            next_line = None if paragraph_final else lines[i + 1][1]

            form: Optional[str] = None
            verdict = ''
            decided_by = ''
            left = right = ''
            if next_line is not None:
                pair = break_pair(text, next_line)
                if pair is not None:
                    left, right = pair[0], pair[1]
                    evidence = pairs.get((left, right))
                    if evidence is not None:
                        form = evidence.form
                        verdict = evidence.verdict
                        decided_by = evidence.decided_by

            value, source = break_type(char, form, verdict, decided_by,
                                       next_line)
            if value is None:
                # No break type: either an error the correction step has not
                # applied, or the rule reading the mark as OCR noise. Counted
                # and reported, never encoded.
                counts['unmarked'] += 1
                counts[f'unmarked:{source}'] += 1
            else:
                cert = 'low' if source in LOW_CERTAINTY_SOURCES else ''
                line.set(BREAK_ATTR, value)
                if cert:
                    line.set(CERT_ATTR, cert)
                counts[f'break:{value}'] += 1
                counts[f'source:{source}'] += 1
                if cert:
                    counts['cert-low'] += 1

            if not report_all and source in (SRC_EVIDENCE, SRC_DECISION):
                continue
            rows.append({
                'file': filename,
                'page': page,
                'region': region,
                'subtype': subtype,
                # A `<l>` carries no xml:id of its own; it points at the Line
                # zone in `<facsimile>`, and that id is what locates it.
                'line': (line.get('facs') or '').lstrip('#'),
                'char': char,
                'break': value or '',
                'cert': 'low' if source in LOW_CERTAINTY_SOURCES else '',
                'source': source,
                'left': left,
                'right': right,
                'form': form or '',
                'verdict': verdict,
                'notation': notation,
                'text': text,
                'link': _link(filename, page),
            })

    return counts, rows, notation


def _link(filename: str, page: int) -> str:
    """A Transkribus deep link for the report, empty when unavailable."""
    if get_transkribus_link is None:
        return ''
    try:
        return get_transkribus_link(filename, page or None)
    except Exception:      # pragma: no cover - a link is a convenience only
        return ''


# ============================================================================
# Command line
# ============================================================================

REPORT_FIELDS = ['file', 'page', 'region', 'subtype', 'line', 'char', 'break',
                 'cert', 'source', 'left', 'right', 'form', 'verdict',
                 'notation', 'text', 'link']


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Mark split words and compound words at the end of a line '
                    'in the TEI-XML files.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--decisions', type=Path,
                        default=PATHS['csv'] / 'hyphen_decisions.csv')
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--ratio', type=float, default=3.0)
    parser.add_argument('--skip-subtypes',
                        default=','.join(SKIPPED_SUBTYPES),
                        help='Comma-separated TextRegion subtypes to leave '
                             "untouched, e.g. 'ad,ad-frame'")
    parser.add_argument('--force', action='store_true',
                        help='Re-mark files already at this version')
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', default='word_break_marks.csv')
    parser.add_argument('--report-all', action='store_true',
                        help='Report confident marks as well, not only the '
                             'ones needing attention')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    skip_subtypes = {s.strip() for s in args.skip_subtypes.split(',')
                     if s.strip()}

    decisions = load_decisions(args.decisions)
    if not args.quiet:
        print(f"Decision store: {len(decisions):,} pairs settled "
              f"({args.decisions})")

    # The analysis always runs over the whole corpus - the lexicon must not
    # shrink just because only a few files are being marked.
    pairs, _ = analyse(args.xml_folder, None, args.ratio, args.quiet,
                       decisions)

    files = sorted(args.xml_folder.glob('*.xml'))
    if args.max_files:
        files = files[:args.max_files]

    totals: Counter = Counter()
    notations: Counter = Counter()
    all_rows: List[dict] = []
    files_marked = 0
    files_skipped = 0
    when = datetime.date.today().isoformat()

    for n, xml_path in enumerate(files, 1):
        tree = parse_tei(xml_path)
        if not args.force and is_up_to_date(tree, APP_IDENT, VERSION):
            files_skipped += 1
            continue

        counts, rows, notation = mark_tree(
            tree, xml_path.name, pairs, skip_subtypes, args.report_all)
        notations[notation] += 1
        totals.update(counts)
        all_rows.extend(rows)
        if any(key.startswith('break:') for key in counts):
            files_marked += 1

        if not args.dry_run:
            # `wsb:break` needs its prefix declared on the root, or lxml
            # invents one at the point of use, halfway down the document.
            ensure_wsb_namespace(tree)
            # The marker goes on even when this file happened to need no mark:
            # what it records is that the pass ran, which is what makes the
            # absence of a `@break` a statement rather than a gap.
            set_marker(
                tree, APP_IDENT, VERSION, when,
                label=MARKER_LABEL,
                change=f'Word breaks marked on <l> (@wsb:break/@cert) '
                       f'({APP_IDENT} {VERSION}).')
            if args.backup:
                backup_path = xml_path.with_suffix('.xml.bak')
                if not backup_path.exists():
                    shutil.copy2(xml_path, backup_path)
            write_with_single_quotes(tree, xml_path)

        if not args.quiet and n % 100 == 0:
            so_far = sum(v for k, v in totals.items()
                         if k.startswith('break:'))
            print(f"  {n}/{len(files)} files, {so_far:,} marks")

    _report(args, files, files_marked, files_skipped, totals, notations,
            all_rows)
    return 0


def _report(args, files, files_marked, files_skipped, totals, notations,
            all_rows) -> None:
    """Print the summary and write the per-mark CSV."""
    marked = sum(v for k, v in totals.items() if k.startswith('break:'))

    print("\n" + "=" * 62)
    print("DRY RUN - nothing written" if args.dry_run else "XML files updated")
    print("=" * 62)
    print(f"  files marked    {files_marked:,} / {len(files):,}")
    if files_skipped:
        print(f"  files skipped   {files_skipped:,} "
              f"(already at version {VERSION}; --force to re-mark)")
    print(f"  line ends marked {marked:,}")
    for value in (BREAK_WORD, BREAK_COMPOUND, BREAK_UNKNOWN):
        label = f"break='{value}'"
        print(f"    {label:20s} {totals.get(f'break:{value}', 0):8,}")
    print(f"  of those cert='low' {totals.get('cert-low', 0):,} "
          f"(the import's rule, no evidence either way)")
    print("  by source")
    for source in (SRC_EVIDENCE, SRC_DECISION, SRC_RULE, SRC_PARA_FINAL):
        count = totals.get(f'source:{source}', 0)
        if count:
            print(f"    {source:22s} {count:8,}")

    # Two quite different reasons a marked line end gets no break type, and only
    # the first is a to-do.
    pending = sum(totals.get(f'unmarked:{form}', 0)
                  for form in SEPARATING_FORMS)
    if pending:
        detail = ', '.join(
            f"{form} {totals[f'unmarked:{form}']:,}" for form in SEPARATING_FORMS
            if totals.get(f'unmarked:{form}'))
        print(f"\n  left unmarked, correction pending  {pending:,}")
        print(f"    a mark the evidence reads as separate words ({detail}) -")
        print("    run correct_xml_hyphens.py, then mark again.")
    noise = (totals.get(f'unmarked:{SRC_GUARD}', 0)
             + totals.get(f'unmarked:{SRC_RULE}', 0))
    if noise:
        print(f"\n  left unmarked, read as OCR noise   {noise:,}")
        print("    the next word cannot be a word-part at all "
              "('Eugen¬ | v. Balogh'),")
        print("    so the mark is a stray one and the line ends its token.")
    if totals.get('paragraphs-skipped'):
        print(f"\n  paragraphs held back by --skip-subtypes: "
              f"{totals['paragraphs-skipped']:,}")

    mix = ', '.join(f"{char} {count:,}" for char, count in
                    notations.most_common())
    if mix:
        print(f"\n  notation        {mix}  (detected per file, reported only)")

    if all_rows:
        date_prefix = datetime.date.today().strftime('%Y%m%d')
        report_path = PATHS['output'] / f"{date_prefix}_{args.report}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        scope = 'every mark' if args.report_all else 'marks needing attention'
        print(f"\nPer-mark report ({scope}): {report_path}")


if __name__ == '__main__':
    raise SystemExit(main())
