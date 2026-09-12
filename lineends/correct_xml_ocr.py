#!/usr/bin/env python3
"""
XML OCR Correction
==================

Applies OCR corrections directly to the Transkribus XML files in `data/xml/`.
The string-level correction logic is shared with
`prepare_factoids.py` (downstream, not shipped) (which acts on the database after import)
via `utils.ocr_text_corrections`, so both apply exactly the same rules and the
XML files themselves are also usable outside of the pipeline.

Operations:
    1. Delete wrongly detected text lines inside image regions
       - Removes `<l>` elements in the <text> section that belong to a
         TextRegion with `subtype='image'`
       - Removes `Line` sub-zones inside image TextRegion zones in <facsimile>
       - Removes empty `<lg>` and `<p>` wrappers left behind
    2. Correct common transcription errors using `data/csv/replacement.csv`
    3. Correct OCR digit errors (1→r, 0→o, 2→z, 5→s) in:
       - Year tokens (1700–1930) preceded by a space
       - Day tokens (1–31) preceded by a space and followed by '.'
       - "Nummer N[N]" / "Seite N[N]" (1–99)
    4. Insert a missing dot in times before " Uhr": 3- or 4-digit run with
       no leading digit, 12-hour style (hour 1–12, minute 00–59),
       e.g. "730 Uhr" → "7.30 Uhr", "1145 Uhr" → "11.45 Uhr"
    5. Insert a missing slash in two-year date ranges where the OCR dropped
       it, e.g. "191415" → "1914/15" (first year 1860–1938, second part the
       following consecutive year)
    6. Apply the positional word fixes in `data/csv/ocr_corrections.csv`
       (see `ocr_corrections.py`): a word misread at a line break, or a
       line-final character misread, both of which are wrong only *there* and
       so cannot go into `replacement.csv`

Corrections are found on the joined paragraph text
--------------------------------------------------
Operations 2–5 used to run on each `<l>` element separately, and that missed two
whole classes of correction, which is why `prepare_factoids.py` (downstream, not shipped) had
to run the same list a second time over the joined factoid text in the database:

    a word split across the break     "Vortrage" as "Vor¬" + "trage"; neither
                                      line contains the string a rule matches

    a space-padded rule at a line     " österreich" cannot match a token that
    edge                              starts an `<l>` element, because the
                                      space before it is the line break

Measured over 120 files, 450 corrections in 128 rules fell into those two
classes — around 3,000 corpus-wide — against 53 that a line-by-line pass finds.
So this script now joins each paragraph exactly as the import step (not shipped)
will (`hyphen_utils.join_lines_with_origin`), corrects that text through the same
`utils.ocr_text_corrections.correct_text` the database step uses, then *diffs*
the result to find where each correction landed and writes it back into the one
`<l>` element it belongs to. The line structure of the transcription is
preserved: only the line actually holding the changed characters is rewritten.

Why a diff rather than locating each rule's match: `replacement.csv` restores
diacritics in stages (`Lancut -> Lańcut -> Łańcut`), so the rules have to be
applied in order, over each other's output. There is no set of independent match
positions to collect — but there is a before and an after, and that is enough.

Four things can happen to a correction (all recorded in the report):

    applied           the changed characters sit in one line — the great
                      majority, and free: the padding spaces of a rule like
                      " Vortrage " land in the unchanged parts of the diff.
    join-artifact     the rule wants to change the space or hyphen that the
                      *join* introduced at the break, e.g. "Drag. Rgt" ->
                      "Drag.-Rgt". No `<l>` element contains that character, and
                      writing one would be deciding the line ends in `-` —
                      which is `correct_xml_hyphens.py`'s verdict to make, from
                      evidence, not a substring rule's. Skipped.
    straddles-break   the changed characters themselves span the break. The
                      replacement goes to the line holding most of them (a pure
                      insertion at the boundary goes to the left line) and the
                      rest are deleted from the other. Rare, so read them.
    changes-break     the correction changes what the *join itself reads*, and
                      the whole paragraph is skipped. See `correct_paragraph`.

The line-end hyphen character is never touched here; that is
`correct_xml_hyphens.py`'s business, and the two stay disjoint.

Every paragraph is rejoined and checked after its edits are planned, so what
lands in the XML always rejoins to exactly what `correct_text` produced — the
parity with the database step is enforced, not assumed.

Usage:
    python correct_xml_ocr.py --dry-run
    python correct_xml_ocr.py
    python correct_xml_ocr.py --max-files 10
    python correct_xml_ocr.py --min-confidence medium --dry-run
    python correct_xml_ocr.py --csv path/to/replacement.csv

Flags:
    --xml-folder P    Path to XML files folder (default: PATHS['xml']).
    --csv PATH        Path to replacement CSV (default: PATHS['csv']/replacement.csv).
    --corrections P   Positional word-fix store (default:
                      PATHS['csv']/ocr_corrections.csv). Pass '' to skip it.
    --min-confidence L
                      Lowest confidence to apply from that store without an
                      explicit approval: high (default), medium or low.
                      `approved=yes` applies regardless; `approved=no` never
                      applies.
    --approved-only   From that store, apply only rows approved by hand.
    --max-files N     Maximum number of files to process.
    --skip-subtypes S Comma-separated region subtypes to leave untouched
                      (default: none; 'ad,ad-frame' holds back the noisiest).
    --backup          Write <name>.xml.bak before modifying a file.
    --dry-run         Report changes without writing files.
    --report NAME     Per-change CSV written to output/ (default:
                      ocr_corrections_applied.csv).
    --quiet           Suppress progress messages.

Author: Christian Lendl
Created: 2026-05-18
Last Modified: 2026-08-14
"""

import argparse
import csv
import datetime
import difflib
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from lxml import etree
from tqdm import tqdm


# Add parent directory to path for config + shared utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS  # noqa: E402
from validate_xml_hyphens import SKIPPED_SUBTYPES  # noqa: E402
# Shared, string-level OCR correction logic (also used by
# the database import step (not shipped)) lives in utils.ocr_text_corrections.
from utils.ocr_text_corrections import (  # noqa: E402
    correct_text,
    load_replacement_dict,
)
from hyphen_utils import (  # noqa: E402
    join_lines_with_hyphen_cleanup,
    join_lines_with_origin,
)
from ocr_corrections import (  # noqa: E402
    CONFIDENCE_LEVELS,
    WILDCARD,
    Correction,
    load as load_corrections,
)
from validate_xml_hyphens import (  # noqa: E402
    HYPHEN_CHARS,
    MIN_TOKEN_LEN,
    STRIP_CHARS,
    read_subtypes,
    strip_token,
)
# Shared TEI serialization, so every lineends script writes the files back
# in the single-quoted style Transkribus emits.
from xml_io import write_with_single_quotes  # noqa: E402


# ============================================================================
# Configuration
# ============================================================================

TEI_NS = 'http://www.tei-c.org/ns/1.0'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NS = {'tei': TEI_NS}

DEFAULT_XML_FOLDER = str(PATHS['xml'])
DEFAULT_CSV = PATHS['csv'] / 'replacement.csv'
DEFAULT_CORRECTIONS = PATHS['csv'] / 'ocr_corrections.csv'

# What became of one correction found on the joined paragraph text
APPLIED = 'applied'
JOIN_ARTIFACT = 'join-artifact'
STRADDLES = 'straddles-break'
CHANGES_BREAK = 'changes-break'


# ============================================================================
# XML processing
# ============================================================================

def _strip_facs_hash(facs_attr: Optional[str]) -> Optional[str]:
    """Strip leading '#' from a facs attribute (e.g. '#facs_1_tr_1' → 'facs_1_tr_1')."""
    if facs_attr is None:
        return None
    return facs_attr.lstrip('#') or None


def remove_image_region_lines(root: etree._Element) -> Tuple[int, int, int]:
    """
    Remove text content that belongs to image regions.

    - In <text>: removes <l> elements whose facs points to a child of an
      image-subtype TextRegion. Empty <lg> and <p> wrappers are removed too.
    - In <facsimile>: removes 'Line' sub-zones inside image TextRegion zones.

    Returns:
        Tuple of (lines_removed_in_text, line_zones_removed_in_facs,
                  paragraphs_removed_in_text).
    """
    # Collect xml:ids of image TextRegion zones
    image_zones = root.xpath(
        "//tei:zone[@rendition='TextRegion' and @subtype='image']",
        namespaces=NS,
    )
    image_zone_ids = {
        zone.get(f'{{{XML_NS}}}id') for zone in image_zones
        if zone.get(f'{{{XML_NS}}}id')
    }

    # Map every line-zone xml:id to the parent TextRegion xml:id so that
    # individual <l> elements can be matched against an image region.
    line_zone_to_region: Dict[str, str] = {}
    for region in root.xpath(
        "//tei:zone[@rendition='TextRegion']",
        namespaces=NS,
    ):
        region_id = region.get(f'{{{XML_NS}}}id')
        if not region_id:
            continue
        for line_zone in region.xpath(".//tei:zone[@rendition='Line']", namespaces=NS):
            line_id = line_zone.get(f'{{{XML_NS}}}id')
            if line_id:
                line_zone_to_region[line_id] = region_id

    # 1. Remove Line sub-zones inside image TextRegion zones
    line_zones_removed = 0
    for zone in image_zones:
        for line_zone in zone.xpath(".//tei:zone[@rendition='Line']", namespaces=NS):
            line_zone.getparent().remove(line_zone)
            line_zones_removed += 1

    # 2. Remove <l> elements in <text> whose region is an image region
    lines_removed = 0
    for l_elem in root.xpath("//tei:l", namespaces=NS):
        facs_id = _strip_facs_hash(l_elem.get('facs'))
        if not facs_id:
            continue
        region_id = line_zone_to_region.get(facs_id)
        if region_id and region_id in image_zone_ids:
            l_elem.getparent().remove(l_elem)
            lines_removed += 1

    # 3. Clean up empty <lg> elements
    for lg in root.xpath("//tei:lg", namespaces=NS):
        if len(lg) == 0 and (lg.text is None or not lg.text.strip()):
            lg.getparent().remove(lg)

    # 4. Remove <p> elements that point to image regions and are now empty,
    #    plus any other <p> elements left empty by the line removals.
    paragraphs_removed = 0
    for p in root.xpath("//tei:p", namespaces=NS):
        facs_id = _strip_facs_hash(p.get('facs'))
        is_image_p = facs_id in image_zone_ids if facs_id else False
        is_empty = len(p) == 0 and (p.text is None or not p.text.strip())
        if is_image_p or is_empty:
            p.getparent().remove(p)
            paragraphs_removed += 1

    return lines_removed, line_zones_removed, paragraphs_removed


def line_elements(paragraph: etree._Element) -> List[etree._Element]:
    """
    The `<l>` elements of one paragraph that hold plain text.

    A line containing markup is left out, the same way `correct_xml_hyphens.py`
    leaves it alone: rewriting it would mean deciding which text node a
    correction belongs to. No line in the corpus has any (0 of 147,011 checked),
    so this is a guard rather than a common case — but a paragraph holding one
    is skipped entirely by `correct_paragraph`, since dropping a line from the
    middle would silently join the wrong words together.
    """
    return [el for el in paragraph.xpath('.//tei:l', namespaces=NS)
            if len(el) == 0 and (el.text or '').strip()]


def resolve_edit(origin, start: int, end: int
                 ) -> Tuple[Optional[str], Dict[int, Tuple[int, int]]]:
    """
    Work out which lines a range of the joined text belongs to.

    Args:
        origin: the map from `join_lines_with_origin`
        start, end: a half-open range of *changed* characters in joined
            coordinates. `start == end` is a pure insertion point.

    Returns:
        Tuple of (outcome, {line_index: (start_offset, end_offset)}), where
        outcome is None when the edit can be written, `JOIN_ARTIFACT` when it
        cannot, and `STRADDLES` when it can but spans two lines. The offsets are
        into the *stripped* line text, which is what was joined.
    """
    if start == end:
        # An insertion has no characters of its own, so it borrows a position
        # from its neighbours. The left one is preferred: for "Savoyensche" ->
        # "Savoyen'sche" broken as "Savoyen¬" + "sche", appending to the left
        # line and prepending to the right both rejoin correctly, and appending
        # keeps the inserted character on the line whose word it completes.
        left = origin[start - 1] if start > 0 else (None, None)
        if left[0] is not None:
            return None, {left[0]: (left[1] + 1, left[1] + 1)}
        right = origin[start] if start < len(origin) else (None, None)
        if right[0] is not None:
            return None, {right[0]: (right[1], right[1])}
        return JOIN_ARTIFACT, {}

    spans: Dict[int, Tuple[int, int]] = {}
    for k in range(start, end):
        line_index, offset = origin[k]
        if line_index is None:
            # The rule is reaching for the space or hyphen the join invented;
            # no line contains it. See the module docstring.
            return JOIN_ARTIFACT, {}
        if line_index in spans:
            spans[line_index] = (spans[line_index][0], offset + 1)
        else:
            spans[line_index] = (offset, offset + 1)

    return (None if len(spans) == 1 else STRADDLES), spans


def plan_line_edits(before: str, after: str, origin
                    ) -> Tuple[List[Tuple[int, int, int, str]], Counter,
                               List[Tuple[str, str, str]]]:
    """
    Turn "this paragraph should read like that" into per-line rewrites.

    `difflib` reduces the two versions to the characters that actually differ,
    which is what makes the common case free: a rule like " Vortrage " ->
    " Vorträge " matches across a line break, but the one character it changes
    sits on a single line, and the padding spaces are in the unchanged parts.

    Returns:
        Tuple of (edits, outcome counts, skipped) where each edit is
        (line_index, start_offset, end_offset, replacement) in stripped-line
        coordinates, and `skipped` holds (before, after, outcome) for the
        corrections that could not be applied.
    """
    edits: List[Tuple[int, int, int, str]] = []
    outcomes: Counter = Counter()
    skipped: List[Tuple[str, str, str]] = []

    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        replacement = after[j1:j2]
        outcome, spans = resolve_edit(origin, i1, i2)

        if outcome == JOIN_ARTIFACT:
            outcomes[JOIN_ARTIFACT] += 1
            skipped.append((before[i1:i2], replacement, JOIN_ARTIFACT))
            continue

        if outcome == STRADDLES:
            # The changed characters themselves cross the break. Give the
            # replacement to the line holding most of them and delete the rest
            # from the others, so the two lines still rejoin to the right text.
            outcomes[STRADDLES] += 1
            main = max(spans, key=lambda i: spans[i][1] - spans[i][0])
            for line_index, (line_start, line_end) in spans.items():
                edits.append((line_index, line_start, line_end,
                              replacement if line_index == main else ''))
            continue

        outcomes[APPLIED] += 1
        line_index, (line_start, line_end) = next(iter(spans.items()))
        edits.append((line_index, line_start, line_end, replacement))

    return edits, outcomes, skipped


def apply_line_edits(lines: List[str],
                     edits: List[Tuple[int, int, int, str]]) -> Dict[int, str]:
    """
    Perform the planned rewrites, latest offset first.

    Working backwards keeps every remaining offset valid without recomputing
    anything. Two edits overlapping on one line cannot both be right, so the
    later one is dropped; `difflib` opcodes never overlap, but a straddling edit
    contributes a span per line and the guard costs nothing.

    Returns:
        {line_index: new_text} for the lines that changed.
    """
    changed: Dict[int, str] = {}
    by_line: Dict[int, List[Tuple[int, int, str]]] = {}
    for line_index, start, end, replacement in edits:
        by_line.setdefault(line_index, []).append((start, end, replacement))

    for line_index, line_edits in by_line.items():
        text = lines[line_index]
        covered_from = len(text) + 1
        for start, end, replacement in sorted(line_edits, reverse=True):
            if end > covered_from:
                continue
            text = text[:start] + replacement + text[end:]
            covered_from = start
        if text != lines[line_index]:
            changed[line_index] = text
    return changed


def correct_paragraph(paragraph: etree._Element, replacements: Dict[str, str],
                      filename: str
                      ) -> Tuple[List[dict], Counter]:
    """
    Correct one `<p>` element on its joined text, writing back per line.

    Returns:
        Tuple of (change records, outcome counts).
    """
    elements = line_elements(paragraph)
    if not elements:
        return [], Counter()
    # A paragraph with a markup-carrying line cannot be joined faithfully, so it
    # is left alone entirely rather than joined out of order.
    if len(elements) != len(paragraph.xpath('.//tei:l', namespaces=NS)):
        return [], Counter({'skipped-markup': 1})

    raw = [el.text or '' for el in elements]
    # The import joins `''.join(line.itertext()).strip()`, so the joined text is
    # built from stripped lines; the leading-whitespace width is kept per line so
    # an offset can be mapped back onto the attribute value as it really is.
    lines = [text.strip() for text in raw]
    lead = [len(text) - len(text.lstrip()) for text in raw]

    joined, origin = join_lines_with_origin(lines)
    corrected, _, _ = correct_text(joined, replacements)
    if corrected == joined:
        return [], Counter()

    edits, outcomes, skipped = plan_line_edits(joined, corrected, origin)
    changed = apply_line_edits(lines, edits)

    # Rejoin and check, rather than trust the mapping. A correction can change
    # the very thing the join reads: `join_lines_with_hyphen_cleanup` keeps a
    # line-end hyphen before a capitalised word and drops it before a lowercase
    # one, so a rule that lowercases the first letter of a line silently flips
    # that decision ("Huppmann-" + "Valbella" -> "Huppmann-valócz..." where the
    # join then removes the hyphen and yields "Huppmannvalócz..."). The origin
    # map was built from the lines as they were and cannot see this coming.
    #
    # Rather than repair the break — deciding what a line break means from a
    # substring rule is exactly what the join-artifact case refuses to do — the
    # whole paragraph is left alone and reported. A handful in the whole corpus,
    # and worth reading: the first run of this check found a `replacement.csv`
    # rule that was destroying a real family name everywhere it occurred.
    if changed:
        rejoined = join_lines_with_hyphen_cleanup(
            [changed.get(i, line) for i, line in enumerate(lines)])
        if rejoined != corrected and not outcomes[JOIN_ARTIFACT]:
            # Report the place the two versions part company, not the head of
            # the paragraph — the break in question can be anywhere in it.
            at = next((i for i, (a, b) in enumerate(zip(corrected, rejoined))
                       if a != b), min(len(corrected), len(rejoined)))
            return [{
                'file': filename,
                'region': (paragraph.get('facs') or '').lstrip('#'),
                'line': '',
                'source': 'replacement.csv',
                'outcome': CHANGES_BREAK,
                'before': corrected[max(0, at - 60):at + 40],
                'after': rejoined[max(0, at - 60):at + 40],
                'note': 'not applied: correcting this would change how the '
                        'line break joins',
            }], Counter({CHANGES_BREAK: 1})

    records: List[dict] = []
    region = (paragraph.get('facs') or '').lstrip('#')
    for line_index, new_line in sorted(changed.items()):
        element = elements[line_index]
        before = raw[line_index]
        # Put back exactly the surrounding whitespace that was there
        element.text = (before[:lead[line_index]] + new_line
                        + before[lead[line_index] + len(lines[line_index]):])
        records.append({
            'file': filename,
            'region': region,
            'line': element.get('facs', '').lstrip('#'),
            'source': 'replacement.csv',
            'outcome': APPLIED,
            'before': lines[line_index],
            'after': new_line,
            'note': '',
        })
    for before, after, outcome in skipped:
        records.append({
            'file': filename,
            'region': region,
            'line': '',
            'source': 'replacement.csv',
            'outcome': outcome,
            'before': before,
            'after': after,
            'note': 'not applied',
        })
    return records, outcomes


def correct_paragraph_texts(
    root: etree._Element,
    replacements: Dict[str, str],
    filename: str,
    skip_subtypes: Set[str] = frozenset(),
    subtypes: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], Counter]:
    """
    Apply the replacement and digit corrections to every paragraph.

    Returns:
        Tuple of (change records, outcome counts).
    """
    records: List[dict] = []
    outcomes: Counter = Counter()
    subtypes = subtypes or {}

    for paragraph in root.xpath('//tei:p', namespaces=NS):
        if skip_subtypes:
            region = (paragraph.get('facs') or '').lstrip('#')
            if subtypes.get(region, '') in skip_subtypes:
                continue
        paragraph_records, paragraph_outcomes = correct_paragraph(
            paragraph, replacements, filename)
        records.extend(paragraph_records)
        outcomes.update(paragraph_outcomes)

    return records, outcomes


# ============================================================================
# Positional word fixes (data/csv/ocr_corrections.csv)
# ============================================================================

def build_plan(corrections: Dict[Tuple[str, str], Correction],
               min_confidence: str, approved_only: bool
               ) -> Dict[Tuple[str, str], Correction]:
    """Reduce the store to the rows that are actually allowed to apply."""
    if approved_only:
        return {key: c for key, c in corrections.items()
                if c.approved == 'yes' and not c.is_noop}
    return {key: c for key, c in corrections.items()
            if c.applies(min_confidence)}


def replace_token(raw: str, stripped: str, corrected: str) -> str:
    """
    Swap the word inside a surface token, keeping its punctuation.

    `stripped` was produced from `raw` by removing STRIP_CHARS at both ends (see
    `validate_xml_hyphens.strip_token`), so the word sits at a known offset and
    the brackets, quotes and full stops around it survive: "(Antra)" ->
    "(Antrag)".
    """
    prefix = len(raw) - len(raw.lstrip(STRIP_CHARS))
    suffix = len(raw) - len(raw.rstrip(STRIP_CHARS))
    body = raw[prefix:len(raw) - suffix]
    if body != stripped:          # not the token we think it is; leave it
        return raw
    return raw[:prefix] + corrected + (raw[len(raw) - suffix:] if suffix
                                       else '')


def correct_word_pairs(root: etree._Element, plan, filename: str,
                       skip_subtypes: Set[str] = frozenset(),
                       subtypes: Optional[Dict[str, str]] = None
                       ) -> List[dict]:
    """
    Apply the positional word fixes from `ocr_corrections.py`.

    Walks every line of every paragraph. For a line with a successor, the pair
    (last token, first token of the next line) is looked up first — the same
    break `correct_xml_hyphens.py` works on, but rewriting the words rather than
    the character between them. Failing that, and only where the line does not
    end in a hyphen, the last token is looked up on its own as a wildcard row,
    which is how a line-final character misread is fixed even on the last line of
    a paragraph.

    The line-end hyphen is read to decide whether a wildcard row may apply, and
    is then put back exactly as it was found. Deciding *which* character belongs
    there is `correct_xml_hyphens.py`'s job, from evidence.

    Returns:
        List of change records.
    """
    changes: List[dict] = []
    subtypes = subtypes or {}

    for paragraph in root.xpath('//tei:p', namespaces=NS):
        region = (paragraph.get('facs') or '').lstrip('#')
        if skip_subtypes and subtypes.get(region, '') in skip_subtypes:
            continue
        elements = line_elements(paragraph)

        for i, element in enumerate(elements):
            following_element = elements[i + 1] if i + 1 < len(elements) \
                else None
            current = (element.text or '').strip()
            if not current.split():
                continue

            had_hyphen = current.endswith(HYPHEN_CHARS)
            hyphen = current[-1] if had_hyphen else ''
            body = current[:-1].rstrip() if had_hyphen else current
            if not body.split():
                continue

            raw_left = body.split()[-1]
            left = strip_token(raw_left)
            if len(left) < MIN_TOKEN_LEN:
                continue

            following = ((following_element.text or '').strip()
                         if following_element is not None else '')
            raw_right = following.split()[0] if following.split() else ''
            right = strip_token(raw_right)

            # An exact pair row is the more specific statement and wins. The
            # wildcard is only consulted for a line that really ends here: on a
            # hyphenated line the token is a fragment, not the word it names.
            correction = None
            if right and len(right) >= MIN_TOKEN_LEN:
                correction = plan.get((left, right))
            if correction is None and not had_hyphen:
                correction = plan.get((left, WILDCARD))
            if correction is None:
                continue

            new_left, new_right = raw_left, raw_right
            if correction.changes_left:
                new_left = replace_token(raw_left, left,
                                         correction.left_correct)
            if correction.changes_right:
                new_right = replace_token(raw_right, right,
                                          correction.right_correct)
            if new_left == raw_left and new_right == raw_right:
                continue

            record = {
                'file': filename,
                'region': region,
                'source': 'ocr_corrections.csv',
                'outcome': APPLIED,
                'note': f"[{correction.confidence}/{correction.source}] "
                        f"{correction.note}"[:200],
            }

            if new_left != raw_left:
                # Only the final token is rewritten, so an identical word
                # earlier in the same line is left alone.
                head = body[:len(body) - len(raw_left)]
                new_current = head + new_left + hyphen
                changes.append(dict(
                    record, line=element.get('facs', '').lstrip('#'),
                    before=current, after=new_current))
                element.text = new_current
                current = new_current

            if new_right != raw_right:
                tail = following[len(raw_right):]
                new_following = new_right + tail
                changes.append(dict(
                    record,
                    line=following_element.get('facs', '').lstrip('#'),
                    before=following, after=new_following))
                following_element.text = new_following

    return changes


# ============================================================================
# Per-file driver
# ============================================================================

def process_xml_file(
    xml_path: Path,
    replacements: Dict[str, str],
    plan: Optional[Dict[Tuple[str, str], Correction]] = None,
    dry_run: bool = False,
    skip_subtypes: Set[str] = frozenset(),
    backup: bool = False,
) -> Tuple[Dict[str, int], List[dict]]:
    """
    Process a single XML file: remove image-region lines and apply OCR
    corrections.

    Returns:
        Tuple of (per-file statistics, change records).
    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    lines_removed, line_zones_removed, paragraphs_removed = \
        remove_image_region_lines(root)

    # Region subtypes come from <facsimile>; read once per file, and only when
    # something is actually being held back.
    subtypes = (read_subtypes(xml_path.read_text(encoding='utf-8'))
                if skip_subtypes else {})

    records, outcomes = correct_paragraph_texts(
        root, replacements, xml_path.name, skip_subtypes, subtypes)
    if plan:
        records.extend(correct_word_pairs(
            root, plan, xml_path.name, skip_subtypes, subtypes))

    lines_changed = sum(1 for r in records if r['outcome'] == APPLIED)
    stats = {
        'lines_removed': lines_removed,
        'line_zones_removed': line_zones_removed,
        'paragraphs_removed': paragraphs_removed,
        'lines_changed': lines_changed,
        'join_artifacts_skipped': outcomes[JOIN_ARTIFACT],
        'straddling_edits': outcomes[STRADDLES],
        'break_changing_skipped': outcomes[CHANGES_BREAK],
        'paragraphs_skipped_markup': outcomes['skipped-markup'],
    }

    # A skipped correction is worth reporting but is not a change to the file,
    # so it must not trigger a rewrite of an otherwise untouched document.
    any_change = bool(lines_changed or lines_removed or line_zones_removed
                      or paragraphs_removed)
    if any_change and not dry_run:
        if backup:
            backup_path = xml_path.with_suffix('.xml.bak')
            if not backup_path.exists():
                shutil.copy2(xml_path, backup_path)
        write_with_single_quotes(tree, xml_path)

    return stats, records


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Apply OCR corrections directly to Transkribus XML files',
    )
    parser.add_argument(
        '--xml-folder',
        default=DEFAULT_XML_FOLDER,
        help=f'Path to XML files folder (default: {DEFAULT_XML_FOLDER})',
    )
    parser.add_argument(
        '--csv',
        default=str(DEFAULT_CSV),
        help=f'Replacement CSV (default: {DEFAULT_CSV})',
    )
    parser.add_argument(
        '--corrections',
        default=str(DEFAULT_CORRECTIONS),
        help=f'Positional word-fix store (default: {DEFAULT_CORRECTIONS}); '
             f"pass '' to skip it",
    )
    parser.add_argument(
        '--min-confidence',
        default='high',
        choices=list(CONFIDENCE_LEVELS),
        help='Lowest confidence to apply from the correction store without an '
             'explicit approval (default: high)',
    )
    parser.add_argument(
        '--approved-only',
        action='store_true',
        help='From the correction store, apply only rows with approved=yes',
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process',
    )
    parser.add_argument(
        '--skip-subtypes',
        default=','.join(SKIPPED_SUBTYPES),
        help='Comma-separated region subtypes to leave untouched, e.g. '
             "'ad,ad-frame'",
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Write <name>.xml.bak before modifying a file',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report changes without writing files',
    )
    parser.add_argument(
        '--report',
        default='ocr_corrections_applied.csv',
        help='Per-change CSV written to output/',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages',
    )
    args = parser.parse_args()

    xml_folder = Path(args.xml_folder)
    if not xml_folder.exists():
        print(f"Error: XML folder not found: {xml_folder}")
        return 1

    xml_files = sorted(xml_folder.glob('*.xml'))
    if not xml_files:
        print(f"No XML files found in {xml_folder}")
        return 0

    if args.max_files is not None and args.max_files > 0:
        xml_files = xml_files[:args.max_files]

    skip_subtypes = {s.strip() for s in args.skip_subtypes.split(',')
                     if s.strip()}
    replacements = load_replacement_dict(Path(args.csv))

    # The positional store is optional: on a first run it does not exist yet,
    # and the replacement rules are useful on their own.
    plan: Dict[Tuple[str, str], Correction] = {}
    corrections: Dict[Tuple[str, str], Correction] = {}
    if args.corrections:
        corrections = load_corrections(Path(args.corrections))
        plan = build_plan(corrections, args.min_confidence, args.approved_only)

    if not args.quiet:
        print("=" * 70)
        print("XML OCR CORRECTION")
        print("=" * 70)
        print(f"XML folder:        {xml_folder}")
        print(f"Replacement CSV:   {args.csv}  ({len(replacements)} rules)")
        if args.corrections:
            gate = ('approved=yes only' if args.approved_only
                    else f'approved=yes, or confidence >= '
                         f'{args.min_confidence}')
            wild = sum(1 for c in plan.values() if c.is_wildcard)
            print(f"Correction store:  {args.corrections}  "
                  f"({len(corrections)} rows)")
            print(f"  applying:        {len(plan)} rows "
                  f"({len(plan) - wild} pair, {wild} line-final)  [{gate}]")
        else:
            print("Correction store:  (skipped)")
        if skip_subtypes:
            print(f"Skipped subtypes:  {', '.join(sorted(skip_subtypes))}")
        print(f"Files to process:  {len(xml_files)}")
        if args.dry_run:
            print("Mode:              DRY RUN (no files written)")
        print()

    totals: Counter = Counter()
    all_records: List[dict] = []

    start = time.time()
    for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file",
                         disable=args.quiet):
        try:
            stats, records = process_xml_file(
                xml_file, replacements, plan, dry_run=args.dry_run,
                skip_subtypes=skip_subtypes, backup=args.backup)
        except Exception as exc:
            print(f"\nError processing {xml_file.name}: {exc}")
            continue

        # Only a real modification counts. A skipped join-artifact correction is
        # worth reporting but leaves the file byte-identical, and counting it
        # would make a second, idempotent run claim to have changed files.
        if (stats['lines_changed'] or stats['lines_removed']
                or stats['line_zones_removed'] or stats['paragraphs_removed']):
            totals['files_changed'] += 1
        totals.update(stats)
        all_records.extend(records)

    elapsed = time.time() - start
    by_source = Counter(r['source'] for r in all_records
                        if r['outcome'] == APPLIED)

    if not args.quiet:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Files processed:                  {len(xml_files):,}")
        print(f"Files changed:                    "
              f"{totals['files_changed']:,}")
        print(f"Image-region <l> removed:         "
              f"{totals['lines_removed']:,}")
        print(f"Image-region Line zones removed:  "
              f"{totals['line_zones_removed']:,}")
        print(f"Empty/image <p> removed:          "
              f"{totals['paragraphs_removed']:,}")
        print(f"<l> elements with text corrected: "
              f"{totals['lines_changed']:,}")
        for source, count in by_source.most_common():
            print(f"  from {source:22s}      {count:,}")
        if totals['straddling_edits']:
            print(f"  of which straddle a break:      "
                  f"{totals['straddling_edits']:,}  (read the report)")
        if totals['join_artifacts_skipped']:
            print(f"Corrections NOT applied:          "
                  f"{totals['join_artifacts_skipped']:,}  "
                  f"(would change the line break itself)")
        if totals['break_changing_skipped']:
            print(f"Paragraphs NOT applied:           "
                  f"{totals['break_changing_skipped']:,}  "
                  f"(would change how the line break joins)")
        if totals['paragraphs_skipped_markup']:
            print(f"Paragraphs skipped (markup):      "
                  f"{totals['paragraphs_skipped_markup']:,}")
        print(f"Processing time:                  {elapsed:.2f}s")
        if args.dry_run:
            print("\n(dry run — no XML files were modified)")

    if all_records:
        date_prefix = datetime.date.today().strftime('%Y%m%d')
        report_path = PATHS['output'] / f"{date_prefix}_{args.report}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(
                fh, fieldnames=['file', 'region', 'line', 'source', 'outcome',
                                'before', 'after', 'note'])
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nPer-change report: {report_path}")
        if args.dry_run:
            print("Read it before rerunning without --dry-run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
