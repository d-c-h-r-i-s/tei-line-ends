#!/usr/bin/env python3
"""
The Text of a Region
====================

One definition of what a transcribed region *says*, for everything that needs
it: the XML annotation scripts, the database import, the apps. A region's text
is not simply its `<l>` elements concatenated — the lines break words in half,
and putting them back together is a decision per break.

Until now that decision was made twice. `hyphen_utils.py` inferred it
from the next word (capitalised or `und` or a Hungarian street name keeps the
hyphen, lowercase drops it), and every other consumer either imported that
module or, worse, re-derived something close to it. `mark_xml_word_breaks.py`
has since written the answer into the files themselves:

    <l … break='word'>… Frau Erzherzogin Auguste, Ge¬</l>
    <l …>mahlin des Herrn Erzherzogs Josef, …</l>

so where the marks are present there is nothing left to infer, and joining is
six lines with no rule table and no notation to know about. `region_text()`
reads the marks when the file carries them and falls back to the old rule when
it does not, which keeps the text identical for an unmarked file and correct
for a marked one.

**Absence of a mark is a statement, not a gap** — but only in a file whose
header says the marking pass ran. That is why `use_marks` is decided per file
from the `<application ident='<prefix>-wordbreaks'>` marker rather than per
line from whether an attribute happens to be there: in an unmarked file every
line is unmarked, and reading that as "no line continues a word" would silently
glue the corpus together wrongly.

Exported:
    WORDBREAK_IDENT              The marker that says the marks are present
    BREAK_ATTR, CERT_ATTR        Attribute names
    BREAK_WORD, BREAK_COMPOUND, BREAK_UNKNOWN
    has_word_break_marks(tree)   Whether to read the marks in this file
    line_texts(p)                A region's transcribed lines
    marked_lines(p)              …paired with their break marks
    join_marked_lines(lines)     The consumer-side join, from the marks alone
    region_text(p, use_marks)    A region's text, however this file says it
    paragraphs_by_zone(root)     zone xml:id -> its <p>, in one pass

Author: Christian Lendl
Created: 2026-08-15
Last Modified: 2026-08-15
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MARKER_PREFIX
from hyphen_utils import HYPHEN_CHARS, join_lines_with_hyphen_cleanup
from xml_io import NS, XML_ID, read_marker_version, tei, wsb

# ============================================================================
# The word-break vocabulary
# ============================================================================

# The marker `mark_xml_word_breaks.py` signs its work with. A file carrying it
# has been through the pass, so an unmarked line in it means "nothing continues
# here" rather than "nobody looked".
WORDBREAK_IDENT = f'{MARKER_PREFIX}-wordbreaks'

# `wsb:break`, not a bare `break`. TEI *has* a `@break` — `att.breaking`, on
# the milestone-like elements (`<lb>`, `<pb>`, `<cb>`, `<gb>`, `<milestone>`) —
# and `<l>` is not among them, so a bare `break='word'` here is rejected by
# `tei_all`, verified against the schema rather than assumed. Its `no`/`yes`/
# `maybe` vocabulary also has no value meaning "continues, but keep the
# hyphen". Both facts point the same way: this is the project's own attribute
# and belongs in the project's own namespace, where it cannot be mistaken for
# TEI's or silently ignored by something expecting TEI's.
BREAK_ATTR = wsb('break')

# `@cert` in contrast IS TEI's own (att.global.responsibility) and valid on
# `<l>`, so it stays unprefixed: the certainty of an annotation is exactly what
# TEI means by it.
CERT_ATTR = 'cert'

# One word broken across the break; the mark is not part of the word.
BREAK_WORD = 'word'
# A compound whose hyphen belongs to the word; the hyphen is kept.
BREAK_COMPOUND = 'compound'
# A mark whose kind cannot be settled here — in practice a paragraph's last
# line, whose continuation is in another paragraph entirely.
BREAK_UNKNOWN = 'unknown'

BREAK_VALUES = frozenset({BREAK_WORD, BREAK_COMPOUND, BREAK_UNKNOWN})


def has_word_break_marks(tree: etree._ElementTree) -> bool:
    """Whether this file has been through the word-break marking pass."""
    return read_marker_version(tree, WORDBREAK_IDENT) is not None


# ============================================================================
# Reading a region's lines
# ============================================================================

def line_texts(paragraph: etree._Element) -> List[str]:
    """
    The transcribed lines of one `<p>`, in document order.

    Each line is stripped and empty ones are dropped, which is what
    the import step (not shipped) has always done — an empty `<l>` is a Line
    zone with nothing written in it and contributes no text and no break.
    """
    texts = []
    for line in paragraph.xpath('.//tei:l', namespaces=NS):
        text = ''.join(line.itertext()).strip()
        if text:
            texts.append(text)
    return texts


def marked_lines(
    paragraph: etree._Element,
) -> List[Tuple[str, Optional[str]]]:
    """The same lines, each paired with its `@break` value (or None)."""
    lines = []
    for line in paragraph.xpath('.//tei:l', namespaces=NS):
        text = ''.join(line.itertext()).strip()
        if text:
            lines.append((text, line.get(BREAK_ATTR)))
    return lines


# ============================================================================
# Joining
# ============================================================================

def join_marked_lines(lines: Sequence[Tuple[str, Optional[str]]]) -> str:
    """
    Join a region the way a consumer of the marks is meant to.

    No guessing from the next word, no table of abbreviations, no notation to
    know about: the file says what each break is. This is the whole point of
    the marking, and it is deliberately identical to the reference
    implementation `test_and_debug/test_word_break_marks.py` checks the marks
    against — that test keeps its own copy on purpose, so that the property
    "the marks mean what the join does" is verified against something other
    than the code under test.
    """
    parts = []
    for index, (text, mark) in enumerate(lines):
        last = index == len(lines) - 1
        if mark in (BREAK_WORD, BREAK_COMPOUND) and not last:
            body = text[:-1]
            parts.append(body if mark == BREAK_WORD else body + '-')
        else:
            # The token ends here. A break character still sitting at the end
            # is residue of a mark the evidence rejected and
            # `correct_xml_hyphens.py` has not removed yet — no part of the
            # word either way. The last line keeps its mark: whatever
            # continues it is in another region.
            body = text
            if not last and text.endswith(tuple(HYPHEN_CHARS)):
                body = text[:-1]
            parts.append(body + ('' if last else ' '))
    return ''.join(parts)


def region_text(paragraph: etree._Element, use_marks: bool) -> str:
    """
    The text of one region, joined however this file says it should be.

    `use_marks` comes from `has_word_break_marks(tree)` — once per file, not
    once per line. See the module docstring for why that distinction matters.
    """
    if use_marks:
        return join_marked_lines(marked_lines(paragraph))
    return join_lines_with_hyphen_cleanup(line_texts(paragraph))


# ============================================================================
# Finding a region's text
# ============================================================================

def paragraphs_by_zone(root: etree._Element) -> Dict[str, etree._Element]:
    """
    Map every facsimile zone's `xml:id` to the `<p>` transcribing it.

    The link is the `<p>`'s `@facs`, which points at the zone with a leading
    `#`. Built in one pass because the alternative — an XPath per zone, as
    `import_xml_files.py` does — is quadratic, and these files hold up to a
    few thousand regions each.

    Regions with no transcription (`image`, `separator-star`,
    `separator-short`) have no `<p>` and are simply absent from the mapping;
    that is normal and not an error.
    """
    mapping = {}
    for paragraph in root.iter(tei('p')):
        facs = paragraph.get('facs')
        if facs:
            mapping[facs.lstrip('#')] = paragraph
    return mapping


def zone_id(zone: etree._Element) -> str:
    """The `xml:id` of a zone, or '' if it has none."""
    return zone.get(XML_ID, '')
