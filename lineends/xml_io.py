#!/usr/bin/env python3
"""
Shared TEI XML Reading and Writing
==================================

The scripts in `lineends/` all rewrite the Transkribus TEI files in
`data/xml/` in place, and they all need to hand the result back in exactly the
style Transkribus emits. lxml serializes attributes with double quotes and
writes an XML declaration with double quotes; the source files use single
quotes throughout. Converting back keeps `git diff` limited to the lines that
really changed, which is the whole point — a formatting-only rewrite of 735
files would bury every real correction.

This module holds that shared serialization step, plus the namespace constants
and the parser configuration the scripts agree on.

It also holds the `<teiHeader>` marker every annotating script writes: a
`<application>` entry naming the script and the version of the annotation it
applied, which is what makes a re-run cheap (a file already at the current
version is skipped) and what tells a consumer that the *absence* of an
annotation is a statement rather than a gap. The mechanics of finding or
creating `<encodingDesc>/<appInfo>`, keeping `<revisionDesc>` last in the
header's content model and indenting a newly built subtree to match the
surrounding file are identical for every such script, so they live here rather
than being copied into each one.

It also holds the order those scripts run in (`STEPS`), which is what lets a
script that changes a file clear the markers of everything built on top of it.
`STEPS` here lists the three steps these tools ship; in the project they were
extracted from it continues with the reading order, the classification and the
identifiers, and the mechanism is the same — add your own steps to the tuple.

Exported:
    TEI_NS, NS, XML_ID           Namespace constants
    parse_tei(path)              Parse preserving whitespace
    write_with_single_quotes()   Serialize in the source files' style
    read_marker_version()        The version a script last recorded
    is_up_to_date()              Whether a file already carries that version
    set_marker()                 Record that a script annotated the file
    indent_subtree()             Tab-indent a newly built subtree
    STEPS, steps_after()         The pipeline order, and what follows a step
    remove_marker()              Take one script's marker back out
    clear_markers_after()        Invalidate every step downstream of one
    marker_version_fast()        Read a marker without parsing the file
    is_up_to_date_fast()         `is_up_to_date` without parsing the file

Author: Christian Lendl
Created: 2026-08-05
Last Modified: 2026-08-15
"""

import os
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MARKER_PREFIX, PROJECT_NS  # noqa: E402

# ============================================================================
# Namespaces
# ============================================================================

TEI_NS = 'http://www.tei-c.org/ns/1.0'

# This project's own namespace, for the handful of things TEI has no attribute
# for. Everything the Guidelines *do* cover is written in plain TEI: the
# reading order is `@n`, the classification `@ana`, the fragment links
# `@part`/`@next`/`@prev`, the certainty of a mark `@cert`. What is left over is
# genuinely ours — a word-break kind, and the project's public identifier for a
# factoid — and putting it here rather than inventing a bare attribute name is
# what keeps a file valid against a schema instead of merely well-formed.
#
# Set it to a domain you control in `config.toml`; nothing fetches the URL, it
# only has to be stable and yours. The `wsb:` prefix is kept as the local name
# throughout so that files written by the Wiener Salonblatt project, where these
# tools come from, still round-trip.
WSB_NS = PROJECT_NS

NS = {'tei': TEI_NS, 'wsb': WSB_NS}

# xml:id, in the form lxml expects for attribute access
XML_ID = '{http://www.w3.org/XML/1998/namespace}id'


def tei(tag: str) -> str:
    """Return `tag` in the TEI namespace, in lxml's `{uri}local` notation."""
    return f'{{{TEI_NS}}}{tag}'


def wsb(name: str) -> str:
    """Return `name` in the project namespace, in lxml's `{uri}local` notation."""
    return f'{{{WSB_NS}}}{name}'


def ensure_wsb_namespace(tree: etree._ElementTree) -> bool:
    """
    Declare the `wsb:` prefix on the root element. True if it was added.

    Without this, lxml invents a prefix (`ns0:`) at the point of use and
    declares it on whichever element happens to carry the first attribute — so
    the same annotation would be spelled differently in different files, and
    the declaration would sit on an `<l>` halfway down a 3,000-line document.

    lxml will not let an existing element's `nsmap` be extended, so the root is
    rebuilt with the wider map and its children moved across. Only the root
    element's own line changes; everything below it is the same objects.
    """
    root = tree.getroot()
    if root.nsmap.get('wsb') == WSB_NS:
        return False

    nsmap = dict(root.nsmap)
    nsmap['wsb'] = WSB_NS
    replacement = etree.Element(root.tag, nsmap=nsmap)
    replacement.text = root.text
    replacement.tail = root.tail
    for key, value in root.attrib.items():
        replacement.set(key, value)
    replacement.extend(list(root))
    tree._setroot(replacement)
    return True


# ============================================================================
# Reading
# ============================================================================

def parse_tei(xml_path: Path) -> etree._ElementTree:
    """
    Parse a TEI file preserving whitespace.

    `remove_blank_text=False` matters: the indentation in these files lives in
    the text/tail nodes, and dropping it would reformat every line on write.
    """
    parser = etree.XMLParser(remove_blank_text=False)
    return etree.parse(str(xml_path), parser)


# ============================================================================
# Writing
# ============================================================================

# One attribute, as lxml writes it: double-quoted, the value free of literal
# `"` because lxml escapes those as &quot;.
_ATTR_DQ_RE = re.compile(r'([\w:.-]+)\s*=\s*"([^"]*)"')


def _iter_chunks(text: str):
    """
    Split serialized XML into (is_markup, chunk), markup being `<…>`.

    Quoted attribute values are skipped over while looking for the closing
    `>`, so a `>` inside one cannot end the tag early. Comments and CDATA are
    returned whole and marked as markup: their content is not text to be
    escaped, and neither appears in this corpus anyway.
    """
    position, length = 0, len(text)
    while position < length:
        start = text.find('<', position)
        if start == -1:
            yield False, text[position:]
            return
        if start > position:
            yield False, text[position:start]
        if text.startswith('<!--', start):
            end = text.find('-->', start)
            end = length if end == -1 else end + 3
        elif text.startswith('<![CDATA[', start):
            end = text.find(']]>', start)
            end = length if end == -1 else end + 3
        else:
            cursor, quote = start + 1, ''
            while cursor < length:
                character = text[cursor]
                if quote:
                    if character == quote:
                        quote = ''
                elif character in '"\'':
                    quote = character
                elif character == '>':
                    break
                cursor += 1
            end = cursor + 1
        yield True, text[start:end]
        position = end


# Elements Transkribus writes as an empty *pair* rather than self-closed.
# Measured over all 787 files: `<l …></l>` occurs 1,775 times in 460 files and
# is the only such element; `graphic`, `pb` and childless `zone` are always
# self-closed, exactly as lxml would write them. An empty `<l>` is a Line zone
# with nothing transcribed in it.
_EMPTY_PAIR_ELEMENTS = frozenset({'l'})

_SELF_CLOSING_RE = re.compile(r'<([A-Za-z][\w:.-]*)(.*?)\s*/>\Z', re.DOTALL)


def _restore_empty_pairs(markup: str) -> str:
    """Write an empty element of `_EMPTY_PAIR_ELEMENTS` as `<x …></x>`."""
    match = _SELF_CLOSING_RE.match(markup)
    if match is None or match.group(1) not in _EMPTY_PAIR_ELEMENTS:
        return markup
    return f'<{match.group(1)}{match.group(2)}></{match.group(1)}>'


def _single_quote_attributes(markup: str) -> str:
    """Rewrite `name="value"` as `name='value'` within one tag."""
    def replace(match: 're.Match[str]') -> str:
        name, value = match.group(1), match.group(2)
        # A value containing an apostrophe cannot be single-quoted without
        # escaping it. None exist in this corpus; keeping the double quotes is
        # the one option that stays well-formed if one ever does.
        if "'" in value:
            return match.group(0)
        return f"{name}='{value}'"
    return _ATTR_DQ_RE.sub(replace, markup)


def write_with_single_quotes(tree: etree._ElementTree, xml_path: Path) -> None:
    """
    Serialize the tree in the style Transkribus emits.

    Two differences have to be undone, both of them pure notation — the parsed
    document is identical either way — and both of which would otherwise put
    every file these scripts touch into the diff wholesale:

    * **Attribute quoting.** lxml writes `facs="…"`, the source files use
      `facs='…'`.
    * **Predefined entities in text.** Transkribus writes `&quot;` and
      `&apos;`; lxml resolves them on parse and re-emits the bare characters,
      since neither needs escaping in text content. The corpus is full of
      them — the magazine sets its quotation marks as `„…“` but straight
      quotes survive in around 3 lines per file — so a corrector that changed
      one line was rewriting ten others alongside it.
    * **Empty `<l>` elements.** Written as `<l …></l>`, which lxml self-closes.

    Together these were changing around 14 lines in half the files in the
    corpus on a rewrite that changed nothing at all.

    Doing this by walking markup and text separately, rather than with one
    regex over the whole document, is what makes the second fix safe: the text
    now legitimately contains `"` and `'`, and a global `="…"` substitution
    would eventually find a pair of them inside a transcribed line and quietly
    rewrite the transcription.
    """
    raw = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')
    parts = []
    for is_markup, chunk in _iter_chunks(raw.decode('utf-8')):
        if is_markup:
            parts.append(_restore_empty_pairs(_single_quote_attributes(chunk)))
        else:
            parts.append(chunk.replace('"', '&quot;').replace("'", '&apos;'))
    text = ''.join(parts)
    # Transkribus ends every file with a newline (all 787 in the corpus do) and
    # lxml ends with `</TEI>`. Without this the closing tag lands in the diff of
    # every file any of these scripts touches, for no change at all.
    if not text.endswith('\n'):
        text += '\n'

    # Write to a sibling and rename, rather than truncating the real file and
    # filling it in. `os.replace` is atomic, so a reader sees either the whole
    # old file or the whole new one — never the empty file that a truncate
    # leaves behind for the moment it takes to write 8 MB. That window is not
    # theoretical: two of these scripts running at once, or one of them killed
    # mid-write, will otherwise hand the next reader `Document is empty, line
    # 1, column 1`. The temporary carries the pid so two writers cannot collide
    # on it, and it sits in the same directory to keep the rename on one
    # filesystem.
    path = Path(xml_path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ============================================================================
# The header marker
# ============================================================================

def read_marker_version(tree: etree._ElementTree,
                        ident: str) -> Optional[str]:
    """Return the version `ident` last recorded in the file, if any."""
    app = tree.find(f".//tei:appInfo/tei:application[@ident='{ident}']", NS)
    return app.get('version') if app is not None else None


def _version_tuple(version: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split('.'))
    except ValueError:
        return (0,)


def is_up_to_date(tree: etree._ElementTree, ident: str,
                  version: str) -> bool:
    """True if the file already carries this version of `ident`'s annotation."""
    recorded = read_marker_version(tree, ident)
    if recorded is None:
        return False
    return _version_tuple(recorded) >= _version_tuple(version)


def indent_subtree(element: etree._Element, depth: int) -> None:
    """
    Tab-indent a newly built subtree so it matches the surrounding file.

    Only ever called on elements a script creates. Existing subtrees are left
    untouched — reformatting them would put the whole header into the diff,
    which is precisely what the single-quote serialization avoids.
    """
    children = list(element)
    if not children:
        return
    element.text = '\n' + '\t' * (depth + 1)
    for child in children:
        child.tail = '\n' + '\t' * (depth + 1)
        indent_subtree(child, depth + 1)
    children[-1].tail = '\n' + '\t' * depth


def set_marker(tree: etree._ElementTree, ident: str, version: str, when: str,
               label: str, change: str) -> None:
    """
    Record that a script annotated the file.

    `<appInfo>` inside `<encodingDesc>` is TEI's own mechanism for saying which
    software acted on a document, and carrying the version there is what makes
    re-runs cheap. A human-readable `<change>` goes in `<revisionDesc>`
    alongside it.

    `label` describes what the annotation is (it lands in `<application>`'s
    `<label>` and is the one place a consumer of the corpus can read what the
    attributes mean); `change` is the `<revisionDesc>` line.
    """
    header = tree.find('.//tei:teiHeader', NS)
    if header is None:
        raise AssertionError('no <teiHeader>')

    # --- encodingDesc/appInfo/application, updated in place if already there
    app = header.find(f".//tei:appInfo/tei:application[@ident='{ident}']", NS)
    if app is None:
        encoding_desc = header.find('tei:encodingDesc', NS)
        if encoding_desc is None:
            encoding_desc = etree.Element(tei('encodingDesc'))
            # encodingDesc follows fileDesc in the TEI header's content model
            file_desc = header.find('tei:fileDesc', NS)
            position = header.index(file_desc) + 1 if file_desc is not None else 0
            header.insert(position, encoding_desc)
        app_info = encoding_desc.find('tei:appInfo', NS)
        if app_info is None:
            app_info = etree.SubElement(encoding_desc, tei('appInfo'))
        app = etree.SubElement(app_info, tei('application'))
        etree.SubElement(app, tei('label'))
        # Indent the whole of encodingDesc, not only a subtree just created.
        # A second script adding its `<application>` to an `<appInfo>` the first
        # one built would otherwise append it unindented, on a single line — and
        # the corpus is meant to be read. Everything under here is written by
        # these scripts (no Transkribus export carries an `<encodingDesc>`), so
        # re-indenting it reformats nothing that is not ours, and this branch
        # only runs when the element is being added anyway.
        indent_subtree(encoding_desc, depth=2)
    # The label is rewritten on every run: it documents the current version's
    # vocabulary, so a file carrying an older one must not keep its old wording.
    label_element = app.find('tei:label', NS)
    if label_element is None:
        label_element = etree.SubElement(app, tei('label'))
    label_element.text = label
    app.set('ident', ident)
    app.set('version', version)
    app.set('when', when)

    # --- revisionDesc/change, replacing this script's previous entry
    revision_desc = header.find('tei:revisionDesc', NS)
    if revision_desc is None:
        revision_desc = etree.SubElement(header, tei('revisionDesc'))
    for previous in revision_desc.findall('tei:change', NS):
        if previous.get('n') == ident:
            revision_desc.remove(previous)
    change_element = etree.SubElement(revision_desc, tei('change'))
    change_element.set('when', when)
    change_element.set('n', ident)
    change_element.text = change

    # revisionDesc is last in the TEI header's content model
    header.remove(revision_desc)
    header.append(revision_desc)
    indent_subtree(revision_desc, depth=2)

    # Only the header's own children are re-tailed: one per line, with the
    # last one closing the header. Their subtrees are never touched.
    children = list(header)
    header.text = '\n\t\t'
    for child in children[:-1]:
        child.tail = '\n\t\t'
    children[-1].tail = '\n\t'


# ============================================================================
# The annotation chain
# ============================================================================

class Step(NamedTuple):
    """One script that rewrites `data/xml/`."""

    name: str                # script stem, e.g. 'set_xml_readingorder'
    ident: Optional[str]     # its `<application>` marker, None if it writes none
    automated: bool          # run by `main.py --annotate`, or only by hand?
    note: str = ''           # why, when it is not automated


# Every script that rewrites `data/xml/`, in the order the pipeline runs them.
# This is the one definition of that order: `clear_markers_after` reads it, and
# so does `main.py --annotate`, so the two cannot drift apart.
#
# A step that writes no marker still holds a position here — it needs one to
# know what comes *after* it. The two correctors write none deliberately: they
# are driven by decision stores that keep growing as the LLM passes settle more
# cases, so "this file has been corrected" is never true for good the way "this
# file has been ordered" is.
#
# `automated` is the difference between a step that derives everything from the
# file plus a committed CSV, and one that applies verdicts a model had to
# produce first. Running the latter unattended would rewrite the corpus from
# whatever half of the decisions happened to exist at the time.
STEPS: Tuple[Step, ...] = (
    # Order matters, and this is the one definition of it.
    #
    # The OCR correction runs first and never touches the character at the line
    # end; the hyphen correction decides that character, from evidence. Running
    # them the other way round lets a substring rule overwrite a verdict.
    Step('correct_xml_ocr', None, False,
         'applies data/csv/ocr_corrections.csv - run by hand, after --dry-run'),
    Step('correct_xml_hyphens', None, False,
         'applies data/csv/hyphen_decisions.csv - run by hand, after --dry-run'),
    # Last, and dependent on both: the break marker records how a line end was
    # resolved, so the text it describes has to be final first.
    Step('mark_xml_word_breaks', f'{MARKER_PREFIX}-wordbreaks', True),
)


def step_names() -> List[str]:
    """The pipeline steps, in order."""
    return [step.name for step in STEPS]


def steps_after(step: str) -> List[str]:
    """
    Return the marker idents of every step that runs after `step`.

    Raises KeyError for a step not in STEPS — a typo here would silently clear
    nothing, which is the failure mode this whole mechanism exists to prevent.
    """
    names = step_names()
    if step not in names:
        raise KeyError(
            f'unknown pipeline step {step!r}; known steps: {", ".join(names)}'
        )
    position = names.index(step)
    return [s.ident for s in STEPS[position + 1:] if s.ident is not None]


def _drop_element(element: etree._Element) -> None:
    """
    Remove an element, leaving the surrounding indentation intact.

    lxml carries an element's trailing whitespace in its own `tail`, so
    removing the *last* child of a parent takes the newline-and-indent that
    closed the parent with it, and the closing tag ends up on the content line.
    Handing that tail to the new last child keeps the file readable.
    """
    parent = element.getparent()
    if parent is None:
        return
    siblings = list(parent)
    if siblings and siblings[-1] is element and len(siblings) > 1:
        siblings[-2].tail = element.tail
    parent.remove(element)


def remove_marker(tree: etree._ElementTree, ident: str) -> bool:
    """
    Remove `ident`'s marker from the header. True if anything was removed.

    Takes the `<application>` and the matching `<revisionDesc><change>` with
    it, and then any container left empty: TEI requires `<appInfo>`,
    `<encodingDesc>` and `<revisionDesc>` to have content, so leaving an empty
    one behind would turn a cleared marker into an invalid file.
    """
    header = tree.find('.//tei:teiHeader', NS)
    if header is None:
        return False

    removed = False

    app = header.find(f".//tei:appInfo/tei:application[@ident='{ident}']", NS)
    if app is not None:
        app_info = app.getparent()
        _drop_element(app)
        removed = True
        if len(app_info) == 0:
            encoding_desc = app_info.getparent()
            _drop_element(app_info)
            if len(encoding_desc) == 0:
                _drop_element(encoding_desc)

    revision_desc = header.find('tei:revisionDesc', NS)
    if revision_desc is not None:
        for change in revision_desc.findall('tei:change', NS):
            if change.get('n') == ident:
                _drop_element(change)
                removed = True
        if len(revision_desc) == 0:
            _drop_element(revision_desc)

    return removed


def clear_markers_after(tree: etree._ElementTree, step: str) -> List[str]:
    """
    Clear the markers of every step downstream of `step`. Returns what it removed.

    A script that changes a file has to say so, or the next run of everything
    built on top of that file will skip it as up to date. A Transkribus
    re-export needs no help here — it arrives with no markers at all — but a
    correction applied in place does.

    This is deliberately a blanket rule rather than a dependency graph: it
    clears *everything* after the step, not only what the change can actually
    have invalidated. Correcting a transcription does not move a region, so the
    reading order it clears was in fact still good. Re-running that costs
    seconds over the whole corpus; working out which of eight steps a given
    edit really touches, and being wrong once, costs a corpus that quietly
    disagrees with itself.
    """
    cleared = [ident for ident in steps_after(step) if remove_marker(tree, ident)]
    return cleared


# ============================================================================
# Reading a marker without parsing the file
# ============================================================================

# `<application ident='…' version='…' when='…'>`, in either quoting style and
# in any attribute order.
_APPLICATION_RE = re.compile(r'<application\b([^>]*)>')
_IDENT_RE = re.compile(r'\bident\s*=\s*[\'"]([^\'"]*)[\'"]')
_VERSION_RE = re.compile(r'\bversion\s*=\s*[\'"]([^\'"]*)[\'"]')

# How much of a file to read before giving up on finding `</teiHeader>`. The
# headers in this corpus run to about 1 KB; 256 KB is far past any plausible
# one and still a rounding error against an 8 MB file.
_HEADER_MAX_BYTES = 262_144


def marker_version_fast(xml_path: Path, ident: str,
                        chunk_size: int = 8192) -> Optional[str]:
    """
    Return the version `ident` recorded in a file, without parsing the file.

    The corpus is heading for 2,731 files of ~8 MB, and every annotating script
    asks this one question of every one of them on every run. Parsing 21 GB to
    read 2,731 headers costs minutes per script per run; the header is the
    first kilobyte, so read that instead.

    Raises ValueError if `</teiHeader>` is not found within the first
    `_HEADER_MAX_BYTES` — that is a malformed file rather than an unmarked one,
    and silently reporting "no marker" would make the pipeline reprocess it
    forever.
    """
    with open(xml_path, 'rb') as handle:
        head = handle.read(chunk_size)
        while b'</teiHeader>' not in head and len(head) < _HEADER_MAX_BYTES:
            more = handle.read(chunk_size)
            if not more:
                break
            head += more

    text = head.decode('utf-8', errors='replace')
    end = text.find('</teiHeader>')
    if end == -1:
        raise ValueError(
            f'{xml_path}: no </teiHeader> in the first {len(head)} bytes'
        )

    for match in _APPLICATION_RE.finditer(text, 0, end):
        attributes = match.group(1)
        found = _IDENT_RE.search(attributes)
        if found is not None and found.group(1) == ident:
            version = _VERSION_RE.search(attributes)
            return version.group(1) if version is not None else None
    return None


def is_up_to_date_fast(xml_path: Path, ident: str, version: str) -> bool:
    """`is_up_to_date` without parsing the file. See `marker_version_fast`."""
    recorded = marker_version_fast(xml_path, ident)
    if recorded is None:
        return False
    return _version_tuple(recorded) >= _version_tuple(version)
