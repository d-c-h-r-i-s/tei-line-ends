#!/usr/bin/env python3
"""
XML Line-End Hyphen Validation
==============================

Detects wrong or missing line-end hyphens (`¬`, `-`, `=`) in the Transkribus
XML. Both error directions are covered:

    False positive — a `¬` where the two parts are really two separate words
                     ("die Leiden¬ / der Bedauernswerten" is *not*
                     "Leidender"). The current import logic in
                     `hyphen_utils.py` trusts every `¬` and would
                     silently glue these together.

    False negative — a line break with no `¬` although the word continues
                     ("Maje | stät", "Prinzes | sin", "deut | schen").

Method — the corpus is its own dictionary
-----------------------------------------
No external word list and no LLM is needed for the bulk of the decisions.
A frequency lexicon is built from the *interior* tokens of every line, i.e.
every token that is neither the first nor the last one on its line. Those
tokens can by construction never be affected by the line-break hyphen problem,
so the lexicon is unbiased with respect to the very error we are looking for.
From the same tokens a bigram table is collected.

Every line break then poses the same question — how should this pair be
written? — and the `¬` the OCR produced is only a (fallible) prior. Five
readings compete, each scored from its own table of line-interior evidence:

    join     unigram frequency of  A+B        "Prinzessin"
    hyphen   frequency of  A-B  written with a real dash inside a line
             "Windisch-Graetz"
    period   frequency of  "A." followed by B  — the `¬` is a misread full
             stop, which is how the abbreviated titles this corpus is full of
             get restored: "Geh. Rat", "Se. Majestät", "Präs. des"
    comma    frequency of  "A," followed by B  — the `¬` is a misread comma,
             before a subordinate clause or inside an enumeration:
             "fand statt, wo …", "Lingerie, Trousseaux, Layettes"
    split    plain bigram frequency of  A B    "Leiden der"

Decision rules, in order:

    1. right part capitalised -> dashed compound vs. separated words
       (this is where "Windisch¬ Graetz" and "in¬ Wien" part ways)
    2. left part is not a word on its own -> JOIN
       ("Prin" does not exist, "Prinzessin" does)
    3. compound clearly beats separation  -> JOIN
    4. separation clearly beats the compound -> the `¬` is spurious
    5. how the corpus writes this very pair everywhere else
    6. neither attested, both parts are common words -> separated
    7. otherwise -> REVIEW

Whenever a rule lands on "separated", a second question follows: does anything
stand between the two words? Comma and full-stop occurrences count towards
separation either way — they part the words just as a space does — so the
character is chosen only afterwards, by `separator_form`, and adding the two
punctuation readings therefore cannot disturb the join/split balance. A full
stop is not confined to capitalised right-hand words: the abbreviations this
corpus is built on are followed by lowercase ones constantly ("Präs. des",
"Kl. und"), so all three separators are weighed the same way.

Overturning the OCR additionally requires the opposing reading to be
essentially unattested. Where both readings are real words the pair is
genuinely contested — "nachdem"/"nach dem", "wieder"/"wie der",
"Leidender"/"Leiden der" — and only the surrounding sentence can settle it,
so it goes to the review queue instead (see --export-review).

Measured on the full corpus (735 issues, 889k judged line breaks):

    OK               97.9%
    FALSE_POSITIVE    8,067 occurrences   (spurious `¬`)
    FALSE_NEGATIVE    5,415 occurrences   (dropped `¬`)
    PERIOD              302 occurrences   (`¬` that is really a full stop)
    COMMA                67 occurrences   (`¬` that is really a comma)
    REVIEW            5,126 occurrences / 1,464 distinct pairs

Because the queue is deduplicated by word pair, the LLM is asked ~1,500
questions rather than being run over 735 files. Answers land in the decision
store (`data/csv/hyphen_decisions.csv`) via `resolve_hyphens_llm.py` and are
consulted on every later run, so growing the corpus to ~2,700 issues only ever
costs questions about pairs that have never been seen before.

Antiqua and Fraktur
-------------------
The issues set in Antiqua mark a continued word with `¬` and write a real
compound hyphen as `-`. The older ones set in Fraktur use the Doppelbindestrich
`=` for both, and write in-line compounds the same way ("Franz Josefs=Kai").
Nothing above depends on which of them a file uses: a break is a break whatever
character marks it, and the in-line compound table accepts either spelling
under one key, so the `hyphen` reading keeps its evidence in a Fraktur issue
too. What is decided here is a *form*, never a character.

Which notation a file is in is therefore only reported (`detect_notation`, by
majority over its line ends) and matters one step later, where
`correct_xml_hyphens.py` has to write a character back and a Fraktur issue must
not acquire a `¬` it never used.

Paragraph-final hyphens are out of scope
----------------------------------------
A hyphen on the *last* line of a paragraph is never counted, flagged or
corrected here. Only breaks between two lines of the same paragraph are
examined, so those lines are skipped by construction (3,710 of them across the
corpus, none touched). Such a paragraph is normally continued in the next
column or on the next page; it is joined later by
the paragraph-merge step (downstream, not shipped), and whatever hyphens remain after that join
are reported by `utils/check_remaining_hyphens.py` for manual correction.

Regions outside `CORRECTED_SUBTYPES` are excluded by default
(`--skip-subtypes`), from the frequency lexicon as well as from the judged
breaks. Two reasons pointing the same way: `ad`/`ad-frame` hold real text whose
OCR is the noisiest in the corpus by a factor of 16
(docs/ocr-word-correction.md), while `image`, `mode` and the `separator*`
classes hold no text at all — what the line model reports inside them is stray
marks read as characters ("K / 2 / 8 / K / K"). Either way they would pollute
the evidence tables and dominate the report — which is what running without the
exclusion looks like. `--skip-subtypes ''` reopens them.

Note on line geometry: whether a line reaches the right margin of its
TextRegion was tested as an additional signal and does *not* separate the
classes (median gap to the margin 23px for correct vs 27px for spurious
hyphens). It is deliberately not used.

Output:
    Everything a run produces lands in output/ under the same date prefix: the
    verdict CSV, and the two optional JSONL exports.

    A CSV (date-prefixed, in output/) with one row per distinct word pair,
    its verdict, the supporting counts and an example location, mirroring the
    other validation scripts. The `find`/`replace` columns are written against
    the joined paragraph text — i.e. after `hyphen_utils.py` has
    already glued the lines together — and use the surface forms as they
    really occur, punctuation included ("österr.¬ungar.").

Usage:
    python validate_xml_hyphens.py
    python validate_xml_hyphens.py --max-files 50
    python validate_xml_hyphens.py --report false-positives
    python validate_xml_hyphens.py --export-review review.jsonl

Flags:
    --xml-folder P      Path to XML files folder (default: PATHS['xml']).
    --max-files N       Maximum number of files to process.
    --skip-subtypes S   Comma-separated region subtypes to leave out of the
                        lexicon and the judged breaks (default: the regions
                        outside CORRECTED_SUBTYPES;
                        pass '' for none).
    --report R          Which verdicts to write: all (default),
                        false-positives (spurious `¬` only),
                        false-negatives (missing `¬` only),
                        review (contested cases only).
    --min-count N       Only report pairs seen at least N times (default: 1).
    --ratio R           Evidence ratio needed for a confident verdict
                        (default: 3.0).
    --export-review NAME
                        Additionally write contested cases with their sentence
                        context as JSONL, ready to be sent to a local LLM.
    --export-benchmark NAME
                        Write a labelled test set as JSONL: pairs the
                        statistics settled confidently, in the same shape as
                        the review queue but with the answer attached. Lets a
                        model be scored before it is trusted with the cases
                        nobody knows (see resolve_hyphens_llm.py --benchmark).

                        Both JSONL exports follow the CSV: a bare filename is
                        written to output/ with the run's date in front, so the
                        artefacts of one run stay together. Pass a path with a
                        directory (or an absolute one) to override that.
    --benchmark-size N  How many pairs to sample, stratified over the forms
                        (default: 200).
    --decisions P       Accumulated LLM/manual verdicts, consulted before the
                        statistics (default: PATHS['csv']/hyphen_decisions.csv).
    --output NAME       Output CSV filename (default: hyphen_validation.csv).
    --quiet             Suppress progress messages.

Author: Christian Lendl
Created: 2026-07-22
Last Modified: 2026-07-29
"""

import argparse
import csv
import datetime
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS
from hyphen_decisions import (
    DEFAULT_NOTATION,
    NOTATIONS,
    Decision,
    load as load_decisions,
)

try:
    from utils.get_transkribus_link import get_transkribus_link
except ImportError:  # pragma: no cover - link is a convenience only
    get_transkribus_link = None


# Hyphen characters that mark a word continuation at the end of a line. `¬` is
# the Antiqua convention, `=` the Fraktur Doppelbindestrich of the older
# issues, and `-` occurs in both.
HYPHEN_CHARS = ('¬', '-', '=')

# Punctuation stripped before a token is looked up in the lexicon. `=` is in
# here because Fraktur OCR reads an opening quotation mark as one ("=Der
# Dichter"); a `=` *inside* a token is a compound hyphen and is dealt with by
# `strip_token`, since only leading and trailing characters are removed here.
STRIP_CHARS = '.,;:!?»«„“”"\'()[]*='

# Inside a token, the Fraktur Doppelbindestrich is the same compound hyphen a
# `-` is, so lookup keys are written with `-` throughout. Without this the same
# name would key the tables twice ("Pachta-Rayhofen" and "Pachta=Rayhofen"),
# and a pair the LLM already settled in an Antiqua issue would be asked again
# for a Fraktur one.
INNER_DASH_RE = re.compile(r"(?<=[A-Za-zÄÖÜäöüßÀ-ÿ])=(?=[A-Za-zÄÖÜäöüßÀ-ÿ])")

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüßÀ-ÿ]+")

# A compound written with a real hyphen inside one line - "Windisch-Graetz" in
# Antiqua, "Franz Josefs=Kai" in Fraktur. Tokens normally reach this already
# normalised by `strip_token`; accepting `=` as well keeps the pattern true on
# its own, and the halves come from the match rather than from splitting on a
# fixed character so either spelling yields the same key.
DASHED_RE = re.compile(r"([A-Za-zÄÖÜäöüßÀ-ÿ]+)[-=]([A-Za-zÄÖÜäöüßÀ-ÿ]+)")

# Shortest token still worth judging - single letters are initials or ad noise
MIN_TOKEN_LEN = 2

# `[^>]*` after the facs attribute is not cosmetic: `set_xml_readingorder.py`
# writes a reading-order index onto every paragraph (`<p facs='#…' n='1'>`), and
# a pattern demanding `>` right after the closing quote matched none of them —
# the whole analysis silently read an empty corpus. Tolerating further
# attributes is what stops the next one from breaking this again.
LINE_RE = re.compile(r"<l\s+facs='#([^']*)'[^>]*>(.*?)</l>", re.S)
PARA_RE = re.compile(r"<p\s+facs='#([^']*)'[^>]*>(.*?)</p>", re.S)
PAGE_RE = re.compile(r"<pb\s+facs='#facs_(\d+)'")
TAG_RE = re.compile(r"<[^>]+>")

# The layout class of a region, as these scripts spell it: the
# `subtype` of the `<zone rendition='TextRegion'>` a paragraph's `facs` points
# at. The whole attribute list is captured and the two attributes read out of it
# separately, so nothing here depends on the order Transkribus writes them in.
REGION_RE = re.compile(r"<zone\s+([^>]*rendition='TextRegion'[^>]*)>")
ATTR_RE = re.compile(r"(\S+)='([^']*)'")

# ---------------------------------------------------------------------------
# What the correction scripts are allowed to touch
# ---------------------------------------------------------------------------
#
# One scope, shared by every script in `lineends` that reads or writes the
# transcription, so a region cannot be evidence in one pass and invisible in
# the next. It is stated as two explicit lists rather than one, because the
# question "is this subtype in scope?" must have an answer for every subtype
# the corpus contains - and when it does not, that is a fact worth reporting
# rather than a default worth applying (see `unknown_subtypes`).
#
# Corrected: every region holding running prose the magazine actually set.
# `ad-content` belongs here and `ad`/`ad-frame` do not: the content of an
# advertisement is typeset text, while the frame and the unstructured ad body
# are the noisiest OCR in the corpus by a wide margin (docs/ocr-word-correction.md:
# a truncated line-final word is 16x more likely inside one).
CORRECTED_SUBTYPES = ('paragraph', 'heading', 'heading-sub', 'image-caption',
                      'image-credit', 'ad-content', 'imprint', 'footnote')

# Left alone, for two different reasons that happen to point the same way:
#
#   ad, ad-frame              real text, unusable OCR
#   image, mode               no text at all. What the line model reports
#                             inside them is stray marks read as characters
#                             ("K / 2 / 8 / K / K"), and correcting it would
#                             mean inventing words for noise that is going to
#                             be removed.
#   volume                    real text, but only ever the issue's date, volume
#                             and number ("Nr. 1. / Sonntag, den 4. Jänner
#                             1914. / 45. Jahrgang."). A fixed, closed form
#                             checked by its own script, where a general word
#                             correction has nothing to contribute and a wrong
#                             one would corrupt a field other steps parse.
#   separator*                printer's rules, not language
SKIPPED_SUBTYPES = ('ad', 'ad-frame',
                    'image', 'mode', 'volume',
                    'separator', 'separator-short', 'separator-star')

# Kept as the historical name for the two subtypes this started with, since
# `--skip-subtypes ad,ad-frame` appears in the README, in run logs and in
# people's shell history. New code should use SKIPPED_SUBTYPES.
NOISY_SUBTYPES = ('ad', 'ad-frame')

# Stats key for the break marks on a paragraph's last line. Not a verdict —
# nothing was judged — so it is deliberately kept out of the verdict table and
# out of its total, and reported on its own line underneath.
UNJUDGED = 'UNJUDGED_PARAGRAPH_FINAL'


def unknown_subtypes(seen) -> set:
    """
    Subtypes the scope above does not mention.

    A new region class is not a hypothetical: `heading-sub` was added to this
    corpus in August 2026, and a scope expressed only as "skip these" would
    have started correcting it silently on the next run. Reporting the gap is
    the cheap half of the fix; deciding which list it belongs in is a person's
    job and cannot be guessed from the name.
    """
    known = set(CORRECTED_SUBTYPES) | set(SKIPPED_SUBTYPES)
    return {s for s in seen if s and s not in known}


@dataclass
class PairEvidence:
    """Accumulated evidence for one distinct (left, right) word pair."""
    left: str
    right: str
    hyphenated: int = 0          # times seen with a `¬`/`-` at the break
    plain: int = 0               # times seen with no hyphen at the break
    join_count: int = 0          # lexicon frequency of left+right
    split_count: int = 0         # bigram frequency of (left, right), no comma
    comma_count: int = 0         # frequency of "left," followed by "right"
    left_count: int = 0          # lexicon frequency of left alone
    right_count: int = 0         # lexicon frequency of right alone
    dash_count: int = 0          # frequency of "left-right" written with a dash
    period_count: int = 0        # frequency of "left." followed by "right"
    form: Optional[str] = None   # decided form: join / hyphen / period / split
    decided_by: str = ""         # set when the form came from the decision store
    verdict: str = ""
    reason: str = ""
    bad_hyphenated: int = 0      # hyphenated occurrences judged wrong
    bad_plain: int = 0           # plain occurrences judged wrong
    examples: Dict[str, Tuple[str, int, str]] = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)
    surface_left: Counter = field(default_factory=Counter)
    surface_right: Counter = field(default_factory=Counter)

    def surface_pair(self) -> Tuple[str, str]:
        """The most common surface forms, punctuation included."""
        return (self.surface_left.most_common(1)[0][0],
                self.surface_right.most_common(1)[0][0])

    @property
    def separation(self) -> int:
        """
        Total evidence that the two words stand apart.

        A comma or a full stop between them separates them just as a plain
        space does, so all three tables count towards the join/split question;
        which of them supplies the evidence only decides the character written
        at the break (see `separator_form`).
        """
        return self.split_count + self.comma_count + self.period_count

    @property
    def total(self) -> int:
        return self.hyphenated + self.plain

    @property
    def affected(self) -> int:
        """Occurrences that would actually change if the verdict is applied."""
        return self.bad_hyphenated + self.bad_plain


def strip_token(token: str) -> str:
    """
    Reduce a token to the form under which it is looked up.

    Surrounding punctuation is removed and a Fraktur compound hyphen is written
    as `-`, so an Antiqua and a Fraktur issue produce the same key for the same
    word. This is the one place both the lexicon, the pair table and the
    correction step pass through, which is what keeps their keys in step.

    The surface forms the CSV reports are kept separately and stay untouched -
    they have to match the text as it really is.
    """
    return INNER_DASH_RE.sub('-', token.strip(STRIP_CHARS))


def break_pair(current: str, following: str
               ) -> Optional[Tuple[str, str, str, str]]:
    """
    Derive the lookup key for the break between two consecutive lines.

    Returns `(left, right, raw_left, raw_right)` — the two normalised tokens
    the pair table is keyed by, plus the surface forms they were read from — or
    None when this break carries no usable pair (an empty line, or a token too
    short to judge).

    Every caller that has to *find* a break in the corpus goes through this:
    `collect_breaks` when the table is built, `correct_xml_hyphens.py` when a
    verdict is written back, `mark_xml_word_breaks.py` when a break is
    annotated. They must derive the key identically or a verdict is looked up
    under a key the analysis never stored, and the failure is silent — the
    correction simply does not happen.
    """
    tail = following.split()
    if not tail:
        return None

    has_hyphen = current.endswith(HYPHEN_CHARS)
    body = current[:-1] if has_hyphen else current
    if not body.split():
        return None

    raw_left = body.split()[-1]
    raw_right = tail[0]
    # The right word can be the whole of its own line and so carry the *next*
    # break mark ("bei="). That mark is gone from the joined text the
    # `find`/`replace` columns are written against, so it must not end up in
    # the surface form either. Only a one-token line qualifies: a `-` further
    # along the line ("Hof- und") is a real hyphen and survives the join.
    if len(tail) == 1 and raw_right.endswith(HYPHEN_CHARS):
        raw_right = raw_right[:-1]
    left = strip_token(raw_left)
    right = strip_token(raw_right)
    # Single letters are initials, ad fragments and stray marks; they carry no
    # usable frequency signal.
    if len(left) < MIN_TOKEN_LEN or len(right) < MIN_TOKEN_LEN:
        return None
    return left, right, raw_left, raw_right


def detect_notation(lines: Iterable[str],
                    default: str = DEFAULT_NOTATION) -> str:
    """
    Which line-break mark this text is transcribed with.

    An issue set in Antiqua ends a continued line with `¬`, an older one set in
    Fraktur with the Doppelbindestrich `=`. Neither character means anything
    else, so counting how often each closes a line tells the two typesettings
    apart reliably, file by file - a corpus may well hold both. A plain `-` is
    not a candidate: it is the ordinary compound hyphen in either style and so
    says nothing about the notation.

    Returns `default` when neither mark occurs, which is what an issue with no
    line breaks worth speaking of gets.
    """
    counts: Counter = Counter()
    for line in lines:
        counts[line.rstrip()[-1:]] += 1
    best = max(NOTATIONS, key=lambda char: counts[char])
    return best if counts[best] else default


def read_subtypes(xml_text: str) -> Dict[str, str]:
    """
    Map every TextRegion's xml:id to its layout subtype.

    Read from `<facsimile>`, which is where the geometry and the layout class
    live; the `<p>` elements in `<text>` only point at it through `facs`. A
    region carrying no `subtype` maps to '', the same way an unclassified region
    reads elsewhere in these scripts.
    """
    subtypes: Dict[str, str] = {}
    for match in REGION_RE.finditer(xml_text):
        attrs = dict(ATTR_RE.findall(match.group(1)))
        region_id = attrs.get('xml:id')
        if region_id:
            subtypes[region_id] = attrs.get('subtype', '')
    return subtypes


def read_paragraphs(xml_text: str, min_lines: int = 2,
                    skip_subtypes: FrozenSet[str] = frozenset()
                    ) -> List[Tuple[str, int, str, List[str]]]:
    """
    Extract paragraphs as lists of line strings.

    `min_lines` is what a caller needs the paragraph for. The hyphen analysis
    judges the break *between* two lines and so needs at least two of them,
    which is the default; the line-end analysis judges where a line stops and
    needs no following line at all, so it passes 1 and a one-line paragraph is
    judged like any other.

    `skip_subtypes` drops whole layout classes before anything is counted —
    `NOISY_SUBTYPES` is what a caller normally wants. Filtering here rather than
    at the call site means an excluded region cannot reach a frequency lexicon
    either, which is the point: ad vocabulary must not become evidence about
    ordinary prose.

    Returns:
        List of (region_id, page_number, region_subtype, [line_text, ...]).
    """
    body_start = xml_text.find('<text')
    if body_start == -1:
        return []
    # Subtypes come from <facsimile>, which sits *before* <text>, so they are
    # read from the head of the document rather than from the body slice.
    subtypes = read_subtypes(xml_text[:body_start])
    body = xml_text[body_start:]

    # Map each paragraph to the page it appears on by tracking <pb/> markers
    page_at: List[Tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in PAGE_RE.finditer(body)
    ]

    paragraphs = []
    for pm in PARA_RE.finditer(body):
        region_id = pm.group(1)
        subtype = subtypes.get(region_id, '')
        if subtype in skip_subtypes:
            continue
        page = 0
        for pos, num in page_at:
            if pos < pm.start():
                page = num
            else:
                break
        lines = []
        for lm in LINE_RE.finditer(pm.group(2)):
            text = TAG_RE.sub('', lm.group(2)).strip()
            if text:
                lines.append(text)
        if len(lines) >= min_lines:
            paragraphs.append((region_id, page, subtype, lines))
    return paragraphs


def read_paragraph_lines(xml_text: str, min_lines: int = 2,
                         skip_subtypes: FrozenSet[str] = frozenset()
                         ) -> List[Tuple[str, int, str,
                                         List[Tuple[str, str]]]]:
    """
    `read_paragraphs`, but each line paired with its `facs` id.

    A sibling rather than a parameter on `read_paragraphs`. That function is
    what `validate_line_end_chars.py`, `correct_xml_hyphens.py` and
    `mark_xml_word_breaks.py` all key their tables through, it is guarded by
    `test_and_debug/test_xml_readers.py` because it has silently stopped
    matching before, and a per-occurrence stage is no reason to make three
    other scripts unpack a shape they have no use for.

    The id is what an occurrence verdict is stored under: a break belongs to
    the line it *ends*, and `facs` names that line uniquely within its file.
    `LINE_RE` has always captured it; nothing here but `read_paragraphs`
    threw it away.

    Returns:
        List of (region_id, page_number, region_subtype,
                 [(line_id, line_text), ...]).
    """
    body_start = xml_text.find('<text')
    if body_start == -1:
        return []
    subtypes = read_subtypes(xml_text[:body_start])
    body = xml_text[body_start:]

    page_at: List[Tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in PAGE_RE.finditer(body)
    ]

    paragraphs = []
    for pm in PARA_RE.finditer(body):
        region_id = pm.group(1)
        subtype = subtypes.get(region_id, '')
        if subtype in skip_subtypes:
            continue
        page = 0
        for pos, num in page_at:
            if pos < pm.start():
                page = num
            else:
                break
        lines = []
        for lm in LINE_RE.finditer(pm.group(2)):
            text = TAG_RE.sub('', lm.group(2)).strip()
            if text:
                lines.append((lm.group(1), text))
        if len(lines) >= min_lines:
            paragraphs.append((region_id, page, subtype, lines))
    return paragraphs


def build_lexicon(paragraphs_per_file) -> Tuple[Counter, Counter, Counter,
                                                Counter, Counter]:
    """
    Build unigram, bigram, dashed-compound and sentence-boundary tables from
    line-interior tokens only.

    A token that is neither the first nor the last one on its line cannot have
    been damaged by a line-break hyphen, which makes these counts a clean
    reference for judging the tokens that sit at the breaks.

    `bigrams`, `periods` and `commas` are kept apart on purpose: "aus Karl",
    "aus. Karl" and "statt, wo" are different evidence. Only the second
    supports reading a line-end `¬` as a misrecognised full stop, and only the
    third supports reading it as a comma.
    """
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    dashed: Counter = Counter()
    periods: Counter = Counter()
    commas: Counter = Counter()

    for _, _, lines in paragraphs_per_file:
        for line in lines:
            interior = line.split()[1:-1]
            cleaned = [strip_token(t) for t in interior]
            words = [w for w in cleaned if WORD_RE.fullmatch(w)]
            unigrams.update(words)
            for token in cleaned:
                # Genuine compound hyphens inside a line ("Windisch-Graetz",
                # "Franz Josefs=Kai"), the reference for judging a break mark
                # before a capitalised word. The two halves are taken from the
                # match rather than by splitting on a fixed character, so both
                # notations land under the same key.
                dash = DASHED_RE.fullmatch(token)
                if dash:
                    dashed[dash.group(1, 2)] += 1
            for i in range(len(interior) - 1):
                a, b = cleaned[i], cleaned[i + 1]
                if not (WORD_RE.fullmatch(a) and WORD_RE.fullmatch(b)):
                    continue
                # The punctuation between the two decides which table they
                # land in
                tail = interior[i].rstrip('»«"\')]')
                if tail.endswith('.'):
                    periods[(a, b)] += 1
                elif tail.endswith(','):
                    commas[(a, b)] += 1
                else:
                    bigrams[(a, b)] += 1

    return unigrams, bigrams, dashed, periods, commas


def collect_breaks(all_paragraphs) -> Dict[Tuple[str, str], PairEvidence]:
    """Collect every in-paragraph line break as a (left, right) word pair."""
    pairs: Dict[Tuple[str, str], PairEvidence] = {}

    for filename, page, lines in all_paragraphs:
        for i in range(len(lines) - 1):
            current, following = lines[i], lines[i + 1]
            pair = break_pair(current, following)
            if pair is None:
                continue
            left, right, raw_left, raw_right = pair
            has_hyphen = current.endswith(HYPHEN_CHARS)

            key = (left, right)
            ev = pairs.get(key)
            if ev is None:
                ev = pairs[key] = PairEvidence(left=left, right=right)
            # Remember the surface forms so the search & replace strings match
            # the text as it really appears ("österr.¬ungar.", not "österrungar")
            ev.surface_left[raw_left] += 1
            ev.surface_right[raw_right] += 1
            if has_hyphen:
                ev.hyphenated += 1
            else:
                ev.plain += 1

            # Keep one example per occurrence class: a FALSE_NEGATIVE has to be
            # illustrated by a break that really lacks the hyphen, and vice
            # versa.
            slot = 'hyphenated' if has_hyphen else 'plain'
            if slot not in ev.examples:
                ev.examples[slot] = (filename, page,
                                     f"{current} | {following}")
            if len(ev.contexts) < 3:
                ev.contexts.append(f"{current} | {following}")

    return pairs


def separator_form(ev: PairEvidence, ratio: float) -> Tuple[str, str]:
    """
    The two words are separate — decide whether anything stands between them.

    Called only once the join/split question is already settled in favour of
    separation, so punctuation never has to compete with the compound reading:
    a comma or full stop separates the two words just as a space does, and only
    the character written at the break is at stake here.

    A full stop is not confined to capitalised right-hand words — the
    abbreviations this corpus is built on are followed by lowercase words all
    the time ("Präs. des", "Kl. und"), so all three are weighed the same way.
    """
    comma, period, plain = ev.comma_count, ev.period_count, ev.split_count
    best = max(comma, period, plain)
    if best >= 2:
        if period == best and period >= ratio * max(comma, plain, 1):
            return 'period', 'sentence-boundary-attested'
        if comma == best and comma >= ratio * max(period, plain, 1):
            return 'comma', 'comma-attested'
    return 'split', 'bigram-attested'


def decide_form(ev: PairEvidence, ratio: float) -> Tuple[Optional[str], str]:
    """
    Decide how this word pair should be written, independently of the hyphen
    the OCR happened to produce.

    Returns (form, reason) where form is 'join' (one word), 'hyphen' (a real
    compound hyphen), 'period'/'comma' (punctuation the OCR read as a hyphen)
    or 'split' (two words), and None when the evidence is too weak or
    contradictory to decide.
    """
    j, s = ev.join_count, ev.separation

    # A capitalised right part has four competing readings: a genuine compound
    # hyphen carried over the line break ("Windisch-Graetz"), a `¬` that does
    # not belong there at all ("in¬ Wien"), a full stop the OCR read as a
    # hyphen ("nicht aus¬ | Karl Schäfer" -> "nicht aus. Karl Schäfer"), or a
    # comma ("Graf Szapáry¬ | Graf Nemes" -> "Graf Szapáry, Graf Nemes").
    # The import step currently turns every one of them into a real "-", so all
    # of them are compared explicitly against their own evidence table.
    if ev.right[:1].isupper():
        d = ev.dash_count
        if max(d, s) < 2:
            return None, 'capitalised-right'
        if d >= 2 and d >= ratio * max(s, 1):
            return 'hyphen', 'dashed-compound-attested'
        if s >= 2 and s >= ratio * max(d, 1):
            return separator_form(ev, ratio)
        return None, 'capitalised-right'

    # The left part is not a word on its own -> it can only be a fragment.
    if ev.left_count == 0 and j >= 2 and s == 0:
        return 'join', 'left-is-fragment'

    # Direct competition between the compound and the two-word reading.
    if j >= 2 and j >= ratio * max(s, 1):
        return 'join', 'compound-attested'
    if s >= 2 and s >= ratio * max(j, 1):
        return separator_form(ev, ratio)
    if j >= 2 and s >= 2:
        return None, 'contested'

    # How this very pair is written everywhere else in the corpus. A pair that
    # is hyphenated hundreds of times and plain a handful of times identifies
    # those few as dropped hyphens.
    if ev.hyphenated >= 5 and ev.hyphenated >= ratio * max(ev.plain, 1):
        return 'join', 'pair-usually-hyphenated'
    if ev.plain >= 5 and ev.plain >= ratio * max(ev.hyphenated, 1) and j == 0:
        return 'split', 'pair-usually-plain'

    # Both sides are common words and the compound was never seen.
    if j == 0 and ev.left_count >= 5 and ev.right_count >= 5:
        return 'split', 'both-common-words'

    return None, 'insufficient-evidence'


def judge(ev: PairEvidence, unigrams: Counter, bigrams: Counter,
          dashed: Counter, periods: Counter, commas: Counter, ratio: float,
          decisions: Optional[Dict[Tuple[str, str], 'Decision']] = None) -> str:
    """
    Attach lexicon evidence and derive a verdict per occurrence class.

    The same pair can be right in one place and wrong in another, so the
    hyphenated and the plain occurrences are judged separately against one
    shared decision about how the pair ought to be written.

    Sets `verdict` to one of:
        OK              hyphen usage agrees with the evidence
        FALSE_POSITIVE  hyphen present but the parts are two words
        FALSE_NEGATIVE  no hyphen although the word continues
        PERIOD          the `¬` is a misrecognised full stop
        COMMA           the `¬` is a misrecognised comma
        REVIEW          evidence too weak or contradictory

    A pair listed in `decisions` (the accumulated LLM/manual verdicts) is
    settled from there and never enters the review queue again.
    """
    ev.join_count = unigrams[ev.left + ev.right]
    ev.split_count = bigrams[(ev.left, ev.right)]
    ev.left_count = unigrams[ev.left]
    ev.right_count = unigrams[ev.right]
    ev.dash_count = dashed[(ev.left, ev.right)]
    ev.period_count = periods[(ev.left, ev.right)]
    ev.comma_count = commas[(ev.left, ev.right)]

    settled = decisions.get((ev.left, ev.right)) if decisions else None
    if settled is not None:
        ev.form, ev.reason = settled.form, f'decided:{settled.source}'
        ev.decided_by = settled.source
    else:
        ev.form, ev.reason = decide_form(ev, ratio)

    if ev.form is None:
        # Undecided. Only the hyphenated occurrences are questionable - a plain
        # line break between two words needs no explanation.
        if ev.hyphenated and ev.plain:
            ev.bad_hyphenated = ev.hyphenated
            return 'REVIEW'
        return 'OK'

    # Overturning the OCR requires the opposing reading to be essentially
    # unattested. When both readings are real words the pair is genuinely
    # contested - "nachdem" vs "nach dem", "wieder" vs "wie der" - and only
    # the surrounding sentence can settle it, so it goes to the review queue.
    # An explicit decision has already resolved that and is never second-guessed.
    if ev.form == 'join':
        if not ev.plain:
            return 'OK'
        ev.bad_plain = ev.plain
        # A comma between the two words argues against joining them just as a
        # plain space does, so the counter-evidence is the full separation
        # count, not only the comma-free bigrams.
        if ev.separation >= 2 and not settled:
            ev.reason = 'contested-both-readings'
            return 'REVIEW'
        return 'FALSE_NEGATIVE'

    if ev.form == 'hyphen':
        # A real compound hyphen: correct as long as the OCR marked the break.
        return 'OK'

    if ev.form == 'skip':
        # Explicitly marked unresolvable - leave every occurrence alone.
        return 'OK'

    if ev.form == 'context':
        # Both readings are real and the pair is answered one occurrence at a
        # time in `hyphen_occurrences.csv`. Nothing corpus-wide is true of it,
        # so it contributes no verdict here rather than a misleading `OK`:
        # `OK` would be this analysis vouching for occurrences it has not
        # looked at, which is the exact claim `context` exists to withdraw.
        return 'CONTEXT'

    if ev.form in ('period', 'comma'):
        if not ev.hyphenated:
            return 'OK'
        ev.bad_hyphenated = ev.hyphenated
        return ev.form.upper()

    if not ev.hyphenated:
        return 'OK'
    ev.bad_hyphenated = ev.hyphenated
    if (ev.join_count >= 2 or ev.dash_count >= 2) and not settled:
        ev.reason = 'contested-both-readings'
        return 'REVIEW'
    return 'FALSE_POSITIVE'


def analyse(xml_folder: Path, max_files: Optional[int], ratio: float,
            quiet: bool, decisions=None,
            skip_subtypes: FrozenSet[str] = frozenset()
            ) -> Tuple[Dict[Tuple[str, str], PairEvidence], Counter]:
    """
    Run the full analysis over the XML corpus.

    `skip_subtypes` drops whole layout classes before anything is counted, so
    an excluded region reaches neither the lexicon nor the report. It stays
    empty here rather than defaulting to `NOISY_SUBTYPES`, because the callers
    that correct the XML (`correct_xml_hyphens.py`, `mark_xml_word_breaks.py`)
    decide for themselves which regions they write to; only this script's own
    `--skip-subtypes` sets it.
    """
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No XML files found in {xml_folder}")

    if not quiet:
        print(f"Reading {len(files)} XML files from {xml_folder} ...")

    all_paragraphs = []
    notations: Counter = Counter()
    unjudged = 0
    for n, path in enumerate(files, 1):
        text = path.read_text(encoding='utf-8')
        file_lines: List[str] = []
        # Read with `min_lines=1` and filter afterwards, rather than letting
        # the reader drop the one-line paragraphs. They hold no *pair* and so
        # must stay out of `all_paragraphs` — admitting them would change the
        # lexicon and with it every verdict — but a one-line paragraph ending
        # in a break mark is in exactly the same position as any other
        # paragraph's last line, and counting the omission only where it is
        # convenient to see is how an under-report gets called complete.
        for region, page, subtype, lines in read_paragraphs(
                text, min_lines=1, skip_subtypes=skip_subtypes):
            if lines and lines[-1].endswith(HYPHEN_CHARS):
                unjudged += 1
            if len(lines) < 2:
                continue
            all_paragraphs.append((path.name, page, lines))
            file_lines.extend(lines)
        notations[detect_notation(file_lines)] += 1
        if not quiet and n % 100 == 0:
            print(f"  {n}/{len(files)} files")

    if not quiet:
        # Which typesetting the corpus is in decides nothing here - the
        # verdicts are forms, not characters - but it is what the correction
        # step writes back, so it is worth seeing before that runs.
        mix = ', '.join(f"{char} {count:,} files"
                        for char, count in notations.most_common())
        print(f"Line-break notation: {mix}")
        print("Building lexicon from line-interior tokens ...")

    # Files that parse but yield no paragraphs are not an empty corpus, they are
    # a reader that has stopped matching. That is exactly how the `n='1'`
    # attribute `set_xml_readingorder.py` added went unnoticed: the run carried
    # on to a division by zero several screens later, far from the cause.
    if not all_paragraphs:
        raise SystemExit(
            f"Read {len(files)} XML files from {xml_folder} but found no "
            f"paragraphs.\nEither every region was skipped by "
            f"--skip-subtypes, or the files are not empty and PARA_RE/LINE_RE "
            f"in "
            f"{Path(__file__).name} no longer match what `<p>`/`<l>` look "
            f"like — check their attributes.")

    unigrams, bigrams, dashed, periods, commas = build_lexicon(
        all_paragraphs)
    if not quiet:
        print(f"  {len(unigrams):,} word types / {sum(unigrams.values()):,} tokens")
        print(f"  {len(bigrams):,} bigram types")
        print("Judging line breaks ...")

    pairs = collect_breaks(all_paragraphs)
    stats: Counter = Counter()

    # Break marks this analysis cannot reach: the last line of a paragraph
    # carries one, but its continuation is in the next column or on the next
    # page and no script in `lineends` can see across that boundary yet. They are
    # not judged, not counted as OK, and — until now — not mentioned, which is
    # the part worth fixing: an omission nobody can see reads exactly like an
    # absence of errors. Closing it needs the `@next` chain
    # `merge_xml_factoids.py` will write; see
    # docs/hyphen-occurrence-disambiguation.md.
    stats[UNJUDGED] = unjudged
    for ev in pairs.values():
        ev.verdict = judge(ev, unigrams, bigrams, dashed, periods, commas,
                           ratio, decisions)
        if ev.verdict == 'OK':
            stats['OK'] += ev.total
        elif ev.verdict == 'CONTEXT':
            # Every occurrence is counted, not just the ones a form would have
            # changed. There is no form here to change anything, and folding
            # these into `OK` would report the analysis as having approved
            # breaks it deliberately declined to judge.
            stats['CONTEXT'] += ev.total
        else:
            # Only the occurrences that would actually change count as errors;
            # the rest of the pair's occurrences were fine.
            stats[ev.verdict] += ev.affected
            stats['OK'] += ev.total - ev.affected

    return pairs, stats


def export_to_csv(pairs, output_path: Path, report: str, min_count: int):
    """Write the decision table."""
    wanted = {
        'all': {'FALSE_POSITIVE', 'FALSE_NEGATIVE', 'PERIOD', 'COMMA',
                'REVIEW', 'CONTEXT'},
        'false-positives': {'FALSE_POSITIVE'},
        'false-negatives': {'FALSE_NEGATIVE'},
        'review': {'REVIEW'},
        'context': {'CONTEXT'},
    }[report]

    # A CONTEXT pair changes nothing corpus-wide, so `affected` is 0 for all of
    # them and the `min_count` floor - which counts changes - would drop every
    # one. They are held to their occurrence count instead, that being the
    # quantity they actually have.
    rows = [ev for ev in pairs.values()
            if ev.verdict in wanted
            and (ev.total if ev.verdict == 'CONTEXT' else ev.affected)
            >= min_count]
    rows.sort(key=lambda e: (e.verdict, -e.affected))

    fieldnames = [
        'verdict', 'reason', 'left', 'right', 'find', 'replace',
        'affected', 'seen_hyphenated', 'seen_plain', 'compound_freq',
        'bigram_freq', 'dashed_freq', 'period_freq', 'comma_freq',
        'left_freq', 'right_freq',
        'example_context', 'example_file', 'example_page', 'transkribus_link',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ev in rows:
            # `find`/`replace` are written against the joined paragraph text,
            # i.e. after the import step has already glued the lines together.
            sl, sr = ev.surface_pair()
            if ev.verdict == 'FALSE_POSITIVE':
                find, replace = f"{sl}{sr}", f"{sl} {sr}"
                slot = 'hyphenated'
            elif ev.verdict == 'FALSE_NEGATIVE':
                find, replace = f"{sl} {sr}", f"{sl}{sr}"
                slot = 'plain'
            elif ev.verdict in ('PERIOD', 'COMMA'):
                # What the import currently produces from the break mark: a
                # real dash before a capitalised word, otherwise the two halves
                # glued together. `-` regardless of whether the source had `¬`,
                # `-` or `=`, since the import normalises all three
                # (see hyphen_utils.determine_hyphen_action).
                find = f"{sl}-{sr}" if sr[:1].isupper() else f"{sl}{sr}"
                mark = '.' if ev.verdict == 'PERIOD' else ','
                replace = f"{sl.rstrip('.,')}{mark} {sr}"
                slot = 'hyphenated'
            else:
                find = replace = ""
                slot = 'hyphenated' if 'hyphenated' in ev.examples else 'plain'

            filename, page, context = ev.examples.get(slot) or \
                next(iter(ev.examples.values()), ("", 0, ""))
            link = ""
            if get_transkribus_link and filename:
                try:
                    link = get_transkribus_link(filename, page=page)
                except Exception:
                    link = ""
            writer.writerow({
                'verdict': ev.verdict,
                'reason': ev.reason,
                'left': ev.left,
                'right': ev.right,
                'find': find,
                'replace': replace,
                'affected': ev.affected,
                'seen_hyphenated': ev.hyphenated,
                'seen_plain': ev.plain,
                'compound_freq': ev.join_count,
                'bigram_freq': ev.split_count,
                'dashed_freq': ev.dash_count,
                'period_freq': ev.period_count,
                'comma_freq': ev.comma_count,
                'left_freq': ev.left_count,
                'right_freq': ev.right_count,
                'example_context': context,
                'example_file': filename,
                'example_page': page,
                'transkribus_link': link,
            })
    print(f"\nDecision table saved to: {output_path}  ({len(rows):,} rows)")


def in_output(value: Path, date_prefix: str) -> Path:
    """
    Place a JSONL export next to the CSV in output/, date-prefixed.

    A bare filename ("review.jsonl") lands in `PATHS['output']` with the run's
    date in front, matching the CSV and every other validation script. A value
    that carries a directory ("/tmp/x.jsonl", "sub/x.jsonl") is taken literally,
    so a one-off destination still works. A name that already starts with the
    date is not prefixed twice.
    """
    if value.parent != Path('.'):
        return value
    name = value.name
    if not name.startswith(f"{date_prefix}_"):
        name = f"{date_prefix}_{name}"
    PATHS['output'].mkdir(parents=True, exist_ok=True)
    return PATHS['output'] / name


def question_record(ev: PairEvidence) -> dict:
    """One pair as a question for the LLM: the readings, evidence and context."""
    sl, sr = ev.surface_pair()
    return {
        'left': ev.left,
        'right': ev.right,
        # The five readings, spelled out as the model should answer
        'candidates': {
            'join': f"{sl}{sr}",
            'hyphen': f"{sl}-{sr}",
            'period': f"{sl.rstrip('.,')}. {sr}",
            'comma': f"{sl.rstrip('.,')}, {sr}",
            'split': f"{sl} {sr}",
        },
        'affected': ev.affected,
        'evidence': {
            'compound_freq': ev.join_count,
            'bigram_freq': ev.split_count,
            'dashed_freq': ev.dash_count,
            'period_freq': ev.period_count,
            'comma_freq': ev.comma_count,
            'left_freq': ev.left_count,
            'right_freq': ev.right_count,
        },
        'reason': ev.reason,
        'contexts': ev.contexts,
    }


def export_review_jsonl(pairs, output_path: Path, min_count: int):
    """
    Write the contested cases as JSONL for LLM adjudication.

    One record per distinct pair - not per occurrence - so the same question
    is never asked twice. Each record carries the sentence context the model
    needs to decide.
    """
    rows = [ev for ev in pairs.values()
            if ev.verdict == "REVIEW" and ev.affected >= min_count]
    rows.sort(key=lambda e: -e.affected)

    with open(output_path, 'w', encoding='utf-8') as fh:
        for ev in rows:
            fh.write(json.dumps(question_record(ev), ensure_ascii=False) + '\n')
    print(f"Review set saved to: {output_path}  ({len(rows):,} pairs)")


def export_benchmark_jsonl(pairs, output_path: Path, size: int, seed: int = 0):
    """
    Write a labelled test set for comparing models against each other.

    The pairs the statistics settled *confidently* double as ground truth: they
    are the same kind of question as the review queue, only with a known
    answer, so a model can be scored on them before being trusted with the
    cases nobody knows. Sampling is stratified over the forms, because `join`
    and `split` would otherwise crowd out the rare `period`/`comma`/`hyphen`
    decisions that are the interesting ones.

    Pairs settled from the decision store are excluded — their answer came from
    a model, and scoring a model against another model's output measures
    agreement, not correctness.
    """
    import random

    by_form: Dict[str, List[PairEvidence]] = {}
    for ev in pairs.values():
        if (ev.form in (None, 'skip', 'context') or ev.decided_by
                or ev.verdict == 'REVIEW'):
            continue
        # Needs enough evidence to be trustworthy, and context to reason from
        if ev.total < 2 or not ev.contexts:
            continue
        # A left word that already carries punctuation ("Wien,") makes a
        # confusing question - several forms then render alike, or produce a
        # spelling nobody would write - so those are kept out of the test set.
        if ev.surface_pair()[0].rstrip()[-1:] in '.,;:':
            continue
        # The expected answer still has to be identifiable among the rest.
        spellings = question_record(ev)['candidates']
        if list(spellings.values()).count(spellings[ev.form]) > 1:
            continue
        by_form.setdefault(ev.form, []).append(ev)

    rng = random.Random(seed)
    per_form = max(1, size // max(len(by_form), 1))
    sample: List[PairEvidence] = []
    for form, evs in sorted(by_form.items()):
        evs.sort(key=lambda e: -e.total)
        sample.extend(rng.sample(evs, min(per_form, len(evs))))
    rng.shuffle(sample)

    with open(output_path, 'w', encoding='utf-8') as fh:
        for ev in sample:
            record = question_record(ev)
            record['expected'] = ev.form
            record['expected_reason'] = ev.reason
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')

    counts = Counter(ev.form for ev in sample)
    print(f"Benchmark set saved to: {output_path}  ({len(sample):,} pairs, "
          f"{dict(counts)})")


def main():
    parser = argparse.ArgumentParser(
        description="Validate line-end hyphens in Transkribus TEI-XML.")
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--skip-subtypes', default=','.join(SKIPPED_SUBTYPES),
                        help='Comma-separated region subtypes to leave out of '
                             'the lexicon and the judged breaks (default: '
                             f"{','.join(SKIPPED_SUBTYPES)})")
    parser.add_argument('--report', default='all',
                        choices=['all', 'false-positives', 'false-negatives',
                                 'review', 'context'])
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=3.0)
    parser.add_argument('--export-review', type=Path)
    parser.add_argument('--export-benchmark', type=Path,
                        help='Write a labelled test set of confidently settled '
                             'pairs, for comparing models')
    parser.add_argument('--benchmark-size', type=int, default=200)
    parser.add_argument('--decisions', type=Path,
                        default=PATHS['csv'] / 'hyphen_decisions.csv',
                        help='Accumulated LLM/manual verdicts, consulted first')
    parser.add_argument('--output', default='hyphen_validation.csv')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    skip_subtypes = frozenset(
        s.strip() for s in args.skip_subtypes.split(',') if s.strip())
    if skip_subtypes and not args.quiet:
        print(f"Excluded region subtypes: {', '.join(sorted(skip_subtypes))}")

    decisions = load_decisions(args.decisions)
    if decisions and not args.quiet:
        print(f"Decision store: {len(decisions):,} pairs already settled "
              f"({args.decisions})")

    pairs, stats = analyse(args.xml_folder, args.max_files, args.ratio,
                           args.quiet, decisions, skip_subtypes)

    verdicts = ('OK', 'FALSE_POSITIVE', 'FALSE_NEGATIVE', 'PERIOD',
                'COMMA', 'REVIEW', 'CONTEXT')
    # The unjudged count is not a verdict and must not dilute the percentages:
    # it counts breaks nobody looked at, not breaks that came back clean.
    total = sum(stats[verdict] for verdict in verdicts)
    print("\n" + "=" * 62)
    print("Line-break verdicts (by occurrence)")
    print("=" * 62)
    for verdict in verdicts:
        count = stats[verdict]
        print(f"  {verdict:15s} {count:9,}  ({count / max(total, 1):6.2%})")
    print(f"  {'TOTAL':15s} {total:9,}")
    unjudged = stats[UNJUDGED]
    if unjudged:
        print(f"\n  {unjudged:,} further break marks sit on a paragraph's last "
              f"line and were\n  not judged at all: the word continues in the "
              f"next column or on the next\n  page, which nothing in lineends can "
              f"reach until `merge_xml_factoids.py`\n  writes the `@next` chain "
              f"(docs/hyphen-occurrence-disambiguation.md).")

    date_prefix = datetime.date.today().strftime('%Y%m%d')
    output_path = PATHS['output'] / f"{date_prefix}_{args.output}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_csv(pairs, output_path, args.report, args.min_count)

    if args.export_review:
        export_review_jsonl(pairs, in_output(args.export_review, date_prefix),
                            args.min_count)

    if args.export_benchmark:
        export_benchmark_jsonl(pairs,
                               in_output(args.export_benchmark, date_prefix),
                               args.benchmark_size)


if __name__ == '__main__':
    main()
