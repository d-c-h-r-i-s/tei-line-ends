#!/usr/bin/env python3
"""
XML Line-End Truncation Detection
=================================

Finds line-final words the OCR cut short - "Gra" for "Graf", "arrangier" for
"arrangiert", "Hote" for "Hotel" - and writes them out for review.
This is detector A of `docs/ocr-word-correction.md`, which measured the class
and designed the rules; `validate_line_end_chars.py` names it as the failure
mode its substitutions cannot repair:

    Scht -> Schw      The token is *truncated*, not substituted - the OCR
    Geno -> Genf      dropped characters rather than misreading one.

A substitution detector proposing a fix for a truncation makes the text worse,
so the two classes need different candidates: the substituted final letter
there, the *added* final letters here.

Method - the same lexicon trick, turned around
----------------------------------------------
The reference is the one every line-end script uses: tokens from the interior
of a line, a position no line-end error can reach, via `build_lexicon`. A
line-final token is flagged when all of these hold:

  1. The line does not end in a break mark. `Ver¬` is a fragment on purpose,
     and the hyphen scripts own it.
  2. The raw token ends in a letter. A cut-off word cannot end in `.`, `,` or
     `»`, and this is the single strongest filter the plan found - it removes
     `Frank,`, `überreicht.`, `Tir.`, `geg.` in one step.
  3. The stripped token is alphabetic, at least `--min-length` letters, and
     essentially unknown inside a line (`--max-observed`, default 1).
  4. Adding one letter gives a word the corpus uses (`--min-count`, default
     10). One prefix index over the lexicon, built once. `--max-missing`
     allows two or three, which the hand check found worthless (below).
  5. It is not a dropped hyphen. "Hard | egg-Gudenus" and "Öster | reich-Este"
     are a word continued on the next line, not a truncated one, so the token
     is tested joined to the next line's first token, to that token's leading
     letters, and as the left half of a dashed compound.
  6. Some completion has been seen beside this token's neighbours - after the
     previous word or before the next one, inside a line somewhere in the
     corpus. "den [Winte] / immer" passes on "Winter immer"; "seine
     [Bezugsrechte] / verkauft" does not, because nothing says "Bezugsrechtes
     verkauft", and it is not cut off. `--keep-unsupported` lifts it.

Rules 5 and 6 need the next line, so a paragraph's last line is not judged.
Its continuation is in another column or on another page; the count of lines
that pass rules 1-4 there is reported, so the gap is visible rather than
silent.

What the hand check settled (60 rows, 2026-09-18)
--------------------------------------------------
A random sample of the first full run, rules 1-5 and up to three letters, came
back 21 truncated with a right completion, 2 truncated without one, 3 dropped
hyphens, 34 not truncated: **38% against the plan's 80% bar.** Two things
separated them, and they became rules 4 and 6:

                                          rows   truncated
    one letter missing                      40      22   (55%)
    two or three letters                    20       1   ( 5%)
    a completion seen beside a neighbour    29      22   (76%)
    none seen                               31       1   ( 3%)
    both                                    25      21   (84%)

The neighbour test was written down before the labels existed and held on the
half of the sample nobody had looked at (11 of 16 against 0 of 14). Together
the two keep all 21 right answers; the only truncations lost are the two
whose completions were all wrong anyway. What they remove is mostly correct
German that is merely rare - "selbständige", "eingetroffene", "olympische",
whose "-n" form is commoner - then names and foreign words ("Salmhof",
"Commendator", "Chas"), and compounds split over a dropped hyphen
("Filial / Reservespitale"), which no frequency test can settle because the
compound occurs nowhere else. 84% is 21 of 25 rows: an estimate, not a
guarantee.

What is proposed, and what is not
---------------------------------
Up to three completions per token, never one answer: detection is often right
where the completion is not ("ers" -> "ersten" / "erster" / "erst"). A token
qualifies on a one-letter completion, but completions of up to three letters
are offered beside it, because the answer can be longer than what qualified
it: "Gräfin [Gabrie]" qualifies on "Gabriel" and is "Gabriele". They are
ordered by how often the corpus has seen each beside this token's neighbours,
then by frequency - an ordering for the reviewer, not a verdict. Each row also
carries the best one-letter *substitution*, when there is one, because "Grafer"
may be a cut-off "Grafern" or a misread "Grafen" and the two detectors should
not pretend the other does not exist.

Nothing here writes to `data/xml/`. A truncation is wrong at one line end only,
so the fix belongs at that line and not in a global replacement rule; the
correction step for it does not exist yet.

Not in scope, and where it lives instead:

  * A cut-off word that is *itself* a word - "au[f]", "de[r]", "is[t]" - is
    invisible to rule 3 by design. It is the ambiguous class, and belongs with
    `resolve_line_end_context.py` as a completion candidate beside its
    substitutions (measured 2026-09-17: ~240 line ends).
  * A missing abbreviation dot ("H[.]", "ung[.]"). Many abbreviations appear
    without their dot inside a line too, so the statistics cannot see it.

The gap to the region border (`gap_px`)
---------------------------------------
Measured before this was built, over 646,799 line ends with a next line: the
hits do **not** end at their region's right border, which is where a word
clipped by the region polygon would end. They end further from it - median
25 px against 17 px for all line ends - and the hit rate rises with the gap:

    gap <= 5 px    0.048%       15-40 px   0.143%
    5-15 px        0.052%       40-100 px  0.254%

So the region border is not what cuts these words; the line polygon stopping
short of the word is the likelier mechanism. The hand check agrees: the real
truncations end a median 29 px from the border, the rest 18 px, and of the 21
rows within 15 px only 3 were truncated. It stays a column and not a rule -
rule 6 already holds back most of what it would, and 60 rows are too few to
set a second threshold on.

Usage:
    python validate_line_end_truncations.py
    python validate_line_end_truncations.py --max-files 50
    python validate_line_end_truncations.py --sample 60
    python validate_line_end_truncations.py --export-review truncations.jsonl

Author: Christian Lendl
Created: 2026-09-17
"""

import argparse
import csv
import datetime
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS
from validate_line_end_chars import ALPHABET
from validate_xml_hyphens import (
    ATTR_RE,
    HYPHEN_CHARS,
    SKIPPED_SUBTYPES,
    WORD_RE,
    build_lexicon,
    in_output,
    read_paragraph_lines,
    strip_token,
)

try:
    from utils.get_transkribus_link import get_transkribus_link
except ImportError:  # pragma: no cover - link is a convenience only
    get_transkribus_link = None

# The plan's thresholds, kept as they were measured: a completion the corpus
# uses at least ten times, a token it has seen at most once inside a line.
MIN_COUNT = 10
MAX_OBSERVED = 1

# Three letters, because two-letter tokens are initials, particles and the
# ambiguous class ("de", "zu") far more often than they are truncations.
MIN_LENGTH = 3

# One. The plan allowed three, and the hand check found 1 truncation among 20
# rows that needed more than one letter - and for that one, no completion
# offered was right. The reviewer had seen the same in Transkribus.
MAX_MISSING = 1

# What is *offered*, as against what qualifies. A token qualifies on a
# one-letter completion, but the right answer can still be longer: "Gräfin
# [Gabrie]" qualifies on "Gabriel" and is "Gabriele". Limiting the list to
# what qualifies would put the wrong name first on every such row.
SHOWN_MISSING = 3

TOP_COMPLETIONS = 3

# Words of the next line shown in the CSV context. More than one, because the
# 2026-09-17 hand check found the deciding word often further away than that.
NEXT_WORDS = 6


@dataclass
class Truncation:
    """One line-final token that looks cut short."""
    file: str
    page: int
    line_facs: str
    region_id: str
    subtype: str
    raw_token: str
    observed: str
    line: str
    next_line: str
    previous_line: Optional[str]
    observed_freq: int
    # (word, corpus frequency, times seen beside this token's neighbours)
    completions: List[Tuple[str, int, int]] = field(default_factory=list)
    substitution: Optional[Tuple[str, int]] = None
    gap_px: Optional[int] = None
    type_count: int = 0

    @property
    def best(self) -> Tuple[str, int, int]:
        return self.completions[0]

    @property
    def missing(self) -> int:
        return len(self.best[0]) - len(self.observed)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def read_corpus(xml_folder: Path, max_files: Optional[int], skip_subtypes,
                quiet: bool):
    """
    Every paragraph once, in the two shapes needed.

    `for_lexicon` is what `build_lexicon` takes; `regions` keeps the line ids,
    the page and the region id, which is what a row needs to be found again -
    in Transkribus by the reviewer, in `<facsimile>` for its geometry.
    """
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No XML files found in {xml_folder}")
    if not quiet:
        print(f"Reading {len(files)} XML files from {xml_folder} ...")

    for_lexicon, regions = [], []
    for number, path in enumerate(files, 1):
        text = path.read_text(encoding='utf-8')
        for region_id, page, subtype, lines in read_paragraph_lines(
                text, min_lines=1, skip_subtypes=skip_subtypes):
            for_lexicon.append((path.name, page, [t for _, t in lines]))
            regions.append((path.name, page, region_id, subtype, lines))
        if not quiet and number % 200 == 0:
            print(f"  {number}/{len(files)} files")
    return for_lexicon, regions


def completion_index(unigrams: Counter, min_count: int, max_missing: int,
                     min_length: int) -> Dict[str, List[Tuple[str, int]]]:
    """
    Prefix -> the frequent words it is a prefix of, by one to `max_missing`
    letters.

    Built once over the lexicon rather than generated per token: a token has
    `30 ** 3` three-letter extensions and the corpus has a few hundred
    thousand line ends, while the lexicon holds only the words that exist.
    """
    index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for word, count in unigrams.items():
        if count < min_count or not WORD_RE.fullmatch(word):
            continue
        for cut in range(1, max_missing + 1):
            if len(word) - cut >= min_length:
                index[word[:-cut]].append((word, count))
    return index


def continues_on_next_line(observed: str, next_raw: str, unigrams: Counter,
                           dashed: Counter) -> bool:
    """
    Rule 5: is this a word carried over a dropped hyphen instead?

    Tested three ways because each leaks on its own. The plain join misses a
    next token that is itself a compound ("Hard | egg-Gudenus" strips to
    "egg-Gudenus"); its leading letters catch that; and a token that is the
    left half of a dashed compound ("Öster | reich-Este") is a break too.
    """
    following = strip_token(next_raw)
    if unigrams[observed + following]:
        return True
    lead = WORD_RE.match(following)
    if lead and unigrams[observed + lead.group(0)]:
        return True
    return bool(dashed[(observed, following)]) or bool(
        lead and dashed[(observed, lead.group(0))])


def best_substitution(observed: str, unigrams: Counter,
                      min_count: int) -> Optional[Tuple[str, int]]:
    """The commonest one-letter substitution of the final letter, if any."""
    found = [(observed[:-1] + char, unigrams[observed[:-1] + char])
             for char in ALPHABET if char != observed[-1].lower()]
    found = [f for f in found if f[1] >= min_count]
    return max(found, key=lambda f: f[1]) if found else None


def detect(regions, unigrams: Counter, bigrams: Counter, dashed: Counter,
           args) -> Tuple[List[Truncation], Counter]:
    """
    Apply rules 1-6 to every line end.

    Returns the hits and what was held back after rule 4: paragraph-final line
    ends that could not be judged, having no next line for rules 5 and 6, and
    the ones rule 6 turned down.
    """
    index = completion_index(unigrams, args.min_count,
                             max(args.max_missing, SHOWN_MISSING),
                             args.min_length)
    hits: List[Truncation] = []
    held: Counter = Counter()

    for filename, page, region_id, subtype, lines in regions:
        for position, (line_facs, line) in enumerate(lines):
            tokens = line.split()
            if not tokens or line.endswith(HYPHEN_CHARS):            # rule 1
                continue
            raw = tokens[-1]
            if not raw[-1].isalpha():                                 # rule 2
                continue
            observed = strip_token(raw)
            if (not observed.isalpha() or len(observed) < args.min_length
                    or unigrams[observed] > args.max_observed):       # rule 3
                continue
            candidates = index.get(observed) or []
            qualifying = [(word, count) for word, count in candidates
                          if len(word) - len(observed) <= args.max_missing]
            if not qualifying:                                        # rule 4
                continue
            if position + 1 >= len(lines):
                held['paragraph-final'] += 1
                continue
            next_line = lines[position + 1][1]
            next_tokens = next_line.split()
            if next_tokens and continues_on_next_line(
                    observed, next_tokens[0], unigrams, dashed):      # rule 5
                continue

            previous = strip_token(tokens[-2]) if len(tokens) > 1 else ''
            following = strip_token(next_tokens[0]) if next_tokens else ''
            def beside(word: str) -> int:
                return bigrams[(previous, word)] + bigrams[(word, following)]
            if (not any(beside(word) for word, _ in qualifying)
                    and not args.keep_unsupported):                   # rule 6
                held['no completion beside a neighbour'] += 1
                continue
            ranked = sorted(((word, count, beside(word))
                             for word, count in candidates),
                            key=lambda c: (-c[2], -c[1], len(c[0])))
            hits.append(Truncation(
                file=filename, page=page, line_facs=line_facs,
                region_id=region_id, subtype=subtype, raw_token=raw,
                observed=observed, line=line, next_line=next_line,
                previous_line=lines[position - 1][1] if position else None,
                observed_freq=unigrams[observed],
                completions=ranked[:TOP_COMPLETIONS],
                substitution=best_substitution(observed, unigrams,
                                               args.min_count)))

    per_type = Counter(hit.observed for hit in hits)
    for hit in hits:
        hit.type_count = per_type[hit.observed]
    return hits, held


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def zone_polygon(head: str, zone_id: str) -> Optional[List[Tuple[float, float]]]:
    """
    The `points` polygon of one `<zone>` in `<facsimile>`.

    Looked up by id rather than parsed wholesale: a page's line polygons run to
    megabytes of coordinates and only the few hundred flagged lines need theirs.
    """
    at = head.find(f"xml:id='{zone_id}'")
    if at == -1:
        return None
    start = head.rfind('<zone', 0, at)
    end = head.find('>', at)
    points = dict(ATTR_RE.findall(head[start:end])).get('points')
    if not points:
        return None
    return [tuple(map(float, p.split(','))) for p in points.split() if ',' in p]


def right_edge_at(polygon: List[Tuple[float, float]], y: float) -> float:
    """
    The region's rightmost x at height `y`.

    Not simply its maximum x: a region shaped around an image is an L, and a
    line in the notch has its border much further left than the region's
    widest point.
    """
    crossings = []
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 <= y < y2) or (y2 <= y < y1):
            crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return max(crossings) if crossings else max(x for x, _ in polygon)


def attach_gaps(hits: List[Truncation], xml_folder: Path) -> None:
    """Distance from each flagged line's end to its region's right border."""
    by_file: Dict[str, List[Truncation]] = defaultdict(list)
    for hit in hits:
        by_file[hit.file].append(hit)
    for filename, group in by_file.items():
        text = (xml_folder / filename).read_text(encoding='utf-8')
        head = text[:text.find('<text')]
        for hit in group:
            line = zone_polygon(head, hit.line_facs)
            region = zone_polygon(head, hit.region_id)
            if not line or not region:
                continue
            ys = [y for _, y in line]
            middle = (min(ys) + max(ys)) / 2
            hit.gap_px = round(right_edge_at(region, middle)
                               - max(x for x, _ in line))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = ['verdict', 'observed', 'completion', 'missing', 'completions',
              'substitution', 'observed_freq', 'type_count', 'gap_px',
              'context', 'file', 'page', 'line_facs', 'region_subtype',
              'transkribus_link']


def context_string(hit: Truncation) -> str:
    """
    The whole line with the raw token marked, then the start of the next line.

    The raw token, punctuation and all: the sample CSVs of the context stage
    showed the stripped one, and a reviewer reading "[ung]" had no way to see
    the full stop the XML actually carried.
    """
    head = hit.line.rsplit(None, 1)[0] if len(hit.line.split()) > 1 else ''
    tail = ' '.join(hit.next_line.split()[:NEXT_WORDS])
    return f"{head} [{hit.raw_token}] / {tail}".strip()


def format_completions(hit: Truncation) -> str:
    return '; '.join(f"{word} {count:,}" + (f" (beside {fit:,})" if fit else '')
                     for word, count, fit in hit.completions)


def row_for(hit: Truncation, links: Dict[Tuple[str, int], str]) -> dict:
    return {
        'verdict': '',
        'observed': hit.observed,
        'completion': hit.best[0],
        'missing': hit.missing,
        'completions': format_completions(hit),
        'substitution': (f"{hit.substitution[0]} {hit.substitution[1]:,}"
                         if hit.substitution else ''),
        'observed_freq': hit.observed_freq,
        'type_count': hit.type_count,
        'gap_px': '' if hit.gap_px is None else hit.gap_px,
        'context': context_string(hit),
        'file': hit.file,
        'page': hit.page,
        'line_facs': hit.line_facs,
        'region_subtype': hit.subtype,
        'transkribus_link': links.get((hit.file, hit.page), ''),
    }


def transkribus_links(hits: List[Truncation]) -> Dict[Tuple[str, int], str]:
    """One lookup per page, not per row; a missing issue is a gap, not a failure."""
    links: Dict[Tuple[str, int], str] = {}
    if not get_transkribus_link:
        return links
    for key in {(hit.file, hit.page) for hit in hits}:
        try:
            links[key] = get_transkribus_link(key[0], page=key[1])
        except Exception:
            links[key] = ''
    return links


def export_csv(hits: List[Truncation], path: Path,
               links: Dict[Tuple[str, int], str]) -> None:
    """
    One row per line end, repeated truncations together.

    Per occurrence rather than per type as the plan had it: a truncation is
    wrong at one line only, which is where both the reviewer in Transkribus and
    a later correction step have to find it. `type_count` keeps the per-type
    view the plan wanted.
    """
    ordered = sorted(hits, key=lambda h: (-h.type_count, h.observed.lower(),
                                          h.file, h.line_facs))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for hit in ordered:
            writer.writerow(row_for(hit, links))
    print(f"\nTruncation table saved to: {path}  ({len(hits):,} rows)")


def export_sample(hits: List[Truncation], path: Path, size: int,
                  links: Dict[Tuple[str, int], str], seed: int = 0) -> None:
    """
    A random hand-check sample, `verdict` left blank.

    Plain random, not stratified: the question it answers is the one the plan
    set as the bar - is detector A right more than 80% of the time - and that
    is a question about the whole output.
    """
    picked = random.Random(seed).sample(hits, min(size, len(hits)))
    picked.sort(key=lambda h: (h.file, h.page, h.line_facs))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for hit in picked:
            writer.writerow(row_for(hit, links))
    print(f"Hand-check sample saved to: {path}  ({len(picked):,} rows)")
    print("  Fill `verdict` with y (truncated, and a completion is right), "
          "t (truncated, but no\n  completion offered is right) or n (not "
          "truncated).")


def export_review_jsonl(hits: List[Truncation], path: Path) -> None:
    """
    One record per line end, for a person or a model to settle.

    The transcribed form is always the first candidate, so "leave it alone" is
    as easy an answer as any completion - the same rule `resolve_line_end_llm`
    follows. Context crosses line boundaries: the previous, current and next
    line, because the word that settles an inflection is often not on the line
    that ends in it.
    """
    with open(path, 'w', encoding='utf-8') as fh:
        for hit in hits:
            fh.write(json.dumps({
                'file': hit.file,
                'page': hit.page,
                'line_facs': hit.line_facs,
                'observed': hit.observed,
                'raw_token': hit.raw_token,
                'candidates': [hit.observed] + [c[0] for c in hit.completions],
                'evidence': {
                    'observed_freq': hit.observed_freq,
                    **{word: count for word, count, _ in hit.completions},
                },
                'substitution': hit.substitution[0] if hit.substitution else '',
                'context': {
                    'previous_line': hit.previous_line or '',
                    'line': hit.line,
                    'next_line': hit.next_line,
                },
                'gap_px': hit.gap_px,
                'type_count': hit.type_count,
            }, ensure_ascii=False) + '\n')
    print(f"Review set saved to: {path}  ({len(hits):,} line ends)")


def report(hits: List[Truncation], held: Counter, judged_lines: int) -> None:
    print("\n" + "=" * 62)
    print("Line-end truncations")
    print("=" * 62)
    print(f"  line ends flagged      {len(hits):7,}  "
          f"({len(hits) / max(judged_lines, 1):.3%} of {judged_lines:,})")
    print(f"  distinct tokens        {len({h.observed for h in hits}):7,}")
    for reason, count in held.items():
        print(f"  held back, {reason}: {count:,}")

    print("\n  letters missing (best completion):")
    for missing, count in sorted(Counter(h.missing for h in hits).items()):
        print(f"    {missing}   {count:6,}")

    with_sub = sum(1 for h in hits if h.substitution)
    print(f"\n  also have a one-letter substitution: {with_sub:,} "
          f"(truncation or misread - the reviewer decides)")

    print("\n  by region subtype:")
    for subtype, count in Counter(h.subtype for h in hits).most_common():
        print(f"    {subtype or '(none)':16s} {count:6,}")

    gaps = sorted(h.gap_px for h in hits if h.gap_px is not None)
    if gaps:
        print(f"\n  gap to region border: median {gaps[len(gaps) // 2]} px "
              f"(all line ends: 17 px, see docstring)")

    print("\n  most frequent:")
    for observed, count in Counter(h.observed for h in hits).most_common(10):
        example = next(h for h in hits if h.observed == observed)
        print(f"    {count:4,}  {observed:>14} -> {format_completions(example)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Detect line-final words the OCR cut short.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--skip-subtypes', default=','.join(SKIPPED_SUBTYPES),
                        help='Comma-separated region subtypes to leave out of '
                             'the lexicon and the scan alike')
    parser.add_argument('--min-count', type=int, default=MIN_COUNT,
                        help='Interior occurrences a completion needs '
                             f'(default: {MIN_COUNT})')
    parser.add_argument('--max-observed', type=int, default=MAX_OBSERVED,
                        help='Interior occurrences the token may have and '
                             f'still count as unknown (default: {MAX_OBSERVED})')
    parser.add_argument('--min-length', type=int, default=MIN_LENGTH)
    parser.add_argument('--max-missing', type=int, default=MAX_MISSING,
                        help=f'Letters a completion may add (default: '
                             f'{MAX_MISSING}; the plan had 3, see the docstring)')
    parser.add_argument('--keep-unsupported', action='store_true',
                        help='Lift rule 6: keep tokens none of whose '
                             'completions has been seen beside a neighbour')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='Also write a random N-row hand-check sample')
    parser.add_argument('--export-review', type=Path,
                        help='Write the hits as JSONL with three lines of '
                             'context each')
    parser.add_argument('--output', default='line_end_truncations.csv')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    skip_subtypes = frozenset(
        s.strip() for s in args.skip_subtypes.split(',') if s.strip())
    for_lexicon, regions = read_corpus(args.xml_folder, args.max_files,
                                       skip_subtypes, args.quiet)
    unigrams, bigrams, dashed, _, _ = build_lexicon(for_lexicon)
    if not args.quiet:
        print(f"  {len(unigrams):,} interior word types / "
              f"{sum(unigrams.values()):,} tokens")

    hits, held = detect(regions, unigrams, bigrams, dashed, args)
    attach_gaps(hits, args.xml_folder)
    judged = sum(1 for *_, lines in regions for _, line in lines
                 if line.split() and not line.endswith(HYPHEN_CHARS))
    report(hits, held, judged)

    date_prefix = datetime.date.today().strftime('%Y%m%d')
    output_path = PATHS['output'] / f"{date_prefix}_{args.output}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    links = transkribus_links(hits)
    export_csv(hits, output_path, links)

    if args.sample:
        stem = Path(args.output).stem
        export_sample(hits, PATHS['output'] / f"{date_prefix}_{stem}_sample.csv",
                      args.sample, links)
    if args.export_review:
        export_review_jsonl(hits, in_output(args.export_review, date_prefix))
    return 0


if __name__ == '__main__':
    sys.exit(main())
