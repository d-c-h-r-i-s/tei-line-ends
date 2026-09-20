#!/usr/bin/env python3
"""
XML Line-End Character Validation
=================================

Transkribus misreads the *last character of a line* far more often than a
character anywhere else — the letterform is clipped by the margin, sits next to
the break hyphen, or is simply the last thing the line model sees. The signature
case is a final `n` or `m` coming out as `r`:

    Kronen -> Kroner        Wien -> Wier         Grafen -> Grafer
    Damen  -> Damer         Sohn -> Sohr         Namen  -> Namer

`r` is only the loudest case. The same test over every final letter finds
`l->n`, `r->s`, `e->t`, `f->ß` and more behaving identically, so `--confusions`
decides how wide to cast the net rather than the code doing it.

Method — the same lexicon trick as the hyphens
----------------------------------------------
`validate_xml_hyphens.py` needs no dictionary because it builds its lexicon from
a position the error cannot reach: the *interior* tokens of a line, neither the
first nor the last, cannot have been damaged by a line break. A line-end
character misread has exactly the same property, so the same reference works
here — and it is built from the very same `read_paragraphs`/`strip_token` pair,
so both scripts key their tables identically and Antiqua and Fraktur land under
one key.

For each line-final token, every one-character substitution of its final letter
is scored against that lexicon:

    observed    how often the spelling as transcribed occurs inside a line
    candidate   how often each substituted spelling does

A candidate wins only when it is well attested (`--min-count`, default 20) and
the observed spelling is essentially absent (`--ratio`, default 50 — the
observed form must be at most 1/50 of the candidate). Anything less clear-cut is
reported but never applied.

Verdicts:
    OK        the spelling as transcribed is a word this corpus uses
    MISREAD   exactly one candidate dominates and the observed form is absent
    REVIEW    two or more candidates compete, or the observed form is attested
              after all, so only the sentence can decide (see --export-review,
              then resolve_line_end_llm.py)

Measured on the 787-issue corpus (880,806 judged line ends):

    OK        877,587   99.63%
    MISREAD     2,350    0.27%   (1,371 distinct tokens)
    REVIEW        869    0.10%   (  318 distinct tokens)

Why a MISREAD is not simply applied
-----------------------------------
On *this* corpus the confident verdicts include ones that are plainly wrong, and
they are wrong in two ways worth naming, because neither can be fixed by moving
a threshold:

    dorf -> dort      `-dorf` is a place-name ending. Reiss, Bahr, Mariae the
    Reiss -> Reise    same: this newspaper is largely surnames and place names,
    Mariae -> Marian  and a rare real name looks exactly like a misread word.

    Scht -> Schw      The token is *truncated*, not substituted — the OCR
    Geno -> Genf      dropped characters rather than misreading one. No
                      single-character substitution can repair it, and
                      proposing one makes the text worse than leaving it alone.
                      That class belongs to a different detector:
                      validate_line_end_truncations.py.

So the confident verdicts go into the correction store as `source=rule`, which
`resolve_line_end_llm.py` can overturn and which `correct_xml_ocr.py` applies
only at `confidence=high`. **Run the resolver over them** (`--export-review
--review-misread`) before correcting anything: an LLM reading the sentence
settles exactly the two failure modes above, and its verdict outranks this one.

What the statistics cannot see at all
-------------------------------------
When the misread spelling is *itself* a common word, no frequency table can
detect it. "den" and "dem" misread as "der" is exactly this: `der` is one of the
commonest words in German, so a line ending in `der` looks perfectly normal
however it got there. Only the syntax of the sentence settles it.

Those cases are therefore **not** flagged by default — doing so would queue
every line-final `der`, `die`, `und` in the corpus. `--ambiguous` turns them into
REVIEW rows for LLM adjudication, which is a much larger and much lower yield
job than the default run; start without it.

Scope
-----
Only lines that really end where they appear to. A line ending in a hyphen holds
a *fragment* of a word continuing on the next line, whose final character
legitimately belongs mid-word ("Ver¬"), so it is skipped; those breaks are the
hyphen scripts' business. Unlike the hyphen scripts, the last line of a
paragraph *is* judged — a line end needs no following line — so paragraphs are
read with `min_lines=1`.

Regions outside `validate_xml_hyphens.CORRECTED_SUBTYPES` are excluded by
default (`--skip-subtypes`), from the lexicon as well as from the scan.
`ad`/`ad-frame` hold real text whose OCR is the noisiest in the corpus by a
factor of 16 (docs/ocr-word-correction.md); `image`, `mode` and the
`separator*` classes hold no text at all, and their stray single characters
("K", "2", "8") are exactly the shape this scan mistakes for a misread word.
`--skip-subtypes ''` reopens them.

Output
------
A verdict CSV in `output/` under the run's date, and two optional exports:

    --export-corrections   writes the confident MISREAD verdicts into the OCR
                           correction store as wildcard rows ("Wier, *, Wien"),
                           ready for correct_xml_ocr.py
    --export-review        writes the contested ones as JSONL with sentence
                           context, for resolve_line_end_llm.py

Usage:
    python validate_line_end_chars.py
    python validate_line_end_chars.py --max-files 50
    python validate_line_end_chars.py --confusions rnm
    python validate_line_end_chars.py --export-review linechar_review.jsonl \
        --review-misread
    python validate_line_end_chars.py --export-corrections

Flags:
    --xml-folder P    Path to XML files folder (default: PATHS['xml']).
    --max-files N     Maximum number of files to process.
    --skip-subtypes S Comma-separated region subtypes to leave out of the
                      lexicon and the scan (default: ad,ad-frame; pass '' for
                      none).
    --confusions S    Which final letters may be substituted, and by what.
                      'all' (default) tries every letter; a bare letter set
                      like 'rnm' restricts both sides to those letters;
                      'r:nm,l:r' spells out observed:candidates pairs.
    --min-count N     A candidate needs this many line-interior occurrences
                      (default: 20).
    --ratio R         The observed spelling must be at most 1/R of the
                      candidate's frequency (default: 50).
    --min-length N    Shortest line-final token to judge (default: 4). Short
                      words are mostly function words where the substituted
                      form is another real word.
    --max-observed N  A MISREAD verdict needs the observed spelling to occur at
                      most N times inside a line (default: 1). This is the
                      guard against German inflection: raising it starts
                      "correcting" datives like "am späten" and proper names.
                      Above N the token becomes a REVIEW.
    --ambiguous       Also queue tokens whose observed spelling is itself a
                      common word (der/den/dem). Large, low-yield: REVIEW only.
    --export-corrections
                      Write confident MISREAD verdicts into the correction
                      store as wildcard rows. Read the precision note above
                      first.
    --corrections P   Correction store to write (default:
                      PATHS['csv']/ocr_corrections.csv).
    --export-review NAME
                      Write contested cases as JSONL with sentence context.
    --review-misread  Also put the confident MISREAD verdicts in that export,
                      so resolve_line_end_llm.py checks them. Worth doing on
                      this corpus — see the precision note above.
    --output NAME     Verdict CSV filename (default: line_end_chars.csv).
    --quiet           Suppress progress messages.

Author: Christian Lendl
Created: 2026-08-14
"""

import argparse
import csv
import datetime
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS
from ocr_corrections import WILDCARD, Correction, merge
from validate_xml_hyphens import (
    HYPHEN_CHARS,
    SKIPPED_SUBTYPES,
    WORD_RE,
    in_output,
    read_paragraphs,
    strip_token,
)

try:
    from utils.get_transkribus_link import get_transkribus_link
except ImportError:  # pragma: no cover - link is a convenience only
    get_transkribus_link = None

DEFAULT_CORRECTIONS = PATHS['csv'] / 'ocr_corrections.csv'

# Every letter the transcription uses, for --confusions all
ALPHABET = 'abcdefghijklmnopqrstuvwxyzäöüß'

# A token this common in line-interior position is a word in its own right, so a
# substitution cannot be told from a correct reading by frequency alone.
AMBIGUOUS_AT = 20


@dataclass
class Candidate:
    """One competing spelling of a line-final token."""
    word: str
    count: int


@dataclass
class TokenEvidence:
    """Accumulated evidence for one distinct line-final token."""
    word: str
    seen: int = 0                 # times seen at a line end
    observed_count: int = 0       # its own line-interior frequency
    candidates: List[Candidate] = field(default_factory=list)
    verdict: str = ''
    reason: str = ''
    contexts: List[str] = field(default_factory=list)
    examples: List[Tuple[str, int, str, str]] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def runner_up(self) -> Optional[Candidate]:
        return self.candidates[1] if len(self.candidates) > 1 else None


def parse_confusions(spec: str) -> Dict[str, str]:
    """
    Turn a --confusions argument into {observed_char: candidate_chars}.

    Three spellings, in rising order of specificity:
        'all'      every letter may become every other
        'rnm'      only these letters, and only among themselves
        'r:nm,l:r' exactly these substitutions
    """
    spec = spec.strip()
    if spec == 'all':
        return {ch: ALPHABET.replace(ch, '') for ch in ALPHABET}
    if ':' in spec:
        mapping: Dict[str, str] = {}
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            observed, _, candidates = part.partition(':')
            observed, candidates = observed.strip(), candidates.strip()
            if len(observed) != 1 or not candidates:
                raise argparse.ArgumentTypeError(
                    f"bad confusion {part!r}: expected 'x:abc'")
            mapping[observed] = candidates.replace(observed, '')
        return mapping
    letters = ''.join(dict.fromkeys(spec))
    if not letters:
        raise argparse.ArgumentTypeError('--confusions may not be empty')
    return {ch: letters.replace(ch, '') for ch in letters}


def collect(xml_folder: Path, max_files: Optional[int],
            skip_subtypes: FrozenSet[str], quiet: bool
            ) -> Tuple[Counter, Dict[str, TokenEvidence]]:
    """
    Read the corpus once, building the interior lexicon and the line-end tally.

    Returns:
        (interior unigram counter, {token: TokenEvidence})
    """
    files = sorted(xml_folder.glob('*.xml'))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No XML files found in {xml_folder}")

    if not quiet:
        print(f"Reading {len(files)} XML files from {xml_folder} ...")

    interior: Counter = Counter()
    endings: Dict[str, TokenEvidence] = {}

    for n, path in enumerate(files, 1):
        text = path.read_text(encoding='utf-8')
        # min_lines=1: a line end needs no following line, so a one-line
        # paragraph is judged like any other.
        paragraphs = read_paragraphs(text, min_lines=1,
                                     skip_subtypes=skip_subtypes)
        for region_id, page, subtype, lines in paragraphs:
            for line in lines:
                tokens = line.split()
                if not tokens:
                    continue
                for token in tokens[1:-1]:
                    word = strip_token(token)
                    if WORD_RE.fullmatch(word):
                        interior[word] += 1

                # A hyphenated line ends on a fragment, not a word
                if line.endswith(HYPHEN_CHARS):
                    continue
                word = strip_token(tokens[-1])
                if not WORD_RE.fullmatch(word):
                    continue
                ev = endings.get(word)
                if ev is None:
                    ev = endings[word] = TokenEvidence(word=word)
                ev.seen += 1
                if len(ev.contexts) < 3:
                    ev.contexts.append(line)
                    ev.examples.append((path.name, page, region_id, subtype))
        if not quiet and n % 100 == 0:
            print(f"  {n}/{len(files)} files")

    if not interior:
        raise SystemExit(
            f"Read {len(files)} XML files from {xml_folder} but found no "
            f"line-interior tokens. Either every region was skipped by "
            f"--skip-subtypes, or the reader in validate_xml_hyphens.py no "
            f"longer matches this corpus.")

    return interior, endings


def judge(ev: TokenEvidence, interior: Counter, confusions: Dict[str, str],
          min_count: int, ratio: float, ambiguous: bool,
          max_observed: int) -> str:
    """
    Score the substitutions of this token's final character.

    The observed spelling's own line-interior frequency is the thing to beat: a
    word the corpus uses inside lines is a word, whatever a variant's count
    says. Only when the observed form is essentially absent does a well attested
    variant become the better reading.

    `max_observed` is what keeps German grammar out of the results, and it
    matters far more than the ratio does. The corpus writes "später" far more
    often than "späten", so a frequency test alone calls a line-final "späten" a
    misread `r` — but "am späten Abend" is simply correct, and so are "von echt
    deutschem", "nach altem". A token the corpus uses at all therefore goes to
    REVIEW rather than being corrected, and only the sentence can settle it.
    """
    ev.observed_count = interior[ev.word]
    final = ev.word[-1].lower()
    for char in confusions.get(final, ''):
        variant = ev.word[:-1] + char
        count = interior[variant]
        if count >= min_count:
            ev.candidates.append(Candidate(variant, count))
    ev.candidates.sort(key=lambda c: -c.count)

    best = ev.best
    if best is None:
        ev.reason = 'no-attested-variant'
        return 'OK'

    # The observed spelling is a word this corpus uses. Frequency cannot tell a
    # correct reading from a misreading here - only the sentence can.
    if ev.observed_count >= AMBIGUOUS_AT:
        ev.reason = 'observed-is-a-word'
        return 'REVIEW' if ambiguous else 'OK'

    if ev.observed_count * ratio > best.count:
        ev.reason = 'observed-attested'
        return 'OK'

    runner_up = ev.runner_up
    if runner_up is not None and runner_up.count * 2 > best.count:
        # "Wien" and "Wiem" would both explain "Wier"; the corpus does not
        # separate them, so the sentence has to.
        ev.reason = 'competing-variants'
        return 'REVIEW'

    ev.reason = f'{final}->{best.word[-1]}'
    if ev.observed_count > max_observed:
        # A word the corpus uses, even rarely. Almost always a dative ending or
        # a proper name rather than a misreading - see the docstring.
        ev.reason += f'?  observed {ev.observed_count}x inside a line'
        return 'REVIEW'
    return 'MISREAD'


def export_to_csv(rows: List[TokenEvidence], output_path: Path) -> None:
    """Write the verdict table, worst offenders first."""
    fieldnames = ['verdict', 'reason', 'observed', 'suggested',
                  'seen_at_line_end', 'observed_freq', 'suggested_freq',
                  'runner_up', 'runner_up_freq', 'example_context',
                  'example_file', 'example_page', 'region_subtype',
                  'transkribus_link']
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ev in rows:
            best, runner_up = ev.best, ev.runner_up
            filename, page, _region, subtype = (
                ev.examples[0] if ev.examples else ('', 0, '', ''))
            link = ''
            if get_transkribus_link and filename:
                try:
                    link = get_transkribus_link(filename, page=page)
                except Exception:
                    link = ''
            writer.writerow({
                'verdict': ev.verdict,
                'reason': ev.reason,
                'observed': ev.word,
                'suggested': best.word if best else '',
                'seen_at_line_end': ev.seen,
                'observed_freq': ev.observed_count,
                'suggested_freq': best.count if best else 0,
                'runner_up': runner_up.word if runner_up else '',
                'runner_up_freq': runner_up.count if runner_up else 0,
                'example_context': ev.contexts[0] if ev.contexts else '',
                'example_file': filename,
                'example_page': page,
                'region_subtype': subtype,
                'transkribus_link': link,
            })
    print(f"\nVerdict table saved to: {output_path}  ({len(rows):,} rows)")


def export_review_jsonl(rows: List[TokenEvidence], output_path: Path) -> None:
    """
    Write tokens as JSONL for LLM adjudication.

    Each record carries the statistical verdict, so the model is doing one of
    two jobs depending on which: deciding a REVIEW the corpus could not settle,
    or checking a MISREAD it settled but might have got wrong. On this corpus
    the second is not a formality — see the precision note in the module
    docstring.
    """
    with open(output_path, 'w', encoding='utf-8') as fh:
        for ev in rows:
            fh.write(json.dumps({
                'observed': ev.word,
                'candidates': [c.word for c in ev.candidates],
                'statistical_verdict': ev.verdict,
                'suggested': ev.best.word if ev.best else '',
                'seen_at_line_end': ev.seen,
                'evidence': {
                    'observed_freq': ev.observed_count,
                    **{c.word: c.count for c in ev.candidates},
                },
                'reason': ev.reason,
                'contexts': ev.contexts,
            }, ensure_ascii=False) + '\n')
    print(f"Review set saved to: {output_path}  ({len(rows):,} tokens)")


def export_corrections(rows: List[TokenEvidence], path: Path,
                       model: str) -> None:
    """
    Write the confident verdicts into the OCR correction store.

    As wildcard rows (`right` = `*`): the error is in the word itself and does
    not depend on what follows it. `source=rule` keeps them below any LLM or
    hand-made row, so `resolve_line_end_llm.py` can overturn one and a manual
    `approved=no` retires it for good.
    """
    corrections = [
        Correction(left=ev.word, right=WILDCARD,
                   left_correct=ev.best.word, right_correct=WILDCARD,
                   confidence='high', source='rule', model=model,
                   note=(f"line-end {ev.reason}; '{ev.word}' unattested "
                         f"inside a line ({ev.observed_count}x), "
                         f"'{ev.best.word}' {ev.best.count}x"))
        for ev in rows if ev.best is not None
    ]
    merge(path, corrections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Detect misread characters at the end of a line.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--skip-subtypes', default=','.join(SKIPPED_SUBTYPES),
                        help='Comma-separated region subtypes to leave out of '
                             'the lexicon and the scan (default: '
                             f"{','.join(SKIPPED_SUBTYPES)})")
    parser.add_argument('--confusions', default='all',
                        help="'all', a letter set like 'rnm', or explicit "
                             "pairs like 'r:nm,l:r' (default: all)")
    parser.add_argument('--min-count', type=int, default=20)
    parser.add_argument('--ratio', type=float, default=50.0)
    parser.add_argument('--min-length', type=int, default=4)
    parser.add_argument('--max-observed', type=int, default=1,
                        help='A MISREAD verdict needs the observed spelling '
                             'to occur at most this often inside a line '
                             '(default: 1); above it the token goes to REVIEW')
    parser.add_argument('--ambiguous', action='store_true',
                        help='Also queue tokens whose observed spelling is '
                             'itself a common word (large, low yield)')
    parser.add_argument('--export-corrections', action='store_true')
    parser.add_argument('--corrections', type=Path,
                        default=DEFAULT_CORRECTIONS)
    parser.add_argument('--export-review', type=Path)
    parser.add_argument('--review-misread', action='store_true',
                        help='Include the confident MISREAD verdicts in the '
                             'review export, so the LLM checks them too')
    parser.add_argument('--output', default='line_end_chars.csv')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    skip_subtypes = frozenset(
        s.strip() for s in args.skip_subtypes.split(',') if s.strip())
    try:
        confusions = parse_confusions(args.confusions)
    except argparse.ArgumentTypeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    interior, endings = collect(args.xml_folder, args.max_files, skip_subtypes,
                                args.quiet)
    if not args.quiet:
        if skip_subtypes:
            print(f"Excluded region subtypes: "
                  f"{', '.join(sorted(skip_subtypes))}")
        print(f"  {len(interior):,} interior word types / "
              f"{sum(interior.values()):,} tokens")
        print(f"  {len(endings):,} distinct line-final tokens")
        print("Judging line ends ...")

    stats: Counter = Counter()
    for ev in endings.values():
        # Short words are mostly function words whose one-letter variants are
        # other real words; judging them by frequency invites false positives.
        if len(ev.word) < args.min_length:
            ev.verdict, ev.reason = 'OK', 'too-short'
        else:
            ev.verdict = judge(ev, interior, confusions, args.min_count,
                               args.ratio, args.ambiguous, args.max_observed)
        stats[ev.verdict] += ev.seen

    total = sum(stats.values())
    misread = [ev for ev in endings.values() if ev.verdict == 'MISREAD']
    review = [ev for ev in endings.values() if ev.verdict == 'REVIEW']
    misread.sort(key=lambda e: -e.seen)
    review.sort(key=lambda e: -e.seen)

    print("\n" + "=" * 62)
    print("Line-end character verdicts (by occurrence)")
    print("=" * 62)
    for verdict in ('OK', 'MISREAD', 'REVIEW'):
        count = stats[verdict]
        print(f"  {verdict:10s} {count:9,}  ({count / max(total, 1):6.2%})")
    print(f"  {'TOTAL':10s} {total:9,}")
    print(f"\n  distinct tokens: {len(misread):,} misread, "
          f"{len(review):,} to review")

    by_sub = Counter(ev.reason for ev in misread)
    if by_sub:
        print("\n  substitutions found (distinct tokens):")
        for sub, count in by_sub.most_common(12):
            print(f"    {sub:10s} {count:6,}")

    date_prefix = datetime.date.today().strftime('%Y%m%d')
    output_path = PATHS['output'] / f"{date_prefix}_{args.output}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_csv(misread + review, output_path)

    if args.export_review:
        queue = (review + misread) if args.review_misread else review
        queue.sort(key=lambda e: -e.seen)
        export_review_jsonl(queue, in_output(args.export_review, date_prefix))
        print("Next: python resolve_line_end_llm.py "
              f"{in_output(args.export_review, date_prefix)}")

    if args.export_corrections:
        export_corrections(misread, args.corrections,
                           f'line-end-stats/{args.confusions}')
        if not args.review_misread:
            # Saying so here rather than only in the docstring: this is the
            # step at which a surname or a truncated token becomes a pending
            # rewrite of data/xml/.
            print("\n[note] These are statistical verdicts only. On this "
                  "corpus they include surnames ('Reiss') and truncated "
                  "tokens ('Scht').\n       Run resolve_line_end_llm.py over "
                  "them before correct_xml_ocr.py.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
