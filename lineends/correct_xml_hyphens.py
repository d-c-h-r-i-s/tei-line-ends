#!/usr/bin/env python3
"""
XML Line-End Hyphen Correction
==============================

Applies the verdicts of `validate_xml_hyphens.py` to the Transkribus XML files
in `data/xml/` themselves, so the corrected transcription — not just the
imported database text — is what gets published to other researchers.

The correction is a single character at the end of a `<l>` element, which is
exactly where the error sits. That makes the XML the natural place to fix it:

    join    the word continues        line must end with `¬`
    hyphen  real compound hyphen      line must end with `-`
    period  the `¬` was a full stop   line must end with `.`
    comma   the `¬` was a comma       line must end with `,`
    split   two separate words        line must not end with a hyphen at all

Only line breaks whose verdict is an actual error are touched — a line already
carrying the right character is left byte-identical. Attribute quoting and the
XML declaration are restored to the single-quote style Transkribus emits, so
the diff stays limited to the lines that really changed.

Antiqua and Fraktur
-------------------
The table above is the Antiqua notation. The older issues set in Fraktur use
the Doppelbindestrich `=` instead, and use it for both jobs — a continued word
and a real compound hyphen look alike there — so in such a file `join` and
`hyphen` both write `=`, and `period`, `comma` and `split` are unaffected.
Which notation a file is in is read off the file itself, by majority over its
line ends, so a corpus holding both is corrected correctly without being sorted
first; `--notation antiqua|fraktur` forces it for the whole run. The `notation`
column of the per-change report records what was used for each line.

Nothing is lost by Fraktur collapsing the two forms: the join/hyphen decision
lives in the store, and it is `hyphen_utils.py` — not the XML — that
needs it, since it normalises `¬`, `-` and `=` alike when joining the lines.

Where the verdicts come from
----------------------------
The same analysis `validate_xml_hyphens.py` performs, plus the accumulated
decision store (`data/csv/hyphen_decisions.csv`). By default only verdicts the
Anything the statistics could not settle stays a `REVIEW` and is left alone
until it appears in the store, which is what `resolve_hyphens_llm.py` writes
back into. A stored answer always wins over the statistics; storing the form
`skip` is how a pair is declared unresolvable and kept untouched for good.

Paragraph-final hyphens are out of scope
----------------------------------------
A hyphen on the *last* line of a paragraph is never counted, flagged or
corrected here. Only breaks between two lines of the same paragraph are
examined, so those lines are skipped by construction (3,710 of them across the
corpus, none touched). Such a paragraph is normally continued in the next
column or on the next page; it is joined later by
the paragraph-merge step (downstream, not shipped), and whatever hyphens remain after that join
are reported by `utils/check_remaining_hyphens.py` for manual correction.

Safety
------
`--dry-run` reports what would change without writing. `--backup` copies each
file to `<name>.xml.bak` before the first modification. Since the corpus is
under version control, a plain `git diff` is the other way to inspect the
result.

Usage:
    python correct_xml_hyphens.py --dry-run
    python correct_xml_hyphens.py --backup
    python correct_xml_hyphens.py --min-count 2

Flags:
    --xml-folder P    Path to XML files folder (default: PATHS['xml']).
    --decisions P     Decision store to consult (default:
                      PATHS['csv']/hyphen_decisions.csv).
    --max-files N     Maximum number of files to process.
    --min-count N     Only apply verdicts backed by at least N corpus
                      occurrences of that word pair (default: 1).
    --ratio R         Evidence ratio required for a verdict (default: 3.0).
    --notation N      Line-break notation to write: 'antiqua' (¬ / -),
                      'fraktur' (= / =), or 'auto' (default) to read it off
                      each file.
    --normalize-hyphens
                      Also rewrite a line-end '-' to the file's continuation
                      mark where the word continues. Notation only, not an
                      error fix, so it is off by default to keep the published
                      diff meaningful.
    --skip-subtypes S Comma-separated TextRegion subtypes to leave untouched.
                      Advertisements carry the noisiest OCR and account for
                      about a tenth of all corrections: --skip-subtypes ad,ad-frame
    --backup          Write <name>.xml.bak before modifying a file.
    --dry-run         Report changes without writing files.
    --report NAME     Write a per-change CSV to output/ (default:
                      hyphen_corrections.csv).
    --quiet           Suppress progress messages.

Author: Christian Lendl
Created: 2026-07-22
Last Modified: 2026-07-29
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
import break_occurrences
from validate_xml_hyphens import SKIPPED_SUBTYPES, UNJUDGED  # noqa: E402
from hyphen_decisions import (
    NOTATION_NAMES,
    load as load_decisions,
    line_end_for,
)
from validate_xml_hyphens import (
    HYPHEN_CHARS,
    MIN_TOKEN_LEN,
    analyse,
    detect_notation,
    strip_token,
)
from xml_io import write_with_single_quotes

TEI_NS = 'http://www.tei-c.org/ns/1.0'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NS = {'tei': TEI_NS}

# Verdicts that describe a real error and therefore justify rewriting a line
CORRECTABLE = ('FALSE_POSITIVE', 'FALSE_NEGATIVE', 'PERIOD', 'COMMA')


def build_plan(pairs, min_count: int) -> Dict[Tuple[str, str], str]:
    """
    Reduce the analysis to a lookup of (left, right) -> target form.

    Only pairs whose verdict is an actual error end up here; everything the
    OCR already got right is deliberately absent so those lines stay untouched.

    Pairs settled in the decision store arrive already carrying a definite
    verdict rather than REVIEW, so an LLM or manual answer is always applied.
    A `skip` decision is the way to say "leave this one alone".
    """
    plan: Dict[Tuple[str, str], str] = {}
    for ev in pairs.values():
        # `context` is excluded here and not merely skipped downstream: the
        # pair has no corpus-wide form, and the plan it would contribute is
        # keyed by pair alone, which is precisely the resolution this form
        # rejects. Its occurrences are corrected from `hyphen_occurrences.csv`.
        if ev.form in (None, 'skip', 'context') or ev.affected < min_count:
            continue
        if ev.verdict in CORRECTABLE:
            plan[(ev.left, ev.right)] = ev.form
    return plan


def correct_file(xml_path: Path, plan, backup: bool, dry_run: bool,
                 normalize: bool = False,
                 skip_subtypes: Set[str] = frozenset(),
                 notation: Optional[str] = None,
                 occurrences: Optional[Dict[str, object]] = None,
                 stale: Optional[List[dict]] = None
                 ) -> Tuple[int, List[dict], str]:
    """
    Apply the plan to one XML file.

    `notation` forces the line-break mark to write; left at None it is read off
    this very file, so an Antiqua issue keeps its `¬`/`-` and a Fraktur one its
    `=` no matter what the rest of the corpus is set in.

    `occurrences` is this file's slice of `hyphen_occurrences.csv`, keyed by the
    `facs` id of the line the break ends. It is consulted for the pairs the
    plan deliberately does not carry - those settled `context`, which have no
    corpus-wide form - and it *outranks* the plan for every break it covers,
    because a verdict about this break was made with this sentence in front of
    it and a verdict about the pair was not.

    A row whose pair no longer matches the line is recorded in `stale` and
    never applied. The `facs` id contains a Transkribus region id, which need
    not survive a re-export, and a key that has come to point at a different
    break is the one failure this store cannot notice by itself.

    Returns:
        (number of lines changed, list of change records, notation used)
    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    if notation is None:
        # Read off every line, not just the ones this run may correct: the
        # majority over the whole file is what makes the verdict stable.
        notation = detect_notation(
            ''.join(el.itertext())
            for el in root.xpath('//tei:l', namespaces=NS))

    changes: List[dict] = []

    # Region id -> subtype, so whole categories (advertisements above all) can
    # be held back from correction.
    subtypes = {}
    if skip_subtypes:
        for zone in root.xpath("//tei:zone[@rendition='TextRegion']",
                               namespaces=NS):
            zone_id = zone.get(f'{{{XML_NS}}}id')
            if zone_id:
                subtypes[zone_id] = zone.get('subtype', '')

    for paragraph in root.xpath('//tei:p', namespaces=NS):
        if skip_subtypes:
            region = (paragraph.get('facs') or '').lstrip('#')
            if subtypes.get(region, '') in skip_subtypes:
                continue
        lines = paragraph.xpath('.//tei:l', namespaces=NS)
        # Lines carrying markup are left alone: rewriting them would mean
        # deciding which text node the hyphen belongs to.
        texts = [(el, el.text) for el in lines
                 if len(el) == 0 and (el.text or '').strip()]

        for i in range(len(texts) - 1):
            element, text = texts[i]
            following = texts[i + 1][1]
            current = text.strip()
            if not following.strip().split() or not current.split():
                continue

            had_hyphen = current.endswith(HYPHEN_CHARS)
            body = current[:-1].rstrip() if had_hyphen else current
            if not body.split():
                continue

            left = strip_token(body.split()[-1])
            right = strip_token(following.strip().split()[0])
            if len(left) < MIN_TOKEN_LEN or len(right) < MIN_TOKEN_LEN:
                continue

            form = plan.get((left, right))
            occurrence = (occurrences or {}).get(
                (element.get('facs') or '').lstrip('#'))
            if occurrence is not None:
                if occurrence.matches(left, right):
                    form = occurrence.form
                elif stale is not None:
                    stale.append({'file': xml_path.name,
                                  'line_facs': occurrence.line_facs,
                                  'expected': f"{occurrence.left}|"
                                              f"{occurrence.right}",
                                  'found': f"{left}|{right}"})
            if form is None:
                continue
            ending = line_end_for(form, notation)
            if ending is None:          # 'skip' or 'context'
                continue

            if had_hyphen:
                # Swapping one hyphen character for another ("-" -> "¬", or a
                # stray "¬" in a Fraktur issue -> "=") only normalises
                # notation, it does not fix a transcription error, so it stays
                # out of the diff unless explicitly asked for.
                if ending in HYPHEN_CHARS and not normalize:
                    continue
                # Replace the hyphen the OCR produced with the right character
                corrected = body + ending
            elif form == 'join':
                # The only correction that adds something to an unmarked break
                corrected = body + ending
            else:
                # 'split' needs no hyphen and there is none; 'period'/'hyphen'
                # describe a character that was misread, so with no hyphen
                # present there is nothing here to correct.
                continue

            # Never double a full stop the line already ends with
            if ending == '.' and body.endswith('.'):
                corrected = body
            if corrected == current:
                continue

            element.text = corrected
            changes.append({
                'file': xml_path.name,
                'left': left,
                'right': right,
                'form': form,
                'notation': notation,
                'before': current,
                'after': corrected,
            })

    if changes and not dry_run:
        if backup:
            backup_path = xml_path.with_suffix('.xml.bak')
            if not backup_path.exists():
                shutil.copy2(xml_path, backup_path)
        write_with_single_quotes(tree, xml_path)

    return len(changes), changes, notation


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Apply line-end hyphen corrections to the TEI-XML files.')
    parser.add_argument('--xml-folder', type=Path, default=PATHS['xml'])
    parser.add_argument('--decisions', type=Path,
                        default=PATHS['csv'] / 'hyphen_decisions.csv')
    parser.add_argument('--occurrences', type=Path,
                        default=PATHS['csv'] / 'hyphen_occurrences.csv',
                        help='Per-occurrence verdicts for the pairs settled '
                             '`context`; missing file means none are applied')
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=3.0)
    parser.add_argument('--normalize-hyphens', action='store_true',
                        help="Also rewrite a line-end '-' to the file's "
                             'continuation mark where the word continues '
                             '(notation only)')
    parser.add_argument('--notation', default='auto',
                        choices=['auto', *sorted(NOTATION_NAMES)],
                        help="Line-break notation to write: 'antiqua' (¬/-), "
                             "'fraktur' (=/=), or 'auto' (default) to read it "
                             'off each file')
    parser.add_argument('--skip-subtypes',
                        default=','.join(SKIPPED_SUBTYPES),
                        help='Comma-separated TextRegion subtypes to leave '
                             "untouched, e.g. 'ad,ad-frame'")
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', default='hyphen_corrections.csv')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    skip_subtypes = {s.strip() for s in args.skip_subtypes.split(',')
                     if s.strip()}

    decisions = load_decisions(args.decisions)
    if not args.quiet:
        print(f"Decision store: {len(decisions):,} pairs settled "
              f"({args.decisions})")

    # The analysis always runs over the whole corpus - the lexicon must not
    # shrink just because only a few files are being rewritten.
    pairs, stats = analyse(args.xml_folder, None, args.ratio, args.quiet,
                       decisions)
    plan = build_plan(pairs, args.min_count)

    if not args.quiet:
        forms = Counter(plan.values())
        print(f"\nCorrection plan: {len(plan):,} word pairs "
              f"({dict(forms)})")

    # The per-occurrence verdicts, for the pairs the plan has no form for.
    # Grouped by file once rather than indexed per line: the store only grows,
    # and searching it for every break would make the pass quadratic in it.
    occurrences = break_occurrences.load(args.occurrences)
    per_file = break_occurrences.by_file(occurrences)
    stale: List[dict] = []
    if occurrences and not args.quiet:
        pairs = len({o.pair for o in occurrences.values()})
        print(f"Occurrence store: {len(occurrences):,} breaks over {pairs:,} "
              f"pairs ({args.occurrences})")

    files = sorted(args.xml_folder.glob('*.xml'))
    if args.max_files:
        files = files[:args.max_files]

    # 'auto' leaves the notation to each file, which is the only setting that
    # copes with a corpus holding both typesettings.
    forced_notation = NOTATION_NAMES.get(args.notation)

    total_changes = 0
    files_changed = 0
    all_changes: List[dict] = []
    notations: Counter = Counter()
    for n, xml_path in enumerate(files, 1):
        count, changes, notation = correct_file(
            xml_path, plan, args.backup, args.dry_run, args.normalize_hyphens,
            skip_subtypes, forced_notation,
            per_file.get(xml_path.name), stale)
        notations[notation] += 1
        if count:
            files_changed += 1
            total_changes += count
            all_changes.extend(changes)
        if not args.quiet and n % 100 == 0:
            print(f"  {n}/{len(files)} files, {total_changes:,} lines changed")

    unjudged = stats.get(UNJUDGED, 0)
    if unjudged:
        # Reported here and not only by the validator, because this is the
        # script that decides what the corpus ends up saying: "15,749 lines
        # corrected" reads as a complete job unless the breaks nobody could
        # look at are counted next to it.
        print(f"\n  {unjudged:,} break marks on a paragraph's last line were "
              f"left alone — their\n  continuation is in the next column or "
              f"page, unreachable until\n  the paragraph-merge step (downstream, not shipped) writes the "
              f"`@next` chain.")

    if stale:
        # Loud, and not fatal. A handful of these means a re-export moved some
        # ids; a storeful means the key scheme has stopped working and no
        # occurrence verdict in the file is trustworthy.
        print(f"\n  {len(stale):,} occurrence rows point at a break that is no "
              f"longer there and were skipped:")
        for row in stale[:5]:
            print(f"    {row['file']} {row['line_facs']}: expected "
                  f"{row['expected']}, found {row['found']}")
        if len(stale) > 5:
            print(f"    ... and {len(stale) - 5:,} more")

    by_form = Counter(c['form'] for c in all_changes)
    print("\n" + "=" * 62)
    print("DRY RUN - nothing written" if args.dry_run else "XML files updated")
    print("=" * 62)
    print(f"  files touched   {files_changed:,} / {len(files):,}")
    print(f"  lines corrected {total_changes:,}")
    for form, count in by_form.most_common():
        print(f"    {form:8s} {count:6,}")
    mix = ', '.join(f"{char} {count:,}" for char, count in
                    notations.most_common())
    print(f"  notation        {mix}"
          f"{'' if forced_notation else '  (detected per file)'}")

    if all_changes:
        date_prefix = datetime.date.today().strftime('%Y%m%d')
        report_path = PATHS['output'] / f"{date_prefix}_{args.report}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(
                fh, fieldnames=['file', 'left', 'right', 'form', 'notation',
                                'before', 'after'])
            writer.writeheader()
            writer.writerows(all_changes)
        print(f"\nPer-change report: {report_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
