#!/usr/bin/env python3
"""
Extract OCR Word Corrections from the Hyphen Decision Notes
===========================================================

While `resolve_hyphens_llm.py` was deciding how a line-break word pair should be
written, the model kept noticing something it was never asked about: that one of
the two words is itself misread. Those observations sit in the `note` column of
`data/csv/hyphen_decisions.csv`, in free German prose —

    "Es handelt sich um das Wort 'Atmungsorgane', bei dem die OCR-Erkennung
     Buchstaben (u, O) ausgelassen hat."
    "Es handelt sich um zwei getrennte Wörter ('Antiquitäten' und 'aus'),
     wobei das erste Wort durch die OCR vermutlich verstümmelt wurde."

— which is far too varied to unpick with a regex. So the notes go back to a
model, one per pair, and come back as structured rows for `ocr_corrections.py`.

Of the 1,458 settled pairs in the store, 109 carry a note mentioning an OCR
error, so this is a minutes-long run rather than an overnight one. The yield
grows every time `resolve_hyphens_llm.py` drains more of the hyphen review
queue, which is the reason to keep it around: the notes are otherwise dead text.

What is asked
-------------
Not "is this word right?" but "what does this note say about these two words?".
The model is reading a note, not the newspaper, and it must not invent a
correction the note does not claim. Both corrected forms are asked for at once,
because a fix can land on either side — and because a word broken across the
break has to stay broken across it:

    Athmungs + rgane  ->  Athmungs + organe   ("Athmungsorgane")

Historical orthography is not an OCR error. "Cassa", "Theil", "Bureau",
"Athmungsorgane" are how this newspaper spells things, and the prompt says so at
length: modernising them would quietly rewrite the source.

Why the answers are not simply applied
--------------------------------------
The extraction is faithful to the note; the note itself is a guess made from
text context by a model that never saw the scan, and it was made while answering
a different question. Every row therefore carries a `confidence`, and
`correct_xml_ocr.py` applies only what clears the bar or what a human approved by
hand. See `ocr_corrections.py`.

Resuming, sharding, failure
---------------------------
Pairs already in the correction store are skipped before any request is made, so
a run can be stopped and restarted freely, and `--shard k/n` splits the queue
round-robin over several machines exactly as the hyphen review queue is split. A
server failure stops the run rather than recording a verdict nobody gave; the
answers already given are kept. The end of a run is pushed through
`utils/notify.py`.

Usage:
    python extract_ocr_corrections.py --dry-run
    python extract_ocr_corrections.py
    python extract_ocr_corrections.py \
        --review output/20260729_review.jsonl
    python extract_ocr_corrections.py --shard 1/4 \
        --corrections data/csv/ocr_corrections_1of4.csv

Flags:
    --decisions P      Hyphen decision store to read notes from (default:
                       PATHS['csv']/hyphen_decisions.csv).
    --corrections P    Correction store to write (default:
                       PATHS['csv']/ocr_corrections.csv).
    --review P         Review JSONL from validate_xml_hyphens.py
                       --export-review. Optional but worth passing: it supplies
                       the sentence context the note was written about, which is
                       what lets the model tell a real claim from a hedge.
    --match RE         Which notes to send (default: 'OCR', case-insensitive).
    --shard K/N        Take every Nth pair starting at K, for running several
                       machines in parallel. Give each its own --corrections.
    --backend NAME     'ollama' or 'lmstudio' (default: ollama).
    --model NAME       Model to query (default: [llm.ocr_extractor] in
                       config.toml, per backend).
    --url URL          Server base URL (default depends on --backend).
    --limit N          Stop after N pairs (useful for a first sanity batch).
    --temperature T    Sampling temperature (default: [llm].temperature).
    --dry-run          Show the prompts, ask nothing, write nothing.
    --quiet            Suppress per-pair output.
    --no-notify        Do not push a notification when the run ends.

Author: Christian Lendl
Created: 2026-08-14
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS, LLM_TEMPERATURE
from hyphen_decisions import load as load_decisions
from llm_backend import (
    DEFAULT_BACKEND,
    BackendError,
    backend_defaults,
    call_llm,
    ensure_model,
    format_duration,
    parse_json_reply,
)
from ocr_corrections import (
    CONFIDENCE_LEVELS,
    Correction,
    load as load_corrections,
    merge,
)
from utils.notify import notify

DEFAULT_DECISIONS = PATHS['csv'] / 'hyphen_decisions.csv'
DEFAULT_CORRECTIONS = PATHS['csv'] / 'ocr_corrections.csv'

BACKEND_DEFAULTS = backend_defaults('ocr_extractor')

NOTIFY_TITLE = 'Salonblatt | Extract OCR Corrections'

# The break mark never belongs to a word; strip it if the model puts one back
# anyway. Same three characters validate_xml_hyphens.py knows about.
HYPHEN_CHARS = '¬-='

SYSTEM_PROMPT = """\
Du bist Experte für deutsche Sprache und historische Zeitungstexte. Du \
arbeitest an einer österreichischen Zeitschrift des späten 19. und frühen 20. \
Jahrhunderts ("Wiener Salonblatt"), die über Adel und Gesellschaft berichtet.

Ein anderes Modell hat für ein Wortpaar am Zeilenumbruch entschieden, wie es \
geschrieben wird, und dabei notiert, dass eines der beiden Wörter von der OCR \
falsch gelesen wurde. Deine Aufgabe ist es, DIESE NOTIZ auszuwerten: Welche \
korrigierte Schreibweise behauptet sie für die beiden Wörter? Antworte \
AUSSCHLIESSLICH mit einem JSON-Objekt:

{"left": "<linkes Wort, korrigiert>", "right": "<rechtes Wort, korrigiert>", \
"confidence": "<high|medium|low>", "reason": "<kurze Begründung>"}

Regeln:
1. Du wertest die NOTIZ aus, nicht deine eigene Vermutung. Behauptet sie \
   keinen Wortfehler, gib beide Wörter UNVERÄNDERT zurück.
2. Ein Wort, das in Ordnung ist, wird unverändert zurückgegeben. Oft ist nur \
   eines der beiden Wörter betroffen.
3. Ein über den Zeilenumbruch getrenntes Wort BLEIBT getrennt. Die beiden \
   Teile müssen aneinandergefügt das richtige Wort ergeben:
   "Athmungs" + "rgane" (gemeint: "Athmungsorgane")
   -> {"left": "Athmungs", "right": "organe"}
4. Historische Schreibweisen sind KEINE OCR-Fehler und werden NICHT \
   modernisiert. "Cassa", "Theil", "Bureau", "Curse", "Thor", "Cur", \
   "Athmungsorgane", "-iren" statt "-ieren": alles korrekt so. Ebenso bleiben \
   Umlaut- und ß-Schreibung der Vorlage erhalten.
5. Eigennamen bleiben stehen. Diese Zeitschrift besteht großteils aus Namen \
   von Personen, Familien, Schlössern und Orten; ein seltener echter Name ist \
   kein OCR-Fehler.
6. Der Trennstrich am Zeilenende ("¬", "-" oder "=") gehört NICHT zum Wort und \
   kommt in deiner Antwort nicht vor. Er wird an anderer Stelle behandelt.
7. confidence:
   - "high":   Die Notiz nennt die richtige Schreibweise eindeutig.
   - "medium": Die Notiz behauptet einen Fehler, die Schreibweise ist aber \
               nur erschlossen.
   - "low":    Die Notiz ist vage, widersprüchlich, oder nennt nur einen \
               fehlenden Buchstaben ohne das ganze Wort.

Antworte nur mit dem JSON-Objekt.\
"""

USER_TEMPLATE = """\
Wortpaar am Zeilenumbruch: "{left}" + "{right}"
Entschiedene Form: {form}

Notiz des anderen Modells:
{note}
{contexts}
Antworte nur mit dem JSON-Objekt."""

CONTEXT_TEMPLATE = """
Textstellen (| markiert den Zeilenumbruch):
{contexts}
"""


def load_contexts(path: Optional[Path]) -> Dict[Tuple[str, str], List[str]]:
    """
    Read the sentence contexts from a review JSONL, keyed by word pair.

    Optional: without it the model sees only the note, which is usually enough
    for an explicit claim but leaves it no way to check a hedged one.
    """
    contexts: Dict[Tuple[str, str], List[str]] = {}
    if path is None or not path.exists():
        return contexts
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            contexts[(record['left'], record['right'])] = \
                record.get('contexts', [])
    return contexts


def build_prompt(left: str, right: str, form: str, note: str,
                 contexts: List[str]) -> str:
    """Render the user prompt for one pair."""
    block = ''
    if contexts:
        rendered = '\n'.join(f'  {i}. {c}'
                             for i, c in enumerate(contexts, 1))
        block = CONTEXT_TEMPLATE.format(contexts=rendered)
    return USER_TEMPLATE.format(left=left, right=right, form=form or '?',
                                note=note, contexts=block)


def clean_token(value: str, fallback: str) -> str:
    """
    Sanity-check one corrected token before it is allowed into the store.

    The model is asked for a single word and mostly obliges, but a reply that
    carries the line-break hyphen, whitespace, or a whole phrase would apply as
    a silent corruption later on, so anything unexpected falls back to the
    original token — i.e. "no change on this side".
    """
    token = (value or '').strip().strip(HYPHEN_CHARS).strip()
    if not token or len(token.split()) != 1:
        return fallback
    # A "correction" many times longer than the original is a sentence, not a
    # word; the joined-word case ("Abge" -> "Ab") makes the reverse legitimate.
    if len(token) > max(len(fallback) * 3, len(fallback) + 12):
        return fallback
    return token


def parse_answer(reply: str, left: str, right: str
                 ) -> Tuple[str, str, str, str]:
    """
    Pull (left_correct, right_correct, confidence, reason) from the reply.

    A malformed answer degrades to "no change, low confidence" rather than to a
    guess: the store is meant to be reviewed, and a row that changes nothing
    still records that this pair has been looked at.
    """
    data, error = parse_json_reply(reply)
    if data is None:
        return left, right, 'low', error

    confidence = str(data.get('confidence', '')).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = 'low'
    return (
        clean_token(str(data.get('left', '')), left),
        clean_token(str(data.get('right', '')), right),
        confidence,
        str(data.get('reason', '')).strip()[:200],
    )


def build_queue(decisions, corrections, pattern: re.Pattern,
                shard: Optional[Tuple[int, int]], limit: Optional[int]
                ) -> List[dict]:
    """
    The pairs whose note mentions an OCR error and that are not settled yet.

    Sorted by pair so a queue is reproducible, then sharded round-robin, which
    keeps the mix even across machines the way the review queue is split.
    """
    queue = []
    settled = 0
    for key in sorted(decisions):
        decision = decisions[key]
        if not pattern.search(decision.note or ''):
            continue
        if key in corrections:
            settled += 1
            continue
        queue.append({'left': decision.left, 'right': decision.right,
                      'form': decision.form, 'note': decision.note})
    if settled:
        print(f"Already extracted, not asking again: {settled:,} pairs")
    if shard:
        k, n = shard
        queue = [record for i, record in enumerate(queue) if i % n == k - 1]
    if limit:
        queue = queue[:limit]
    return queue


def parse_shard(value: str) -> Tuple[int, int]:
    """Parse a '1/4' shard argument into (k, n), validating the range."""
    match = re.fullmatch(r'(\d+)\s*/\s*(\d+)', value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"expected K/N (e.g. 1/4), got {value!r}")
    k, n = int(match.group(1)), int(match.group(2))
    if n < 1 or not 1 <= k <= n:
        raise argparse.ArgumentTypeError(
            f"shard {value!r} out of range: need 1 <= K <= N")
    return k, n


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Turn the OCR remarks in the hyphen decision notes into '
                    'reviewable word corrections.')
    parser.add_argument('--decisions', type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument('--corrections', type=Path,
                        default=DEFAULT_CORRECTIONS)
    parser.add_argument('--review', type=Path,
                        help='Review JSONL, for the sentence context the note '
                             'was written about')
    parser.add_argument('--match', default='OCR',
                        help="Which notes to send (default: 'OCR')")
    parser.add_argument('--shard', type=parse_shard, metavar='K/N',
                        help='Take every Nth pair starting at K, for running '
                             'several machines in parallel')
    parser.add_argument('--backend', choices=sorted(BACKEND_DEFAULTS),
                        default=DEFAULT_BACKEND)
    parser.add_argument('--model')
    parser.add_argument('--url')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--temperature', type=float, default=LLM_TEMPERATURE)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-notify', action='store_true',
                        help='Do not push a notification when the run ends')
    args = parser.parse_args()

    defaults = BACKEND_DEFAULTS[args.backend]
    args.model = args.model or defaults['model']
    args.url = args.url or defaults['url']
    pattern = re.compile(args.match, re.I)

    decisions = load_decisions(args.decisions)
    if not decisions:
        print(f"No decisions found in {args.decisions}.\n"
              f"Run resolve_hyphens_llm.py first.", file=sys.stderr)
        return 1
    corrections = load_corrections(args.corrections)
    print(f"Decision store:   {len(decisions):,} pairs ({args.decisions})")
    print(f"Correction store: {len(corrections):,} pairs already extracted")

    contexts = load_contexts(args.review)
    if args.review and not contexts:
        print(f"[warn] no contexts read from {args.review}", file=sys.stderr)

    queue = build_queue(decisions, corrections, pattern, args.shard,
                        args.limit)
    shard_note = (f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard
                  else '')
    print(f"To extract: {len(queue):,} pairs{shard_note}\n")
    if not queue:
        print("Nothing left to ask.")
        return 0

    if args.dry_run:
        for record in queue[:3]:
            print('=' * 62)
            print(build_prompt(record['left'], record['right'],
                               record['form'], record['note'],
                               contexts.get((record['left'],
                                             record['right']), [])))
        print('=' * 62)
        print(f"\nDRY RUN - {len(queue):,} pairs would be sent to "
              f"{args.model}, nothing written.")
        return 0

    # Fail before any work when the server is down or the model is absent.
    try:
        ensure_model(args.model, args.url, args.backend)
    except BackendError as exc:
        print(f"\n{exc}", file=sys.stderr)
        if not args.no_notify:
            notify(f"Did not start: {str(exc).splitlines()[0]}", NOTIFY_TITLE)
        return 1

    started = time.time()
    extracted: List[Correction] = []
    changed = 0
    by_confidence: Dict[str, int] = {}
    for n, record in enumerate(queue, 1):
        left, right = record['left'], record['right']
        prompt = build_prompt(left, right, record['form'], record['note'],
                              contexts.get((left, right), []))
        try:
            reply = call_llm(SYSTEM_PROMPT, prompt, args.model, args.url,
                             args.temperature, args.backend)
        except BackendError as exc:
            # The server, not the model: keep what was really extracted and
            # stop, rather than record answers nobody gave.
            merge(args.corrections, extracted, quiet=True)
            print(f"\nAborted at pair {n} of {len(queue):,}: {exc}",
                  file=sys.stderr)
            print(f"{len(extracted):,} extracted pairs were saved to "
                  f"{args.corrections}; rerun to continue where this left "
                  f"off.", file=sys.stderr)
            if not args.no_notify:
                notify(f"Aborted at pair {n:,} of {len(queue):,} after "
                       f"{format_duration(time.time() - started)}: "
                       f"{str(exc).splitlines()[0]}", NOTIFY_TITLE)
            return 1

        left_correct, right_correct, confidence, reason = parse_answer(
            reply, left, right)
        correction = Correction(
            left=left, right=right,
            left_correct=left_correct, right_correct=right_correct,
            confidence=confidence, source='llm', model=args.model,
            note=reason,
        )
        extracted.append(correction)
        if not correction.is_noop:
            changed += 1
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
        if not args.quiet:
            if correction.is_noop:
                print(f"  [{n}/{len(queue)}] {left} + {right} -> no word fix")
            else:
                print(f"  [{n}/{len(queue)}] {left} + {right} -> "
                      f"{left_correct} + {right_correct}  "
                      f"[{confidence}] ({reason[:50]})")

        # Persist as we go: a long run that dies halfway keeps its answers.
        if n % 50 == 0:
            merge(args.corrections, extracted, quiet=True)
            extracted = []

    if extracted:
        merge(args.corrections, extracted, quiet=True)

    elapsed = format_duration(time.time() - started)
    stored = load_corrections(args.corrections)
    real = sum(1 for c in stored.values() if not c.is_noop)
    print("\n" + "=" * 62)
    print(f"Extracted {len(queue):,} pairs with {args.model} in {elapsed}")
    print(f"  word fixes found  {changed:,}")
    for level in reversed(CONFIDENCE_LEVELS):
        if by_confidence.get(level):
            print(f"    {level:8s} {by_confidence[level]:6,}")
    print(f"  no word fix       {len(queue) - changed:,}")
    print(f"\nCorrection store now at {len(stored):,} pairs, {real:,} with a "
          f"word fix -> {args.corrections}")
    print("Next: review the store, then python correct_xml_ocr.py --dry-run")

    if not args.no_notify:
        notify(f"Extracted {len(queue):,} pairs with {args.model} in "
               f"{elapsed}.\n{changed:,} word fixes found. Store now at "
               f"{len(stored):,} pairs.", NOTIFY_TITLE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
