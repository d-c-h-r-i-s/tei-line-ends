#!/usr/bin/env python3
"""
OCR Text Corrections (shared)
=============================

Pure, string-level OCR correction helpers shared by:
    - lineends/correct_xml_ocr.py  (operates on the Transkribus XML files)
    - the database import step (not shipped) (operates on factoids in the database)

Keeping the correction logic in one place ensures both entry points apply
exactly the same rules. The functions here have no knowledge of XML or the
database — they take a string and return a corrected string plus details
about what changed.

Provided corrections:
    1. Replacement dictionary (transcription errors) — apply_replacements()
    2. OCR digit errors (1↔r, 0↔o, 2↔z, 5↔s) in years, days, "Nummer"/"Seite"
       numbers, and times before " Uhr" — correct_ocr_digits()
    3. Missing slash in two-year date ranges, e.g. "191415" → "1914/15"
       (first year in 1860–1938, second part the following consecutive year)
       — also part of correct_ocr_digits()

One entry point, and why the rules must run in order
----------------------------------------------------
`correct_text()` applies everything below in the canonical order and is what
both callers use, so neither can apply the rules differently or skip a stage.

The rules are applied one after another, each over the *result* of the last, and
that is load-bearing rather than incidental: `replacement.csv` restores
diacritics in stages, `Lancut -> Lańcut -> Łańcut`. Matching every rule against
the original text in one pass instead would stop one step short and write
"Lańcut". `rule_chains()` lists the 21 chains this relies on, so the constraint
is discoverable rather than folklore — it rules out the obvious optimisation, and
`lineends/correct_xml_ocr.py` is built around it: rather than locating each
rule's match, it corrects a whole paragraph through `correct_text()` and then
*diffs* the result to find out where the corrections landed.

Author: Christian Lendl
Created: 2026-06-12
Last Modified: 2026-08-14
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A single correction is reported as (original_token, corrected_token, kind).
Change = Tuple[str, str, str]


# ============================================================================
# Replacement loading
# ============================================================================

def load_replacement_dict(csv_file: Path, quiet: bool = False) -> Dict[str, str]:
    """
    Load OCR replacement rules from a semicolon-delimited CSV file.

    Quoting is switched off (`QUOTE_NONE`). Every field in this file is a literal
    string to search for or write, and `"` is one of the characters being
    corrected — the magazine sets its quotation marks as `“`, so the rule `"` ->
    `“` is a real and wanted rule. With csv's default quoting that rule instead
    opened a quoted field: everything up to the next `"` in the file became one
    giant value, and 595 rules between line 1111 (`";“`) and line 1706
    (`Maroii";Maročić`) silently never loaded at all. Nothing here needs quoting,
    so nothing here is quoted.

    Two kinds of problem in the file are still reported rather than left to be
    discovered as strange corrections months later. Neither stops the load — the
    list is hand-maintained and has to keep working — but each means some rule is
    not doing what its author intended:

    Non-idempotent rules (`old in new`)
        The rule compounds every time it is applied. `Hofmeister ->
        OHofmeister` is how the corpus once acquired `OOOHofmeister`; the current
        rules are padded with spaces, which fixes it, and this warning is what
        keeps the next one from repeating the history.

    Conflicting duplicate keys
        The same `old` given two different `new` values. The last row silently
        wins, so which correction applies depends on file order.

    Args:
        csv_file: Path to CSV with two columns (old_text;new_text)
        quiet: Suppress the warnings (the rules still load)

    Returns:
        Mapping from incorrect text to corrected text. Empty if the file is
        missing (a warning is printed).
    """
    replacements: Dict[str, str] = {}

    csv_file = Path(csv_file)
    if not csv_file.exists():
        print(f"Warning: Replacement CSV not found at {csv_file}")
        return replacements

    conflicts: List[Tuple[str, str, str]] = []
    compounding: List[Tuple[str, str]] = []

    with open(csv_file, 'r', encoding='utf-8-sig') as fh:
        # QUOTE_NONE: see the docstring. `"` is data here, not syntax.
        reader = csv.reader(fh, delimiter=';', quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 2:
                continue
            old, new = row[0], row[1]
            if old and old in new and old != new:
                compounding.append((old, new))
            if old in replacements and replacements[old] != new:
                conflicts.append((old, replacements[old], new))
            replacements[old] = new

    if not quiet:
        for old, new in compounding:
            print(f"Warning: rule {old!r} -> {new!r} contains its own input "
                  f"and compounds on every application")
        for old, first, second in conflicts:
            print(f"Warning: key {old!r} is defined twice ({first!r} and "
                  f"{second!r}); the later one wins")

    return replacements


def apply_replacements(text: str, replacements: Dict[str, str]) -> Tuple[str, int]:
    """
    Apply transcription-error replacements to a string.

    Returns:
        Tuple of (corrected_text, total_number_of_replacements).
    """
    count = 0
    for old, new in replacements.items():
        if old in text:
            count += text.count(old)
            text = text.replace(old, new)
    return text, count


# ============================================================================
# OCR digit error correction
# ============================================================================

# 4-char year token preceded by space; char2 must be 7, 8, 9, or 'o' (misread 9).
# char1 may be 'r' (misread 1), char3 may be 'o'/'z' (misread 0/2),
# char4 may be 'r'/'s'/'z' (misread 1/5/2). Negative lookahead prevents
# matching a prefix of a longer digit sequence.
YEAR_PATTERN = re.compile(r'(?<= )[1r][789o][0-9osz][0-9rsz](?!\d)')

# A number token must not run straight into a letter. Without this guard the
# patterns below eat the first letter of the following *word*: "Seite sieht"
# became "Seite 5ieht", "Seite zeigen" became "Seite 2eigen", because `s` and `z`
# are candidate misread digits and nothing said the match had to end. 251 such
# corruptions are already in data/xml/ and 125 factoids carry them in the
# database — see docs/line-end-correction.md.
NOT_LETTER_OR_DIGIT = r'(?![0-9A-Za-zÄÖÜäöüßÀ-ÿ])'

# 1-2 char day token preceded by space and followed by '.'.
# char1: 'r' (misread 1) or literal 1-3; optional char2: digit, 'o', 'r', or 's'.
# 'z' (misread 2) is excluded — "z." is a noble title abbreviation (zu).
DAY_PATTERN = re.compile(r'(?<= )[r1-3][0-9ors]?(?=\.)')

# 1-2 char number after "Nummer " or "Seite ".
NUMMER_PATTERN = re.compile(
    r'(?<=Nummer )[0-9rosz]{1,2}' + NOT_LETTER_OR_DIGIT)
SEITE_PATTERN = re.compile(
    r'(?<=Seite )[0-9rosz]{1,2}' + NOT_LETTER_OR_DIGIT)

# 3- or 4-digit time before " Uhr", 12-hour style: hour 1-12, minute 00-59.
# (?<!\d) prevents matching the tail of a longer digit run like "1730 Uhr".
UHR_TIME_PATTERN = re.compile(r'(?<!\d)(1[0-2]|[1-9])([0-5]\d)(?= Uhr)')

# 6-digit run with no digit on either side, e.g. "191415". The two halves are
# validated in try_correct_year_range() (first year 1860-1938, second part the
# following consecutive year). The OCR commonly drops the '/' between the years.
YEAR_RANGE_PATTERN = re.compile(r'(?<!\d)\d{6}(?!\d)')


def try_correct_year(token: str) -> Optional[str]:
    if not any(c in token for c in 'rosz'):
        return None
    # 'o' at position 1 (second digit) represents '9', not '0' (e.g. "1o13" → "1913").
    if token[1] == 'o':
        token = token[0] + '9' + token[2:]
    corrected = token.replace('r', '1').replace('o', '0').replace('z', '2').replace('s', '5')
    try:
        if 1700 <= int(corrected) <= 1930:
            return corrected
    except ValueError:
        pass
    return None


def try_correct_day(token: str) -> Optional[str]:
    if not any(c in token for c in 'ros'):
        return None
    corrected = token.replace('r', '1').replace('o', '0').replace('s', '5')
    try:
        if 1 <= int(corrected) <= 31:
            return corrected
    except ValueError:
        pass
    return None


def try_correct_short_num(token: str) -> Optional[str]:
    if not any(c in token for c in 'rosz'):
        return None
    corrected = token.replace('r', '1').replace('o', '0').replace('z', '2').replace('s', '5')
    try:
        if 1 <= int(corrected) <= 99:
            return corrected
    except ValueError:
        pass
    return None


def try_correct_year_range(token: str) -> Optional[str]:
    """
    Re-insert a missing '/' in a two-year date range.

    The OCR frequently fails to recognise the slash, merging a range such as
    "1914/15" into "191415". A 6-digit token is treated as a range only when:
        - the first four digits are a year in 1860-1938, and
        - the last two digits are the *following* consecutive year
          (i.e. (year + 1) modulo 100, so "1899/00" for 1899→1900).

    Returns:
        The token with the slash inserted (e.g. "1914/15"), or None if the
        token is not a valid consecutive-year range.
    """
    year = int(token[:4])
    second = int(token[4:])
    if 1860 <= year <= 1938 and second == (year + 1) % 100:
        return f"{token[:4]}/{token[4:]}"
    return None


def correct_ocr_digits(text: str) -> Tuple[str, List[Change]]:
    """
    Apply all OCR digit/range corrections to a single string.

    Returns:
        Tuple of (corrected_text, changes) where changes is a list of
        (original_token, corrected_token, kind) tuples. ``len(changes)`` is
        the number of substitutions made; an empty list means no change.
    """
    changes: List[Change] = []

    def _make_repl(corrector, kind):
        def repl(match):
            token = match.group(0)
            corrected = corrector(token)
            if corrected:
                changes.append((token, corrected, kind))
                return corrected
            return token
        return repl

    def repl_uhr(match):
        token = match.group(0)
        corrected = f"{match.group(1)}.{match.group(2)}"
        changes.append((token, corrected, 'uhr'))
        return corrected

    new_text = YEAR_RANGE_PATTERN.sub(_make_repl(try_correct_year_range, 'year_range'), text)
    new_text = YEAR_PATTERN.sub(_make_repl(try_correct_year, 'year'), new_text)
    new_text = DAY_PATTERN.sub(_make_repl(try_correct_day, 'day'), new_text)
    new_text = NUMMER_PATTERN.sub(_make_repl(try_correct_short_num, 'nummer'), new_text)
    new_text = SEITE_PATTERN.sub(_make_repl(try_correct_short_num, 'seite'), new_text)
    new_text = UHR_TIME_PATTERN.sub(repl_uhr, new_text)
    return new_text, changes


# ============================================================================
# Combined entry point
# ============================================================================

def rule_chains(replacements: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Rules whose output is itself matched by another rule.

    `apply_replacements` walks the rules one after another over the *result* of
    the previous one, so a rule can feed another. In this list that is
    load-bearing and clearly deliberate — diacritics are restored in stages:

        Lancut     -> Lańcut     -> Łańcut
        Dobrzensky -> Dobržensky -> Dobrženský
        Czizek     -> Czjzek     -> Cžjžek

    Anything that wants to reproduce `apply_replacements` therefore cannot match
    all the rules against the original text in one pass; it has to apply them in
    order, or it stops one step short and writes "Lańcut". This function is here
    so that constraint is discoverable rather than folklore.

    Returns:
        List of (old, new, next_old) triples, where applying `old -> new`
        creates text that `next_old` then matches.
    """
    chains: List[Tuple[str, str, str]] = []
    for old, new in replacements.items():
        if not old or old == new:
            continue
        for other in replacements:
            if other and other != old and other in new and other not in old:
                chains.append((old, new, other))
    return chains


def correct_text(
    text: str,
    replacements: Dict[str, str],
) -> Tuple[str, int, List[Change]]:
    """
    Apply every correction in this module to one string, in the canonical order.

    The single entry point for callers that want "the text, corrected" — the
    XML corrector and the database step both go through here, so neither can
    apply the rules in a different order or forget one of the two stages.

    Order matters and is not arbitrary: the replacement rules run first, because
    several of them create or repair the tokens the digit patterns then look at
    (a rule fixing " Selte " to " Seite " is what lets the `Seite N` pattern
    match at all).

    Returns:
        Tuple of (corrected_text, number_of_replacements, digit_changes).
    """
    corrected, replaced = apply_replacements(text, replacements)
    corrected, digit_changes = correct_ocr_digits(corrected)
    return corrected, replaced, digit_changes
