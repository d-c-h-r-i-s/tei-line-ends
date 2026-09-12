#!/usr/bin/env python3
"""
Transkribus Link Generator
==========================

Generates Transkribus links for a given issue date.

Usage (CLI):
    python get_transkribus_link.py 19190501
    python get_transkribus_link.py 1919-05-01 --page 5

Usage (as module):
    from utils.get_transkribus_link import get_transkribus_link

    link = get_transkribus_link("19190501")
    link = get_transkribus_link("1919-05-01", page=5)
    link = get_transkribus_link("ONB_wsb_19190501.xml", page=3)

Flags:
    --page N           Jump to specific page number in Transkribus.

Configuration:
    The collection id comes from `TRANSKRIBUS_COLLECTION_ID` (environment or
    `.secret.env`), or `[transkribus] collection_id` in config.toml. The
    issue-date-to-document-id mapping is `data/csv/transkribus_filenames.csv`.
    Both are optional: with neither, every validator simply writes no link.
    See docs/transkribus-links.md.

Note:
    Accepts date formats: YYYYMMDD or YYYY-MM-DD (dashes are stripped automatically).
    Also accepts XML filenames, parsed with `[transkribus] filename_pattern`.

Author: Christian Lendl
Created: 2026-01-22
Last Modified: 2026-01-22
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, TRANSKRIBUS

# ============================================================================
# Configuration
# ============================================================================

# A collection id identifies *your* Transkribus collection, so it is not in the
# code: `config.py` reads it from `TRANSKRIBUS_COLLECTION_ID` in the environment
# or `.secret.env`, falling back to `[transkribus] collection_id` in
# config.toml for a collection that is already public. With none configured,
# every caller degrades to "no link" — see `links_available()`.
COLLECTION_ID = TRANSKRIBUS['collection_id']
BASE_URL = f"{TRANSKRIBUS['base_url']}/{COLLECTION_ID}/doc"

# Which part of an XML filename is the issue date. One capture group, in the
# form the first column of the mapping CSV uses.
FILENAME_PATTERN = re.compile(TRANSKRIBUS['filename_pattern'])

# Issue date -> Transkribus document id. Semicolon-separated, two columns,
# header `ISSUE_DATE;TRANSKRIBUS_ID`. The file is yours to fill in: see
# docs/transkribus-links.md for how to get the ids out of Transkribus.
CSV_PATH = PATHS['csv'] / 'transkribus_filenames.csv'

# Tracks issue dates already warned about, so callers that hit the same missing
# date many times only get a single reminder per process.
_WARNED_MISSING_DATES: set = set()


# ============================================================================
# Functions
# ============================================================================

def load_transkribus_mapping() -> dict:
    """
    Load the mapping from issue date to Transkribus ID from CSV.

    Returns:
        Dict mapping issue date (str) to Transkribus ID (str)
    """
    mapping = {}

    if not CSV_PATH.exists():
        return mapping

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            issue_date = row['ISSUE_DATE'].strip()
            transkribus_id = row['TRANSKRIBUS_ID'].strip()
            if issue_date and transkribus_id:
                mapping[issue_date] = transkribus_id

    return mapping


def links_available() -> bool:
    """
    Whether a link can be built at all: a collection id and a mapping file.

    Callers import this module inside a `try`, so a missing collection is
    already survivable — this is for the ones that would rather leave the
    column out than fill it with warnings.
    """
    return bool(COLLECTION_ID) and CSV_PATH.exists()


def extract_date_from_filename(filename: str) -> str:
    """
    Extract issue date from XML filename.

    Args:
        filename: Any filename the configured pattern matches, e.g.
                  'ONB_wsb_19190501.xml' under the default eight-digit pattern

    Returns:
        Date string like '19190501' or empty string if not found
    """
    match = FILENAME_PATTERN.search(filename)
    if match:
        return match.group(1)
    return ""


def get_transkribus_link(issue_date: str, page: int = None) -> str:
    """
    Generate a Transkribus link for a given issue date or XML filename.

    Args:
        issue_date: Issue date (YYYYMMDD, YYYY-MM-DD) or XML filename
                    (e.g., 'ONB_wsb_19190501.xml')
        page: Optional page number

    Returns:
        Transkribus URL string

    Raises:
        ValueError: If issue date not found in mapping
    """
    if not COLLECTION_ID:
        raise ValueError(
            "No Transkribus collection configured. Set "
            "TRANSKRIBUS_COLLECTION_ID in .secret.env or [transkribus] "
            "collection_id in config.toml; see docs/transkribus-links.md."
        )

    mapping = load_transkribus_mapping()

    # If input looks like a filename, extract the date
    if issue_date.endswith('.xml'):
        issue_date = extract_date_from_filename(issue_date)
        if not issue_date:
            raise ValueError(f"Could not extract date from filename")

    # Normalize issue date (remove any dashes or other formatting)
    issue_date = issue_date.replace('-', '').replace('/', '')

    if issue_date not in mapping:
        if issue_date not in _WARNED_MISSING_DATES:
            _WARNED_MISSING_DATES.add(issue_date)
            print(
                f"WARNING: Issue date '{issue_date}' not found in {CSV_PATH.name}. "
                f"Add it to enable Transkribus links for this issue.",
                file=sys.stderr,
            )
        raise ValueError(f"Issue date '{issue_date}' not found in transkribus_filenames.csv")

    transkribus_id = mapping[issue_date]

    if page is not None:
        return f"{BASE_URL}/{transkribus_id}/edit?pageNr={page}"
    else:
        return f"{BASE_URL}/{transkribus_id}"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Transkribus link for a given issue date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python get_transkribus_link.py 19190501
    python get_transkribus_link.py 19190501 --page 5
        """
    )
    parser.add_argument(
        'issue_date',
        help='Issue date in format YYYYMMDD (e.g., 19190501)'
    )
    parser.add_argument(
        '--page', '-p',
        type=int,
        default=None,
        help='Page number to link to'
    )

    args = parser.parse_args()

    try:
        link = get_transkribus_link(args.issue_date, args.page)
        print(link)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
