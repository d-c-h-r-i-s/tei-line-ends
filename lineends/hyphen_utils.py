#!/usr/bin/env python3
"""
Hyphen Processing Utilities for XML Import
===========================================

This module provides functions for intelligent hyphen removal during XML import.
Line-end hyphens (-, ¬, =) are detected and processed based on the following word.

Hyphen Removal Rules:
    - OCR errors (followed by geb., v., gew.): Replace with space
    - Uppercase following: Keep as hyphen (-)
    - "und" following: Keep as hyphen (-)
    - Hungarian street abbreviations (út, üt, ut, utca): Keep as hyphen (-)
    - Lowercase following: Remove entirely (word continuation)

Public Functions:
    - detect_line_end_hyphen(line_text): Detect if a line ends with a hyphen
    - get_next_word(next_line_text): Extract the first word from a line
    - determine_hyphen_action(hyphen_char, next_word): Determine hyphen handling
    - join_lines_with_hyphen_cleanup(lines): Join text lines with hyphen handling
    - join_lines_with_origin(lines): The same join, plus the source line and
      offset of every character, so a correction found in the joined text can be
      written back into the line it belongs to
    - cleanup_hyphens_legacy(originaltext, removethis): Legacy cleanup function

This logic was originally in prepare_factoids.py but has been moved here
to process text during XML import when line boundaries are still available.

Author: Christian Lendl
Created: 2025-12-27
Last Modified: 2025-12-27
"""

import re
from typing import List, Optional, Tuple


# Constants for hyphen processing
HYPHEN_CHARS = ['¬', '-', '=']
OCR_SPACE_PATTERNS = ['geb.', 'v.', 'gew.', 'z.', '(']
HUNGARIAN_STREET_ABBRS = ['út', 'üt', 'ut', 'utca']


def detect_line_end_hyphen(line_text: str) -> Tuple[bool, str, str]:
    """
    Detect if a line ends with a hyphen character.

    Args:
        line_text: Text from a single line

    Returns:
        Tuple of (has_hyphen, hyphen_char, text_without_hyphen):
        - has_hyphen: True if line ends with ¬, -, or =
        - hyphen_char: The actual hyphen character found (or empty string)
        - text_without_hyphen: Line text with trailing hyphen removed
    """
    if not line_text:
        return (False, '', '')

    # Check if line ends with any hyphen character
    for hyphen in HYPHEN_CHARS:
        if line_text.endswith(hyphen):
            return (True, hyphen, line_text[:-len(hyphen)])

    return (False, '', line_text)


def get_next_word(next_line_text: str) -> str:
    """
    Extract the first word from the next line.

    Args:
        next_line_text: Text from the following line

    Returns:
        First word (up to space, punctuation, or end of line)
    """
    if not next_line_text:
        return ''

    # Match word characters at start of line
    # Include periods for abbreviations like "geb.", "v."
    match = re.match(r'^([a-zA-ZäöüÄÖÜßáéíóúÁÉÍÓÚ.]+)', next_line_text)

    if match:
        return match.group(1)

    return ''


def determine_hyphen_action(hyphen_char: str, next_word: str) -> str:
    """
    Determine what to do with the hyphen based on the next word.

    Args:
        hyphen_char: The hyphen character found (¬, -, or =)
        next_word: First word from the next line

    Returns:
        Action to take:
        - 'space': Replace with space (OCR error)
        - 'hyphen': Keep as hyphen (-)
        - 'remove': Remove entirely (word continuation)
    """
    if not next_word:
        # No next word, remove the hyphen
        return 'remove'

    # Check for OCR errors that should be spaces
    if next_word in OCR_SPACE_PATTERNS:
        return 'space'

    # Check if next word starts with uppercase
    if next_word[0].isupper():
        return 'hyphen'

    # Check for "und"
    if next_word == 'und':
        return 'hyphen'

    # Check for Hungarian street abbreviations
    if next_word in HUNGARIAN_STREET_ABBRS:
        return 'hyphen'

    # Default: lowercase continuation, remove hyphen
    return 'remove'


def join_lines_with_hyphen_cleanup(lines: List[str]) -> str:
    """
    Join text lines with intelligent hyphen handling.

    Processes each line to detect line-end hyphens and applies appropriate
    cleanup rules based on the next line's starting word.

    Args:
        lines: List of text lines from a paragraph

    Returns:
        Concatenated text with proper hyphen handling
    """
    if not lines:
        return ''

    if len(lines) == 1:
        # Single line, no hyphen processing needed
        return lines[0]

    result_parts = []

    for i in range(len(lines)):
        current_line = lines[i]

        # Check if this line ends with a hyphen
        has_hyphen, hyphen_char, line_without_hyphen = detect_line_end_hyphen(current_line)

        if has_hyphen and i < len(lines) - 1:
            # There's a hyphen and there's a next line
            next_line = lines[i + 1]
            next_word = get_next_word(next_line)

            action = determine_hyphen_action(hyphen_char, next_word)

            if action == 'space':
                # OCR error - replace hyphen with space
                result_parts.append(line_without_hyphen + ' ')
            elif action == 'hyphen':
                # Keep the hyphen (normalize to -)
                result_parts.append(line_without_hyphen + '-')
            else:  # action == 'remove'
                # Word continuation - remove hyphen entirely
                result_parts.append(line_without_hyphen)
        else:
            # No hyphen at end, or last line - add as-is with space
            result_parts.append(current_line)

            # Add space between lines (except for the last line)
            if i < len(lines) - 1:
                # Only add space if we didn't just handle a hyphen
                if not has_hyphen:
                    result_parts.append(' ')

    return ''.join(result_parts)


# Marks a character of the joined text that no source line contains: the space
# or hyphen the join itself puts at a line break.
JOIN_ARTIFACT = (None, None)


def join_lines_with_origin(
    lines: List[str],
) -> Tuple[str, List[Tuple[Optional[int], Optional[int]]]]:
    """
    Join lines exactly as join_lines_with_hyphen_cleanup does, and say where
    every character of the result came from.

    Why this exists
    ---------------
    `data/csv/replacement.csv` used to be applied twice: once to each `<l>`
    element of the XML, and again to the joined factoid text in the database,
    because a rule spanning a line break cannot match a single line. Correcting
    the *XML* on the joined text removes the need for the second pass — but a
    correction found in joined coordinates has to be written back into the line
    it actually belongs to, and that needs this map.

    Returns:
        Tuple of (joined_text, origin) where `origin[k]` is
        `(line_index, offset_in_line)` for a character taken from a source line,
        and `JOIN_ARTIFACT` (i.e. `(None, None)`) for one the join introduced —
        the space between two unhyphenated lines, or the `-`/` ` a line-end
        hyphen is turned into. Those belong to no line and so can never be
        written back; see `0_validate/correct_xml_ocr.py`, which reports such a
        correction instead of applying it.

    The text returned is byte-identical to `join_lines_with_hyphen_cleanup`'s
    for the same input, and `scripts/test_and_debug/test_xml_readers.py` asserts
    that over the whole corpus. The map is worthless if the text it annotates is
    not the text the import produces.
    """
    if not lines:
        return '', []

    if len(lines) == 1:
        return lines[0], [(0, k) for k in range(len(lines[0]))]

    parts: List[str] = []
    origin: List[Tuple[Optional[int], Optional[int]]] = []

    def emit(text: str, line_index: Optional[int], start: int = 0) -> None:
        """Append text, recording where each of its characters came from."""
        parts.append(text)
        if line_index is None:
            origin.extend([JOIN_ARTIFACT] * len(text))
        else:
            origin.extend((line_index, start + k) for k in range(len(text)))

    for i in range(len(lines)):
        current_line = lines[i]

        has_hyphen, hyphen_char, line_without_hyphen = \
            detect_line_end_hyphen(current_line)

        if has_hyphen and i < len(lines) - 1:
            next_word = get_next_word(lines[i + 1])
            action = determine_hyphen_action(hyphen_char, next_word)

            emit(line_without_hyphen, i)
            if action == 'space':
                # OCR error - the hyphen becomes a space that is in no line
                emit(' ', None)
            elif action == 'hyphen':
                # A real compound hyphen, normalized to '-'; still not a
                # character any single line contains, since the line holds
                # whichever of ¬/-/= the OCR produced.
                emit('-', None)
            # action == 'remove': the word continues, nothing is emitted
        else:
            emit(current_line, i)
            if i < len(lines) - 1 and not has_hyphen:
                emit(' ', None)

    return ''.join(parts), origin


# For backward compatibility and testing
def cleanup_hyphens_legacy(originaltext: str, removethis: str) -> str:
    """
    Legacy hyphen cleanup function (kept for comparison/testing).

    This is the original algorithm from prepare_factoids.py.
    New code should use join_lines_with_hyphen_cleanup() instead.

    Args:
        originaltext: Text to clean
        removethis: Character sequence to remove (e.g., "¬ ", "- ")

    Returns:
        Cleaned text with -XXX markers (caller should replace with "- ")
    """
    # Process as long as there is something to remove
    while removethis in originaltext:
        # Find the characters to remove
        start = originaltext.find(removethis)

        if start == -1:
            break

        # Get the text after the hyphen character
        after_hyphen_start = start + len(removethis)

        # Check for OCR errors (geb., v.) - should be spaces
        if any(after_hyphen_start + len(pattern) <= len(originaltext) and
               originaltext[after_hyphen_start:after_hyphen_start + len(pattern)] == pattern
               for pattern in OCR_SPACE_PATTERNS):
            # Replace hyphen with space (OCR error correction)
            newtext = originaltext.replace(removethis, ' ', 1)

        # If character after hyphen is upper case, replace by "-"
        elif after_hyphen_start < len(originaltext) and originaltext[after_hyphen_start].isupper():
            newtext = originaltext.replace(removethis, '-', 1)

        # If the next word is "und" then replace by "-XXX" (to avoid infinite loop)
        elif after_hyphen_start + 3 <= len(originaltext) and originaltext[after_hyphen_start:after_hyphen_start + 3] == 'und':
            newtext = originaltext.replace(removethis, '-XXX', 1)

        # Check for Hungarian street abbreviations (út, üt, ut, utca)
        elif any(after_hyphen_start + len(abbr) <= len(originaltext) and
                originaltext[after_hyphen_start:after_hyphen_start + len(abbr)] == abbr
                for abbr in HUNGARIAN_STREET_ABBRS):
            # Replace with hyphen (even if it was ¬ or =)
            newtext = originaltext.replace(removethis, '-', 1)

        # If character after hyphen is lower case, remove completely
        else:
            newtext = originaltext.replace(removethis, '', 1)

        originaltext = newtext

    return originaltext
