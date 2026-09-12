#!/usr/bin/env python3
"""
Configuration
=============

Every path and every tunable in one place, so that adapting these tools to
another corpus is an edit to `config.toml` rather than a search through the
scripts.

Exported:
    PATHS               Project paths, derived from this file's location
    LLM_TEMPERATURE     Sampling temperature for every local-model call
    llm_models_for()    Default models for one `[llm.<section>]` table
    PROJECT_NS          The XML namespace for this project's own attributes
    MARKER_PREFIX       Prefix of the `<application ident='...'>` markers
    TRANSKRIBUS         Collection id, base URL and the filename pattern

Paths are **not** read from `config.toml`: they are derived from where this
file sits, so a clone works with no configuration at all. Point the scripts at
a different corpus with `--xml-folder`, which every one of them accepts.

Usage in scripts:
    from config import PATHS

    xml_files = list(PATHS['xml'].glob('*.xml'))

Usage:
    python config.py    # print the resolved paths and say what is missing
"""

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_CONFIG_TOML = ROOT / 'config.toml'
if not _CONFIG_TOML.exists():
    raise FileNotFoundError(
        f"Configuration file not found at {_CONFIG_TOML}. "
        "It must sit next to config.py."
    )

with open(_CONFIG_TOML, 'rb') as _f:
    _SETTINGS = tomllib.load(_f)


# ============================================================================
# Paths
# ============================================================================

PATHS = {
    'root':    ROOT,
    'data':    ROOT / 'data',
    'csv':     ROOT / 'data' / 'csv',
    'prompts': ROOT / 'data' / 'prompts',
    'xml':     ROOT / 'data' / 'xml',
    'output':  ROOT / 'output',
}


# ============================================================================
# The corpus this is pointed at
# ============================================================================

# The namespace for the handful of things TEI has no attribute for — the word
# break marker and its certainty. Everything the Guidelines *do* cover is
# written in plain TEI. Change it to your own domain; nothing reads the URL,
# it only has to be stable and yours.
PROJECT_NS = _SETTINGS['corpus']['project_namespace']

# Prefix of the `<application ident='...'>` markers a script writes into
# `<encodingDesc>` to record that it has run over a file. `mark_xml_word_breaks`
# writes `<prefix>-wordbreaks`.
MARKER_PREFIX = _SETTINGS['corpus']['marker_prefix']


# ============================================================================
# Local models
# ============================================================================

LLM_TEMPERATURE = _SETTINGS['llm']['temperature']


def llm_models_for(section: str) -> dict:
    """
    Default local models for one `[llm.<section>]` table, keyed by backend.

    Each script that talks to a local model names its own section —
    `hyphen_resolver`, `ocr_extractor`, `line_end_resolver`, `break_context` —
    so the model can be changed per task without touching the others. A section
    that does not exist yields an empty mapping and the caller falls back to its
    own default (`lineends/llm_backend.FALLBACK_MODELS`), which keeps adding a
    script from being a config.toml edit as well.
    """
    return dict(_SETTINGS.get('llm', {}).get(section, {}))


# ============================================================================
# Transkribus deep links  (optional — see docs/transkribus-links.md)
# ============================================================================

def _secret(name: str) -> str:
    """
    Read one value from the environment, falling back to `.secret.env`.

    Scripts are usually started with a plain `uv run ...` in a shell that never
    sourced that file, so reading it directly is what makes the value actually
    arrive. Missing is not an error anywhere: every caller degrades to "no
    link".
    """
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / '.secret.env'
    if not env_file.exists():
        return ''
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip().removeprefix('export ').strip()
        if line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ''


# A Transkribus collection id identifies *your* collection and is not something
# to commit, so it is read from the environment or `.secret.env` first. The
# config.toml value is the fallback, for a collection that is already public.
TRANSKRIBUS = {
    'collection_id': (_secret('TRANSKRIBUS_COLLECTION_ID')
                      or str(_SETTINGS['transkribus'].get('collection_id', ''))),
    'base_url': _SETTINGS['transkribus']['base_url'],
    # How to get an issue date out of an XML filename. One capture group, the
    # date, in whatever form the first column of transkribus_filenames.csv uses.
    'filename_pattern': _SETTINGS['transkribus']['filename_pattern'],
}


# ============================================================================
# Self-check
# ============================================================================

if __name__ == '__main__':
    print(f"Project root: {ROOT}\n")
    for key, path in PATHS.items():
        state = 'ok' if path.exists() else 'MISSING'
        print(f"  {key:<8} {state:<8} {path}")
    print(f"\nProject namespace: {PROJECT_NS}")
    print(f"Marker prefix:     {MARKER_PREFIX}")
    print(f"LLM temperature:   {LLM_TEMPERATURE}")
    collection = TRANSKRIBUS['collection_id'] or '(not configured — links off)'
    print(f"Transkribus:       {collection}")
