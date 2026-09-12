#!/usr/bin/env python3
"""
Push Notifications for Long-Running Scripts
===========================================

Sends a short message to an [ntfy](https://ntfy.sh) topic, so a script that runs
for hours can say when it finished instead of being watched. Meant for anything
in this workflow that outlives the attention span of the person who started it —
the LLM hyphen resolution, NER batches, image pipelines.

The topic URL lives in `.secret.env` at the project root, next to the other
credentials:

    export NTFY_CHANNEL=https://ntfy.sh/your-private-topic-name

Anyone who knows an ntfy topic URL can read and post to it, which is why it
belongs in `.secret.env` (gitignored) and not in `config.py`. Pick a long,
unguessable name.

The variable is read from the environment first, so a sourced shell or an
exported value wins. Failing that, `.secret.env` is parsed directly — scripts
are often started with plain `uv run ...` in a shell that never sourced it, and
a notification that silently never arrives is worse than no notification at all.
With no channel configured anywhere, `notify()` does nothing and says so once.

Notifications are best-effort: a dead network, a typo in the URL or ntfy being
down can never break the run that is reporting.

Usage (as module):
    from utils.notify import notify

    notify("Resolved 1,432 pairs in 3h 12m", title="resolve_hyphens_llm")

Usage (CLI, to test the setup):
    python utils/notify.py "test message" --title "line ends"

Author: Christian Lendl
Created: 2026-07-27
Last Modified: 2026-07-27
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS

# ============================================================================
# Configuration
# ============================================================================

ENV_VAR = 'NTFY_CHANNEL'
SECRET_ENV = PATHS['root'] / '.secret.env'

# Resolved channel, cached after the first lookup. `False` means "looked and
# found nothing" and is kept apart from `None` ("not looked yet"), so an
# unconfigured setup is not re-read from disk on every notification.
_CHANNEL = None

# One hint per process when nothing is configured; a long run must not turn
# into a wall of warnings.
_WARNED_MISSING = False


# ============================================================================
# Functions
# ============================================================================

def _read_secret_env(name: str) -> Optional[str]:
    """
    Pull a single variable out of `.secret.env` without sourcing it.

    The file is shell syntax (`export KEY=value`), but only the plain
    assignments used in this project are understood — no expansion, no
    substitution. Unreadable or absent file simply means "not configured".
    """
    try:
        lines = SECRET_ENV.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None

    for line in lines:
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        key, sep, value = line.partition('=')
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def get_channel() -> Optional[str]:
    """
    Return the configured ntfy topic URL, or None if there is none.

    Environment wins over `.secret.env`, so a one-off run can redirect its
    notifications with `NTFY_CHANNEL=... uv run script.py`.
    """
    global _CHANNEL

    if _CHANNEL is None:
        _CHANNEL = os.environ.get(ENV_VAR) or _read_secret_env(ENV_VAR) or False
    return _CHANNEL or None


def notify(message: str, title: Optional[str] = None) -> bool:
    """
    Send `message` to the configured ntfy channel.

    Args:
        message: Notification body. Keep it to a line or two — this is read on
                 a phone lock screen.
        title:   Optional notification title, so each script can name itself.

    Returns:
        True if the notification was handed to ntfy, False if no channel is
        configured or the request failed.

    Never raises: a failed notification must not end a run that succeeded.
    """
    global _WARNED_MISSING

    channel = get_channel()
    if not channel:
        if not _WARNED_MISSING:
            _WARNED_MISSING = True
            print(f"[note] no {ENV_VAR} in the environment or {SECRET_ENV.name}"
                  f" - not sending notifications.", file=sys.stderr)
        return False

    try:
        import requests  # lazy: keep this module importable without the venv

        headers = {'Title': title} if title else None
        response = requests.post(channel, data=message.encode('utf-8'),
                                 headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 - notifications are best-effort
        print(f"[warn] could not send ntfy notification: {exc}",
              file=sys.stderr)
        return False


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Send a test notification to the configured ntfy channel.')
    parser.add_argument('message', nargs='?', default='Test notification',
                        help='Message body (default: "Test notification")')
    parser.add_argument('--title', help='Notification title')
    args = parser.parse_args()

    channel = get_channel()
    if not channel:
        print(f"No {ENV_VAR} configured.\n"
              f"Add it to {SECRET_ENV}:\n"
              f"    export {ENV_VAR}=https://ntfy.sh/your-private-topic-name",
              file=sys.stderr)
        return 1

    print(f"Channel: {channel}")
    return 0 if notify(args.message, args.title) else 1


if __name__ == '__main__':
    raise SystemExit(main())
