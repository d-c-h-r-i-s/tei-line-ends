#!/usr/bin/env python3
"""
Local LLM Backend shared by the lineends scripts
==================================================

`resolve_hyphens_llm.py`, `resolve_line_end_llm.py` and
`extract_ocr_corrections.py` all send thousands of small German questions to a
local model, and all three have to survive a server that dies halfway through a
run that takes hours. The wire protocol, the model check and the retry policy
live here so the three cannot drift apart.

Ollama and LM Studio disagree on almost everything: native `/api/chat` versus
the OpenAI-compatible `/v1/chat/completions`, `/api/tags` versus `/v1/models`,
different ports, different ideas of what a model id looks like. `backend` picks
which set applies; everything above that line is identical for both.

A failure of the *server* is kept strictly apart from a bad answer. An
unparseable reply is a verdict about one question; an unreachable server says
nothing about the question at all, and recording it as one would poison a
decision store with answers nobody gave. Server failures raise `BackendError`
so the caller can stop, keep what it has, and be restarted.

Exported:
    DEFAULT_BACKEND, BACKEND_URLS   Which server, and where it listens
    backend_defaults(section)       Per-backend url + model for one script
    BackendError                    The server failed, not the model
    ensure_model()                  Fail before any work when it cannot run
    call_llm()                      One chat request, with retries
    parse_json_reply()              The JSON object out of a reply
    load_prompt()                   The prompt sections out of data/prompts/
    format_duration()               '3h 12m', '12m 30s', '47s'

Author: Christian Lendl
Created: 2026-08-14
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LLM_TEMPERATURE, PATHS, llm_models_for  # noqa: E402

DEFAULT_BACKEND = 'ollama'

# Fixed infrastructure defaults; the models come from config.toml
BACKEND_URLS = {
    'ollama': 'http://localhost:11434',
    'lmstudio': 'http://127.0.0.1:1234',
}

# Used only when config.toml names no model for a backend
FALLBACK_MODELS = {
    'ollama': 'qwen2.5:14b',
    'lmstudio': 'qwen3.5-35b-a3b',
}

# A prompt file is one text file per script in data/prompts/, cut into named
# sections by a line reading `### <name>`. Anything before the first such line
# is a comment block: what the file is for, and which placeholders its sections
# may use. `###` is what German prose does not begin a line with, which is why
# it marks the sections rather than something more conventional.
PROMPT_SECTION_RE = re.compile(r'^###[ \t]*([A-Za-z0-9_-]+)[ \t]*$', re.M)

JSON_RE = re.compile(r'\{.*?\}', re.S)
THINK_RE = re.compile(r'<think>.*?</think>', re.S)

# Re-exported so a script needs one import for its defaults
DEFAULT_TEMPERATURE = LLM_TEMPERATURE


def backend_defaults(section: str) -> Dict[str, Dict[str, str]]:
    """
    Per-backend URL and default model for one `[llm.<section>]` config table.

    Lets each script name its own model without duplicating the URL table:
    `backend_defaults('hyphen_resolver')`, `backend_defaults('ocr_extractor')`,
    `backend_defaults('line_end_resolver')`.
    """
    models = llm_models_for(section)
    return {
        backend: {'url': url,
                  'model': models.get(backend, FALLBACK_MODELS[backend])}
        for backend, url in BACKEND_URLS.items()
    }


class BackendError(RuntimeError):
    """
    The server could not be reached or refused the request.

    Kept apart from a model that simply answered badly: an unparseable answer
    is a verdict about one question, whereas this says nothing about the
    question at all and must never be recorded as one.
    """


def ensure_model(model: str, base_url: str, backend: str) -> None:
    """
    Check the server is up and the model is present, before any work starts.

    Ollama lists models at `/api/tags` and answers a request for one it does
    not have with a plain 404; LM Studio lists them at the OpenAI-compatible
    `/v1/models` and 404s the wrong endpoint instead - which, read as an empty
    model list, used to surface as a misleading "model not installed" for every
    model. Failing here instead names the actual problem and lists what is
    present.
    """
    import requests

    if backend == 'ollama':
        list_url, key, name_key = f'{base_url}/api/tags', 'models', 'name'
    else:
        list_url, key, name_key = f'{base_url}/v1/models', 'data', 'id'

    try:
        response = requests.get(list_url, timeout=10)
        response.raise_for_status()
        installed = [m[name_key] for m in response.json().get(key, [])]
    except requests.RequestException as exc:
        raise BackendError(
            f"cannot reach {backend} at {base_url} ({exc}).\n"
            f"Start the server, or point --url elsewhere."
        ) from exc

    # Ollama reports "name:tag"; an untagged request means the ":latest" tag.
    # LM Studio's ids are exact, no implicit tag.
    if model in installed or (backend == 'ollama'
                              and f'{model}:latest' in installed):
        return
    hint = (f"Pull it with `ollama pull {model}`" if backend == 'ollama' else
            f"Load it in LM Studio (or check `curl {base_url}/v1/models`)")
    raise BackendError(
        f"model {model!r} is not installed on {base_url}.\n"
        f"Installed: {', '.join(sorted(installed)) or '(none)'}\n"
        f"{hint}, or choose one of the above with --model."
    )


def call_llm(system: str, user: str, model: str, base_url: str,
             temperature: float, backend: str, retries: int = 2) -> str:
    """
    Send one chat request and return the raw reply text.

    Ollama's native `/api/chat` and LM Studio's OpenAI-compatible
    `/v1/chat/completions` want differently shaped requests and responses, so
    the request/response handling branches on `backend`; the retry policy
    around it is shared. A transient failure is retried a couple of times — a
    single dropped connection should not end a run of well over a thousand
    questions — but a persistent one raises `BackendError` so the caller can
    stop rather than record a verdict nobody gave.
    """
    import requests

    if backend == 'ollama':
        url = f'{base_url}/api/chat'
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'stream': False,
            'format': 'json',
            'options': {'temperature': temperature},
        }
    else:
        url = f'{base_url}/v1/chat/completions'
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'stream': False,
            'temperature': temperature,
        }

    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            if backend == 'ollama':
                return data.get('message', {}).get('content', '')
            return data['choices'][0]['message'].get('content', '')
        except requests.RequestException as exc:
            last = exc
            # A 404 means the model vanished mid-run; retrying cannot help
            status = getattr(getattr(exc, 'response', None),
                             'status_code', None)
            if status == 404:
                break
            if attempt < retries:
                time.sleep(2 * (attempt + 1))

    raise BackendError(f"request to {url} failed: {last}")


def parse_json_reply(reply: str) -> Tuple[Optional[dict], str]:
    """
    Pull the JSON object out of a model's reply.

    Returns (data, '') on success and (None, reason) when there is nothing
    usable, so the caller decides what a bad answer means for its own store —
    a `skip` verdict for the hyphens, a `low` confidence for the word fixes.

    Reasoning models (e.g. LM Studio's qwen3.5) preface the answer with a
    `<think>...</think>` block that itself often contains braces - the system
    prompt's own JSON example gets echoed while the model reasons about it. A
    non-greedy search over the raw reply would grab that instead of the real
    answer at the end, so the think block is stripped first.
    """
    reply = THINK_RE.sub('', reply or '')
    match = JSON_RE.search(reply)
    if not match:
        return None, f'unparseable reply: {(reply or "")[:120]}'
    try:
        return json.loads(match.group(0)), ''
    except json.JSONDecodeError:
        return None, f'invalid JSON: {match.group(0)[:120]}'


def parse_json_object(reply: str) -> Tuple[Optional[dict], str]:
    """
    The outermost JSON object of a reply, for answers that nest.

    `parse_json_reply` searches non-greedily and so returns the *first* `{...}`
    it can see, which is right for the one-flat-object answers the hyphen and
    line-end resolvers ask for and wrong for anything containing a list of
    objects: it would stop at the first inner closing brace and hand back
    invalid JSON. This one scans for balanced braces instead, ignoring braces
    inside strings, and is what `resolve_headings_llm.py` uses to read a whole
    span's worth of answers in one reply.

    Returns (data, '') or (None, reason), the same contract as its neighbour,
    so a bad answer stays a verdict about one question rather than an
    exception.
    """
    text = THINK_RE.sub('', reply or '')
    start = text.find('{')
    if start == -1:
        return None, f'no JSON object in reply: {(reply or "")[:120]}'

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                chunk = text[start:index + 1]
                try:
                    return json.loads(chunk), ''
                except json.JSONDecodeError as exc:
                    return None, f'invalid JSON ({exc}): {chunk[:120]}'
    return None, f'unterminated JSON object: {text[start:start + 120]}'


def load_prompt(filename: str,
                required: Tuple[str, ...] = ()) -> Dict[str, str]:
    """
    Read a sectioned prompt file from `data/prompts/`.

    The prompts are text rather than string literals so that they can be
    revised, versioned and diffed without touching the script that sends them —
    the same reason every other prompt in the project it came from does. A bare filename is
    read from `PATHS['prompts']`; a path with a directory in it is taken as
    given, so an experiment can be run against a file outside the store.

    `required` names the sections the caller cannot work without. They are
    checked here, at startup, because the alternative is a run of several
    thousand questions dying on the first one — or worse, sending a prompt with
    a section silently missing.

    Returns:
        {section name: text}, each stripped of its surrounding blank lines.
    """
    path = Path(filename)
    if not path.parent.name:
        path = PATHS['prompts'] / path
    if not path.exists():
        raise SystemExit(f"Prompt file not found: {path}")

    text = path.read_text(encoding='utf-8')
    matches = list(PROMPT_SECTION_RE.finditer(text))
    if not matches:
        raise SystemExit(
            f"{path} contains no sections. Each one is introduced by a line "
            f"reading '### <name>'; expected: {', '.join(required) or 'any'}.")

    sections: Dict[str, str] = {}
    for n, match in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end():end].strip('\n')

    missing = [name for name in required if not sections.get(name)]
    if missing:
        raise SystemExit(
            f"{path} is missing the section(s) {', '.join(missing)}.\n"
            f"Found: {', '.join(sections) or 'none'}.")
    return sections


def format_duration(seconds: float) -> str:
    """Render a run length as '3h 12m', '12m 30s' or '47s'."""
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes:02d}m'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'
