"""One thin wrapper around the Anthropic Messages API, with an on-disk cache.

Why a cache: a twin's conversation is replayed turn by turn, so a crash at turn 12
would otherwise pay again for turns 1-11. Every request is hashed (model, effort,
system, messages, max_tokens) and its response is saved to cache/llm/<hash>.json.
Re-running the same request reads the file instead of calling the API.

Why two cache_control markers: the persona sits in the system prompt and is ~40K
tokens. Marking the system block lets a twin's second condition reuse it, and the
top-level marker caches the growing conversation turn by turn. Both are Anthropic
prompt caching (5-minute TTL), separate from the on-disk cache above.

Why we return the raw content blocks: Claude Haiku 5.5 thinks before answering.
Its thinking blocks must be sent back unchanged on the next turn, so the caller
appends `content` (not just the text) to the conversation.

The API key comes from the ANTHROPIC_API_KEY environment variable. At import, the
repo-root .env (gitignored) is loaded; a variable already set in the shell wins.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "llm"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
    except ImportError:
        for line in path.read_text().splitlines() if path.exists() else []:
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("'\""))


load_env()
MAX_TOKENS = 16000  # thinking counts toward this; a reply itself is a few tokens


def request_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def call(model: str, effort: str, system: str, messages: list[dict], rep: int = 0) -> dict:
    """Send one request; return {"content", "text", "stop_reason", "usage"}.

    `rep` > 0 gives an identical request its own cache entry, for repeated samples.
    It enters only the cache key, never the request sent to the API.
    """
    payload = {"model": model, "effort": effort, "system": system,
               "messages": messages, "max_tokens": MAX_TOKENS}
    if rep:
        payload["rep"] = rep
    path = CACHE / f"{request_key(payload)}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]

    import anthropic  # imported here so dry runs work without the SDK configured

    msg = anthropic.Anthropic(max_retries=8).messages.create(  # SDK retries 429/5xx with backoff
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": effort},
        cache_control={"type": "ephemeral"},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    u = msg.usage
    response = {
        "content": [block.model_dump(exclude_none=True) for block in msg.content],
        "text": "".join(block.text for block in msg.content if block.type == "text"),
        "stop_reason": msg.stop_reason,
        "usage": {"input": u.input_tokens, "output": u.output_tokens,
                  "cache_write": u.cache_creation_input_tokens or 0,
                  "cache_read": u.cache_read_input_tokens or 0},
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # write then rename, so a crash never leaves a half-written cache file
    tmp.write_text(json.dumps({"request": payload, "response": response}))
    tmp.replace(path)
    return response
