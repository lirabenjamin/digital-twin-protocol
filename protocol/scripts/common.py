"""Shared helpers: config, prompt filling, JSON parsing, cached Claude calls, embeddings.

Every script imports this file. Nothing here is specific to one effect: the effect lives in the
config file (config/example_effect.yaml) and in the prompt templates it names.
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent  # protocol/: config, prompts and output paths are relative to it
REPO = ROOT.parent
CACHE = REPO / "cache"  # shared with the rest of the repository; gitignored
TWINS = REPO / "twins"  # written by code/download_twins.py
DEFAULT_CONFIG = ROOT / "config" / "example_effect.yaml"
MAX_TOKENS = 16000  # thinking counts toward this; a rating reply itself is a few tokens


# ---- config and prompts ----------------------------------------------------------------
def load_env(path: Path = REPO / ".env") -> None:
    """Read KEY=VALUE lines from .env. A variable already set in the shell wins."""
    for line in path.read_text().splitlines() if path.exists() else []:
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="effect config (YAML)")
    return ap


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["output_dir"] = ROOT / cfg["output_dir"]
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def template(cfg: dict, step: str) -> str:
    return (ROOT / cfg["prompts"][step]).read_text()


def fill(text: str, values: dict) -> str:
    """Replace {name} with values[name]. Unknown names stay as they are, so JSON examples in a prompt survive."""
    return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


def view(cfg: dict, name: str, stimulus: dict) -> str:
    """Render a stimulus with one of the config's views (e.g. 'line' for embeddings, 'full' for prompts)."""
    return fill(cfg["views"][name], {**cfg_values(cfg), **stimulus}).strip()


def cfg_values(cfg: dict) -> dict:
    """Effect-level text that any prompt may use."""
    return {"effect_description": cfg["effect"]["description"].strip(),
            "condition_names": " and ".join(c["name"] for c in cfg["conditions"])}


def field_schema(fields: dict) -> str:
    return json.dumps({k: f"<{v}>" for k, v in fields.items()})


def parse_json(text: str) -> dict:
    """The first JSON object in a reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.JSONDecoder().raw_decode(text[m.start():])[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


# ---- Claude calls ----------------------------------------------------------------------
def spent(cfg: dict) -> float:
    return sum(r["usd"] for r in read_jsonl(cfg["output_dir"] / "spend.jsonl"))


def call(cfg: dict, role: str, system: str, user: str, step: str, cache: bool = True, rep: int = 0) -> dict:
    """One Claude request with the model settings of `role` (see `models` in the config).

    cache=True saves the reply in cache/llm/<hash>.json and reads it back on a rerun, so a crash never pays
    twice. `rep` > 0 gives an identical request its own cache entry (for repeated ratings); it never reaches
    the API. Generation uses cache=False, because two identical generator prompts must give two draws.
    The system prompt is marked for Anthropic prompt caching, so a long persona is read from cache on
    later calls. Every call adds its cost to <output_dir>/spend.jsonl; calls stop at budget_usd.
    """
    m = cfg["models"][role]
    payload = {"model": m["name"], "effort": m["effort"], "system": system, "user": user, "rep": rep}
    path = CACHE / "llm" / f"{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    if spent(cfg) >= cfg["budget_usd"]:
        raise RuntimeError(f"budget reached: ${spent(cfg):.2f} of ${cfg['budget_usd']}")
    import anthropic  # imported here so dry runs need no SDK and no key
    load_env()
    msg = anthropic.Anthropic(max_retries=8).messages.create(
        model=m["name"], max_tokens=MAX_TOKENS, output_config={"effort": m["effort"]},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}])
    u = msg.usage
    usage = {"input": u.input_tokens, "output": u.output_tokens,
             "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0}
    out = {"text": "".join(b.text for b in msg.content if b.type == "text"), "stop_reason": msg.stop_reason,
           "usage": usage}
    append_jsonl(cfg["output_dir"] / "spend.jsonl", [{"time": time.time(), "step": step, **usage,
                                                      "usd": cost(m, usage)}])
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


def cost(model: dict, usage: dict, batch: bool = False) -> float:
    """Dollars for one call or a sum of calls. Prices per million tokens come from the config."""
    p = model["price"]
    usd = (usage["input"] * p["input"] + usage["output"] * p["output"]
           + usage["cache_write"] * p["cache_write"] + usage["cache_read"] * p["cache_read"]) / 1e6
    return usd * (0.5 if batch else 1)


# ---- embeddings ------------------------------------------------------------------------
def embed(cfg: dict, texts: list[str], offline: bool = False) -> np.ndarray:
    """Unit-length embeddings, one row per text, cached on disk by text.

    offline=True uses a hashed bag of words instead of the API. It is crude, but it lets you test the
    pipeline with no key and no network. Never use it for a real run.
    """
    if offline:
        X = np.zeros((len(texts), 512))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z']+", t.lower()):
                X[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    model = cfg["models"]["embedding"]["name"]
    folder = CACHE / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{hashlib.sha256(f'{model}|{t}'.encode()).hexdigest()[:32]}.json" for t in texts]
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        from openai import OpenAI
        load_env()
        resp = OpenAI(max_retries=8).embeddings.create(model=model, input=[texts[i] for i in missing])
        for i, item in zip(missing, resp.data):
            paths[i].write_text(json.dumps(item.embedding))
    X = np.array([json.loads(p.read_text()) for p in paths])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cosine_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 1 - A @ B.T
