"""Shared pieces: stimulus text, generation, embeddings, and coverage metrics.

Generation uses Claude (one uncached call per draw, so repeated draws are
independent). Embeddings use OpenAI text-embedding-3-large, cached on disk by
text in cache/embeddings/, so a stimulus is never embedded twice.

Coverage: two stimuli are the same "species" when the cosine distance between
their full 3,072-dimensional embeddings is below EPSILON (set by
calibrate_epsilon.py from paraphrase distances). `walk` replays the draws in
order with leader clustering. All metrics use the raw embeddings; PCA and UMAP
are for the map only. Seeds are not draws; they appear on the map and in "seeds
recovered".
"""
import fcntl
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import llm  # noqa: E402  (loads the repo .env)

EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "claude-haiku-5-5"      # draws: full twin in the prompt, thinking off
PARAPHRASE_MODEL = "claude-sonnet-5-5"  # epsilon calibration (calibrate_epsilon.py)
EMBED_CACHE = ROOT / "cache" / "embeddings"
OUTPUT = HERE / "output"
DRAWS = OUTPUT / "draws.jsonl"
FAILURES = OUTPUT / "failures.jsonl"
SEEDS = json.loads((HERE / "seeds.json").read_text()) if (HERE / "seeds.json").exists() else []
GEN_SYSTEM, GEN_USER = (HERE / "prompts" / "generator.txt").read_text().removeprefix("=== SYSTEM ===\n").split("\n=== USER ===\n")
TWINS = ROOT / "twins"


def pair_text(options: list[dict]) -> str:
    """Render options as "x for $y or w for z" (three options: "a, b, or c")."""
    parts = [f"{o['label'][0].lower() + o['label'][1:]} for ${o['cost']}" for o in options]
    text = ", ".join(parts[:-1]) + f", or {parts[-1]}" if len(parts) > 2 else " or ".join(parts)
    return text[0].upper() + text[1:]


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:  # a JSON object followed by more text: keep the first object
        return json.JSONDecoder().raw_decode(text[match.start():])[0]


def clean_label(label: str) -> str:
    """Drop a trailing "for $N" the model sometimes puts in the label; pair_text adds the cost."""
    return re.sub(r"\s+for \$[\d,]+(\.\d+)?\s*$", "", label.strip())


# ---- embeddings --------------------------------------------------------------------
def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, one row per text; only uncached texts hit the API."""
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    paths = [EMBED_CACHE / f"{hashlib.sha256(f'{EMBED_MODEL}|{t}'.encode()).hexdigest()[:32]}.json" for t in texts]
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        from openai import OpenAI
        response = OpenAI(max_retries=8).embeddings.create(model=EMBED_MODEL, input=[texts[i] for i in missing])
        for i, item in zip(missing, response.data):
            paths[i].write_text(json.dumps({"text": texts[i], "model": EMBED_MODEL, "embedding": item.embedding}))
    X = np.array([json.loads(p.read_text())["embedding"] for p in paths])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cosine_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 1 - A @ B.T


# ---- generation --------------------------------------------------------------------
def generate() -> dict:
    """One independent draw: a random Twin-2K-500 twin, the same prompt every time, never cached."""
    import random
    import anthropic
    pid = random.choice(sorted(int(f.stem) for f in TWINS.glob("*.txt")))
    examples = "\n".join(f"- {s['text']}" for s in SEEDS)
    msg = anthropic.Anthropic(max_retries=8).messages.create(
        model=GEN_MODEL, max_tokens=2000, thinking={"type": "disabled"},
        system=GEN_SYSTEM.replace("{persona}", (TWINS / f"{pid}.txt").read_text()),
        messages=[{"role": "user", "content": GEN_USER.replace("{examples}", examples)}],
    )
    reply = "".join(b.text for b in msg.content if b.type == "text")
    draw = parse_json(reply)
    options = [{"label": clean_label(o["label"]), "cost": int(o["cost"])} for o in draw["options"]]
    if len(options) != 2 or options[0]["cost"] <= options[1]["cost"]:
        raise ValueError(f"need two options, expensive first: {reply[:300]}")
    return {"text": pair_text(options), "model_text": draw.get("text", ""), "scenario": draw.get("scenario", ""),
            "options": options, "model": GEN_MODEL, "pid": pid,
            "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}}


def safe_draw(_=None) -> dict | None:
    """generate() + embed; a failed draw is logged to FAILURES and returns None (never retried into the pool)."""
    try:
        d = generate()
        embed([d["text"]])
        return d
    except Exception as e:
        append_lines(FAILURES, [{"time": time.time(), "error": str(e)[:500]}])
        return None


def append_lines(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # the app and batch.py may write at the same time
        f.writelines(json.dumps(r) + "\n" for r in rows)


def append_draws(draws: list[dict]) -> None:
    """Number the draws after the ones already in the pool, under one lock, and append them."""
    OUTPUT.mkdir(exist_ok=True)
    with DRAWS.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        start = sum(1 for _ in f)
        f.writelines(json.dumps({"id": f"draw{start + k + 1:04d}", "time": time.time(), **d}) + "\n"
                     for k, d in enumerate(draws))


# ---- coverage metrics --------------------------------------------------------------
def coverage(n: int, f1: int, f2: int) -> float:
    """Sample coverage (Chao & Jost 2012): Good-Turing 1 - f1/n with a small f2 correction."""
    if (n - 1) * f1 + 2 * f2 == 0:
        return 1 - f1 / n
    return 1 - f1 / n * (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)


def richness(sizes: list[int]) -> dict:
    """Estimators of the total number of groups from abundance data (draws per group).

    chao1     bias-corrected Chao1 (Chao 1987; Chao 2005 form with (n-1)/n): uses f1, f2.
    ichao1    improved Chao1 (Chiu, Wang, Walther & Chao 2014): adds f3, f4; corrects Chao1's downward bias.
    ace       abundance-based coverage estimator (Chao & Lee 1992), rare groups = 10 draws or fewer.
    jack1/2   first- and second-order jackknife (Burnham & Overton 1978/79, abundance form).
    Chao2 and ICE are the incidence-data versions; with one draw per sampling unit they reduce to
    Chao1 and ACE, so they are not computed separately.
    """
    sz = np.asarray(sizes)
    n, S = int(sz.sum()), len(sz)
    f = {k: int((sz == k).sum()) for k in (1, 2, 3, 4)}
    f1, f2, f3, f4 = f[1], f[2], f[3], f[4]
    a = (n - 1) / n
    chao1 = S + a * f1 * (f1 - 1) / (2 * (f2 + 1))
    ichao1 = chao1 + f3 / (4 * max(f4, 1)) * max(f1 - f2 * f3 / (2 * max(f4, 1)), 0)
    rare = sz[sz <= 10]
    n_rare, s_rare, s_abund = int(rare.sum()), len(rare), S - len(rare)
    c_ace = 1 - f1 / n_rare if n_rare else 1
    if c_ace > 0 and n_rare > 1:
        gamma2 = max(s_rare / c_ace * sum(k * (k - 1) * int((rare == k).sum()) for k in range(1, 11))
                     / (n_rare * (n_rare - 1)) - 1, 0)
        ace = s_abund + s_rare / c_ace + f1 / c_ace * gamma2
    else:
        ace = float("nan")
    jack1 = S + f1 * a
    jack2 = S + f1 * (2 * n - 3) / n - f2 * (n - 2) ** 2 / (n * (n - 1))
    return {"S": S, "n": n, "f1": f1, "f2": f2, "chao1": chao1, "ichao1": ichao1, "ace": ace,
            "jack1": jack1, "jack2": jack2}


def walk(X: np.ndarray, X_seeds: np.ndarray, epsilon: float, window: int = 20) -> list[dict]:
    """Replay the draws in arrival order and record, for each one, what it did to the counts.

    Leader clustering: a draw is compared with the FOUNDER of every group. If the
    nearest founder is closer than epsilon, the draw is a repeat and joins that
    group; otherwise it founds a new group. Founders are therefore at least epsilon
    apart, and they form the unique stimulus set. Repeats stay in the counts:
    they are what makes doubletons, Chao1 and coverage possible.
    """
    steps, founders, group_of, sizes, new_flags, nn_dists = [], [], [], [], [], []
    pair_sum, seed_min = 0.0, np.full(len(X_seeds), np.inf)
    D, D_seeds = cosine_distances(X, X), cosine_distances(X_seeds, X)  # computed once
    for i in range(len(X)):
        d_draws = D[i, :i]
        nn_draw = int(d_draws.argmin()) if i else None
        d_found = d_draws[founders] if founders else np.array([])
        near = int(d_found.argmin()) if founders else None
        joined = near is not None and d_found[near] < epsilon
        if joined:
            group = group_of[founders[near]]
            sizes[group] += 1
        else:
            group = len(founders)
            founders.append(i)
            sizes.append(1)
        group_of.append(group)
        new_flags.append(not joined)
        if i:
            nn_dists.append(float(d_draws[nn_draw]))
        pair_sum += float(d_draws.sum())
        d_seeds = D_seeds[:, i]
        seed_min = np.minimum(seed_min, d_seeds)
        sz = np.array(sizes)
        s_obs, f1, f2, n = len(sz), int((sz == 1).sum()), int((sz == 2).sum()), i + 1
        est = s_obs + (n - 1) / n * f1 * (f1 - 1) / (2 * (f2 + 1))  # bias-corrected Chao1, as in richness()
        steps.append({
            "draw": i, "nn_draw": nn_draw, "nn_dist": None if nn_draw is None else float(d_draws[nn_draw]),
            "near_founder": None if near is None else founders[near],
            "near_founder_dist": None if near is None else float(d_found[near]),
            "joined": bool(joined), "group": group, "group_size": sizes[group], "founder": founders[group],
            "nearest_seed": int(d_seeds.argmin()), "nearest_seed_dist": float(d_seeds.min()),
            "n": n, "s_obs": s_obs, "f1": f1, "f2": f2, "chao1": est, "completeness": s_obs / est,
            "coverage": coverage(n, f1, f2),
            "novelty_rate": float(np.mean(new_flags[-window:])), "window": min(window, n),
            "mean_nn_recent": float(np.mean(nn_dists[-window:])) if nn_dists else None,
            "mean_pairwise": pair_sum / (n * (n - 1) / 2) if n > 1 else None,
            "seeds_recovered": int((seed_min < epsilon).sum()),
        })
    return steps


def umap_2d(X: np.ndarray) -> np.ndarray:
    """UMAP on cosine distance; fixed seed so the same points give the same layout."""
    import warnings
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "n_jobs overridden by random_state": expected
        return umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1, metric="cosine",
                         random_state=0).fit_transform(X)


def pca_2d(X: np.ndarray, anchor: int = 0) -> np.ndarray:
    """First two principal components, signs fixed so the anchor point (seed01) stays in the upper right."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    return P * np.where(P[anchor] < 0, -1, 1)
