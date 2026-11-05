"""Generate choice/rejection stimuli, 10 per call, each call conditioned on the running list.

No twins. Each call sends the 7 seeds plus every UNIQUE stimulus so far and asks for 10 new ones that
differ as much as possible from all of them. Each new stimulus is embedded and compared with the
founders (seeds + earlier unique stimuli). Within EPSILON of a founder = repeat: it stays in draws.jsonl
(and counts toward Chao1) but is left out of the running list.

Chao1 and Good-Turing assume independent draws; these draws are not independent (each sees the list),
so the estimates are reported for curiosity only.

Stop: budget ($40, calibration included), 1,000 unique stimuli, or 3 batches in a row with < 3 new.

    python3 generate.py --dry-run
    python3 generate.py --max-batches 150
"""
import argparse
import json
import sys

import numpy as np

from common import BUDGET, HERE, OUTPUT, SEEDS, append_lines, call, cosine_distances, embed, line, spent

DRAWS, BATCHES = OUTPUT / "draws.jsonl", OUTPUT / "batches.jsonl"
SYSTEM, USER = (HERE / "prompts" / "generator.txt").read_text().split("=== USER ===\n")
SYSTEM = SYSTEM.replace("=== SYSTEM ===\n", "").strip()
FIELDS = ("situation", "category", "option_a", "option_b", "source", "channel", "choice_text", "rejection_text")
MAX_UNIQUE, DRY_RUNS, MIN_NEW = 1000, 3, 3


def epsilon() -> float:
    return json.loads((OUTPUT / "epsilon.json").read_text())["epsilon"]


def pool() -> list[dict]:
    return [json.loads(x) for x in DRAWS.read_text().splitlines()] if DRAWS.exists() else []


def prompt(unique: list[dict], batch: int) -> str:
    listed = "\n".join(f"{i + 1}. {line(d)}" for i, d in enumerate(unique)) or "(none yet)"
    return (USER.replace("{examples}", "\n".join(f"- {line(s)}" for s in SEEDS))
                .replace("{n_so_far}", str(len(unique))).replace("{running_list}", listed)
                .replace("{batch_size}", str(batch)))


def richness(draws: list[dict]) -> dict:
    """Chao1 (bias-corrected) and Good-Turing coverage over draws grouped by founder."""
    sizes = {}
    for d in draws:
        sizes[d["group"]] = sizes.get(d["group"], 0) + 1
    n, s_obs = len(draws), len(sizes)
    f1 = sum(v == 1 for v in sizes.values())
    f2 = sum(v == 2 for v in sizes.values())
    chao1 = s_obs + f1 * (f1 - 1) / (2 * (f2 + 1))
    return {"n_draws": n, "groups": s_obs, "f1": f1, "f2": f2, "chao1": chao1,
            "share_found": s_obs / chao1 if chao1 else None, "coverage": 1 - f1 / n if n else None}


def run_batch(batch: int, b: int) -> dict:
    draws = pool()
    unique = [d for d in draws if not d["repeat"]]
    new = call(SYSTEM, prompt(unique, batch), step=f"generate:{b}")["stimuli"]
    new = [d for d in new if all(k in d for k in FIELDS)]
    founders = [{"id": f"seed{s['study']}", **s} for s in SEEDS] + unique
    X_f = embed([line(d) for d in founders])
    X_new = embed([line(d) for d in new])
    eps, rows = epsilon(), []
    for k, d in enumerate(new):
        dist = cosine_distances(X_new[k:k + 1], X_f)[0]
        j = int(dist.argmin())
        rid = f"d{len(draws) + k + 1:04d}"
        repeat = bool(dist[j] < eps)
        rows.append({"id": rid, "batch": b, **{f: d[f] for f in FIELDS}, "nn_dist": float(dist[j]),
                     "nn_id": founders[j]["id"], "repeat": repeat, "group": founders[j]["id"] if repeat else rid})
        if not repeat:  # later stimuli in the same batch are compared with this one too
            founders.append(rows[-1])
            X_f = np.vstack([X_f, X_new[k]])
    append_lines(DRAWS, rows)
    stats = {"batch": b, "returned": len(new), "new_unique": sum(not r["repeat"] for r in rows),
             "unique_total": len(unique) + sum(not r["repeat"] for r in rows),
             "mean_nn": float(np.mean([r["nn_dist"] for r in rows])) if rows else None,
             "spent": round(spent(), 3), **richness(draws + rows)}
    append_lines(BATCHES, [stats])
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=1)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(SYSTEM, "\n---\n", prompt([d for d in pool() if not d["repeat"]], a.batch))
        sys.exit()
    done = len(BATCHES.read_text().splitlines()) if BATCHES.exists() else 0
    dry = 0
    for b in range(done + 1, done + a.max_batches + 1):
        if spent() >= BUDGET - 1.0:  # keep $1 headroom: one call costs well under that
            print(f"stop: budget (${spent():.2f})"); break
        try:
            s = run_batch(a.batch, b)
        except Exception as e:  # a failed call is logged and skipped
            append_lines(OUTPUT / "failures.jsonl", [{"batch": b, "error": str(e)[:500]}])
            print(f"batch {b} failed: {e}"); continue
        print(json.dumps(s), flush=True)
        dry = dry + 1 if s["new_unique"] < MIN_NEW else 0
        if dry >= DRY_RUNS:
            print("stop: generator ran dry"); break
        if s["unique_total"] >= MAX_UNIQUE:
            print("stop: 1,000 unique"); break
