"""Steps 2-3. Generate new stimuli in batches and drop near-duplicates.

Each call sends the seeds and asks for `batch_size` new stimuli (prompts/generator.txt). Every new stimulus
is embedded and compared with the founders (seeds plus earlier unique stimuli). If its nearest founder is
closer than epsilon (from calibrate_epsilon.py), it is a repeat: it stays in draws.jsonl and joins that
founder's group, but it is not a new stimulus.

Two modes (generation.mode in the config):
  list         each call also sees every unique stimulus so far and is asked to differ from all of them.
               Draws are not independent, so Chao1 and coverage are printed for information only.
               Stop when the model runs dry: `dry_batches` batches in a row with fewer than `min_new` new.
  independent  each call sees only the seeds, so draws are independent and the species estimators hold.
               Stop when Good-Turing coverage reaches `target_coverage`.
Both modes also stop at `max_unique` unique stimuli or at the budget.

    python scripts/generate.py --dry-run          # one filled prompt, no API calls
    python scripts/generate.py --max-batches 150  # writes draws.jsonl and batches.jsonl (resumable)
"""
import json

import numpy as np

from common import append_jsonl, call, cfg_values, cosine_distances, embed, field_schema, fill, load_config, \
    parse_json, parser, read_jsonl, spent, template, view


def prompt(cfg: dict, unique: list[dict]) -> tuple[str, str]:
    g = cfg["generation"]
    system, user = template(cfg, "generator").split("=== USER ===\n")
    block = ""
    if g["mode"] == "list":
        listed = "\n".join(f"{i + 1}. {view(cfg, 'line', d)}" for i, d in enumerate(unique)) or "(none yet)"
        block = f"\nStimuli written so far ({len(unique)}):\n{listed}\n"
    values = {**cfg_values(cfg), "system": g["system"], "batch_size": g["batch_size"],
              "examples": "\n".join(f"- {view(cfg, 'line', s)}" for s in cfg["seeds"]),
              "running_list_block": block, "dimensions": g["dimensions"].strip(), "rules": g["rules"].strip(),
              "schema": field_schema(cfg["stimulus_fields"])}
    return fill(system.replace("=== SYSTEM ===\n", ""), values).strip(), fill(user, values).strip()


def richness(draws: list[dict]) -> dict:
    """Bias-corrected Chao1 and Good-Turing coverage over draws grouped by founder."""
    sizes = {}
    for d in draws:
        sizes[d["group"]] = sizes.get(d["group"], 0) + 1
    n, s_obs = len(draws), len(sizes)
    f1, f2 = sum(v == 1 for v in sizes.values()), sum(v == 2 for v in sizes.values())
    chao1 = s_obs + (n - 1) / n * f1 * (f1 - 1) / (2 * (f2 + 1)) if n else 0
    return {"n_draws": n, "groups": s_obs, "f1": f1, "f2": f2, "chao1": round(chao1, 1),
            "coverage": round(1 - f1 / n, 3) if n else None}


def run_batch(cfg: dict, b: int, eps: float, offline: bool) -> dict:
    path = cfg["output_dir"] / "draws.jsonl"
    draws = read_jsonl(path)
    unique = [d for d in draws if not d["repeat"]]
    system, user = prompt(cfg, unique)
    new = parse_json(call(cfg, "generator", system, user, step=f"generate:{b}", cache=False)["text"])["stimuli"]
    new = [d for d in new if all(k in d for k in cfg["stimulus_fields"])]
    founders = cfg["seeds"] + unique
    X_f = embed(cfg, [view(cfg, "line", d) for d in founders], offline)
    X_new = embed(cfg, [view(cfg, "line", d) for d in new], offline)
    rows = []
    for k, d in enumerate(new):
        dist = cosine_distances(X_new[k:k + 1], X_f)[0]
        j = int(dist.argmin())
        rid = f"d{len(draws) + k + 1:04d}"
        repeat = bool(dist[j] < eps)
        rows.append({"id": rid, "batch": b, **{f: d[f] for f in cfg["stimulus_fields"]}, "nn_dist": float(dist[j]),
                     "nn_id": founders[j]["id"], "repeat": repeat, "group": founders[j]["id"] if repeat else rid})
        if not repeat:  # later stimuli in the same batch are compared with this one too
            founders.append(rows[-1])
            X_f = np.vstack([X_f, X_new[k]])
    append_jsonl(path, rows)
    n_new = sum(not r["repeat"] for r in rows)
    stats = {"batch": b, "returned": len(new), "new_unique": n_new, "unique_total": len(unique) + n_new,
             "spent": round(spent(cfg), 2), **richness(draws + rows)}
    append_jsonl(cfg["output_dir"] / "batches.jsonl", [stats])
    return stats


def main():
    ap = parser(__doc__)
    ap.add_argument("--max-batches", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline-embeddings", action="store_true", help="hashed bag of words; for testing only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    g, out = cfg["generation"], cfg["output_dir"]
    if a.dry_run:
        system, user = prompt(cfg, [d for d in read_jsonl(out / "draws.jsonl") if not d["repeat"]])
        print(f"1 request per batch of {g['batch_size']} ({cfg['models']['generator']['name']}, mode {g['mode']}); "
              f"stops at {g['max_unique']} unique stimuli or ${cfg['budget_usd']}\n")
        print(f"--- system ---\n{system}\n--- user ---\n{user}")
        return
    eps = json.loads((out / "epsilon.json").read_text())["epsilon"]
    done, dry = len(read_jsonl(out / "batches.jsonl")), 0
    for b in range(done + 1, done + a.max_batches + 1):
        try:
            s = run_batch(cfg, b, eps, a.offline_embeddings)
        except Exception as e:  # a failed call is logged and skipped; the budget error ends the run
            append_jsonl(out / "failures.jsonl", [{"batch": b, "error": str(e)[:500]}])
            print(f"batch {b} failed: {e}")
            if "budget" in str(e):
                break
            continue
        print(json.dumps(s), flush=True)
        dry = dry + 1 if s["new_unique"] < g["min_new"] else 0
        if g["mode"] == "list" and dry >= g["dry_batches"]:
            print("stop: the generator ran dry"); break
        if g["mode"] == "independent" and s["coverage"] is not None and s["coverage"] >= g["target_coverage"]:
            print(f"stop: coverage {s['coverage']}"); break
        if s["unique_total"] >= g["max_unique"]:
            print(f"stop: {g['max_unique']} unique stimuli"); break


if __name__ == "__main__":
    main()
