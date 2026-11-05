"""Step 3a. Set epsilon, the distance below which two stimuli count as the same stimulus.

Claude rewrites each seed n times in new words (prompts/paraphrase.txt). Epsilon is a high quantile
(default 95th) of the cosine distance between each seed and its own rewrites: a new stimulus closer than
that to an earlier one differs from it no more than a paraphrase does. For contrast, the script also
reports each seed's distance to its nearest other seed.

    python scripts/calibrate_epsilon.py --dry-run   # request count and one filled prompt
    python scripts/calibrate_epsilon.py             # writes paraphrases.json and epsilon.json
"""
import json

import numpy as np

from common import call, cfg_values, cosine_distances, embed, field_schema, fill, load_config, parse_json, parser, \
    template, view


def prompt(cfg: dict, seed: dict) -> str:
    e = cfg["epsilon"]
    schema = field_schema({f: cfg["stimulus_fields"][f] for f in e["fields"]})
    return fill(template(cfg, "paraphrase"), {**cfg_values(cfg), "stimulus": view(cfg, "full", seed),
                                              "n": e["n_paraphrases"], "keep": e["keep"], "schema": schema})


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds, out = cfg["seeds"], cfg["output_dir"]
    if a.dry_run:
        print(f"{len(seeds)} requests ({cfg['models']['generator']['name']}), "
              f"{len(seeds) * cfg['epsilon']['n_paraphrases']} paraphrases to embed\n")
        print(f"--- system ---\n{cfg['generation']['system']}\n--- user (seed 1) ---\n{prompt(cfg, seeds[0])}")
        return
    rows = []
    for s in seeds:
        reply = parse_json(call(cfg, "generator", cfg["generation"]["system"], prompt(cfg, s), step="calibrate")["text"])
        rows += [{**s, **r, "seed": s["id"]} for r in reply["rewrites"]]  # unchanged fields come from the seed
    (out / "paraphrases.json").write_text(json.dumps(rows, indent=1))

    X_seed = embed(cfg, [view(cfg, "line", s) for s in seeds])
    X_para = embed(cfg, [view(cfg, "line", r) for r in rows])
    ids = [s["id"] for s in seeds]
    to_own = np.array([1 - X_seed[ids.index(r["seed"])] @ X_para[i] for i, r in enumerate(rows)])
    D = cosine_distances(X_seed, X_seed)
    np.fill_diagonal(D, np.inf)
    eps = float(np.quantile(to_own, cfg["epsilon"]["quantile"]))
    summary = {"epsilon": eps, "quantile": cfg["epsilon"]["quantile"], "n_paraphrases": len(rows),
               "paraphrase_median": float(np.median(to_own)), "paraphrase_max": float(to_own.max()),
               "seed_nn_min": float(D.min()), "seed_nn_median": float(np.median(D.min(1))),
               "seeds_closer_than_epsilon": int((D.min(1) < eps).sum())}
    (out / "epsilon.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
