"""Set EPSILON: 5 paraphrases of each seed; epsilon = 95th percentile of seed-to-own-paraphrase distance."""
import json

import numpy as np

from common import HERE, OUTPUT, SEEDS, call, cosine_distances, embed, line

PROMPT = (HERE / "prompts" / "paraphrase.txt").read_text()
SYSTEM = "You write stimuli for a consumer-behavior study about how other people's decisions influence our own."

def fill(text: str, s: dict) -> str:
    for k in ("situation", "source", "channel", "choice_text", "rejection_text"):
        text = text.replace("{" + k + "}", s[k])
    return text


rows = []
for s in SEEDS:
    reply = call(SYSTEM, fill(PROMPT, s),
                 step=f"calibrate:{s['study']}")
    rows += [{"seed": s["study"], **r} for r in reply["rewrites"]]
(OUTPUT / "paraphrases.json").write_text(json.dumps(rows, indent=1))

X_seeds, X_para = embed([line(s) for s in SEEDS]), embed([line(r) for r in rows])
ids = [s["study"] for s in SEEDS]
to_own = np.array([cosine_distances(X_seeds[ids.index(r["seed"])][None], X_para[i][None])[0, 0] for i, r in enumerate(rows)])
D = cosine_distances(X_seeds, X_seeds)
np.fill_diagonal(D, np.inf)
summary = {"epsilon": float(np.quantile(to_own, 0.95)), "n_paraphrases": len(rows),
           "paraphrase_median": float(np.median(to_own)), "paraphrase_max": float(to_own.max()),
           "seed_nn_min": float(D.min()), "seed_nn_median": float(np.median(D.min(1)))}
(OUTPUT / "epsilon.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
