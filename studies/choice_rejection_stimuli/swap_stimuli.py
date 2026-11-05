"""Replace dropped picks with their nearest unused unique stimulus (same stimulus id, so the deal is unchanged).
Dropped after the 2026-11-02 screen audit (output/screen_audit.json): garbled situations."""
import json

import numpy as np

from common import HERE, OUTPUT, embed, line

DROP = {"s010": "volunteer slides your change across before you decide",
        "s019": "sister phones you while standing next to you; quote in the wrong person",
        "s044": "decaf espresso 'would let you stay alert'",
        "s055": "'used ... cello lessons'"}
stimuli = json.loads((HERE / "stimuli.json").read_text())
unique = [d for d in (json.loads(x) for x in (OUTPUT / "draws.jsonl").read_text().splitlines()) if not d["repeat"]]
used = {s["id"] for s in stimuli}
X = embed([line(d) for d in unique])
index = {d["id"]: i for i, d in enumerate(unique)}
for n, s in enumerate(stimuli):
    if s["stimulus"] not in DROP or s.get("swapped_from"):
        continue
    dist = 1 - X @ X[index[s["id"]]]
    j = next(int(j) for j in np.argsort(dist) if unique[j]["id"] not in used)
    used.add(unique[j]["id"])
    new = {k: v for k, v in unique[j].items() if k not in ("repeat", "group", "nn_id")}
    stimuli[n] = {"stimulus": s["stimulus"], **new, "swapped_from": s["id"], "swap_reason": DROP[s["stimulus"]]}
    print(s["stimulus"], s["id"], "->", new["id"], f"(distance {dist[j]:.3f}):", new["situation"][:90])
(HERE / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
