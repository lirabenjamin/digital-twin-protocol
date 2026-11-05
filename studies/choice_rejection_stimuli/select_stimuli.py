"""Pick the 100 unique stimuli that best cover the space: k-medoids (greedy facility-location start,
then alternate) on cosine distances between the unique stimuli. Writes stimuli.json and a coverage plot."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import HERE, OUTPUT, cosine_distances, embed, line  # noqa: E402

sys.path.insert(0, str(HERE.parent / "stimulus_augmentation"))
from selection import _kmedoids  # noqa: E402

K = 100
draws = [json.loads(x) for x in (OUTPUT / "draws.jsonl").read_text().splitlines()]
unique = [d for d in draws if not d["repeat"]]
X = embed([line(d) for d in unique])
D = cosine_distances(X, X).clip(0)
picks = _kmedoids(D, K, np.ones(len(unique)))
near = D[:, picks].min(1)
rnd = np.random.default_rng(0)
rand_cover = [D[:, rnd.choice(len(unique), K, replace=False)].min(1).mean() for _ in range(200)]
iu = np.triu_indices(K, 1)
summary = {"n_unique": len(unique), "k": K, "mean_dist_to_pick": float(near.mean()),
           "random_100_mean_dist": float(np.mean(rand_cover)), "max_dist_to_pick": float(near.max()),
           "min_pairwise_picks": float(D[np.ix_(picks, picks)][iu].min()),
           "mean_pairwise_picks": float(D[np.ix_(picks, picks)][iu].mean()),
           "mean_pairwise_all": float(D[np.triu_indices(len(unique), 1)].mean())}
out = [{"stimulus": f"s{i + 1:03d}", **{k: v for k, v in unique[j].items() if k not in ("repeat", "group", "nn_id")}}
       for i, j in enumerate(sorted(picks))]
(HERE / "stimuli.json").write_text(json.dumps(out, indent=1))
(OUTPUT / "selection.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))

batches = [json.loads(x) for x in (OUTPUT / "batches.jsonl").read_text().splitlines()]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
n = [b["n_draws"] for b in batches]
ax[0].plot(n, [b["unique_total"] for b in batches], color="#2a78d6", label="Unique stimuli found")
ax[0].plot(n, [b["chao1"] for b in batches], color="#eb6834", label="Chao1 (assumption violated)")
ax[0].set_xlabel("Stimuli generated"); ax[0].set_ylabel("Count"); ax[0].legend(frameon=False)
ax[1].plot([b["batch"] for b in batches], [b["new_unique"] for b in batches], color="#2a78d6")
ax[1].set_xlabel("Batch"); ax[1].set_ylabel("New unique stimuli (of 10)"); ax[1].set_ylim(0, 10.5)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT / "generation_curve.png", dpi=150)
