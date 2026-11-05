"""Step 4. Pick the k unique stimuli that best cover the generated set, with k-medoids.

k-medoids chooses k stimuli so that every unique stimulus is as close as possible (cosine distance) to its
nearest pick. Start: greedy facility location. Then alternate: assign each stimulus to its nearest pick,
re-pick each cluster's medoid, until nothing changes. As a check, the script compares the mean distance
to the nearest pick with that of 200 random sets of k.

    python scripts/select_stimuli.py --dry-run   # counts only
    python scripts/select_stimuli.py             # writes stimuli.json and selection.json
"""
import json

import numpy as np

from common import cosine_distances, embed, load_config, parser, read_jsonl, view


def kmedoids(D: np.ndarray, k: int) -> list[int]:
    picks, best = [], np.full(len(D), np.inf)
    for _ in range(k):  # greedy facility location
        cost = np.minimum(best[:, None], D).sum(0)
        cost[picks] = np.inf
        j = int(cost.argmin())
        picks.append(j)
        best = np.minimum(best, D[:, j])
    for _ in range(100):  # alternate: assign to the nearest pick, then re-pick each cluster's medoid
        labels = D[:, picks].argmin(1)
        new = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            new.append(int(idx[D[np.ix_(idx, idx)].sum(1).argmin()]) if len(idx) else picks[c])
        if sorted(new) == sorted(picks):
            break
        picks = new
    return picks


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline-embeddings", action="store_true", help="hashed bag of words; for testing only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["output_dir"]
    unique = [d for d in read_jsonl(out / "draws.jsonl") if not d["repeat"]]
    k = min(cfg["selection"]["k"], len(unique))
    if a.dry_run:
        print(f"{len(unique)} unique stimuli in {out / 'draws.jsonl'}; would pick k = {k}. "
              f"Embeddings are read from cache/embeddings/ (no Claude calls).")
        return
    X = embed(cfg, [view(cfg, "line", d) for d in unique], a.offline_embeddings)
    D = cosine_distances(X, X).clip(0)
    picks = kmedoids(D, k)
    near = D[:, picks].min(1)
    rng = np.random.default_rng(0)
    random_k = [D[:, rng.choice(len(unique), k, replace=False)].min(1).mean() for _ in range(200)]
    iu = np.triu_indices(k, 1)
    summary = {"n_unique": len(unique), "k": k, "mean_dist_to_pick": float(near.mean()),
               "random_k_mean_dist": float(np.mean(random_k)), "max_dist_to_pick": float(near.max()),
               "min_pairwise_picks": float(D[np.ix_(picks, picks)][iu].min()),
               "mean_pairwise_picks": float(D[np.ix_(picks, picks)][iu].mean()),
               "mean_pairwise_all": float(D[np.triu_indices(len(unique), 1)].mean())}
    fields = list(cfg["stimulus_fields"])
    stimuli = [{"stimulus": f"s{i + 1:03d}", "draw": unique[j]["id"], **{f: unique[j][f] for f in fields}}
               for i, j in enumerate(sorted(picks))]
    (out / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
