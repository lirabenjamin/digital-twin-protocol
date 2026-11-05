"""Pick k stimuli that span the unique set (the group founders at a given epsilon).

Methods (all on cosine distance between full embeddings):
  kmedoids   k-medoids: choose k founders so every founder is as close as possible to its
             nearest pick. Greedy facility-location start, then alternate assign / re-pick
             medoid until stable.
  kmeans     k-means clusters, then the medoid of each cluster.
  hdbscan    HDBSCAN on a 10-d UMAP of the founders (density-based; it chooses the number
             of clusters itself and leaves "noise" founders unassigned), then each cluster's medoid.

weighted=False counts every distinct situation once; weighted=True weights each founder
by the number of draws in its group, so common situations pull the picks toward them.
"""
import os
import warnings

import numpy as np

from stimuli import cosine_distances

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")  # silences a joblib core-count warning on macOS


def _medoids(D: np.ndarray, labels: np.ndarray, w: np.ndarray) -> list[int]:
    out = []
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        out.append(int(idx[np.argmin((D[np.ix_(idx, idx)] * w[idx]).sum(1))]))
    return out


def _kmedoids(D: np.ndarray, k: int, w: np.ndarray) -> list[int]:
    sel, best = [], np.full(len(D), np.inf)
    for _ in range(k):  # greedy facility location
        cost = (w[:, None] * np.minimum(best[:, None], D)).sum(0)
        cost[sel] = np.inf
        j = int(cost.argmin())
        sel.append(j)
        best = np.minimum(best, D[:, j])
    for _ in range(100):  # alternate: assign to nearest pick, re-pick each cluster's medoid
        new = _medoids(D, np.argmin(D[:, sel], 1), w)
        if sorted(new) == sorted(sel):
            break
        sel = new
    return sel


def select(XF: np.ndarray, w: np.ndarray, k: int, method: str = "kmedoids", weighted: bool = False,
           min_cluster_size: int = 5) -> dict:
    """Return picks (indices into XF), each founder's cluster (index into picks, -1 = noise), and fit stats."""
    D = cosine_distances(XF, XF).clip(0)
    ww = w.astype(float) if weighted else np.ones(len(XF))
    if method == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(k, n_init=10, random_state=0).fit(XF, sample_weight=ww).labels_
        picks = _medoids(D, labels, ww)
    elif method == "hdbscan":
        import umap
        from sklearn.cluster import HDBSCAN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine",
                          random_state=0).fit_transform(XF)
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit(U).labels_
        picks = _medoids(D, labels, ww)
    else:
        picks = _kmedoids(D, k, ww)
    near = np.argmin(D[:, picks], 1)
    cluster = near if method != "hdbscan" else np.where(labels < 0, -1, near)
    dist = D[np.arange(len(XF)), np.array(picks)[near]]
    iu = np.triu_indices(len(picks), 1)
    return {
        "picks": picks, "cluster": cluster.tolist(), "dist_to_pick": dist.tolist(),
        "mean_dist_unweighted": float(dist.mean()), "mean_dist_weighted": float((w * dist).sum() / w.sum()),
        "min_pairwise": float(D[np.ix_(picks, picks)][iu].min()) if len(picks) > 1 else None,
        "mean_pairwise": float(D[np.ix_(picks, picks)][iu].mean()) if len(picks) > 1 else None,
        "noise": int((cluster == -1).sum()),
    }


def representatives(draws: list[dict], X: np.ndarray, steps: list[dict], cap: int | None) -> tuple[list[int], np.ndarray, int]:
    """One draw per group. Without a cap it is the founder. With a cap on the expensive option's
    price, it is the group member under the cap that is closest to the founder; groups with no
    member under the cap are dropped. Returns draw indices, draws per kept group, groups dropped."""
    groups = {}
    for s in steps:
        groups.setdefault(s["group"], []).append(s["draw"])
    reps, sizes = [], []
    for members in groups.values():  # members[0] is the founder
        ok = [i for i in members if not cap or draws[i]["options"][0]["cost"] <= cap]
        if ok:
            reps.append(min(ok, key=lambda i: float(1 - X[i] @ X[members[0]])))
            sizes.append(len(members))
    return reps, np.array(sizes), len(groups) - len(reps)
