"""Exhaustive opposite-tertile matching with transparent contrast/balance objectives.

All distances use the original 0--10 ratings. Outcomes and PCA coordinates never
enter the search. Correlation eigenvalues above one discount shared movement;
inverse directional weights are capped at one so rare directions are not
amplified. This proposed geometry is not ordinary Mahalanobis distance or a
validated measure of semantic redundancy. Raw feature differences remain visible.
"""
import numpy as np
import pandas as pd


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def correlation_weights(correlation):
    """Return Q diag(1/max(lambda, 1)) Q' for a valid correlation matrix."""
    r = np.asarray(correlation, dtype=float)
    _require(r.ndim == 2 and r.shape[0] == r.shape[1] and len(r) > 0, "Correlation must be nonempty and square")
    _require(np.isfinite(r).all(), "Correlation must be finite")
    _require(np.allclose(r, r.T, atol=1e-10) and np.allclose(np.diag(r), 1), "Invalid correlation symmetry or diagonal")
    eigenvalues, eigenvectors = np.linalg.eigh(r)
    _require(eigenvalues.min() >= -1e-8, "Correlation must be positive semidefinite")
    directional_weights = 1 / np.maximum(eigenvalues, 1)
    return (eigenvectors * directional_weights) @ eigenvectors.T


def balance_geometry(reference):
    """Fit once to all reference stimuli's non-focal columns; constants are orthogonal."""
    x = np.asarray(reference, dtype=float)
    _require(x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] >= 1 and np.isfinite(x).all(),
             "Need finite reference data with at least two rows and one column")
    centered = x - x.mean(axis=0)
    lengths = np.linalg.norm(centered, axis=0)
    normalized = np.divide(centered, lengths, out=np.zeros_like(centered), where=lengths > 0)
    correlation = normalized.T @ normalized
    np.fill_diagonal(correlation, 1)
    eigenvalues = np.linalg.eigvalsh(correlation)
    return {"correlation": correlation, "weights": correlation_weights(correlation),
            "eigenvalues": eigenvalues, "directional_weights": 1 / np.maximum(eigenvalues, 1)}


def adjusted_distance(differences, weights):
    """Joint point-distance; no division by number of features or sample SDs."""
    delta, w = np.asarray(differences, dtype=float), np.asarray(weights, dtype=float)
    _require(delta.ndim == 2 and w.shape == (delta.shape[1], delta.shape[1]), "Difference/weight dimensions do not match")
    _require(np.isfinite(delta).all() and np.isfinite(w).all(), "Distance inputs must be finite")
    squared = np.einsum("ij,jk,ik->i", delta, w, delta)
    _require((squared >= -1e-9).all(), "Squared distances cannot be negative")
    return np.sqrt(np.maximum(squared, 0))


def _validate(features, focal, match, prices):
    _require(len(features) >= 2 and features.index.is_unique, "Need at least two unique stimulus IDs")
    _require(features.index.notna().all(), "Stimulus IDs cannot be missing")
    _require(all(isinstance(s, str) and s for s in features.index), "Stimulus IDs must be nonempty strings")
    _require(features.columns.is_unique, "Feature names must be unique")
    _require(focal in features, "Unknown focal feature")
    _require(len(match) > 0 and len(set(match)) == len(match), "Matching features must be nonempty and unique")
    _require(focal not in match and set(match) <= set(features), "Matching features must be known and nonfocal")
    x = features.to_numpy(float)
    _require(np.isfinite(x).all() and ((0 <= x) & (x <= 10)).all(), "Ratings must be finite and in 0--10")
    _require(prices.index.is_unique and set(prices.index) == set(features.index), "Prices must join one-to-one with exactly the stimulus IDs")
    _require({"expensive", "cheaper"} <= set(prices), "Missing price columns")
    p = prices[["expensive", "cheaper"]].to_numpy(float)
    _require(np.isfinite(p).all() and (p[:, 1] >= 0).all() and (p[:, 0] > p[:, 1]).all(), "Prices must be finite with 0 <= cheaper < expensive")


def pair_table(features, focal, match, prices):
    """Enumerate n(n-1)/2 pairs, oriented by focal score; ties use sorted IDs."""
    _validate(features, focal, match, prices)
    features = features.sort_index()
    prices = prices.loc[features.index]
    values = features[focal].to_numpy(float)
    a, b = np.triu_indices(len(features), 1)
    low = np.where(values[a] <= values[b], a, b)
    high = np.where(values[a] <= values[b], b, a)
    nonfocal = features[match].to_numpy(float)
    signed = nonfocal[high] - nonfocal[low]
    deltas = np.abs(signed)
    geometry = balance_geometry(nonfocal)
    adjusted = adjusted_distance(signed, geometry["weights"])
    raw_squared = np.sum(signed ** 2, axis=1)
    retained = np.divide(adjusted ** 2, raw_squared, out=np.ones_like(adjusted), where=raw_squared > 0)
    expensive = prices.expensive.to_numpy(float)
    increment = (prices.expensive - prices.cheaper).to_numpy(float)
    free = prices.cheaper.eq(0).to_numpy()
    qlow, qhigh = np.quantile(values, [1 / 3, 2 / 3])
    gap = values[high] - values[low]
    sd = np.std(values, ddof=1)
    table = pd.DataFrame({
        "low_id": features.index.to_numpy()[low], "high_id": features.index.to_numpy()[high],
        "low_score": values[low], "high_score": values[high], "focal_gap": gap,
        "focal_sd_gap": gap / sd if sd > 0 else np.zeros_like(gap),
        "tertile_low_cutoff": qlow, "tertile_high_cutoff": qhigh,
        "other_adjusted": adjusted,
        "other_mean_abs": deltas.mean(axis=1), "other_sum_abs": deltas.sum(axis=1),
        "other_max": deltas.max(axis=1), "other_rms": np.sqrt(np.mean(deltas ** 2, axis=1)),
        "other_redundancy_discount": np.clip(1 - retained, 0, 1),
        "other_count_ge1": (deltas >= 1 - 1e-10).sum(axis=1),
        "other_count_ge2": (deltas >= 2 - 1e-10).sum(axis=1),
        "worst_feature": np.asarray(match)[deltas.argmax(axis=1)],
        "expensive_ratio": np.maximum(expensive[a], expensive[b]) / np.minimum(expensive[a], expensive[b]),
        "price_gap_ratio": np.maximum(increment[a], increment[b]) / np.minimum(increment[a], increment[b]),
        "free_option_match": free[a] == free[b],
        "opposite_tertiles": (values[low] <= qlow + 1e-10) & (values[high] >= qhigh - 1e-10),
    })
    return table.sort_values(["low_id", "high_id"]).reset_index(drop=True)


def eligible_pairs(table, max_other_gap=None, min_focal_gap=0., max_price_ratio=None,
                   match_free=False, require_tertiles=True):
    """Default eligibility is opposite tertiles and a strictly positive focal gap.

Legacy calipers are explicit opt-ins. No default price/free-option matching.
"""
    for value in (min_focal_gap,) + (() if max_other_gap is None else (max_other_gap,)):
        _require(np.isfinite(value) and value >= 0, "Rating calipers must be finite and nonnegative")
    if max_price_ratio is not None:
        _require(np.isfinite(max_price_ratio) and max_price_ratio >= 1, "Price ratio must be finite and >= 1")
    ok = (table.focal_gap >= min_focal_gap - 1e-10) & (table.focal_gap > 0)
    if max_other_gap is not None:
        ok &= table.other_max <= max_other_gap + 1e-10
    if require_tertiles:
        ok &= table.opposite_tertiles
    if match_free:
        ok &= table.free_option_match
    if max_price_ratio is not None:
        ok &= table.expensive_ratio <= max_price_ratio + 1e-10
        ok &= table.price_gap_ratio <= max_price_ratio + 1e-10
    return table.loc[ok].copy()


def best_pair(table):
    """Smallest adjusted non-focal distance; ties prefer focal gap, then IDs."""
    if table.empty:
        return None
    return rank_pairs(table).iloc[0]


def rank_pairs(table):
    """Order eligible pairs by balance; price never enters even tie-breaking."""
    order = ["other_adjusted", "focal_gap", "low_id", "high_id"]
    return _rank(table, order, [True, False, True, True])


def rank_weighted_pairs(table, penalty):
    """Maximize focal gap - penalty * adjusted distance on an eligible pool.

    Gap and distance keep their existing units. Numerical ties prefer smaller
    distance, then larger gap, then IDs. Prices never affect ranking.
    """
    _require(np.isscalar(penalty) and np.isfinite(penalty) and penalty >= 0,
             "Penalty must be finite and nonnegative")
    required = {"focal_gap", "other_adjusted", "low_id", "high_id"}
    _require(required <= set(table), "Missing weighted-ranking columns")
    values = table[["focal_gap", "other_adjusted"]].to_numpy(float)
    _require(np.isfinite(values).all() and (values >= 0).all()
             and (values[:, 0] <= 10 + 1e-10).all(), "Invalid focal gaps or distances")
    scored = table.assign(penalty=float(penalty), score=table.focal_gap - penalty * table.other_adjusted)
    return _rank(scored, ["score", "other_adjusted", "focal_gap", "low_id", "high_id"],
                 [False, True, False, True, True])


def _rank(table, columns, ascending):
    """Thirds in the mean ratings can differ by roundoff; preserve raw metrics."""
    return table.sort_values(columns, ascending=ascending,
                             key=lambda col: col.round(10) if pd.api.types.is_numeric_dtype(col) else col)


def tradeoff_frontier(table):
    """Diagnostic frontier: increase contrast while decreasing adjusted imbalance.

One representative per equal contrast/balance combination is retained, with
deterministic IDs breaking ties. Input must already meet other rules. The first
row is the minimum-imbalance choice; extra contrast is not the primary objective.
"""
    order = ["other_adjusted", "focal_gap", "low_id", "high_id"]
    ordered = _rank(table, order, [True, False, True, True])
    keep, largest = [], -np.inf
    for index, row in ordered.iterrows():
        if row.focal_gap > largest + 1e-10:
            keep.append(index)
            largest = row.focal_gap
    return ordered.loc[keep].reset_index(drop=True)
