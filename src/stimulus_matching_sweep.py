"""Compare contrast/balance penalties while fixing data, eligibility and geometry."""
import numpy as np
import pandas as pd

from stimulus_matching import eligible_pairs, pair_table, rank_pairs, rank_weighted_pairs


def _unique_nonempty(values, label):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be nonempty and unique")


def _selection(ranked, tags):
    best = ranked.iloc[0]
    if best.score < ranked.score.max() - 1e-9:
        raise ValueError("Selected pair does not maximize the weighted score")
    margin = best.score - ranked.iloc[1].score if len(ranked) > 1 else np.nan
    return {**tags, **best.to_dict(), "n_eligible": len(ranked),
            "runner_up_score_gap": max(0., margin) if np.isfinite(margin) else margin,
            "n_best_score_ties": int(ranked.score.round(10).eq(round(best.score, 10)).sum())}


def _check_tradeoff(selected):
    for _, rows in selected.groupby("focal", sort=False):
        ordered = rows.sort_values("penalty")
        if (np.diff(ordered.other_adjusted) > 1e-9).any() or (np.diff(ordered.focal_gap) > 1e-9).any():
            raise ValueError("Increasing the balance penalty must not increase selected distance or gap")


def penalty_sweep(features, prices, focals, penalties):
    """Search every pair for each moderator; tertiles only, with no price matching."""
    _unique_nonempty(focals, "Focal moderators")
    _unique_nonempty(penalties, "Penalties")
    selected, top, eligibility, baseline = [], [], [], []
    for focal in focals:
        table = pair_table(features, focal, [f for f in features if f != focal], prices)
        if len(table) != len(features) * (len(features) - 1) // 2:
            raise ValueError("Pair enumeration changed the expected row count")
        pool = eligible_pairs(table)
        if pool.empty:
            raise ValueError(f"No eligible opposite-tertile pair for {focal}")
        tags = {"focal": focal, "price_rule": "unrestricted"}
        baseline.append({**tags, **rank_pairs(pool).iloc[0].to_dict()})
        eligibility.append({**tags, "n_all_pairs": len(table), "n_eligible": len(pool),
                            "low_cutoff": table.iloc[0].tertile_low_cutoff,
                            "high_cutoff": table.iloc[0].tertile_high_cutoff})
        for penalty in penalties:
            ranked = rank_weighted_pairs(pool, penalty)
            selected.append(_selection(ranked, tags))
            top.extend({**tags, "rank": i, **row.to_dict()}
                       for i, (_, row) in enumerate(ranked.head(10).iterrows(), 1))
    chosen = pd.DataFrame(selected)
    _check_tradeoff(chosen)
    return {"selected_pairs": chosen, "top_candidates": pd.DataFrame(top),
            "eligibility": pd.DataFrame(eligibility), "minimum_distance_baseline": pd.DataFrame(baseline)}
