"""Audit the existing twin study and export transparent high/low pair searches.

Run from the repository: .venv/bin/python scripts/match_stimuli.py
No model calls or changes to the original study data; results are design candidates.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm

from stimulus_matching import pair_table, eligible_pairs, rank_pairs, tradeoff_frontier, balance_geometry
from stimulus_matching_report import write_report
from stimulus_matching_inputs import load_matching_inputs
from twin_effects.data import read_tables, dataset_from_tables, verify_source_hashes

OUT = ROOT / "outputs/stimulus_matching"
# Justifiability is excluded as a focal hypothesis; all original nonfocal ratings
# still enter balance. See studies/delarosa_stimuli/HUMAN_STUDY_PLAN.md.
TARGETS = ["f09_splurge", "f05_lasting", "f19_upgrade", "f02_deferrable",
           "f03_impulse", "f01_necessity", "f06_frequency"]
COMPARISONS = ["f11_emotion", "f17_convenience", "f18_money_salience"]


def audit_moderators(path, feature_ids):
    table = pd.read_csv(path)
    if len(table) != 41 or table.duplicated(["feature", "spec"]).any():
        raise ValueError("Expected 41 unique feature/specification rows")
    features = table.loc[table.feature.ne("log_price")].copy()
    expected = {(f, s) for f in feature_ids for s in ("base", "price")}
    if set(zip(features.feature, features.spec)) != expected:
        raise ValueError("Moderator table must contain all 20 features in both specifications")
    if not np.isfinite(features[["b", "se", "p", "bh"]]).all().all():
        raise ValueError("Missing or nonfinite moderator estimates")
    if not (features.se.gt(0).all() and features.p.between(0, 1).all()
            and features.bh.between(0, 1).all() and features.converged.eq(True).all()):
        raise ValueError("Invalid standard errors, p-values, or convergence flags")
    if not np.allclose(features.p, 2 * norm.sf(np.abs(features.b / features.se))):
        raise ValueError("Stored p-values do not reconstruct from estimates/SEs")
    for _, group in features.groupby("spec"):
        if not np.allclose(group.bh, false_discovery_control(group.p.to_numpy())):
            raise ValueError("Stored BH correction does not reconstruct")
    # log_price is outside the two 20-feature correction families; missing BH is structural.
    price = table.loc[table.feature.eq("log_price")]
    if len(price) != 1 or price.iloc[0].spec != "price_only" or price.bh.notna().any():
        raise ValueError("Unexpected price-only row")
    features["ci_low"] = features.b - 1.96 * features.se
    features["ci_high"] = features.b + 1.96 * features.se
    features["role"] = np.where(features.feature.isin(TARGETS), "user_nominated",
                                 np.where(features.feature.isin(COMPARISONS), "proposed_comparison", "other"))
    return features


def load_inputs():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    hashes = verify_source_hashes(protocol["source_root"], protocol["source_sha256"])
    tables = read_tables(protocol["source_root"])
    data = dataset_from_tables(tables)
    features = pd.DataFrame(data.features, index=data.ids, columns=data.feature_ids)
    prices = pd.DataFrame([{"stimulus": s["stimulus"], "expensive": s["options"][0]["cost"],
                            "cheaper": s["options"][1]["cost"]} for s in data.stimuli]).set_index("stimulus")
    path = Path(protocol["source_root"]) / "studies/delarosa_stimuli/output/full/moderation_mlm.csv"
    evidence = audit_moderators(path, data.feature_ids)
    hashes["output/full/moderation_mlm.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data, tables["metadata"], features, prices, evidence, hashes


def geometry_records(features, focal):
    columns = [f for f in features if f != focal]
    geometry = balance_geometry(features[columns].to_numpy())
    matrices = [{"focal": focal, "row_feature": a, "column_feature": b,
                 "correlation": geometry["correlation"][i, j], "weight": geometry["weights"][i, j]}
                for i, a in enumerate(columns) for j, b in enumerate(columns)]
    eigenvalues = [{"focal": focal, "direction": i + 1, "eigenvalue": value,
                    "directional_weight": weight}
                   for i, (value, weight) in enumerate(zip(geometry["eigenvalues"], geometry["directional_weights"]))]
    return matrices, eigenvalues


def metric_choice(table, column):
    chosen = table.sort_values([column, "focal_gap", "low_id", "high_id"],
                               ascending=[True, False, True, True],
                               key=lambda x: x.round(10) if pd.api.types.is_numeric_dtype(x) else x).iloc[0]
    return f"{chosen.low_id} → {chosen.high_id}"


def search(features, prices):
    chosen, top, frontier, eligibility, matrices, eigenvalues = [], [], [], [], [], []
    for focal in TARGETS + COMPARISONS:
        table = pair_table(features, focal, [f for f in features if f != focal], prices)
        if len(table) != 4950:
            raise ValueError("Must enumerate all 4,950 pairs")
        matrix, eigen = geometry_records(features, focal)
        matrices.extend(matrix)
        eigenvalues.extend(eigen)
        for price_rule in ("unrestricted", "matched"):
            tags = {"scope": "all20", "focal": focal, "price_rule": price_rule}
            kwargs = {} if price_rule == "unrestricted" else {"max_price_ratio": 1.25, "match_free": True}
            pool = eligible_pairs(table, **kwargs)
            ranked = rank_pairs(pool)
            qlow, qhigh = table.iloc[0][["tertile_low_cutoff", "tertile_high_cutoff"]]
            eligibility.append({**tags, "low_cutoff": qlow, "high_cutoff": qhigh, "n_eligible": len(pool),
                                "n_low": int(features[focal].le(qlow + 1e-10).sum()),
                                "n_high": int(features[focal].ge(qhigh - 1e-10).sum())})
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            if not np.isclose(best.other_adjusted, pool.other_adjusted.min(), atol=1e-10):
                raise ValueError("Selected pair does not minimize non-focal distance")
            chosen.append({**tags, "n_eligible": len(pool), **best.to_dict(),
                           "runner_up_distance_gap": ranked.iloc[1].other_adjusted - best.other_adjusted if len(ranked) > 1 else np.nan,
                           "raw_rms_pair": metric_choice(pool, "other_rms"),
                           "raw_max_pair": metric_choice(pool, "other_max")})
            top.extend({**tags, "rank": rank, **row.to_dict()} for rank, (_, row) in enumerate(ranked.head(10).iterrows(), 1))
            frontier.extend({**tags, "frontier_rank": rank, **row.to_dict()}
                            for rank, (_, row) in enumerate(tradeoff_frontier(pool).iterrows(), 1))
    return {"selected_pairs": pd.DataFrame(chosen), "top_candidates": pd.DataFrame(top),
            "pareto_pairs": pd.DataFrame(frontier), "eligibility": pd.DataFrame(eligibility),
            "geometry": pd.DataFrame(matrices), "geometry_eigenvalues": pd.DataFrame(eigenvalues)}


def describe_pairs(selected, features, prices, stimuli, metadata):
    names = {f["id"]: f["name"] for f in metadata}
    scenarios = {s["stimulus"]: s for s in stimuli}
    details = selected.copy()
    details["moderator"] = details.focal.map(names)
    details["worst_feature_name"] = details.worst_feature.map(names)
    details["role"] = np.where(details.focal.isin(TARGETS), "user_nominated", "proposed_comparison")
    records = []
    for i, row in details.iterrows():
        for side in ("low", "high"):
            sid = row[f"{side}_id"]
            details.loc[i, f"{side}_label"] = scenarios[sid]["options"][0]["label"]
            details.loc[i, f"{side}_expensive"] = prices.loc[sid, "expensive"]
            details.loc[i, f"{side}_cheaper"] = prices.loc[sid, "cheaper"]
        delta = features.loc[row.high_id] - features.loc[row.low_id]
        for fid in features:
            records.append({"focal": row.focal, "price_rule": row.price_rule, "low_id": row.low_id,
                            "high_id": row.high_id, "feature": fid, "feature_name": names[fid],
                            "is_focal": fid == row.focal, "low_rating": features.loc[row.low_id, fid],
                            "high_rating": features.loc[row.high_id, fid], "signed_difference": delta[fid]})
    return details, pd.DataFrame(records)


def check_outputs(tables):
    chosen = tables["selected_pairs"]
    if len(chosen) != 20 or chosen.duplicated(["focal", "price_rule"]).any():
        raise ValueError("Expected ten unique moderator selections in each version")
    if not (chosen.opposite_tertiles.all() and chosen.focal_gap.gt(0).all()):
        raise ValueError("Selected stimuli violate opposite-tertile eligibility")
    for row in chosen.itertuples():
        diffs = tables["pair_feature_differences"].query("focal == @row.focal and price_rule == @row.price_rule")
        if len(diffs) != 20 or diffs.feature.duplicated().any():
            raise ValueError("Each pair must have exactly twenty unique feature differences")
        other = diffs.loc[~diffs.is_focal, "signed_difference"]
        if not np.allclose([row.other_mean_abs, row.other_rms, row.other_max],
                           [other.abs().mean(), np.sqrt(np.mean(other ** 2)), other.abs().max()]):
            raise ValueError("Pair diagnostics do not reconstruct from raw ratings")
        if row.other_adjusted ** 2 > np.sum(other ** 2) + 1e-9:
            raise ValueError("Correlation correction must not amplify squared movement")
    matched = chosen.loc[chosen.price_rule.eq("matched")]
    if not (matched.expensive_ratio.le(1.25 + 1e-10).all() and matched.price_gap_ratio.le(1.25 + 1e-10).all()
            and matched.free_option_match.all()):
        raise ValueError("Optional price-matched version violates its price rules")


def write_audit(hashes):
    specification = {
        "version": 2, "date": "2026-10-18", "source_sha256": hashes,
        "user_nominated": TARGETS, "proposed_comparisons": COMPARISONS,
        "primary_matching_scope": "all other 19 rated features",
        "eligibility": "bottom/top empirical tertiles including ties, with high > low",
        "minimum_focal_gap": None, "nonfocal_tolerance": None,
        "price_matching_default": False, "price_enters_default_tie_break": False,
        "optional_price_matching": {"max_ratio": 1.25, "variables": ["expensive", "expensive-minus-cheaper"], "match_free_alternative": True},
        "primary_objective": "minimize sqrt(delta.T @ W @ delta)",
        "weights": "W = Q diag(1/max(lambda, 1)) Q.T from Pearson correlation of all 100 reference stimuli on the 19 nonfocal features",
        "distance_scale": "original 0-10 rating differences; no division by SD or feature count",
        "interpretation": "proposed discount-only metric; not ordinary Mahalanobis or validated semantic de-duplication",
        "tie_break": "larger focal gap, then stimulus IDs; numerical keys rounded to 10 decimals",
        "pair_count_per_moderator": 4950, "stimuli": 100, "features": 20, "ratings": 6000,
        "audit": "versioned input hashes, identities, values, missingness, joins, rating reconstruction, model evidence, and selected-pair diagnostics passed",
        "pair_ranking_uses_outcomes": False, "moderator_selection_informed_by_twin_results": True,
        "unique_stimuli_across_pairs_required": False, "tests_first_commit": "1b03376",
        "prior_version_commit": "7d6a941",
    }
    (OUT / "audit.json").write_text(json.dumps(specification, indent=2) + "\n")


def main():
    data = load_matching_inputs(ROOT)
    metadata, features, prices, hashes = data.metadata, data.features, data.prices, data.hashes
    evidence = audit_moderators(data.moderator_path, features.columns)
    tables = search(features, prices)
    selected, differences = describe_pairs(tables["selected_pairs"], features, prices, data.stimuli, metadata)
    tables.update({"selected_pairs": selected, "pair_feature_differences": differences,
                   "twin_moderator_evidence": evidence})
    check_outputs(tables)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    write_audit(hashes)
    path = write_report(ROOT, data, selected)
    for superseded in ("caliper_search.csv", "closest_pairs.csv"):
        (OUT / superseded).unlink(missing_ok=True)
    print(f"Wrote {path}")
    print(f"Selected {len(selected)} pairs: ten default and ten optional price-matched; all output checks passed.")


if __name__ == "__main__":
    main()
