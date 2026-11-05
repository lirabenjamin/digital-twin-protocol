"""Run the requested lambda=0,0.5,1,2,3 comparison with audited, frozen inputs.

.venv/bin/python scripts/compare_matching_penalties.py
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from match_stimuli import TARGETS, COMPARISONS, describe_pairs, geometry_records, audit_moderators
from stimulus_matching import balance_geometry
from stimulus_matching_inputs import load_matching_inputs
from stimulus_matching_sweep import penalty_sweep
from stimulus_matching_sweep_report import write_report
from stimulus_pair_explorer import build_payload, render_explorer
from stimulus_survey_export import write_survey_csvs

PENALTIES = [0., .5, 1., 2., 3.]
OUT = ROOT / "outputs/stimulus_matching/lambda_sweep"


def describe_sweep(selected, features, prices, stimuli, metadata):
    details, differences = [], []
    lookup = {s["stimulus"]: s for s in stimuli}
    for penalty, group in selected.groupby("penalty", sort=False):
        described, diffs = describe_pairs(group, features, prices, stimuli, metadata)
        diffs["penalty"] = penalty
        nonfocal = diffs.loc[~diffs.is_focal].assign(absolute_gap=lambda x: x.signed_difference.abs())
        described["other_median_abs"] = described.focal.map(nonfocal.groupby("focal").absolute_gap.median())
        for side in ("low", "high"):
            described[f"{side}_scenario"] = described[f"{side}_id"].map(lambda sid: lookup[sid]["scenario"])
            described[f"{side}_alternative"] = described[f"{side}_id"].map(lambda sid: lookup[sid]["options"][1]["label"])
        details.append(described)
        differences.append(diffs)
    return pd.concat(details, ignore_index=True), pd.concat(differences, ignore_index=True)


def check_outputs(tables, features):
    selected, differences = tables["selected_pairs"], tables["pair_feature_differences"]
    expected = {(f, p) for f in TARGETS + COMPARISONS for p in PENALTIES}
    if len(selected) != len(expected) or set(zip(selected.focal, selected.penalty)) != expected:
        raise ValueError("Expected one selection per moderator/penalty combination")
    if selected.isna().any().any() or differences.isna().any().any():
        raise ValueError("Selected-pair outputs must be complete")
    if len(differences) != len(expected) * len(features.columns) or differences.duplicated(["focal", "penalty", "feature"]).any():
        raise ValueError("Expected twenty unique feature differences per selected pair")
    if not (selected.opposite_tertiles.all() and selected.focal_gap.gt(0).all()
            and selected.price_rule.eq("unrestricted").all()):
        raise ValueError("Invalid eligibility or price rule")
    for row in selected.itertuples():
        if row.penalty == 0 and not np.isclose(
                row.focal_gap, features[row.focal].max() - features[row.focal].min()):
            raise ValueError("Zero penalty must select the full observed focal range")
        columns = [f for f in features if f != row.focal]
        delta = (features.loc[row.high_id] - features.loc[row.low_id])[columns].to_numpy()
        weights = balance_geometry(features[columns])["weights"]
        direct = [np.sqrt(delta @ weights @ delta), np.abs(delta).mean(),
                  np.median(np.abs(delta)),
                  np.sqrt(np.mean(delta ** 2)), np.abs(delta).max()]
        stored = [row.other_adjusted, row.other_mean_abs, row.other_median_abs, row.other_rms, row.other_max]
        if not np.allclose(stored, direct) or not np.isclose(row.score, row.focal_gap - row.penalty * direct[0]):
            raise ValueError("Selected metrics do not reconstruct from original ratings")
        diffs = differences.query("focal == @row.focal and penalty == @row.penalty").set_index("feature")
        raw = features.loc[row.high_id] - features.loc[row.low_id]
        if set(diffs.index) != set(features.columns) or not np.allclose(diffs.loc[raw.index].signed_difference, raw):
            raise ValueError("Exported feature differences do not reconstruct")


def summarize(selected):
    records = []
    for scope, rows in [("seven nominated", selected.loc[selected.focal.isin(TARGETS)]),
                        ("all ten including comparisons", selected)]:
        for penalty, group in rows.groupby("penalty", sort=True):
            records.append({"scope": scope, "penalty": penalty, "pairs": len(group),
                            "unique_stimuli": len(set(group.low_id) | set(group.high_id)),
                            "mean_focal_gap": group.focal_gap.mean(),
                            "mean_adjusted_distance": group.other_adjusted.mean(),
                            "mean_absolute_nonfocal_difference": group.other_mean_abs.mean(),
                            "mean_pair_max_nonfocal_difference": group.other_max.mean(),
                            "largest_nonfocal_difference": group.other_max.max()})
    return pd.DataFrame(records)


def write_audit(hashes, out=OUT):
    selection_count = len(TARGETS + COMPARISONS) * len(PENALTIES)
    specification = {
        "date": "2026-10-19", "source_sha256": hashes, "penalties": PENALTIES,
        "objective": "maximize F - penalty * D", "F": "high-minus-low focal rating, original 0-10 units",
        "D": "sqrt(delta.T @ W @ delta), all other 19 rated features, original rating units",
        "weights": "W = Q diag(1 / max(eta, 1)) Q.T; eta are correlation eigenvalues",
        "geometry_reference": "all 100 stimuli, fixed across penalties, focal feature excluded",
        "geometry_changed_from_previous_search": False,
        "eligibility": "low in bottom empirical tertile; high in top; boundary ties included; high > low",
        "minimum_focal_gap": None, "nonfocal_tolerance": None, "price_matching": False,
        "tie_break": "smaller D, larger F, then IDs; numeric sorting keys rounded to 10 decimal places",
        "pair_ranking_uses_outcomes": False, "unique_stimuli_across_pairs_required": False,
        "user_nominated": TARGETS, "proposed_comparisons": COMPARISONS,
        "stimuli": 100, "features": 20, "ratings": 6000, "pairs_enumerated_per_moderator": 4950,
        "tests_first_commit": "b407473",
        "lambda_three_and_explorer_tests_first_commit": "8ca319d",
        "lambda_zero_tests_first_commit": "7d5f017",
        "nonfocal_median_tests_first_commit": "9094d7c",
        "portable_reproduction_tests_first_commit": "7352715",
        "input_scope": "versioned original stimuli, definitions, 6000 ratings, means, and moderator estimates",
        "checks_passed": ["frozen input hashes, rating reconstruction, and moderator-table audit", "all eligible-pair score optima",
                          "monotonic selected F and D as penalty increases", f"{selection_count} unique selections",
                          f"{selection_count * 20} unique and complete feature differences",
                          "zero penalty selects full observed focal range", "metrics reconstructed from raw ratings"],
    }
    (out / "audit.json").write_text(json.dumps(specification, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Destination root for outputs/ and reports/; defaults to this checkout.")
    args = parser.parse_args()
    data = load_matching_inputs(ROOT)
    metadata, features, prices, hashes = data.metadata, data.features, data.prices, data.hashes
    audit_moderators(data.moderator_path, features.columns)
    tables = penalty_sweep(features, prices, TARGETS + COMPARISONS, PENALTIES)
    selected, differences = describe_sweep(tables["selected_pairs"], features, prices, data.stimuli, metadata)
    tables.update(selected_pairs=selected, pair_feature_differences=differences, summary=summarize(selected))
    check_outputs(tables, features)
    matrices, eigenvalues = [], []
    for focal in TARGETS + COMPARISONS:
        matrix, eigen = geometry_records(features, focal)
        matrices.extend(matrix)
        eigenvalues.extend(eigen)
    tables.update(geometry=pd.DataFrame(matrices), geometry_eigenvalues=pd.DataFrame(eigenvalues))
    out = args.output_root / "outputs/stimulus_matching/lambda_sweep"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    write_audit(hashes, out)
    path = write_report(args.output_root, selected, tables["summary"], data.stimuli)
    payload = build_payload(features, prices, data.stimuli, metadata, TARGETS + COMPARISONS, PENALTIES, COMPARISONS)
    explorer = path.parent / "pair-explorer.html"
    explorer.write_text(render_explorer(payload))
    survey_paths = write_survey_csvs(selected, data.stimuli, path.parent / "survey", TARGETS + COMPARISONS)
    print(f"Wrote {path}, {explorer}, and {len(survey_paths)} survey CSVs; "
          f"{len(selected)} selections and {len(differences)} feature differences passed all checks.")
    print(tables["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
