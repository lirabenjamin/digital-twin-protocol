"""Read-only source loading, cardinality checks, and reconstruction of paired labels."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    ids: list
    feature_ids: list
    features: np.ndarray
    effects: pd.DataFrame
    stimuli: list
    daily: np.ndarray
    weekly: np.ndarray
    twin_ids: list


def read_tables(root):
    base = Path(root) / "studies/delarosa_stimuli"
    folder = base / "output/full"
    names = {"effects": "stimulus_effects.csv", "features": "stimulus_features.csv",
             "ratings": "stimulus_features_long.csv", "choices": "choices.csv", "runs": "results.csv"}
    tables = {key: pd.read_csv(folder / name, dtype={"pid": str}) for key, name in names.items()}
    tables["stimuli"] = json.loads((base / "stimuli.json").read_text())
    tables["metadata"] = json.loads((base / "features.json").read_text())
    return tables


def verify_source_hashes(root, hashes):
    base = Path(root) / "studies/delarosa_stimuli"
    observed = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in hashes}
    if observed != hashes:
        raise ValueError("Source files changed after protocol freeze")
    return observed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_stimuli(tables):
    stimuli, meta = tables["stimuli"], tables["metadata"]
    ids = [s["stimulus"] for s in stimuli]
    keys = [f["id"] for f in meta]
    require(len(ids) == len(set(ids)) == 100, "Need 100 unique stimulus IDs")
    require(len(keys) == len(set(keys)) == 20, "Need 20 unique feature IDs")
    for s in stimuli:
        require(len(s["options"]) == 2 and bool(s["scenario"].strip()), "Invalid stimulus scenario/options")
        hi, lo = (o["cost"] for o in s["options"])
        require(np.isfinite([hi, lo]).all() and 0 <= lo < hi <= 500, "Invalid price range or ordering")
    return sorted(ids), keys


def _validate_features(tables, ids, keys):
    for name in ("effects", "features"):
        frame = tables[name]
        require(not frame.stimulus.duplicated().any(), f"{name}: duplicate stimulus; IDs must be unique")
        require(len(frame) == 100 and set(frame.stimulus) == set(ids), f"{name}: must cover all 100 stimuli")
    values = tables["features"][keys].to_numpy(float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 10)).all(), "Invalid feature rating range or missing/nonfinite scores")
    require((np.std(values, axis=0) > 0).all(), "Every feature must vary")
    r = tables["ratings"]
    require(len(r) == 6000 and not r.duplicated(["stimulus", "feature", "rep"]).any(), "Need 6000 unique feature rating repetitions")
    require(set(r.stimulus) == set(ids) and set(r.feature) == set(keys) and set(r.rep) == {1, 2, 3}, "Invalid rating identities or repetitions")
    require(r.rating.notna().all() and r.rating.between(0, 10).all() and (r.rating % 1 == 0).all(), "Invalid integer rating range")
    means = r.pivot_table(index="stimulus", columns="feature", values="rating", aggfunc="mean").loc[ids, keys]
    expected = tables["features"].set_index("stimulus").loc[ids, keys]
    require(np.allclose(means, expected, atol=1e-12), "Aggregated feature scores do not match rating repetitions")


def _validate_choices(tables, ids):
    c, runs = tables["choices"], tables["runs"]
    require(not c.duplicated(["pid", "condition", "stimulus"]).any(), "duplicate participant-condition-stimulus choice")
    require(len(c) == 57624, "Choice row count must be 57624 with complete pairs")
    require(set(c.condition) == {"daily", "weekly"}, "Invalid condition labels")
    require(c.expensive.notna().all() and c.expensive.isin([0, 1]).all(), "Choices must be binary")
    require(set(c.stimulus) == set(ids) and not c[["pid", "condition", "slot", "stimulus"]].isna().any().any(), "Missing choice IDs")
    require(c.slot.between(1, 14).all() and not c.duplicated(["pid", "condition", "slot"]).any(), "Invalid or duplicate decision slot")
    require((c.groupby(["pid", "condition"]).size() == 14).all(), "Each twin-condition must have 14 choices")
    pair = c.groupby(["pid", "stimulus"])
    require((pair.condition.nunique() == 2).all() and (pair.slot.nunique() == 1).all(), "Missing condition pair or mismatched slot")
    require(len(runs) == 4116 and not runs.duplicated(["pid", "condition"]).any(), "Need 4116 unique twin runs")
    require(runs.pid.nunique() == 2058 and runs.status.eq("ok").all(), "Need 2058 complete twin pairs")
    require(set(runs.pid) == set(c.pid), "Run and choice twin IDs differ")
    return c


def _paired_matrices(choices, ids):
    pids = sorted(choices.pid.unique())
    matrices = [choices.loc[choices.condition.eq(condition)].pivot(index="pid", columns="stimulus", values="expensive").reindex(index=pids, columns=ids).to_numpy(float)
                for condition in ("daily", "weekly")]
    require(np.array_equal(np.isnan(matrices[0]), np.isnan(matrices[1])), "Incomplete paired support")
    return pids, matrices[0], matrices[1]


def _validate_effects(frame, daily, weekly, stimuli, ids):
    effects = frame.set_index("stimulus").loc[ids]
    values = effects[["daily", "weekly", "effect", "se", "lo", "hi"]].to_numpy(float)
    require(np.isfinite(values).all(), "Effects must be finite and complete")
    n = np.isfinite(daily).sum(axis=0)
    delta = daily - weekly
    checks = [np.array_equal(n, effects.n), np.allclose(np.nanmean(delta, axis=0), effects.effect, atol=1e-12),
              np.allclose(np.nanmean(daily, axis=0), effects.daily, atol=1e-12),
              np.allclose(np.nanmean(weekly, axis=0), effects.weekly, atol=1e-12),
              np.allclose(np.nanstd(delta, axis=0, ddof=1) / np.sqrt(n), effects.se, atol=1e-12)]
    require(all(checks), "Stored effect estimates do not reconstruct from paired choices")
    by_id = {s["stimulus"]: s for s in stimuli}
    for sid, row in effects.iterrows():
        hi, lo = by_id[sid]["options"]
        require(row.cost_expensive == hi["cost"] and row.cost_cheaper == lo["cost"], "Price join mismatch")
    return effects


def dataset_from_tables(tables):
    ids, keys = _validate_stimuli(tables)
    _validate_features(tables, ids, keys)
    choices = _validate_choices(tables, ids)
    pids, daily, weekly = _paired_matrices(choices, ids)
    effects = _validate_effects(tables["effects"], daily, weekly, tables["stimuli"], ids)
    by_id = {s["stimulus"]: s for s in tables["stimuli"]}
    features = tables["features"].set_index("stimulus").loc[ids, keys].to_numpy(float)
    return Dataset(ids, keys, features, effects, [by_id[s] for s in ids], daily, weekly, pids)
