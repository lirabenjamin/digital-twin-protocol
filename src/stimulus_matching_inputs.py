"""Load the frozen, versioned inputs needed for stimulus matching alone."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twin_effects.data import require

STUDY = Path("studies/delarosa_stimuli")
FILES = {
    "stimuli.json": STUDY / "stimuli.json",
    "features.json": STUDY / "features.json",
    **{f"output/full/{name}": STUDY / "matching_inputs" / name for name in (
        "stimulus_features.csv", "stimulus_features_long.csv", "moderation_mlm.csv")},
}


@dataclass
class MatchingInputs:
    stimuli: list
    metadata: list
    features: pd.DataFrame
    prices: pd.DataFrame
    moderator_path: Path
    hashes: dict


def validate_scenario(stimulus):
    """Keep the original wording; reject incomplete text or invalid option prices."""
    text = stimulus.get("scenario")
    options = stimulus.get("options", [])
    require(isinstance(text, str) and bool(text.strip()), "Missing scenario text")
    require(len(options) == 2, "Expected exactly two purchase options")
    for option in options:
        label = option.get("label")
        require(isinstance(label, str) and bool(label.strip()), "Missing option label")
    costs = [option.get("cost") for option in options]
    require(all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in costs),
            "Prices must be numeric")
    require(np.isfinite(costs).all() and 0 <= costs[1] < costs[0] <= 500,
            "Invalid price range or ordering")


def _validate_repetitions(ratings, features):
    columns = ["stimulus", "feature", "rep", "rating"]
    require(set(columns) <= set(ratings), "Missing rating columns")
    require(not ratings[columns].isna().any().any(), "Missing rating values or identities")
    require(len(ratings) == 6000 and not ratings.duplicated(columns[:3]).any(),
            "Need 6000 unique rating repetitions")
    require(set(ratings.stimulus) == set(features.index)
            and set(ratings.feature) == set(features.columns)
            and set(ratings.rep) == {1, 2, 3}, "Invalid rating identities or repetitions")
    values = ratings.rating.to_numpy(float)
    require(np.isfinite(values).all() and ((0 <= values) & (values <= 10)).all()
            and (values % 1 == 0).all(), "Invalid integer rating range")
    reconstructed = ratings.pivot_table(
        index="stimulus", columns="feature", values="rating", aggfunc="mean"
    ).loc[features.index, features.columns]
    require(np.allclose(reconstructed, features, rtol=0, atol=1e-12),
            "Mean feature ratings do not reconstruct from repetitions")


def validate_matching_tables(stimuli, metadata, means, ratings):
    """Validate keys, joins, bounds and all 2,000 means before using the inputs."""
    ids = [s.get("stimulus") for s in stimuli]
    keys = [f.get("id") for f in metadata]
    require(all(isinstance(s, str) and s for s in ids), "Missing stimulus IDs")
    require(all(isinstance(f, str) and f for f in keys), "Missing feature IDs")
    require(len(ids) == len(set(ids)) == 100, "Need 100 unique stimuli")
    require(len(keys) == len(set(keys)) == 20, "Need 20 unique features")
    for stimulus in stimuli:
        validate_scenario(stimulus)
    require({"stimulus", *keys} <= set(means), "Missing mean-rating columns")
    require(len(means) == 100 and not means.stimulus.duplicated().any()
            and set(means.stimulus) == set(ids), "Mean ratings must join one-to-one")
    features = means.set_index("stimulus").loc[sorted(ids), keys]
    values = features.to_numpy(float)
    require(np.isfinite(values).all() and ((0 <= values) & (values <= 10)).all(),
            "Invalid mean rating range or missing values")
    require((np.std(values, axis=0) > 0).all(), "Every feature must vary")
    _validate_repetitions(ratings, features)
    prices = pd.DataFrame([
        {"stimulus": s["stimulus"], "expensive": s["options"][0]["cost"],
         "cheaper": s["options"][1]["cost"]} for s in stimuli
    ]).set_index("stimulus").loc[features.index]
    return features, prices


def load_matching_inputs(root):
    """Read only files in this checkout; never consult a private source checkout."""
    root = Path(root)
    manifest = json.loads((root / STUDY / "matching_inputs/manifest.json").read_text())
    expected = manifest["source_sha256"]
    require(set(expected) == set(FILES), "Unexpected frozen input manifest")
    hashes = {name: hashlib.sha256((root / path).read_bytes()).hexdigest()
              for name, path in FILES.items()}
    require(hashes == expected, "Frozen matching input hashes changed")
    stimuli = json.loads((root / FILES["stimuli.json"]).read_text())
    metadata = json.loads((root / FILES["features.json"]).read_text())
    means = pd.read_csv(root / FILES["output/full/stimulus_features.csv"])
    ratings = pd.read_csv(root / FILES["output/full/stimulus_features_long.csv"])
    features, prices = validate_matching_tables(stimuli, metadata, means, ratings)
    by_id = {s["stimulus"]: s for s in stimuli}
    return MatchingInputs(
        [by_id[sid] for sid in features.index], metadata, features, prices,
        root / FILES["output/full/moderation_mlm.csv"], hashes,
    )
