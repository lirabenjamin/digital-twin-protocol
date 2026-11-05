"""Offline checks for feature definitions, theory notes, and complete rating grids."""
import json
import math
import re

import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_catalog(features, expected_count):
    require(isinstance(features, list) and len(features) == expected_count,
            f"Expected {expected_count} feature definitions")
    fields = ("id", "name", "group", "question", "anchor0", "anchor10")
    for feature in features:
        require(all(nonempty(feature.get(k)) for k in fields), "Incomplete feature definition")
        require(re.fullmatch(r"f\d{2}_[a-z_]+", feature["id"]), "Invalid feature ID")
        require(feature["anchor0"] != feature["anchor10"], "Scale anchors must differ")
    for key in ("id", "name", "question"):
        require(len({f[key] for f in features}) == expected_count, f"Duplicate feature {key}")


def validate_stimuli(stimuli, expected_count):
    ids = [s.get("stimulus") for s in stimuli]
    require(len(ids) == len(set(ids)) == expected_count, "Wrong count or duplicate stimulus IDs")
    require(all(isinstance(i, str) and re.fullmatch(r"s\d{3}", i) for i in ids), "Invalid stimulus ID")
    for stimulus in stimuli:
        require(nonempty(stimulus.get("scenario")), "Empty scenario")
        options = stimulus.get("options", [])
        require(len(options) == 2, "Each stimulus needs two options")
        require(all(nonempty(o.get("label")) for o in options), "Empty option label")
        costs = [o.get("cost") for o in options]
        require(all(type(c) in (int, float) and math.isfinite(c) for c in costs), "Invalid price")
        require(0 <= costs[1] < costs[0] <= 500, "Invalid price range or option order")


def validate_theory(theory, features, stimuli):
    feature_ids = {f["id"] for f in features}
    stimulus_ids = {s["stimulus"] for s in stimuli}
    sources = theory["sources"]
    source_ids = {s["id"] for s in sources}
    require(len(source_ids) == len(sources), "Duplicate source IDs")
    for source in sources:
        require(all(nonempty(source.get(k)) for k in ("id", "citation", "url", "finding")), "Incomplete source")
        require(source["url"].startswith("https://"), "Source needs an HTTPS link")
    notes = theory["features"]
    require(len(notes) == 30 and {n["id"] for n in notes} == {f["id"] for f in features[20:]},
            "Theory notes must cover all 30 additions exactly once")
    for note in notes:
        for field in ("rationale", "hypothesis", "distinction", "coverage_note"):
            require(nonempty(note.get(field)), f"Missing theory field: {field}")
        require(note.get("basis") in {"literature-informed", "theoretical extension"}, "Invalid evidence label")
        require(note.get("moderation_evidence") == "untested in this study", "Do not imply established moderation")
        require(note.get("coverage_risk") in {"check", "likely sparse"}, "Invalid coverage risk")
        for field, valid in (("source_ids", source_ids), ("overlap_ids", feature_ids), ("example_ids", stimulus_ids)):
            values = note.get(field, [])
            require(bool(values) and len(values) == len(set(values)) and set(values) <= valid,
                    f"Invalid theory links: {field}")
        require(note["id"] not in note["overlap_ids"], "Feature cannot overlap with itself")


def parse_rating(text):
    """Accept only the requested integer JSON; never turn 4.5 into 4."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Rating response is not JSON") from exc
    require(isinstance(payload, dict), "Rating response must be an object")
    value = payload.get("rating")
    require(type(value) is int and 0 <= value <= 10, "Rating must be an integer from 0 to 10")
    return value


def validate_ratings(frame, stimulus_ids, feature_ids, reps=3):
    require({"stimulus", "feature", "rep", "rating"} <= set(frame.columns), "Missing rating columns")
    require(len(stimulus_ids) == len(set(stimulus_ids)) > 0, "Duplicate or empty expected stimuli")
    require(len(feature_ids) == len(set(feature_ids)) > 0, "Duplicate or empty expected features")
    require(type(reps) is int and reps > 0, "Invalid repetition count")
    keys = ["stimulus", "feature", "rep"]
    require(not frame[keys].isna().any().any(), "Missing rating identifiers")
    require(not frame.duplicated(keys).any(), "Duplicate rating key")
    expected = pd.MultiIndex.from_product([stimulus_ids, feature_ids, range(1, reps + 1)], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    require(len(actual) == len(expected) and actual.difference(expected).empty
            and expected.difference(actual).empty, "Incomplete or invalid stimulus-feature-repetition grid")
    values = pd.to_numeric(frame.rating, errors="coerce")
    require(values.notna().all() and values.between(0, 10).all() and values.mod(1).eq(0).all(),
            "Missing, noninteger, or out-of-range rating")


def aggregate_ratings(frame, stimulus_ids, feature_ids, reps=3):
    validate_ratings(frame, stimulus_ids, feature_ids, reps)
    data = frame.assign(rating=pd.to_numeric(frame.rating))
    summary = data.groupby(["stimulus", "feature"]).rating.agg(["mean", "std"]).unstack("feature")
    columns = [(stat, feature) for stat in ("mean", "std") for feature in feature_ids]
    summary = summary.loc[stimulus_ids, columns]
    summary.columns = [feature if stat == "mean" else f"{feature}_sd" for stat, feature in columns]
    return summary.reset_index()
