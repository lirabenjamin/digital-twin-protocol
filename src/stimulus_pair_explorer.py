"""Build a self-contained explorer from audited data and the exact matching rules."""
import json
from pathlib import Path

from stimulus_matching import eligible_pairs, pair_table, rank_weighted_pairs
from stimulus_matching_sweep_report import SCENARIO_LABELS


def _lookup(records, key, expected):
    result = {r[key]: r for r in records}
    if len(result) != len(records) or set(result) != set(expected):
        raise ValueError(f"{key} metadata must join one-to-one with all ratings")
    return result


def _stimuli(features, prices, stimuli):
    lookup = _lookup(stimuli, "stimulus", features.index)
    output = []
    for sid, row in features.iterrows():
        s = lookup[sid]
        if [o["cost"] for o in s["options"]] != prices.loc[sid, ["expensive", "cheaper"]].tolist():
            raise ValueError("Scenario prices do not match the audited price table")
        output.append({"id": sid, "label": SCENARIO_LABELS.get(sid, s["options"][0]["label"]),
                       "text": s["scenario"], "options": s["options"], "ratings": row.tolist()})
    return output


def _moderator(features, prices, focal, metadata, penalties, comparison):
    table = pair_table(features, focal, [f for f in features if f != focal], prices)
    pool = eligible_pairs(table).reset_index(drop=True)
    if pool.empty:
        raise ValueError(f"No eligible pairs for {focal}")
    index = {sid: i for i, sid in enumerate(features.index)}
    pairs = [[index[r.low_id], index[r.high_id], r.focal_gap, r.other_adjusted] for r in pool.itertuples()]
    winners = [int(rank_weighted_pairs(pool, p).index[0]) for p in penalties]
    return {"id": focal, "name": metadata[focal]["name"], "comparison": comparison,
            "featureIndex": list(features.columns).index(focal),
            "cutoffs": [float(pool.iloc[0].tertile_low_cutoff), float(pool.iloc[0].tertile_high_cutoff)],
            "pairs": pairs, "winners": winners}


def build_payload(features, prices, stimuli, metadata, focals, penalties, comparisons):
    """Preserve full precision for scores and include all eligible pairs, not a sample."""
    names = _lookup(metadata, "id", features.columns)
    if not penalties or len(set(penalties)) != len(penalties) or len(set(focals)) != len(focals):
        raise ValueError("Require unique penalties and focal moderators")
    moderators = [_moderator(features, prices, f, names, penalties, f in comparisons) for f in focals]
    return {"penalties": penalties, "features": [names[f] for f in features.columns],
            "stimuli": _stimuli(features, prices, stimuli), "moderators": moderators,
            "date": "2026-10-19", "defaultModerator": focals.index("f06_frequency") if "f06_frequency" in focals else 0}


def render_explorer(payload):
    template = (Path(__file__).parent / "templates/stimulus_pair_explorer.html").read_text()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # A scenario is text, never executable markup, even inside a script-data element.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace("__EXPLORER_DATA__", encoded)
