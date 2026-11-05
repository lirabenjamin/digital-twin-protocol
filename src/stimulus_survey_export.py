"""Export each selected scenario once, preserving original survey wording."""
import csv
import io
from pathlib import Path

from stimulus_matching_inputs import validate_scenario
from twin_effects.data import require

HEADER = [
    "stimulus_id", "scenario_verbatim", "expensive_option_verbatim",
    "expensive_price_usd", "cheaper_option_verbatim", "cheaper_price_usd",
    "low_for_moderators", "high_for_moderators",
]
PENALTIES = (1, 2)


def _validate_selection(selected, stimuli, expected_focals):
    fields = ["focal", "penalty", "low_id", "high_id", "moderator", "role"]
    require(set(fields) <= set(selected), "Missing selected-pair columns")
    require(not selected[fields].isna().any().any(), "Missing selected-pair values")
    require(len(expected_focals) == len(set(expected_focals)) > 0,
            "Expected focal moderators must be unique")
    require(not selected.duplicated(["focal", "penalty"]).any(), "Duplicate selected pair")
    require(selected.role.isin(["user_nominated", "proposed_comparison"]).all(),
            "Unknown moderator role")
    require(selected.moderator.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
            "Missing moderator name")
    require(selected.low_id.ne(selected.high_id).all(), "A pair must use two stimuli")
    lookup = {s["stimulus"]: s for s in stimuli}
    require(len(lookup) == len(stimuli), "Duplicate stimulus metadata")
    require((set(selected.low_id) | set(selected.high_id)) <= set(lookup),
            "Selected IDs do not join to stimulus metadata")
    for stimulus in stimuli:
        validate_scenario(stimulus)
    for penalty in PENALTIES:
        group = selected.loc[selected.penalty.eq(penalty)]
        require(len(group) == len(expected_focals) and set(group.focal) == set(expected_focals),
                f"Missing or unexpected moderator for lambda={penalty}")
    return lookup


def _survey_rows(group, lookup):
    ids = sorted(set(group.low_id) | set(group.high_id))
    memberships = {sid: {"low": [], "high": []} for sid in ids}
    for row in group.itertuples():
        label = row.moderator + (" (comparison)" if row.role == "proposed_comparison" else "")
        for side in ("low", "high"):
            memberships[getattr(row, f"{side}_id")][side].append(label)
    rows = [HEADER]
    for sid in ids:
        stimulus = lookup[sid]
        expensive, cheaper = stimulus["options"]
        rows.append([
            sid, stimulus["scenario"], expensive["label"], expensive["cost"],
            cheaper["label"], cheaper["cost"], "; ".join(memberships[sid]["low"]),
            "; ".join(memberships[sid]["high"]),
        ])
    return rows


def write_survey_csvs(selected, stimuli, destination, expected_focals):
    """Validate both sets before writing UTF-8 CSVs with an Excel-compatible BOM."""
    lookup = _validate_selection(selected, stimuli, expected_focals)
    exports = {}
    for penalty in PENALTIES:
        group = selected.loc[selected.penalty.eq(penalty)].set_index("focal").loc[expected_focals]
        rows = _survey_rows(group, lookup)
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerows(rows)
        content = buffer.getvalue()
        require(list(csv.reader(io.StringIO(content, newline=""))) ==
                [[str(value) for value in row] for row in rows],
                "CSV serialization changed the survey text")
        exports[f"survey_stimuli_lambda_{penalty}.csv"] = content.encode("utf-8-sig")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in exports.items():
        path = destination / name
        path.write_bytes(content)
        require(path.read_bytes() == content, "Saved CSV does not match the verified export")
        paths.append(path)
    return paths
