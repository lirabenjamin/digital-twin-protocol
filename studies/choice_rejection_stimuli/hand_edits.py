"""Hand edits after the 2026-11-02 v3 audit and repair (exact string replacements; fails loudly if a target is gone)."""
import json

from common import HERE

EDITS = {
    "s032": [("scenario", "You have to decide how to spend your money, and you have two options. Option A is the motor "
              "repair, which", "You have to decide how to spend your money. The motor repair"),
             ("scenario", "electricity. Option B is the new machine, which", "electricity. The new machine")],
    "s085": [("choice_text", "then bought one of the two and bought the plush pair, and walked to the register.",
              "and bought one of the two. She chose the plush pair."),
             ("rejection_text", "then bought one of the two and did not buy the slim pair, and walked to the register.",
              "and bought one of the two. She did not choose the slim pair.")],
    "s065": [("scenario", " Both are in good supply, so either one is easy to get.", "")],
    # one sentence described both options, so it could not go into side-by-side boxes (2026-11-03)
    "s075": [("scenario", "takes a few minutes to eat, or a tub of mixed gummies,",
              "takes a few minutes to eat. Or you can get a tub of mixed gummies,")],
}
# layouts set by hand where segment_scenarios.py found no exact split
LAYOUTS = {"s075": {
    "setup": "Imagine that you are at a cinema concession stand with only one minute left before the trailers start, "
             "and you want a snack to take into the film.",
    "a": "You can get a giant pretzel with cheese dip, which is warm, filling, and takes a few minutes to eat.",
    "b": "Or you can get a tub of mixed gummies, which is sweet, easy to snack on in the dark, and keeps well if you "
         "do not finish it.",
    "after": "The pretzel and the gummies cost about the same, and the stand has plenty of both. The line is moving, "
             "and you have time to buy only one of them. Please think about which of the two options you would choose."}}
stimuli = json.loads((HERE / "stimuli.json").read_text())
for s in stimuli:
    for field, old, new in EDITS.get(s["stimulus"], []):
        if old in s[field]:
            s[field] = s[field].replace(old, new)
        elif new not in s[field]:
            raise SystemExit(f"{s['stimulus']}.{field}: target text not found")
    if s["stimulus"] in LAYOUTS and not s.get("layout"):
        s["layout"] = LAYOUTS[s["stimulus"]]
        assert " ".join(s["layout"].values()) == s["scenario"], s["stimulus"]
(HERE / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
