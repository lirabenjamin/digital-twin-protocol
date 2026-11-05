"""Split each scenario, word for word, into setup / Option A / Option B / after, for a layout with the two options
side by side (design decision, 2026-11-03: no hidden text, line breaks, options side by side). A split is kept only if its
sentences are exactly the scenario's sentences; otherwise it is retried, and after 3 failures the scenario stays
one paragraph."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from common import HERE, call

PROMPT = (HERE / "prompts" / "segment.txt").read_text()
SYSTEM = "You format stimuli for a consumer-behavior study. You never change their wording."
PARTS = ("setup", "a", "b", "after")


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def split(s: dict) -> dict:
    text = (PROMPT.replace("{option_a}", s["option_a"]["label"]).replace("{option_b}", s["option_b"]["label"])
            .replace("{scenario}", s["scenario"]))
    for attempt in range(3):
        seg = call(SYSTEM, text, step=f"segment:{s['stimulus']}:{attempt}")
        sentences = [x for p in PARTS for x in seg.get(p, [])]
        ok = (sorted(words(" ".join(sentences))) == sorted(words(s["scenario"]))
              and all(s["scenario"].count(x) == 1 for x in sentences) and seg.get("a") and seg.get("b"))
        if ok:
            return {**s, "layout": {p: " ".join(seg.get(p, [])) for p in PARTS}}
    print(f"{s['stimulus']}: no exact split, kept as one paragraph")
    return {**s, "layout": None}


stimuli = json.loads((HERE / "stimuli.json").read_text())
with ThreadPoolExecutor(8) as ex:
    stimuli = list(ex.map(split, stimuli))
(HERE / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
print(f"{sum(s['layout'] is not None for s in stimuli)} of {len(stimuli)} split")
