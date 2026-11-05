"""Write the scenario (page 1) and the intro line (page 2) for each picked stimulus with Sonnet.
The frame text itself is not rewritten: page 2 is intro + the stimulus's own choice_text or rejection_text,
so the two frames differ only where the generator made them differ."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import HERE, call

PROMPT = (HERE / "prompts" / "scenario.txt").read_text()
SYSTEM = "You write stimuli for a consumer-behavior study about how other people's decisions influence our own."
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="stimuli.json")
ap.add_argument("--dst", default="stimuli.json")
a = ap.parse_args()
stimuli = json.loads((HERE / a.src).read_text())


def fill(s: dict) -> str:
    text = PROMPT
    for k, v in {"situation": s["situation"], "option_a": s["option_a"]["label"], "option_b": s["option_b"]["label"],
                 "source": s["source"], "channel": s["channel"], "choice_text": s["choice_text"],
                 "rejection_text": s["rejection_text"]}.items():
        text = text.replace("{" + k + "}", v)
    return text


def write(s: dict) -> dict:
    if "scenario" in s:
        return s
    return {**s, **call(SYSTEM, fill(s), step=f"screens:{s.get('stimulus') or 'seed' + s['study']}")}


with ThreadPoolExecutor(8) as ex:
    stimuli = list(ex.map(write, stimuli))
(HERE / a.dst).write_text(json.dumps(stimuli, indent=1))
print(f"{sum('scenario' in s for s in stimuli)} of {len(stimuli)} stimuli have screens")
