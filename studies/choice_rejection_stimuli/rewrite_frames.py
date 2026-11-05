"""Rewrite each stimulus's choice/rejection texts as a matched pair (identical except 'chose B' vs. 'did not choose A'),
then drop its old screens so write_screens.py writes new ones. Keeps the generator's texts as orig_choice_text /
orig_rejection_text."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import HERE, call

PROMPT = (HERE / "prompts" / "frames.txt").read_text()
SYSTEM = "You edit stimuli for a consumer-behavior study about how other people's decisions influence our own."
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="stimuli.json")
ap.add_argument("--dst", default="stimuli.json")
ap.add_argument("--only", nargs="*", help="stimulus ids to rewrite (default: all)")
a = ap.parse_args()
stimuli = json.loads((HERE / a.src).read_text())


def fill(s: dict) -> str:
    text = PROMPT
    for k, v in {"situation": s["situation"], "option_a": s["option_a"]["label"], "option_b": s["option_b"]["label"],
                 "source": s["source"], "channel": s["channel"],
                 "choice_text": s.get("orig_choice_text", s["choice_text"]),
                 "rejection_text": s.get("orig_rejection_text", s["rejection_text"])}.items():
        text = text.replace("{" + k + "}", v)
    return text


def rewrite(s: dict) -> dict:
    if a.only and s["stimulus"] not in a.only:
        return s
    new = call(SYSTEM, fill(s), step=f"frames:{s['stimulus']}")
    keep = {k: v for k, v in s.items() if k not in ("scenario", "intro")}
    return {**keep, "orig_choice_text": s.get("orig_choice_text", s["choice_text"]),
            "orig_rejection_text": s.get("orig_rejection_text", s["rejection_text"]), **new}


with ThreadPoolExecutor(8) as ex:
    stimuli = list(ex.map(rewrite, stimuli))
(HERE / a.dst).write_text(json.dumps(stimuli, indent=1))
