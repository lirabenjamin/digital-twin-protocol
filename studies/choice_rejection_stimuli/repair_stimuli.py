"""Fix stimuli flagged 'fix' in an audit file with one Sonnet call each (prompts/repair.txt), then set the closing
sentence of every scenario to one neutral template (bare labels read badly: 'would choose hire a gardener or ...')."""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from common import HERE, OUTPUT, call

CLOSE = "Please think about which of the two options you would choose."
PROMPT = (HERE / "prompts" / "repair.txt").read_text()
SYSTEM = "You edit stimuli for a consumer-behavior study about how other people's decisions influence our own."
FIELDS = ("scenario", "intro", "choice_text", "rejection_text")
ap = argparse.ArgumentParser()
ap.add_argument("--audit", default="screen_audit_v2.json")
ap.add_argument("--notes", action="store_true", help="also repair entries marked note")
a = ap.parse_args()
stimuli = json.loads((HERE / "stimuli.json").read_text())
audit = json.loads((OUTPUT / a.audit).read_text())
fix = {e["stimulus"] for e in audit if e["severity"] == "fix" or a.notes}


def repair(s: dict) -> dict:
    if s["stimulus"] not in fix:
        return s
    shown = {"option_a": s["option_a"]["label"], "option_b": s["option_b"]["label"], **{k: s[k] for k in FIELDS}}
    problems = "\n".join(f"- {e['detail']} Suggested fix: {e['suggested_fix']}" for e in audit
                         if e["stimulus"] == s["stimulus"])
    text = PROMPT.replace("{stimulus}", json.dumps(shown, indent=1)).replace("{problems}", problems)
    new = call(SYSTEM, text, step=f"repair:{s['stimulus']}")
    return {**s, **{k: new[k] for k in FIELDS}}


def close(s: dict) -> dict:
    body = re.sub(r"\s*Please think about[^.]*\.\s*$", "", s["scenario"].strip())
    return {**s, "scenario": f"{body} {CLOSE}"}


with ThreadPoolExecutor(8) as ex:
    stimuli = list(ex.map(repair, stimuli))
stimuli = [close(s) for s in stimuli]
(HERE / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
print(f"repaired {len(fix)}; closing sentence set on {len(stimuli)}")
