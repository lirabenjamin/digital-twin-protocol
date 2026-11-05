"""Rate the 100 stimuli on the 100 candidate moderators in moderators.json, as for De La Rosa
(studies/delarosa_stimuli/code_features.py): claude-haiku-5-5 at effort low, one isolated request per
stimulus x moderator x repetition (no cross-feature anchoring), 3 repetitions, a cell is the mean of its valid
repetitions. Requests are cached on disk by code/llm.py. An unparseable reply is recorded as missing.

    python3 code_moderators.py            # request count and one filled prompt, no API calls
    python3 code_moderators.py --run      # LIVE: 30,000 requests -> output/moderators/
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from run_twins import HERE, ROOT

sys.path.insert(0, str(ROOT / "src"))
from stimulus_feature_catalog import parse_rating  # noqa: E402

MODEL, EFFORT, REPS = "claude-haiku-5-5", "low", 3
SYSTEM = "You rate everyday decisions for a behavioral study."
PROMPT = (HERE / "prompts" / "moderator.txt").read_text()
OUT = HERE / "output" / "moderators"


def scenario(s: dict) -> str:
    lay = s["layout"]
    return " ".join([lay["setup"], lay["a"], lay["b"], lay["after"]])


def prompt(s: dict, m: dict) -> str:
    fill = {"scenario": scenario(s), "option_a": s["option_a"]["label"], "option_b": s["option_b"]["label"],
            "intro": s["intro"], "choice_text": s["choice_text"], "rejection_text": s["rejection_text"],
            **{k: m[k] for k in ("name", "question", "anchor0", "anchor10")}}
    return re.sub(r"\{(\w+)\}", lambda x: str(fill.get(x.group(1), x.group(0))), PROMPT)


def rate(job: tuple) -> dict:
    import llm
    s, m, rep = job
    out = {"stimulus": s["stimulus"], "moderator": m["id"], "rep": rep, "rating": None}
    try:
        out["rating"] = parse_rating(llm.call(MODEL, EFFORT, SYSTEM, [{"role": "user", "content": prompt(s, m)}],
                                              rep=rep)["text"])
    except Exception as exc:  # noqa: BLE001 - recorded as missing, reported
        out["error"] = repr(exc)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    mods = json.loads((HERE / "moderators.json").read_text())
    assert len(stimuli) == 100 and len(mods) == 100 and len({m["id"] for m in mods}) == 100
    jobs = [(s, m, rep) for s in stimuli for m in mods for rep in range(1, REPS + 1)]
    print(f"{len(stimuli)} stimuli x {len(mods)} moderators x {REPS} reps = {len(jobs):,} requests "
          f"({MODEL}, effort {EFFORT})")
    if not a.run:
        print("\n--- system ---\n" + SYSTEM + "\n--- user (stimulus 1, moderator 1) ---\n" + prompt(stimuli[0], mods[0]))
        return
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        long = pd.DataFrame(pool.map(rate, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT / "ratings_long.csv", index=False)
    wide = (long.dropna(subset=["rating"]).groupby(["stimulus", "moderator"])["rating"].mean().unstack("moderator")
            .reindex(index=[s["stimulus"] for s in stimuli], columns=[m["id"] for m in mods]))
    wide.reset_index().to_csv(OUT / "stimulus_moderators.csv", index=False)
    print(f"failed requests {int(long['rating'].isna().sum())}, missing cells {int(wide.isna().sum().sum())}; "
          f"wrote {OUT}")


if __name__ == "__main__":
    main()
