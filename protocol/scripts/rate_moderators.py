"""Step 8. Have Claude rate every stimulus on every candidate moderator (0-10).

One isolated request per stimulus x moderator x repetition (prompts/moderator.txt), so a rating never
anchors on another feature or another stimulus. Each pair is rated `reps` times (default 3) and the cell is
the mean of the valid ratings. A reply that is not {"rating": <whole number 0-10>} counts as missing.
Replies are cached on disk, so a rerun only pays for what is missing.

    python scripts/rate_moderators.py --dry-run   # request count and one filled prompt
    python scripts/rate_moderators.py             # writes ratings_long.csv and stimulus_moderators.csv
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import ROOT, call, cfg_values, fill, load_config, parse_json, parser, template, view


def parse_rating(text: str) -> int | None:
    try:
        value = parse_json(text).get("rating")
    except ValueError:
        return None
    return value if type(value) is int and 0 <= value <= 10 else None


def prompt(cfg: dict, s: dict, m: dict) -> str:
    return fill(template(cfg, "moderator"), {**cfg_values(cfg), "stimulus": view(cfg, "rated", s),
                                             **{k: m[k] for k in ("name", "question", "anchor0", "anchor10")}})


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out, reps = cfg["output_dir"], cfg["moderators"]["reps"]
    path = out / "stimuli.json"
    stimuli = json.loads(path.read_text()) if path.exists() else [{"stimulus": s["id"], **s} for s in cfg["seeds"]]
    mods = json.loads((ROOT / cfg["moderators"]["catalog"]).read_text())
    jobs = [(s, m, rep) for s in stimuli for m in mods for rep in range(1, reps + 1)]
    print(f"{len(stimuli)} stimuli x {len(mods)} moderators x {reps} reps = {len(jobs):,} requests "
          f"({cfg['models']['rater']['name']})")
    if a.dry_run:
        if not path.exists():
            print("no stimuli.json yet: using the seeds; screen fields that step 5 writes stay as {placeholders}")
        print(f"\n--- system ---\n{cfg['moderators']['system']}\n--- user (stimulus 1, moderator 1) ---\n"
              f"{prompt(cfg, stimuli[0], mods[0])}")
        return

    def rate(job: tuple) -> dict:
        s, m, rep = job
        r = {"stimulus": s["stimulus"], "moderator": m["id"], "rep": rep, "rating": None}
        try:
            reply = call(cfg, "rater", cfg["moderators"]["system"], prompt(cfg, s, m), step="moderators", rep=rep)
            r["rating"] = parse_rating(reply["text"])
        except Exception as e:  # recorded as missing
            r["error"] = repr(e)[:200]
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        long = pd.DataFrame(ex.map(rate, jobs))
    long.to_csv(out / "ratings_long.csv", index=False)
    wide = (long.dropna(subset=["rating"]).groupby(["stimulus", "moderator"])["rating"].mean().unstack()
            .reindex(index=[s["stimulus"] for s in stimuli], columns=[m["id"] for m in mods]))
    wide.reset_index().to_csv(out / "stimulus_moderators.csv", index=False)
    print(f"missing ratings {int(long['rating'].isna().sum())}, missing cells {int(wide.isna().sum().sum())}")


if __name__ == "__main__":
    main()
