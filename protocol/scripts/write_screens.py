"""Step 5. Write the screens for each picked stimulus.

One Claude call per stimulus (prompts/screens.txt) adds the fields in `screens.fields` of the config (in the
example: the scenario and a one-line intro). The condition texts themselves are not rewritten, so the two
conditions still differ only where the generator made them differ. Stimuli that already have screens are
skipped, so a rerun only fills the gaps.

    python scripts/write_screens.py --dry-run   # request count and one filled prompt
    python scripts/write_screens.py             # adds the screen fields to stimuli.json
"""
import json
from concurrent.futures import ThreadPoolExecutor

from common import call, cfg_values, field_schema, fill, load_config, parse_json, parser, template, view


def prompt(cfg: dict, s: dict) -> str:
    sc = cfg["screens"]
    fields = "\n".join(f'- "{k}": {fill(v, s)}' for k, v in sc["fields"].items())
    return fill(template(cfg, "screens"), {**cfg_values(cfg), "stimulus": view(cfg, "full", s), "fields": fields,
                                           "rules": fill(sc["rules"], s).strip(),
                                           "schema": field_schema({k: "..." for k in sc["fields"]})})


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = cfg["output_dir"] / "stimuli.json"
    stimuli = json.loads(path.read_text()) if path.exists() else cfg["seeds"]
    keys = list(cfg["screens"]["fields"])
    todo = [s for s in stimuli if not all(k in s for k in keys)]
    if a.dry_run:
        source = path if path.exists() else "the seeds (no stimuli.json yet)"
        print(f"{len(todo)} of {len(stimuli)} stimuli need screens ({cfg['models']['generator']['name']}); "
              f"read from {source}\n")
        print(f"--- system ---\n{cfg['screens']['system']}\n--- user (stimulus 1) ---\n{prompt(cfg, stimuli[0])}")
        return

    def write(s: dict) -> dict:
        if all(k in s for k in keys):
            return s
        reply = parse_json(call(cfg, "generator", cfg["screens"]["system"], prompt(cfg, s), step="screens")["text"])
        return {**s, **{k: reply[k] for k in keys}}

    if not path.exists():
        raise SystemExit(f"{path} not found: run select_stimuli.py first")
    with ThreadPoolExecutor(a.workers) as ex:
        stimuli = list(ex.map(write, stimuli))
    path.write_text(json.dumps(stimuli, indent=1))
    print(f"{sum(all(k in s for k in keys) for s in stimuli)} of {len(stimuli)} stimuli have screens")


if __name__ == "__main__":
    main()
