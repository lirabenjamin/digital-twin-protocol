"""Step 6. Run every twin on every stimulus in both conditions.

Each twin x stimulus x condition is ONE isolated request: the persona in the system prompt
(prompts/twin_system.txt) and the screens in the user message (the config's twin_task template). The twin
has no memory of its other stimuli, so there are no order or carry-over effects, and a within-twin design
costs nothing in independence. If the config sets dv.counterbalance, the two options are listed in random
order per twin x stimulus (the same order in both conditions) and the answer is recoded so that the top of
the scale always means the second counterbalanced field.

Two ways to send the same requests:
  interactive  one call at a time per twin, so the persona is read from the prompt cache after the first call;
  batch        the Message Batches API at half price, with a 1-hour cache on the persona. Batches are sized to
               stay under the API limits (100,000 requests, 256 MB). --batch-all submits them all, keeps at
               most --in-flight waiting, and can be stopped and restarted at any time.

    python scripts/run_twins.py --dry-run              # request count, batch size, one filled prompt
    python scripts/run_twins.py --n 5                  # LIVE pilot, interactive: writes pilot/choices.csv
    python scripts/run_twins.py --batch-all            # LIVE full run: writes batches/*, then choices.csv
"""
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import MAX_TOKENS, TWINS, call, cost, fill, load_config, parser, template

MAX_REQUESTS, MAX_BYTES = 100_000, 200e6  # per batch; the API limit is 256 MB, so keep a margin
PERSONA_CHARS = 135_000  # a Twin-2K-500 persona is 126-134 KB; used to size batches before the download


def twin_ids(cfg: dict) -> list[str]:
    pids = sorted((f.stem for f in TWINS.glob("*.txt")), key=lambda p: (len(p), p))
    return pids[:cfg["twins"]["n"]]


def persona(pid: str) -> str:
    path = TWINS / f"{pid}.txt"
    return path.read_text() if path.exists() else f"<persona of twin {pid}: run code/download_twins.py>"


def jobs(cfg: dict, pid: str, stimuli: list[dict]) -> list[dict]:
    """One twin's requests. The option order is drawn per twin x stimulus, so it is the same in both conditions."""
    rng, task, cb, out = random.Random(f"{cfg['twins']['seed']}-{pid}"), template(cfg, "twin_task"), cfg["dv"].get("counterbalance"), []
    for s in stimuli:
        shown = rng.sample(cb, 2) if cb else []
        for treat, c in enumerate(cfg["conditions"]):
            values = {**s, "condition_text": s[c["text_field"]]}
            values.update({f"option_{i + 1}": s[f] for i, f in enumerate(shown)})
            out.append({"id": f"{pid}-{s['stimulus']}-{c['name']}", "pid": pid, "stimulus": s["stimulus"],
                        "condition": c["name"], "treat": treat, "reversed": bool(cb) and shown[1] != cb[1],
                        "user": fill(task, values)})
    return out


def parse_answer(text: str, lo: int, hi: int) -> int | None:
    """The first whole number on the scale in the reply."""
    for m in re.finditer(r"-?\d+", text):
        if lo <= int(m.group()) <= hi:
            return int(m.group())
    return None


def row(cfg: dict, job: dict, text: str, usage: dict) -> dict:
    lo, hi = cfg["dv"]["scale"]
    rating = parse_answer(text, lo, hi)
    y = None if rating is None else (lo + hi - rating if job["reversed"] else rating)
    return {k: job[k] for k in ("pid", "stimulus", "condition", "treat", "reversed")} | {
        "reply": text[:50], "rating": rating, "y": y, **{f"tok_{k}": v for k, v in usage.items()}}


def run_interactive(cfg: dict, pid: str, stimuli: list[dict]) -> list[dict]:
    system = fill(template(cfg, "twin_system"), {"persona": persona(pid)})
    return [row(cfg, j, (r := call(cfg, "twin", system, j["user"], step="twins"))["text"], r["usage"])
            for j in jobs(cfg, pid, stimuli)]


def batch_requests(cfg: dict, pids: list[str], stimuli: list[dict]) -> list[dict]:
    """A twin's requests sit next to each other, which gives the prompt cache its best chance."""
    m, reqs = cfg["models"]["twin"], []
    for pid in pids:
        system = [{"type": "text", "text": fill(template(cfg, "twin_system"), {"persona": persona(pid)}),
                   "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        for j in jobs(cfg, pid, stimuli):
            reqs.append({"custom_id": j["id"], "params": {
                "model": m["name"], "max_tokens": MAX_TOKENS, "output_config": {"effort": m["effort"]},
                "system": system, "messages": [{"role": "user", "content": j["user"]}]}})
    return reqs


def twins_per_batch(cfg: dict, pids: list[str], stimuli: list[dict]) -> int:
    per_twin = batch_requests(cfg, pids[:1], stimuli)
    size = len(json.dumps(per_twin).encode())
    if not (TWINS / f"{pids[0]}.txt").exists():  # placeholder persona: add the size of a real one
        size += PERSONA_CHARS * len(per_twin)
    return max(1, min(MAX_REQUESTS // len(per_twin), int(MAX_BYTES // size)))


def collect(cfg: dict, client, stimuli: list[dict], folder: Path) -> pd.DataFrame:
    """Read an ended batch. Ids of requests that did not succeed go to failed.json, to resend later."""
    info = json.loads((folder / "batch.json").read_text())
    index = {j["id"]: j for pid in info["pids"] for j in jobs(cfg, pid, stimuli)}
    rows, failed = [], []
    for r in client.messages.batches.results(info["batch_id"]):
        if r.result.type != "succeeded":
            failed.append(r.custom_id)
            continue
        msg, u = r.result.message, r.result.message.usage
        rows.append(row(cfg, index[r.custom_id], "".join(b.text for b in msg.content if b.type == "text"), {
            "input": u.input_tokens, "output": u.output_tokens,
            "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0}))
    (folder / "failed.json").write_text(json.dumps(failed))
    print(f"{folder.name}: {len(rows)} succeeded, {len(failed)} not", flush=True)
    return pd.DataFrame(rows)


def batch_all(cfg: dict, stimuli: list[dict], pids: list[str], in_flight: int) -> None:
    """Resumable: a batch folder with choices.csv is done; one with only batch.json is still waiting."""
    import anthropic
    from common import load_env
    load_env()
    client, size = anthropic.Anthropic(max_retries=8), twins_per_batch(cfg, pids, stimuli)
    root = cfg["output_dir"] / "batches"
    groups = {f"batch{i // size:03d}": pids[i:i + size] for i in range(0, len(pids), size)}
    todo = [n for n in groups if not (root / n / "batch.json").exists()]
    waiting = [n for n in groups if (root / n / "batch.json").exists() and not (root / n / "choices.csv").exists()]
    while todo or waiting:
        for n in list(waiting):
            bid = json.loads((root / n / "batch.json").read_text())["batch_id"]
            if client.messages.batches.retrieve(bid).processing_status == "ended":
                collect(cfg, client, stimuli, root / n).to_csv(root / n / "choices.csv", index=False)
                waiting.remove(n)
        if todo and len(waiting) < in_flight:
            n = todo.pop(0)
            b = client.messages.batches.create(requests=batch_requests(cfg, groups[n], stimuli))
            (root / n).mkdir(parents=True, exist_ok=True)
            (root / n / "batch.json").write_text(json.dumps({"batch_id": b.id, "pids": groups[n]}))
            print(f"{time.strftime('%H:%M')} submitted {n} ({len(groups[n])} twins): {b.id}", flush=True)
            waiting.append(n)
        else:
            time.sleep(60)
    df = pd.concat(pd.read_csv(root / n / "choices.csv", dtype={"pid": str}) for n in groups)
    report(cfg, df, cfg["output_dir"], batch=True)


def report(cfg: dict, df: pd.DataFrame, out: Path, batch: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "choices.csv", index=False)
    t = {k: df[f"tok_{k}"].sum() for k in ("input", "output", "cache_write", "cache_read")}
    m = dict(cfg["models"]["twin"])
    if batch:  # batches use the 1-hour cache, which costs more to write
        m["price"] = {**m["price"], "cache_write": m["price"]["cache_write_1h"]}
    usd, n = cost(m, t, batch), df["pid"].nunique()
    hit = t["cache_read"] / max(1, t["cache_read"] + t["cache_write"] + t["input"])
    print(f"{len(df)} calls, {n} twins; unparsed answers {df['rating'].isna().sum()}")
    print(f"cache hit rate {hit:.1%}; cost ${usd:.2f} (${usd / n:.3f} per twin)")
    print(df.groupby("condition")["y"].agg(["mean", "std"]).round(2))


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="interactive pilot on the first n twins")
    ap.add_argument("--batch-all", action="store_true")
    ap.add_argument("--in-flight", type=int, default=10, help="batches waiting at once")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = cfg["output_dir"] / "stimuli.json"
    stimuli = json.loads(path.read_text()) if path.exists() else [{"stimulus": s["id"], **s} for s in cfg["seeds"]]
    pids = twin_ids(cfg)
    if a.dry_run:
        n_twins = len(pids) or cfg["twins"]["n"]
        pids = pids or ["1"]
        per_twin = len(stimuli) * len(cfg["conditions"])
        print(f"{len(stimuli)} stimuli x {len(cfg['conditions'])} conditions = {per_twin} requests per twin; "
              f"{n_twins} twins = {n_twins * per_twin:,} requests ({cfg['models']['twin']['name']})")
        print(f"one batch = {twins_per_batch(cfg, pids, stimuli)} twins "
              f"(limits: {MAX_REQUESTS:,} requests, {MAX_BYTES / 1e6:.0f} MB)")
        if not path.exists():
            print("no stimuli.json yet: using the seeds; screen fields that step 5 writes stay as {placeholders}")
        j = jobs(cfg, pids[0], stimuli)[1]
        system = fill(template(cfg, "twin_system"), {"persona": persona(pids[0])})
        print(f"\n--- system (twin {pids[0]}, {len(system):,} characters; first 400) ---\n{system[:400]}\n...")
        print(f"--- user ({j['id']}, option order reversed: {j['reversed']}) ---\n{j['user']}")
    elif a.batch_all:
        batch_all(cfg, stimuli, pids, a.in_flight)
    elif a.n:
        with ThreadPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(lambda p: run_interactive(cfg, p, stimuli), pids[:a.n]) for r in rs]
        report(cfg, pd.DataFrame(rows), cfg["output_dir"] / "pilot", batch=False)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
