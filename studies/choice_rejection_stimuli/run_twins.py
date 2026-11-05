"""Run Twin-2K-500 twins on the 100 picked stimuli in both frames.

Design: within twin and within stimulus. Each twin sees all 100 stimuli in both frames, and every
twin x stimulus x frame is ONE independent call (no memory of other vignettes): scenario, then intro + the
frame text, then the choice. The answer order (which option is listed first) is random per twin x stimulus
and the same in both frames. claude-haiku-5-5, effort low, persona in the cached system prompt (as in
delarosa_s2). 2,058 twins x 200 = 411,600 calls.

Outcome (7-point relative preference, 2026-11-03): rating = the raw answer (1 = strongly prefer the option
listed first, 7 = strongly prefer the one listed second); pref_b = the same answer recoded so that 7 = strongly
prefer option B (the option the other person ended up with) and 1 = strongly prefer A. The 27 twins run on
2026-11-03 with the binary A/B question are in output/pilot_binary (prompt saved there).

Two ways to send the same requests (same model, prompt and settings, so the answers do not depend on the way):
- interactive: one call at a time per twin, so its persona is read from the prompt cache;
- batch: the Message Batches API at half price. A batch holds at most 256 MB, and each request carries a
  ~130 KB persona, so one batch holds BATCH_TWINS twins. Cache hits inside a batch are best-effort, so
  check the cost of the first batch before sending more.

    python3 run_twins.py --dry-run                              # print 1 prompt, no API calls
    python3 run_twins.py --n 10 --out pilot                     # LIVE, interactive
    python3 run_twins.py --batch --start 0 --out batch000       # LIVE: submit one batch (BATCH_TWINS twins)
    python3 run_twins.py --collect --out batch000               # wait for it, write choices.csv + cost
    python3 run_twins.py --batch-all                            # LIVE: every twin in batches of BATCH_TWINS,
                                                                # resumable; then output/full/choices.csv
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import llm  # noqa: E402

MODEL, EFFORT, SEED, BATCH_TWINS = "claude-haiku-5-5", "low", 20261101, 9  # 9 x 200 x ~135 KB ~ 245 MB
IN_FLIGHT = 10  # batches submitted but not yet collected, in --batch-all
# claude-haiku-5-5, $ per million tokens (prompts under 100K tokens). Cache write 1.25x (5 min) or 2x (1 h),
# cache read 0.1x; batch halves everything. Check platform.claude.com pricing before relying on these.
PRICE = {"input": 0.10, "output": 0.50, "cache_write_5m": 0.125, "cache_write_1h": 0.20, "cache_read": 0.01}
SYSTEM = (ROOT / "studies" / "delarosa_s2" / "prompts" / "system.txt").read_text().replace("{{", "{").replace("}}", "}")
TASK = (HERE / "prompts" / "twin_task.txt").read_text()
TWINS = ROOT / "twins"
FRAMES = {"choice": "choice_text", "rejection": "rejection_text"}


def scenario_text(s: dict, order: list[int]) -> str:
    """Screen 1 as paragraphs, as in the survey: setup, the two options (each headed by its name, like the
    side-by-side boxes) in the same order as the answer scale on screen 3, then the shared sentences and the
    closing line. order_free.py made each description read well in either position."""
    lay = s["layout"]
    opts = [(s["option_a"]["label"], lay["a"]), (s["option_b"]["label"], lay["b"])]
    boxes = [f"{name[0].upper() + name[1:]}: {text}" for name, text in (opts[i] for i in order)]
    return "\n\n".join([lay["setup"], *boxes, lay["after"]])


def prompt(s: dict, frame: str, order: list[int]) -> tuple[str, list[str]]:
    labels = [s["option_a"]["label"], s["option_b"]["label"]]
    shown = [labels[i] for i in order]
    text = TASK
    for k, v in {"scenario": scenario_text(s, order), "intro": s["intro"], "frame_text": s[FRAMES[frame]],
                 "option_1": shown[0], "option_2": shown[1]}.items():
        text = text.replace("{" + k + "}", v)
    return text, ["A" if i == 0 else "B" for i in order]


def jobs(pid: str, stimuli: list[dict]) -> list[dict]:
    """One twin's 200 calls. The answer order is drawn per twin x stimulus, so it is the same in both frames."""
    rng, out = random.Random(f"{SEED}-{pid}"), []
    for s in stimuli:
        order = rng.sample([0, 1], 2)
        for frame in FRAMES:
            user, codes = prompt(s, frame, order)
            out.append({"id": f"{pid}-{s['stimulus']}-{frame}", "pid": pid, "stimulus": s["stimulus"],
                        "frame": frame, "codes": codes, "user": user})
    return out


def system_for(pid: str) -> str:
    return SYSTEM.replace("{persona}", (TWINS / f"{pid}.txt").read_text())


def parse_rating(text: str) -> int | None:
    """The number 1-7 in the reply: after a colon if the reply is JSON (e.g. {"1": 5}), else the first one."""
    m = re.search(r':\s*"?([1-7])\b', text) or re.search(r'\b([1-7])\b', text)
    return int(m.group(1)) if m else None


def row(job: dict, text: str, usage: dict) -> dict:
    rating = parse_rating(text)
    pref_b = None if rating is None else (rating if job["codes"][0] == "A" else 8 - rating)
    return {"pid": job["pid"], "stimulus": job["stimulus"], "frame": job["frame"], "first_shown": job["codes"][0],
            "reply": text[:50], "rating": rating, "pref_b": pref_b,
            **{f"tok_{k}": v for k, v in usage.items()}}


def run_interactive(pid: str, stimuli: list[dict]) -> list[dict]:
    """One twin's calls in sequence, so its persona is written to the cache once and then read."""
    system = system_for(pid)
    return [row(j, (r := llm.call(MODEL, EFFORT, system, [{"role": "user", "content": j["user"]}]))["text"],
                r["usage"]) for j in jobs(pid, stimuli)]


def batch_requests(pids: list[str], stimuli: list[dict]) -> list[dict]:
    """Same request as llm.call, with a 1-hour cache on the persona (a batch can run for up to an hour or more).
    A twin's requests sit next to each other, which gives the cache its best chance."""
    reqs = []
    for pid in pids:
        system = [{"type": "text", "text": system_for(pid), "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        for j in jobs(pid, stimuli):
            reqs.append({"custom_id": j["id"], "params": {
                "model": MODEL, "max_tokens": llm.MAX_TOKENS, "output_config": {"effort": EFFORT},
                "system": system, "messages": [{"role": "user", "content": j["user"]}]}})
    return reqs


def submit(client, pids: list[str], stimuli: list[dict], out: Path) -> str:
    b = client.messages.batches.create(requests=batch_requests(pids, stimuli))
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.json").write_text(json.dumps({"batch_id": b.id, "pids": pids}))
    return b.id


def collect(client, stimuli: list[dict], out: Path) -> pd.DataFrame:
    """Read an ended batch; ids of requests that did not succeed go to failed.json (to resend later)."""
    info = json.loads((out / "batch.json").read_text())
    index = {j["id"]: j for pid in info["pids"] for j in jobs(pid, stimuli)}
    rows, failed = [], []
    for r in client.messages.batches.results(info["batch_id"]):
        if r.result.type != "succeeded":
            failed.append(r.custom_id)
            continue
        m, u = r.result.message, r.result.message.usage
        text = "".join(blk.text for blk in m.content if blk.type == "text")
        rows.append(row(index[r.custom_id], text, {
            "input": u.input_tokens, "output": u.output_tokens,
            "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0}))
    (out / "failed.json").write_text(json.dumps(failed))
    print(f"{out.name}: {len(rows)} succeeded, {len(failed)} not", flush=True)
    return pd.DataFrame(rows)


def batch_all(stimuli: list[dict], all_pids: list[str]) -> None:
    """Every twin, BATCH_TWINS per batch, at most IN_FLIGHT batches waiting at once. Resumable: a batch folder
    with choices.csv is done; one with only batch.json is still waiting."""
    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    names = {f"batch{i // BATCH_TWINS:03d}": all_pids[i:i + BATCH_TWINS] for i in range(0, len(all_pids), BATCH_TWINS)}
    folder = lambda n: HERE / "output" / n  # noqa: E731
    todo = [n for n in names if not (folder(n) / "batch.json").exists()]
    waiting = [n for n in names if (folder(n) / "batch.json").exists() and not (folder(n) / "choices.csv").exists()]
    while todo or waiting:
        for n in list(waiting):
            bid = json.loads((folder(n) / "batch.json").read_text())["batch_id"]
            if client.messages.batches.retrieve(bid).processing_status == "ended":
                collect(client, stimuli, folder(n)).to_csv(folder(n) / "choices.csv", index=False)
                waiting.remove(n)
        if todo and len(waiting) < IN_FLIGHT:
            n = todo[0]
            try:
                print(f"{time.strftime('%H:%M')} submitted {n}: {submit(client, names[n], stimuli, folder(n))}", flush=True)
                todo.pop(0)
                waiting.append(n)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                print(f"{time.strftime('%H:%M')} {n} not submitted ({type(e).__name__}); retrying", flush=True)
                time.sleep(120)
        else:
            time.sleep(60)
    df = pd.concat(pd.read_csv(folder(n) / "choices.csv") for n in names)
    report(df, HERE / "output" / "full", batch=True)


def cost(df: pd.DataFrame, batch: bool) -> float:
    t = df.filter(like="tok_").sum()
    write = PRICE["cache_write_1h"] if batch else PRICE["cache_write_5m"]
    usd = (t["tok_input"] * PRICE["input"] + t["tok_output"] * PRICE["output"]
           + t["tok_cache_write"] * write + t["tok_cache_read"] * PRICE["cache_read"]) / 1e6
    return usd * (0.5 if batch else 1)


def report(df: pd.DataFrame, out: Path, batch: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "choices.csv", index=False)
    t = df.filter(like="tok_").sum()
    hit = t["tok_cache_read"] / (t["tok_cache_read"] + t["tok_cache_write"] + t["tok_input"])
    usd = cost(df, batch)
    print(f"{len(df)} calls, {df['pid'].nunique()} twins; unparsed {df['rating'].isna().sum()}")
    print(f"cache hit rate {hit:.1%}; cost ${usd:.2f} (${usd / df['pid'].nunique():.3f} per twin, "
          f"~${usd / df['pid'].nunique() * 2058:.0f} for 2,058 twins)")
    print("pref_b (1 = strongly A, 7 = strongly B) by frame:")
    print(df.groupby("frame")["pref_b"].agg(["mean", "std"]).round(2))
    print("rating distribution:", df["rating"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="twins, interactive mode")
    ap.add_argument("--start", type=int, default=0, help="first twin (index into the sorted twin list), batch mode")
    ap.add_argument("--out", default="pilot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--batch-all", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    all_pids = sorted((f.stem for f in TWINS.glob("*.txt")), key=int)
    out = HERE / "output" / a.out
    if not all_pids:
        raise SystemExit(f"No twins found in {TWINS.relative_to(ROOT)}/. From the repository root, run "
                         "python3 code/download_twins.py first (about 270 MB, no API key needed).")

    if a.dry_run:
        print(jobs(all_pids[0], stimuli)[0]["user"])
        reqs = batch_requests(all_pids[:BATCH_TWINS], stimuli)
        mb = len(json.dumps({"requests": reqs}).encode()) / 1e6
        print(f"\n{len(all_pids) * 2 * len(stimuli)} calls for {len(all_pids)} twins; one batch = {BATCH_TWINS} twins, "
              f"{len(reqs)} requests, {mb:.0f} MB (limit 256 MB)")
    elif a.batch_all:
        batch_all(stimuli, all_pids)
    elif a.batch:
        import anthropic
        pids = all_pids[a.start:a.start + BATCH_TWINS]
        print(f"submitted {submit(anthropic.Anthropic(), pids, stimuli, out)}: twins {pids[0]}..{pids[-1]}")
    elif a.collect:
        import anthropic
        client, info = anthropic.Anthropic(), json.loads((out / "batch.json").read_text())
        while (b := client.messages.batches.retrieve(info["batch_id"])).processing_status != "ended":
            print(f"{b.processing_status}: {b.request_counts.processing} processing"); time.sleep(60)
        report(collect(client, stimuli, out), out, batch=True)
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(lambda p: run_interactive(p, stimuli), all_pids[:a.n]) for r in rs]
        report(pd.DataFrame(rows), out, batch=False)
