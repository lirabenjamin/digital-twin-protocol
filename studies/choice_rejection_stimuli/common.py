"""Shared helpers for choice_rejection_stimuli: model, prices, text rendering, spend log."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stimulus_augmentation"))
from stimuli import append_lines, cosine_distances, embed, parse_json  # noqa: E402,F401

MODEL, EFFORT = "claude-sonnet-5-5", "medium"
PRICE_IN, PRICE_OUT = 2.00 / 1e6, 10.00 / 1e6  # Sonnet 5.5, $ per token (output includes thinking)
BUDGET = 40.0  # at most $40 on stimulus generation (calibration included)
OUTPUT = HERE / "output"
SPEND = OUTPUT / "spend.jsonl"
SEEDS = json.loads((HERE / "seeds.json").read_text())


def line(d: dict) -> str:
    """The text that is listed in the prompt and embedded."""
    return (f"{d['situation']} | from {d['source']}, via {d['channel']} | "
            f"choice: {d['choice_text']} | rejection: {d['rejection_text']}")


def spent() -> float:
    return sum(json.loads(x)["usd"] for x in SPEND.read_text().splitlines()) if SPEND.exists() else 0.0


def call(system: str, user: str, step: str) -> dict:
    """One Sonnet call; logs its cost to spend.jsonl and refuses to start past the budget."""
    import anthropic
    if spent() >= BUDGET:
        raise RuntimeError(f"budget reached: ${spent():.2f} of ${BUDGET:.0f}")
    msg = anthropic.Anthropic(max_retries=8).messages.create(
        model=MODEL, max_tokens=16000, system=system, output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": user}])
    usd = msg.usage.input_tokens * PRICE_IN + msg.usage.output_tokens * PRICE_OUT
    OUTPUT.mkdir(exist_ok=True)
    append_lines(SPEND, [{"time": time.time(), "step": step, "input": msg.usage.input_tokens,
                          "output": msg.usage.output_tokens, "usd": usd}])
    if msg.stop_reason == "refusal":
        raise RuntimeError("refusal")
    return parse_json("".join(b.text for b in msg.content if b.type == "text"))
