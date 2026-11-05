"""Flag pairs where the choice text names Option A or the rejection text names Option B, or the two texts differ
by more than the decision phrase (a long run of differing words)."""
import difflib
import json
import re

from common import HERE

STOP = {"the", "a", "an", "with", "and", "or", "of", "for", "to", "in", "on", "at", "your", "you", "one", "plan",
        "option", "two", "both"}


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def own(label: str, other: str) -> set[str]:
    return {w for w in words(label) if w not in STOP and len(w) > 2} - set(words(other))


def problems(s: dict) -> list[str]:
    a, b = s["option_a"]["label"], s["option_b"]["label"]
    out = []
    if own(a, b) & set(words(s["choice_text"])):
        out.append(f"choice text names A: {sorted(own(a, b) & set(words(s['choice_text'])))}")
    if own(b, a) & set(words(s["rejection_text"])):
        out.append(f"rejection text names B: {sorted(own(b, a) & set(words(s['rejection_text'])))}")
    c, r = words(s["choice_text"]), words(s["rejection_text"])
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
                  difflib.SequenceMatcher(None, c, r).get_opcodes() if op != "equal")
    if changed > 8:
        out.append(f"texts differ in {changed} words")
    return out


if __name__ == "__main__":
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    bad = {s["stimulus"]: p for s in stimuli if (p := problems(s))}
    for k, p in bad.items():
        print(k, "; ".join(p))
    print(f"{len(bad)} of {len(stimuli)} flagged")
    print("IDS", " ".join(bad))
