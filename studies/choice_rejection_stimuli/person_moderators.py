"""Score 10 person-level traits from the Twin-2K-500 profiles (twins/<pid>.txt). No API calls.

    python3 studies/choice_rejection_stimuli/person_moderators.py   # writes output/person/traits.csv

Each Likert scale is found by the text of its first item, then read as the run of "N. text / Answer: k" lines
that follows. Scores are item means after reverse-keying (reversed = min + max - k). A scale score is NA if more
than 20% of its items are missing ("[No Answer Provided]").

  uniqueness    Consumer Need for Uniqueness, short form (Ruvio, Shoham & Makovec Brencic 2008), 12 items,
                1-5, none reversed (creative choice, unpopular choice, avoidance of similarity; 4 items each)
  self_monitor  Revised Self-Monitoring Scale (Lennox & Wolfe 1984), 13 items, 1-6 here (0-5 in the original),
                items 4 and 6 reversed
  collectivism  Horizontal + vertical collectivism from the 16-item culture orientation scale (Triandis &
                Gelfand 1998), items 9-16, 1-5. The profile has this scale, not Singelis's (1994)
                independent/interdependent self-construal scale, so collectivism stands in for interdependence.
  nfc           Need for Cognition, 18-item short form (Cacioppo, Petty & Kao 1984), 1-5,
                items 3, 4, 5, 7, 8, 9, 12, 16, 17 reversed
  closure       Need for Closure, 15-item short form (Roets & Van Hiel 2011), 1-5, none reversed
  maximizing    Maximization Scale, 6-item short form (Nenkov, Morrin, Ward, Schwartz & Hulland 2008),
                1-5 here (1-7 in the original), none reversed
  clarity       Self-Concept Clarity (Campbell et al. 1996), 12 items, 1-5, all reversed except 6 and 11
                (higher = clearer self-concept)
  agreeable     BFI-44 Agreeableness (John, Donahue & Kentle 1991), items 2R 7 12R 17 22 27R 32 37R 42, 1-5
  crt           Cognitive reflection: correct answers out of 4 on the CRT-2 (Thomson & Oppenheimer 2016):
                Emily's father (Emily), hole of dirt (0), passing second place (2), sheep (8)
  age           age band from twins/index.csv (1 = 18-29, 2 = 30-49, 3 = 50-64, 4 = 65+)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TWINS = HERE.parents[1] / "twins"
OUT = HERE / "output" / "person" / "traits.csv"

# name: (regex for item 1's text, items in the block, keep these items (None = all), reversed items, min, max)
SCALES = {
    "uniqueness": (r"I often combine possessions in such a way", 12, None, [], 1, 5),
    "self_monitor": (r"In social situations, I have the ability to alter my behavior", 13, None, [4, 6], 1, 6),
    "collectivism": (r"I.d rather depend on myself than others", 16, range(9, 17), [], 1, 5),
    "nfc": (r"I would prefer complex to simple problems", 18, None, [3, 4, 5, 7, 8, 9, 12, 16, 17], 1, 5),
    "closure": (r"I don.t like situations that are uncertain", 15, None, [], 1, 5),
    "maximizing": (r"When I am in the car listening to the radio", 6, None, [], 1, 5),
    "clarity": (r"My beliefs about myself often conflict with one another", 12, None,
                [1, 2, 3, 4, 5, 7, 8, 9, 10, 12], 1, 5),
    "agreeable": (r"Is talkative", 44, [2, 7, 12, 17, 22, 27, 32, 37, 42], [2, 12, 27, 37], 1, 5),
}
ITEM = re.compile(r"(\d+)\. (.*)\nAnswer: (.*)")
CRT = [(r"Emily.s father has three daughters", lambda a: a.strip().lower() == "emily"),
       (r"How many cubic feet of dirt are there in a hole", lambda a: num(a) == 0),
       (r"If you.re running a race and you pass the person in second place", lambda a: num(a) == 2),
       (r"A farmer had 15 sheep and all but 8 died", lambda a: num(a) == 8)]
AGE = {"18-29": 1, "30-49": 2, "50-64": 3, "65+": 4}


def num(a: str) -> float | None:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", a)
    return float(m.group(1)) if m else None


def block(text: str, anchor: str, n: int) -> list[tuple[str, int | None]]:
    """The n items starting at the item whose text matches anchor, as (item text, answer or None)."""
    m = re.search(r"^1\. " + anchor, text, re.M)
    if not m:
        return []
    items = []
    for k, it in enumerate(ITEM.finditer(text, m.start()), start=1):
        if int(it.group(1)) != k or k > n:
            break
        items.append((it.group(2).strip(), int(a.group()) if (a := re.match(r"\d+", it.group(3))) else None))
    return items


def answer_after(text: str, stem: str) -> str | None:
    m = re.search(stem, text)
    a = m and re.search(r"^Answer: (.*)$", text[m.end():], re.M)
    return a.group(1) if a else None


def score(text: str) -> tuple[dict, dict]:
    row, raw = {}, {}
    for name, (anchor, n, keep, rev, lo, hi) in SCALES.items():
        items = block(text, anchor, n)
        raw[name] = items
        if len(items) != n:
            row[name] = np.nan
            continue
        x = [(lo + hi - a if i in rev else a) if a is not None else np.nan
             for i, (_, a) in enumerate(items, start=1) if keep is None or i in keep]
        raw[name + "_keyed"] = x
        row[name] = np.nanmean(x) if np.isnan(x).mean() <= 0.2 else np.nan
    ans = [answer_after(text, stem) for stem, _ in CRT]
    row["crt"] = sum(ok(a) for a, (_, ok) in zip(ans, CRT)) if all(a is not None for a in ans) else np.nan
    return row, raw


def alpha(m: np.ndarray) -> float:
    m = m[~np.isnan(m).any(axis=1)]
    k = m.shape[1]
    return k / (k - 1) * (1 - m.var(axis=0, ddof=1).sum() / m.sum(axis=1).var(ddof=1))


idx = pd.read_csv(TWINS / "index.csv")
rows, keyed, texts = [], {s: [] for s in SCALES}, {s: {} for s in SCALES}
for pid in idx.pid:
    row, raw = score((TWINS / f"{pid}.txt").read_text())
    rows.append({"pid": pid, **row})
    for s in SCALES:
        if s + "_keyed" in raw:
            keyed[s].append(raw[s + "_keyed"])
        for i, (t, _) in enumerate(raw[s], start=1):
            texts[s].setdefault(i, set()).add(t)
out = pd.DataFrame(rows)
out["age"] = idx.age.map(AGE).values
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("Item check (items found per twin; distinct item texts per position across twins):")
for s, (_, n, keep, *_) in SCALES.items():
    m = np.array(keyed[s], dtype=float)
    variants = max(len(v) for v in texts[s].values())
    print(f"  {s:13s} block {n} items, scored {m.shape[1]}, twins parsed {len(m)}/{len(out)}, "
          f"max text variants per item {variants}, missing item answers {int(np.isnan(m).sum())}, "
          f"alpha {alpha(m):.2f}")
crt = np.array([[ok(answer_after((TWINS / f"{p}.txt").read_text(), st) or "") for st, ok in CRT]
                for p in idx.pid], dtype=float)
print(f"  {'crt':13s} 4 items, share correct per item {crt.mean(0).round(2).tolist()}, KR-20 alpha {alpha(crt):.2f}")
print("\nMissing per trait:", out.drop(columns="pid").isna().sum().to_dict())
print(out.drop(columns="pid").describe().round(2).to_string())
print("\nCRT distribution:", out.crt.value_counts().sort_index().to_dict())
print("Age distribution:", out.age.value_counts().sort_index().to_dict())
