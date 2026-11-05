"""Download the Twin-2K-500 digital twins and save one plain-text persona per person.

Source: Hugging Face dataset LLM-Digital-Twin/Twin-2K-500, config `full_persona`
(Toubia et al. 2025, CC-BY-4.0). The config ships as 7 parquet chunks with four
columns: pid, persona_text, persona_summary, persona_json.

We keep `persona_text`, the full survey transcript, because it is the dataset's
default "text persona". Every study reads twins/<pid>.txt and never the parquet
files, so swapping in a different persona field later only touches this script.

Output (gitignored):
    twins/<pid>.txt   the persona_text, unchanged
    twins/index.csv   pid, persona length, and the 14 demographic answers

The demographics come from the "Demographics" block of persona_json, the same
answers that appear at the top of persona_text, so studies can sample or stratify
twins without parsing the text.

    python3 code/download_twins.py
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TWINS = ROOT / "twins"
PARQUET_CACHE = ROOT / "cache" / "twin2k500"
BASE_URL = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/full_persona/chunks"
N_CHUNKS = 7
N_TWINS = 2058

# Demographics block question IDs -> short column names for index.csv.
DEMOGRAPHICS = {
    "QID11": "region", "QID12": "sex", "QID13": "age", "QID14": "education",
    "QID15": "race", "QID16": "us_citizen", "QID17": "marital_status",
    "QID18": "religion", "QID19": "religious_attendance", "QID20": "party",
    "QID21": "family_income", "QID22": "political_views",
    "QID23": "household_size", "QID24": "employment",
}


def download_chunks() -> list[Path]:
    """Fetch the parquet chunks once; later runs reuse the local copies."""
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(1, N_CHUNKS + 1):
        path = PARQUET_CACHE / f"persona_chunk_{k:03d}.parquet"
        if not path.exists():
            print(f"downloading {path.name}")
            urllib.request.urlretrieve(f"{BASE_URL}/{path.name}", path)
        paths.append(path)
    return paths


def demographics(persona_json: str) -> dict:
    """Pull the selected answer text for each Demographics question."""
    blocks = json.loads(persona_json)
    demo = next(b for b in blocks if b.get("BlockName", "").strip() == "Demographics")
    out = {}
    for q in demo["Questions"]:
        if q["QuestionID"] in DEMOGRAPHICS:
            out[DEMOGRAPHICS[q["QuestionID"]]] = q.get("Answers", {}).get("SelectedText")
    return out


def main():
    TWINS.mkdir(exist_ok=True)
    rows = []
    for path in download_chunks():
        chunk = pd.read_parquet(path, columns=["pid", "persona_text", "persona_json"])
        for r in chunk.itertuples():
            pid = str(r.pid)
            (TWINS / f"{pid}.txt").write_text(r.persona_text)
            rows.append({"pid": pid, "persona_chars": len(r.persona_text), **demographics(r.persona_json)})
    index = pd.DataFrame(rows).sort_values("pid", key=lambda s: s.astype(int))
    assert len(index) == N_TWINS and index.pid.is_unique, f"expected {N_TWINS} unique twins, got {len(index)}"
    index.to_csv(TWINS / "index.csv", index=False)
    print(f"wrote {len(index)} personas to {TWINS}/ and index.csv")


if __name__ == "__main__":
    main()
