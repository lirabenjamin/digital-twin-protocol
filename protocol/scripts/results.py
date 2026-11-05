"""Step 7. Estimate the effect overall, per stimulus and per twin, with reliability and heterogeneity.

Effect = DV in the second condition minus DV in the first, paired within twin x stimulus (the "cell").

  d_av, d_z         mean effect / average SD of the two conditions; mean effect / SD of the cell effects
  Cochran's Q, I2   do the stimulus effects differ more than sampling error allows? I2 = share of their
                    spread that is true heterogeneity; tau = SD of the true stimulus effects
  variance shares   two-way random-effects decomposition of the cell effects (method of moments): stimulus,
                    twin, and residual (twin x stimulus interaction plus response noise; with one response
                    per cell the two cannot be separated)
  split-half        stimulus effects from a random half of the twins vs. the other half (Spearman-Brown
                    corrected); the same for twin effects over a random half of the stimuli

    python scripts/results.py --dry-run              # what it reads and writes
    python scripts/results.py                        # reads <output_dir>/choices.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import load_config, parser


def split_half(cells: pd.DataFrame, unit: str, other: str, seed: int = 1) -> float:
    keys = cells[other].unique()
    half = set(np.random.default_rng(seed).permutation(keys)[: len(keys) // 2])
    mask = cells[other].isin(half)
    r = cells[mask].groupby(unit)["effect"].mean().corr(cells[~mask].groupby(unit)["effect"].mean())
    return 2 * r / (1 + r)


def analyze(d: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = d.dropna(subset=["y"])
    cells = d.pivot_table(index=["pid", "stimulus"], columns="treat", values="y").dropna().reset_index()
    cells["effect"] = cells[1] - cells[0]

    st = cells.groupby("stimulus").agg(n=("effect", "size"), mean_0=(0, "mean"), mean_1=(1, "mean"),
                                       effect=("effect", "mean"), sd=("effect", "std"))
    st["se"] = st["sd"] / np.sqrt(st["n"])
    st["ci_lo"], st["ci_hi"] = st["effect"] - 1.96 * st["se"], st["effect"] + 1.96 * st["se"]
    tw = cells.groupby("pid")["effect"].agg(["mean", "std", "count"])
    tw["p"] = 2 * stats.t.sf(np.abs(tw["mean"] / (tw["std"] / np.sqrt(tw["count"]))), tw["count"] - 1)

    w = 1 / st["se"] ** 2  # Cochran's Q across stimuli
    q = float((w * (st["effect"] - (w * st["effect"]).sum() / w.sum()) ** 2).sum())
    q_df = len(st) - 1
    tau = np.sqrt(max(0.0, (q - q_df) / (w.sum() - (w ** 2).sum() / w.sum())))

    wide = cells.pivot(index="pid", columns="stimulus", values="effect")  # variance decomposition
    n_t, n_s = wide.notna().sum(axis=0).mean(), wide.notna().sum(axis=1).mean()
    grand = cells["effect"].mean()
    ms_stim = n_t * ((wide.mean(axis=0) - grand) ** 2).sum() / (wide.shape[1] - 1)
    ms_twin = n_s * ((wide.mean(axis=1) - grand) ** 2).sum() / (wide.shape[0] - 1)
    resid = wide.sub(wide.mean(axis=1), axis=0).sub(wide.mean(axis=0), axis=1) + grand
    ms_res = (resid ** 2).sum().sum() / ((wide.shape[0] - 1) * (wide.shape[1] - 1))
    v_stim, v_twin = max(0.0, (ms_stim - ms_res) / n_t), max(0.0, (ms_twin - ms_res) / n_s)
    v_tot = v_stim + v_twin + ms_res

    res = {"n_twins": int(d["pid"].nunique()), "n_stimuli": int(d["stimulus"].nunique()), "n_cells": len(cells),
           "n_unparsed": int(d["rating"].isna().sum()) if "rating" in d else 0,
           "mean_condition_0": d.loc[d.treat == 0, "y"].mean(), "mean_condition_1": d.loc[d.treat == 1, "y"].mean(),
           "effect_mean": grand, "d_av": grand / d.groupby("treat")["y"].std().mean(),
           "d_z": grand / cells["effect"].std(),
           "stimuli_positive": int((st["effect"] > 0).sum()), "stimuli_ci_above_0": int((st["ci_lo"] > 0).sum()),
           "stim_effect_min": st["effect"].min(), "stim_effect_max": st["effect"].max(),
           "stim_q": q, "stim_q_df": q_df, "stim_q_p": float(stats.chi2.sf(q, q_df)), "stim_tau": tau,
           "stim_i2_pct": max(0.0, (q - q_df) / q) * 100,
           "var_pct_stimulus": v_stim / v_tot * 100, "var_pct_twin": v_twin / v_tot * 100,
           "var_pct_residual": ms_res / v_tot * 100,
           "twins_positive": int((tw["mean"] > 0).sum()), "twins_p05_pct": float((tw["p"] < .05).mean() * 100),
           "reliability_stimulus_effects": split_half(cells, "stimulus", "pid"),
           "reliability_twin_effects": split_half(cells, "pid", "stimulus")}
    res = {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in res.items()}
    return res, st.reset_index(), tw.reset_index().rename(columns={"mean": "effect", "count": "n"})


def main():
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    folder = load_config(a.config)["output_dir"]
    if a.dry_run:
        print(f"reads {folder / 'choices.csv'} (exists: {(folder / 'choices.csv').exists()}); writes results.json, "
              f"stimulus_effects.csv, twin_effects.csv there. No API calls.")
        return
    res, st, tw = analyze(pd.read_csv(folder / "choices.csv", dtype={"pid": str}))
    (folder / "results.json").write_text(json.dumps(res, indent=1))
    st.sort_values("effect", ascending=False).to_csv(folder / "stimulus_effects.csv", index=False)
    tw.to_csv(folder / "twin_effects.csv", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
