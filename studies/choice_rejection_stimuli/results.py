"""Headline numbers for the twin study (7-point DV) -> output/full/results.json.
Effect = rejection minus choice in pref_b (1-7, 7 = strongly prefer B, the option the other person ended up with),
paired within twin x stimulus. Split-half reliabilities use a fixed random split."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from run_twins import HERE

OUT = HERE / "output" / "full"
d = pd.read_csv(OUT / "choices.csv")
cells = (d.dropna(subset=["pref_b"]).pivot_table(index=["pid", "stimulus"], columns="frame", values="pref_b")
         .dropna().reset_index())
cells["effect"] = cells["rejection"] - cells["choice"]
by_stim = cells.groupby("stimulus")["effect"].mean()
by_twin = cells.groupby("pid")["effect"].mean()


def split_half(unit: str, other: str) -> float:
    """Correlation of `unit` effects computed from two random halves of `other` (Spearman-Brown corrected)."""
    rng = np.random.default_rng(20261104)
    keys = cells[other].unique()
    half = set(rng.permutation(keys)[: len(keys) // 2])
    mask = cells[other].isin(half)
    r = cells[mask].groupby(unit)["effect"].mean().corr(cells[~mask].groupby(unit)["effect"].mean())
    return 2 * r / (1 + r)


# Cohen's d: d_av = mean paired difference / average SD of the two frames (comparable to a between-subjects d);
# d_z = mean paired difference / SD of the paired differences (twin x stimulus cells)
sd_frames = d.groupby("frame")["pref_b"].std().mean()
d_av, d_z = cells["effect"].mean() / sd_frames, cells["effect"].mean() / cells["effect"].std()

# heterogeneity across stimuli: Cochran's Q on the 100 stimulus effects, tau = SD of the true stimulus effects
st = cells.groupby("stimulus")["effect"].agg(["mean", "var", "count"])
se2 = st["var"] / st["count"]
w = 1 / se2
q = float((w * (st["mean"] - (w * st["mean"]).sum() / w.sum()) ** 2).sum())
q_df = len(st) - 1
tau = np.sqrt(max(0.0, (q - q_df) / (w.sum() - (w ** 2).sum() / w.sum())))

# I^2: share of the spread in the 100 stimulus effects that is true heterogeneity, not sampling error
i2 = max(0.0, (q - q_df) / q) * 100

# variance decomposition of the twin x stimulus effects (two-way random effects, method of moments, one rep
# per cell): stimulus, twin, and residual (twin x stimulus interaction plus response noise, not separable)
wide = cells.pivot(index="pid", columns="stimulus", values="effect")
n_t, n_s = wide.notna().sum(axis=0).mean(), wide.notna().sum(axis=1).mean()  # twins per stimulus, stimuli per twin
grand = cells["effect"].mean()
ms_stim = n_t * ((wide.mean(axis=0) - grand) ** 2).sum() / (wide.shape[1] - 1)
ms_twin = n_s * ((wide.mean(axis=1) - grand) ** 2).sum() / (wide.shape[0] - 1)
resid = wide.sub(wide.mean(axis=1), axis=0).sub(wide.mean(axis=0), axis=1) + grand
ms_res = (resid ** 2).sum().sum() / ((wide.shape[0] - 1) * (wide.shape[1] - 1))
v_stim, v_twin = max(0.0, (ms_stim - ms_res) / n_t), max(0.0, (ms_twin - ms_res) / n_s)
v_tot = v_stim + v_twin + ms_res

# each twin as its own 100-stimulus study: paired t-test of its 100 effects against 0
tw = cells.groupby("pid")["effect"].agg(["mean", "std", "count"])
tw_p = 2 * stats.t.sf(np.abs(tw["mean"] / (tw["std"] / np.sqrt(tw["count"]))), tw["count"] - 1)

lean = (d["pref_b"] > 4).groupby(d["frame"]).mean() * 100
res = {
    "n_twins": d["pid"].nunique(), "n_calls": len(d), "n_unparsed": int(d["rating"].isna().sum()),
    "mean_choice": d.loc[d.frame == "choice", "pref_b"].mean(),
    "mean_rejection": d.loc[d.frame == "rejection", "pref_b"].mean(),
    "effect_mean": cells["effect"].mean(),
    "lean_b_choice": lean["choice"], "lean_b_rejection": lean["rejection"],
    "lean_b_diff": lean["rejection"] - lean["choice"],
    "twins_positive": int((by_twin > 0).sum()), "stimuli_positive": int((by_stim > 0).sum()),
    "stim_effect_min": by_stim.min(), "stim_effect_max": by_stim.max(), "stim_effect_sd": by_stim.std(),
    "twin_effect_sd": by_twin.std(),
    "d_av": d_av, "d_z": d_z,
    "stim_q": q, "stim_q_df": q_df, "stim_q_p": float(stats.chi2.sf(q, q_df)), "stim_tau": tau,
    "stim_i2": i2, "var_pct_stim": v_stim / v_tot * 100, "var_pct_twin": v_twin / v_tot * 100,
    "var_pct_resid": ms_res / v_tot * 100, "twin_true_sd": np.sqrt(v_twin),
    "twin_effect_median": tw["mean"].median(), "twin_effect_min": tw["mean"].min(),
    "twin_effect_max": tw["mean"].max(), "twins_sig_pct": float((tw_p < .05).mean() * 100),
    "stim_reliability": split_half("stimulus", "pid"), "twin_reliability": split_half("pid", "stimulus"),
}
# twin traits as moderators (person_moderation.R), if that analysis has run
person = OUT.parent / "person" / "person_moderation.csv"
if person.exists():
    pm = pd.read_csv(person).set_index("trait")
    tc = pd.read_csv(OUT.parent / "person" / "trait_cor.csv").set_index("trait")
    res.update({"n_traits": len(pm), "n_traits_sig": int((pm["bh"] < .05).sum()),
                "nfc_b": pm.loc["nfc", "b"], "closure_b": pm.loc["closure", "b"],
                "nfc_bh_p": float(f"{pm.loc['nfc', 'bh']:.1g}"), "closure_bh_p": float(f"{pm.loc['closure', 'bh']:.1g}"),
                "nfc_rel_pct": abs(pm.loc["nfc", "rel"]) * 100, "closure_rel_pct": pm.loc["closure", "rel"] * 100,
                "nfc_closure_r": tc.loc["nfc", "closure"]})
# stimulus moderators (code_moderators.py + moderation.R), if that analysis has run
mod_dir = OUT.parent / "moderators"
if (mod_dir / "moderation.csv").exists():
    mm = pd.read_csv(mod_dir / "moderation.csv").sort_values("p")
    mc = pd.read_csv(mod_dir / "moderator_cor.csv").set_index("moderator").to_numpy()
    pca = pd.read_csv(mod_dir / "moderator_pca.csv")
    top = mm.iloc[0]
    theory = mm[mm["moderator"].str.contains("flaw_inference|taste_based|idiosyncratic|prior_preference|rejection_implies")]
    # smallest |r| (n = 100 stimuli) whose p passes BH for the single best test (p < .05 / 100)
    t_crit = stats.t.isf(.05 / 100 / 2, len(by_stim) - 2)
    res.update({"n_mods": len(mm), "n_mods_sig": int((mm["bh"] < .05).sum()), "mods_min_bh": mm["bh"].min(),
                "top_mod_b": top["b"], "top_mod_r": top["r_stimulus"], "top_mod_raw_p": float(f"{top['p']:.1g}"),
                "mods_max_abs_r": mm["r_stimulus"].abs().max(),
                "theory_r_min": theory["r_stimulus"].min(), "theory_r_max": theory["r_stimulus"].max(),
                "r_needed_bh": t_crit / np.sqrt(t_crit ** 2 + len(by_stim) - 2),
                "mod_pairs_r07": int((np.abs(mc[np.triu_indices_from(mc, 1)]) > .7).sum()),
                "mod_pairs": int(len(mc) * (len(mc) - 1) / 2),
                "mod_pcs_80": int((pca["cumulative"] < .8).sum() + 1)})
res = {k: (float(f"{v:.3g}") if k.endswith("_p") else (round(float(v), 3) if k.endswith("_b") else round(float(v), 2))) if isinstance(v, float) else v
       for k, v in res.items()}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
