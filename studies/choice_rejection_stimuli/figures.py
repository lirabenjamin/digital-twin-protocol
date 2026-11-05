"""Figures for the twin study report -> output/full/figures/*.png.

Uses the same cells and definitions as results.py: effect = rejection minus choice in pref_b
(1-7, 7 = strongly prefer B, the option the other person ended up with), paired within twin x stimulus.
SD, d and split-half reliabilities are read from results.json, so the figures match the text.

    python3 figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = __import__("pathlib").Path(__file__).resolve().parent
OUT = HERE / "output" / "full"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# colorblind-safe pair (dataviz reference palette, slots 1-2) and recessive ink
COL = {"choice": "#2a78d6", "rejection": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EFFECT_LABEL = "Rejection minus choice (points on the 7-point scale)"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 220, "savefig.bbox": "tight",
})

res = json.loads((OUT / "results.json").read_text())
d = pd.read_csv(OUT / "choices.csv", usecols=["pid", "stimulus", "frame", "pref_b"]).dropna(subset=["pref_b"])
cells = (d.pivot_table(index=["pid", "stimulus"], columns="frame", values="pref_b").dropna().reset_index().rename_axis(columns=None))
cells = cells.assign(effect=cells["rejection"] - cells["choice"])
by_stim = cells.groupby("stimulus")["effect"].agg(["mean", "std", "count"])
by_twin = cells.groupby("pid")["effect"].mean()


def fig1_means() -> None:
    """Mean pref_b by frame with SEs across twins; distribution of ratings by frame."""
    twin_means = d.groupby(["pid", "frame"])["pref_b"].mean().unstack()
    m = d.groupby("frame")["pref_b"].mean()
    se = twin_means.std() / np.sqrt(len(twin_means))
    frames = ["choice", "rejection"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1, 2.2]})

    for i, f in enumerate(frames):
        a.bar(i, m[f] - 1, bottom=1, width=0.55, color=COL[f])
        a.errorbar(i, m[f], yerr=se[f], color=INK, capsize=4, lw=1.2)
        a.text(i, m[f] + 0.15, f"{m[f]:.2f}", ha="center", va="bottom", color=INK)
    a.set_xticks([0, 1], ["Choice", "Rejection"])
    a.set_ylim(1, 7)
    a.set_yticks(range(1, 8))
    a.axhline(4, color=MUTED, lw=0.8, ls=":")
    a.set_ylabel("Preference for B (1 = strongly A, 7 = strongly B)")
    a.set_title(f"A. Mean rating\nDifference = {m['rejection'] - m['choice']:.2f} points, d = {res['d_av']:.2f}",
                loc="left", fontsize=10)

    pct = d.groupby("frame")["pref_b"].value_counts(normalize=True).unstack().reindex(columns=range(1, 8)) * 100
    for j, f in enumerate(frames):
        b.bar(np.arange(1, 8) + (j - 0.5) * 0.38, pct.loc[f], width=0.36, color=COL[f], label=f.capitalize())
    b.set_xticks(range(1, 8), ["1\nStrongly A", "2", "3", "4\nNo preference", "5", "6", "7\nStrongly B"])
    b.set_ylabel("Percent of ratings")
    b.yaxis.grid(True, color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.legend(frameon=False, loc="upper left")
    b.set_title("B. Distribution of ratings", loc="left", fontsize=10)
    fig.text(0, -0.04, f"Error bars: ±1 SE across {len(twin_means):,} twins (smaller than the line width). "
             "B = the option the other person ended up with.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_means_by_frame.png")
    plt.close(fig)


def fig2_stimuli() -> None:
    """Effect per stimulus, sorted, with 95% CIs (SE = SD of paired differences / sqrt(n twins))."""
    stim = {s["stimulus"]: s for s in json.loads((HERE / "stimuli.json").read_text())}

    def short(s: dict) -> str:
        a, b = (s[k]["label"].removeprefix("the ") for k in ("option_a", "option_b"))
        return f"{b} (vs. {a})"

    st = by_stim.sort_values("mean")
    ci = 1.96 * st["std"] / np.sqrt(st["count"])
    y = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(7.5, 14))
    ax.errorbar(st["mean"], y, xerr=ci, fmt="o", ms=3.5, color=COL["rejection"], ecolor=MUTED, elinewidth=1,
                capsize=0)
    ax.axvline(0, color=INK, lw=0.9)
    ax.axvline(res["effect_mean"], color=MUTED, lw=0.9, ls="--")
    ax.text(res["effect_mean"] + 0.03, len(st) - 0.5, f"Mean = {res['effect_mean']:.2f}", color=MUTED, fontsize=8,
            va="bottom")
    ax.set_yticks(y, [short(stim[s]) for s in st.index], fontsize=6.3)
    ax.set_ylim(-1, len(st) + 1.5)
    ax.set_xlabel(EFFECT_LABEL)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Effect of the rejection frame for each of the 100 stimuli\n"
                 "Label: option B, the option the other person ended up with (vs. option A)",
                 loc="left", fontsize=10)
    ax.annotate(f"Dots: mean paired difference across {st['count'].min():,}-{st['count'].max():,} twins per stimulus. Bars: 95% CI.",
                (0, -0.06), xycoords="axes fraction", fontsize=8, color=MUTED)
    fig.savefig(FIG / "fig2_effect_by_stimulus.png")
    plt.close(fig)


def fig3_twins_vs_stimuli() -> None:
    """Distribution of twin effects vs stimulus effects on one x axis, with SD and split-half reliability."""
    bins = np.arange(-0.4, 2.75, 0.05)
    rows = [("A. Twins", by_twin, res["twin_effect_sd"], res["twin_reliability"], "averaged over 100 stimuli"),
            ("B. Stimuli", by_stim["mean"], res["stim_effect_sd"], res["stim_reliability"],
             f"averaged over {len(by_twin):,} twins")]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, (name, x, sd, rel, note), col in zip(axes, rows, [COL["choice"], COL["rejection"]]):
        w = np.full(len(x), 100 / len(x))
        ax.hist(x, bins=bins, weights=w, color=col, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_ylabel(f"Percent of {name.split()[1].lower()}")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{name} (n = {len(x):,}; each effect {note})", loc="left", fontsize=10)
        ax.text(0.98, 0.9, f"SD = {sd:.2f}\nSplit-half reliability = {rel:.2f}", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=9)
    axes[1].set_xlabel(EFFECT_LABEL)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_twins_vs_stimuli.png")
    plt.close(fig)


def fig5_moderators():
    """Moderation of the frame effect by each of the 100 stimulus moderators (moderation.R), sorted, 95% CIs."""
    path = HERE / "output" / "moderators" / "moderation.csv"
    if not path.exists():
        return
    m = pd.read_csv(path).sort_values("b").reset_index(drop=True)
    lo, hi = m["b"] - 1.96 * m["se"], m["b"] + 1.96 * m["se"]
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.5, 16))
    raw = m["p"] < .05
    ax.hlines(y, lo, hi, color=np.where(raw, INK, MUTED), lw=1)
    ax.scatter(m["b"], y, s=14, color=np.where(raw, INK, "white"), edgecolor=np.where(raw, INK, MUTED), zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, m["name"], fontsize=6.5)
    ax.set_ylim(-1, len(m))
    ax.set_xlabel("Change in the rejection effect per SD of the moderator (points on the 7-point scale)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"None of the {len(m)} moderators passes Benjamini-Hochberg correction "
                 f"(smallest corrected p = {m['bh'].min():.2f})", fontsize=9, color=INK, loc="left")
    ax.text(-0.35, -0.045, "Filled dots: uncorrected p < .05. Bars: 95% CIs. Mean effect at the average stimulus: "
             f"{m['effect_at_mean'].mean():.2f} points.", fontsize=7.5, color=MUTED, transform=ax.transAxes)
    fig.savefig(FIG / "fig5_moderators.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_means()
    fig2_stimuli()
    fig3_twins_vs_stimuli()
    fig5_moderators()
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")
