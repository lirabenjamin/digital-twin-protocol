"""Readable report for the revised opposite-tertile matching design."""
from pathlib import Path

import pandas as pd


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def candidates_table(details, variant):
    rows = []
    for r in details.loc[details.price_rule.eq(variant)].itertuples():
        rows.append([r.moderator, f"{r.low_id} → {r.high_id}", f"{r.low_score:.2f} → {r.high_score:.2f}",
                     f"{r.other_adjusted:.2f}", f"{r.other_mean_abs:.2f}", f"{r.other_rms:.2f}",
                     f"{r.other_max:.2f}", r.worst_feature_name])
    return md_table(["Moderator", "Low → high", "Focal ratings", "Adjusted distance", "Mean absolute Δ", "RMS Δ", "Max absolute Δ", "Largest non-focal change"], rows)


def appendix(details, stimuli):
    by_id = {s["stimulus"]: s for s in stimuli}
    parts = []
    for r in details.loc[details.price_rule.eq("unrestricted")].itertuples():
        parts.extend([f"### {r.moderator}", "",
                      f"Eligible pairs: {r.n_eligible}; focal gap {r.focal_gap:.2f}; adjusted non-focal distance {r.other_adjusted:.2f}; "
                      f"mean absolute change {r.other_mean_abs:.2f}, RMS {r.other_rms:.2f}, max {r.other_max:.2f} "
                      f"({r.worst_feature_name}). {r.other_count_ge1} of 19 non-focal features move ≥1 point and "
                      f"{r.other_count_ge2} move ≥2 points. These counts are descriptive, not eligibility rules.", ""])
        for side in ("low", "high"):
            sid = getattr(r, f"{side}_id")
            s = by_id[sid]
            parts.extend([f"**{side.title()} — {sid}, rating {getattr(r, f'{side}_score'):.2f}:** {s['options'][0]['label']}", "",
                          s["scenario"], "", "Options: " + "; ".join(f"{o['label']} (${o['cost']:g})" for o in s["options"]) + ".", ""])
    return "\n".join(parts)


def method_text():
    return r"""## Selection rule

For each moderator, examine all 4,950 pairs and retain pairs whose low stimulus is at or below the 33⅓% rating quantile and whose high stimulus is at or above the 66⅔% quantile, with high > low. **No two-point minimum, no non-focal tolerance, and no price matching in the default version.** Tied scores at quantile boundaries stay together, so the low/high groups may contain more than one third of the stimuli.

Among eligible pairs, minimize the correlation-adjusted distance on **all other 19 rated features**. Only when that distance ties to numerical precision does a larger focal gap break the tie, followed by stimulus IDs. Prices do not enter the default ranking or tie-break. Pairs are optimized independently, so stimuli can recur across moderators.

## Measuring movement without counting related ratings as independent

Each non-focal feature contributes its signed **high-minus-low difference on the original 0–10 scale**. A one-point change has the same initial weight on every feature; differences are not divided by sample SDs. We estimate the Pearson correlation matrix R of the 19 non-focal ratings across all 100 stimuli, once for each focal moderator. Neither the focal feature nor twin outcomes enter that matrix.

If R = Q diag(λ) Qᵀ, define:

    W = Q diag[1 / max(λ, 1)] Qᵀ
    adjusted distance = sqrt(Δᵀ W Δ)

Shared movement in a correlated direction is discounted. The inverse weights are capped at 1, so low-variance directions are **not amplified**. Every direction remains in the score; there is no two-dimensional PCA truncation. This is a proposed, capped inverse-correlation distance, **not ordinary Mahalanobis distance**. It continues the discount-only geometry discussed in the earlier stimulus-validity work.

For an isolated block of two perfectly duplicated ratings, moving both by one point contributes the same squared distance as one independent one-point change. At correlation .90, same-direction one-point movement contributes 2/1.90 ≈ 1.05 instead of 2. Opposite-direction movement is not discounted in this example. This illustrates statistical overlap; it does not prove that correlated psychological constructs are interchangeable, and exact duplicate invariance is not claimed for arbitrary correlated systems.

Ordinary covariance inversion can amplify poorly estimated small-eigenvalue directions; see the [scikit-learn covariance guide](https://scikit-learn.org/stable/modules/covariance.html). Our capped weighting is a declared design choice, not a validated psychological metric. Correlations come from this selected 100-stimulus reference and need not generalize beyond it.

**The adjusted score is a joint distance, not an average rating difference.** It is used to rank pairs for a given moderator. Alongside it, the report shows raw mean absolute movement, RMS movement, largest movement and its feature, counts moving ≥1/≥2 points, and every signed feature difference. These raw diagnostics are not selection cutoffs. The redundancy-discount column in the CSV is the fraction of raw squared joint movement removed by W, not variance explained or causal validity.
"""


def sensitivity_table(details):
    rows = [[r.moderator, f"{r.low_id} → {r.high_id}", r.raw_rms_pair, r.raw_max_pair]
            for r in details.loc[details.price_rule.eq("unrestricted")].itertuples()]
    return md_table(["Moderator", "Adjusted-distance selection", "Raw RMS selection", "Raw maximum-gap selection"], rows)


def write_report(root, data, details):
    root = Path(root)
    primary = details.loc[details.price_rule.eq("unrestricted")]
    counts = pd.Series(primary.low_id.tolist() + primary.high_id.tolist()).value_counts()
    reused = ", ".join(f"{sid} ({n} pairs)" for sid, n in counts.items() if n > 1) or "none"
    files = [("Selected pairs, prices, and diagnostics", "selected_pairs.csv"),
             ("Ten best eligible pairs per moderator and version", "top_candidates.csv"),
             ("All signed feature differences", "pair_feature_differences.csv"),
             ("Tertile cutoffs, tied-group sizes, and eligible counts", "eligibility.csv"),
             ("Correlation and weight matrices", "geometry.csv"),
             ("Eigenvalues and capped directional weights", "geometry_eigenvalues.csv"),
             ("Diagnostic contrast–balance frontiers", "pareto_pairs.csv"),
             ("Original twin moderator evidence", "twin_moderator_evidence.csv"),
             ("Audit and updated specification", "audit.json")]
    links = "\n".join(f"- [{label}](../../outputs/stimulus_matching/{name})" for label, name in files)
    text = f"""# Opposite-tertile stimulus matching — revised 2026-10-18

The default selects the **most balanced pair on all other 19 rated features**, among bottom-versus-top-tertile stimuli. Price matching is an optional second version. There is no minimum focal gap beyond high > low, and no non-focal tolerance.

{method_text()}

## Default results: no price matching

{candidates_table(details, 'unrestricted')}

The first seven moderators were nominated by the user. Emotional pull, convenience, and money salience remain proposed comparison hypotheses; their earlier nonsignificant twin results do not establish zero effects. The ten pairs contain **{len(counts)} distinct stimuli**. Reuse: {reused}.

Small focal gaps can qualify because tertiles are relative to this inventory. In particular, purchase frequency has cutoffs 1.33 and 2.00, and money salience has cutoffs 0.67 and 1.00. This is the requested rule, and the report leaves those contrasts visible rather than reinstating a two-point minimum.

## Optional version: price matching

The focal-tertile rule, all-19-feature distance, and ranking are unchanged. This secondary version additionally requires the expensive prices and the expensive-minus-cheaper price increments each to differ by a factor ≤1.25, and both cheaper options to be free or both paid. These are the previous provisional price rules, retained only for this optional comparison.

{candidates_table(details, 'matched')}

## Sensitivity to the distance definition

These are diagnostic comparisons using the same default eligible pairs. They do not choose or tune the primary distance after observing results. Each metric's ties prefer a larger focal gap, then IDs. Raw RMS treats the dimensions as separate; raw maximum movement ignores the number of other differences. Both are reported to make the role of the correlation adjustment visible.

{sensitivity_table(details)}

## Interpretation

- Minimum distance means best under the declared score among these 100 stimuli; it does not establish that all non-focal differences are small. Check the largest raw movement and the full text.
- Independent ratings should check the intended contrast and other features. Repeated LLM ratings are not human validation; the feature inventory itself was developed after inspecting twin outcomes.
- A single pair supports a test of that stimulus contrast. It cannot establish a general causal effect of the named moderator. Randomized pay conditions are still needed to compare the two stimuli's daily-minus-weekly effects.
- No prices enter the default search, so a selected pair can have a large price difference. Both option prices are disclosed in the CSV and appendix.

## Files and reproduction

{links}

Run `.venv/bin/python scripts/match_stimuli.py`. The versioned input hashes, rating reconstruction, and moderator-table checks run before selection. See [setup and reproduction instructions](REPRODUCE.md). Unit tests cover the revised eligibility/ranking, duplicate-block discounting, signed correlations, no amplification, singular correlations, constant features, feature-order/reverse-coding invariance, and data validity. Old tolerance-search outputs were superseded; the prior version remains in Git history.

## Stimulus texts for the default selections

{appendix(details, data.stimuli)}
"""
    path = root / "reports/stimulus_matching/report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")
    return path
