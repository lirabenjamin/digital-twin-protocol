# Selecting stimulus pairs for the human study

This report describes the analysis conducted on 18 October 2026 and extended on 19 October to include a balance weight of zero and rerun the selection at a weight of two.

The human study will test whether being paid daily rather than weekly changes people's likelihood of choosing the more expensive purchase option, and whether that effect differs across purchase situations. A **moderator** is a feature of a situation that may be associated with a larger or smaller pay-frequency effect.

We want pairs of scenarios that differ clearly on one feature while remaining similar on the other rated features. For example, a pair chosen to examine necessity should differ in necessity while being similar in impulse, long-term value, and the other features we measured.

These goals can compete. A larger difference in necessity may come with larger differences elsewhere. This analysis compares five settings for balancing those goals, including one that gives no weight to balance.

**Our recommended starting point is ten pairs covering seven primary features and three comparison features, using a balance weight of 2.** These pairs contain 18 distinct scenarios. The report explains the selection rule, what the balance weight means, and the remaining differences the team should review. All results use the original 20 features and their original ratings.

## What information did we use?

We started with 100 purchase scenarios. Each presents a choice between a more expensive option and a cheaper option. The original coding pipeline used Claude Haiku 5.5, with reasoning effort set to low. Each fresh request contained one scenario, its two options and prices, and one feature question with anchors for 0 and 10. We averaged three separate integer ratings for each scenario and feature, producing 2,000 mean ratings from 6,000 requests. These ratings guide the selection; human participants may perceive the scenarios differently. The [original rating guide](../../studies/delarosa_stimuli/FEATURES_REVIEW.md) contains all 20 questions and their anchors.

We selected one pair for each of seven features nominated from the twin-study results: splurge, long-term value, upgrade versus a different thing, deferrability, impulse, necessity, and purchase frequency. These are candidate moderators: features that may be associated with differences in the treatment effect across scenarios.

We also examined emotional pull, convenience, and money salience as comparison features. Their estimated moderation effects were small in twin-study models that did and did not adjust for price. This does not establish that their true effects are zero.

**Justifiability is excluded from the features we propose to test.** The concern is that judging whether a purchase is justifiable is subjective and too close to the purchase decision we are trying to explain. It remains among the original rated features used to balance pairs, consistent with the decision to match on all other rated features.

The **feature of interest** is the feature we want a particular pair to differ on. The **other features** are the remaining 19 rated features.

## Required step: check distributions and coverage after rating

After rating the stimuli, and before finalizing the human-study selection, we will inspect which parts of the feature space our pool represents. We will flag potential gaps for review, rather than assume that a skewed distribution is a problem or that every part of a scale should be equally common.

1. **Inspect each feature across the full pool.** Show its distribution on the full 0–10 scale, alongside the scale anchors. Report the observed range, tertile cutoffs, and group sizes with ties included. Flag skew, tight clustering, concentrations at the endpoints, and sparsely represented or empty portions of the scale. Relative ranks alone can hide these patterns.
2. **Inspect relevant combinations of features.** Use plots or counts for theoretically important feature pairs to flag missing combinations. Having a broad range on each feature separately does not guarantee coverage of their combinations. Compare the eventual selected set with the full pool as well.
3. **Assess what each flag means.** A sparse region may reflect the intended population: perhaps most relevant purchases are not splurges. It may instead reflect how stimuli were generated or selected, or how the rating question was interpreted. Check the scenario texts and available evidence about the intended population. If the explanation is uncertain, record that uncertainty; this pool alone cannot establish the population distribution.
4. **Record a decision for each flag.** Note the feature or combination, the sparse region and number of stimuli, the possible explanation, and the response. Options include retaining a justified imbalance, reviewing the ratings, adding suitable stimuli, or narrowing the study's claims. The response should reflect the study's objective; estimating typical effects and testing strong moderator contrasts may require different coverage.

We will not automatically force distributions to be uniform or change ratings to fill gaps.

This is a required review before the human-study set is finalized. The pair tables below remain candidates for that review; they do not establish that this distribution-and-coverage check has been completed.

### Purchase frequency may need broader coverage

**Purchase frequency may be an important moderator that the current scope does not adequately capture.** It is already on our shortlist, but our pool contains relatively few scenarios near the frequent-purchase end of the scale. Of the 100 scenarios, 72 have mean ratings of 2 or below, and only seven score above 6. The scale runs from 0, “once every few years or less,” to 10, “several times a week.” Intermediate ratings do not have specified calendar-frequency anchors.

The upper-tertile cutoff is only 2.00. A scenario can therefore qualify as “high frequency” relative to this pool without representing a frequent purchase. Selecting opposite tertiles cannot supply contrasts that the pool lacks, and balancing the other features can further narrow the contrast we select.

This is a reason to investigate coverage, not evidence that we have missed a moderation effect. Before finalizing the study, we should review whether the intended scope includes routine, frequent purchases and whether the ratings reflect the scenario texts. If it does, we should consider adding suitable examples and rating them with the same procedure. If we retain the current scope, our conclusions about purchase frequency should acknowledge its limited coverage.

## Which pairs could be selected?

For each feature of interest, we ordered the scenarios by their ratings. The low scenario had to be in the bottom third, and the high scenario had to be in the top third. These groups are called **tertiles**. Scenarios tied at a cutoff were included, so a group could contain more than one third of the scenarios.

The high rating had to exceed the low rating, with no additional minimum gap. Differences on other features reduced the pair's score rather than automatically excluding it. We did not match scenarios on price.

We examined all 4,950 possible pairs and scored those that met these rules. We searched separately for each of the ten features, allowing scenarios to appear in multiple pairs. The twin-study results helped us nominate the features to investigate. Once those features were chosen, pair scores used feature ratings, not estimated treatment effects.

## How did we measure the two goals?

First, we calculated the **gap on the feature of interest** by subtracting the low rating from the high rating. Ratings of 2 and 8 give a gap of 6 points. Larger gaps provide stronger contrasts. We call this gap **F**.

Second, we calculated an **adjusted distance**, called **D**, that combines differences on all 19 other features. Smaller distances indicate greater similarity.

The distance accounts for correlations among the other features. When several features tend to vary together across the 100 scenarios, changes along that shared pattern receive less weight. This reduces repeated counting of correlated differences while retaining every feature. It does not guarantee that all conceptual overlap has been removed.

This adjustment accounts for both positive and negative correlations. For example, a feature and its reverse-coded version carry the same information even though their correlation is negative. Reversing the direction of a rating scale does not change the adjusted distance. A pair that departs from a usual relationship between features can still receive a larger distance than a pair that follows that relationship.

An adjusted distance of 4 does **not** mean an average difference of 4 rating points. The tables therefore also show the **average gap on the other features** and the **largest gap on another feature**. These use ordinary absolute differences in rating points: differences of +2 and −2 both count as gaps of 2.

## What is lambda, and how does it affect selection?

**Lambda**, written **λ**, is the weight we give to similarity on the other features. It is a setting we choose, rather than a quantity estimated from the data. We compared five settings: 0, 0.5, 1, 2, and 3.

For every eligible pair, we calculated a score using this rule:

**Score = gap on the feature of interest − balance weight × adjusted distance.**

Using the symbols defined above, the same rule is **score = F − λD**. We selected the pair with the highest score for each feature and each weight.

At a weight of 0.5, an extra unit of adjusted distance reduces the score by half a point. At a weight of 2, it reduces the score by two points. Higher weights therefore favor greater similarity on the other features. When adjusted distances are equal, the larger gap on the feature of interest always wins.

At a weight of zero, the score is simply the gap on the feature of interest. Every selected pair therefore spans that feature's full observed range. This is a contrast in feature ratings, not an estimated daily-versus-weekly treatment effect. The tertile rule still applies. If several pairs share the largest gap, we choose the one with the smallest adjusted distance; balance cannot compensate for a smaller focal gap at this setting.

The following example uses **hypothetical pairs**, not the study results. Pair A has a gap of 8 and an adjusted distance of 4. Pair B has a smaller gap of 6, but a smaller adjusted distance of 2.5.

| Balance weight (λ) | Pair A's score | Pair B's score | Selected pair |
| --- | ---: | ---: | --- |
| 0 | 8.00 | 6.00 | Pair A is selected. |
| 0.5 | 6.00 | 4.75 | Pair A is selected. |
| 1 | 4.00 | 3.50 | Pair A is selected. |
| 2 | 0.00 | 1.00 | Pair B is selected. |
| 3 | −4.00 | −1.50 | Pair B is selected. |

With less weight on balance, Pair A's larger contrast wins. With more weight, Pair B's greater similarity wins. Negative scores are possible; the highest score still wins. Scores are comparable only within the same feature and weight. We choose the weight by examining the selected scenarios and their remaining differences.

## What happened with our scenarios?

The table summarizes the seven primary pairs. Each pair contributes equally to the averages. The average gap on other features covers all 19 other features in all seven pairs. The largest gap is the single largest such difference. We want larger gaps on the feature of interest and smaller differences elsewhere.

$summary_table

**At a weight of zero, the seven primary pairs have an average focal gap of 8.00 points, compared with 5.29 at a weight of 2.** However, their average gap on the other features is 2.21 points rather than 0.85, and the largest non-focal gap is 9.33 rather than 4.00. The larger focal contrasts come with much larger differences elsewhere.

For purchase frequency, zero selects a new bathroom vanity and faucet (s050; rating 0.67) versus coffee-shop cold brew (s058; rating 8.67). The focal gap is 8.00 points, but another feature differs by 8.00 points. At a weight of 2, the focal gap is 2.67 points and no other feature differs by more than 2.00. This comparison reinforces the need to review frequency coverage and the available balanced contrasts.

**We recommend starting with a balance weight of 2 for team review.** Compared with a weight of 0.5, it reduces the average gap on other features from 1.15 to 0.85 points. The cost is a smaller average gap on the feature of interest: 5.29 instead of 6.90 points. This is a judgment about the tradeoff, not a uniquely correct setting identified by the analysis.

Purchase frequency illustrates the tradeoff particularly clearly. At weights of 0.5 and 1, the selected pair compares a Napa anniversary weekend with a coffee-shop cold brew. The purchase-frequency gap is 7.67 points, but convenience also differs by 7 points. At a weight of 2, the selected pair compares a cruise excursion booked ahead with a church-group dinner. Its purchase-frequency gap is smaller, at 2.67 points, but its largest difference on any other feature is 2 points.

Increasing the weight from 2 to 3 changes only the upgrade pair among the seven primary pairs. It also changes the money-salience comparison pair. Across the primary pairs, the average gap on other features falls only slightly, from 0.85 to 0.84 points, while the average gap on the feature of interest falls from 5.29 to 4.90. However, the upgrade pair's largest difference on another feature falls from 3.67 to 2.33 points. A weight of 3 may therefore be useful if that particular improvement matters to the team.

## Which scenarios were selected?

The tables show every selected pair, combining weights that choose the same pair. “Low” and “high” refer to the rating on the feature of interest. Each entry includes the scenario's identifier, such as `s048`, and its rating. Complete scenario texts follow the main report.

The last column names the other feature with the largest gap. Increasing the balance weight improves or preserves the combined distance, but may leave some individual feature gaps larger.

### The seven primary features

$targets_table

### The three proposed comparison features

$comparisons_table

Including the three comparison features gives ten selected pairs. These contain **18 distinct scenarios** at weights of 0.5, 1, and 2, and **19 distinct scenarios** at weights of 0 and 3. The count is below twenty because some scenarios appear in more than one pair.

## The selected set at a balance weight of 2

The rerun selects the same ten pairs as before, comprising seven primary pairs and three comparison pairs. Together they contain **18 distinct stimuli**. The table below gives the complete selection at this setting. All gaps are measured in points on the original 0–10 rating scales.

The median gap is the middle of the 19 absolute differences on the other features. The largest gap shows the greatest of those differences and names the feature. These summaries do not change the selection scores.

$lambda_two_table

The water-park scenario (s035) appears in both the upgrade and money-salience pairs. The church-group dinner (s065) appears in both the impulse and purchase-frequency pairs. These repeated appearances explain why ten pairs use 18 distinct stimuli. The seven primary pairs alone use 13 distinct stimuli.

## The selected set at a balance weight of 1

At this weight, the ten selected pairs also contain **18 distinct stimuli**, covering seven primary features and three comparison features. The table uses the same columns and original 0–10 rating scales as the table above.

$lambda_one_table

## What should the team review before using these pairs?

The pairs still differ beyond the intended contrast. At a weight of 2, the long-term-value pair differs by 4 points on upgrade versus a different thing. Human rating checks would help establish whether participants perceive both the intended contrast and these remaining differences.

A difference in treatment effects within a pair would not establish that the feature of interest caused it. The scenarios also differ in wording, context, and potentially unmeasured characteristics. The analysis should also account for scenarios that appear in multiple pairs.

Finally, this procedure deliberately selects contrasts. It does not produce a representative sample for estimating the average treatment effect across all 100 scenarios or across a broader population of purchase situations.

## How can readers explore the results?

Open the accompanying [interactive pair explorer](pair-explorer.html) in a browser. Choose a feature and a balance weight to see the selected pair, its complete scenario texts, and the differences on every other rated feature. Each dot represents an eligible **pair of scenarios**. Higher dots have larger gaps on the feature of interest, and dots farther left have smaller adjusted distances. The highlighted pair has the highest score at the selected weight.

The explorer works offline because its data are included in the HTML file. To share it as a GitHub Gist, include `pair-explorer.html` and this report. Recipients can download the files and open the HTML file in their browser.

## Technical appendix: calculation and verification

This appendix gives the details needed to reproduce the calculation. It is not required to interpret the tables.

For each feature of interest, we calculated a correlation matrix, called **R**, for the other 19 features using all 100 scenarios. The vector **Δ** contains the 19 signed rating differences between the high and low scenarios. We kept these differences in the original rating units.

We decomposed R into its eigenvectors, collected in **Q**, and its eigenvalues, denoted by **η**. The eigenvectors identify directions in the feature space, and the eigenvalues describe the amount of variation in those directions on the correlation scale. The weight matrix **W** and the distance **D** were then calculated as follows:

```text
R = Q diag(η) Qᵀ
W = Q diag(1 / max(η, 1)) Qᵀ
D = sqrt(Δᵀ W Δ)
```

Here, `diag` forms a diagonal matrix and the superscript `ᵀ` denotes a transpose. The maximum is taken separately for each eigenvalue. This gives squared differences less weight along directions with eigenvalues above one and leaves the weight at one for the remaining directions. We did not divide the rating differences by their standard deviations or divide the resulting distance by the number of features. This is the proposed correlation adjustment used for this analysis; it is not standard Mahalanobis distance. The distance calculation stayed fixed across the five balance weights.

If pair scores tied, we preferred the smaller adjusted distance, then the larger gap on the feature of interest, and then the stimulus identifiers in alphabetical order. Numerical sorting values were rounded to ten decimal places to prevent floating-point noise from deciding ties. Prices did not enter these decisions.

The input checks passed for all 100 scenarios, 20 features, and 6,000 individual ratings. These checks covered identifiers, missing values, rating ranges, matching records across files, and consistency with the saved source data. All $selected_count selected-pair results and $difference_count feature differences were checked against the original ratings. Each selected pair had the highest score among the eligible pairs for its feature and weight.

The following files contain the detailed results:

- The [selected-pair file](../../outputs/stimulus_matching/lambda_sweep/selected_pairs.csv) contains scenario texts, scores, prices, and measures of similarity.
- The [feature-difference file](../../outputs/stimulus_matching/lambda_sweep/pair_feature_differences.csv) contains all 20 rating differences for every selected pair.
- The [candidate file](../../outputs/stimulus_matching/lambda_sweep/top_candidates.csv) contains the ten highest-scoring pairs for each feature and weight.
- The [audit record](../../outputs/stimulus_matching/lambda_sweep/audit.json) records the settings and checks used in the analysis.

To reproduce the analysis, interactive explorer, and both survey CSVs from this checkout, run `.venv/bin/python scripts/compare_matching_penalties.py`. The original rating inputs are versioned in the repository. See the [setup and reproduction instructions](REPRODUCE.md) for dependencies, validation checks, and the command that rebuilds only the survey CSVs.

## Appendix: complete original scenario texts

The texts below reproduce every scenario selected by at least one feature and weight. The two purchase options belong to that individual scenario; they are different from the low and high scenarios paired in the analysis.

$full_scenarios
