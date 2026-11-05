# Reproduce the stimulus-selection report and survey CSVs

This is the shortlist algorithm: for each focal moderator, it picks a pair of stimuli that differ as much as possible on that moderator while staying similar on all the other moderators (a balance weight, lambda, sets the trade-off). The worked inputs are the ratings of the 100 generated stimuli of an earlier proof of concept (the pay-frequency effect, `studies/delarosa_stimuli/`). The same code applies to any stimulus x moderator rating table.

The current 20-feature analysis can be rebuilt from this repository alone. The code, templates, original scenario texts, feature definitions, and frozen rating inputs are versioned. No API key, new coding run, private participant data, embedding cache, or checkout at a particular local path is required.

## Set up Python

From the repository root, create a virtual environment and install the pinned dependencies. The workflow was verified with Python 3.12.14.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-matching.txt
```

## Rebuild the current analysis

```sh
.venv/bin/python scripts/compare_matching_penalties.py
```

This command reruns the pair search at balance weights 0, 0.5, 1, 2, and 3. It writes:

- `reports/stimulus_matching/lambda_sweep.md`: the report, including the selected sets at weights 2 and 1 and their median and largest non-focal gaps.
- `reports/stimulus_matching/pair-explorer.html`: the self-contained interactive explorer.
- `outputs/stimulus_matching/lambda_sweep/`: selected pairs, candidate pairs, feature differences, summary statistics, geometry, and the audit record.
- `reports/stimulus_matching/survey/survey_stimuli_lambda_1.csv`.
- `reports/stimulus_matching/survey/survey_stimuli_lambda_2.csv`.

To write a separate copy for comparison, add `--output-root /path/to/reproduction`. The command creates `reports/` and `outputs/` below that directory. It always reads inputs from the checkout containing the script.

The pair search starts from the original ratings; it does not read selected pairs from an existing report. Report tables and the complete scenario appendix are generated. The explanatory prose is maintained in `src/templates/stimulus_selection_report.md`.

## Where the inputs come from

The original 100 scenarios and 20 feature definitions are in `studies/delarosa_stimuli/stimuli.json` and `features.json`. The small [frozen input bundle](../../studies/delarosa_stimuli/matching_inputs/README.md) contains:

- All 6,000 original individual ratings: three ratings per scenario and feature.
- The original table of 2,000 mean ratings, with its saved standard deviations.
- The original aggregate moderator-estimate table used as background for the shortlist.
- A manifest of source hashes that match the previously saved analysis audit.

Loading checks file hashes, unique identifiers, complete joins, numeric bounds, missing values, and reconstruction of every feature mean from the individual ratings. The analysis then checks the pair scores, tertile eligibility, and non-focal diagnostics. The survey exporter checks memberships and CSV text preservation.

The shortlist and scoring choices remain the existing ones. Adding this reproduction workflow does not change the selected pairs, original ratings, or CSV contents.
