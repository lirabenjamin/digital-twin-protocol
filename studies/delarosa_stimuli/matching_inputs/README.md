# Frozen inputs for stimulus matching

These three CSVs are exact copies of the original study output files used in the existing stimulus-selection analysis:

- `stimulus_features_long.csv`: 6,000 integer ratings, identified by stimulus, feature, and repetition.
- `stimulus_features.csv`: 100 rows with means and standard deviations for the 20 original features.
- `moderation_mlm.csv`: 41 aggregate rows from the original moderator models.

`manifest.json` records their original SHA-256 hashes, plus hashes of `../stimuli.json` and `../features.json`. All five match the earlier analysis audit. The `output/full/` prefixes in the manifest identify the original source locations; the portable loader maps those entries to this folder.

These are the original 20-feature ratings. The later agent ratings and the stopped isolated-coding run do not enter this bundle.

The bundle supports reproducing stimulus matching from the original ratings. It does not include participant-level choices and therefore does not itself support refitting the twin-outcome models that produced the moderator-estimate table.

See [reproduction instructions](../../../reports/stimulus_matching/REPRODUCE.md).
