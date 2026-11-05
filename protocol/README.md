# Use the protocol for a new effect

This folder is a generic, config-driven version of the pipeline in `studies/choice_rejection_stimuli/`. The scripts hold no study-specific text. One YAML file describes the effect: the two conditions, the stimulus fields, the DV and its scale, the seed stimuli, the model settings and the prompt templates. The worked example is the choice-versus-rejection effect, in `config/example_effect.yaml`.

Run every command from this folder. Each script reads `--config` (default: the example).

## Adapt it to your effect

1. Copy `config/example_effect.yaml` and edit every block: `effect`, `conditions`, `stimulus_fields`, `views`, `seeds`, `generation`, `epsilon`, `screens`, `dv`.
2. Write your twin task screen (see `prompts/examples/choice_rejection/twin_task.txt`) and your moderator catalog (`moderators.json`, same format).
3. The generic templates in `prompts/` (`generator.txt`, `paraphrase.txt`, `screens.txt`, `twin_system.txt`, `moderator.txt`) take their effect-specific text from the config. Edit them only if your design needs it.
4. Run each step with `--dry-run` first and read the filled prompt.

## Steps and outputs

All outputs go to `output_dir` from the config.

| Step | Command | Writes |
|---|---|---|
| 0. Twins | `python ../code/download_twins.py` | `../twins/` |
| 3a. Epsilon | `python scripts/calibrate_epsilon.py` | `paraphrases.json`, `epsilon.json` |
| 2-3. Generate | `python scripts/generate.py --max-batches 150` | `draws.jsonl`, `batches.jsonl` |
| 4. Select | `python scripts/select_stimuli.py` | `stimuli.json`, `selection.json` |
| 5. Screens | `python scripts/write_screens.py` | adds the screen fields to `stimuli.json` |
| 6. Twins | `python scripts/run_twins.py --n 5` (pilot), then `--batch-all` | `pilot/choices.csv`, `batches/`, `choices.csv` |
| 7. Effects | `python scripts/results.py` | `results.json`, `stimulus_effects.csv`, `twin_effects.csv` |
| 8. Ratings | `python scripts/rate_moderators.py` | `ratings_long.csv`, `stimulus_moderators.csv` |
| 9. Moderation | `Rscript scripts/moderation.R output/<name>` | `moderation.csv`, `moderator_cor.csv`, `moderator_pca.csv`, `moderator_cor.png` |

Claude calls are cached on disk in `../cache/`, so a rerun only pays for what is missing. Every call adds its cost to `spend.jsonl`, and calls stop at `budget_usd`.
