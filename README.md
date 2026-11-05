# Digital-twin stimulus pre-testing protocol

This repository holds a protocol for pre-testing many stimuli of a behavioral effect with LLM "digital twins" before running a human study. It also holds the complete proof of concept: the choice-versus-rejection framing effect (Nan, Park & Yang 2023), tested on 100 generated stimuli with 2,058 twins.

The twins are the Twin-2K-500 personas (Toubia et al. 2025). Each persona is the full survey record of one real US adult. Claude answers each study screen as that person would.

## The protocol in one paragraph

Start from the published stimuli of an effect. Have an LLM write hundreds of new stimuli and drop near-duplicates. Pick a diverse subset. Run every twin on every stimulus in both conditions. Estimate the effect for each stimulus and each twin, and check how much it varies. Have an LLM rate each stimulus on candidate moderators and test each one. Then pick the stimuli for the human study from these results.

## Repository map

| Path | What it holds |
|---|---|
| `studies/choice_rejection_stimuli/` | The proof of concept: code, prompts, the 100 final stimuli and the outputs |
| `studies/choice_rejection_stimuli/stimuli.json` | The 100 final stimuli, with their screens |
| `studies/choice_rejection_stimuli/prompts/` | `generator.txt`, `paraphrase.txt`, `scenario.txt`, `segment.txt`, `frames.txt`, `repair.txt`, `twin_task.txt`, `moderator.txt` |
| `studies/choice_rejection_stimuli/output/full/` | `choices.csv` (411,600 twin responses), `results.json` (headline numbers), `figures/` |
| `studies/choice_rejection_stimuli/output/moderators/` | Moderator ratings (`ratings_long.csv`, `stimulus_moderators.csv`) and moderation results |
| `studies/choice_rejection_stimuli/output/person/` | Twin traits scored from the personas and their moderation results |
| `code/` | `download_twins.py` (Twin-2K-500 from Hugging Face) and `llm.py` (one Claude call, cached on disk) |
| `studies/delarosa_s2/prompts/system.txt` | The twin system prompt (the persona goes here); shared with an earlier study |
| `studies/stimulus_augmentation/` | Shared helpers: embeddings, parsing, k-medoids (`stimuli.py`, `selection.py`) |
| `src/stimulus_feature_catalog.py` | Strict parser for the 0-10 moderator ratings |
| `scripts/compare_matching_penalties.py`, `src/`, `reports/stimulus_matching/REPRODUCE.md` | The shortlist algorithm: pairs of stimuli that differ on one focal moderator and stay similar on the others |
| `protocol/` | A generic, config-driven version of the pipeline. **Use the protocol for a new effect:** see [protocol/README.md](protocol/README.md) |
| `studies/choice_rejection_human/`, `reports/choice_rejection_human/` | The human study's analysis code and tables, added by the human-study authors |

## Requirements

- Python 3.12 or later: `pip install -r requirements.txt`
- R 4.3 or later with `tidyverse`, `lme4` and `jsonlite` for the moderation models
- API keys only for new paid runs: copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` (Claude) and `OPENAI_API_KEY` (embeddings for deduplication and selection). The analysis of the shipped outputs needs no key.
- The twins: `python3 code/download_twins.py` writes `twins/<pid>.txt` (2,058 files, about 270 MB, no key needed).

## The steps

The proof-of-concept scripts run from `studies/choice_rejection_stimuli/`. Each paid script has an offline preview that prints the request count and one filled prompt. The `protocol/` column gives the generic command for a new effect (run from `protocol/`).

| Step | Proof of concept | Generic (`protocol/`) | Writes |
|---|---|---|---|
| 1. Seed stimuli | `seeds.json` (the 7 published studies, with human conformity rates) | `seeds:` in `config/example_effect.yaml` | |
| 2. Generate new stimuli | `python3 generate.py --dry-run`, then `--max-batches 150` | `python scripts/generate.py` | `output/draws.jsonl`, `batches.jsonl` |
| 3. Deduplicate and stop | `calibrate.py` (epsilon), then the checks inside `generate.py` | `python scripts/calibrate_epsilon.py` | `epsilon.json`, `paraphrases.json` |
| 4. Pick a diverse subset | `python3 select_stimuli.py` (k-medoids, k = 100) | `python scripts/select_stimuli.py` | `stimuli.json` |
| 5. Write the screens | `write_screens.py`, `segment_scenarios.py`, `rewrite_frames.py`, `order_free.py`, `repair_stimuli.py`, `hand_edits.py`; `check_frames.py` and `export_screens.py` to review | `python scripts/write_screens.py` | `stimuli.json`, `screens.md` |
| 6. Run the twins | `python3 run_twins.py --dry-run`, then `--batch-all` | `python scripts/run_twins.py --batch-all` | `output/full/choices.csv` |
| 7. Effects and heterogeneity | `python3 results.py`, `figures.py` | `python scripts/results.py` | `output/full/results.json`, `figures/` |
| 8. Rate the moderators | `python3 code_moderators.py` (preview), then `--run` | `python scripts/rate_moderators.py` | `output/moderators/stimulus_moderators.csv` |
| 9. Test the moderators | `Rscript moderation.R`; twin traits: `person_moderators.py`, `person_moderation.R` | `Rscript scripts/moderation.R <output dir>` | `output/moderators/moderation.csv`, `moderator_cor.*` |
| 10. Choose stimuli for humans | `scripts/compare_matching_penalties.py` (see `reports/stimulus_matching/REPRODUCE.md`) | per-stimulus effects in `stimulus_effects.csv` | survey CSVs |

Step details:

1. **Seeds.** The published stimuli anchor the generator and calibrate epsilon. Keep the human results next to them, so you can compare the twins with people.
2. **Generation.** Each call asks for 10 new stimuli that differ from all the stimuli so far on named dimensions (in the example: the product, who conveys the decision, the channel, and the wording).
3. **Deduplication.** Every new stimulus is embedded (OpenAI `text-embedding-3-large`). If it lies within epsilon (cosine distance) of an earlier stimulus, it counts as a repeat. Generation stops when the model runs dry: 3 batches in a row with fewer than 3 new stimuli out of 10. With independent draws, you can stop on Good-Turing coverage or Chao1 instead.
4. **Selection.** k-medoids picks the k stimuli that leave every unique stimulus closest to a pick.
5. **Screens.** An LLM writes the scenario and the lead-in. The two condition texts stay word for word the same except for the manipulated phrase.
6. **Twin run.** One isolated request per twin x stimulus x condition. The option order is random per twin x stimulus and the same in both conditions.
7. **Effects.** `results.py` reports the mean effect, Cohen's d, the effect per stimulus and per twin, Cochran's Q, I², tau, a variance decomposition (stimulus, twin, residual) and split-half reliabilities.
8. **Moderator ratings.** Claude Haiku rates each stimulus on each candidate moderator from 0 to 10. One isolated request per stimulus x moderator x repetition, 3 repetitions.
9. **Moderation.** One mixed model per moderator: `y ~ treat * z + (1 | twin) + (1 + treat | stimulus)`, with Benjamini-Hochberg correction across moderators, plus the moderator correlation matrix and a PCA.
10. **Human study.** Use the per-stimulus effects and the moderator results to choose stimuli. The shortlist algorithm finds pairs of stimuli that differ on one focal moderator and match on the rest.

## Cost

- **Twin run.** In our run, 2,058 twins x 200 calls with Claude Haiku 5.5 cost about $153, or $0.074 per twin. Two features make this cheap. The Message Batches API costs 50% of the normal price. Prompt caching stores the persona (about 50,000 tokens) once per twin and reads it from cache for that twin's other calls; 98.5% of input tokens were cache reads.
- **Generation and screens.** Claude Sonnet 5.5; the scripts stop at a set budget ($40 in our run).
- **Moderator ratings.** 100 stimuli x 100 moderators x 3 repetitions = 30,000 short calls with Claude Haiku 5.5.

Prices change. Check the current price list before a large run, and check the cost of the first batch before you send the rest.

## Design choices

- **Isolated calls.** Each twin sees each stimulus and condition in a fresh request with no memory of the others. This removes order and carry-over effects, so a within-twin design keeps each response independent of the twin's other responses.
- **Epsilon from paraphrases.** The duplicate threshold comes from the data: it is the 95th percentile of the distance between each seed and its own paraphrases. Two stimuli closer than that differ no more than two wordings of one stimulus.
- **k-medoids.** The picks are real stimuli (medoids), and they cover the generated set better than a random set of the same size.
- **Three rating repetitions.** LLM ratings vary from call to call. The mean of 3 isolated ratings is more reliable than one rating, and the spread shows how stable each moderator is.
- **Benjamini-Hochberg correction.** With 100 candidate moderators, about 5 would pass p < .05 by chance. BH controls the false discovery rate across all of them.
- **Split-half reliability.** It shows whether the stimulus effects (or the twin effects) replicate across independent halves of the data. In our run, stimulus effects were highly reliable (0.99) and twin effects were not (0.20).
- **Moderator correlations.** Moderators that measure the same thing will pass or fail together. The correlation matrix and PCA show how many separate dimensions the catalog holds.

## How to cite

> [Citation placeholder: add the manuscript reference here.]

## References

- Nan, L. X., Park, S. K., & Yang, Y. (2023). Rejections are more contagious than choices. *Journal of Consumer Research*.
- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions. arXiv:2505.17479. Data: Hugging Face, `LLM-Digital-Twin/Twin-2K-500`.

## License

Code: MIT (see `LICENSE`). The Twin-2K-500 personas are CC-BY-4.0 and are downloaded from their source, not stored here.
