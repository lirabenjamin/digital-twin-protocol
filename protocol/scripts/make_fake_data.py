"""Write a small synthetic dataset, so you can run the offline steps and the analysis with no API key.

Writes output/fake/:
  config.yaml               the example config, with output_dir set to output/fake
  draws.jsonl               60 fake generated stimuli (for select_stimuli.py --offline-embeddings)
  choices.csv               80 twins x 24 stimuli x 2 conditions on a 1-7 scale (for results.py)
  stimulus_moderators.csv   6 fake moderators for the 24 stimuli (for moderation.R)

The truth is known: the effect is 0.6 scale points at the average stimulus, and only `fake_driver` moderates
it (+0.4 per SD). `fake_correlated` is built to correlate with fake_driver (r = .8 in the population, less in a
sample of 24 stimuli), so it looks like a moderator too; the four `fake_noise` moderators are pure noise. Twins also differ in the size of the effect (SD 0.3).

    python scripts/make_fake_data.py
    python scripts/select_stimuli.py --config output/fake/config.yaml --offline-embeddings
    python scripts/results.py --config output/fake/config.yaml
    Rscript scripts/moderation.R output/fake

Run these from the protocol/ folder.
"""
import json

import numpy as np
import pandas as pd
import yaml

from common import DEFAULT_CONFIG, ROOT

OUT = ROOT / "output" / "fake"
N_TWINS, N_STIM, N_DRAWS = 80, 24, 60
rng = np.random.default_rng(7)
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
cfg["output_dir"] = "output/fake"
cfg["selection"]["k"] = N_STIM
(OUT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

# draws: random words for every field, so that the offline embeddings have something to compare
words = "apple river lamp budget phone hotel ticket coffee garden laptop sofa bike course plan doctor shoe".split()
draws = []
for i in range(N_DRAWS):
    w = rng.choice(words, 6)
    draws.append({"id": f"d{i + 1:04d}", "batch": i // 10 + 1, "situation": f"choosing a {w[0]} or a {w[1]} for {w[2]}",
                  "category": w[0], "option_a": f"the {w[0]}", "option_b": f"the {w[1]}", "source": f"a {w[3]} friend",
                  "channel": f"a {w[4]} message", "choice_text": f"She chose the {w[1]} near the {w[5]}.",
                  "rejection_text": f"She did not choose the {w[0]} near the {w[5]}.", "repeat": False})
(OUT / "draws.jsonl").write_text("".join(json.dumps(d) + "\n" for d in draws))

# moderators (per stimulus, 0-10 like the real ratings)
driver = rng.normal(0, 1, N_STIM)
mods = {"fake_driver": driver, "fake_correlated": 0.8 * driver + 0.6 * rng.normal(0, 1, N_STIM)}
mods.update({f"fake_noise_{k}": rng.normal(0, 1, N_STIM) for k in range(1, 5)})
stims = [f"s{j + 1:03d}" for j in range(N_STIM)]
pd.DataFrame({"stimulus": stims, **{k: np.clip(np.round(5 + 2 * v, 2), 0, 10) for k, v in mods.items()}}) \
    .to_csv(OUT / "stimulus_moderators.csv", index=False)

# twin responses: y = 4 + stimulus base + twin base + treat x (stimulus effect + twin effect) + noise, on 1-7
stim_base, stim_eff = rng.normal(0, 0.8, N_STIM), 0.6 + 0.4 * driver + rng.normal(0, 0.2, N_STIM)
twin_base, twin_eff = rng.normal(0, 0.6, N_TWINS), rng.normal(0, 0.3, N_TWINS)
rows = []
for t in range(N_TWINS):
    for j in range(N_STIM):
        for treat, cond in enumerate(["choice", "rejection"]):
            latent = 4 + stim_base[j] + twin_base[t] + treat * (stim_eff[j] + twin_eff[t]) + rng.normal(0, 1.0)
            y = int(np.clip(np.round(latent), 1, 7))
            rows.append({"pid": str(t + 1), "stimulus": stims[j], "condition": cond, "treat": treat,
                         "reversed": False, "reply": str(y), "rating": y, "y": y})
pd.DataFrame(rows).to_csv(OUT / "choices.csv", index=False)
print(f"wrote {OUT}: {len(rows)} responses, {N_STIM} stimuli, {N_TWINS} twins, {len(mods)} moderators, "
      f"{N_DRAWS} draws; true mean effect {stim_eff.mean():.2f}")
