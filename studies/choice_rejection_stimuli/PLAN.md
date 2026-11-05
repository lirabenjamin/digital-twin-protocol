# choice_rejection_stimuli: the review-frame effect across 100 generated stimuli

Status: written 2026-11-01, before any paid API calls. Kept as the dated plan; later changes are in the scripts' docstrings.

Source: a genericized choice/rejection vignette template from the research team (2026-11-01).

## Stimulus = context × source × channel × wording

Design decision (2026-11-01): stimuli vary in the product, in who conveys the other person's decision (reviewer, friend, salesperson...), in how it is learned, and in how the accept or reject is said (chose / did not choose, kept / returned, put in basket / put back...). Each generated stimulus carries its own matched `choice_text` and `rejection_text`; both say the other person ended up with B.

## The manipulation (template version)

The participant reads a neutral scenario: a binary choice between Option A and Option B. Then the participant reads one review. The two reviews are identical except for one clause:

- Choice frame (c = 1): "...and we chose [B] after doing some research."
- Rejection frame (c = 2): "...and we did not choose [A] after doing some research."

DV: "Which option would you [choose]?" A or B.

Outcome: `chose_B` = 1 if the twin picks the option the review names as chosen (choice frame) or the option the review does not name (rejection frame). Effect for stimulus j = P(B | rejection) − P(B | choice).

## Mapping to the De La Rosa procedure

| Step | delarosa_stimuli | choice_rejection_stimuli |
|---|---|---|
| 1. Seeds | 15 Study 2 decisions | 7 studies from Nan, Park & Yang (2023, JCR, 'Rejections are more contagious than choices'): `seeds.json`, with the human conformity rates. Human effects run from +8.7 to +27.9 points (rejection > choice) |
| 2. Generator | random Twin-2K-500 profile + seeds → one decision (Haiku, no thinking, no cache) | NO twins (design decision, 2026-11-01). `generate.py`: Sonnet gets the seeds plus the FULL running list and writes 10 new decisions that differ as much as possible from all of them (`prompts/generator.txt`) |
| 3. Same stimulus = within ε | ε from paraphrases at the 95th percentile | same code (`calibrate_epsilon.py`), new paraphrase prompt |
| 4. Stop rule | Good–Turing coverage, Chao1 | Chao1 and Good–Turing assume independent draws, and list-conditioned draws are not independent. Stop instead when the model runs dry: 3 batches in a row with fewer than 3 of 10 new decisions beyond ε of the pool, or when the mean nearest-neighbor distance of a batch falls below ε. Hard cap: 1,000 decisions |
| 5. Pick 100 | k-medoids on founders | same (`selection.py`) |
| 6. Write the screens | `prompts/scenario.txt` → one decision screen | `prompts/scenario.txt` fills every template slot: scenario, review stem, identical neutral evaluation, DV wording. The two reviews are built in code from one stem, so they cannot differ except in the clause |
| 7. Twin run | each twin: 14 stimuli × 2 pay conditions, same stimuli, random order | each twin: k stimuli × 2 frames, same stimuli; one isolated call per twin × stimulus × frame (see Q3) |
| 8. Effects | paired CI per stimulus; `glmer(expensive ~ daily + slot + (1 + daily \| stimulus) + (1 \| pid))` | `glmer(chose_B ~ rejection + (1 + rejection \| stimulus) + (1 \| pid))`; same paired CIs, LRT on the random slope, method-of-moments noise check |
| 9. Moderators | 20 (then 100) LLM-coded features, split-half hold-out | new catalog for this effect (draft below); same hold-out rule |

## Counterbalancing

Within each stimulus, which real option plays "A" (the rejected one) and which plays "B" (the chosen one) is random per twin, fixed across that twin's two frames. Without this, the frame is confounded with the content of B. The order of A and B on the DV screen is also random.

## Draft moderator catalog (for the frame effect)

1. Binariness: how clearly A and B are the only options (the rejection frame tells you about B only if the choice is binary).
2. Prior preference: how much a typical person would prefer A over B before the review.
3. Status quo or default: whether A is the default, usual or expected option.
4. Price gap between A and B.
5. Hedonic vs. utilitarian.
6. Risk or uncertainty about quality.
7. Expertise needed to judge the options.
8. Reversibility of the choice.
9. Social visibility of the choice.
10. Credibility of the review source.

The coding prompt is reviewed before it runs.

## Cost (estimates)

- Generator: about $0.005 per draw. 500–1,000 draws → $3–5.
- Twin run, isolated calls, k = 14: 2,058 × 14 × 2 = 57,624 calls. At about $0.005 per call (51K-token persona; less with prompt caching) → under $300. With k = 7: under $150.
- Feature coding: under $5.

## Open questions

Resolved during the project; see the scripts' docstrings.
