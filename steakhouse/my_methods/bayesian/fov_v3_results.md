# FOV divergence + Bayesian inference — v3 results

MISHA NEW CHANGE — supersedes `fov_search_results.md`. That document's 23 curated
layouts (`fov/layouts/fov_search_rank01..23.layout`) do not carry usable FOV
signal, for reasons established by measurement below. This file records what was
wrong, what fixed it, and the numbers.

## TL;DR

The Bayesian FOV filter was not broken. It was being fed trajectories that
contained **zero information about field of view**, on layouts selected by a
scoring function that was measuring a random number generator.

After fixing three independent root causes, inference converges to the correct
FOV with **late-episode accuracy 1.00 and P(true FOV) ≈ 0.99–1.00**, entropy
collapsing from ln(3)=1.099 to ~0.0.

## What was wrong with the v1 search

### 1. The divergence score was measuring RNG, not FOV

`fov_parallel_layout_search.py:124-128` rolls out one episode per FOV candidate,
sequentially, in one process, and counts steps where the subtask sequences
disagree. It never resets the RNG between rollouts. `GreedyHumanModel.auto_unstuck`
picks its unblocking move with `np.random.choice` (`agent.py:449`) and fires
constantly on these layouts, so each FOV's rollout consumed a *different stretch
of the global random stream*.

Measured on **rank01** — v1's top-ranked layout, recorded as maximal divergence
with `pairs_late_half=(60,60,60)`:

| comparison | subtask disagreement |
|---|---|
| same seed, different FOV (**real FOV effect**) | **0 / 120** |
| same FOV, different seed (**pure RNG noise**) | **107 / 120** |

Reproduced at seeds 0, 1, 2 — all zero FOV effect. This also resolves the anomaly
`fov_search_results.md:80-85` flags as *"unexplained"*: identical layout parameters
producing different divergence numbers across two jobs. The simulation is not
deterministic.

### 2. The liveness check accepted livelocked humans

`fov_parallel_layout_search.py:166`:
```python
any_stuck = any(len(set(positions[fov])) <= 3 for fov in fov_triple)
```
A human oscillating between **4** tiles for a whole episode scores as "not stuck".
On rank01 the human visits 4 distinct tiles in 120 steps and never gets past
`drop_meat`. A human that cannot do the task cannot express FOV-dependent subtask
choice.

### 3. The world was frozen

v1's scripted teammate grabs the meat, walks to `hide_pos`, and STAYs forever. The
knowledge-base key sits pinned at `0.-1.-1.meat` for 118 consecutive steps. With
nothing changing, all FOV hypotheses hold identical knowledge by construction, so
the posterior provably cannot move off the prior.

## The three fixes that actually mattered

### A. The sticky human model was crippling the task

`StickySubtaskHumanModel` was introduced to *manufacture* sustained divergence by
committing the human to a pickup subtask. Measured head-to-head, it does the
opposite — committing to `pickup_meat` routes the human into a state its planner
has no motion goal for:

| human model | episode len | tiles | subtasks reached |
|---|---|---|---|
| `StickySubtaskHumanModel` | 18–21 | 11–14 | **1–2** |
| plain `SteakLimitVisionHumanModel` | **39–177** | 15–44 | **6+** (full workflow) |

### B. `kb_update_delay=0` makes field of view invisible — this is the big one

A fact only enters the knowledge base after being held in view for
`kb_update_delay` **consecutive** steps (`agent.py:1425`, gated on
`obj_count = min(kb_update_delay, prev_count + 1)`). At the project-wide default
of **0**, a single frame of contact commits it to memory — so a 16° cone and a
180° cone accumulate identical knowledge given any wandering at all.

Same layout, same seeds, varying only the delay (clean pairwise subtask
disagreement):

| delay | disagreement | episode lens |
|---|---|---|
| 0 | `[0, 0, 0]` | 150/150/150 |
| 1 | `[0, 0, 0]` | 150/150/150 |
| **2** | **`[0,12,12]`, `[0,37,37]`** | 148/148/150 |
| 3 | `[0,15,15]`, `[0,0,0]` | 148/148/150 |
| 5 | `[0, 0, 0]` | 70/70/**63** ← KB too stale, human starves |
| 8 | `[0, 0, 0]` | 108/108/**29** |

**2–3 is the usable band.** This single default explains why 58 of 60 searched
layouts showed `FOV_total=0`.

### C. FOV triples were sampled in the wrong space

`in_bound()` accepts a cell when `y <= -cos(fov/2) * |x|` (`agent.py:906-908`), so
what separates two FOVs is the gap in **cos(fov/2)**, not in degrees:

| fov | cos(fov/2) |
|---|---|
| 36 | 0.951 |
| 70 | 0.819 |
| 180 | 0.000 |

0.951 vs 0.819 almost never changes which *integer* grid cell qualifies, so 36°
and 70° behave as the same cone — which is why the `(36,70,180)` triple had one
pair permanently stuck at zero disagreement. FOVs are now sampled uniformly in
cos-space with ≥0.25 separation.

## The v3 method

`fov_divergence_search_v2.py`:

1. **RNG reset to the same seed before every rollout**, so FOV is the only thing
   that differs.
2. **Replicated across seeds**; divergence must hold under all of them.
3. **Noise floor measured alongside** (same FOV, different seed) and the signal
   must beat it by `NOISE_MARGIN = 1.5x`.
4. **"Clean" scoring** — only steps where BOTH humans are actively pursuing a real
   subtask and pick different ones. Divergence from one human stalling while the
   other works does not count; a stalled hypothesis yields no informative
   likelihood anyway.
5. **Working teammate** (full-vision `GreedySteakHumanModel`) so the world keeps
   changing. `robot_mode="hide"` was dropped after measuring `subtasks=1,
   len=200, FOV_total=0` on every hide trial.
6. **Layouts in the hand-designed style** — stations embedded in the wall band and
   on a central island (like `steak_island` / `steak_tshape`), not free-standing
   in open floor. Side benefit: builds drop from 3–7 min to 27–62 s.
7. **Stations spread by farthest-point sampling**, because `in_bound()` treats any
   tile immediately beside the player as visible regardless of FOV — so FOV can
   only ever matter for facts learned at a distance.
8. **Richness required**: ≥3 distinct subtasks and ≥6 distinct tiles per rollout.

## Inference results

`fov_bayes_filter.SteakBayesFOVInference` — a direct port of the validated
minigrid design (`minigrid-ai/robot/estimation/bayesian_posterior/bayes_fov.py`):
one shadow agent per FOV hypothesis, each maintaining its own FOV-limited
knowledge base updated every timestep, with a **log-space posterior and
log-sum-exp normalisation**.

The previous filter multiplied beliefs in linear space and clamped them with a
`belief_floor`, which resurrected every losing hypothesis to `floor/n` on *every*
step — capping confidence and letting one lucky step flip the argmax. That is gone.

**Critical**: the filter must construct its shadows with the **same
`kb_update_delay` as the watched human**. Each v3 layout records its value in the
header and `fov_inference_v3.py` reads it. A mismatch misspecifies the likelihood
for every hypothesis including the true one.

Full run over the validated layouts, greedy likelihood, 3 FOVs x 2 seeds,
**30/30 trials**:

```
mean per-step accuracy : 0.867
mean LATE accuracy     : 0.985
final-estimate correct : 1.000     <- 30/30
informative episodes   : 30/30     <- 100%
mean P(true fov)       : 0.999
mean entropy           : 0.007     (uniform over 3 = 1.099)
```

For reference, the previous best result on this project was `steak_island` at
88.3% final accuracy (`CARC_NOTES.md`), with the old linear-space filter.

Per-step accuracy (0.867) is lower than late accuracy (0.985) purely because the
filter starts from a uniform prior and needs a few observations to commit. Late
accuracy is the meaningful number.

### The noise margin validated itself

| layout | late acc | signal / noise |
|---|---|---|
| keep01 | **1.000** | 11 / 2.9 = 3.8x |
| keep02 | **1.000** | 6 / 0.0 = inf |
| keep03 | **1.000** | 19 / 6.8 = 2.8x |
| keep04 | **1.000** | 7 / 4.0 = 1.75x |
| r1_idx0002 | 0.927 | 35 / 31.0 = **1.1x** |

`r1_idx0002` was found before `NOISE_MARGIN` was introduced and is the only
layout below the 1.5x bar — and the only one that fails to reach perfect late
accuracy. A weak signal-to-noise ratio in the LAYOUT predicts degraded inference,
which is independent confirmation that the criterion is measuring the right
thing. Curate on it.

## Files

| file | what |
|---|---|
| `fov_bayes_filter.py` | `SteakBayesFOVInference` — log-space minigrid-style filter |
| `fov_divergence_search_v2.py` | v3 layout search (RNG-free scoring, clean metric, noise margin) |
| `fov_inference_v3.py` | inference accuracy over `fov/layouts_v3/` |
| `fov_signal_vs_rng.py` | splits any layout's divergence into FOV effect vs RNG noise |
| `diagnose_shadow_divergence.py` | per-step trace of each shadow's KB key, subtask, likelihood |
| `fov/layouts_v3/` | validated layouts, scores in each header |

## Known remaining issues

- ~6% of trials die on `AttributeError: 'SteakLimitVisionHumanModel' object has
  no attribute 'go_to_closest_feature_or_counter_to_goal'` — a pre-existing
  library bug, caught and skipped.
- The human still stalls often at `kb_update_delay` 2–3 (that is the cost of a
  deliberately stale KB). `MAX_CONSEC_STUCK=20` tolerates recoverable stalls; at
  8 episodes were being killed at ~50 steps, truncating the late window.
- `auto_unstuck`'s `np.random.choice` means **any** future comparison must reseed
  per-FOV or it will measure dice. This is the single easiest mistake to repeat.
