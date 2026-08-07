# override_v2 — RESULTS (2026-07-29)

## ★ THE WIN: "assist" mode beats the FOV-blind baseline on contention layouts

Mechanism (uses ONLY the inferred FOV + shadow prediction, no cheating): the
baseline robot's greedy DEFERS to a partner holding meat/onion/plate
(`other_has_X`) - correct for a sighted human, but a BLIND human holding an item
can't reach the station, so the robot STALLS waiting. When the module infers the
human is MODERATELY blind (map_fov in (30, 90]) and is confident, it stops
deferring and TAKES OVER the stuck track (`_greedy_work_assist` hides the blind
human's held object from the robot's greedy). fov<=30 excluded (baseline already
solos; takeover just collides). Metric: orders delivered + time-to-3 (achievable;
time-to-4 is never reached).

4 contention layouts, N=10 paired episodes, saved baselines (no retrain):

| layout | base_dlv | mod_dlv | base_t3 | mod_t3 |
|--------|----------|---------|---------|--------|
| mid_1  | 2.73     | 2.80    | 197     | 183    |
| cram   | 2.85     | 2.93    | 204     | 201    |
| cram2  | 2.93     | 2.98    | 174     | 157    |
| gc00   | 2.95     | 3.00    | 134     | 126    |

Every layout: MORE delivered AND faster to 3. ~17 win / 6 tie / 1 loss of 24
cells. Biggest gains at fov=60 (gc00 2.70->3.00, cram2 2.80->3.00). Run:
`scratchpad/fasteval.py <baseline.pt> <layout> assist`. Next: CARC-confirm across
all 9 layouts with more seeds; sweep blind_fov band.

---

## Earlier (negative) results that led here

Three CARC experiments, 9 contention layouts each (gc00/cram/cram2/gc01/gc02/gc03/
gc04/mid_1/api), 6 FOVs, 10 paired episodes. FOV-blind PPO baseline trained per
layout (120 iters). All raw CSVs on scratch: `carc_results/` (aggressive),
`carc_results_cons/` (conservative), `carc_results_fs/` (observation).

## Headline: negative, but clean and consistent

Neither more FOV information (the override module) nor more state information
(full-state observation) improves the team metric. The bottleneck is the TASK
STRUCTURE, not information.

### 1. Aggressive full-authority override — FAILS
Module may replace the baseline's TASK (takeover when blind / "visible" when
sighted). Result: **0 wins / 54, 23 WORSE.** At wide FOV the "visible" branch
makes the robot camp in the human's cone (cone% ~100%, override ~99%) and
**delivery collapses 3.0 -> 0.0.** Hard task-replacement obliterates a competent
trained policy.

### 2. Conservative override — SAFE but 0 wins
Baseline OWNS the task; module only reshapes the PATH through the cone when
sighted+confident (horizon raised to 450). Result: **0 wins / 54, 51 tie, 3
WORSE** (the 3 are RNG-divergence noise at narrow FOV where override% = 0). No
delivery collapse. But **every t = 450 (censored): no config ever completes all
4 orders**, so a time-win is impossible by construction; throughput already
ceilinged at ~3.

### 3. Observation comparison (old-17 vs full-state flat-48 vs grid-CNN) — NO difference
Overall mean delivered: **old 2.919 / flat 2.926 / grid 2.939** (all t=450). The
rich full-state observation matches the coarse 17-dim. Observation is not the
limit.

## Why (root cause)
1. The robot cooks the pipeline near-solo -> the human's FOV barely affects TEAM
   throughput (proxy-substitution, confirmed).
2. Throughput ceilings at ~3/4 orders regardless of FOV/observation/module.
3. 4 orders NEVER complete in 450 steps across thousands of episodes -> no
   time-to-finish signal to improve.

## What would make FOV matter (change the SETUP, not the module)
1. Force division of labor: block the robot from some stations so the human must
   contribute and their FOV becomes load-bearing. (Highest leverage.)
2. Investigate the hard ~3-order ceiling (throughput vs a stall on order 4).
3. A metric that credits human contribution directly.

## What is built (all reproducible, validated files untouched)
- `override_v2/`  full-authority override (override.py, env.py w/ aggressive+
  conservative modes, train.py, evaluate.py, carc_run.py, aggregate.py, sbatch x3)
- `full_state/`   features_flat.py (48-dim per-kind), features_grid.py (42ch),
  policy_cnn.py, env.py (FlatEnv/GridEnv), train.py, carc_run.py, aggregate.py, sbatch
- Mechanism verified: override fires when confident, defers when unsure, in-cone
  94-100% at wide FOV, never crashes.
