# override_v2 — STATUS (pick up here)

**Goal.** A FOV module with FULL AUTHORITY: the frozen FOV-blind baseline only
*suggests* a task; the module, from its inferred-FOV posterior + entropy, may
**override anything** — the task, the low-level path, both — or defer when unsure.
Question: does it deliver in LESS time with NO WORSE delivery than the baseline?

## Architecture (locked with user)
- **baseline = what task** (trained, FOV-blind, frozen — never retrained)
- **module = can override the task AND/OR reshape the path**, gated by entropy
  - high entropy (unsure) → **defer** to baseline
  - confident + `fov <= blind_fov_max(90)` → **takeover**: cook the pipeline itself
  - confident + `fov >= sighted_fov_min(120)` → **visible**: act at a station
    INSIDE the inferred cone + hug that cone en route (so the human sees & yields)
- deployment-time only → any gain is attributable to FOV info (clean control)
- cone computed by reusing the human's own `visible()` at the INFERRED angle +
  the human's OBSERVED pose — **no cheating** (never reads the true FOV)

## Files (all NEW; validated baseline/, module/fov_module.py, inference/ untouched, imported read-only)
- `override.py` — `FOVOverride` (task authority, angle thresholds) + `ConeReroute` (trajectory authority, BFS + `knob`)
- `env.py` — `OverrideEnv(RobotAssistEnv)`; toggles `override_task`, `reroute`, `knob`, `conf_min`, thresholds; diagnostics `n_override/n_takeover/n_visible/n_in_cone`
- `train.py` — self-contained PPO, trains the FOV-blind baseline checkpoint (quiet rollouts, progress→stderr)
- `evaluate.py` — baseline vs override, PAIRED per (fov, seed): delivered, t_complete, override%, in_cone%, WIN/tie/WORSE
- `tests.py` — mechanism tests (control never overrides; narrow-fov takes over; wide-fov visible fires; reroute never lowers in-cone)
- `_util.py` — `quiet()` to silence the greedy model's per-step debug prints

## DONE
- Package written, `py_compile` OK.
- Mechanism proven earlier on a RANDOM baseline: override fired 78–85% when
  confident, defers at high entropy, delivery ran, no crash. (Reroute correctly
  no-ops on narrow cones; matters for wide cones only — consistent with design.)
- Removed the two interim prototypes (`module/fov_override.py`, `module/env_override.py`) — superseded by this package.

## NEXT (run in order, next session — NOTHING has been run yet on a trained net)
```bash
cd steakhouse && export PYTHONPATH="$PWD:$PYTHONPATH"
PY=/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python
# 1. mechanism tests (seconds)
$PY -m fov.robot.policy.override_v2.tests steak_side_2
# 2. train a FOV-blind baseline (minutes; try 100 iters first, watch stderr return climb)
$PY -m fov.robot.policy.override_v2.train 100 steak_side_2 baseline_steak_side_2.pt
# 3. the honest comparison
$PY -m fov.robot.policy.override_v2.evaluate baseline_steak_side_2.pt steak_side_2 8
```
Then: sweep `knob` / thresholds / layouts (contention layouts steak_gc00/gc06/cram best);
if it wins locally, scale the sweep on **CARC** (account biyik_1173, scratch1, sbatch — never login node).

## OPEN / to decide with user
- Is hard task-override (takeover/visible) better than the old SOFT logit-bias
  module (`module/fov_module.py`)? evaluate both, compare honestly.
- `visible` currently targets in-cone stations that "need attention"; may want to
  restrict to stations where the robot can make a genuinely NEW visible change.
- **Separate queued thread:** `features_full` observation redesign (flat per-kind
  + grid/CNN exact-per-object) so the BASELINE sees full state (robot self, full
  human, 8-class held vocab, keep age/drop distance, normalized xy). Not started.
