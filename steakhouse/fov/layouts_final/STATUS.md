# STATUS — read this first to catch up (2026-07-27)

Project: **inferring a limited-vision human's field-of-view for seen/unseen
assistance** in Overcooked "steak". End goal: an **FOV-aware robot completes
orders in less time than an FOV-unaware robot**, using only physical action (no
communication) — so its one lever is **stepping into / out of the human's cone**.

Current phase: **the human agent + layout library + settings are FINALIZED, and
the end-to-end FOV-aware-robot experiment is now BUILT and run** (2026-07-27/28).

**Robot end-to-end — see [`fov/robot/policy/RESULTS.md`](../robot/policy/RESULTS.md).**
A FOV-blind PPO baseline (`baseline/`) vs a FOV-aware module (`module/fov_module.py`)
that biases the baseline's logits from `SamplingBayesFOVInference` (the only place
FOV enters). Honest finding: the FOV-aware robot completes orders faster; the size
depends on the baseline — **~72% of settings win vs well-trained baselines (mean
−13 steps at narrow FOV, up to −185 on hard layouts), and ~2× vs baselines lacking
the station-idle/human-position proxies.** FOV inference and those proxies are
SUBSTITUTES (proven by the `STEAK_MINIMAL_FEAT` ablation). Winning module config:
work-only bias, `strength 2-3, kw 2-4, ks=0, kt=0`. Eval scripts in `fov/eval/`
(compare_time / search_layout / ablate). Trained baselines in
`fov/robot/policy/models/`. A remaining lever (untried): a module that
COMPLEMENTS a sighted human at wide FOV instead of staying out.

## What exists now

- **ONE human file**: [`fov/human/agent/limited_vision_human.py`](../human/agent/limited_vision_human.py).
  - Samples subtasks from FOV-gated decaying beliefs; starts knowing nothing;
    routes by its own BFS over seen floor (no full-map planner). Metric = **team**
    delivery ("team win is a win").
  - Two always-on teammate-receptiveness channels: **fetch-suppression** (see robot
    carrying X → don't fetch X) and **station-yield** (see robot at a station →
    yield it). Both FOV-gated, soft, no-cheat.
  - Two opt-in toggles, **default off = identical to base**: `avoid_robot`
    (traffic-avoidance) and `occlude` (line-of-sight / bigger blind spots).
- **31 validated layouts** in [`layouts/`](layouts/); catalogued in
  [`LAYOUTS.md`](LAYOUTS.md).
- Docs: [`SUMMARY.md`](SUMMARY.md) (human + setup + properties), `LAYOUTS.md`
  (full battery per layout + perf), this file.
- Test harnesses persisted in [`fov/eval/`](../eval/) (scratchpad may be wiped on
  restart — these are the saved copies).

## Validated results (all fresh, 2026-07-27)

| property | result |
|---|---|
| FOV → different high-level behaviour (P1) | ✅ batch_validate 12/12 separation, minDiv 7–16; battery P1 4–22 |
| exact Bayesian FOV inference | ✅ **0.872** overall (canonical, 8 layouts incl. contention); ~0.84 across 32 |
| no cheating | ✅ real guarantees (no illegal writes, start UNKNOWN) pass everywhere |
| team-win every FOV | ✅ 12/12 (batch) · 31/31 (battery) |
| P2 blind spots grow as FOV narrows | ✅ 31/31 |
| robot-in-view **influence** (aware-vs-blind divergence) | up to **28.5**; 25/31 layouts "influential" (≥8) |
| **performance value** of that influence (aware vs blind human, team throughput) | **+12.5% contention (up to +21%)**, +4.7% spread |

## Key decisions / findings (so you don't re-litigate them)

- **Contention layouts are the win.** Cramped/clustered maps (`steak_gc00` INFL
  28.5, `gc06`, `cram`, `cram2`, `mid_1`) make seeing the robot matter most AND
  cash out to the biggest throughput gain (+21%). Use these as the primary
  testbed. Open layouts: influence & gain are modest (greedy partner compensates).
- **Rejected teammate channels** (tested in throwaway files, all failed the
  gauntlet): held-item redundancy (anti-gated), staging (idle-harm), belief-transfer
  (redundant without occlusion), assembly-suppress (dead), complementary-boost /
  lookahead-prep (hurt divergence). Only fetch-suppress + station-yield survived.
- **Suppression strength doesn't change influence** (swept 0.3→0.0, flat): the
  per-tick influence ceiling is structural; **contention**, not reaction-strength,
  is what raises it.
- **Occlusion** enlarges blind spots (P2) but does NOT raise influence and can
  over-blind on tight layouts → kept as an **opt-in toggle**, not always-on.

## What's NEXT (the not-yet-done proof)

1. **Build the FOV-aware and FOV-unaware robot policies** and measure
   **time-to-complete** aware vs unaware. This is the actual thesis; everything
   above is the human-side scaffolding that makes it *possible*. Run it first on
   the **contention layouts**. (A prior premature robot was deleted — start clean.)
2. Optional: full re-validation with `occlude=True` (perception-model change), and
   the `avoid_robot` toggle, if the end-to-end run needs them.

## Environment / how to run

- Env: `steakhouse-ai` (py3.8) — `/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python`.
- Full battery on one layout: `python fov/eval/fulltest_layout.py <layout> 4 3 12 300`.
- Canonical: `python -m fov.human.batch_validate 6 3`,
  `python -m fov.robot.inference.evaluate_sampling_inference <layout> 6`,
  `python -m fov.human.test_no_cheating <layout> 6`.
- Heavy compute: USC CARC (`ssh mishafu@discovery.usc.edu`, account `biyik_1173`,
  use scratch1, sbatch not login node).
- Workflow rule: risky changes go in a NEW file, tested, merged only if proven;
  keep ONE clean human file.
