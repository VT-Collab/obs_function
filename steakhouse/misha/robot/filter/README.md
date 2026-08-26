# robot/filter/ -- layout

Mirrors no_larping/robot/filter/'s package shape, self-contained: nothing here
imports from no_larping at runtime.

    core/  <-  harness/  <-  { analysis/, tests/ }

Nothing in `analysis/` or `tests/` is imported by `core/`. Delete either and the
robot still runs; delete `core/` and nothing does.

## core/ -- what the robot actually runs

Two INDEPENDENT filters live here, sharing only `fov_posterior.py`'s cone
inference and nothing else -- they do not call into each other.

| file | role |
|---|---|
| `fov_filter.py` | **the validated one, but not what's wired up right now** -- see `my_fov_filter.py` below. `FOVFilter`: no task knowledge at all, only "over the ticks I spend doing this sub-task, how often and how soon would the human see me", priced in ticks of the baseline's own cost. `C(plan) = R + D - bonus`, `bonus = cap * SUM decay**(j-1) * gamma**(k-1)`, bounded by `max_influence = cap/(1-decay)`. Self-contained -- does not use `value_tail.py` or `progress.py`. `RESULTS.md` has its cap=8 measurements (all losses -- see that file). |
| `qmdp.py` | the OTHER filter, not used by this grid. Re-derives what the kitchen needs (recipe leg, orders left, a stash's handoff) and scores a plan by ticks-to-finish via a bounded A* tail. Uses `value_tail.py` and `progress.py`. |
| `fov_posterior.py` | the inference both filters read. `P(theta \| the human's actions)` over vision cones -- one shadow `LimitedVisionHuman` per cone, reweighted by how well it predicted the human's last action. |
| `value_tail.py` | `qmdp.py`'s tail: bounded A* on the ladder graph. Not used by `fov_filter.py`. |
| `progress.py` | `orders_remaining`, which `value_tail.py` calls. Not used by `fov_filter.py`. |
| `my_fov_filter.py` | **the one `robot/methods.py` actually wires up now.** Misha's own hand-written FoV-only filter -- progress-gated `seen` credit, per-cell budget/floor constraint, `committed_pos` anti-reversal. `_fov()`'s `build()` imports this class and constructs it at its own `__init__` defaults, silently dropping any `cap`/`frozen_fov`/`fov_decay` kwarg a row passes (it doesn't accept them yet) -- see the note below and `RESULTS_my_fov_filter.md`. |

## harness/ -- produces grids

| file | role |
|---|---|
| `evaluate.py` | headless episode runner, one JSON line per episode. No pygame, so it runs on a compute node. Supports `exec:fov:<baseline>:<k=v,...>` for one-off cap/decay sweeps not worth a registry row. |
| `run_local.sh` | the grid across local cores, one process per (layout, cone). |
| `run_carc.sbatch` | the same grid as a 120-task SLURM array (30 cells x 4 pairings). |

## analysis/ -- reads grids

| file | role |
|---|---|
| `report.py` | the gate: each filter against the baseline it wraps, per cell. Owns `NAME_FIX`/`PAIRS`. |
| `compare_grids.py` | diff two grids cell by cell. Refuses to tally across a layout change. |
| `results_tables.py` | emits `RESULTS.md`'s and `RESULTS_my_fov_filter.md`'s tables from a grid directory. Numbers in both documents are generated, never typed. |
| `layout_facts.py` | which stations each agent can WALK to, derived from the `.layout` files. Run this first when two grids disagree. |

## tests/

`test_fov_filter.py` -- the checks that guard `core/fov_filter.py`: parity at
cap=0, the bounded-influence theorem over a cap sweep, the bonus identity, the
tier-safety guard, and the two refusal guards. Run it directly.

## Running things

    python -m robot.filter.harness.evaluate --layouts divide --fovs 30 --seeds 0 --methods "handoff,fov-c8"
    ./robot/filter/harness/run_local.sh OUT 8 "greedy,fov-greedy-c8,solo,fov-solo-c8,handoff,fov-c8,bayes,fov-bayes-c8"
    python -m robot.filter.analysis.report OUT/*.jsonl
    python -m robot.filter.analysis.results_tables OUT
    python -m robot.filter.analysis.layout_facts
    python -m robot.filter.tests.test_fov_filter

## The registry, robot/methods.py

A baseline column and the FoV-only filter over each of them, at several caps:

    greedy  solo  handoff  bayes        four theta-blind baselines, all DRAWING
                                        their sub-task from their own pi
    bayes-noip                          evidence control for bayes
    qmdp  qmdp-greedy  qmdp-solo  qmdp-bayes    the OTHER filter (not this grid)
    qmdp-base  qmdp-fixed               qmdp's parity/theta-blind controls
    fov  fov-greedy  fov-solo  fov-bayes         my_fov_filter.FOVFilter at its
                                                  OWN defaults, over each baseline
    fov-base  fov-fixed  fov-free       nominally FOVFilter's parity / theta-blind /
                                        unbounded controls (cap=0 / frozen 90 / cap=1e6)
                                        -- INERT right now, same as the row above
    fov-c1  fov-c2  fov-c8  fov-c21     nominally the cap SWEEP over handoff, 1/2/8/10.7
                                        -- INERT right now, all four identical
    fov-greedy-c8  fov-solo-c8  fov-bayes-c8    nominally cap=8 over the other three
                                                baselines -- INERT, see below
    fov-d02  fov-d08                    nominally the decay sweep at fixed budget=8
                                        -- INERT right now, identical to the row above

    Every `fov-*` row above currently builds the exact SAME object
    (`my_fov_filter.FOVFilter(mdp, base, post, agent_index=robot_idx)`, no
    kwargs) -- `_fov()`'s `build()` drops `cap`/`frozen_fov`/`fov_decay`
    because `my_fov_filter.FOVFilter.__init__` doesn't accept them yet (see
    the `#---- CHANGED` comment on `_fov()`). The row names/blurbs describe
    what `core/fov_filter.py` used to do when these names were wired to it;
    they are stale labels on an inert sweep until `my_fov_filter.py` grows
    those knobs. `RESULTS_my_fov_filter.md` is the grid that actually measures
    what these rows do today.

`fov-c8` is scored against `handoff`, `fov-solo-c8` against `solo`, and so on --
the filter against the exact object it wraps. See `RESULTS.md` for what the
`cap=8` row did back when it was wired to `core/fov_filter.py` (all losses);
see `RESULTS_my_fov_filter.md` for what it does today, wired to
`my_fov_filter.py` at that file's own defaults.

## Before comparing any two grids

Every row carries `layout_sha`. `misha/layout/layouts` and `no_larping/layout/layouts`
are DIFFERENT kitchens (confirmed by fingerprint, not by eye) -- do not compare a
number from this package's grids against a number in `no_larping/robot/filter/RESULTS.md`.
`compare_grids.py` refuses to tally across a fingerprint mismatch within this
package; there is no tool that compares across packages, so check the
fingerprints by hand if you ever need to.
