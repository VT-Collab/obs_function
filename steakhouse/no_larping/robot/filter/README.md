# robot/filter/ -- layout

Four directories, and the dependency runs ONE WAY:

    core/  <-  harness/  <-  { analysis/, dev/, tests/ }

Nothing in `analysis/`, `dev/` or `tests/` is imported by `core/`. Delete any of
those three and the robot still runs; delete `core/` and nothing does.

## core/ -- what the robot actually runs

The only directory `robot/methods.py` and `play.py` import from.

| file | role |
|---|---|
| `qmdp.py` | the controller, and the only one. INTERACT-legality is a predicate on (position, orientation) over the union of the top-m jobs' cells; every reachable (cell, arrival tick) is scored to depth 30. |
| `fov_posterior.py` | the inference. P(theta \| the human's actions) over vision cones. |
| `value_tail.py` | the tail of `C = t_end + tail`. Bounded A* on the ladder graph. |
| `progress.py` | `orders_remaining`, which is what makes the tail need-aware. Imported by `value_tail.py` and that is the whole reason it is in `core/` -- `potential()` and its three helpers below it are NOT wired to anything. |

## harness/ -- produces grids

| file | role |
|---|---|
| `evaluate.py` | headless episode runner, one JSON line per episode. No pygame, so it runs on a compute node. |
| `run_local.sh` | the grid across local cores, one process per (layout, cone). |
| `run_carc.sbatch` | the same grid as a 30-task SLURM array. |

## analysis/ -- reads grids

| file | role |
|---|---|
| `report.py` | the gate: each filter against the baseline it wraps, per cell. Also owns `NAME_FIX`, which normalises method names so a grid written under an earlier vocabulary still reports. |
| `compare_grids.py` | diff two grids cell by cell. Did a change move anything? Refuses to tally across a layout change. |
| `results_tables.py` | emits the RESULTS.md tables from a grid directory. Numbers in that document are generated, never typed. |
| `layout_facts.py` | which stations each agent can WALK to, derived from the .layout files rather than transcribed. RUN THIS FIRST when two grids disagree. |

## tests/

`test_qmdp.py` -- the checks that guard `core/`, including the
degrade-to-baseline parity check. Run it directly.

## dev/

Measurement instruments written to answer one question each. See `dev/README.md`.

## Running things

    python -m robot.filter.harness.evaluate --layouts divide --fovs 30 --seeds 0 --methods qmdp
    ./robot/filter/harness/run_local.sh OUT 6 "greedy,solo,handoff,bayes,qmdp"
    python -m robot.filter.analysis.report OUT/*.jsonl
    python -m robot.filter.analysis.results_tables OUT
    python -m robot.filter.analysis.layout_facts
    python -m robot.filter.tests.test_qmdp
    python -m robot.filter.dev.subtask_dev --layouts divide --fovs 60 --seeds 0

## The eleven rows, one filter

A baseline column and one filter over each of them, nothing else:

    greedy  solo  handoff  bayes        four theta-blind baselines, all DRAWING
                                        their sub-task from their own pi
    bayes-noip                          evidence control for bayes
    qmdp-greedy  qmdp-solo  qmdp  qmdp-bayes    ONE filter class over each
    qmdp-base  qmdp-fixed               parity control, theta-blind control

`qmdp` is scored against `handoff`, `qmdp-solo` against `solo`, and so on -- the
filter against the exact object it wraps. There is no deterministic variant of
any baseline: the filter consumes a distribution over sub-tasks and a policy that
never draws does not have one, so keeping both kinds meant scoring a filter fed a
modelled `pi` against a baseline that never held those preferences.

## Before comparing any two grids

Every row carries `layout_sha`. Four of the six .layout files were edited
mid-experiment and the hand-transcribed station table went on describing a kitchen
that no longer existed -- it had been correct when written, which is worse than
being wrong. Two grids with different fingerprints are measuring different
kitchens, however tempting the diff looks. `compare_grids.py` checks this; so does
`dev/subtask_dev.py --load`, which refuses to pool mismatched rows.
