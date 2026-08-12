# dev/ -- measurement scripts, not part of the filter

Nothing in `core/`, `harness/` or `analysis/` imports anything in here. Deleting
this whole directory leaves the filter working. These are the instruments that
produced numbers, kept so the numbers can be reproduced or disputed rather than
taken on trust. The dependency runs one way only:

    core/  <-  harness/  <-  { analysis/, dev/, tests/ }

The one thing they DO need from `core/` is three `info` keys that
`qmdp_exec.action()` emits and nothing else reads: `base_subtask`, `top_subtask`,
`base_pair`, plus `pair_idx` / `act_idx` / `base_pair_scored`. They are additive
readouts. If you ever want them gone, delete them and delete this directory
together.

| file | what it answers |
|---|---|
| `subtask_dev.py` | How often does the layer commit to a job its baseline did not pick, does that change the robot's move, and does it follow the job through to an INTERACT? |
| `swaps.py` | WHICH job does it swap for which, per layout, with the estimated tick gain it thought it was buying. Reads `subtask_dev --trace-out`. |
| `why_probe.py` | On the ticks it drops a rung of the ladder, is the job it picked actually CLOSER? Tests "trades urgency for proximity". |
| `parity_probe.py` | Does `pair_k=1` really collapse the search to one candidate? (It does not -- see below.) |
| `run_dev.sh` | The grid, one process per (layout, cone) cell, like `../harness/run_local.sh`. |

## Running them

    python -m robot.filter.dev.subtask_dev --layouts divide --fovs 60 --seeds 0
    ./robot/filter/dev/run_dev.sh OUT 6 "qmdp,qmdp-greedy,qmdp-solo,qmdp-base" "0-9"
    python -m robot.filter.dev.subtask_dev --load OUT/*.jsonl --only qmdp

`--only` matters. The pooled block is meaningless across methods: `qmdp-base` is a
parity control whose entire purpose is not to deviate, so pooling it with the live
pairings drags every rate down by roughly a quarter.

## The two things to know before reading any output

**`deviated_frac` in `harness/evaluate.py` is a different question and the two
counts are not nested.** It counts ticks where the emitted ACTION differs. A
different job can produce the same action -- two jobs sharing the first ten steps
down a corridor -- and the same job can produce a different action, a longer way
round or a STAY. So neither count bounds the other, and an action-deviation rate
cannot be read as a job-switching rate. That is a fact about the two definitions,
not a measurement; no rate is quoted here because none has been reproduced against
the current tree.

**`enum_gap` is now an INVARIANT CHECK, not a confound to subtract.** It was
written when `pairs[0]` was the top job by aggregated pi mass, which is not the
same object as the baseline's realised pick -- so the committed job could differ
without the search having moved anything, and the parity control looked like it
deviated when it had not. DESIGN.md §3.0b closed that: `_jobs` now takes the
realised pick and forces it to index 0 with its own cell first. So `enum_gap`
should read ~0, firing only when the pick is missing from `ranked` altogether. If a
run shows it materially above zero, the anchor is broken again -- that is the
reason to keep the counter rather than delete it.

`off_job_q` (job differs AND `pair_idx > 0`) is still the number to quote, because
it is the one that stays meaningful whichever way the anchor behaves.

## What `parity_probe.py` found

`qmdp_exec.action()`:

    pairs = pairs[:max(1, self.pair_k - 1)] + [must]

With `pair_k=1`, `max(1, 0)` is `1`, so the must-keep append yields TWO pairs, not
one -- the `qmdp-base` control is not actually collapsed to a single candidate.
`only_base=True` gives each pair exactly one action, so two pairs means two
candidate actions and the min can land on the committed pair's rather than the
baseline's. That is a mechanism for the parity break RESULTS.md section 4 records
as real and unexplained.

MECHANISM CONFIRMED, LOCATION NOT ESTABLISHED. `n_pairs > 1` fires on ~0.9% of
ticks, so the collapse genuinely does not hold, and that is a code fact readable
from the line above. Where it turns into an action deviation is NOT established:
two runs disagreed (pantry fov90 vs pantry fov180), which cannot both be right on a
seed-independent episode. `core/qmdp.py` and `core/value_tail.py` were both
edited while those grids were in flight, including the realised-pick anchor fix,
which changes every candidate list. Neither run's cell attribution should be
quoted. RESULTS.md section 4 names pantry fov180 from the CARC grid; that is the
number to trust until someone re-runs it against one fixed tree.

NOTHING IN THIS DIRECTORY HAS BEEN RE-RUN SINCE that anchor fix, so no rate any of
these scripts printed is currently reproducible. The scripts are correct; the
numbers are stale.
