# `my_fov_filter.py` -- full grid, all four baselines

`FOVFilter` (`core/my_fov_filter.py`) -- Misha's hand-written prototype, over all
four baselines, default hyperparameters (`weight_progress=1`, `weight_seen=1`,
`seen_bonus=10`, `budget_mult=1.5`, `stick=0.5`, `m=3`, `depth=12`).

**Do not confuse this with `RESULTS.md`**, which is `core/fov_filter.py` (a
different class, different file) at `cap=8`. `robot/methods.py`'s `fov-*-c8`
row names are a holdover from when they wired to `core/fov_filter.py`; the
`_fov()` builder now imports `my_fov_filter.FOVFilter` and silently **drops**
the `cap`/`frozen_fov`/`fov_decay` kwargs those rows still pass -- every
`fov-*` row (the whole cap sweep, the parity/theta-blind/unbounded controls,
the decay sweep) now constructs the exact same object (see the `#---- CHANGED`
comment on `_fov()` in `methods.py`). So despite the `-c8` suffix, this grid
is `my_fov_filter.py` at its own defaults, not any cap=8 configuration --
there is no cap/decay knob to sweep here, only the hyperparameters above.

**Grid.** CARC job `11225684`. 6 layouts x 5 cones x 4 pairings x 20 seeds x
horizon 400 = **4800 episodes**. `sacct`: 120/120 COMPLETED, 120 output files,
0 empty, exactly 4800 lines total. Output:
`/scratch1/mishafu/fov_myproto_20seed_20260820` (not synced to `/home1` --
quota). Layouts: `back_bar`, `butchery`, `divide`, `pantry` are the versions
re-synced to CARC on 2026-08-20 (`divide` carries its newest edit); `banquet_pass`
and `chefs_table` are unchanged since 08-17. Current copies of all six are
snapshotted at `layout/layouts/_snapshots/2026-08-20/`.

**Regenerate:**

    ssh mishafu@10.72.0.13   # discovery1 -- see CARC_NOTES.md, discovery2 is flaky
    cd ~/steakhouse/misha
    python -m robot.filter.analysis.results_tables /scratch1/mishafu/fov_myproto_20seed_20260820

---

## 1. Headline

| pairing | win | tie | loss | dishes: baseline -> layer | change |
|---|---|---|---|---|---|
| `fov-greedy-c8` vs greedy | 18 | 1 | 11 | 36.8 -> 43.7 | +6.9 |
| `fov-solo-c8` vs solo | 22 | 2 | 6 | 35.6 -> 43.2 | +7.7 |
| `fov-c8` vs handoff | 22 | 2 | 6 | 35.6 -> 43.2 | +7.7 |
| `fov-bayes-c8` vs bayes | 22 | 2 | 6 | 25.5 -> 35.2 | +9.7 |

win/tie/loss counts CELLS (more dishes wins; at equal dishes, fewer ticks).
Positive on every pairing, majority win rate everywhere (18-22 of 30 cells).

## 2. The whole table

```
layout       | fov | greedy   +fov8    | solo     +fov8    | handoff  +fov8    | bayes    +fov8
back_bar     |  30 | 0.1/400  0.1/400  X | 0.1/400  0.1/400  * | 0.1/400  0.1/400  * | 0.0/400  0.1/400  *
back_bar     |  60 | 1.1/376  1.2/376  * | 1.0/381  1.1/377  * | 1.0/381  1.1/377  * | 0.8/398  1.1/385  *
back_bar     |  90 | 1.1/362  1.9/307  * | 1.6/343  1.9/311  * | 1.6/343  1.9/311  * | 1.1/383  1.6/321  *
back_bar     | 180 | 1.2/348  1.8/287  * | 1.5/298  1.2/339  X | 1.5/298  1.2/339  X | 1.2/368  1.7/296  *
back_bar     | 360 | 1.7/297  2.0/262  * | 1.4/311  2.0/266  * | 1.4/311  2.0/266  * | 1.4/354  1.6/303  *
banquet_pass |  30 | 0.4/394  1.8/369  * | 0.6/388  1.6/380  * | 0.6/388  1.6/380  * | 0.2/400  1.2/378  *
banquet_pass |  60 | 1.4/363  1.2/348  X | 1.1/372  1.2/346  * | 1.1/372  1.2/346  * | 0.5/400  1.1/377  *
banquet_pass |  90 | 1.5/352  1.4/346  X | 1.4/367  1.5/342  * | 1.4/367  1.5/342  * | 0.3/400  1.1/372  *
banquet_pass | 180 | 1.9/316  1.9/300  X | 1.8/320  1.8/310  * | 1.8/320  1.8/310  * | 0.5/394  1.4/371  *
banquet_pass | 360 | 1.9/295  2.0/309  * | 1.9/314  2.0/297  * | 1.9/314  2.0/297  * | 0.8/392  1.3/386  *
butchery     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400
butchery     |  60 | 1.1/370  1.9/324  * | 1.1/348  1.9/320  * | 1.1/348  1.9/320  * | 1.4/374  1.0/372  X
butchery     |  90 | 0.9/370  0.7/394  X | 0.9/361  0.7/394  X | 0.9/361  0.7/394  X | 1.1/360  1.1/364  X
butchery     | 180 | 0.1/400  1.2/325  * | 0.1/400  1.3/325  * | 0.1/400  1.3/325  * | 0.7/384  0.6/396  X
butchery     | 360 | 0.9/334  0.9/356  X | 0.8/351  0.9/356  * | 0.8/351  0.9/356  * | 1.3/349  0.9/356  X
chefs_table  |  30 | 0.1/400  0.0/400  X | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400
chefs_table  |  60 | 0.9/376  1.9/370  * | 0.8/386  1.9/371  * | 0.8/386  1.9/371  * | 0.4/400  0.8/388  *
chefs_table  |  90 | 0.8/395  0.8/394  X | 1.1/393  0.7/398  X | 1.1/393  0.7/398  X | 0.1/400  0.4/400  *
chefs_table  | 180 | 1.4/357  1.4/381  X | 1.2/372  1.4/382  * | 1.2/372  1.4/382  * | 0.5/398  1.1/389  *
chefs_table  | 360 | 1.6/356  1.7/350  * | 1.3/362  1.8/331  * | 1.3/362  1.8/331  * | 0.3/400  1.1/394  *
divide       |  30 | 1.1/394  0.9/400  X | 0.9/398  0.9/400  X | 0.9/398  0.9/400  X | 0.6/399  0.7/400  *
divide       |  60 | 1.7/352  1.6/357  X | 1.9/340  1.6/356  X | 1.9/340  1.6/356  X | 1.5/375  1.4/379  X
divide       |  90 | 1.6/355  1.9/330  * | 1.6/355  1.9/334  * | 1.6/355  1.9/334  * | 1.4/359  1.4/377  X
divide       | 180 | 1.8/273  2.0/265  * | 1.9/267  2.0/255  * | 1.9/267  2.0/255  * | 1.7/304  2.0/253  *
divide       | 360 | 1.6/284  2.0/258  * | 1.6/284  2.0/255  * | 1.6/284  2.0/255  * | 1.6/315  2.0/272  *
pantry       |  30 | 1.7/332  1.9/307  * | 1.4/355  1.9/289  * | 1.4/355  1.9/289  * | 0.7/393  1.1/374  *
pantry       |  60 | 1.7/307  1.9/303  * | 1.6/338  1.9/298  * | 1.6/338  1.9/298  * | 0.7/390  1.6/364  *
pantry       |  90 | 1.6/332  2.0/273  * | 1.4/362  2.0/278  * | 1.4/362  2.0/278  * | 1.5/363  1.9/318  *
pantry       | 180 | 1.8/288  2.0/242  * | 1.9/283  2.0/247  * | 1.9/283  2.0/247  * | 1.7/327  2.0/297  *
pantry       | 360 | 1.9/244  2.0/270  * | 2.0/253  2.0/261  X | 2.0/253  2.0/261  X | 1.7/326  2.0/261  *
```

`*` the layer beat its baseline, `X` lost, blank tie. dishes/ticks, mean over
20 seeds. `solo` and `handoff` columns are IDENTICAL on every row -- expected
on this suite, `solo`'s contention demotion rarely fires differently from
`handoff` here.

A cell can show `X` even when both numbers round to the same displayed value
(e.g. `back_bar/30` greedy: `0.1/400` vs `0.1/400 X`) -- the verdict compares
the full-precision per-seed means, the table only prints 1 decimal. That
particular cell is a near-starvation one (mostly 0 dishes across every method,
same as `butchery/30` and `chefs_table/30` below), so a single seed's outcome
swings the whole verdict; see `analysis/results_tables.py`'s `verdict()` and
`cell()` if this comes up again elsewhere.

**`butchery/30` and `chefs_table/30` tie at 0.0 dishes for every method** --
these two (layout, fov) cells appear to be too hard for any policy in this set
to deliver even one dish in 400 ticks, baseline or filtered. Worth a look
before trusting the aggregate win-rate too far; a cell where nobody can score
anything can't distinguish good policies from bad ones.

## 3. Cost

```
    fov-bayes-c8    deviates 32.2% of ticks     68 ms/tick
    fov-c8          deviates 23.1% of ticks     51 ms/tick
    fov-greedy-c8   deviates 22.6% of ticks     53 ms/tick
    fov-solo-c8     deviates 23.1% of ticks     50 ms/tick
```

## 4. Layouts this grid ran against

```
    back_bar      ffe9e746a6
    banquet_pass  053a1874f3
    butchery      24eecb475d
    chefs_table   226ad55e9f
    divide        496de8131f
    pantry        a4df69934e
```

## 5. Known limits and what is missing

- **No cap/decay sweep** -- unlike `RESULTS.md`, there is nothing to sweep here
  yet. `my_fov_filter.py`'s hyperparameters (`weight_progress`, `weight_seen`,
  `seen_bonus`, `budget_mult`, `stick`, `m`, `depth`) were all left at their
  `__init__` defaults for this grid. A sweep over any of these (`seen_bonus`
  in particular, currently `10`, `budget_mult`, currently `1.5`, and `stick`,
  currently `0.5`) has not been run.
- **`stick=0.5` is known to be too small in at least one diagnosed case.**
  `pantry`/`fov60`/`fov-solo` (seed 2 under the layout as of 2026-08-19) showed
  a 30+ tick ping-pong between two interchangeable STASH targets a genuine
  1-tick apart in cost -- the incumbency bonus couldn't hold either one.
  Raising `stick` to 1.5 kills that ping-pong almost everywhere but causes a
  *worse* regression elsewhere (a different seed drops to zero delivered
  dishes over a full episode), so it was not applied; 0.9 does nothing at all
  (the gap was exactly 1.0, and `stick` only wins a comparison it's strictly
  larger than). No fix has been applied to `my_fov_filter.py` for this. The
  `pantry` layout was independently redesigned on 2026-08-19/20 and its losses
  mostly disappeared as a result (see layout SHAs above vs. earlier grids in
  git/session history) -- consistent with, but not proof of, that redesign
  having removed the specific tile geometry behind the bug.
- **`fov-base`/`fov-fixed` controls not run** -- same gap `RESULTS.md` section
  6 flags for the other filter: no structural check that the wins are
  attributable to the FoV term specifically rather than some other divergence
  from baseline behavior, and no isolation of how much is the cone inference
  vs. the underlying search/budget mechanics. Also moot until `my_fov_filter.py`
  actually reads `cap`/`frozen_fov` (see the disambiguation note at the top).
- This grid reflects the state of `my_fov_filter.py`/`baselines.py` as of
  2026-08-19 (facing-aware `remaining()`, verb-keyed vs terrain-keyed timer
  dispatch, `_top_jobs` job-level candidate retention, progress-gated `seen`
  credit, per-cell budget/floor constraint, `committed_pos` anti-reversal) and
  the `back_bar`/`butchery`/`divide`/`pantry` layouts as re-synced 2026-08-20.
  No filter-code fixes were applied after this grid was submitted.
