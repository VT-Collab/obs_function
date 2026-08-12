# qmdp: full results

The execution-level QMDP layer, built to `DESIGN.md`. Everything measured,
including what does not work.

**Grid.** 6 layouts x 5 cones x 10 seeds x horizon 400, every baseline and the
layer wrapping each of them, plus two controls. **3000 episodes**, CARC job
`10970646`, a 30-task array, one task per (layout, cone).

**The comparison is PAIRED, and both sides are the same object.** Each filter is
scored against the baseline it wraps -- `qmdp` over handoff against handoff --
because the layer re-ranks its own baseline's jobs and falls back to its own
baseline's action, so that baseline is both its floor and the thing it must beat.
Every baseline DRAWS its sub-task from its own distribution, which is exactly what
the filter consumes; section 12 says why that is a correctness requirement and not
a flavour.

**Pass rule.** More dishes wins; at equal dishes, fewer ticks. Cells are averaged
over 10 seeds before comparison. An unfinished episode scores the horizon, 400.
Two dishes is a full score: `is_terminal` fires at `len(order_list) <= 1`.

**Regenerate every number here** -- none of it is typed by hand:

    python -m robot.filter.analysis.results_tables  GRID_DIR
    python -m robot.filter.analysis.report          GRID_DIR/*.jsonl
    python -m robot.filter.analysis.layout_facts
    python -m robot.filter.tests.test_qmdp

**The kitchens this grid ran in.** The layouts are experimental material and they
change; these numbers are only about these six files, and every row carries its
fingerprint:

    back_bar 76f0323586   banquet_pass 46fec8e416   butchery 57da296901
    chefs_table 47631e1490   divide c9d93b6640      pantry faeee8fb24

---

## 1. Headline

> **SUPERSEDED -- see section 2d.** Measured on `.layout` files that have since
> been edited; five of the six kitchens in this table no longer exist.


| pairing | win | tie | loss |
|---|---|---|---|
| `qmdp-greedy` vs greedy | 9 | 5 | **16** |
| `qmdp-solo` vs solo | 11 | 8 | 11 |
| `qmdp` vs handoff | 11 | 8 | 11 |
| `qmdp-bayes` vs **bayes** | **17** | 4 | 9 |

**The gate is not met.** Two pairings are exactly even, one is clearly negative,
and only `qmdp-bayes` wins -- over the strongest baseline, which is the one place
a win is worth having.

Cell counting flatters the layer, because a win can be a few ticks and a loss can
be a whole dish. Here is the same grid as total dishes delivered, summed over all
30 cells:

    baseline          with the layer      change
    greedy   17.8  ->  qmdp-greedy 15.7   -2.1
    solo     11.2  ->  qmdp-solo    9.0   -2.2
    handoff  11.2  ->  qmdp         9.0   -2.2
    bayes    30.8  ->  qmdp-bayes  33.5   +2.7

**Only `qmdp-bayes` delivers more food than the policy it wraps.** The three
ladder pairings each lose about two dishes across the grid even where they draw
level on cell count. Both views are reported because they disagree, and the
disagreement is the finding.

Cost, averaged over the grid:

    qmdp          deviates from its baseline on  6.8% of ticks    82 ms/tick
    qmdp-solo                                    6.8%             82
    qmdp-greedy                                  7.8%             93
    qmdp-bayes                                   9.9%            128
    qmdp-fixed                                   7.8%             81
    qmdp-base                                    0.0%              9   (parity)

Biggest movements:

    qmdp        divide      fov60   0.3/400 -> 1.4/374   +1.1 dishes
    qmdp        divide      fov180  0.1/400 -> 0.5/397   +0.4
    qmdp        chefs_table fov360  1.1/379 -> 0.0/400   -1.1
    qmdp        butchery    fov360  1.4/338 -> 0.1/400   -1.3
    qmdp-bayes  divide      fov60   0.7/393 -> 1.6/371   +0.9
    qmdp-bayes  divide      fov360  0.4/400 -> 1.3/392   +0.9
    qmdp-bayes  pantry      fov180  1.4/350 -> 0.8/368   -0.6

**divide is where the layer works** -- the top win for all four pairings, and the
layout where the robot holds pot, board and sink while the human holds meat, onion
and the only hatch, so every dish must be traded. **butchery fov360 and
chefs_table fov360 are where it fails**, by whole dishes.

## 2. The whole table

> **SUPERSEDED -- see section 2d.** Same stale layouts as section 1.


`dishes/ticks`, mean over 10 seeds. `*` the layer beat its baseline, `X` lost,
blank a tie.

```
layout       | fov | greedy   +qmdp    | solo     +qmdp    | handoff  +qmdp    | bayes    +qmdp
back_bar     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.1/400  0.1/400
back_bar     |  60 | 0.2/400  0.6/400  * | 0.0/400  0.4/400  * | 0.0/400  0.4/400  * | 1.2/381  1.2/375  *
back_bar     |  90 | 0.5/383  0.5/400  X | 0.0/400  0.1/400  * | 0.0/400  0.1/400  * | 1.9/305  2.0/269  *
back_bar     | 180 | 0.6/367  0.2/400  X | 0.1/400  0.1/400    | 0.1/400  0.1/400    | 1.9/276  2.0/224  *
back_bar     | 360 | 0.6/383  0.1/400  X | 0.2/400  0.0/400  X | 0.2/400  0.0/400  X | 2.0/270  2.0/214  *
banquet_pass |  30 | 0.1/400  0.0/400  X | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.4/399  0.7/400  *
banquet_pass |  60 | 0.2/400  0.0/400  X | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 1.4/391  1.3/392  X
banquet_pass |  90 | 0.5/400  0.0/400  X | 0.6/400  0.0/400  X | 0.6/400  0.0/400  X | 1.0/400  1.2/397  *
banquet_pass | 180 | 1.4/389  1.2/393  X | 1.3/398  1.2/399  X | 1.3/398  1.2/399  X | 1.9/361  1.7/369  X
banquet_pass | 360 | 1.8/371  2.0/359  * | 1.7/371  2.0/359  * | 1.7/371  2.0/359  * | 1.7/369  1.6/373  X
butchery     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400
butchery     |  60 | 1.0/390  0.8/386  X | 0.6/400  0.8/392  * | 0.6/400  0.9/392  * | 0.9/384  0.7/400  X
butchery     |  90 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400
butchery     | 180 | 0.1/400  0.0/400  X | 0.5/388  0.0/400  X | 0.5/388  0.0/400  X | 0.6/395  0.4/400  X
butchery     | 360 | 1.4/333  0.6/389  X | 1.4/338  0.0/400  X | 1.4/338  0.1/400  X | 0.7/400  1.1/393  *
chefs_table  |  30 | 0.5/400  0.0/400  X | 0.3/400  0.0/400  X | 0.3/400  0.0/400  X | 0.5/396  0.4/400  X
chefs_table  |  60 | 0.8/391  0.6/400  X | 0.5/400  0.9/399  * | 0.5/400  0.8/400  * | 1.3/379  1.5/382  *
chefs_table  |  90 | 0.9/400  0.6/381  X | 0.7/400  0.0/400  X | 0.7/400  0.0/400  X | 1.6/358  1.6/361  X
chefs_table  | 180 | 0.9/392  1.2/347  * | 0.4/400  0.0/400  X | 0.4/400  0.0/400  X | 1.5/370  1.9/337  *
chefs_table  | 360 | 1.1/371  1.4/365  * | 1.1/379  0.0/400  X | 1.1/379  0.0/400  X | 1.9/344  1.9/331  *
divide       |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400
divide       |  60 | 1.7/309  1.4/374  X | 0.3/400  1.4/374  * | 0.3/400  1.4/374  * | 0.7/393  1.6/371  *
divide       |  90 | 0.3/400  0.3/400    | 0.2/400  0.4/400  * | 0.2/400  0.4/400  * | 1.2/394  0.9/398  X
divide       | 180 | 0.0/400  0.5/400  * | 0.1/400  0.6/397  * | 0.1/400  0.5/397  * | 0.7/395  1.2/389  *
divide       | 360 | 0.0/400  0.1/400  * | 0.4/400  0.2/400  X | 0.4/400  0.2/400  X | 0.4/400  1.3/392  *
pantry       |  30 | 0.1/400  0.3/400  * | 0.2/400  0.0/400  X | 0.2/400  0.0/400  X | 0.2/400  0.7/396  *
pantry       |  60 | 0.7/379  0.6/398  X | 0.1/400  0.1/400    | 0.1/400  0.1/400    | 0.9/377  0.9/362  *
pantry       |  90 | 1.0/355  1.3/325  * | 0.2/392  0.2/388  * | 0.2/392  0.2/388  * | 1.8/306  1.8/302  *
pantry       | 180 | 0.7/392  0.8/386  * | 0.1/400  0.3/400  * | 0.1/400  0.3/400  * | 1.4/350  0.8/368  X
pantry       | 360 | 0.7/372  0.6/395  X | 0.2/400  0.3/389  * | 0.2/400  0.3/389  * | 1.0/383  1.0/370  *
```

Three things read straight off it.

**Only 2 of 30 cells are dead** -- butchery/30 and butchery/90, where no method
in the registry delivers anything. Under the earlier deterministic baselines 6
cells were dead and roughly half the grid read `0.0/400` for every policy.
Drawing the sub-task spreads the outcomes out, so nearly every cell now carries
information: the grid became more discriminative as a side effect of making it
fair. (Cells that read `0.0/400` across all four columns above, such as
divide/30, are not dead -- something outside those four columns scores there.)

**`solo` and `handoff` are near-identical** -- 11.2 dishes each and the same value
in most cells. handoff's staging only fires when nothing better than a stash is on
offer, so on this suite it rarely separates from solo.

**`greedy` beats both of them**, 17.8 dishes against 11.2. See section 5.

## 2b. Removing the candidate budgets -- this is now what `qmdp` IS

> **SUPERSEDED -- see section 2d.** Stale layouts, and the filter it describes
> has since been replaced.


Numbered `2b` rather than `3` so the `section N` cross-references in this package
keep pointing where they did.

**READ SECTIONS 1 AND 2 AS HISTORY.** They were measured with the filter that
enumerated `(v, g, a)` triples and truncated its candidates at `cell_k=6` /
`pair_k=6`. That filter no longer exists: there is now ONE class,
`core/qmdp.py:QMDPFilter`, and this section is the grid that justified replacing
the old one. The layouts are identical (fingerprints below), so the two are
comparable -- but section 1 does not describe the shipped code and section 2b
does.

**The grid in this section ran under the name `qmdp-enum`**, which was the
budgets-removed configuration before the consolidation. `analysis/report.py`'s
`NAME_FIX` maps that spelling onto `qmdp`, so the saved JSONL still reports.

**Why those were wrong.** For a STASH the baseline offers its counters in ITS OWN
order, which is distance from the ROBOT, and `cell_k=6` kept the first six. Measured
at fov 90 over 6 layouts x 3 seeds, on ticks where the committed job was a stash:

    layout        counters offered   scored   visible to human: offered / scored
    back_bar              22.8          6.0            2.20 / 0.18
    banquet_pass          33.3          6.0            2.00 / 0.00
    butchery              26.1          6.0            2.40 / 0.47
    chefs_table           30.1          6.0            4.78 / 1.11
    divide                51.2          6.0           12.13 / 1.23
    pantry                33.4          6.0            3.65 / 1.10

On back_bar the six-cell shortlist held NO counter the human could see on **55% of
stash ticks**, on banquet_pass **39%**, and banquet_pass scored 0.00 visible
counters on average. For a layer whose whole purpose is choosing a counter the human
can see, the shortlist was excluding the answer. `T=25` was the same bug in a
different place: the farthest legal target measured over 655 ticks is **28 steps**
away (banquet_pass), and 24-27 on back_bar, chefs_table and pantry, so four of six
layouts had their far counters unreachable inside the head. `T=30` covers all of them.

**Grid.** CARC job `10974948`, a 120-task array (one per layout x cone x pairing),
**2400 episodes**, horizon 400, 10 seeds, zero incomplete cells, zero tracebacks.
The layout fingerprints are IDENTICAL to the grid in section 1, so the two tables
are directly comparable rather than merely adjacent.

| pairing | win | tie | loss | section 1's `qmdp` |
|---|---|---|---|---|
| `qmdp-enum-greedy` vs greedy | 11 | 4 | 15 | 9 / 5 / 16 |
| `qmdp-enum-solo` vs solo | 12 | 8 | 10 | 11 / 8 / 11 |
| `qmdp-enum` vs handoff | 12 | 8 | 10 | 11 / 8 / 11 |
| `qmdp-enum-bayes` vs bayes | 15 | 4 | 11 | 17 / 4 / 8 |

**Cell counts barely move. Dishes do, and that is the result:**

    baseline           with the layer         change    section 1's qmdp
    greedy   17.8  ->  qmdp-enum-greedy 15.2    -2.6         -2.1
    solo     11.2  ->  qmdp-enum-solo   12.6    +1.4         -2.2
    handoff  11.2  ->  qmdp-enum        12.6    +1.4         -2.2
    bayes    30.8  ->  qmdp-enum-bayes  32.2    +1.4         +2.7

**`qmdp` and `qmdp-solo` go from -2.2 dishes to +1.4** -- a 3.6-dish swing on the
same six layouts, from removing two truncations. That is the largest movement any
change in this package's history has produced, and it is the movement the stash
measurement above predicts: solo and handoff are the pairings that stash most, so
they are the ones the six-cell shortlist hurt most.

Two results that go the other way and are not to be buried. **`qmdp-enum-bayes` is
WORSE than `qmdp-bayes`** on cells, 15/4/11 against 17/4/8, while still gaining 1.4
dishes -- bayes's ranking already moves every tick, so widening its candidate set
adds churn as well as reach. And **`greedy` is untouched at -2.6**: removing the
truncation did not help it at all, which says its loss is not a candidate-set
problem. Section 5 is about greedy and this does not change it.

**It really does use the reach.** On ticks where the chosen plan is a stash, which
counter it picks and whether that is the nearest -- the nearest being exactly what
handoff picks and what `cell_k=6` forced:

    layout        stash ticks   counters scored   picks nearest   mean rank of pick
    back_bar             138           22.5            3%             5.1
    banquet_pass         143           33.4            2%             8.7
    divide                20           53.6           10%            18.9
    pantry               128           32.6           28%             6.9

It picks a counter OTHER than the nearest on 72-98% of stash ticks, at mean rank
5-19. On divide it routinely takes the 19th-nearest of 53 -- a cell the old shortlist
could not score at any price.

**Cost.** 277 ms/tick against `qmdp`'s 82, for ~91 rollouts a tick instead of ~19.

```
layout       | fov | greedy   +enum    | solo     +enum    | handoff  +enum    | bayes    +enum   
back_bar     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.1/400  0.1/400   
back_bar     |  60 | 0.2/400  0.6/400  * | 0.0/400  0.3/400  * | 0.0/400  0.3/400  * | 1.2/381  1.2/366  *
back_bar     |  90 | 0.5/383  0.5/400  X | 0.0/400  0.1/400  * | 0.0/400  0.1/400  * | 1.9/305  2.0/266  *
back_bar     | 180 | 0.6/367  0.2/400  X | 0.1/400  0.1/400    | 0.1/400  0.1/400    | 1.9/276  2.0/234  *
back_bar     | 360 | 0.6/383  0.1/400  X | 0.2/400  0.0/400  X | 0.2/400  0.0/400  X | 2.0/270  2.0/223  *
banquet_pass |  30 | 0.1/400  0.0/400  X | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.4/399  0.5/400  *
banquet_pass |  60 | 0.2/400  0.0/400  X | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 1.4/391  1.2/391  X
banquet_pass |  90 | 0.5/400  0.0/400  X | 0.6/400  0.0/400  X | 0.6/400  0.0/400  X | 1.0/400  1.2/398  *
banquet_pass | 180 | 1.4/389  1.2/393  X | 1.3/398  1.2/399  X | 1.3/398  1.2/399  X | 1.9/361  1.6/373  X
banquet_pass | 360 | 1.8/371  2.0/360  * | 1.7/371  2.0/360  * | 1.7/371  2.0/360  * | 1.7/369  1.4/382  X
butchery     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400   
butchery     |  60 | 1.0/390  0.8/388  X | 0.6/400  1.0/392  * | 0.6/400  1.0/392  * | 0.9/384  0.7/400  X
butchery     |  90 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400   
butchery     | 180 | 0.1/400  0.0/400  X | 0.5/388  0.0/400  X | 0.5/388  0.0/400  X | 0.6/395  0.4/400  X
butchery     | 360 | 1.4/333  0.1/400  X | 1.4/338  0.3/400  X | 1.4/338  0.3/400  X | 0.7/400  1.1/393  *
chefs_table  |  30 | 0.5/400  0.2/400  X | 0.3/400  0.1/400  X | 0.3/400  0.1/400  X | 0.5/396  0.4/400  X
chefs_table  |  60 | 0.8/391  0.4/400  X | 0.5/400  0.5/400    | 0.5/400  0.5/400    | 1.3/379  1.3/382  X
chefs_table  |  90 | 0.9/400  1.1/380  * | 0.7/400  1.2/364  * | 0.7/400  1.2/364  * | 1.6/358  1.7/350  *
chefs_table  | 180 | 0.9/392  1.0/354  * | 0.4/400  0.3/389  X | 0.4/400  0.3/389  X | 1.5/370  2.0/322  *
chefs_table  | 360 | 1.1/371  1.3/371  * | 1.1/379  0.2/393  X | 1.1/379  0.2/393  X | 1.9/344  1.9/342  *
divide       |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400   
divide       |  60 | 1.7/309  1.4/380  X | 0.3/400  1.7/364  * | 0.3/400  1.7/364  * | 0.7/393  1.4/387  *
divide       |  90 | 0.3/400  0.2/400  X | 0.2/400  0.3/400  * | 0.2/400  0.3/400  * | 1.2/394  0.7/398  X
divide       | 180 | 0.0/400  0.3/400  * | 0.1/400  0.6/397  * | 0.1/400  0.6/397  * | 0.7/395  1.1/392  *
divide       | 360 | 0.0/400  0.1/400  * | 0.4/400  0.2/400  X | 0.4/400  0.2/400  X | 0.4/400  1.3/393  *
pantry       |  30 | 0.1/400  0.3/400  * | 0.2/400  0.0/400  X | 0.2/400  0.0/400  X | 0.2/400  0.7/390  *
pantry       |  60 | 0.7/379  0.8/394  * | 0.1/400  0.7/400  * | 0.1/400  0.7/400  * | 0.9/377  0.7/379  X
pantry       |  90 | 1.0/355  1.2/348  * | 0.2/392  0.4/400  * | 0.2/392  0.4/400  * | 1.8/306  1.9/305  *
pantry       | 180 | 0.7/392  1.0/358  * | 0.1/400  0.7/371  * | 0.1/400  0.7/371  * | 1.4/350  0.9/360  X
pantry       | 360 | 0.7/372  0.4/400  X | 0.2/400  0.7/361  * | 0.2/400  0.7/361  * | 1.0/383  0.8/374  X
```

`*` the layer beat its baseline, `X` lost, blank a tie. dishes/ticks, mean over 10
seeds. Regenerate with:

    python -m robot.filter.analysis.results_tables GRID_DIR --family enum

**WHAT THIS SECTION DOES NOT HAVE, and it matters.** No parity control and no
theta-blind control: `qmdp-enum-base` and `qmdp-enum-fixed` were not run. The parity
control is what caught the filter anchoring its fallback on `argmax pi` instead of
the baseline's realised draw (section 4), so until it runs, part of the +1.4 could be
a degrade-to-baseline artifact rather than better play. Do not quote the +1.4 as
settled without it.

## 2c. The current filter, measured (CARC job 10987487)

> **SUPERSEDED -- see section 2d.** This grid ran against CARC's copies of the
> layouts, which were five edits behind the working tree. The numbers here
> describe kitchens that do not exist.


**THIS IS THE ONLY SECTION THAT DESCRIBES THE SHIPPED CODE.** Sections 1, 2 and 2b
were measured with filters that no longer exist -- the enumerate-(v, g, a) one and
its budgets-removed variant. Read them as history. The layouts are identical
(fingerprints below match theirs), so the numbers are comparable; the code is not.

**Grid.** 6 layouts x 5 cones x 4 pairings x 10 seeds, horizon 400. 2392 of 2400
episodes; the six short cells are banquet_pass fov60 and fov90 under handoff, qmdp
and qmdp-greedy, at 8-9 seeds instead of 10. Zero tracebacks. Regenerate with:

    python -m robot.filter.analysis.results_tables GRID_DIR

    WARNING, and it nearly corrupted this table. The output directory already held
    a 3000-row grid from 2026-08-10 whose method names (`handoff-stoch`,
    `bayes-post`) map onto the current ones through report.NAME_FIX, so reading the
    directory pooled two grids under identical keys -- half the rows from code that
    ran at 0.4 ms/tick against this code's 900. Moved to `_prior_grid_2026-08-10/`.
    Glob `qmdp*.jsonl`, not `*.jsonl`.

```
layout       | fov | greedy   +qmdp      | solo     +qmdp      | handoff  +qmdp      | bayes    +qmdp   
back_bar     |  30 | 0.0/400  0.8/396  * | 0.0/400  0.8/397  * | 0.0/400  0.8/397  * | 0.1/400  1.5/391  *
back_bar     |  60 | 0.2/400  2.0/308  * | 0.0/400  2.0/313  * | 0.0/400  2.0/313  * | 1.2/381  1.5/362  *
back_bar     |  90 | 0.5/383  2.0/293  * | 0.0/400  2.0/297  * | 0.0/400  2.0/297  * | 1.9/305  2.0/256  *
back_bar     | 180 | 0.6/367  2.0/244  * | 0.1/400  2.0/232  * | 0.1/400  2.0/232  * | 1.9/276  2.0/229  *
back_bar     | 360 | 0.6/383  2.0/236  * | 0.2/400  2.0/230  * | 0.2/400  2.0/230  * | 2.0/270  2.0/232  *
banquet_pass |  30 | 0.1/400  2.0/380  * | 0.0/400  1.9/382  * | 0.0/400  1.9/382  * | 0.4/399  1.0/400  *
banquet_pass |  60 | 0.2/400  2.0/396  * | 0.0/400  1.8/397  * | 0.0/400  2.0/396  * | 1.4/391  1.5/390  *
banquet_pass |  90 | 0.5/400  1.0/400  * | 0.6/400  1.0/400  * | 0.6/400  1.0/400  * | 1.0/400  1.5/383  *
banquet_pass | 180 | 1.4/389  1.1/395  X | 1.3/398  1.2/390  X | 1.3/398  1.2/390  X | 1.9/361  1.0/400  X
banquet_pass | 360 | 1.8/371  1.0/400  X | 1.7/371  1.0/400  X | 1.7/371  1.0/400  X | 1.7/369  1.5/372  X
butchery     |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400   
butchery     |  60 | 1.0/390  1.8/288  * | 0.6/400  1.6/291  * | 0.6/400  1.6/291  * | 0.9/384  1.9/327  *
butchery     |  90 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400   
butchery     | 180 | 0.1/400  0.8/386  * | 0.5/388  0.6/390  * | 0.5/388  0.6/390  * | 0.6/395  0.8/384  *
butchery     | 360 | 1.4/333  2.0/286  * | 1.4/338  1.9/281  * | 1.4/338  1.9/281  * | 0.7/400  2.0/299  *
chefs_table  |  30 | 0.5/400  1.0/400  * | 0.3/400  1.0/400  * | 0.3/400  1.0/400  * | 0.5/396  0.4/400  X
chefs_table  |  60 | 0.8/391  1.3/366  * | 0.5/400  1.3/366  * | 0.5/400  1.3/366  * | 1.3/379  1.8/361  *
chefs_table  |  90 | 0.9/400  1.9/355  * | 0.7/400  1.8/359  * | 0.7/400  1.8/359  * | 1.6/358  2.0/324  *
chefs_table  | 180 | 0.9/392  1.9/348  * | 0.4/400  1.9/369  * | 0.4/400  1.9/369  * | 1.5/370  1.9/324  *
chefs_table  | 360 | 1.1/371  2.0/328  * | 1.1/379  2.0/329  * | 1.1/379  2.0/329  * | 1.9/344  1.5/354  X
divide       |  30 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.1/400  *
divide       |  60 | 1.7/309  2.0/272  * | 0.3/400  2.0/272  * | 0.3/400  2.0/272  * | 0.7/393  2.0/294  *
divide       |  90 | 0.3/400  2.0/247  * | 0.2/400  2.0/245  * | 0.2/400  2.0/245  * | 1.2/394  1.9/292  *
divide       | 180 | 0.0/400  2.0/237  * | 0.1/400  2.0/237  * | 0.1/400  2.0/237  * | 0.7/395  2.0/301  *
divide       | 360 | 0.0/400  2.0/234  * | 0.4/400  2.0/235  * | 0.4/400  2.0/235  * | 0.4/400  2.0/290  *
pantry       |  30 | 0.1/400  2.0/274  * | 0.2/400  2.0/289  * | 0.2/400  2.0/289  * | 0.2/400  0.7/400  *
pantry       |  60 | 0.7/379  0.0/400  X | 0.1/400  0.0/400  X | 0.1/400  0.0/400  X | 0.9/377  1.2/345  *
pantry       |  90 | 1.0/355  2.0/297  * | 0.2/392  2.0/296  * | 0.2/392  2.0/296  * | 1.8/306  1.9/283  *
pantry       | 180 | 0.7/392  0.0/400  X | 0.1/400  0.2/392  * | 0.1/400  0.2/392  * | 1.4/350  1.1/355  X
pantry       | 360 | 0.7/372  0.9/380  * | 0.2/400  0.7/395  * | 0.2/400  0.7/395  * | 1.0/383  0.2/385  X
```

`*` the layer beat its baseline, `X` lost, blank a tie. dishes/ticks, mean over
seeds.

### summary

| pairing | win | tie | loss | dishes: baseline -> layer | change |
|---|---|---|---|---|---|
| `qmdp-greedy` vs greedy | 23 | 3 | 4 | 17.8 -> 41.5 | +23.7 |
| `qmdp-solo` vs solo | 24 | 3 | 3 | 11.2 -> 40.7 | +29.5 |
| `qmdp` vs handoff | 24 | 3 | 3 | 11.2 -> 40.9 | +29.7 |
| `qmdp-bayes` vs bayes | 22 | 2 | 6 | 30.8 -> 40.9 | +10.1 |

Cell counts and dishes agree for once, and they did not before: in section 1 the
ladder pairings were level on cells (11/8/11) and DOWN 2.2 dishes. Both views now
point the same way and by a wide margin.

**ALL FOUR PAIRINGS LAND ON THE SAME NUMBER -- 41.5, 40.7, 40.9, 40.9 dishes --
whichever baseline they wrap.** That is the most important line in this section and
it is not a good sign on its own. Through every earlier grid the pairings stayed
spread, because the layer re-ranked its baseline's jobs and inherited its floor --
and their BASELINES are still spread here, 17.8 / 11.2 / 11.2 / 30.8, the same
numbers as section 1. Landing on one number says the baseline now supplies only the
candidate set and the search decides everything else. That is either the layer
working as designed or the layer having stopped listening to its baseline, and THIS
GRID CANNOT TELL THOSE APART, because:

- **there is no parity control in it.** `qmdp-base` and `qmdp-fixed` were not in the
  array. `qmdp-base` collapses the mask to the baseline's own realised cell and must
  therefore tie its baseline in every cell; it is exactly the check that would
  distinguish "correctly better" from "ignoring its baseline", and it is the check
  that caught the argmax-pi anchor bug (section 4). Run it before quoting +29.7.
- **the deviation rate is 37-42% of ticks**, against 6-10% in section 1. The layer
  now overrides its baseline on two ticks in five.

Cells that were dead for every policy in section 1 now deliver: back_bar 60-360
goes `0.0/400 -> 2.0/230-313`, divide 90-360 goes `0.0-0.4/400 -> 2.0/234-247`,
pantry 30 goes `0.2/400 -> 2.0/274`. The losses cluster on banquet_pass 180/360,
where all four pairings lose.

**Cost.** 476-999 ms/tick, against section 1's 82. `tail_ticks` is 91% of it: a
stash tick scores 360 plans (24 counters x 5 first actions x 3 arrival ticks) and
calls the tail once each, where a fetch tick calls it 45 times. Two exact
speedups are identified and NOT yet applied -- prune on `avail` (which spans 12-223
while the tail spans 108-124, so with tail >= 0 most plans can never win) and dedupe
the tail across the 360 plans' 120 distinct (cell, press tick) pairs.

**What changed since section 2b**, all of it unmeasured until this grid:

| change | where |
|---|---|
| union action mask over the top-m jobs' cells; INTERACT legal at any of them, at every step | `core/qmdp.py` |
| every (cell, arrival tick) enumerated to depth 30, so the farthest counter (28 steps) is reachable | `core/qmdp.py` |
| a stash costs until the PARTNER HOLDS IT: first look at or after the press, then their walk | `core/qmdp.py` |
| one `no_handoff` penalty for both failure modes -- never looked at, or cannot reach | `core/qmdp.py` |
| a finished `dish` short-circuits the whole leg chain to `deliver` | `core/value_tail.py` |
| `steak_dish` and `garnish_dish` short-circuit the streams they contain | `core/value_tail.py` |
| remaining orders priced by inventory in dish-equivalents rather than a flat constant each | `core/value_tail.py` |
| `FETCH_W` tie-break so work sitting in someone else's slack is not free | `core/value_tail.py` |

The last four are in the TAIL. The tail is shared by every FILTER but by no
baseline -- greedy, solo, handoff and bayes never call it -- so the baselines are a
genuine control on the rest of the stack, and they reproduce section 1 to the tenth
of a dish: 17.8, 11.2, 11.2, 30.8 in both. The human model, the layouts and the
harness are therefore unchanged, and the 23.7-to-29.7 dish delta is attributable to
the filter rather than to something that moved underneath it.

That is the strongest evidence in this section, and it is worth more than the
headline: it rules out the whole class of explanations where the grid moved because
the environment did. It does NOT rule out the layer ignoring its baseline, which is
what the parity control above is for.

## 2d. FINAL: the two human models, on the layouts that currently exist

**EVERYTHING ABOVE THIS SECTION IS SUPERSEDED.** Sections 1, 2, 2b and 2c were all
measured on `.layout` files that have since been edited, and CARC was additionally
holding pre-edit copies of five of them -- `robot/` and `common/` were being synced,
`layout/` never was. The fingerprints make it unambiguous:

    layout         section 2c    current
    back_bar       76f0323586    76f0323586   same
    banquet_pass   46fec8e416    252a7bcc0d   DIFFERENT
    butchery       57da296901    a86a1d541b   DIFFERENT
    chefs_table    47631e1490    0aec924651   DIFFERENT
    divide         c9d93b6640    57e9e2bdf5   DIFFERENT
    pantry         faeee8fb24    707da824f6   DIFFERENT

Five of six kitchens changed, so none of the earlier dish counts describe anything
that exists. This is the third time this package has been bitten by exactly that,
and `analysis/layout_facts.py`'s docstring says why it is worse than being wrong:
the numbers were correct when written and nothing about them looks stale.

**Grid.** CARC jobs `11001542` (v1) and `11001543` (v2). 6 layouts x 5 cones x 4
pairings x 10 seeds x horizon 400 = **2400 episodes per arm**. Every cell has exactly
10 seeds, every row's `layout_sha` matches the current file, zero tracebacks. The two
arms differ in ONE thing:

    v1   the shipped human           robot.filter.harness.evaluate
    v2   composites saturate         robot.filter.dev.eval_human_v2
         (a garnish_dish is a garnish AND a plate, so it stops the garnish chain
          AND the plate chain -- common/tasks_containment.py, EXPERIMENTAL)

Same robot code, same seeds, same layouts. The baselines in each arm are the control
on everything else.

### v1 -- the shipped human model

```
layout       | fov | greedy   +qmdp    | solo     +qmdp    | handoff  +qmdp    | bayes    +qmdp   
back_bar     |  30 | 0.0/400  0.8/396  * | 0.0/400  0.8/397  * | 0.0/400  0.8/397  * | 0.1/400  1.5/391  *
back_bar     |  60 | 0.2/400  2.0/308  * | 0.0/400  2.0/313  * | 0.0/400  2.0/313  * | 1.2/381  1.5/362  *
back_bar     |  90 | 0.5/383  2.0/293  * | 0.0/400  2.0/297  * | 0.0/400  2.0/297  * | 1.9/305  2.0/256  *
back_bar     | 180 | 0.6/367  2.0/244  * | 0.1/400  2.0/232  * | 0.1/400  2.0/232  * | 1.9/276  2.0/229  *
back_bar     | 360 | 0.6/383  2.0/236  * | 0.2/400  2.0/230  * | 0.2/400  2.0/230  * | 2.0/270  2.0/232  *
banquet_pass |  30 | 0.2/400  2.0/337  * | 0.0/400  2.0/337  * | 0.0/400  2.0/337  * | 0.7/398  2.0/348  *
banquet_pass |  60 | 0.2/400  2.0/332  * | 0.0/400  2.0/333  * | 0.0/400  2.0/333  * | 1.5/358  2.0/345  *
banquet_pass |  90 | 0.2/400  2.0/330  * | 0.0/400  2.0/343  * | 0.0/400  2.0/343  * | 1.5/377  2.0/343  *
banquet_pass | 180 | 0.2/400  2.0/286  * | 0.0/400  2.0/286  * | 0.0/400  2.0/286  * | 1.9/338  2.0/299  *
banquet_pass | 360 | 0.2/400  2.0/302  * | 0.1/400  2.0/297  * | 0.1/400  2.0/297  * | 1.8/343  2.0/310  *
butchery     |  30 | 0.0/400  0.3/400  * | 0.0/400  0.3/400  * | 0.0/400  0.3/400  * | 0.2/400  0.0/400  X
butchery     |  60 | 0.0/400  1.2/304  * | 0.0/400  1.2/304  * | 0.0/400  1.2/304  * | 2.0/310  2.0/264  *
butchery     |  90 | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 0.0/400  0.0/400    | 1.2/323  0.8/373  X
butchery     | 180 | 0.0/400  1.1/338  * | 0.0/400  1.1/338  * | 0.0/400  1.1/338  * | 1.8/311  1.2/338  X
butchery     | 360 | 0.0/400  2.0/260  * | 0.0/400  2.0/260  * | 0.0/400  2.0/260  * | 1.9/274  2.0/265  *
chefs_table  |  30 | 0.3/400  1.0/400  * | 0.2/400  1.1/396  * | 0.2/400  1.1/396  * | 0.9/398  0.9/394  *
chefs_table  |  60 | 0.4/400  1.2/393  * | 0.4/400  1.1/396  * | 0.4/400  1.1/396  * | 1.2/363  1.9/347  *
chefs_table  |  90 | 0.4/394  1.4/387  * | 0.5/400  1.3/389  * | 0.5/400  1.3/389  * | 1.2/352  1.7/369  *
chefs_table  | 180 | 0.5/400  2.0/254  * | 0.6/400  2.0/254  * | 0.6/400  2.0/254  * | 1.0/360  1.4/360  *
chefs_table  | 360 | 0.3/400  2.0/248  * | 0.5/400  2.0/248  * | 0.5/400  2.0/248  * | 0.7/382  1.2/376  *
divide       |  30 | 0.1/400  2.0/313  * | 0.1/400  2.0/313  * | 0.1/400  2.0/313  * | 1.2/371  2.0/296  *
divide       |  60 | 0.1/400  2.0/310  * | 0.1/400  2.0/310  * | 0.1/400  2.0/310  * | 1.8/332  1.9/286  *
divide       |  90 | 0.0/400  1.9/283  * | 0.0/400  1.9/276  * | 0.0/400  1.9/276  * | 1.9/280  2.0/258  *
divide       | 180 | 0.0/400  2.0/211  * | 0.0/400  2.0/210  * | 0.0/400  2.0/210  * | 2.0/242  2.0/198  *
divide       | 360 | 0.0/400  2.0/233  * | 0.0/400  2.0/232  * | 0.0/400  2.0/232  * | 2.0/235  2.0/224  *
pantry       |  30 | 0.4/389  1.8/318  * | 0.2/400  1.9/302  * | 0.2/400  1.9/302  * | 0.4/391  0.9/370  *
pantry       |  60 | 0.4/400  2.0/292  * | 0.1/400  2.0/289  * | 0.1/400  2.0/289  * | 1.2/371  2.0/305  *
pantry       |  90 | 0.6/366  0.9/400  * | 0.4/386  0.9/400  * | 0.4/386  0.9/400  * | 1.6/340  1.9/260  *
pantry       | 180 | 0.5/387  2.0/289  * | 0.0/400  2.0/289  * | 0.0/400  2.0/289  * | 1.9/278  2.0/241  *
pantry       | 360 | 0.6/384  2.0/232  * | 0.2/400  2.0/232  * | 0.2/400  2.0/232  * | 1.8/293  2.0/248  *
```

| pairing | win | tie | loss | dishes: baseline -> layer | change |
|---|---|---|---|---|---|
| `qmdp-greedy` vs greedy | 29 | 1 | 0 | 7.5 -> 49.6 | +42.1 |
| `qmdp-solo` vs solo | 29 | 1 | 0 | 3.7 -> 49.6 | +45.9 |
| `qmdp` vs handoff | 29 | 1 | 0 | 3.7 -> 49.6 | +45.9 |
| `qmdp-bayes` vs bayes | 27 | 0 | 3 | 42.4 -> 50.8 | +8.4 |

```
    qmdp          deviates 51.7% of ticks   1468 ms/tick
    qmdp-bayes    deviates 48.5% of ticks    697 ms/tick
    qmdp-greedy   deviates 51.5% of ticks   1211 ms/tick
    qmdp-solo     deviates 51.7% of ticks   1182 ms/tick
```

### v2 -- composites count toward saturation (experimental)

```
layout       | fov | greedy   +qmdp    | solo     +qmdp    | handoff  +qmdp    | bayes    +qmdp   
back_bar     |  30 | 0.0/400  0.8/396  * | 0.0/400  0.8/396  * | 0.0/400  0.8/396  * | 0.2/400  1.3/398  *
back_bar     |  60 | 0.2/400  2.0/308  * | 0.0/400  2.0/314  * | 0.0/400  2.0/314  * | 1.2/381  1.5/362  *
back_bar     |  90 | 0.5/383  2.0/280  * | 0.0/400  2.0/285  * | 0.0/400  2.0/285  * | 1.8/312  2.0/256  *
back_bar     | 180 | 0.6/367  2.0/243  * | 0.1/400  2.0/232  * | 0.1/400  2.0/232  * | 2.0/278  2.0/229  *
back_bar     | 360 | 0.6/383  2.0/235  * | 0.2/400  2.0/230  * | 0.2/400  2.0/230  * | 2.0/268  2.0/230  *
banquet_pass |  30 | 0.2/400  2.0/337  * | 0.0/400  2.0/337  * | 0.0/400  2.0/337  * | 0.8/398  2.0/348  *
banquet_pass |  60 | 0.2/400  2.0/332  * | 0.0/400  2.0/333  * | 0.0/400  2.0/333  * | 1.5/358  2.0/345  *
banquet_pass |  90 | 0.2/400  2.0/330  * | 0.0/400  2.0/343  * | 0.0/400  2.0/343  * | 1.6/376  2.0/343  *
banquet_pass | 180 | 0.2/400  2.0/286  * | 0.0/400  2.0/286  * | 0.0/400  2.0/286  * | 1.9/338  2.0/299  *
banquet_pass | 360 | 0.2/400  2.0/302  * | 0.1/400  2.0/297  * | 0.1/400  2.0/297  * | 1.8/343  2.0/310  *
butchery     |  30 | 0.0/400  0.3/400  * | 0.0/400  0.3/400  * | 0.0/400  0.3/400  * | 0.2/400  0.0/400  X
butchery     |  60 | 0.0/400  1.2/300  * | 0.0/400  1.2/302  * | 0.0/400  1.2/302  * | 2.0/316  2.0/264  *
butchery     |  90 | 0.0/400  2.0/275  * | 0.0/400  2.0/275  * | 0.0/400  2.0/275  * | 2.0/296  2.0/283  *
butchery     | 180 | 0.0/400  2.0/244  * | 0.0/400  2.0/244  * | 0.0/400  2.0/244  * | 2.0/266  2.0/267  X
butchery     | 360 | 0.0/400  2.0/252  * | 0.0/400  2.0/252  * | 0.0/400  2.0/252  * | 1.9/276  2.0/254  *
chefs_table  |  30 | 0.3/400  1.1/397  * | 0.2/400  1.2/394  * | 0.2/400  1.2/394  * | 0.9/398  1.2/394  *
chefs_table  |  60 | 0.4/400  2.0/346  * | 0.4/400  2.0/343  * | 0.4/400  2.0/343  * | 1.3/362  2.0/319  *
chefs_table  |  90 | 0.3/400  1.3/385  * | 0.7/396  1.4/379  * | 0.7/396  1.4/379  * | 1.2/367  1.9/348  *
chefs_table  | 180 | 1.1/390  2.0/254  * | 1.3/390  2.0/254  * | 1.3/390  2.0/254  * | 1.4/344  2.0/320  *
chefs_table  | 360 | 1.0/390  2.0/248  * | 1.3/385  2.0/248  * | 1.3/385  2.0/248  * | 2.0/308  2.0/332  X
divide       |  30 | 0.1/400  2.0/313  * | 0.1/400  2.0/313  * | 0.1/400  2.0/313  * | 1.2/371  2.0/296  *
divide       |  60 | 0.1/400  2.0/310  * | 0.1/400  2.0/310  * | 0.1/400  2.0/310  * | 1.9/332  2.0/279  *
divide       |  90 | 0.0/400  2.0/262  * | 0.0/400  2.0/262  * | 0.0/400  2.0/262  * | 2.0/274  2.0/262  *
divide       | 180 | 0.0/400  2.0/212  * | 0.0/400  2.0/210  * | 0.0/400  2.0/210  * | 2.0/247  2.0/198  *
divide       | 360 | 0.0/400  2.0/232  * | 0.0/400  2.0/231  * | 0.0/400  2.0/231  * | 2.0/232  2.0/240  X
pantry       |  30 | 0.4/389  1.8/318  * | 0.2/400  1.9/302  * | 0.2/400  1.9/302  * | 0.4/391  0.9/370  *
pantry       |  60 | 0.4/400  2.0/292  * | 0.1/400  2.0/289  * | 0.1/400  2.0/289  * | 1.2/371  2.0/305  *
pantry       |  90 | 0.6/366  1.1/388  * | 0.4/386  1.1/388  * | 0.4/386  1.1/388  * | 1.7/338  2.0/249  *
pantry       | 180 | 0.5/387  2.0/273  * | 0.0/400  2.0/273  * | 0.0/400  2.0/273  * | 1.9/268  2.0/241  *
pantry       | 360 | 0.7/369  2.0/232  * | 0.3/387  2.0/232  * | 0.3/387  2.0/232  * | 1.8/281  2.0/242  *
```

| pairing | win | tie | loss | dishes: baseline -> layer | change |
|---|---|---|---|---|---|
| `qmdp-greedy` vs greedy | 30 | 0 | 0 | 8.8 -> 53.6 | +44.8 |
| `qmdp-solo` vs solo | 30 | 0 | 0 | 5.5 -> 53.9 | +48.4 |
| `qmdp` vs handoff | 30 | 0 | 0 | 5.5 -> 53.9 | +48.4 |
| `qmdp-bayes` vs bayes | 26 | 0 | 4 | 45.8 -> 54.8 | +9.0 |

```
    qmdp          deviates 52.3% of ticks   1201 ms/tick
    qmdp-bayes    deviates 49.2% of ticks    723 ms/tick
    qmdp-greedy   deviates 52.0% of ticks   1347 ms/tick
    qmdp-solo     deviates 52.3% of ticks   1268 ms/tick
```

### What the pair says

**The human patch does not explain the layer's gain, and slightly helps it.** The
qmdp deltas are +42.1 / +45.9 / +45.9 / +8.4 under v1 and +44.8 / +48.4 / +48.4 /
+9.0 under v2. If the layer had been exploiting a partner that over-produces, fixing
the partner would have shrunk the gain; it grew by about 2.5 dishes. That was the
hypothesis this experiment existed to kill, and it is dead.

**The patch helps BOTH agents, which is why it is not a confound.** Baselines move
7.5 -> 8.8 (greedy) and 3.7 -> 5.5 (solo/handoff) and 42.4 -> 45.8 (bayes). A human
that stops fetching onions it already has is a better partner for anything.

**The three ladder pairings are now 29/1/0 and 30/0/0.** Near-total, and they land on
the same number (49.6 under v1, 53.6-53.9 under v2) whichever baseline they wrap while
their baselines stay far apart (7.5 / 3.7 / 3.7). That is the same pattern section 2c
flagged and it still has the same two readings -- the search deciding everything, or
the search ignoring its baseline -- and THERE IS STILL NO PARITY CONTROL IN THIS GRID
to tell them apart. `qmdp-base` must tie its baseline in every cell; it was not run.
Do not quote +45.9 as settled without it.

**bayes is the one pairing that loses cells**, 3 under v1 and 4 under v2, while still
gaining ~8-9 dishes. It is the strong baseline, and at a wide cone it already has the
information the FOV layer exists to supply -- chefs_table fov360 is a measured
example: `bayes` 1.9/344t against `qmdp-bayes` 1.5/354t under the old layouts, and the
robot pays ~900 ms/tick and a ~40% deviation rate to search for it.

**Cost** is in the two blocks above, 0.5-1.0 s/tick against section 1's 82 ms.
`tail_ticks` is 91% of a stash tick -- 360 plans, one tail call each -- and two exact
speedups are identified and unapplied: prune on `avail` (it spans 12-223 while the
tail spans 108-124, so with tail >= 0 most plans cannot win) and dedupe the tail
across the 360 plans' 120 distinct (cell, press tick) pairs.

**Regenerate:**

    python -m robot.filter.analysis.results_tables /scratch1/$USER/qmdp_final_v1
    python -m robot.filter.analysis.results_tables /scratch1/$USER/qmdp_final_v2

Glob `qmdp*.jsonl`, never `*.jsonl`: an older grid once sat in the same directory
under names that `report.NAME_FIX` maps onto the current ones, and reading the
directory pooled two code versions under identical keys.

## 3. The fact the whole design rests on

Regenerate with `python -m robot.filter.analysis.layout_facts`; do not read it
off the grids by eye.

```
layout        robot     human      can assemble alone
back_bar      .B.MOD.   PBW.O.S    neither
banquet_pass  ..WMO..   PBWMODS    human
butchery      PBWMOD.   .BWMODS    robot
chefs_table   .BWM.D.   PBW.ODS    neither
divide        PBW..D.   .B.MO.S    neither
pantry        .B.MOD.   PBW.O.S    neither

P pot  B board  W sink  M meat  O onion  D plate  S serve;  '.' = cannot reach
```

**The robot can reach a serve hatch on NO layout**, so no dish is deliverable by
one agent anywhere and every dish crosses the divide through a counter both can
stand at. The stash cell is not a nicety, it is the only channel -- and a value
function that cannot express a handoff cannot price anything here. `value_tail.py`
is built around exactly this.

Assembling alone is rarer still: only butchery's robot and banquet_pass's human
hold all six of pot, board, sink, meat, onion and plate.

## 4. The controls

**PARITY (`qmdp-base`) -- IDENTICAL IN 30 OF 30 CELLS**, with a dish total exactly
equal to its baseline's, 11.2. The search is collapsed to the baseline's own
candidate while the real rollout machinery still runs, so this is the
degrade-to-baseline guarantee measured end to end at horizon 400.

The previous grid had one break, 29/30. The cause was found and fixed: the layer
anchored candidate 0 on `argmax pi` rather than on the baseline's REALISED draw.
`pi` is a distribution and the pick is a sample from it, so the two diverge
routinely once the baseline draws -- and the "baseline's own action" the tie-break
falls back to was not the baseline's action. DESIGN.md §3.0b. Both the 24-cell
unit check and this 30-cell grid check pass now.

One caveat survives, a real defect rather than a rounding detail: at `pair_k=1`
the must-keep step computes `pairs[:max(1, pair_k - 1)] + [must]`, which yields
TWO pairs, not one. The collapse is therefore not structurally airtight and this
parity is empirical. It fires on ~0.9% of ticks. **Not fixed** -- the fix is
`pairs[:pair_k - 1]` or special-casing `pair_k == 1`, and it was deliberately left
alone so that the code and this grid describe the same program. DESIGN.md §5(i).

**THETA-BLIND (`qmdp-fixed`, cone frozen at 90).** Same rollouts, same tail, same
guards; the only difference is whether the cone posterior's weights are used.

    inferring theta   better 9   identical 13   worse 8
    dishes            qmdp 9.0   vs   qmdp-fixed 9.4

**This is a wash, and on total dishes the frozen cone is marginally AHEAD.** The
per-cell count leans very slightly to inference, the food count leans the other
way, and neither margin supports a claim. An earlier grid read 4/24/2 in favour of
inference, but that grid had different baselines underneath it and half its cells
could not discriminate, so the two are not comparable.

That is the honest state of the research question: **the layer's one solid result
is over bayes, and this control says the cone posterior is not what produces it.**

## 5. Why greedy beats solo and handoff

Not a filter result, but it sets the bar the filter has to clear, and it turned up
while verifying a docstring rather than while looking for it.

`solo` is `greedy` plus one term in the sort key: any candidate cell the human's
shortest path reaches sooner is demoted **within its tier**. The description these
docstrings carried for a long time -- "gives up a station in the dividing wall" --
was wrong in a way that mattered:

- the penalty applies to EVERY candidate the ladder emits, not just stations, and
  about **88% of firings land on `stash` targets**;
- a stash is not a contention. `tasks.py` says so directly: every counter the human
  can reach the robot can reach too, so an ordinary stash IS the handoff. The term
  penalises exactly the cells that are the team's only channel, on the grounds that
  the human is nearer them;
- same-state head-to-head over 18 episodes: 72 ticks diverge, 46 are genuine job
  cession, and **26 of the 26 remaining are stash-vs-stash, where greedy's counter
  is reachable from BOTH rooms and solo's only from its own**;
- 54 matched episodes: greedy 289 drops on divide counters and 35 delivered; solo
  237 and 22.

It is conditional rather than absolute: the demotion is `+1` in a key that sorts
after `tier`, so a contested sub-task alone in its tier is still taken.

## 6. WITHDRAWN: the "can the robot assemble alone" predictor

An earlier version of this file claimed the layer's success was predicted by
whether the robot could assemble a dish unaided, with chefs_table and butchery as
the two "yes" layouts that failed. **That claim is withdrawn.** It rested on a
station table that was correct when written and then went stale -- four `.layout`
files were edited mid-experiment and chefs_table lost the robot's pot and onion.
On the current layouts the robot can assemble alone on butchery ONLY, so the
predictor does not describe even its own headline example.

What survives is one weak observation: butchery is the only layout where the robot
can assemble alone, and butchery/fov360 is among the worst cells for every ladder
pairing. One layout is not a predictor.

**What was measured and still stands**, from watching episodes:

- **An abandonment tax of about a third.** Counting job switches with no INTERACT
  since the previous switch -- an approach walked part-way and thrown away -- the
  layer pays ~33% against the baselines' 6-21%. It is paid on the layouts it WINS
  too (divide 24%, pantry 29%), so it is a symptom, not the discriminator.
- **Target-cell switching is negligible**: 0-5 per episode on every layout. The
  lever the tail was built to price is almost never pulled.
- **INTERACT deferral** (arriving and declining to press) is 0-7%, highest on a
  layout the filter WINS. Not the cause.
- **The candidate set is not the problem**: the baseline's realised job is in the
  layer's top three 100% of the time on every layout.

So the layer picks a different job from a correct shortlist, and on several
layouts that choice is worse. The tail mis-valuing jobs is the surviving
hypothesis and it is untested.

## 7. Two things that make grid comparison unsafe

**Layout drift.** Four of the six `.layout` files were edited mid-experiment. An
earlier pair of grids was diffed across that edit and the change attributed to
code. The tell was mechanical: **47 of 120 baseline cells moved**, and the
baselines share no code with the filter, so nothing the filter did could have
moved them.

**Method drift.** `handoff` names a different object before and after the
baselines were made to draw their sub-task. Grids either side of that change use
the same word for two policies, and nothing in a row says so.

Both are now checkable rather than guessable. Every row carries `layout_sha`;
`compare_grids.py` refuses to tally a layout whose fingerprint differs and, for
older fingerprint-less grids, falls back to baseline drift as proof. `report.py`
normalises method names on load so any grid still reads.

Earlier grids' numbers are not reproduced here. They describe different kitchens,
different baselines, or both.

## 8. The value function

`C_theta(v, g, a) = t_end + tail(state at t_end)`, in TICKS, lower better.

**Head**: the real mdp stepped forward against a faithful deep copy of the
theta-shadow, from the forced first action until **the first INTERACT** --
DESIGN.md §3.2's termination rule. **Tail**: bounded A* on the ladder graph from
the head's end configuration. Nodes are sets of completed recipe legs plus each
agent's (position, free-at); edges are one leg assigned to one agent OR relayed
between both through a pass counter; the objective is makespan. Bounded at 1200
expansions, degrading to A*'s own tightest lower bound rather than to a cap.

`Q(v,g,a) = sum over live theta of p(theta) C_theta` (live by `min_p`, MAP
fallback). `Q(a) = MIN over (v,g)`. The sum/min asymmetry is the layer in one
line: theta is genuinely unknown so it gets an expectation; which job to pursue is
ours to choose so it gets a minimum.

**It prices a stash with no rule saying it should**, checked in
`tests/test_qmdp.py`:

| stash cell | tail |
|---|---|
| pass counter the human KNOWS about | **79.0** |
| pass counter in their blind spot | 79.0 |
| not a pass counter at all | 80.0 |

The ordering falls out of geometry plus the human's own `BeliefView`: a leg whose
source the human has never seen is charged `blind` extra ticks.

## 9. Eight bugs, in the order they mattered

| # | bug | symptom |
|---|---|---|
| 1 | **`t_end` fixed at `T`** | `C = T + tail` makes t_end a constant, so the ONE tick the action choice controls is thrown away and must be recovered from a tail that resolves at 4-8 ticks. `C` ranked a step directly AWAY from the target above the greedy step on **86% of ticks on divide**. Terminating the head at the first INTERACT took that to 81% correct. |
| 2 | **anchored on `argmax pi`, not the realised draw** | The fallback pointed at a job the baseline was not doing. Invisible under a deterministic baseline, routine under one that draws: 3 of 24 parity cells diverged. Fixing it took grid parity from 29/30 to **30/30**. |
| 3 | **`eps` on the wrong axis** | Jobs differ by ~29 ticks, actions by ~2. One threshold across both made the layer re-pick its job every tick by argmin. Split into `eps` and `eps_a`. |
| 4 | **greedy list-schedule instead of A\*** | unstable by construction: one tick flips an assignment and the estimate moves tens of ticks. sd 13-16 ticks/tick against a true signal of ~1. |
| 5 | **clocks mutated during search** | every option merely CONSIDERED was charged to the team, so the answer depended on enumeration order. |
| 6 | **agent positions stored as station cells** | stations are not walkable, so `room(pos)` returned -1 and the A* never once reached its goal. |
| 7 | **hitting the expansion bound returned `cap`** | a 10^6-tick spike in the middle of a comparison poisons every candidate near it. |
| 8 | **`cell_k`/`pair_k` truncated the stash shortlist by DISTANCE FROM THE ROBOT** | The six counters kept were the six nearest the robot, which is the theta-blind ordering the layer exists to overcome. On back_bar the shortlist held no counter the human could see on 55% of stash ticks; on banquet_pass it averaged 0.00 visible counters. `T=25` clipped the far counters on four of six layouts (the farthest legal target is 28 steps). Removing both took `qmdp` and `qmdp-solo` from -2.2 dishes to +1.4 -- see section 2b. |

Settled configuration, now the class defaults: `T=25, eps=2.0, eps_a=2.0,
min_p=0.05, blind=8.0, m=3, cell_k=6, pair_k=6`.

## 10. Prerequisite bug in the baselines

`_BaseRobot.rank_subtasks` and `greedy`'s copy called `legal_subtasks` WITHOUT the
reachability predicate, then filtered unreachable candidates out afterwards.
`bayesian_delegation.py:240` already passed it, which is why bayes was the only
baseline that worked. One missing argument caused two failures at once:

- `actionable("dish", view, ok)` answered "yes, there is a serve station" about a
  hatch across the divide the robot can never reach, so the robot stashed a
  finished dish and picked it straight back up -- **115 stashes and 114 take_dish
  in one 400-tick episode on divide**, 0 delivered. `tasks.py:76-91` predicts
  exactly this in its own docstring.
- A non-empty legal list suppressed the stash fallback, every candidate was then
  dropped for being unreachable, the list came back empty and the robot stood
  still -- **98% of ticks** on back_bar and pantry.

Every grid since is meaningless without it.

## 11. Performance

Two optimisations, each proven equivalent BEFORE being relied on:

| | check | result |
|---|---|---|
| BFS distance field replacing `path_len`'s A* | 396,615 exhaustive (every walkable start x every cell, 6 layouts) | 0 mismatches, 2.9x |
| memoised `visible_cells`, bounded at 4000 entries | 39,840 exhaustive | 0 mismatches |

The tail costs 3-11 ms a call; the filter runs at 82 ms/tick over handoff and 128
over bayes. `qmdp-base` at 9 ms/tick is the cost of the machinery with the search
switched off. The full grid is ~1.5 h wall-clock as a 30-task CARC array.

Operational notes:

- Stopping a `run_local.sh` grid kills the wrapper but NOT the `xargs` tree, which
  gets reparented to init and keeps running. Three abandoned grids' worth of
  orphans turned a 55-second episode into a 20-minute one and looked exactly like
  the machine being slow. `run_local.sh` now traps EXIT/INT/TERM and kills its own
  process group.
- `discovery.usc.edu` does not always resolve through the system resolver even
  with the VPN up. The internal nameserver has it as `10.72.0.13`;
  `ssh -o HostKeyAlias=discovery.usc.edu 10.72.0.13` works when the name does not.
- Rsync to CARC once reported success without transferring; caught by comparing
  SHA256. Use `rsync --checksum` when the library there matters.

## 12. Known limits

- **The gate is not met.** 11/8/11 for the headline pairing, and -2.2 dishes over
  the grid. Only `qmdp-bayes` is positive on both counts.
- **Cone inference is not carrying the result.** `qmdp` vs `qmdp-fixed` is 9/13/8
  with the frozen cone marginally ahead on dishes. The one clear win is over
  bayes, and this control says it is not the posterior producing it.
- **butchery fov360 and chefs_table fov360 lose by whole dishes** under the ladder
  pairings. The most concentrated failures on the grid.
- **`pair_k=1` does not yield one pair**, so `qmdp-base` parity is empirical
  rather than structural. Open. Section 4.
- **An abandonment tax of about a third**, paid on winning layouts too, so a
  symptom rather than the discriminator. Unexplained. Section 6.
- **Commitment persistence is kept on the argument, not the numbers.** Holding a
  committed pair while it stays LEGAL rather than while it stays in the top-m
  moved two cells out of twenty when it was isolated, one of them a loss.
- **Never probes.** No term values a sharper `p(theta)`, so the robot never spends
  a tick disambiguating cones. A property of QMDP itself; probing would be a
  separate, ablatable bonus.
- **One-step deviation per tick.** A detour whose payoff needs the first TWO steps
  to both be non-greedy is invisible until the branch point arrives.
- **As good as the human model.** Every `C_theta` is a claim about what a
  theta-human would do, so a change to the ladder silently changes every score.
  The filter and the rollouts must always run the same ladder.
- **`blind` is a modelling choice**, not a tuning knob: it asserts a partner will
  eventually sweep past an item they have not seen, at a cost of `blind` ticks. 8
  was chosen because 30 made the estimate jump by 30 ticks every time a cone
  crossed a counter.

## 13. Methods

A baseline column, the same filter over each of them, two controls, and two
later candidate-set variants (section 2b).

    greedy         nearest job first -- no partner model at all
    solo           greedy + a within-tier demotion of cells the human reaches
                   sooner. Section 5: on this suite it costs more than it buys.
    handoff        solo + staging. Two changes, not one: it prepends reachable
                   free counters when holding something shareable, and biases the
                   `stash` verb within its tier -- which does not reorder its own
                   ranking but does shift the pi the layer reads.
    bayes          + intent: a belief over sub-task ALLOCATIONS updated by inverse
                   planning. Models the mind, never the eyes.
    bayes-noip     EVIDENCE CONTROL: inverse planning off. NOT frozen at the prior
                   -- the belief is still carried and re-projected each tick --
                   but evidence is the only knob that differs.

    qmdp           THE filter, over handoff. One class, core/qmdp.py:QMDPFilter.
                   INTERACT-legality is a PREDICATE on (position, orientation)
                   over the union of the top-m jobs' legal cells, and every
                   (cell, arrival tick) reachable inside depth=30 is scored. So
                   every stash counter is in scope -- 23 to 51 of them, not the
                   6 nearest the robot -- and "same counter, two ticks later" is
                   a distinct plan. Q = C at the MAP cone only, and no search at
                   all below certainty=0.9.
    qmdp-greedy    over greedy's jobs
    qmdp-solo      over solo's jobs
    qmdp-bayes     over bayes's jobs -- the only positive pairing, and the slowest
    qmdp-base      PARITY CONTROL: search collapsed to the baseline's own
                   candidate; 30/30 identical on this grid
    qmdp-fixed     THETA-BLIND CONTROL: same rollouts, cone frozen at 90. It
                   freezes the posterior's WEIGHTS, not the posterior -- that is
                   still built, still fed the human's action, still reported.

**EVERY BASELINE DRAWS ITS SUB-TASK, and no name carries a `-stoch` suffix because
there is nothing left to contrast with.** `METHOD_KEYS` is `greedy, solo, handoff,
bayes, bayes-noip` plus the six `qmdp-*`. No deterministic ladder is reachable
from a command line; `BASELINES` in `robot/nominal_policy/` still holds the
classes if one is ever needed directly.

`handoff` therefore MEANS something different than it once did -- a change of
meaning rather than a rename, and deliberate. DESIGN.md §3.0 consumes a
distribution over sub-tasks and a deterministic policy does not have one, so
`true_pi` could only hand the layer a lifted Boltzmann RECONSTRUCTION of what a
policy that never draws might have drawn, and it was then scored against a
baseline that never held those preferences. Wrong on both sides at once. Now
`qmdp` wraps `handoff` and is scored against `handoff` -- the same object, reading
the very `pi` that baseline sampled from. The interim spellings (`handoff-stoch`,
`bayes-post`, `bayes-prior`) resolve through `ALIASES`, and `analysis/report.py`
normalises them on load.

The superseded subtask-level filter (`qmdp_fov.QMDPFilter`) has been deleted; the
cone inference it shared a module with survives as `core/fov_posterior.py`.

## 14. Package layout

`README.md` has the map. In short: `core/` is what the robot runs, `harness/`
produces grids, `analysis/` reads them, `tests/` guards `core/`, and `dev/` holds
measurement instruments. The dependency runs one way,
`core/ <- harness/ <- {analysis, dev, tests}`, so deleting any of the last three
leaves the robot working.
