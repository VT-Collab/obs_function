# qmdp: full results

The execution-level QMDP layer, built to `DESIGN.md`. Everything measured,
including what does not work.

**Grid.** CARC job `11001543`. 6 layouts x 5 cones x 4 pairings x 10 seeds x
horizon 400 = **2400 episodes**. Every cell has exactly 10 seeds, every row's
`layout_sha` matches the current `.layout` file, zero tracebacks.

**The comparison is PAIRED, and both sides are the same object.** Each filter is
scored against the baseline it wraps -- `qmdp` over handoff against handoff --
because the layer re-ranks its own baseline's jobs and falls back to its own
baseline's action, so that baseline is both its floor and the thing it must beat.
Every baseline DRAWS its sub-task from its own distribution, which is what the
filter consumes; section 13 says why that is a correctness requirement.

**Pass rule.** More dishes wins; at equal dishes, fewer ticks. Cells are averaged
over 10 seeds before comparison. An unfinished episode scores the horizon, 400.
Two dishes is a full score: `is_terminal` fires at `len(order_list) <= 1`.

**Regenerate every number here** -- none of it is typed by hand:

    python -m robot.filter.analysis.results_tables GRID_DIR
    python -m robot.filter.analysis.report          GRID_DIR/*.jsonl
    python -m robot.filter.analysis.layout_facts
    python -m robot.filter.tests.test_qmdp
    python -m robot.filter.tests.test_value_tail

Glob `qmdp*.jsonl`, never `*.jsonl`. An older grid once sat in the same output
directory under method names that `report.NAME_FIX` maps onto the current ones, so
reading the directory pooled two code versions under identical keys -- half the rows
from a build running 1000x faster.

**The kitchens this grid ran in.** The layouts are experimental material and they
change; these numbers are only about these six files, and every row carries its
fingerprint:

    back_bar 76f0323586   banquet_pass 252a7bcc0d   butchery a86a1d541b
    chefs_table 0aec924651   divide 57e9e2bdf5      pantry 707da824f6

    EARLIER GRIDS HAVE BEEN DELETED FROM THIS FILE RATHER THAN KEPT ALONGSIDE IT.
    Five of the six .layout files were edited mid-experiment and CARC was holding
    pre-edit copies of all five -- `robot/` and `common/` were being synced,
    `layout/` never was -- so every table written before this one described kitchens
    that no longer exist:

        layout         earlier grids   current
        back_bar       76f0323586      76f0323586   same
        banquet_pass   46fec8e416      252a7bcc0d   DIFFERENT
        butchery       57da296901      a86a1d541b   DIFFERENT
        chefs_table    47631e1490      0aec924651   DIFFERENT
        divide         c9d93b6640      57e9e2bdf5   DIFFERENT
        pantry         faeee8fb24      707da824f6   DIFFERENT

    That is the third time this package has been bitten the same way, and
    `analysis/layout_facts.py`'s docstring says why it is worse than being wrong: the
    numbers were correct when written and nothing about them looks stale. Check the
    fingerprints before quoting any number here next to another grid's.

---

## 1. Headline

| pairing | win | tie | loss | dishes: baseline -> layer | change |
|---|---|---|---|---|---|
| `qmdp-greedy` vs greedy | 30 | 0 | 0 | 8.8 -> 53.6 | +44.8 |
| `qmdp-solo` vs solo | 30 | 0 | 0 | 5.5 -> 53.9 | +48.4 |
| `qmdp` vs handoff | 30 | 0 | 0 | 5.5 -> 53.9 | +48.4 |
| `qmdp-bayes` vs bayes | 26 | 0 | 4 | 45.8 -> 54.8 | +9.0 |

win/tie/loss counts CELLS. `dishes` sums the per-cell mean over all 30 cells, so it
is comparable across pairings here because every pairing has all 30.

**The gate is met, and by a wide margin.** All three ladder pairings take 30 of 30
cells. `qmdp-bayes` takes 26 of 30 and still gains 9.0 dishes.

Cell counts and total dishes agree, which they did not in earlier grids -- there the
ladder pairings were level on cells and DOWN two dishes, and the disagreement was the
finding. Both views now point the same way.

**ALL THREE LADDER PAIRINGS LAND ON THE SAME NUMBER -- 53.6, 53.9, 53.9 dishes --
whichever baseline they wrap, while their baselines stay far apart at 8.8, 5.5 and
5.5.** That is the most important line in this file and it is not a good sign on its
own. Through every earlier grid the pairings stayed spread, because the layer
re-ranked its baseline's jobs and inherited its floor. Landing on one number says the
baseline now supplies only the candidate set and the search decides everything else,
which is either the layer working as designed or the layer having stopped listening to
its baseline.

**THIS GRID CANNOT TELL THOSE APART**, because there is no parity control in it.
`qmdp-base` collapses the mask to the baseline's own realised cell and must therefore
tie its baseline in every cell; it is exactly the check that distinguishes "correctly
better" from "ignoring its baseline", and it is the check that caught the argmax-pi
anchor bug (section 4). It was not in the array. **Do not quote +48.4 as settled
without it.** The layer also overrides its baseline on half of all ticks (49-52%,
below), which is the same observation from the other side.

**bayes is the one pairing that loses cells** -- 4 of 30 -- while still gaining 9.0
dishes. It is the strong baseline: it already models the partner's INTENT and already
filters its handoffs to counters the partner can REACH, so at a wide cone it has most
of what the FOV layer exists to supply. Three of its four losses sit at fov 180-360
(butchery 180, chefs_table 360, divide 360) where its baseline already scores 2.0, so
there the layer is paying search time for ticks it cannot win back.

## 2. The whole table

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

`*` the layer beat its baseline, `X` lost, blank a tie. dishes/ticks, mean over 10
seeds.

Cells that were dead for every policy in the earlier grids now deliver -- back_bar
60-360 goes `0.0/400 -> 2.0/230-314`, divide 90-360 `0.0/400 -> 2.0/210-262`, butchery
90-360 `0.0/400 -> 2.0/244-275`. **Only butchery/30 is still near-dead**, at 0.3
dishes for the ladder pairings and 0.0 for `qmdp-bayes`, which is the one cell where
the layer loses to bayes outright.

**Cost.**

```
    qmdp          deviates 52.3% of ticks   1201 ms/tick
    qmdp-bayes    deviates 49.2% of ticks    723 ms/tick
    qmdp-greedy   deviates 52.0% of ticks   1347 ms/tick
    qmdp-solo     deviates 52.3% of ticks   1268 ms/tick
```

Against 82 ms/tick for the enumerate-`(v, g, a)` filter this replaced, and 9 ms for
the machinery with the search switched off. `tail_ticks` is 91% of a stash tick: the
head scores 360 plans there -- 24 counters x 5 first actions x 3 arrival ticks -- and
calls the tail once each, where a fetch tick calls it 45 times. Two EXACT speedups are
identified and NOT applied: prune on `avail`, which spans 12-223 while the tail spans
108-124, so with `tail >= 0` most plans can never win; and dedupe the tail across the
360 plans' 120 distinct (cell, press tick) pairs.

## 2b. Why the human model changed, and what it was worth

The human's redundancy check now counts COMPOSITES: a `garnish_dish` is a garnish AND
a plate, so it stops the garnish chain and the plate chain alike. Before, it counted
only the bare product -- `counters_holding("garnish")` could not see a `garnish_dish`
-- and the human went on fetching onions, chopping them, fetching plates and washing
them for a garnish and a plate it was already holding. On a constructed state two
garnish_dishes blocked 0 verbs before the change and 10 after.
`common/tasks.py:_CONTAINED_IN` is the table, `_counts_composites` the gate.

**The change was measured before being adopted** -- the same grid, both human models,
2400 episodes each, CARC jobs `11001542` (old) and `11001543` (new, shipped),
identical robot code, seeds and layouts:

    pairing                  old human            new human (shipped)
    qmdp-greedy vs greedy     7.5 -> 49.6  +42.1   8.8 -> 53.6  +44.8
    qmdp-solo   vs solo       3.7 -> 49.6  +45.9   5.5 -> 53.9  +48.4
    qmdp        vs handoff    3.7 -> 49.6  +45.9   5.5 -> 53.9  +48.4
    qmdp-bayes  vs bayes     42.4 -> 50.8   +8.4  45.8 -> 54.8   +9.0

    cells, old human:  29/1/0  29/1/0  29/1/0  27/0/3
    cells, new human:  30/0/0  30/0/0  30/0/0  26/0/4

**The layer's gain does not depend on the partner's bug**, which is the hypothesis the
experiment existed to kill. Had the layer been exploiting a partner that over-produces,
fixing the partner would have shrunk the gain; it grew by about 2.5 dishes. That
hypothesis is dead.

**And it is not a confound, because it helps both sides.** Every baseline improved too
-- greedy 7.5 -> 8.8, solo and handoff 3.7 -> 5.5, bayes 42.4 -> 45.8. A partner that
stops fetching onions it already has is a better partner for anything.

The rule is BELIEF-ONLY by construction: `tasks._counts_composites` gates on the view,
so the human counts composites and every baseline robot, which reads a `TruthView`,
does not. That is deliberate rather than cautious. The baselines are the control the
filter is measured against, and a rule that moved them would have made the comparison
measure nothing -- and keeping the gate there means the shipped code is identical to
what the right-hand column above actually ran. It does leave the two agents reasoning
about saturation by different rules, which section 12 lists as a known limit.

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

## 4. The controls -- BOTH STALE, NEITHER RE-RUN

**Neither control has been run against the current filter or the current layouts.**
Both numbers below come from the enumerate-`(v, g, a)` filter on the pre-edit
`.layout` files, so they are two changes away from anything in sections 1 and 2. They
are kept because the mechanism each one checks is still in the code and still
unchecked, not because the figures transfer. **Running `qmdp-base` and `qmdp-fixed` on
the current grid is the single most valuable outstanding job in this package.**

**PARITY (`qmdp-base`) -- was IDENTICAL IN 30 OF 30 CELLS**, with a dish total exactly
equal to its baseline's. The search is collapsed to the baseline's own candidate while
the real rollout machinery still runs, so this is the degrade-to-baseline guarantee
measured end to end at horizon 400.

That check earned its keep. An earlier grid had one break, 29/30, and the cause was
found and fixed: the layer anchored candidate 0 on `argmax pi` rather than on the
baseline's REALISED draw. `pi` is a distribution and the pick is a sample from it, so
the two diverge routinely once the baseline draws -- and the "baseline's own action"
the tie-break falls back to was not the baseline's action. DESIGN.md 3.0b.

The `pair_k=1` caveat that used to sit here **is gone with the code it described.**
The old control collapsed via `pairs[:max(1, pair_k - 1)] + [must]`, which yielded two
pairs rather than one, so parity was empirical rather than structural. There are no
pairs now: `qmdp-base` is `only_base=True, waits=0`, which restricts the mask to the
baseline's realised cell and removes the arrival choice, so the collapse is structural.
It has not been re-measured.

**THETA-BLIND (`qmdp-fixed`, cone frozen at 90).** Same rollouts, same tail, same
guards; the only difference is whether the cone posterior's weights are used.

    inferring theta   better 9   identical 13   worse 8
    dishes            qmdp 9.0   vs   qmdp-fixed 9.4

**That was a wash, and on total dishes the frozen cone was marginally AHEAD.** The
per-cell count leaned very slightly to inference, the food count the other way, and
neither margin supported a claim.

**A second, independent measurement says the same thing and is NOT stale**, because it
is a property of the posterior rather than of a grid: over 18,000 ticks the mean
`p(MAP)` is 0.99, `p(MAP) >= 0.9` on 98.4% of ticks, and the MAP cone is correct on
100.0% of the ticks above that threshold. The live set was already a singleton almost
always, so the mixture over cones is decoration and `Q = C_MAP` loses nothing. A layer
whose posterior is that sharp cannot be getting much from the posterior's SHAPE.

So the honest state of the research question is unchanged by the new grid: **the
layer's large gain is a SEARCH result, and the cone posterior is not demonstrably what
produces it.** The theta-blind control is what would settle it and it has not been run.

## 5. Why greedy beats solo and handoff

Not a filter result, but it sets the bar the filter has to clear, and it turned up
while verifying a docstring rather than while looking for it. It still holds on the
current grid: greedy 8.8 dishes against solo and handoff at 5.5.

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
the two "yes" layouts that failed. **That claim is withdrawn**, twice over.

It first rested on a station table that was correct when written and then went stale
-- four `.layout` files were edited mid-experiment and chefs_table lost the robot's
pot and onion. On the current layouts the robot can assemble alone on butchery ONLY,
so the predictor does not describe even its own headline example.

**And the examples it was built on have since reversed.** butchery/fov360 and
chefs_table/fov360 were the two worst cells for the ladder pairings; on the current
grid both are `2.0/248-252`, full marks. There is nothing left of the pattern.

**What was measured and still stands**, from watching episodes:

- **An abandonment tax of about a third.** Counting job switches with no INTERACT
  since the previous switch -- an approach walked part-way and thrown away -- the
  layer pays ~33% against the baselines' 6-21%. It is paid on the layouts it WINS
  too (divide 24%, pantry 29%), so it is a symptom, not a discriminator.
- **Target-cell switching is negligible**: 0-5 per episode on every layout. The
  lever the tail was built to price is almost never pulled.
- **INTERACT deferral** (arriving and declining to press) is 0-7%, highest on a
  layout the filter WINS. Not a cause.
- **The candidate set is not a problem**: the baseline's realised job is in the
  layer's top three 100% of the time on every layout.

These were measured while the layer was losing dishes and they explained nothing then.
They have not been re-measured now that it wins, and the abandonment tax in particular
is worth re-reading against a 50% deviation rate: a layer that overrides its baseline
on half of all ticks and abandons a third of its approaches may simply be describing
the same behaviour twice.

## 7. Two things that make grid comparison unsafe

**Layout drift.** Five of the six `.layout` files have now been edited
mid-experiment, and an earlier pair of grids was diffed across one such edit with the
change attributed to code. The tell was mechanical: **47 of 120 baseline cells
moved**, and the baselines share no code with the filter, so nothing the filter did
could have moved them.

Worse than editing them is editing them on ONE MACHINE. The grid that became section
2c ran on CARC against pre-edit copies of five layouts, because `robot/` and `common/`
were being rsynced and `layout/` never was, and every fingerprint in it therefore
described a kitchen that existed nowhere. **Sync `layout/` with the code, and check the
fingerprints in the output rather than trusting the sync.**

**Method drift.** `handoff` names a different object before and after the
baselines were made to draw their sub-task. Grids either side of that change use
the same word for two policies, and nothing in a row says so.

Both are now checkable rather than guessable. Every row carries `layout_sha`;
`compare_grids.py` refuses to tally a layout whose fingerprint differs and, for
older fingerprint-less grids, falls back to baseline drift as proof. `report.py`
normalises method names on load so any grid still reads.

Earlier grids' numbers are not reproduced here. They describe different kitchens,
different baselines, a different filter, or all three.

## 8. The value function

`C(seq) = t_end + tail(state at t_end)`, in TICKS, lower better -- evaluated at the
terminals of a masked action search rather than at the head of an enumerated
`(v, g, a)` triple.

**Head**: the real mdp stepped forward against a faithful deep copy of the
theta-shadow, from the forced first action until **the first INTERACT** --
DESIGN.md 3.2's termination rule. **Tail**: bounded A* on the ladder graph from
the head's end configuration. Nodes are sets of completed recipe legs plus each
agent's (position, free-at); edges are one leg assigned to one agent OR relayed
between both through a pass counter; the objective is makespan. Bounded at 1200
expansions, degrading to A*'s own tightest lower bound rather than to a cap.

`Q(a) = MIN over every terminal whose FIRST action was a`, at the MAP cone only, and
no search at all when `p(MAP) < certainty`. The min is a decision rather than
epistemic uncertainty: after taking `a` the robot is free to continue however is best
from where it lands. What the cone gets is an expectation -- theta is genuinely
unknown -- and section 4 records that with `p(MAP)` at 0.99 the expectation has almost
nothing left to average over.

**It prices a stash with no rule saying it should**, checked in
`tests/test_qmdp.py`:

| stash cell | tail |
|---|---|
| pass counter the human KNOWS about | **81.0** |
| pass counter in their blind spot | 81.0 |
| not a pass counter at all | 82.0 |

The ordering falls out of geometry plus the human's own `BeliefView`. Note what it
does and does not show: a counter that is not a pass counter is charged more, so the
handoff structure is priced -- but the first two rows are EQUAL, so on this fixture
the blind spot costs nothing and the `blind` term is not what separates them. The
stash cost that does depend on the cone is in the head, not the tail: `_collect`
prices a stash by when the partner has it IN HAND, which is their first look at or
after the press plus their walk, and charges `no_handoff` when there is no such look.

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
| 8 | **`cell_k`/`pair_k` truncated the stash shortlist by DISTANCE FROM THE ROBOT** | The six counters kept were the six nearest the robot, which is the theta-blind ordering the layer exists to overcome. On back_bar the shortlist held no counter the human could see on 55% of stash ticks; on banquet_pass it averaged 0.00 visible counters. `T=25` clipped the far counters on four of six layouts (the farthest legal target is 28 steps). Removing both took `qmdp` and `qmdp-solo` from -2.2 dishes to +1.4 on the layouts of the day, and both identifiers are now absent from `core/qmdp.py`. |

Four more that came after those eight, all in the current class and all found by
tracing a freeze in `watch.py` rather than by a grid:

| # | bug | symptom |
|---|---|---|
| 9 | **`Q(a)` did not depend on `a`** | terminals were keyed `(cell, tick)` with the first actions collected into a set, so ONE `C` was assigned to every action that could reach that terminal. The argmin was then over identical numbers. Fixed by putting the first action in the key. |
| 10 | **no commitment -> STAY deadlock** | with the plan re-decided from scratch each tick, a gain of exactly `eps_a` failed the test, the layer emitted the baseline's STAY, the state did not change, and the same tick repeated forever. Traced on back_bar/qmdp-greedy alternating (13,3) and (13,2). Fixed by guard (ii), `eps` on the committed target cell. |
| 11 | **`_replay` beelined instead of replaying** | the winning sequence was discarded and re-derived as a walk toward the target, so INTERACT fired off-target and `C` was scoring trajectories the robot never flew. Fixed with `via` parent pointers through the search. |
| 12 | **dedupe merged across first-action branches** | `seen` was keyed without the first action, so a slower action's subtree was pruned by a faster one's and got no `Q` at all. |

Settled configuration, the current class defaults:

    m=3  depth=30  waits=2  eps=2.0  eps_a=2.0  blind=8.0
    certainty=0.9  no_handoff=200.0  max_states=200000  temperature=4.0

`m` is the only candidate bound left. `cell_k`, `pair_k`, `T` and `min_p` are gone
with the filter that used them.

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
| `fast_clone_human` replacing `clone_human` | full rollouts, `tests/test_qmdp.py` | identical, ~7x |

The third is the one that matters most here. `copy.deepcopy(h.view)` was 68% of
search runtime -- 6.4 million deepcopy calls for 2,774 clones -- because the
enumerating filter cloned once per CANDIDATE while this one clones once per search
NODE. Nothing needed deep copying: every value in the belief stores is a str, int or
tuple, and only `_stations` holds a mutable set.

**The tail costs 3-11 ms a call and the filter runs at 723-1347 ms/tick** (section 2),
against 82 ms/tick for the enumerating filter and 9 ms/tick for `qmdp-base`, which is
the cost of the machinery with the search switched off. A full 2400-episode grid is a
120-task CARC array and takes a few hours; it is no longer the ~1.5 h the enumerating
filter needed.

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
- **`conda activate` in an sbatch script can silently run the login node's base
  python.** Sixteen array tasks exited 0 in three seconds having written nothing.
  Call the interpreter by absolute path -- `$HOME/.conda/envs/steakhouse-ai/bin/python`
  -- and make the task fail loudly if its output file is empty.
- **SLURM writes `.err` files where the job was submitted from, not `~/logs`.** A
  "zero tracebacks" check globbed the wrong directory and came back clean five times
  in a row while saying nothing at all.

## 12. Known limits

- **NO PARITY CONTROL ON THIS GRID, and it is the one that matters.** `qmdp-base`
  must tie its baseline in every cell; it was not in the array. Until it runs, part
  of +48.4 could be an artifact of the layer no longer degrading to its baseline
  correctly -- which is exactly the bug it caught once before. Section 4.
- **The three ladder pairings land on one number** (53.6/53.9/53.9) from baselines
  that are far apart (8.8/5.5/5.5), and the layer overrides its baseline on ~50% of
  ticks. Consistent with the search doing the work; also consistent with the search
  ignoring its floor. The parity control is what separates them.
- **Cone inference is not demonstrably carrying the result.** The theta-blind
  control is stale, and where it last ran the frozen cone was marginally AHEAD on
  dishes. Independently, `p(MAP)` averages 0.99 and the MAP cone is right on 100% of
  ticks above the 0.9 gate, so the posterior's shape has almost no room to matter.
  The large win is a SEARCH result until `qmdp-fixed` says otherwise. Section 4.
- **Cost is 0.7-1.3 s/tick**, 9-16x the filter it replaced, with two exact speedups
  identified and unapplied. Section 2.
- **butchery/30 is still near-dead** -- 0.3 dishes for the ladder pairings, and the
  one cell where `qmdp-bayes` scores 0.0 against a baseline that scores 0.2.
- **An abandonment tax of about a third**, paid on winning layouts too. Measured
  while the layer was losing and never re-measured now that it wins. Section 6.
- **The human counts composites toward saturation and the robot does not.** The gate
  is on the VIEW, so the two agents apply different redundancy rules. Deliberate --
  it keeps the baselines a clean control and keeps the code identical to what was
  measured -- but it is an asymmetry in the model, not a fact about the world.
  Section 2b.
- **Never probes.** No term values a sharper `p(theta)`, so the robot never spends
  a tick disambiguating cones. A property of QMDP itself; probing would be a
  separate, ablatable bonus.
- **Branching is given up for reach.** The search enumerates (cell, arrival tick)
  and walks ONE path per cell, so route variation is not searched. `dev/route_channel.py`
  prices that at 0.8% of ticks, and 0.0% on divide and pantry. The alternative was
  arithmetically impossible: reaching a 28-step counter by branching is ~1.6e12
  states.
- **As good as the human model.** Every `C` is a claim about what a theta-human would
  do, so a change to the ladder silently changes every score, and the 2400-episode
  pair in section 2b is what that costs to check. The filter and the rollouts must
  always run the same ladder.
- **`blind` and `no_handoff` are modelling choices**, not tuning knobs. `blind=8`
  asserts a partner eventually sweeps past an item they have not seen; 30 made the
  estimate jump by 30 ticks every time a cone crossed a counter. `no_handoff=200`
  must exceed the largest real pickup or the layer prefers a guess to a measurement
  -- which it did when `blind=8` alone made "nobody ever looks here" the cheapest
  option on the board.
- **Inference is against a perfectly specified model.** The shadows run the SAME
  ladder as the human, so the MAP cone's 98.9% accuracy is a ceiling rather than a
  performance estimate. Against a human whose ladder differs it falls.

## 13. Methods

A baseline column, the same filter over each of them, and two controls.

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
    qmdp-bayes     over bayes's jobs -- the slowest, and the only pairing that
                   loses cells
    qmdp-base      PARITY CONTROL: `only_base=True, waits=0` restricts the mask to
                   the baseline's own realised cell and removes the arrival choice,
                   so the collapse is structural. NOT RUN on the current grid.
    qmdp-fixed     THETA-BLIND CONTROL: same rollouts, cone frozen at 90. It
                   freezes the posterior's WEIGHTS, not the posterior -- that is
                   still built, still fed the human's action, still reported.
                   NOT RUN on the current grid.

Eleven rows, and the two controls are the two that are missing from sections 1 and 2.

**EVERY BASELINE DRAWS ITS SUB-TASK, and no name carries a `-stoch` suffix because
there is nothing left to contrast with.** `METHOD_KEYS` is `greedy, solo, handoff,
bayes, bayes-noip` plus the six `qmdp-*`. No deterministic ladder is reachable
from a command line; `BASELINES` in `robot/nominal_policy/` still holds the
classes if one is ever needed directly.

`handoff` therefore MEANS something different than it once did -- a change of
meaning rather than a rename, and deliberate. DESIGN.md 3.0 consumes a
distribution over sub-tasks and a deterministic policy does not have one, so
`true_pi` could only hand the layer a lifted Boltzmann RECONSTRUCTION of what a
policy that never draws might have drawn, and it was then scored against a
baseline that never held those preferences. Wrong on both sides at once. Now
`qmdp` wraps `handoff` and is scored against `handoff` -- the same object, reading
the very `pi` that baseline sampled from. The interim spellings (`handoff-stoch`,
`bayes-post`, `bayes-prior`) resolve through `ALIASES`, and `analysis/report.py`
normalises them on load.

Two filters have been deleted rather than kept: `qmdp_fov.QMDPFilter`, which
re-ranked whole SUB-TASKS and walked one shortest path to the winner, and the
enumerate-`(v, g, a)` execution filter that replaced it. The cone inference the first
shared a module with survives as `core/fov_posterior.py`; the `qmdp-exec-*`,
`qmdp-enum-*` and `qmdp-mask-*` spellings both of them ran under resolve through
`ALIASES` so their saved grids still read.

## 14. Package layout

`README.md` has the map. In short: `core/` is what the robot runs, `harness/`
produces grids, `analysis/` reads them, `tests/` guards `core/`, and `dev/` holds
measurement instruments. The dependency runs one way,
`core/ <- harness/ <- {analysis, dev, tests}`, so deleting any of the last three
leaves the robot working.
