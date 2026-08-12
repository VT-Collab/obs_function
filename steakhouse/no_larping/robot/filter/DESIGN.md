# QMDP over EXECUTION, with a posterior over the human's cone

The layer sits on top of an unchanged theta-blind baseline. Each tick it takes
the baseline's top three subtasks at the VERB level, then scores every way of
executing each of them — which job, which target cell, which first step, whether
to wait — by rolling the real mdp forward against one faithful simulated human
per cone hypothesis, mixing by the cone posterior, and emitting a score over the
six low-level actions. When the scores are flat it reproduces the baseline
exactly, so it can only improve on the nominal policy, never scramble it.

`RESULTS.md` has the measurements. This file is the design and the reasons.

## Symbols

`theta` a cone hypothesis in {30,60,90,180,360}; `p(theta)` the filter's
posterior; `tau = (tier, verb, cell)` a baseline subtask, tier being the urgency
class (DELIVER down to EXPLORE); `pi(tau)` the baseline's distribution over its
legal tau; `v` a candidate JOB `(tier, verb)`, one of the top `m = 3`; `g` a
candidate target cell; `G(v)` the legal cells for `v`; `a` a first low-level
action; `A(x,g)` the masked action set; `T` the head horizon; `C_theta(v,g,a)`
estimated ticks-to-finish-everything; `Q` the belief-weighted mixture of `C`;
`eps` / `eps_a` the switching margins; `alpha` trust in the human model (~0.9);
`certainty` the floor on `p(MAP)` below which the layer does not search at all.

## 0. The environment fact everything rests on

On EVERY layout in the suite the two agents stand in DISCONNECTED rooms, and the
robot has no serve hatch anywhere. **Regenerate with `python -m
robot.filter.analysis.layout_facts` rather than trusting this copy** — the .layout files
have been edited mid-experiment before, and a transcribed table that was correct
when written went on describing a kitchen that no longer existed:

    layout        robot     human      can assemble alone
    back_bar      .B.MOD.   PBW.O.S    neither
    banquet_pass  ..WMO..   PBWMODS    human
    butchery      PBWMOD.   .BWMODS    robot
    chefs_table   .BWM.D.   PBW.ODS    neither
    divide        PBW..D.   .B.MO.S    neither
    pantry        .B.MOD.   PBW.O.S    neither

    P pot  B board  W sink  M meat  O onion  D plate  S serve;  '.' = cannot reach
    A dish needs PBWMOD, then a serve hatch.

**No dish is deliverable by one agent anywhere** — the robot reaches a serve
hatch on no layout at all. Every dish crosses the divide through a counter both
can stand at. The stash cell is therefore not a nicety, it is the only channel —
and a value function that cannot express a handoff cannot price anything here. It
reports "impossible" for every candidate and the layer goes inert.

**Assembling alone is rarer still**: only butchery's robot and banquet_pass's
human hold all six of pot, board, sink, meat, onion and plate. On the other four
layouts even the sub-assemblies have to be traded.

## 1. Posterior over theta — CURRENT, lives in `core/fov_posterior.py`

Hidden: the human's cone, inferred from their ACTIONS alone. The robot never
reads the human's beliefs.

The human is a deterministic ladder, so a hypothesised `theta` yields exactly one
predicted action. `FOVPosterior` keeps one shadow `LimitedVisionHuman` per cone,
steps each on the TRUE state every tick so it perceives through its own cone and
keeps its own decaying beliefs, and reads the action off it.

**Soft-deterministic likelihood.** A hard match would zero a hypothesis on the
first modelling error and never recover:

    P(a_H | theta) = alpha              if a_H == predicted(theta)     alpha ~ 0.9
                   = (1-alpha)/(|A|-1)  otherwise

`alpha` is trust in our human MODEL, not a property of the human.

**The hidden variable costs nothing.** `b(theta, tau_H) = p(theta) * 1[tau_H is
the theta-shadow's current subtask]`. All the mass sits on one `tau` per cone
because the ladder is deterministic given theta, so the joint hypothesis is READ,
not built — the shadow already IS the conditional. Load-bearing assumption: a
softened (Boltzmann) human breaks this and `tau_H` would need its own posterior.

## 2. The baseline contract

Every baseline exposes two things and the layer reads both without editing
either. Accessors live in `robot/nominal_policy/subtask_dist.py`:

- `true_ranking(bot, state)` — the baseline's OWN ordering, already sorted by its
  own key `(tier, penalty, distance, cell, verb)`. Not modelled.
- `true_subtask(bot, state)` — what it is ACTUALLY doing, stickiness included.
  Only `action()` knows, because only `action()` applies stickiness. SIDE
  EFFECTING: it advances `t`, `log`, `last_subtask`, and for bayes `last_action`,
  which its next `update()` reads back as evidence. Call once per tick.
- `true_pi(bot, state, ...)` — returns `(pi, source)`. A drawing policy hands
  back the distribution it actually sampled from (`"drawn"`); bayes can hand back
  its genuine allocation posterior (`"posterior"`); a deterministic policy has no
  distribution at all and can only be given the lifted Boltzmann (`"modelled"`).

**EVERY BASELINE THIS LAYER WRAPS IS STOCHASTIC, and that is a correctness
requirement, not a preference.** §3.0 consumes a distribution over subtasks. A
deterministic ladder does not have one, so `"modelled"` is a reconstruction of
what a policy that never draws anything might have drawn — and the layer was then
scored against a baseline that never held those preferences. Both halves of that
are wrong at once. As it stands the layer reads the very `pi` its baseline
sampled from: measured live over 30 ticks on divide, source is `"drawn"` on every
tick for all four pairings (bayes shows `"modelled"` on tick 0 only, before
its first draw).

Reconstructing what can be read exactly is how a layer's model of its own
baseline drifts from what the baseline does.

**`true_subtask` is separately load-bearing, and skipping it broke the degrade
guarantee.** `pi` is a distribution; the realised subtask is a DRAW from it. The
two disagree routinely under a drawing baseline, and under any baseline
whenever within-tier stickiness fires. §3.0 must therefore anchor on the realised
pick, not on `argmax pi` — see §3.0b. Assuming the two were the same is what made
`qmdp-base` diverge from its baseline on 3 of 24 parity cells.

## 3. Candidates: an action mask, and every (cell, arrival tick)

**3.0 The unit is the top `m=3` VERBS, not the top three tuples.** Aggregate
`pi(v) = sum over cells of pi(tier, v, cell)` and take the three largest, ties
broken by the baseline's own order. Two reasons. Taking the top three TUPLES
would re-freeze the cell inside the theta-blind baseline — handoff ranks its
stash candidates by distance from ITSELF, so the counter the human can see may
never be scored. Aggregating first hands each job its ENTIRE legal cell set, and nothing
truncates it afterwards -- which is the whole correction described in
`RESULTS.md` §2b, where a six-cell shortlist taken in the baseline's own
(robot-distance) order held NO counter the human could see on 55% of back_bar's
stash ticks.
And three jobs is where cross-job hedging lives: when the baseline itself has two
near-tied options, that is exactly when the rollouts have something to say.

**3.0b The baseline's REALISED pick leads the job list, ahead of the `pi`
ordering.** The job and cell the baseline actually chose this tick — from the very
`action()` call the layer already makes — is forced to the front, with its own
cell first. `pi` is a distribution and the pick is a DRAW from it, so under a
baseline that draws they disagree routinely; anchoring on `argmax pi` pointed the
fallback at a job the baseline was not doing and broke parity on 3 of 24 cells.
The emitted action still falls back to `base_act` itself, so a flat `Q`
reproduces the nominal policy exactly.

Anchoring on `argmax pi` instead looks equivalent and is not. A drawn subtask
disagrees with the argmax routinely, so the "baseline's own
action" being fallen back to was not the baseline's action, and the
degrade-to-baseline guarantee failed outright — `qmdp-base` diverged on 3 of 24
parity cells until this was fixed. Residual gap: when the baseline is holding a
committed subtask that has dropped out of `ranked`, there is nothing to anchor
to and the `pi` ordering is used (measured: 27 of 29 ticks anchored on bayes,
30 of 30 on the ladders).

**3.1 Execution space.** `G(v)` is the job's legal cells, read off the baseline's
own tuples. `A(x,g)` is the four moves and STAY, always, plus INTERACT only when
`arrived(g)`. The mask is what keeps 3.2 true: an off-target INTERACT would pick
up or drop something no plan meant.

A candidate is `(v, g, a)` and that triple is the entire answer to "how do we
change the robot's action". There are no route objects, no wait objects:

    same job, longer way round      vary a with (v,g) fixed
    same counter, two ticks later   a = STAY
    slightly different counter      vary g with v fixed
    different job altogether        vary v

One-step deviations compose across ticks into whole routes and multi-tick waits,
because the controller re-plans from wherever it lands. The route is never
planned as a route.

**3.2 Termination: the subtask ends at its FIRST INTERACT.** This is not
bookkeeping, it is what makes the score mean anything — see §4.

**3.2b The action mask, which replaced the pair budget.** There is no candidate
list to truncate any more, because INTERACT-legality is a PREDICATE rather than an
enumeration: `A(x)` is the four moves and STAY always, plus INTERACT exactly when
the cell directly in front is a legal target of ANY of the top-`m` jobs. A
predicate costs the same whether a job has three legal cells or fifty-one, so every
stash counter is in scope for free — and the robot can press a counter it PASSES on
the way to another, which the old per-candidate mask (`INTERACT` only on
`arrived(g)`) could never express.

What is searched is then `(cell, arrival tick)`: a shortest-path sweep over
`(position, orientation, tick)` collects every reachable press, and each distinct
`(cell, tick)` pair is scored on its own. `waits` bounds how many arrival ticks per
cell are kept, which is what makes "same counter, two ticks later" a scored plan
rather than a re-plan artifact.

**3.2c Why the sweep dedupes on position and not on the whole state.** Two robot
paths meeting at the same tile at the same tick have swept different cells, so
strictly the human could differ between them and the key would have to include the
shadow. Doing that is exact and useless: nothing merges, the frontier grows ~2.5x
per level (55 states at depth 4, 355 at 6, 2281 at 8, measured), and the search
caps out around depth 6 — which cannot reach a 28-step counter at any price.

`dev/route_channel.py` measures what the exactness buys. Over 720 ticks on all six
layouts, the human SEES the robot on 70.7% of ticks but changing the robot's first
action changes the human's next six actions on **0.8%** of them, and **0.0%** on
divide and pantry. The only channel from robot position to human behaviour is the
contention demotion, and it fires only at stations inside the dividing wall. So the
human is forecast ONCE per tick against the baseline's own trajectory and shared by
every branch, the sweep dedupes on `(position, orientation, tick)`, and depth 30 is
affordable. That is the one approximation in the layer and 0.8% is its measured
size.

**3.3 Human-honest rollouts.** The simulated human is a DEEP COPY OF THE WHOLE
SHADOW: its view with its decay clocks, its held subtask and stickiness, its rng.
Because the copy starts identical, its first simulated action equals the shadow's
prediction for this tick, which is the honesty check (`tests/test_qmdp.py`).
Copying only the view is NOT honest: the ladder holds a subtask across ticks and
a fresh human does not, injecting a systematic first-step error into every
rollout.

**3.4 Transitions are deterministic.** The env transition is deterministic given
the joint action, the ladder is deterministic given theta, and A* breaks ties by
cell coordinate. Each `(v,g,a,theta)` is exactly ONE trace: no sampling, no
variance, bit-for-bit reproducible.

## 4. Value: head in simulation, tail on the ladder graph

    C_theta(v, g, a) = t_end + tail(state at t_end)      ticks, LOWER is better

**The head** runs the real mdp from the forced first action until the FIRST
INTERACT (capped at `T`). It is the only place theta can enter the number:
whether the cone sweeps the stashed plate before FORGET_HORIZON expires the
belief, who reaches a divide-embedded station first, whether a narrow cone burns
a stale round trip.

**Why the head must stop at the INTERACT and not at a fixed `T`.** With `t_end =
T` for every candidate, `C = T + tail(s)` — the ONE tick the action choice
controls is thrown away and has to be recovered from a tail that resolves at 4-8
ticks. Measured with a fixed head, `C` ranked a step directly AWAY from the
target above the greedy step on 86% of ticks on divide. Stopping at the INTERACT
puts a wasted step straight into `t_end`, where it belongs, and took that to 81%
correct.

**The tail** is a bounded A* on the LADDER graph, never the raw state graph: the
true joint state carries both positions, both held items, station contents,
timers and the human's beliefs, and half of every transition is the human's
closed-loop choice, which a search cannot make on their behalf. Nodes are sets of
completed recipe legs plus each agent's (position, free-at); edges are one leg
assigned to one agent OR RELAYED between both through a pass counter; the
objective is makespan. It prices the NEXT dish and adds a per-layout constant for
the rest — searching all remaining dishes multiplies the chain length and SCALES
the jitter by the orders left, while a constant cancels out of every comparison.

**Start node is the head's END CONFIGURATION**, so which counter an item ended on
changes the tail. That is what makes drops and handoffs priceable at all.

**The human's legs are priced through THEIR OWN VIEW.** An item left where their
cone has not been is charged `blind` extra ticks, because they will not collect
what they do not know is there until a sweep finds it. Nobody writes "prefer
visible counters" anywhere; the ordering — known pass counter < blind-spot pass
counter < non-pass counter — falls out of geometry plus the `BeliefView`, and is
asserted in `tests/test_qmdp.py`.

**The no-op-by-algebra property.** A robot-only "ticks to deliver everything"
contains no theta, so the mixture would collapse: `sum_theta p(theta) C = C`, the
posterior cancels out of its own equation, and the layer returns the baseline
action at every belief. If `C_theta` does not depend on theta, this layer is a
no-op by algebra rather than by bug.

## 5. Output: mix, collapse, emit, guard

**No mixture: the MAP cone only.** `Q = C` at the single most likely cone, and
no search at all when `p(MAP) < certainty`. This is not an approximation of a
mixture, it is what the mixture already was. Measured over 18,000 ticks (6 layouts
x 5 true cones x 3 seeds): mean `p(MAP)` is **0.99** and `p(MAP) >= 0.9` on **98.4%**
of ticks, so the old `min_p = 0.05` pruning left a singleton anyway. The gate is
what the sharpness buys -- the MAP cone is right on 98.9% of ticks overall and on
**100.0%** of the ticks where `p(MAP) >= 0.9`, so every cone error the posterior
makes sits inside the 1.6% the gate refuses to act on. Lower Q is better; units are
ticks.

    The caveat is not small: the shadows run the SAME ladder as the human, so this
    is inference against a perfectly specified model and 98.9% is a ceiling rather
    than a performance estimate.

**Collapse jobs and targets.** `Q(a) = MIN over (v,g)`. Min, not average, and
never weighted by `pi(v)`: after taking `a` the robot is free to pursue whichever
job is best from where it lands, and its own job choice is a DECISION, not
epistemic uncertainty. Averaging would score an action that is right for none of
the jobs; multiplying by `pi` would double-count the baseline's preference.
**The theta mixture is a sum because theta is genuinely unknown; the (v,g)
collapse is a min because it is ours to choose. That asymmetry is the whole layer
in one line.**

**Emit** the 6-vector with masked actions at `+inf`, and `softmax(-Q/temperature)`
with the masked ones at zero.

**Three guards.**

(i) *Tie-break follows the baseline's own order* — pair index first (its own pair
at 0), then action index (its greedy step at 0). A flat Q reproduces the nominal
policy EXACTLY. `qmdp-base` is meant to collapse the search to that single
candidate, and is checked against the baseline over 24 cells.

`qmdp-base` collapses the mask to that single cell with `waits=0`, so the
baseline's own plan is the only one available and it must play identically.

    Historical note, because it cost a grid. The previous filter's parity control
    used `pair_k=1`, and `pairs[:max(1, pair_k - 1)] + [must]` cannot return fewer
    than one element, so the must-keep append silently made it TWO candidates and
    the control was never collapsed. It fired on ~0.9% of ticks. Collapsing the
    MASK has no such arithmetic to get wrong.

(ii) *Two switching margins, because the two axes have different resolutions.*
Measured: the spread of `C` across JOBS is ~29 ticks, across the six ACTIONS at a
fixed `(v,g)` it is ~2. One threshold across both makes the layer re-pick its job
every tick by argmin, overriding a baseline that ranks by urgency. So `eps`
guards the `(v,g)` switch and `eps_a` the action. `eps_a = 2.0` because one
wasted step costs ~2 ticks, so an alternative must beat a wasted step to be worth
taking. The committed pair is legality-checked against the WHOLE ranking, not
just the top `m` — otherwise the guard silently finds nothing to hold whenever
the committed job drops out of the top three.

(iii) *Forced re-evaluation passes through untouched.* The margin does not bind
when the job completes at its INTERACT, becomes illegal, or a strictly more
urgent tier arrives. All three fall out of the same line — the commitment is
cleared at INTERACT and looked up in the fresh ranking every tick — rather than
being special-cased.

## 5b. The knobs, and what each one is

`QMDPFilter.__init__`; these are the settled defaults, not placeholders.

| knob | default | what it is |
|---|---|---|
| `m` | 3 | jobs whose cells enter the mask. Three is where cross-job hedging lives. |
| `depth` | 30 | how far the sweep walks. Sized so the FARTHEST legal target in the suite (28 steps, banquet_pass) is reachable; 24-27 on back_bar, chefs_table and pantry. The old `T=25` clipped four of six layouts. |
| `waits` | 2 | arrival ticks scored per cell, earliest first. `waits+1` plans per cell, which is what prices waiting. |
| `certainty` | 0.9 | below this `p(MAP)` the search does not run at all and the baseline acts |
| `eps_a` | 2.0 | action margin, on the ~2-tick axis — one wasted step. The only margin left: `eps` guarded a committed `(v,g)` and there is none. |
| `blind` | 8.0 | ticks charged for a pickup the human has not seen. MODELLING CHOICE. |
| `temperature` | 4.0 | only shapes the emitted distribution, never the argmin |
| `frozen_fov` | None | set to 90 for the theta-blind control |
| `only_base` | False | PARITY control: the mask collapses to the baseline's own realised cell, so its own plan is the only one available |
| `hold_target` | False | re-admit last tick's winning cell while it stays legal anywhere in the ranking. Off by default; the first thing to try if it thrashes. |

Gone with the `(v, g, a)` enumeration: `cell_k`, `pair_k`, `T`, `eps`, `min_p`.
The first two were the truncation `RESULTS.md` §2b is about, `T` became `depth`,
`eps` guarded a commitment that no longer exists, and `min_p` pruned a cone mixture
that the posterior collapses on its own -- `p(MAP)` is 0.99 mean and >= 0.9 on 98.4%
of ticks, so the live set was a singleton anyway.

Two more live in `core/value_tail.py`: `MAX_EXPAND = 1200`, the A* node budget, and
`PASS_K = 4`, the pass counters considered per leg. Both are cost knobs. Every
pass counter with both orderings ran to 44 successors a leg and 130 ms a call,
which at ~36 candidates a tick is nine seconds.

## 6. Known limits

- **Never probes.** No term values a sharper `p(theta)`, so the robot never spends
  a tick disambiguating cones. That is a property of QMDP itself. Probing would
  be a separate, ablatable additive bonus and is not part of this design.
- **One-step deviation per tick.** A detour whose payoff needs the first TWO steps
  to both be non-greedy is invisible until the branch point arrives.
- **Exactly as good as the human model.** Every `C_theta` is a claim about what a
  theta-human would do, so a change to the ladder silently changes every score.
  The filter and the rollouts must always run the same ladder.
- **`blind` is a modelling choice**, not a tuning knob: it asserts a partner will
  eventually sweep past an item they have not seen, at a cost of `blind` ticks.
- **An abandonment tax of about a third.** The layer switches job mid-approach on
  ~33% of its switches against the baselines' 6-21%, throwing away the walking
  already done. It is paid on winning layouts too, so it is not what separates
  success from failure — but it is unexplained and unfixed. `RESULTS.md` §5.

## 7. Superseded

`qmdp_fov.QMDPFilter` re-ranked whole SUB-TASKS and walked one shortest path to
the winner. With the target cell inside the subtask it could only choose among
cells the baseline had already shortlisted, and it could not say "same job,
longer way round" or "same counter, two ticks later". It has been deleted; the
cone inference it shared a module with survives as `core/fov_posterior.py`.
