# QMDP-over-actions filter with a posterior over the human's cone

Sits on top of a nominal policy (`robot/nominal_policy/baselines.py`). The
baseline proposes; this re-orders. Neither baseline changes.

**What this can and cannot do.** It re-ranks the baseline's `top_k` candidates
and nothing else. `HandoffRobot` orders its stash candidates by distance from
itself, so the counters that reach a rollout are the ones nearest the ROBOT — and
the counter the human can actually see may never be among them. Everything below
improves the ordering of a shortlist this file does not choose. Widening or
re-ordering that shortlist is a change to the baseline.

## 1. Posterior over theta

Hidden: the human's cone `theta in {30,60,90,180,360}`, inferred from the human's
ACTIONS alone — the robot never sees the human's beliefs.

The human is a deterministic ladder, so a hypothesised `theta` yields exactly one
predicted action. `FOVPosterior` keeps one shadow `LimitedVisionHuman` per cone,
steps each on the TRUE state every tick so it perceives through its own cone and
maintains its own decaying beliefs, and reads the action off it. Position and
orientation come from the true state each tick, so no shadow can drift away from
where the human actually is.

**Soft-deterministic likelihood.** Do NOT use a hard match. A single mispredicted
action would zero out a hypothesis permanently, which makes the posterior
collapse on the first modelling error and never recover:

    P(a_H | theta) = alpha            if a_H == predicted(theta)      (alpha ~ 0.9)
                   = (1-alpha)/(|A|-1) otherwise

alpha is the trust we place in the human model, not a property of the human.
0.9 keeps the posterior sharp while leaving every hypothesis able to come back
from a surprise. Sequential update over the episode, renormalised each tick.

**There is no separate posterior over `tau`, and no query for what the human can
see.** Given `theta` the ladder is deterministic, so the human's subtask is just
whatever that shadow chose, and a distribution over it carries nothing the cone
posterior does not already carry. `subtask_posterior()` and `seen_by_human()`
used to expose those two things; nothing consumed either, and both are gone —
grep confirms no caller anywhere in the package. The shadows' choices are still
reachable as `FOVPosterior.shadows[f].last_subtask` if a readout ever wants one,
and a cone as `geo.visible_cells(...)`.

What the rollouts DO use the shadows for is `beliefs_for(theta)` — a deep copy of
what a theta-human would currently believe, used to seed the simulated human.
Without it the simulated human starts blind and the rollout is fiction.

## 2. QMDP over ACTIONS, not over subtasks

The decision is made at subtask level — take the baseline's top 3 — but each
rollout must be simulated at the ACTION level, with both agents stepping the
real mdp.

This is not an implementation detail, it is the whole mechanism. **FOV is a
function of orientation, and orientation changes every single step.** Which way
the human happens to turn while walking decides what enters their cone, which
decides what they believe, which decides their next subtask. A rollout that
abstracts movement away and jumps subtask-to-subtask throws away exactly the
coupling the robot is supposed to exploit.

So:

    for each candidate subtask s in baseline.rank_subtasks(state)[:3]:
        for each theta weighted by the posterior:
            clone the state
            drive the robot toward s with the real step_towards
            drive the human with LimitedVisionHuman(theta) — its own cone,
                its own decaying beliefs, its own turns
            step the real mdp to delivery or horizon
            value = discounted return   (see §2b)
        Q(s) = sum_theta P(theta) * value(s, theta)
    act on argmax Q

Ties are broken on the candidate's rank in `baseline.rank_subtasks`, never on the
subtask tuple. Sorting `(-Q, tier, verb, cell)` meant that a flat Q fell through
to alphabetical order — `chop` ahead of `wash` at the same T_WORK tier, even when
the baseline had ranked `wash` first because it was nearer. Keying on the
baseline's index makes a flat Q degrade to exactly the nominal policy, so the
filter can only improve on it, never scramble it.

## 2b. Rollout return — deliveries alone are not enough

`sparse` only ever fires on a delivery, and forty ticks does not carry a raw
onion to the pass. Scoring on it alone returned 0.0 for every candidate on
roughly half of all ticks; Q was flat and the filter was decided entirely by its
tie-break. The per-step return is now

    r = W_DELIVER * sparse                     # 5.0 — a served dish is worth 100
      + W_ENV     * shaped                     # 1.0 — the mdp's own shaping
      + W_PROGRESS * (gamma*Phi(s') - Phi(s))  # 20.0 — recipe progress

`shaped` is currently zero and is wired in for the day it is not: for the steak
recipe the only live terms in `overcooked_mdp` are `PLACEMENT_IN_POT_REW` and
`COOKING_STEAK_REW` (chop, wash, plate, collect, combine are all commented out),
and every layout ships `rew_shaping_params: None`. The steak line needs `Phi`.

`Phi` (`progress.py`) is recipe progress in units of one delivered dish: three
streams — plate, steak, garnish — each scored raw → loaded → timing out → done,
plus credit for each merge. It is used as **potential-based shaping**, not as a
bonus, so over a rollout it telescopes to `gamma^T Phi(s_T)` minus a constant
every candidate shares. It cannot be farmed by shuffling an item back and forth.

The load-bearing property is that **`Phi` is need-aware**: components count only
up to the orders still deliverable (`len(order_list) - 1`, because `is_terminal`
is `<= 1`). A fourth garnish against two remaining orders is worth zero, so the
rollout that chops it scores identically to the rollout that stands still, and
washing a plate wins. The ladder in `common/tasks.py` has no notion of "enough" —
every board with an onion emits a `T_WORK` chop forever — and this is where that
gets corrected, at the level of value rather than legality.

Measured on `pillars`, fov 60: sparse-only served 1/2 in 400 ticks with Q flat on
181 of them; the rich return serves 2/2 in 183 ticks with Q flat on 1.

Cones the posterior has already ruled out (`p < min_p`) are skipped, so this gets
cheaper as the belief sharpens instead of costing the same every tick. If that
leaves nothing, we fall back to the MAP cone alone rather than to no rollout.

## 3. What this buys — the expressive behaviour

When "stash the washed plate" is among the top 3, each free counter that made the
shortlist is its own candidate and gets its own rollout. Under a hypothesised
narrow cone:

- counter INSIDE the inferred cone -> the simulated human sees it on the tick it
  lands, a higher tier becomes legal, they collect it -> high value
- counter in the BLIND SPOT -> the simulated human never looks that way, the
  belief expires after FORGET_HORIZON, the item is never collected -> low value

Same action, different counter, different Q. The robot places the plate where
the human can see it **without anyone writing that rule** — it falls out of
rolling the human's own perception forward.

The same machinery gives unseen assistance for free: if the posterior says the
human cannot see the grill, rollouts where the robot quietly cooks score well
because the human never re-plans around it.

## 4. Contention — closed

*Historical, kept because the reasoning is worth not repeating.* The rollouts
call the SAME human and robot policies used for real play, and both used to
converge on the same station and jam (measured: the human spent 397/400 ticks
holding an onion, unable to reach a contested board). A rollout that livelocks
returns the same near-zero value for every candidate, so Q was flat and the
filter could not discriminate.

Two things closed it, and the order matters because the first diagnosis was
wrong. Flat Q was caused primarily by scoring on `sparse` alone (§2b) — a rollout
that never reaches the pass scores zero whether or not it livelocked, so a jam
and a slow honest rollout were indistinguishable. The progress term made a jammed
rollout visibly worse than a productive one, which is what made contention
*measurable* rather than merely suspected.

The fix itself then went where this section said it should: into the ladder, as a
within-tier demotion of a subtask the teammate would reach first
(`_BaseRobot.rank_subtasks`, and the human's own `rank()`), not into unstick
heuristics. Measured: 732 firings, every one of them on a station embedded in the
divide — the only kind both agents can reach. The two agents were adjacent on 0
of 4800 ticks, so there is no collision handling anywhere in the package and none
is needed.
