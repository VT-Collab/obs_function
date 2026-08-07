# QMDP-over-actions filter with joint (theta, tau) inference

Sits on top of a nominal policy (`robot/nominal_policy/baselines.py`). The
baseline proposes; this re-orders. Neither baseline changes.

## 1. Joint posterior over (theta, tau)

Hidden: the human's cone `theta in {30,60,90,180,360}` and their current subtask
`tau`. Both are inferred from the human's ACTIONS alone — the robot never sees
the human's beliefs.

The human is a deterministic ladder, so for a hypothesised `theta` we can replay
its `BeliefView` over the observed history and ask what action it *would* have
emitted. That gives an exact predicted action per hypothesis.

**Soft-deterministic likelihood.** Do NOT use a hard match. A single mispredicted
action would zero out a hypothesis permanently, which makes the posterior
collapse on the first modelling error and never recover:

    P(a_H | theta) = alpha            if a_H == predicted(theta)      (alpha ~ 0.9)
                   = (1-alpha)/(|A|-1) otherwise

alpha is the trust we place in the human model, not a property of the human.
0.9 keeps the posterior sharp while leaving every hypothesis able to come back
from a surprise. Sequential update over the episode, renormalised each tick.

`tau` comes along for free: given `theta`, the ladder's top choice IS the
subtask, so `P(tau | history) = sum_theta P(theta | history) [tau = ladder(theta)]`.
That is the closed-form distribution the robot needs, with no epsilon fudge
anywhere except the deliberate alpha above.

## 2. QMDP over ACTIONS, not over subtasks

The decision is made at subtask level — take the baseline's top 3 — but each
rollout must be simulated at the ACTION level, with both agents stepping the
real mdp.

This is not an implementation detail, it is the whole mechanism. **FOV is a
function of orientation, and orientation changes every single step.** Which way
the human happens to turn while walking決 decides what enters their cone, which
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
            value = discounted deliveries
        Q(s) = sum_theta P(theta) * value(s, theta)
    act on argmax Q

## 3. What this buys — the expressive behaviour

When "stash the washed plate" is among the top 3, the candidate expands to one
rollout per free counter. Under a hypothesised narrow cone:

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

## 4. Prerequisite — fix agent contention first

The rollouts call the SAME human and robot policies used for real play. Both
currently converge on the same station and jam at chokepoints (measured: the
human spent 397/400 ticks holding an onion, unable to reach a contested board).

A rollout that livelocks returns the same near-zero value for every candidate,
so Q is flat and the filter cannot discriminate. **The filter is untestable
until contention is fixed**, and the fix belongs in the ladder — demote a
subtask the teammate is visibly committed to — not in more unstick heuristics.
