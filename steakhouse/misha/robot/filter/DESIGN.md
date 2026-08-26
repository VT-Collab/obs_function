# FOVFilter -- a FoV-only layer, no task knowledge at all

`robot/filter/core/fov_filter.py`. Distilled from that file's own (extensive)
docstrings -- treat this as a map, not a replacement; the file itself is the
source of truth and carries the proofs.

`RESULTS.md` has the cap=8 measurements. This file is the design and the reasons.

**This is not the filter `robot/methods.py` currently wires up.** Every
`fov-*` row builds `my_fov_filter.py`'s `FOVFilter` now, not this one -- see
`README.md`'s `core/` table and `RESULTS_my_fov_filter.md` for what's actually
running today. This document and `RESULTS.md` describe `core/fov_filter.py`
as a design that stands on its own, not as documentation of current behavior.

## 0. What it is not

`qmdp.py` is a second scheduler: it re-derives what the kitchen needs (the
recipe leg, the fetch count, a stash's handoff) and argues with the baseline
about it, so every number it emits mixes "the human should see me" with "this
is the better job." `FOVFilter` keeps exactly two things from that lineage --
the certainty gate on the cone posterior, and the baseline's own sub-task
distribution -- and deletes everything between them. In their place, one
question:

    HOW OFTEN AND HOW SOON, OVER THE TICKS OF DOING THIS SUB-TASK, WOULD THE
    HUMAN SEE ME -- AND IS THE ANSWER WORTH WHAT IT COSTS THE BASELINE?

and one cost:

    C(plan) = R(cell) + D(action, cell) - bonus(plan)

`R` is the sub-task regret in ticks against the baseline's realised draw, `D`
is the detour this first action adds over the shortest route to the cell, and
`bonus` prices being seen. The nominal policy owns task success; this layer
owns only being seen.

## 1. The bonus: a per-sighting price, not a capped fraction

    bonus(plan) = cap * SUM_{j=1..n} decay**(j-1) * gamma**(k_j - 1)

where `k_1 < ... < k_n` are the ticks of the plan's own trajectory on which the
human's cone covers the robot's position (`common/views.py`: `view.robot` is
written iff the robot's cell is in the human's `seen_cells` that tick -- this is
literally the human's belief-refresh signal, not an analogy for it).

    cap     what ONE sighting is worth, in ticks of baseline cost. A plan seen
            once on tick 1 is worth exactly `cap` (decay**0 * gamma**0 == 1).
    decay   in (0, 1). What the (j+1)-th sighting is worth relative to the j-th:
            the human's belief is refreshed by seeing the robot at all, so a
            second look tells them less than the first did, whenever it happens.
    gamma   the OPEN-LOOP time discount (0.9). The forecast human is simulated
            without closing the loop on the robot's future moves, so a sighting
            predicted further out deserves less trust.
    max_influence = cap / (1 - decay)   THE bound, not `cap`. What the whole
            plan can ever be worth, summing the geometric tail to infinity.
            At the defaults (cap=4, decay=0.5) this is 8 ticks.

Two different decays, not one: `gamma**k` distrusts a sighting for being LATE;
`decay**j` distrusts it for being a REPEAT. Both apply, independently, to every
term.

**Why a per-sighting price and not a capped fraction of trajectory length.**
The file used to score `s in [0,1]` -- discounted sightings over discounted
trajectory length, with `cap` as the total budget `s` spent a fraction of.
Two problems: the tenth sighting was worth as much as the first (so "seen once"
and "not seen at all" scored nearly the same), and the same physical sighting
was worth a different amount depending on how long the job around it happened
to be (a 2-tick trajectory seen twice scored the same `s` as a 12-tick one seen
eight times). The per-sighting price fixes both: the FIRST sighting sets the
price, decay is what a REPEAT is worth relative to it, and there is no
denominator left to make one sighting's value depend on an unrelated job's
length.

## 2. Three guards, three different units

**(i) Tier lock.** At or below one rung of the tier ladder (`max_influence <=
21.4` ticks) the FoV term provably cannot move the robot to a different TIER of
job, on any baseline, on any layout. Above a rung it can. `cap=8` gives
`max_influence = 16` -- inside the tier-safe band, but double the file's stated
"intended" budget of 2-8 ticks (cap 1-4).

**(ii) Switching margin (`eps_a`).** The committed target is held unless a
challenger beats it by more than `eps_a` ticks, so the layer does not re-pick
its plan every tick by argmin over noise.

**(iii) Deviation budget (`stall_max`, leaky).** The per-plan bonus resets every
tick -- a receding-horizon controller forgets its own accumulated detour, so
"one more tick of loitering, cheap" is available forever, and decay does not
fix this because it decays sightings WITHIN one plan while every tick starts a
fresh one. `self.debt` increments on a deviating tick and decays back down by 1
on a non-deviating one; at `debt >= stall_max` the baseline's action is forced
for one tick. This is the one place in the file where something drains back
down over time rather than shrinking from a peak -- the opposite direction from
the within-plan `decay` term, and the closest thing here to "reverse decay" if
that phrase ever gets a precise meaning. It is already active by default
(`stall_max=6`) and was not touched for the cap=8 grid.

## 3. Knobs

| knob | default | what it is |
|---|---|---|
| `cap` | 4.0 | ticks the layer will pay for ONE sighting. **Set to 8.0 for this grid.** |
| `fov_decay` | 0.5 | in (0,1). What a repeat sighting is worth relative to the last. **Left at default -- not touched for this grid.** |
| `gamma` | 0.9 | open-loop time discount |
| `depth` | 12 | how far the forecast window runs |
| `m` | 3 | candidate jobs considered |
| `certainty` | 0.9 | below this `p(MAP)`, no search at all |
| `stall_max` | 6 | consecutive deviating ticks before one forced baseline tick |
| `frozen_fov` | None | set to 90 for the theta-blind control |

## 4. Known limits (inherited from the file's own docstrings)

- **No controls in this grid.** `fov-base` (cap=0 parity check) and `fov-fixed`
  (theta-blind, cone frozen at 90) were not run alongside cap=8. Both exist in
  `robot/methods.py` and cost nothing to add to the next grid.
- **cap and its bound are easy to misquote.** `max_influence = cap/(1-decay)`
  is what every safety property is actually stated in terms of; quoting `cap`
  alone is off by a factor of `1/(1-decay)` (2x at the default decay).
- **Never probes.** No term values a sharper `p(theta)`; the layer never spends
  a tick disambiguating cones.
- **The tier lock is a ceiling, not a floor on quality.** Staying inside one
  rung stops the layer from picking a worse TIER of job; it says nothing about
  whether spending the budget it does have is worth it, which is exactly the
  question `RESULTS.md` answers for cap=8.

## 5. Superseded

`qmdp_fov.QMDPFilter` (deleted) re-ranked whole sub-tasks and walked one
shortest path to the winner; the cone inference it shared a module with
survives as `fov_posterior.py`. `qmdp.py`'s enumerate-`(v,g,a)` predecessor is
also deleted; `no_larping/robot/filter/RESULTS.md` section 9 has what replaced
it and why.
