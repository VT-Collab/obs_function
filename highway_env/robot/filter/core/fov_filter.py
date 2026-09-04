"""The FOV filter. No task knowledge.

Mirrors steakhouse/misha/robot/filter/core/my_fov_filter.py's actual
mechanism -- for each legal FIRST action, roll a shadow forward some depth
and score how much of that rollout keeps the robot seen (or unseen, if
weight_seen < 0) by the human, weighted by the CURRENT belief over which FOV
hypothesis is true -- not the old (reverted) design here, which was just a
lookup-table cut to baseline_accel proportional to E[caution]. That version
never rolled anything forward, never scored an actual number of ticks seen,
and could only ever brake, never accelerate away or hold steady when holding
steady is what best manages visibility -- none of which matches the
reference's own rollout-and-score structure.

WHAT CARRIES OVER FROM THE REFERENCE, ADAPTED:
  - baseline's own proposed action is the anchor every candidate is
    generated around and constrained against (there: the BFS/subtask
    search still had to stay within budget_mult of the baseline's own
    target time; here: a candidate accel outside [-max_extra_decel,
    +max_extra_accel] of the baseline accel is never even generated).
  - roll a "shadow" of the other agent forward under the SAME first action,
    tracking whether the robot is seen each tick (there: human1.clone()
    plus a BFS-toward-cell robot rollout; here: straight-line kinematics
    for both sides -- there is no grid/cell/subtask structure in this
    domain, just one continuous scalar to pick and one fixed route to keep
    driving, so the reference's multi-CELL candidate search collapses to a
    multi-ACCEL candidate search).
  - _score's sign convention: weight_seen multiplies a NEGATIVE contribution
    to a score that gets MINIMIZED, so weight_seen > 0 rewards being seen
    (more seen -> lower score -> more likely to win) and weight_seen < 0
    rewards NOT being seen, exactly the reference's own convention.
  - certainty gating and a bounded deviation from baseline both carry over;
    the reference's "committed cell" stickiness does not -- there is only
    ever one committed thing here (ROBOT_ROUTE itself), so nothing plays
    the role a flip-flopping cell choice would.

WHAT'S DELIBERATELY A STAND-IN FOR NOW (see class docstring): the human's
own rollout just holds its CURRENT acceleration constant. The FOV posterior
and the human model are both about to change shape for the user-study work
(discrete subtask commands instead of continuous IDM) -- this filter's own
rollout/scoring machinery is written against the STABLE interface both
already expose (positions/headings/speeds, a scalar acceleration, an
FOVPosterior.beliefs() dict) so it doesn't need to change when they do.
"""
import numpy as np

from common.geometry import in_cone, is_occluded

DEFAULT_DT = 1 / 15


class FOVFilter:
    """Pick the robot's accel THIS tick that best manages how much of a
    short rollout keeps it inside (weight_seen > 0) or outside
    (weight_seen < 0) the human's FOV, subject to never deviating from the
    baseline's own proposed accel by more than [max_extra_decel,
    max_extra_accel].

    weight_seen: reward (per tick, averaged over the rollout) for being
        seen. Positive -> stay visible; negative -> stay hidden; 0 ->
        ignore visibility entirely and this filter is a no-op (always
        returns baseline_accel exactly, since every candidate then scores
        identically and the tie_break term picks the one closest to it).
    depth_s / dt: rollout horizon in seconds, converted to ticks at this
        dt (matches every other STALL_TIMEOUT_S/dt-style duration
        elsewhere in this codebase). Default 4.0s.
    max_extra_accel / max_extra_decel: candidates only ever range over
        [baseline - max_extra_decel, baseline + max_extra_accel] -- the
        hard budget constraint, generated in rather than checked after
        (nothing outside this range is ever a candidate at all).
    n_candidates: how many accels to sample across that range (odd, so
        baseline_accel itself -- the midpoint -- is always one of them;
        the filter can therefore never do WORSE on the visibility score
        than just proposing the baseline unchanged).
    tie_break_weight: small penalty on |candidate - baseline|, just large
        enough to prefer the smallest deviation among otherwise-tied
        candidates (e.g. a robot far enough away that no candidate changes
        whether it's seen at all -- see module/class docstring's own
        worked example) without ever overriding a real difference in the
        visibility score.
    enable_occlusion: whether the rollout also checks line-of-sight against
        `ctx.occluders` (other nearby vehicles, held FIXED at their CURRENT
        positions for the whole rollout -- a short-horizon approximation;
        this filter has no way to predict anyone else's future action
        without becoming task-aware about them too). Silently skipped if
        the caller's ctx doesn't carry `occluders` at all.
    """

    def __init__(self, weight_seen=1.0, depth_s=4.0, dt=DEFAULT_DT,
                 max_extra_accel=1.5, max_extra_decel=3.0, n_candidates=7,
                 tie_break_weight=0.02, enable_occlusion=True, certainty=0.0):
        self.weight_seen = weight_seen
        self.depth = max(1, round(depth_s / dt))
        self.dt = dt
        self.max_extra_accel = max_extra_accel
        self.max_extra_decel = max_extra_decel
        self.n_candidates = n_candidates
        self.tie_break_weight = tie_break_weight
        self.enable_occlusion = enable_occlusion
        self.certainty = certainty  # 0.0 -- unlike the reference, this always searches; see action()'s own gate

    def _rollout_seen_fraction(self, robot_state, robot_accel, human_state, human_accel,
                                belief, occluders):
        """Straight-line-kinematics rollout of both vehicles for self.depth
        ticks (constant heading, constant accel each -- see module
        docstring on why: no lane/route knowledge belongs in this file),
        returning the belief-weighted average fraction of ticks the robot
        is visible to the human. Positions integrate independently of any
        lane geometry -- fine over a few seconds, and the only way this
        stays task-free.

        Vectorized across all `depth` ticks at once (per FOV hypothesis) --
        confirmed as the second-largest per-tick cost anywhere in the whole
        interactive interface: 841,297 individual in_cone() calls across
        just 200 real ticks with 2 robots (depth=60 ticks x 7 accel
        candidates x 5 FOV hypotheses x 2 robots), each recomputing its own
        cos/sin/norm from scratch as a separate Python function call. The
        speed/position rollout itself vectorizes exactly (constant-accel
        motion with a floor at 0 is a plain running max/cumsum -- verified
        term-by-term against the original per-tick loop, including the
        "once floored at 0, stays there under constant non-positive accel"
        behavior, which falls out of np.maximum on a monotonic linear
        sequence automatically). is_occluded() itself is untouched (still
        the plain per-tick, per-hypothesis function) and, exactly as
        before, is only ever called for a tick that already passed the
        in_cone check -- only the in_cone side, which dominated the raw
        call count, needed vectorizing.
        """
        rx0, ry0, rh, rs0 = robot_state
        hx0, hy0, hh, hs0 = human_state
        dt = self.dt
        ticks = np.arange(1, self.depth + 1, dtype=float)

        rs = np.maximum(0.0, rs0 + robot_accel * dt * ticks)
        hs = np.maximum(0.0, hs0 + human_accel * dt * ticks)
        rx = rx0 + np.cos(rh) * dt * np.cumsum(rs)
        ry = ry0 + np.sin(rh) * dt * np.cumsum(rs)
        hx = hx0 + np.cos(hh) * dt * np.cumsum(hs)
        hy = hy0 + np.sin(hh) * dt * np.cumsum(hs)

        dx, dy = rx - hx, ry - hy
        dist = np.sqrt(dx * dx + dy * dy)
        cos_angle = (dx * np.cos(hh) + dy * np.sin(hh)) / np.maximum(dist, 1e-6)
        near_zero = dist < 1e-6  # vacuously "in cone" -- matches in_cone()'s own guard

        seen_per_tick = np.zeros(self.depth)
        for fov, p in belief.items():
            if p <= 0.0:
                continue
            if fov >= 360:
                in_cone_mask = np.ones(self.depth, dtype=bool)
            else:
                in_cone_mask = near_zero | (cos_angle >= np.cos(np.radians(fov / 2.0)) - 1e-9)
            if self.enable_occlusion and occluders:
                for k in np.flatnonzero(in_cone_mask):
                    if is_occluded((hx[k], hy[k]), (rx[k], ry[k]), occluders):
                        in_cone_mask[k] = False
            seen_per_tick += p * in_cone_mask
        return float(seen_per_tick.sum() / self.depth)

    def action(self, ctx):
        """ctx: the same namespace every baseline reads -- .robot, .human,
        .belief (FOVPosterior.beliefs() dict, or None), .front_vehicle,
        .dt, and optionally .occluders (nearby vehicles for the rollout's
        own occlusion check; treated as empty if absent) and
        .crossing_conflict_accel (a hard deceleration ceiling from the
        caller's own scene1_background.crossing_conflict_brake() check --
        see below; treated as no constraint if absent). Falls back to
        the plain baseline accel whenever there's no human/belief to
        reason about, or the belief isn't confident enough yet (self.
        certainty) -- same as every other baseline here, this never
        invents a decision the underlying IDM computation didn't already
        make; it only ever chooses AMONG a bounded neighborhood of it.

        crossing_conflict_accel FOLDED IN HERE, not applied as an external
        min() by the caller afterward -- confirmed as a real, not
        theoretical, difference: a caller-side `min(baseline_result,
        filter_result)` against apply_better_car_following's OWN
        (separately, redundantly recomputed) blended value means ANY tick
        crossing_conflict_brake is active for the robot -- which, in the
        dense traffic this whole module is meant to run under, is often --
        THAT value wins outright regardless of what candidate this filter
        would have picked, so its own visibility preference never gets a
        chance to matter on exactly the ticks proximity to the human makes
        it most relevant. Folding the SAME constraint in as a ceiling on
        every candidate instead (rather than overriding the winner after
        the fact) means: when the conflict is severe enough that EVERY
        candidate gets clipped to it, the outcome is identical (a real
        emergency stop still overrides any FOV preference, as it must);
        when there's genuine room within the constraint, the filter can
        still express a real preference among the safe options instead of
        the whole tick being silently decided by someone else's already-
        computed number.
        """
        base = ctx.robot.acceleration(ego_vehicle=ctx.robot, front_vehicle=ctx.front_vehicle, rear_vehicle=None)
        conflict = getattr(ctx, "crossing_conflict_accel", None)
        if conflict is not None:
            base = min(base, conflict)
        if ctx.belief is None or ctx.human is None or ctx.human.crashed or self.weight_seen == 0.0:
            return base
        if self.certainty > 0.0 and max(ctx.belief.values(), default=0.0) < self.certainty:
            return base

        robot_state = (ctx.robot.position[0], ctx.robot.position[1], ctx.robot.heading, ctx.robot.speed)
        human_state = (ctx.human.position[0], ctx.human.position[1], ctx.human.heading, ctx.human.speed)
        human_accel = ctx.human.action.get("acceleration", 0.0) if isinstance(ctx.human.action, dict) \
            else getattr(ctx.human.action, "acceleration", 0.0)
        occluders = getattr(ctx, "occluders", ())

        candidates = np.linspace(base - self.max_extra_decel, base + self.max_extra_accel, self.n_candidates)
        if conflict is not None:
            candidates = np.minimum(candidates, conflict)
        best_a, best_score = base, None
        for a in candidates:
            seen_frac = self._rollout_seen_fraction(robot_state, float(a), human_state, human_accel,
                                                      ctx.belief, occluders)
            score = -self.weight_seen * seen_frac + self.tie_break_weight * abs(a - base)
            if best_score is None or score < best_score:
                best_score, best_a = score, float(a)
        return best_a
