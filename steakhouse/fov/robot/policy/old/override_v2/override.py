"""
FULL-AUTHORITY FOV OVERRIDE (override_v2).

The baseline (frozen, FOV-blind) only SUGGESTS a task. Given the FOV posterior +
entropy, this module may OVERRIDE ANYTHING - the task, the path, both - or defer.
Deployment-time only: the trained policy is never retrained, so any measured
difference is attributable to FOV information.

AUTHORITY IS GATED BY ENTROPY
  unsure (high entropy)   -> defer to the baseline
  confident (low entropy) -> override by the inferred angle:
      fov <= blind_fov_max     -> TAKEOVER: cook the pipeline yourself (a blind
                                  human cannot find stations; leaving them jobs
                                  wastes time)
      fov >= sighted_fov_min   -> VISIBLE: act at a station INSIDE the inferred
                                  cone and hug that cone en route, so the human
                                  sees the change and yields / stops double-working
      in between               -> defer

NO CHEATING. The cone is computed by reusing the human's own visible() test, fed
the INFERRED angle (map_fov) + the human's OBSERVED pose - never the true FOV.

Thresholds are ANGLES so they read directly against the candidate set
[30,60,90,120,180,360]: default blind<=90, sighted>=120 (so 120/180/360 are
"sighted", 30/60/90 are "blind", nothing is left in the defer band by default).
"""
import math
from collections import deque

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction
from steakhouse.fov.robot.policy.old.baseline.features import ACTIONS

WORK_IDX = ACTIONS.index("work")


def confidence(entropy, n_candidates):
    """1 = posterior collapsed to one FOV, 0 = still uniform / clueless."""
    return max(0.0, 1.0 - entropy / math.log(max(2, n_candidates)))


class ConeReroute:
    """TRAJECTORY authority: bias the robot's next step toward the human's
    INFERRED cone. `knob` = the extra path-cost (in steps) the robot will pay to
    be in view; knob<=0 disables it (identity)."""

    def __init__(self, inf, mdp, robot_index=0, human_index=1, knob=3.0):
        self.inf, self.mdp = inf, mdp
        self.ri, self.hi = robot_index, human_index
        self.knob = knob
        self._valid = set(mdp.get_valid_player_positions())

    def in_cone(self, state, loc):
        """Is grid cell `loc` inside the human's INFERRED cone right now? Uses the
        shadow human at fov=map_fov, reading the REAL human's pose from `state`."""
        sh = self.inf.shadows.get(self.inf.map_fov())
        if sh is None:
            return False
        try:
            return bool(sh.visible(state, tuple(loc)))
        except Exception:
            return False

    def _dist_map(self, targets):
        """BFS step-distance from every reachable cell to the nearest standing
        cell adjacent to any target station."""
        goals = set()
        for g in targets:
            for d in Direction.ALL_DIRECTIONS:
                stand = (g[0] - d[0], g[1] - d[1])
                if stand in self._valid:
                    goals.add(stand)
        dist = {c: 1e9 for c in self._valid}
        dq = deque()
        for c in goals:
            dist[c] = 0.0
            dq.append(c)
        while dq:
            c = dq.popleft()
            for d in Direction.ALL_DIRECTIONS:
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in dist and dist[nb] > dist[c] + 1:
                    dist[nb] = dist[c] + 1
                    dq.append(nb)
        return dist

    def choose(self, state, targets, base_move):
        """Score each immediate move by -(steps still to go) + knob*(in cone?).
        The robot drifts into view only when the detour is worth <= knob steps.
        knob<=0 or already interacting -> base_move unchanged (clean control)."""
        if self.knob <= 0 or not targets or base_move == Action.INTERACT:
            return base_move
        dist = self._dist_map(targets)
        p = state.players[self.ri]
        best_move, best_score = base_move, -1e18
        for move in list(Direction.ALL_DIRECTIONS) + [Action.STAY]:
            if move == Action.STAY:
                npos = p.position
            else:
                cand = (p.position[0] + move[0], p.position[1] + move[1])
                npos = cand if cand in self._valid else p.position   # blocked: turn in place
            ctg = dist.get(npos, 1e9)
            if ctg >= 1e9:
                continue
            score = -ctg + self.knob * (1.0 if self.in_cone(state, npos) else 0.0)
            if score > best_score:
                best_move, best_score = move, score
        return best_move


class FOVOverride:
    """TASK authority: map the baseline's suggested task index to a decision.
    Returns (kind, idx) where kind in {"defer","takeover","visible"}; on "defer"
    idx is the baseline's own task, otherwise idx is None (the env realises the
    takeover / visible behaviour itself)."""

    def __init__(self, inf, conf_min=0.4, blind_fov_max=90, sighted_fov_min=120):
        self.inf = inf
        self.conf_min = conf_min
        self.blind_fov_max = blind_fov_max
        self.sighted_fov_min = sighted_fov_min

    def confident(self):
        return confidence(self.inf.entropy(),
                          len(self.inf.candidate_fovs)) >= self.conf_min

    def decide(self, baseline_idx):
        mf = self.inf.map_fov()
        if mf is None or not self.confident():
            return "defer", baseline_idx
        if mf <= self.blind_fov_max:
            return "takeover", None
        if mf >= self.sighted_fov_min:
            return "visible", None
        return "defer", baseline_idx
