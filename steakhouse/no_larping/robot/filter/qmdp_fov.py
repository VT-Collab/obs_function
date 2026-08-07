"""Joint (theta, tau) inference + QMDP-over-actions, sitting on a nominal policy.

See DESIGN.md. Two pieces:

  FOVPosterior   soft-deterministic Bayesian belief over the human's cone,
                 updated from the human's ACTIONS only.
  QMDPFilter     takes the baseline's top-k subtasks, rolls each one out at the
                 ACTION level under every plausible cone, and acts on argmax Q.

The robot never reads the human's beliefs. Everything it knows about what the
human can see comes from having watched them act.
"""
import copy
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action                  # noqa: E402
from common import geometry as geo                                # noqa: E402
from common.tasks import TIER_NAME                                # noqa: E402
from common.views import TruthView                                # noqa: E402
from human.limited_vision_human import LimitedVisionHuman         # noqa: E402

FOVS = (30, 60, 90, 180, 360)
N_ACTIONS = 6


class FOVPosterior:
    """P(theta | a_H_{0:t}) by soft-deterministic likelihood.

    The human is a deterministic ladder, so a hypothesised cone yields ONE
    predicted action. We do not match it hard: a single modelling error would
    zero a hypothesis forever and the posterior could never recover. alpha is
    how much we trust our human model, not a property of the human.
    """

    def __init__(self, mdp, fovs=FOVS, alpha=0.9, human_index=1, seed=0):
        self.mdp, self.fovs, self.alpha = mdp, tuple(fovs), alpha
        self.human_index = human_index
        self.p = {f: 1.0 / len(self.fovs) for f in self.fovs}
        self.shadows = {f: LimitedVisionHuman(mdp, f, agent_index=human_index,
                                              seed=seed) for f in self.fovs}
        self.predicted = {}

    def update(self, state, human_action):
        """Advance every shadow on the TRUE state, then reweight by agreement.

        Each shadow observes the real world through its own cone, so its beliefs
        stay honest; position and orientation are read from the true state each
        tick, so no shadow can drift away from where the human actually is.
        """
        self.predicted = {}
        for f, sh in self.shadows.items():
            act, _ = sh.action(state)
            self.predicted[f] = act

        miss = (1.0 - self.alpha) / (N_ACTIONS - 1)
        total = 0.0
        new = {}
        for f in self.fovs:
            lik = self.alpha if self.predicted[f] == human_action else miss
            new[f] = self.p[f] * lik
            total += new[f]
        if total <= 0:
            new = {f: 1.0 / len(self.fovs) for f in self.fovs}
            total = 1.0
        self.p = {f: v / total for f, v in new.items()}
        return self.p

    def subtask_posterior(self):
        """P(tau) -- free, because tau is a deterministic function of theta."""
        out = {}
        for f, prob in self.p.items():
            sub = self.shadows[f].last_subtask
            if sub is None:
                continue
            out[sub] = out.get(sub, 0.0) + prob
        return out

    def map_fov(self):
        return max(self.p, key=self.p.get)

    def beliefs_for(self, fov):
        """A copy of what a theta-human would currently believe. Seeds rollouts;
        without it the simulated human starts blind and the rollout is fiction."""
        return copy.deepcopy(self.shadows[fov].view)


class QMDPFilter:
    """Re-orders the baseline's top-k subtasks by rolling each one forward.

    Rollouts are at ACTION level, stepping the real mdp, with the simulated human
    running its own cone and its own decaying beliefs. That is essential rather
    than fussy: the human's cone follows its ORIENTATION, which changes every
    step, so which way it happens to turn while walking decides what it sees and
    therefore what it does next. A subtask-level rollout would abstract away the
    exact coupling the robot is meant to exploit.
    """

    def __init__(self, mdp, baseline, posterior, top_k=3, horizon=40,
                 gamma=0.98, min_p=0.05, agent_index=0):
        self.mdp, self.baseline, self.post = mdp, baseline, posterior
        self.top_k, self.horizon, self.gamma, self.min_p = top_k, horizon, gamma, min_p
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.last_subtask = None
        self.last_Q = {}
        self._last_pos, self._stuck = None, 0
        import random
        self._rng = random.Random(7)

    def _joint(self, robot_act, human_act):
        return (robot_act, human_act) if self.agent_index == 0 else (human_act, robot_act)

    # -- rollout ------------------------------------------------------------
    def _rollout(self, state, cell, fov):
        """Discounted deliveries if we commit to `cell` against a fov-human.

        The simulated human is seeded with what a theta-human would ALREADY
        believe, then left to run on its own perception. Its orientation changes
        as it walks, so the cone sweeps different cells in different rollouts -
        which is precisely how a counter inside the cone comes out worth more
        than one in the blind spot.
        """
        sim = state.deepcopy()
        human = LimitedVisionHuman(self.mdp, fov, agent_index=self.other_index)
        human.view = self.post.beliefs_for(fov)
        shadow_bot = type(self.baseline)(self.mdp, agent_index=self.agent_index)
        target, value, disc = cell, 0.0, 1.0

        for _ in range(self.horizon):
            if self.mdp.is_terminal(sim):
                break
            me = sim.players[self.agent_index]
            pos, orient = tuple(me.position), tuple(me.orientation)
            other = tuple(sim.players[self.other_index].position)

            if target is not None:                  # committed leg
                walk = TruthView(self.mdp, sim).walkable | {pos}
                mv, arrived = geo.step_towards(walk, pos, orient, target, {other})
                if arrived:
                    ar, target = Action.INTERACT, None   # then hand back to the baseline
                else:
                    ar = mv or Action.STAY
            else:
                ar, _ = shadow_bot.action(sim)

            ah, _ = human.action(sim)
            sim, sparse, _, _ = self.mdp.get_state_transition(sim, self._joint(ar, ah))
            value += disc * float(sum(sparse))
            disc *= self.gamma
        return value

    # -- decision -----------------------------------------------------------
    def rank(self, state):
        """[(Q, tier, verb, cell)] best first, over the baseline's top-k."""
        cands = self.baseline.rank_subtasks(state)[:self.top_k]
        live = {f: p for f, p in self.post.p.items() if p >= self.min_p}
        if not live:
            live = {self.post.map_fov(): 1.0}
        norm = sum(live.values())

        out = []
        for tier, verb, cell in cands:
            q = 0.0
            for fov, p in live.items():
                q += (p / norm) * self._rollout(state, cell, fov)
            out.append((-q, tier, verb, cell))
        out.sort()
        self.last_Q = {(v, c): -q for q, _, v, c in out}
        return [(-q, t, v, c) for q, t, v, c in out]

    def action(self, state):
        ranked = self.rank(state)
        if not ranked:
            return Action.STAY, {"subtask": None, "fov_post": dict(self.post.p)}
        q, tier, verb, cell = ranked[0]
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        other = tuple(state.players[self.other_index].position)
        walk = TruthView(self.mdp, state).walkable | {pos}
        mv, arrived = geo.step_towards(walk, pos, orient, cell, {other})
        act = Action.INTERACT if arrived else (mv or Action.STAY)

        if act is not Action.INTERACT and pos == self._last_pos:
            self._stuck += 1
            if self._stuck >= 2:
                act = geo.sidestep(walk, pos, {other}, self._rng) or Action.STAY
                self._stuck = 0
        else:
            self._stuck = 0
        self._last_pos = pos
        return act, {"subtask": self.last_subtask, "Q": q,
                     "fov_post": dict(self.post.p)}

    def set_agent_index(self, i):
        self.agent_index, self.other_index = i, 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
