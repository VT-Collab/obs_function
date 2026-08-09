"""Inference over the human's cone + QMDP-over-actions, on top of a nominal policy.

DESIGN.md next to this file is the long version and is current. Two pieces:

  FOVPosterior   soft-deterministic Bayesian belief over theta, the human's cone,
                 updated from the human's ACTIONS only.
  QMDPFilter     takes the baseline's top-k subtasks, rolls each one out at the
                 ACTION level under every plausible cone, and acts on argmax Q.

The robot never reads the human's beliefs. Everything it knows about what the
human can see comes from having watched them act. There is no separate posterior
over the human's SUBTASK and no "can they see this cell" query: given theta the
ladder is deterministic, so tau is whatever that shadow chose. The two functions
that once exposed them (subtask_posterior, seen_by_human) are gone -- nothing
called either, and a readout nothing consumes is a claim nothing tests.

THE CEILING, STATED HONESTLY. This filter can only RE-RANK the candidates the
baseline hands it, and it sees only the first `top_k` of them. HandoffRobot
orders its stash candidates by DISTANCE FROM ITSELF, so the counters that reach
the rollout stage are the three nearest to the robot -- and the counter the human
can actually SEE may not be among them, in which case no amount of rollout can
choose it. Everything below improves the ordering of a shortlist it does not
control; widening or re-ordering that shortlist is a change to the baseline, not
to this file.
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
from robot.filter.progress import potential                       # noqa: E402

FOVS = (30, 60, 90, 180, 360)
N_ACTIONS = 6

# Rollout reward = W_DELIVER * sparse + W_ENV * shaped + W_PROGRESS * potential.
#
# A delivery pays delivery_reward=20, so W_DELIVER=5 makes one served dish worth
# 100 while a whole dish's worth of UNSERVED progress tops out at 0.91 * 20 ~ 18.
# That ordering is the point: five-to-one means the robot will never dawdle
# assembling a second dish when it could serve the one in its hands, but among
# the candidates that reach no pass within the horizon -- which is most of them
# at depth 40 -- progress is the only thing separating them.
#
# W_ENV rides the mdp's own shaped_reward. It is zero in practice (see
# progress.py) and is here so that building the mdp with BASE_REW_SHAPING_PARAMS
# starts paying without another edit.
W_DELIVER = 5.0
W_ENV = 1.0
W_PROGRESS = 20.0


class FOVPosterior:
    """P(theta | a_H_{0:t}) by soft-deterministic likelihood.

    The human is a deterministic ladder, so a hypothesised cone yields ONE
    predicted action. We do not match it hard: a single modelling error would
    zero a hypothesis forever and the posterior could never recover. alpha is
    how much we trust our human model, not a property of the human.

    One shadow LimitedVisionHuman per cone does the predicting. They are the real
    agent, not an approximation of it, which is why this is exact rather than
    fitted -- and it is also why the whole thing is only as good as that human
    model: a change to the ladder changes what every hypothesis predicts.
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

    RE-ORDERS, never invents. The candidate set is the baseline's own top-k and
    nothing else goes in, so with a flat Q this degrades to exactly the nominal
    policy -- it can improve on the baseline and cannot scramble it. The cost is
    the ceiling in the module docstring: an option the baseline never proposed
    cannot be chosen here however good it would have been.
    """

    def __init__(self, mdp, baseline, posterior, top_k=3, horizon=40,
                 gamma=0.98, min_p=0.05, agent_index=0,
                 w_deliver=W_DELIVER, w_env=W_ENV, w_progress=W_PROGRESS):
        self.mdp, self.baseline, self.post = mdp, baseline, posterior
        self.top_k, self.horizon, self.gamma, self.min_p = top_k, horizon, gamma, min_p
        # w_progress=0, w_env=0, w_deliver=1 reproduces the old sparse-only
        # scoring exactly -- keep that A/B available, it is the control.
        self.w_deliver, self.w_env, self.w_progress = w_deliver, w_env, w_progress
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.last_subtask = None
        self.last_Q = {}
        import random
        self._rng = random.Random(7)

    def _joint(self, robot_act, human_act):
        return (robot_act, human_act) if self.agent_index == 0 else (human_act, robot_act)

    # -- rollout ------------------------------------------------------------
    def _rollout(self, state, cell, fov):
        """Discounted return if we commit to `cell` against a fov-human.

        The simulated human is seeded with what a theta-human would ALREADY
        believe, then left to run on its own perception. Its orientation changes
        as it walks, so the cone sweeps different cells in different rollouts -
        which is precisely how a counter inside the cone comes out worth more
        than one in the blind spot.

        Return is deliveries (large), plus the mdp's shaped reward, plus
        potential-based recipe progress. The progress term is what makes the
        rollouts separable at all: at depth 40 almost nothing reaches the pass,
        and on sparse alone every candidate came back 0.0.
        """
        sim = state.deepcopy()
        human = LimitedVisionHuman(self.mdp, fov, agent_index=self.other_index)
        human.view = self.post.beliefs_for(fov)
        shadow_bot = type(self.baseline)(self.mdp, agent_index=self.agent_index)
        target, value, disc = cell, 0.0, 1.0
        phi = potential(self.mdp, sim) if self.w_progress else 0.0

        for _ in range(self.horizon):
            if self.mdp.is_terminal(sim):
                break
            me = sim.players[self.agent_index]
            pos, orient = tuple(me.position), tuple(me.orientation)

            if target is not None:                  # committed leg
                walk = TruthView(self.mdp, sim).walkable | {pos}
                mv, arrived = geo.step_towards(walk, pos, orient, target)
                if arrived:
                    ar, target = Action.INTERACT, None   # then hand back to the baseline
                else:
                    ar = mv or Action.STAY
            else:
                ar, _ = shadow_bot.action(sim)

            ah, _ = human.action(sim)
            sim, sparse, shaped, _ = self.mdp.get_state_transition(
                sim, self._joint(ar, ah))

            r = self.w_deliver * float(sum(sparse)) + self.w_env * float(sum(shaped))
            if self.w_progress:
                # gamma*Phi(s') - Phi(s), NOT a raw bonus. See progress.py: the
                # difference form telescopes to gamma^T Phi(s_T) minus a constant
                # every candidate shares, so it cannot be farmed by picking an
                # item up and putting it down again.
                nxt = potential(self.mdp, sim)
                r += self.w_progress * (self.gamma * nxt - phi)
                phi = nxt
            value += disc * r
            disc *= self.gamma
        return value

    # -- decision -----------------------------------------------------------
    def rank(self, state):
        """[(Q, tier, verb, cell)] best first, over the baseline's top-k.

        `top_k` and `min_p` are the two cost knobs and they multiply: one rollout
        per candidate per live cone, each stepping the real mdp for `horizon`
        ticks with a full LimitedVisionHuman inside it. min_p drops cones the
        posterior has already ruled out, which is what makes this get CHEAPER as
        the belief sharpens rather than costing the same forever. If it rules
        every cone out -- possible only when the posterior is degenerate -- we
        fall back to the MAP cone alone rather than to no rollout at all, because
        an unranked candidate list is strictly worse than one ranked under a
        single guess.
        """
        cands = self.baseline.rank_subtasks(state)[:self.top_k]
        live = {f: p for f, p in self.post.p.items() if p >= self.min_p}
        if not live:
            live = {self.post.map_fov(): 1.0}
        norm = sum(live.values())

        out = []
        for i, (tier, verb, cell) in enumerate(cands):
            q = 0.0
            for fov, p in live.items():
                q += (p / norm) * self._rollout(state, cell, fov)
            # TIE-BREAK ON THE BASELINE'S OWN RANK, not on (tier, verb, cell).
            # Sorting a tuple that ends in the verb string meant that whenever Q
            # came back flat -- which, on sparse-only scoring, was nearly always
            # -- the order collapsed to alphabetical: "chop" beat "wash" every
            # time, at the same T_WORK tier, even when the baseline had put wash
            # first because it was nearer. Keying on i makes a flat Q degrade to
            # exactly the baseline ordering, so the filter can only ever improve
            # on its nominal policy, never scramble it.
            out.append((-q, i, tier, verb, cell))
        out.sort()
        self.last_Q = {(v, c): -q for q, _, _, v, c in out}
        return [(-q, t, v, c) for q, _, t, v, c in out]

    def action(self, state):
        ranked = self.rank(state)
        if not ranked:
            return Action.STAY, {"subtask": None, "fov_post": dict(self.post.p)}
        q, tier, verb, cell = ranked[0]
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = TruthView(self.mdp, state).walkable | {pos}
        mv, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (mv or Action.STAY)


        return act, {"subtask": self.last_subtask, "Q": q,
                     "fov_post": dict(self.post.p)}

    def set_agent_index(self, i):
        self.agent_index, self.other_index = i, 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
