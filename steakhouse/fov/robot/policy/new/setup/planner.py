"""THE HAND-WRITTEN BASELINE.  Task-competent, partner-naive, no network.

Two variants, both reported:

    blind   task-competent and treats the partner as scenery
    react   the same, plus REACTIVE deconfliction: it backs off a station the
            partner is standing at, and drops a fetch the partner is already
            carrying. Position and held-object only -- no inference, no cone.

===========================================================================
WHY THE BASELINE IS THE HUMAN'S OWN POLICY WITH PERFECT PERCEPTION
===========================================================================
The robot is a fully-observable agent -- that is stated in the problem setup and
it is what the trained baselines were always handed. So the cleanest competent
robot available is the human's decision policy run with the perception limits
removed: fov=360, occlusion off, the whole map known.

    COMPETENT BY CONSTRUCTION   the probe measures this exact policy solving the
                                task alone in ~220 steps on every layout. It is
                                not a strawman.
    COMPARABLE BY CONSTRUCTION  robot and human differ in EXACTLY two things --
                                what they can perceive, and whether a module is
                                attached. Any measured effect is attributable to
                                one of those, not to a policy-class mismatch.
    FOV-BLIND BY CONSTRUCTION   nothing in it represents a cone, its own or
                                anybody's. `react` reacts to where the partner
                                IS, never to what the partner can SEE.

This also removes the single worst problem with the previous setup. The trained
SP/E3T checkpoints were measured at +20 to +190 steps WORSE than a uniform-random
robot when paired with this human, so "beats the baseline" was a bar under the
floor, and the QMDP module's only demonstrated contribution was repairing damage
the baseline itself caused (old/module/QMDP/RESULTS.md section 6c). A competent
baseline makes a win mean what it says.

===========================================================================
WHY `react` EXISTS
===========================================================================
Deconfliction -- "do not do what your partner is doing" -- needs the partner's
subtask, NOT their field of view. It is partner modelling, not FOV reasoning.
The old campaign's own ablations say so: the eight belief-level terms were at
chance (481 win / 482 loss) and only `task_overlap`, which is deconfliction, made
the result significant -- which is exactly why pinning the cone to a constant
performed as well as the inferred posterior.

So `react` is not a courtesy strong baseline. It is the arm that ABSORBS the
non-FOV explanation. Anything the module wins over `react` cannot be attributed
to "a model of the partner helps", because `react` already has one.

===========================================================================
THE TWO HOOKS, AND WHY OVERRIDING IS THE RIGHT SWITCH
===========================================================================
All teammate reactivity in the human model funnels through exactly two methods:

    _robot_redundant(subtask)   is the partner already carrying this fetch?
    _robot_faced_kind()         which station is the partner standing at?

Everything else -- the ROBOT_SUPPRESS down-weight in _weights, the station-yield
block in _available_advancing, the commitment abandon -- reads one of those two.
So neutralising both switches the whole channel off at the source, with no edit
to limited_vision_human.py and no risk of missing a call site.
"""

import _paths  # noqa: F401   MUST be first

from env import make_human, ROBOT_INDEX                              # noqa: E402
from fov.human.agent.limited_vision_human import UNKNOWN             # noqa: E402


class Planner:
    """The robot. `.action(state) -> (Action, info)`, same interface as the human.

    partner_aware=False is `blind`; True is `react`.
    """

    def __init__(self, mdp, seed=0, partner_aware=False, temperature=0.5,
                 agent_index=ROBOT_INDEX):
        #fov=360 + occlude=False + familiar = perfect perception. The agent still
        #runs its own observe()/decay machinery, it just never misses anything,
        #which keeps its internal state shaped exactly like the human's.
        self.agent = make_human(mdp, 360, seed, temperature=temperature,
                                occlude=False, familiar=True)
        self.agent.agent_index = agent_index
        self.partner_aware = partner_aware
        if not partner_aware:
            self._go_blind()

    def _go_blind(self):
        """Sever both teammate channels on THIS INSTANCE only.

        Bound-method assignment, not subclassing, so nothing about the class --
        and therefore nothing about the human or the filter's shadows -- changes.
        `lambda *_a, **_k` swallows whatever arguments the call site passes, so
        this cannot break if a signature gains a parameter.
        """
        self.agent._robot_redundant = lambda *_a, **_k: False
        self.agent._robot_faced_kind = lambda *_a, **_k: ""

    # ---- the interface --------------------------------------------------
    def action(self, state):
        """-> (Action, info). info['subtask'] is this robot's OWN subtask, which
        is its own business to know -- it is not read off the human."""
        return self.agent.action(state)

    @property
    def subtask(self):
        return self.agent._current

    @property
    def held(self):
        return self.agent.prev_chosen_subtask

    def reset_counters(self):
        for c in ("n_checks", "n_wasted_commits", "n_delivered", "n_abandoned",
                  "n_explore"):
            setattr(self.agent, c, 0)


class NoopPlanner:
    """Plays STAY forever. The `no robot at all` floor.

    Every result gets reported next to this and next to RandomPlanner, because
    old/module/QMDP/RESULTS.md section 6b measured a uniform-random robot BEATING
    the trained policy, and section 5 measured scale-matched noise winning 65.9%
    of paired cells. A win rate quoted without these two floors means nothing.
    """

    def __init__(self, *_a, **_k):
        from overcooked_ai_py.mdp.overcooked_mdp import Action
        self._stay = Action.STAY

    def action(self, _state):
        return self._stay, {"subtask": "wait"}

    def reset_counters(self):
        pass


class RandomPlanner:
    """Uniform over the six primitives. The other floor."""

    def __init__(self, mdp=None, seed=0, **_k):
        import random as _random
        from overcooked_ai_py.mdp.overcooked_mdp import Action
        self._rng = _random.Random(seed)
        self._acts = list(Action.ALL_ACTIONS)

    def action(self, _state):
        return self._rng.choice(self._acts), {"subtask": "random"}

    def reset_counters(self):
        pass


def make_planner(kind, mdp, seed=0, temperature=0.5):
    """kind in {blind, react, noop, random}."""
    if kind == "blind":
        return Planner(mdp, seed, partner_aware=False, temperature=temperature)
    if kind == "react":
        return Planner(mdp, seed, partner_aware=True, temperature=temperature)
    if kind == "noop":
        return NoopPlanner()
    if kind == "random":
        return RandomPlanner(mdp, seed)
    raise ValueError("unknown planner kind %r" % kind)
