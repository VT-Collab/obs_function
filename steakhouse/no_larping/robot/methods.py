"""Every policy this package can put in the ROBOT seat, in one registry.

play.py and watch.py both read METHODS and both build through make_robot, so a
new method is added HERE and nowhere else -- neither harness knows the name of a
single policy class. Same reason baselines.py keeps BASELINES: the harnesses
should read a registry, not import a list of classes and grow a special case per
policy. BASELINES is still the registry of NOMINAL policies; this is the registry
of everything runnable, nominal policies and the filter's variants alike.

The split down the middle of the table is the experiment:

  CONTROL    theta-blind. Full state, no cone anywhere -- see the docstrings in
             robot/nominal_policy/. Whatever an FOV-aware policy beats these by
             is attributable to theta and to nothing else.
  FOV        theta-aware. FOVPosterior infers the cone from the human's ACTIONS
             and QMDPFilter re-ranks the nominal policy's top-k under it.

WHY THE qmdp-* ROW EXISTS AT ALL. QMDPFilter takes its `baseline` as a
constructor argument and only ever RE-RANKS what that baseline proposes -- its
own docstring calls this the ceiling: "an option the baseline never proposed
cannot be chosen here however good it would have been". Which baseline it wraps
is therefore a first-class axis of the method, not an implementation detail, and
it was previously nailed to handoff with no way to say otherwise from a command
line. qmdp-sparse, qmdp-map and bayes-noip are the three A/Bs the source already
names as controls; they were reachable only by editing a constructor call.
"""
import collections
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.nominal_policy.baselines import BASELINES            # noqa: E402
from robot.nominal_policy.subtask_dist import (                 # noqa: E402
    BETA, RHO, stochastic)
from robot.filter.qmdp_fov import FOVPosterior, QMDPFilter      # noqa: E402

CONTROL = "theta-blind controls -- no model of what you can see"
FOV = "fov-aware -- infers your CONE from your actions"
STOCH = "stochastic -- sub-task DRAWN from a distribution, not argmaxed"


class Drivers:
    """Everything that wants feeding with the human's action, behind one object.

    Each harness calls ONE update() per tick and reads ONE `.p` for the cone
    readout. That was fine while every method had at most one thing to feed, but
    qmdp wrapped around bayes has two -- the cone posterior and the baseline's
    own allocation belief -- and only one of them holds a belief over CONES. The
    fan-out lives here so neither harness has to learn the difference.

    `.p` is the cone posterior and nothing else. A method with no cone returns
    {} from it, which is exactly the contract BayesianDelegationRobot.p already
    documents: empty means "I am theta-blind", and both HUDs fall through to
    saying so.
    """

    def __init__(self, members=(), cone=None):
        self.members = list(members)
        self.cone = cone

    def __bool__(self):
        return bool(self.members) or self.cone is not None

    __nonzero__ = __bool__            # py2-style guard; harmless on py3

    def update(self, state, human_action):
        for m in self.members:
            m.update(state, human_action)

    @property
    def p(self):
        return self.cone.p if self.cone is not None else {}


Method = collections.namedtuple("Method", "key group blurb hotkey build")


# Every build takes the same `cfg` dict rather than a positional argument list.
# The list had grown to six by the time beta and rho arrived, and every new knob
# meant editing every closure; a dict means the next one touches only the
# closures that read it.
DEFAULTS = {"seed": 0, "top_k": 3, "depth": 40, "beta": BETA, "rho": RHO}


def _nominal(cls, **kw):
    """A bare nominal policy. It drives itself iff it has an update() to drive."""
    def build(mdp, robot_idx, human_idx, cfg):
        bot = cls(mdp, agent_index=robot_idx, seed=cfg["seed"], **kw)
        return bot, Drivers([bot] if hasattr(bot, "update") else [])
    return build


def _stoch(baseline_key, source="prior", **inner_kw):
    """A nominal policy that DRAWS its sub-task instead of taking the argmax.

    Wraps the class, not the built instance, so the result still answers to
    `type(bot)(mdp, agent_index=..., seed=...)` -- see stochastic() for why that
    matters. Delegation keeps `update` working, so bayes-post is fed by the same
    Drivers fan-out as everything else here.
    """
    cls = BASELINES[baseline_key]

    def build(mdp, robot_idx, human_idx, cfg):
        wrapped = stochastic(cls, beta=cfg["beta"], rho=cfg["rho"],
                             source=source, **inner_kw)
        bot = wrapped(mdp, agent_index=robot_idx, seed=cfg["seed"])
        return bot, Drivers([bot] if hasattr(bot.inner, "update") else [])
    return build


def _qmdp(baseline_key, **kw):
    """QMDPFilter over another entry in this table.

    The baseline is built through the registry rather than from BASELINES so a
    baseline that drives itself keeps being driven: qmdp-bayes feeds BOTH the
    cone posterior and the wrapped bayes robot, and the wrapped robot's belief
    would otherwise sit frozen at its prior for the whole episode while the
    filter rolled out a shortlist ranked by it.
    """
    def build(mdp, robot_idx, human_idx, cfg):
        base, base_drv = METHODS[baseline_key].build(
            mdp, robot_idx, human_idx, cfg)
        post = FOVPosterior(mdp, human_index=human_idx, seed=cfg["seed"])
        bot = QMDPFilter(mdp, base, post, top_k=cfg["top_k"], horizon=cfg["depth"],
                         agent_index=robot_idx, **kw)
        return bot, Drivers(base_drv.members + [post], cone=post)
    return build


# Ordered by how much of the human each one models -- the setup screen renders
# this verbatim, so the blurbs are the documentation most people will read.
_TABLE = [
    Method("greedy", CONTROL,
           "nearest job first, every tick -- no model of you at all",
           "g", _nominal(BASELINES["greedy"])),
    Method("solo", CONTROL,
           "greedy plus: gives up a station in the wall if you are closer",
           "s", _nominal(BASELINES["solo"])),
    Method("handoff", CONTROL,
           "solo plus staging -- picks the counter nearest ITSELF",
           "f", _nominal(BASELINES["handoff"])),
    Method("bayes", CONTROL,
           "infers WHICH JOB you are doing, never what you can SEE",
           "b", _nominal(BASELINES["bayes"])),
    # "The control for the control": with inverse planning off the belief is the
    # prior every tick, so any concentration bayes shows above this line is the
    # evidence doing work rather than the value prior. bayesian_delegation.py:337.
    Method("bayes-noip", CONTROL,
           "bayes with inverse planning OFF -- the prior, never the evidence",
           None, _nominal(BASELINES["bayes"], inverse_planning=False)),

    Method("qmdp", FOV,
           "rolls handoff's top-k out against a simulated you",
           "q", _qmdp("handoff")),
    Method("qmdp-greedy", FOV,
           "the same, over greedy's shortlist instead of handoff's",
           None, _qmdp("greedy")),
    Method("qmdp-solo", FOV,
           "the same, over solo's shortlist",
           None, _qmdp("solo")),
    # The interesting one: every other shortlist here is ordered by DISTANCE, so
    # the filter is re-ranking three nearby cells. bayes orders by marginal
    # posterior over allocations, so the top-k that reaches the rollout is a
    # different set of candidates, not the same set in a different order -- and
    # the filter's stated ceiling is exactly which candidates it gets to see.
    #
    # Costs about what plain qmdp costs, measured, which is not obvious: _rollout
    # rebuilds the baseline per rollout and bayes enumerates allocations, but the
    # shadow bot only acts AFTER the committed leg ends and at depth 40 the leg
    # eats most of the horizon. 60 ticks on divide: 10.9s here, 11.7s for qmdp.
    Method("qmdp-bayes", FOV,
           "the same, over bayes's -- the one shortlist not ordered by distance",
           None, _qmdp("bayes")),
    # The scoring A/B named at qmdp_fov.py:140 -- "w_progress=0, w_env=0,
    # w_deliver=1 reproduces the old sparse-only scoring exactly, keep that A/B
    # available, it is the control". At depth 40 almost nothing reaches the pass,
    # so this is the condition where Q comes back flat and the filter degrades to
    # its baseline; that is the point of having it.
    Method("qmdp-sparse", FOV,
           "qmdp scored on DELIVERIES ONLY -- the scoring A/B, no shaping",
           None, _qmdp("handoff", w_deliver=1.0, w_env=0.0, w_progress=0.0)),
    # Point estimate instead of the mixture: min_p above 1 admits no cone at all,
    # and rank() then falls back to the MAP cone alone. So this is QMDP's
    # expectation over theta replaced by its argmax -- does averaging over cones
    # buy anything over just betting on the likeliest one?
    #
    # Honest caveat: qmdp_fov.py frames min_p as a COST knob and that fallback as
    # a degenerate-posterior safety net, not as a labelled ablation. Driving it
    # deliberately is a use of the branch the source documents, but the framing
    # is this file's, not qmdp_fov.py's.
    Method("qmdp-map", FOV,
           "qmdp against the single likeliest cone, not the whole posterior",
           None, _qmdp("handoff", min_p=1.01)),

    # Every policy above picks argmax rank_subtasks(state), and given that pick
    # the action is a deterministic pure function -- measured: no policy reads
    # its own _rng, step_towards is reproducible over 20k identical inputs, and
    # all four feed it the same walkable set. So ALL the behavioural variation a
    # nominal policy has is in the sub-task choice, which is the only place a
    # distribution is worth putting. See subtask_dist.py for pi and for why
    # stickiness leaves it exactly stationary.
    Method("greedy-stoch", STOCH,
           "greedy, sub-task drawn from its own value prior",
           None, _stoch("greedy")),
    Method("solo-stoch", STOCH,
           "solo, drawn from its own value prior",
           None, _stoch("solo")),
    Method("handoff-stoch", STOCH,
           "handoff, drawn from its own value prior",
           None, _stoch("handoff")),
    # The pairing that isolates inverse planning: identical machinery, evidence
    # off and on, sampled the same way. Mirrors bayes-noip / bayes one level up.
    # bayes-prior is the only entry here drawing a PRIOR from bayes's own belief
    # rather than from subtask_pi, so the two rows differ in evidence alone.
    Method("bayes-prior", STOCH,
           "bayes with inverse planning OFF, drawn from its prior",
           None, _stoch("bayes", source="posterior", inverse_planning=False)),
    Method("bayes-post", STOCH,
           "bayes drawn from its TRUE posterior P(tau_robot | history)",
           None, _stoch("bayes", source="posterior")),
]

METHODS = collections.OrderedDict((m.key, m) for m in _TABLE)
METHOD_KEYS = list(METHODS)
GROUPS = [CONTROL, FOV, STOCH]

# Old spellings that must keep working. `qmdp` has always meant qmdp over
# handoff; qmdp-handoff is the same thing said explicitly.
ALIASES = {"qmdp-handoff": "qmdp"}

HOTKEYS = {m.hotkey: m.key for m in _TABLE if m.hotkey}


def in_group(group):
    return [m for m in _TABLE if m.group == group]


def resolve(kind):
    """Canonical key for a user-supplied name, or None if there is no such method."""
    key = ALIASES.get(kind, kind)
    return key if key in METHODS else None


def make_robot(kind, mdp, robot_idx, human_idx, seed=0, top_k=3, depth=40,
               beta=None, rho=None):
    """(agent, driver). `driver` is whatever wants update(state, human_action).

    For most controls that is nobody: they never look at the human's actions, and
    the driver comes back None so the harnesses skip the call exactly as before.
    A policy that infers the partner's INTENT -- bayes -- is its own driver, and
    an FOV-aware one is driven by its cone posterior; both are fed by the same
    call, and a composed method by the same call fanning out. Note that being fed
    the human's actions is NOT a theta leak: an action is public, a cone is not.

    `beta` and `rho` reach only the *-stoch / bayes-* entries; passing them with
    any other method is silently ignored rather than an error, so a command line
    can carry them across a sweep that mixes deterministic and stochastic rows.
    """
    key = resolve(kind)
    if key is None:
        raise SystemExit("no robot method %r. Available:\n  %s"
                         % (kind, "\n  ".join(METHOD_KEYS)))
    cfg = dict(DEFAULTS, seed=seed, top_k=top_k, depth=depth)
    if beta is not None:
        cfg["beta"] = beta
    if rho is not None:
        cfg["rho"] = rho
    agent, drivers = METHODS[key].build(mdp, robot_idx, human_idx, cfg)
    return agent, (drivers or None)


def listing():
    """Multi-line description of the whole table, for --list-robots and --help."""
    out = []
    for group in GROUPS:
        out.append(group)
        for m in in_group(group):
            out.append("  %-14s %s" % (m.key, m.blurb))
    return "\n".join(out)
