"""Every policy this package can put in the ROBOT seat, in one registry.

play.py and watch.py both read METHODS and both build through make_robot, so a
new method is added HERE and nowhere else -- neither harness knows the name of a
single policy class. Same reason baselines.py keeps BASELINES: the harnesses
should read a registry, not import a list of classes and grow a special case per
policy. BASELINES is still the registry of NOMINAL policies; this is the registry
of everything runnable, nominal policies and the filter's variants alike.

ELEVEN ROWS, AND ONLY TWO IDEAS. The table is a baseline column and the same
filter laid on top of each entry in it:

  BASE   four theta-blind baselines. Full state, no cone anywhere -- see the
         docstrings in robot/nominal_policy/. Whatever an FOV-aware policy beats
         these by is attributable to theta and to nothing else. Plus one
         evidence control, bayes-noip.
  FOV    the SAME QMDPFilter (robot/filter/core/qmdp.py) over each of
         them. FOVPosterior (robot/filter/core/fov_posterior.py) infers the cone
         from the human's ACTIONS, and the filter searches over how to EXECUTE
         its baseline's job under it -- which legal target cell, which first
         step, whether to wait a tick. Plus two controls.

    greedy  vs  qmdp-greedy        handoff  vs  qmdp
    solo    vs  qmdp-solo          bayes    vs  qmdp-bayes

WHY THERE IS A ROW PER BASELINE. QMDPFilter takes its `baseline` as a
constructor argument, re-ranks only what that baseline proposes, and falls back
to that baseline's own action whenever the rollouts do not clear its margin. So
the wrapped baseline is both the floor and the thing to beat, which makes WHICH
baseline it wraps a first-class axis of the method rather than an implementation
detail. qmdp-solo is not a cousin of qmdp; it is the same class over a different
floor.

WHY EVERY BASELINE IS STOCHASTIC. The filter consumes a DISTRIBUTION over
sub-tasks. A deterministic ladder has none, so subtask_dist.true_pi could only
hand it a lifted Boltzmann reconstruction of what a policy that never draws
anything might have drawn -- and it was then scored against a baseline that never
held those preferences. There is now ONE kind of baseline and both sides of every
pair use it, which is also why no row carries a `-stoch` suffix: there is nothing
left to contrast with. The old spellings resolve through ALIASES.

Superseded and deleted: qmdp_fov.QMDPFilter, which re-ranked whole SUB-TASKS and
walked one shortest path to the winner. robot/filter/RESULTS.md records what it
could not express and what replacing it bought.
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
from robot.filter.core.fov_posterior import FOVPosterior             # noqa: E402
from robot.filter.core.qmdp import QMDPFilter                        # noqa: E402

BASE = "baselines -- theta-blind, sub-task DRAWN from the policy's own pi"
FOV = "fov-aware -- the SAME layer over each baseline, inferring your CONE"
# Kept as aliases so any external caller still naming the old groups keeps
# working. There are only two groups now: a baseline, and that baseline with
# the FOV layer on top.
CONTROL, STOCH = BASE, BASE


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
    matters. Delegation keeps `update` working, so bayes is fed by the same
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

    The baseline comes through the REGISTRY rather than from BASELINES, so a
    baseline that drives itself keeps being driven: qmdp-bayes feeds both the cone
    posterior and the wrapped bayes robot, whose belief would otherwise sit frozen
    at its prior for the whole episode while the filter ranked candidates by it.

    There is ONE filter class now. The earlier split -- an enumerate-(v,g,a) filter
    beside a masked-search one -- is gone: the enumerating one truncated its stash
    shortlist by distance from the ROBOT and could only press the cell it had
    committed to, and robot/filter/RESULTS.md section 2b has what that cost.
    """
    def build(mdp, robot_idx, human_idx, cfg):
        base, base_drv = METHODS[baseline_key].build(
            mdp, robot_idx, human_idx, cfg)
        post = FOVPosterior(mdp, human_index=human_idx, seed=cfg["seed"])
        bot = QMDPFilter(mdp, base, post, agent_index=robot_idx, **kw)
        return bot, Drivers(base_drv.members + [post], cone=post)
    return build


# Ordered by how much of the human each one models -- the setup screen renders
# this verbatim, so the blurbs are the documentation most people will read.
_TABLE = [
    # THE FOUR BASELINES. Every one of them DRAWS its sub-task from its own
    # distribution. There used to be a deterministic twin of each sitting beside
    # it -- the pair was spelled `handoff` and `handoff-stoch` -- and the twins
    # are gone on purpose, not lost.
    #
    # The filter consumes a DISTRIBUTION over sub-tasks. A deterministic ladder
    # does not have one, so subtask_dist.true_pi could only hand it a lifted
    # Boltzmann RECONSTRUCTION of what a policy that never draws anything might
    # have drawn. Keeping both kinds meant every paired comparison had a choice
    # of baseline and the wrong one was the default: a filter fed a fiction of
    # its baseline's preferences, scored against a baseline that never held
    # them. One kind of baseline, used on both sides, is the only arrangement in
    # which the comparison means anything -- so this is it, and the suffix is
    # dropped because there is nothing left to contrast with.
    #
    # Each of these keeps the pi it sampled from in `last_pi`, which true_pi
    # returns as "drawn"; bayes draws from its genuine allocation posterior.
    # That is what each qmdp-* row wraps, and what report.PAIRS scores against.
    Method("greedy", BASE,
           "nearest job first -- no model of you at all; sub-task DRAWN",
           "g", _stoch("greedy")),
    Method("solo", BASE,
           "greedy plus: demotes any cell you would reach sooner",
           "s", _stoch("solo")),
    Method("handoff", BASE,
           "solo plus staging -- picks the counter nearest ITSELF",
           "f", _stoch("handoff")),
    Method("bayes", BASE,
           "infers WHICH JOB you are doing, never what you can SEE",
           "b", _stoch("bayes", source="posterior")),
    # THE EVIDENCE CONTROL, one level below the FOV question: identical
    # machinery to bayes with inverse planning OFF. NOT frozen at the prior --
    # update() carries the belief and re-projects it onto each tick's allocation
    # set before the gated block, so it still moves. What the flag removes is the
    # action-likelihood multiply and nothing else (bayesian_delegation.py:444-453
    # is the only place `inverse_planning` is read in the repo), so bayes-noip
    # against bayes differs in EVIDENCE ALONE and the gap between them is what
    # the inference buys.
    Method("bayes-noip", BASE,
           "bayes with inverse planning OFF -- belief carried, evidence never folded in",
           None, _stoch("bayes", source="posterior", inverse_planning=False)),

    # ONE FILTER, over each baseline in turn. `qmdp` is the headline; the other
    # three are the same object wrapping a different rung, which is what makes each
    # comparison paired -- report.PAIRS encodes exactly these.
    #
    # WHAT IT DOES, in one line: INTERACT-legality is a PREDICATE on (position,
    # orientation) over the union of the top-m jobs' legal cells, and every
    # (cell, arrival tick) reachable inside `depth` is scored. So every stash
    # counter is in scope -- 23 to 51 of them, not the 6 nearest the robot -- and
    # "same counter, two ticks later" is a distinct plan rather than a re-plan
    # artifact. DESIGN.md has the reasons; RESULTS.md section 2b has the numbers.
    Method("qmdp", FOV,
           "searches WHERE and WHEN to act -- every legal cell, every arrival",
           "q", _qmdp("handoff")),
    Method("qmdp-greedy", FOV,
           "the same filter over greedy's jobs",
           None, _qmdp("greedy")),
    Method("qmdp-solo", FOV,
           "the same filter over solo's jobs",
           None, _qmdp("solo")),
    Method("qmdp-bayes", FOV,
           "the same filter over bayes's jobs -- bayes runs inside every rollout",
           None, _qmdp("bayes")),

    # THE TWO CONTROLS, both the same class with one knob moved.
    #
    # -base is the PARITY control: the mask collapses to the baseline's own
    # realised cell with no wait variants, so the only plan available is the
    # baseline's own and it must play identically. That check earned its keep --
    # pointing it at a baseline that DRAWS its sub-task is what exposed the filter
    # anchoring its fallback on argmax pi instead of the realised draw.
    #
    # -fixed is the THETA-BLIND control: same search, cone frozen at 90, so
    # whatever qmdp beats it by is what INFERRING theta bought rather than what
    # searching bought.
    Method("qmdp-base", FOV,
           "PARITY CONTROL: mask collapsed to the baseline's own realised cell",
           None, _qmdp("handoff", only_base=True, waits=0)),
    Method("qmdp-fixed", FOV,
           "THETA-BLIND CONTROL: the same search, cone frozen at 90",
           None, _qmdp("handoff", frozen_fov=90)),
]

METHODS = collections.OrderedDict((m.key, m) for m in _TABLE)
METHOD_KEYS = list(METHODS)
GROUPS = [BASE, FOV]

# Old spellings that must keep working.
#
# `handoff` NOW MEANS THE DRAWING POLICY. It used to name a deterministic one
# that sat beside it in this table, and that is a change of meaning rather than a
# rename -- deliberately, because `--robot handoff` should give you the object
# this experiment is built on, and there is no longer a second kind to
# disambiguate from. The -stoch / bayes-post / bayes-prior spellings are what
# these rows were called during the changeover, and saved JSONL grids use them.
#
# The qmdp-exec-* spellings are what the execution-level filter was called while
# it was being built; older grids use those.
ALIASES = {"qmdp-handoff": "qmdp",
           # Retired spellings. Saved JSONL grids use them, and analysis/report.py
           # has to keep reading those grids: qmdp-exec-* was the enumerate-(v,g,a)
           # filter, qmdp-enum-* the same with its budgets opened, qmdp-mask-* the
           # masked search before the two were folded into one class.
           "qmdp-exec": "qmdp", "qmdp-exec-solo": "qmdp-solo",
           "qmdp-exec-greedy": "qmdp-greedy", "qmdp-exec-bayes": "qmdp-bayes",
           "qmdp-exec-base": "qmdp-base", "qmdp-exec-fixed": "qmdp-fixed",
           "qmdp-enum": "qmdp", "qmdp-enum-solo": "qmdp-solo",
           "qmdp-enum-greedy": "qmdp-greedy", "qmdp-enum-bayes": "qmdp-bayes",
           "qmdp-mask": "qmdp", "qmdp-mask-solo": "qmdp-solo",
           "qmdp-mask-greedy": "qmdp-greedy", "qmdp-mask-bayes": "qmdp-bayes"}

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

    `beta` and `rho` reach the baseline rows only; passing them with
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
