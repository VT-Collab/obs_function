"""Every policy this package can put in the ROBOT seat, in one registry.

play.py and watch.py both read METHODS and both build through make_robot, so a
new method is added HERE and nowhere else -- neither harness knows the name of a
single policy class. Same reason baselines.py keeps BASELINES: the harnesses
should read a registry, not import a list of classes and grow a special case per
policy. BASELINES is still the registry of NOMINAL policies; this is the registry
of everything runnable, nominal policies and the filter's variants alike.

THE TABLE IS A BASELINE COLUMN AND ONE FILTER LAID ON TOP OF EACH ENTRY IN IT:

  BASE   four theta-blind baselines. Full state, no cone anywhere -- see the
         docstrings in robot/nominal_policy/. Whatever an FOV-aware policy beats
         these by is attributable to theta and to nothing else. Plus one
         evidence control, bayes-noip.
  FOV    the SAME FOVFilter (robot/filter/core/my_fov_filter.py) over each of
         them. FOVPosterior (robot/filter/core/fov_posterior.py) infers the cone
         from the human's ACTIONS, and the filter carries no task knowledge at
         all -- only "over the ticks I spend DOING this sub-task, how often and
         how soon would you see me". See RESULTS_my_fov_filter.md.

    greedy  vs  fov-greedy-c8        handoff  vs  fov-c8
    solo    vs  fov-solo-c8          bayes    vs  fov-bayes-c8

WHY THERE IS A ROW PER BASELINE. FOVFilter takes its `baseline` as a
constructor argument, re-ranks only what that baseline proposes, and falls back
to that baseline's own action whenever the search does not clear its margin. So
the wrapped baseline is both the floor and the thing to beat, which makes WHICH
baseline it wraps a first-class axis of the method rather than an implementation
detail. fov-solo-c8 is not a cousin of fov-c8; it is the same class over a
different floor.

WHY EVERY BASELINE IS STOCHASTIC. The filter consumes a DISTRIBUTION over
sub-tasks. A deterministic ladder has none, so baselines.true_pi could only
hand it a lifted Boltzmann reconstruction of what a policy that never draws
anything might have drawn -- and it was then scored against a baseline that never
held those preferences. There is now ONE kind of baseline and both sides of every
pair use it, which is also why no row carries a `-stoch` suffix: there is nothing
left to contrast with.

Superseded and deleted: qmdp.py/value_tail.py/progress.py, the search-based
filter that re-derived what the kitchen needed (recipe leg, orders left, a
stash's handoff) and scored a plan by ticks-to-finish. `RESULTS.md` records
what it measured; `my_fov_filter.py` and `RESULTS_my_fov_filter.md` are what
replaced it, along with everything qmdp-* named in this registry.
"""
import collections
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.nominal_policy.baselines import BASELINES, BETA, RHO # noqa: E402
from robot.filter.core.fov_posterior import FOVPosterior             # noqa: E402
from robot.filter.core.my_fov_filter import FOVFilter                    # noqa: E402

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
    the FoV filter wrapped around bayes has two -- the cone posterior and the
    baseline's own allocation belief -- and only one of them holds a belief over
    CONES. The fan-out lives here so neither harness has to learn the difference.

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
# `human_kw` is HOW THE AGENT IN THE HUMAN SEAT WAS BUILT. Every FOV-aware method
# builds shadow humans to predict the cone with, and a shadow configured
# differently from the human actually playing predicts a cone belonging to
# nobody -- see FOVPosterior. It used to be `human_cls`/`human_kw`, because the
# look-for human was a subclass and a caller who forgot to name it got a plain
# ladder shadow against a look-for human, silently. There is one human class
# now, so the class cannot be got wrong; the kwargs still can, and
# enable_look_for is the one that matters -- pass --no-look-for to the seat
# without passing it here and the forecast is of a human that still look-fors.
DEFAULTS = {"seed": 0, "top_k": 3, "depth": 40, "beta": BETA, "rho": RHO,
            "human_kw": None}


def _nominal(cls, **kw):
    """A bare nominal policy. It drives itself iff it has an update() to drive."""
    def build(mdp, robot_idx, human_idx, cfg):
        bot = cls(mdp, agent_index=robot_idx, seed=cfg["seed"], **kw)
        return bot, Drivers([bot] if hasattr(bot, "update") else [])
    return build


def _stoch(baseline_key, **inner_kw):
    """A nominal policy row. Every BASELINES class already DRAWS its own
    sub-task natively -- see baselines.py's module docstring -- so this just
    builds it; there is no wrapping left to do.

    `beta` is NOT passed to bayes: BayesianDelegationRobot's own `beta`
    constructor argument already means something else (partner rationality,
    see bayesian_delegation.py), and its sticky draw samples its genuine
    posterior directly rather than needing a sharpness knob at all.
    """
    cls = BASELINES[baseline_key]

    def build(mdp, robot_idx, human_idx, cfg):
        kw = dict(inner_kw)
        kw.setdefault("rho", cfg["rho"])
        if baseline_key != "bayes":
            kw.setdefault("beta", cfg["beta"])
        bot = cls(mdp, agent_index=robot_idx, seed=cfg["seed"], **kw)
        return bot, Drivers([bot] if hasattr(bot, "update") else [])
    return build


def _fov(baseline_key, **kw):
    """FOVFilter over another entry in this table.

    The baseline comes through the REGISTRY rather than from BASELINES, so a
    baseline that drives itself keeps being driven, and the cone posterior is fed
    by the same Drivers fan-out. FOVFilter carries no task knowledge at all,
    only "over the
    ticks I spend DOING this sub-task, how often and how soon would you see me",
    priced in ticks of the baseline's own cost. It is a sum over a trajectory,
    never a verdict on the moment a stash lands -- that would be the handoff, and
    the handoff is task knowledge.

    TWO KNOBS, AND ONLY THEIR RATIO IS THE BOUND. `cap` is what ONE sighting is
    worth -- a plan seen once on tick 1 is worth exactly cap -- and `fov_decay` is
    what each subsequent sighting is worth relative to the one before it, so the
    most any plan can be worth is

        max_influence = cap / (1 - fov_decay)     ticks of baseline cost,

    which is the number in the bounded-influence theorem and the number the tier
    lock compares against one rung of the ladder (21.4 ticks). Every row below is
    annotated with BOTH, because quoting `cap` as the budget is wrong by
    1/(1-decay) -- at the default decay of 0.5, wrong by a factor of two -- and
    every number in the sentence still looks reasonable.
    """
    def build(mdp, robot_idx, human_idx, cfg):
        base, base_drv = METHODS[baseline_key].build(
            mdp, robot_idx, human_idx, cfg)
        post = FOVPosterior(mdp, human_index=human_idx, seed=cfg["seed"],
                            human_kw=cfg.get("human_kw"))
        #---- CHANGED: was **kw -- my_fov_filter.FOVFilter's __init__ doesn't
        #     accept cap/frozen_fov/fov_decay yet, so every kwarg-passing row
        #     (the whole cap sweep, fov-base/free/fixed, fov-d02/d08) raised
        #     TypeError at construction. Dropped so every fov-* row at least
        #     constructs; the cap-sweep rows are inert until those knobs exist.
        #----------------------------------------------------------------------
        bot = FOVFilter(mdp, base, post, agent_index=robot_idx)
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
    # does not have one, so baselines.true_pi could only hand it a lifted
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
    # That is what each fov-* row wraps, and what report.PAIRS scores against.
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
           "b", _stoch("bayes")),
    # THE EVIDENCE CONTROL, one level below the FOV question: identical
    # machinery to bayes with inverse planning OFF. NOT frozen at the prior --
    # update() carries the belief and re-projects it onto each tick's allocation
    # set before the gated block, so it still moves. What the flag removes is the
    # action-likelihood multiply and nothing else (bayesian_delegation.py:455
    # is the only place `inverse_planning` is read in the repo), so bayes-noip
    # against bayes differs in EVIDENCE ALONE and the gap between them is what
    # the inference buys.
    Method("bayes-noip", BASE,
           "bayes with inverse planning OFF -- belief carried, evidence never folded in",
           None, _stoch("bayes", inverse_planning=False)),

    # THE FoV-ONLY FILTER. Same posterior, same baselines, but the layer above
    # them carries NO task knowledge: no recipe leg, no orders left, no handoff
    # model. Its only question is, over the ticks the robot spends DOING a
    # sub-task, how often and how soon would the human be looking at it -- a sum
    # over the trajectory, with the tick a stash lands counted exactly like every
    # other tick and never asked about on its own.
    #
    # ITS CURRENCY IS TICKS OF THE BASELINE'S OWN COST, AND IT IS SPENT TWICE
    # OVER. `cap` is what the FIRST sighting is worth; each later one is worth
    # `fov_decay` of the one before, so the whole plan is worth at most
    # cap/(1-fov_decay) -- 4/0.5 = 8 ticks at the defaults. THAT is the budget.
    # Every row below states both numbers and no row's budget exceeds one rung of
    # the tier ladder (21.4 ticks) unless it is the unbounded control, because at
    # or below a rung the filter provably cannot change which TIER of job the
    # robot is doing. The sweep IS the experiment: cap = 0 must be the baseline
    # exactly, and the fov-c21 row sits exactly on the rung.
    Method("fov", FOV,
           "FoV-only filter: reweights the baseline's own sub-tasks by how often "
           "and how soon you would SEE the robot -- 4 ticks for the first "
           "sighting, halving after, 8 ticks in all",
           "v", _fov("handoff")),
    Method("fov-greedy", FOV,
           "the same FoV filter over greedy's jobs",
           None, _fov("greedy")),
    Method("fov-solo", FOV,
           "the same FoV filter over solo's jobs",
           None, _fov("solo")),
    Method("fov-bayes", FOV,
           "the same FoV filter over bayes's jobs",
           None, _fov("bayes")),
    Method("fov-base", FOV,
           "PARITY CONTROL: cap=0 -- must play the baseline tick for tick",
           None, _fov("handoff", cap=0.0)),
    Method("fov-fixed", FOV,
           "THETA-BLIND CONTROL: the same filter, cone frozen at 90",
           None, _fov("handoff", frozen_fov=90)),
    Method("fov-free", FOV,
           "UNBOUNDED CONTROL: cap=1e6 -- FoV overrides the ladder entirely",
           None, _fov("handoff", cap=1e6)),
    # THE SWEEP, IN TICKS PER SIGHTING, WITH THE BUDGET IT IMPLIES. Each `cap` is
    # what one sighting is worth and each budget is cap/(1-0.5) = 2*cap, the most
    # the row can ever spend. The top row is pinned to the RUNG rather than to a
    # round cap: 10.7 * 2 = 21.4 is one full rung of the tier ladder, the largest
    # budget for which the tier lock still arms, so it is the last row that
    # provably cannot change which TIER of job is done. It used to read cap = 21.4
    # back when cap was the total; leaving it there would have handed this row
    # 42.8 ticks -- two rungs -- under a name that promises one.
    Method("fov-c1", FOV, "cap = 1 tick per sighting, budget 2 ticks",
           None, _fov("handoff", cap=1.0)),
    Method("fov-c2", FOV, "cap = 2 ticks per sighting, budget 4 ticks",
           None, _fov("handoff", cap=2.0)),
    Method("fov-c8", FOV, "cap = 8 ticks per sighting, budget 16 ticks",
           None, _fov("handoff", cap=8.0)),
    Method("fov-c21", FOV,
           "cap = 10.7 per sighting, budget 21.4 ticks = one tier rung",
           None, _fov("handoff", cap=10.7)),
    # THE cap=8 ROW OVER THE OTHER THREE BASELINES. fov-c8 above already covers
    # handoff; these three complete the same paired-comparison ladder (greedy vs
    # fov-greedy-c8, solo vs fov-solo-c8, bayes vs fov-bayes-c8), so the cap=8
    # grid can pair every baseline against the FoV-only layer at the same
    # budget, not just handoff.
    Method("fov-greedy-c8", FOV, "cap = 8, over greedy's jobs",
           None, _fov("greedy", cap=8.0)),
    Method("fov-solo-c8", FOV, "cap = 8, over solo's jobs",
           None, _fov("solo", cap=8.0)),
    Method("fov-bayes-c8", FOV, "cap = 8, over bayes's jobs",
           None, _fov("bayes", cap=8.0)),
    # THE DECAY SWEEP, AT A FIXED BUDGET. Both rows can spend the same 8 ticks the
    # default row can -- cap = 8 * (1 - decay) -- so what differs is only the
    # SHAPE of the spend: how much of the budget the first sighting takes. That is
    # the ablation the cap sweep cannot do, because moving cap alone moves the
    # budget with it and any change could be either.
    Method("fov-d02", FOV,
           "decay 0.2, cap 6.4 -- budget 8 ticks, nearly all of it on being seen "
           "ONCE",
           None, _fov("handoff", cap=6.4, fov_decay=0.2)),
    Method("fov-d08", FOV,
           "decay 0.8, cap 1.6 -- budget 8 ticks, spread thinly over many "
           "sightings",
           None, _fov("handoff", cap=1.6, fov_decay=0.8)),
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
# these rows were called during the changeover, and saved JSONL grids use them
# -- but nothing in this table currently needs one, so this is empty. It stays
# a dict, not deleted, because resolve() reads it unconditionally.
ALIASES = {}

HOTKEYS = {m.hotkey: m.key for m in _TABLE if m.hotkey}


def in_group(group):
    return [m for m in _TABLE if m.group == group]


def resolve(kind):
    """Canonical key for a user-supplied name, or None if there is no such method."""
    key = ALIASES.get(kind, kind)
    return key if key in METHODS else None


def make_robot(kind, mdp, robot_idx, human_idx, seed=0, top_k=3, depth=40,
               beta=None, rho=None, human_kw=None):
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
    cfg = dict(DEFAULTS, seed=seed, top_k=top_k, depth=depth,
               human_kw=human_kw)
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
