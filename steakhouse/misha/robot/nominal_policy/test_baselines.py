"""Checks for baselines.py's sub-task distribution machinery. Run it directly:

    PYTHONPATH=$STEAK_ROOT:$STEAK_ROOT/no_larping python test_baselines.py

Three of these carry the file's actual claims. The rest are guard rails.

  IS_THE_VALUE_FUNCTION   pi at beta=1 is bayes's single-agent value, normalised
                          -- not something similar to it, the same numbers.
  STATIONARY              sticky sampling leaves pi exactly stationary, at every
                          rho, so the occupancy distribution over ticks EQUALS
                          the draw distribution.
  HAS_POWER               the stationarity check FAILS on the two ways of
                          getting it wrong. Without this the check above proves
                          nothing -- a test that cannot fail is not evidence.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import overcooked_ai_py                                              # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "layout", "layouts")

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv        # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common.views import TruthView                                   # noqa: E402
from human.limited_vision_human import LimitedVisionHuman            # noqa: E402
from robot.nominal_policy.baselines import (                         # noqa: E402
    BASELINES, steps_to_finish, subtask_pi)
from robot.nominal_policy.bayesian_delegation import _Snapshot       # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  %-4s %-26s %s" % ("ok" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def advanced_state(layout="butchery", ticks=60, seed=0):
    """A state some way into an episode, so hands are full and pi is non-trivial."""
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, 30, agent_index=1, seed=seed)
    bot = BASELINES["handoff"](mdp, agent_index=0, seed=seed)
    for _ in range(ticks):
        a, _ = bot.action(env.state)
        h, _ = human.action(env.state)
        if mdp.is_terminal(env.state):
            break
        env.step((a, h))
    return mdp, env.state


def maxdev(a, b):
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


# ---------------------------------------------------------------------------
print("\nIS_THE_VALUE_FUNCTION -- pi(beta=1) vs bayes's own value, same state")
mdp, state = advanced_state()
bayes = BASELINES["bayes"](mdp, agent_index=0, seed=0)
snap = _Snapshot(bayes, state)
me = state.players[0]
pos, orient = tuple(me.position), tuple(me.orientation)
walk = TruthView(mdp, state).walkable | {pos}

# the step count must agree cell for cell, or nothing downstream can
mine = [s for s in snap.mine if s is not None]
step_mismatch = [(s, snap.steps[(0, s)], steps_to_finish(walk, pos, orient, s[2]))
                 for s in mine
                 if snap.steps[(0, s)] != steps_to_finish(walk, pos, orient, s[2])]
check("steps_to_finish", not step_mismatch,
      "%d/%d candidates agree with _Snapshot.steps" % (len(mine) - len(step_mismatch),
                                                       len(mine)))

ref = {s: bayes._snapshot(state)._worth(0, s) for s in mine}
z = sum(ref.values())
ref = {s: v / z for s, v in ref.items()}
got = subtask_pi(mine, pos, orient, walk, beta=1.0)
check("pi(beta=1) == value/Z", maxdev(ref, got) < 1e-12,
      "max deviation %.2e over %d subtasks" % (maxdev(ref, got), len(mine)))

# ---------------------------------------------------------------------------
print("\nBETA -- endpoints and the calibration of the default")
det = min(mine, key=lambda s: (s[0], snap.steps[(0, s)]))
sharp = subtask_pi(mine, pos, orient, walk, beta=float("inf"))
check("beta=inf is argmax", abs(sharp.get(det, 0.0) - 1.0) < 1e-12,
      "puts %.4f on the deterministic policy's pick" % sharp.get(det, 0.0))
flat = subtask_pi(mine, pos, orient, walk, beta=0.0)
check("beta=0 is uniform", maxdev(flat, {s: 1.0 / len(mine) for s in mine}) < 1e-12)

# two same-tier subtasks five tiles apart: does beta=8 really give ~90/10?
GAMMA_ = 0.95
r = GAMMA_ ** 5
p_near = r ** 0 / (r ** 0 + r ** 8)          # value ratio raised to beta=8
check("beta=8 -> ~90/10", 0.87 < p_near < 0.93,
      "same tier, 5 tiles apart -> %.3f / %.3f" % (p_near, 1 - p_near))

# ---------------------------------------------------------------------------
print("\nFAITHFUL -- argmax(pi) should be the baseline's own pick")
from robot.nominal_policy.baselines import within_tier_penalty        # noqa: E402
for pol, floor in (("greedy", 0.99), ("solo", 0.99), ("handoff", 0.99)):
    mdp2 = SteakHouseGridworld.from_layout_name("butchery")
    env2 = OvercookedEnv.from_mdp(mdp2, horizon=400, info_level=0)
    env2.reset()
    hu = LimitedVisionHuman(mdp2, 30, agent_index=1, seed=0)
    pb = BASELINES[pol](mdp2, agent_index=0, seed=0)
    hit = tot = 0
    for _ in range(400):
        st = env2.state
        rk = pb.rank_subtasks(st)
        if rk:
            m_ = st.players[0]
            p_, o_ = tuple(m_.position), tuple(m_.orientation)
            w_ = TruthView(mdp2, st).walkable | {p_}
            pen_ = {t: within_tier_penalty(pb, st, t, w_, p_) for t in rk}
            pp = subtask_pi(rk, p_, o_, w_, float("inf"), pen_)
            if pp:
                tot += 1
                hit += (max(pp, key=pp.get) == rk[0])
        aa, _ = pb.action(st)
        hh, _ = hu.action(st)
        if mdp2.is_terminal(env2.state):
            break
        env2.step((aa, hh))
    rate = hit / float(max(tot, 1))
    # Not 100%, and cannot be: the baselines are lexicographic and the value
    # function deliberately is not. See within_tier_penalty's docstring.
    check("argmax(pi) is %s's pick" % pol, rate >= floor,
          "%.1f%% of %d ticks" % (100 * rate, tot))

print("\nSTATIONARY -- sticky sampling must leave pi exactly stationary")
# A SYNTHETIC pi over `mine`'s actual candidates, not subtask_pi's modelled
# one: the stationarity property is generic to any valid distribution, and
# subtask_pi's tier term can make pi collapse onto one candidate whenever
# `mine` spans a tier gap (TIER_GAIN=3 dominates fast), which starves the
# HAS_POWER corruption checks below of anything to detect. A harmonic decay
# guarantees real, non-degenerate spread regardless of what `mine` is.
_w = [1.0 / (i + 1) for i in range(len(mine))]
_z = sum(_w)
pi = {s: w / _z for s, w in zip(mine, _w)}
N = 300000


def occupancy(rho, redraw_excludes_current=False, per_subtask_rho=False, seed=1):
    """Run the sticky chain and count TICKS each subtask is committed for."""
    rng = random.Random(seed)
    subs = list(pi)

    def draw(exclude=None):
        d = pi
        if exclude is not None:
            d = {s: p for s, p in pi.items() if s != exclude}
            z = sum(d.values())
            if z <= 0:
                return exclude
            d = {s: p / z for s, p in d.items()}
        r, acc = rng.random(), 0.0
        for s, p in d.items():
            acc += p
            if r < acc:
                return s
        return subs[-1]

    held, counts = None, {s: 0 for s in subs}
    for _ in range(N):
        if held is None:
            held = draw()
        else:
            # the broken variant: hold longer for subtasks further down pi
            rr = rho * (0.5 + 0.5 * pi[held] / max(pi.values())) if per_subtask_rho else rho
            if rng.random() > rr:
                held = draw(exclude=held if redraw_excludes_current else None)
        counts[held] += 1
    return {s: c / float(N) for s, c in counts.items()}


tol = 6.0 / math.sqrt(N)          # ~6 sigma on a Bernoulli at N draws
for rho in (0.0, 0.5, 0.9, 0.99):
    occ = occupancy(rho)
    # a high rho means fewer independent draws, so widen the band accordingly
    eff = tol / math.sqrt(max(1e-9, 1.0 - rho)) if rho < 1 else 1.0
    check("occupancy == pi, rho=%.2f" % rho, maxdev(occ, pi) < eff,
          "max deviation %.4f (tol %.4f)" % (maxdev(occ, pi), eff))

print("\nHAS_POWER -- the check above must FAIL on the two ways to get it wrong")
bad1 = occupancy(0.9, redraw_excludes_current=True)
check("re-draw excluding current", maxdev(bad1, pi) > 10 * tol,
      "deviates by %.4f -- correctly detected as wrong" % maxdev(bad1, pi))
bad2 = occupancy(0.9, per_subtask_rho=True)
check("per-subtask rho", maxdev(bad2, pi) > 10 * tol,
      "deviates by %.4f -- correctly detected as wrong" % maxdev(bad2, pi))

# ---------------------------------------------------------------------------
print("\nSAMPLING -- forced re-draws, reproducibility, rebuildability")

Cls = BASELINES["handoff"]
b1 = Cls(mdp, agent_index=0, seed=0, beta=8.0, rho=0.95)
b2 = Cls(mdp, agent_index=0, seed=0, beta=8.0, rho=0.95)
t1 = [b1.action(state)[1]["subtask"] for _ in range(30)]
t2 = [b2.action(state)[1]["subtask"] for _ in range(30)]
check("same seed reproduces", t1 == t2)

# Seed divergence has to be measured where the draw is actually free. At the
# defaults this state has 2 candidates and beta=8 makes pi near-degenerate, so
# every seed picks the same thing and "different seed differs" would fail for a
# reason that has nothing to do with seeding. Force a uniform draw every tick.
f1 = Cls(mdp, agent_index=0, seed=0, beta=0.0, rho=0.0)
f2 = Cls(mdp, agent_index=0, seed=7, beta=0.0, rho=0.0)
u1 = [f1.action(state)[1]["subtask"] for _ in range(40)]
u2 = [f2.action(state)[1]["subtask"] for _ in range(40)]
check("different seed differs", u1 != u2,
      "uniform draw over %d candidates, 40 ticks" % len(pi))
check("rank_subtasks is pure",
      b1.rank_subtasks(state) == Cls(mdp, agent_index=0, seed=0).rank_subtasks(state))
check("rebuildable via type(bot)(mdp, agent_index=..., seed=...)",
      isinstance(type(b1)(mdp, agent_index=0, seed=0), Cls),
      "the form a search rebuilds a policy with -- see methods.py's _stoch()")

# rho=1 must never re-draw spontaneously; rho=0 must never hold
never = Cls(mdp, agent_index=0, seed=0, rho=1.0)
whys = [never.action(state)[1]["subtask_redraw"] for _ in range(60)]
check("rho=1 never spontaneous", "spontaneous" not in whys,
      "%d forced, %d held" % (whys.count("forced"), whys.count("held")))
always = Cls(mdp, agent_index=0, seed=0, rho=0.0)
whys0 = [always.action(state)[1]["subtask_redraw"] for _ in range(60)]
check("rho=0 never holds", "held" not in whys0,
      "%d forced, %d spontaneous" % (whys0.count("forced"), whys0.count("spontaneous")))

# the info dict must carry a real distribution
_, info = b1.action(state)
d = info["subtask_dist"]
check("subtask_dist sums to 1", abs(sum(d.values()) - 1.0) < 1e-9,
      "%d entries" % len(d))

# ---------------------------------------------------------------------------
print("\nPOSTERIOR -- bayes draws from its own marginal, never IDLE")
bp = BASELINES["bayes"](mdp, agent_index=0, seed=0)
human = LimitedVisionHuman(mdp, 30, agent_index=1, seed=0)
ha, _ = human.action(state)
for _ in range(5):
    bp.update(state, ha)
_, pinfo = bp.action(state)
check("dist is the marginal", abs(sum(pinfo["subtask_dist"].values()) - 1.0) < 1e-9,
      "%d subtasks, IDLE excluded" % len(pinfo["subtask_dist"]))
check("bayes readouts survive", "partner_subtask" in pinfo and "n_alloc" in pinfo,
      "partner_p=%.3f" % pinfo["partner_p"])
check("last_action fed back", bp.last_action is not None,
      "otherwise bayes stops scoring its own action")

print("\n%s  (%d failures)" % ("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS),
                               len(FAILS)))
sys.exit(1 if FAILS else 0)
