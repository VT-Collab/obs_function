"""Checks for fov_filter.py. Run it directly:

    ~/miniconda3/envs/steakhouse-ai/bin/python robot/filter/tests/test_fov_filter.py

Four of these carry the file's actual claims; the rest are guard rails.

  PARITY          cap = 0 plays the baseline TICK FOR TICK, not approximately.
                  It is asserted on the emitted ACTION, not on `gain == 0`,
                  because a filter that agrees on every number and disagrees on
                  the tie-break is still a different policy. This is the check
                  that caught qmdp anchoring its reference on argmax(pi) rather
                  than on the baseline's realised draw.
  BOUNDED         on every deviating tick, the emitted plan's BASELINE cost is
                  under cap/(1-decay) ticks -- the theorem in the module
                  docstring, run as an assertion over the whole cap sweep rather
                  than argued. IT IS NOT `cap`: cap is what ONE sighting is worth
                  and the plan's whole bonus is a geometric series over sightings,
                  so the bound is the sum of that series. Asserting `cap` here
                  would pass by luck on the small rows and fail on the big ones.
                  Guard (ii)'s held plan is allowed the bound + eps and is
                  reported separately, because that slack is real and hiding it
                  inside the same number would make the bound unfalsifiable.
  BONUS           the cost's FoV half, on a materialised path: a plan seen once
                  on tick 1 is worth exactly `cap` (that identity is what makes
                  the knob's name true), the j-th sighting is worth `decay` times
                  the (j-1)-th, and no plan is ever worth more than cap/(1-decay).
  TIER_SAFETY     at or below one rung of BUDGET (cap/(1-decay) <= 21.4) the FoV
                  term never
                  moves the robot to a different TIER -- on ANY baseline, on any
                  layout, INCLUDING on the ticks where R has collapsed to 0 on
                  every candidate and the cap is restraining nothing. Above a rung
                  it does move. Both halves matter: the first is the safety
                  property, the second is the evidence that the guard is a guard
                  and not a term that does nothing. The per-baseline sweep is not
                  padding -- the R-only version of this property held on the
                  handoff floor and failed on the bayes floor, on the same code at
                  the same cap, and one floor was all this file used to check.
  GUARD_0         where the baseline's own realised draw cannot be priced -- pi
                  gives it no mass, or it is not a row of this tick's ranking --
                  the filter emits the baseline's action and buys nothing. Both
                  refusals are asserted to FIRE on the bayes floor and to never
                  fire on a drawn one, because a guard that never runs and a guard
                  that works look identical from the pass column.
  _tf             one BFS per cell reproduces baselines.steps_to_finish
                  exactly, over thousands of (pos, orient, cell) triples. That
                  equivalence is what licenses reading the whole of a tick's
                  geometry off `field.get(pos)`.

NOTE ON `divide`: the layout in layout/layouts/divide.layout is mid-edit in the
working tree and its 12th row is one character short, so
SteakHouseGridworld.from_layout_name('divide') raises 'Ragged grid' before any of
this code runs. It is excluded from LAYOUTS here rather than worked around, and
nothing in this file touches it. Put it back in LAYOUTS once the grid is square
again.
"""
import ast
import collections
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
_MISHA = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _MISHA)

import overcooked_ai_py                                              # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(_MISHA, "layout", "layouts")

from overcooked_ai_py.mdp.actions import Action                      # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv        # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common.views import TruthView                                   # noqa: E402
from human.limited_vision_human import LimitedVisionHuman            # noqa: E402
from robot.filter.core.fov_filter import FOVFilter                   # noqa: E402
from robot.filter.core.fov_posterior import FOVPosterior             # noqa: E402
from robot.methods import METHODS, make_robot                        # noqa: E402
from robot.nominal_policy.baselines import steps_to_finish           # noqa: E402

FOV_SRC = os.path.join(_MISHA, "robot", "filter", "core", "fov_filter.py")
LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "pantry"]
FAILS = []


def check(name, ok, detail=""):
    print("  %-4s %-30s %s" % ("ok" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def build(layout, kind=None, baseline_key=None, fov=90, seed=0, **kw):
    """(mdp, env, human, bot, drivers). `kind` names a registry row; otherwise
    an FOVFilter is assembled over `baseline_key` with the kwargs given, which is
    how the floors that have no registry row of their own get tested."""
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=100000, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, fov, agent_index=1, seed=seed)
    if kind is not None:
        bot, drv = make_robot(kind, mdp, 0, 1, seed=seed)
    else:
        cfg = {"seed": seed, "top_k": 3, "depth": 40, "beta": 8.0, "rho": 0.95}
        base, base_drv = METHODS[baseline_key].build(mdp, 0, 1, cfg)
        post = FOVPosterior(mdp, human_index=1, seed=seed)
        bot = FOVFilter(mdp, base, post, agent_index=0, **kw)
        from robot.methods import Drivers
        drv = Drivers(base_drv.members + [post], cone=post)
    return mdp, env, human, bot, drv


def episode(layout, kind=None, baseline_key=None, fov=90, seed=0, ticks=200,
            keep=True, **kw):
    """(actions, infos). One paired rollout against a real LimitedVisionHuman."""
    mdp, env, human, bot, drv = build(layout, kind, baseline_key, fov, seed, **kw)
    acts, infos = [], []
    for _ in range(ticks):
        st = env.state
        a, i = bot.action(st)
        h, _ = human.action(st)
        acts.append(a)
        if keep:
            infos.append(i)
        if drv is not None:
            drv.update(st, h)
        if mdp.is_terminal(st):
            break
        env.step((a, h))
    return acts, infos


# ---------------------------------------------------------------------------
print("\nPARITY -- cap = 0 must play the baseline TICK FOR TICK")
# The load-bearing test. Not `gain == 0`: two policies that agree on every number
# and disagree on one tie-break are still two policies, and the failure shows up
# as a different action, not as a different score.
bad, tot, devs = [], 0, 0
for lay in LAYOUTS:
    for seed in (0, 1):
        a_base, _ = episode(lay, kind="handoff", seed=seed, ticks=200, keep=False)
        a_filt, i_filt = episode(lay, kind="fov-base", seed=seed, ticks=200)
        tot += 1
        devs += sum(1 for i in i_filt if i.get("deviated"))
        if a_base != a_filt:
            k = next(j for j in range(min(len(a_base), len(a_filt)))
                     if a_base[j] != a_filt[j])
            bad.append("%s/s%d@t%d" % (lay, seed, k))
check("fov-base == handoff", not bad,
      "%d episodes x 200 ticks over %d layouts%s"
      % (tot, len(LAYOUTS), "" if not bad else "  first divergence " + bad[0]))
check("fov-base never deviates", devs == 0, "%d deviating ticks" % devs)

for floor in ("greedy", "solo", "bayes"):
    a_base, _ = episode("butchery", kind=floor, seed=0, ticks=200, keep=False)
    a_filt, _ = episode("butchery", baseline_key=floor, cap=0.0, seed=0, ticks=200)
    check("cap=0 over %s" % floor, a_base == a_filt,
          "%d ticks" % len(a_base))

# ---------------------------------------------------------------------------
print("\nBOUNDED -- a deviating tick never buys more than cap/(1-decay) ticks")
# The theorem, as an assertion. C(p) <= C(b) forces R_p + D_p <= bonus_p - bonus_b
# <= cap/(1-decay), so the baseline half of the winning plan's cost is the thing to
# watch and the BOUND IS THE SUM OF THE GEOMETRIC SERIES, not `cap`. The decay
# sweep is inside the loop for exactly that reason: at decay 0.8 the same cap buys
# five times as much, and a test that asserted `cap` would have passed on the
# defaults and been wrong about every row it was actually supposed to bound.
CAP_ROWS = [(c, 0.5) for c in (0.0, 1.0, 2.0, 4.0, 8.0, 10.7, 1e6)]
CAP_ROWS += [(c, d) for d in (0.2, 0.8) for c in (2.0, 10.7)]
for cap, dec in CAP_ROWS:
    bound = cap / (1.0 - dec)
    worst_plain, worst_held, n_dev, n_held = -1.0, -1.0, 0, 0
    for lay in ("butchery", "chefs_table"):
        _, infos = episode(lay, baseline_key="handoff", cap=cap,
                           fov_decay=dec, seed=0, ticks=200)
        for i in infos:
            # The bound the filter reports must BE the bound, on every tick,
            # or the number in every log downstream is a different claim from
            # the one asserted here.
            if abs(i.get("max_influence", -1) - bound) > 1e-9:
                worst_plain = float("inf")
            if not i.get("deviated"):
                continue
            n_dev += 1
            bc = i.get("base_cost", 0.0)
            if i.get("held_plan"):
                n_held += 1
                worst_held = max(worst_held, bc)
            else:
                worst_plain = max(worst_plain, bc)
    ok = worst_plain < bound + 1e-9 and worst_held < bound + 2.0 + 1e-9
    check("cap=%-5g decay=%.1f bound holds" % (cap, dec), ok,
          "%d dev ticks (%d held); worst plain %.2f, worst held %.2f, "
          "bound %.2f" % (n_dev, n_held, worst_plain, worst_held, bound))

# ---------------------------------------------------------------------------
print("\nTIER_SAFETY -- at or below one rung the ladder is untouchable, above it is not")
# THE INVARIANT: for a BUDGET of at most one rung -- cap/(1-decay) <= 21.4, which
# is what the lock actually arms on -- the filter never changes which TIER of
# sub-task the robot is pursuing, on any baseline, on any layout. It is asserted
# and not argued because the argued version was WRONG. The old proof went through
# R -- a rung costs 21.4 ticks of regret, the budget is under that, therefore no
# crossing -- and R is 0 for every candidate at once on 12.3% of the bayes floor's
# scoring ticks (pi's keys are last tick's allocation set) and 5.9% of handoff's
# (the realised draw was the least likely candidate). On those ticks the bonus was
# compared against a field of zeros, and fov-bayes crossed rungs at 4 AND 8 ticks
# of budget while this file, which only ever swept the handoff floor, reported
# 0/474.
#
# THE ROWS ARE NAMED BY BUDGET AND THE CAP IS DERIVED, cap = budget * (1 - decay),
# which is the one thing the per-sighting price changed here. A row that passed
# cap = 21.4 straight through would be handing the filter 42.8 ticks -- two rungs
# -- while its own name promised one, and the lock (which reads the budget) would
# have been OFF for it: the test would have been asserting the opposite of what it
# says, and passing only because a crossing is hard to find anyway.
#
# So the sweep is over baselines first and budgets second, and `collapsed` is
# reported beside every count: those are the ticks where R is restraining nothing
# and the absolute tier guard is the only thing holding the property up. A run
# whose `collapsed` is 0 has not tested the guard at all.
DECAY = 0.5                       # the filter's default; cap = budget * (1-DECAY)
def tier_moves(budget, baseline_key="handoff", ticks=200,
               layouts=("butchery", "chefs_table", "back_bar")):
    """(tier moves, live ticks, R-collapsed ticks, moves on R-collapsed ticks).

    `budget` is max_influence = cap/(1-decay), the number the theorem and the tier
    lock are both written in. The cap that buys it is derived here so that no
    caller of this function can accidentally compare a per-sighting price against
    a rung.
    """
    moved = tot = coll = coll_moved = 0
    for lay in layouts:
        _, infos = episode(lay, baseline_key=baseline_key,
                           cap=budget * (1.0 - DECAY), fov_decay=DECAY, seed=0,
                           ticks=ticks)
        for i in infos:
            s, b = i.get("subtask"), i.get("base_subtask")
            if i.get("gated") or s is None or b is None:
                continue
            tot += 1
            mv = (s[0] != b[0])
            moved += mv
            if i.get("r_collapse"):
                coll += 1
                coll_moved += mv
    return moved, tot, coll, coll_moved


m8, t8, c8, _ = tier_moves(8.0)
check("budget=8 never crosses a rung", m8 == 0,
      "%d/%d ticks moved tier (%d of them R-collapsed)" % (m8, t8, c8))
m21, t21, c21, _ = tier_moves(21.4)
check("budget=21.4 = one rung, still locked", m21 == 0,
      "%d/%d ticks moved tier (%d R-collapsed) -- the guard is inclusive at a rung"
      % (m21, t21, c21))
mf, tf, cf, _ = tier_moves(1e6)
check("unbounded DOES cross a rung", mf > 0,
      "%d/%d ticks moved tier -- the safety check above can fail" % (mf, tf))

# EVERY FLOOR, not just the one this file used to sweep. Three layouts and 150
# ticks each per (baseline, cap) keeps the whole sweep near a minute; the failure
# it is guarding against was visible at 1.2% of ticks, so it does not need 200.
for floor in ("greedy", "solo", "handoff", "bayes"):
    tot_m = tot_t = tot_c = tot_cm = 0
    detail = []
    for budget in (4.0, 8.0, 21.4):
        m, t, c, cm = tier_moves(budget, baseline_key=floor, ticks=150)
        tot_m, tot_t, tot_c, tot_cm = tot_m + m, tot_t + t, tot_c + c, tot_cm + cm
        detail.append("budget%g:%d" % (budget, m))
    check("no rung crossed over %s" % floor, tot_m == 0,
          "%d/%d ticks moved tier [%s]; %d ticks had R collapsed to 0 everywhere"
          " and %d of those moved" % (tot_m, tot_t, " ".join(detail), tot_c,
                                      tot_cm))

# ---------------------------------------------------------------------------
print("\nGUARD_0 -- a reference the baseline cannot price is worth no deviation")
# TWO refusals, and each is asserted to EXIST before it is asserted to be
# harmless. A check that only said "and nothing bad happens on those ticks" would
# pass just as well if the branch were dead, which is how the mode-fallback these
# replaced survived unexamined for so long.
#
#   pi_no_ref   pi gives the baseline's own realised cell no mass. On the bayes
#               floor last_pi comes from _marginal(0) over the belief as it stood
#               LAST tick while rank_subtasks re-projects onto this tick's
#               allocation set; when the key sets are disjoint every candidate is
#               floored together and R is 0 everywhere at once.
#   stale_pick  the drawn sub-task is not a ROW of this tick's ranking. The cell
#               usually survives under a DIFFERENT job, which is what makes it
#               dangerous: R gets priced against that job's mass and the tier
#               guard locks onto that job's rung. It is the residue that the tier
#               guard alone cannot see -- 4 ticks in 440 on the bayes floor, all
#               four of them a tier readout contradicting the baseline.
REFUSALS = ("pi_no_ref", "stale_pick")
n_ref = collections.Counter()
bad0 = 0
for lay in LAYOUTS:
    acts, infos = episode(lay, baseline_key="bayes", cap=8.0, seed=0, ticks=200)
    for a, i in zip(acts, infos):
        fired = [f for f in REFUSALS if i.get(f)]
        if not fired:
            continue
        n_ref.update(fired)
        # Asserted on the EMITTED ACTION against info["base_action"], not on
        # info["deviated"], for the same reason PARITY is: a layer that reports
        # deviated=False and emits something else is still a layer that deviated.
        if i.get("deviated") or Action.ACTION_TO_INDEX[a] != i.get("base_action"):
            bad0 += 1
for flag in REFUSALS:
    check("%s fires on bayes" % flag, n_ref[flag] > 0,
          "%d ticks over %d layouts x 200" % (n_ref[flag], len(LAYOUTS)))
check("and emits the baseline there", bad0 == 0,
      "%d of %d refused ticks deviated" % (bad0, sum(n_ref.values())))
n_drawn = collections.Counter()
for floor in ("greedy", "solo", "handoff"):
    _, infos = episode("butchery", baseline_key=floor, cap=8.0, seed=0, ticks=200)
    for i in infos:
        n_drawn.update([f for f in REFUSALS if i.get(f)])
check("neither fires on a drawn floor", not n_drawn,
      "%s over greedy/solo/handoff -- last_pi there IS this tick's pi over this "
      "tick's ranking" % (dict(n_drawn) or "0 ticks"))

# ---------------------------------------------------------------------------
print("\n_tf -- one BFS per cell reproduces steps_to_finish exactly")
mdp, env, human, bot, drv = build("butchery", kind="fov", seed=0)
for _ in range(40):                       # get some way into an episode
    a, _ = bot.action(env.state)
    h, _ = human.action(env.state)
    drv.update(env.state, h)
    env.step((a, h))
rng = random.Random(0)
mismatch = n = 0
for lay in LAYOUTS:
    m2 = SteakHouseGridworld.from_layout_name(lay)
    e2 = OvercookedEnv.from_mdp(m2, horizon=1000, info_level=0)
    e2.reset()
    b2 = FOVFilter(m2, METHODS["handoff"].build(m2, 0, 1, {"seed": 0, "top_k": 3,
                                                           "depth": 40,
                                                           "beta": 8.0,
                                                           "rho": 0.95})[0],
                   FOVPosterior(m2, human_index=1, seed=0), agent_index=0)
    st = e2.state
    walk = set(TruthView(m2, st).walkable)
    ranked = b2.baseline.rank_subtasks(st)
    cells = sorted({c for _t, _v, c in ranked})[:12]
    spots = sorted(walk)
    for cell in cells:
        field = b2._field_to(walk, cell)
        for p in rng.sample(spots, min(len(spots), 40)):
            for o in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                n += 1
                if b2._tf(field, p, o, cell) != steps_to_finish(walk, p, o, cell):
                    mismatch += 1
check("_tf == steps_to_finish", mismatch == 0,
      "%d triples over %d layouts, %d mismatches" % (n, len(LAYOUTS), mismatch))

# ---------------------------------------------------------------------------
print("\nBONUS -- cap is what ONE sighting is worth, and the total is bounded")
D = 12
allcells = [frozenset({(x, y) for x in range(30) for y in range(30)})
            for _ in range(D + 1)]
none = [frozenset() for _ in range(D + 1)]
# THE FIXTURE HAS TO MOVE. It used to be [(3,3)] * D -- a robot standing still --
# and that is a plan with no CARRY at all, which correctly scores 0 whatever the
# cone does: _window strips the parked tail. A stationary fixture asserting a full
# score was pinning the old window (the whole horizon, parked tail included) and
# would have gone on passing while the score meant something else entirely.
path = [(k, 0) for k in range(D)]           # moves every tick: carry = D - 1 ticks
CAP, DEC = bot.cap, bot.fov_decay
BOUND = bot.max_influence
onetick = lambda k: [frozenset()] + [frozenset({path[j]}) if j + 1 == k
                                     else frozenset() for j in range(D)]

# THE DEFINITION OF THE KNOB, and the reason it is allowed to be called `cap`.
# Seen ONCE, on tick 1: decay**0 * gamma**0 == 1, so the bonus is cap exactly.
# Anything that changes the shape of the weight table has to leave this alone or
# `cap` silently starts meaning "cap times some factor nobody wrote down".
b1 = bot._bonus(path, onetick(1))[0]
check("seen once on tick 1 == cap", abs(b1 - CAP) < 1e-12,
      "bonus %.6f, cap %.6f" % (b1, CAP))
b0, k00, k10, ns0 = bot._bonus(path, none)
check("never seen == 0",
      b0 == 0.0 and k00 is None and k10 is None and ns0 == 0,
      "bonus %.6f first=%s last=%s" % (b0, k00, k10))
# A LATER SIGHTING IS WORTH LESS BY gamma, AND ONLY BY gamma. Being the first
# sighting of its plan, tick k is worth cap * gamma**(k-1) whatever k is -- the
# novelty decay is about HOW MANY came before, not about WHEN.
bad_g = [k for k in range(1, D + 1)
         if abs(bot._bonus(path, onetick(k))[0]
                - CAP * bot.gamma ** (k - 1)) > 1e-12]
check("a first sighting is cap * gamma**(k-1)", not bad_g,
      "checked ticks 1..%d, mismatches %s" % (D, bad_g or "none"))

# THE OTHER HALF OF THE NEW SEMANTICS: the j-th sighting is worth `decay` times
# the (j-1)-th. Measured as a DIFFERENCE of bonuses -- what the j-th sighting
# actually added -- with gamma = 1 so the time discount cannot be mistaken for the
# novelty one. This is the assertion that would have failed on the old score,
# where every sighting was worth the same and the decay lived in a normaliser.
bot.gamma = 1.0
ladder = [bot._bonus(path, [frozenset()]
                     + [frozenset({path[j]}) if j < n else frozenset()
                        for j in range(D)])[0]
          for n in range(D + 1)]
marginal = [ladder[n + 1] - ladder[n] for n in range(D)]
jm = bot._weights()[1]                       # sightings past j_max are worth 0
ratios = [marginal[n] / marginal[n - 1] for n in range(1, min(jm, D))]
check("the j-th sighting is decay x the (j-1)-th",
      all(abs(r - DEC) < 1e-12 for r in ratios),
      "marginals %s -- ratios %s, decay %.2f"
      % (" ".join("%.3f" % m for m in marginal[:5]),
         " ".join("%.4f" % r for r in ratios[:4]), DEC))
check("more sightings is still strictly better",
      all(marginal[n] > 0 for n in range(min(jm, D))) and ladder[0] == 0.0,
      "0 seen -> %.3f, %d seen -> %.3f, %d strictly increasing steps"
      % (ladder[0], min(jm, D), ladder[min(jm, D)], min(jm, D)))
# ...and the sum of that series is the bound, approached and never passed. At
# gamma = 1 and n sightings the closed form is cap * (1 - decay**n)/(1 - decay),
# which is the theorem's series with its tail still attached.
worst = max(abs(ladder[n] - CAP * (1 - DEC ** n) / (1 - DEC))
            for n in range(min(jm, D) + 1))
check("gamma=1 is cap*(1-decay**n)/(1-decay)", worst < 1e-12,
      "max deviation %.2e over n = 0..%d; bound %.3f, richest plan %.3f"
      % (worst, min(jm, D), BOUND, ladder[-1]))
bot.gamma = 0.9

# THE PROPERTY THE SCORE RESTS ON: the bonus SUMS sightings, it does not
# adjudicate one. The arrival tick is one sighting like any other -- not zero
# (which would say a drop seen is a drop unseen) and not special (which would make
# the score a verdict on the drop, i.e. the handoff question). At gamma = 1 the
# press tick, the first step and the middle are worth the same.
bot.gamma = 1.0
seen_at_press = bot._bonus(path, onetick(D))[0]
seen_at_step1 = bot._bonus(path, onetick(1))[0]
seen_at_mid = bot._bonus(path, onetick(D // 2))[0]
check("press tick counts the same as any other",
      seen_at_press == seen_at_step1 == seen_at_mid == CAP,
      "press %.4f  first step %.4f  middle %.4f -- each one sighting, worth cap"
      % (seen_at_press, seen_at_step1, seen_at_mid))
bot.gamma = 0.9

# THE BOUND, on random rollouts rather than on a fixture: no assignment of cones
# to a moving path can be worth more than cap/(1-decay), whatever decay is. The
# j_max clamp can only UNDER-count (sightings past it are worth zero), so it
# cannot be what makes this pass -- which is why the high-decay row, where j_max
# hits `depth` and nothing is clamped at all, is in the sweep.
worst_frac = 0.0
for dec in (0.2, 0.5, 0.8):
    bot.fov_decay = dec
    for trial in range(200):
        rng2 = random.Random(trial)
        cones = [frozenset({(rng2.randrange(4), 0)}) for _ in range(D + 1)]
        pth = [(rng2.randrange(4), 0) for _ in range(D)]
        b, _f, _l, _n = bot._bonus(pth, cones)
        worst_frac = max(worst_frac, b / bot.max_influence)
        if b > bot.max_influence + 1e-9:
            worst_frac = float("inf")
bot.fov_decay = DEC
check("bonus <= cap/(1-decay) on random rollouts", worst_frac <= 1.0,
      "worst plan reached %.1f%% of its bound, over 3 decays x 200 rollouts"
      % (100.0 * worst_frac))
# and the whole trajectory in view is worth nearly the bound, so the bound is
# TIGHT rather than merely true -- a bound nothing approaches is not a bound, it
# is a number.
b_all = bot._bonus(path, allcells)[0]
check("and a fully-seen plan approaches it", 0.5 * BOUND < b_all <= BOUND + 1e-9,
      "seen on all %d ticks: %.3f of a %.3f bound" % (D, b_all, BOUND))

# every live tick, from the readouts: slot 7 is the FoV half of the cost, so it
# pins the bonus to [0, max_influence] on real plans rather than on a fixture.
_, infos = episode("chefs_table", kind="fov", seed=0, ticks=200)
bad_s = 0
for i in infos:
    lim = i.get("max_influence", BOUND)
    for row in i.get("cands", []):
        if not (-lim - 0.05 <= row[7] <= 1e-9):
            bad_s += 1
    if not i.get("gated") and "fov_bonus" in i:
        if not (-lim - 1e-9 <= i["fov_bonus"] <= 1e-9):
            bad_s += 1
check("bonus in [0, cap/(1-decay)] on live ticks", bad_s == 0,
      "%d plans over %d ticks, bound %.2f"
      % (sum(len(i.get("cands", [])) for i in infos), len(infos), BOUND))

# ---------------------------------------------------------------------------
print("\nGUARD_III -- the deviation budget is a hard bound on wall-clock authority")
def longest_run(infos):
    run = best = 0
    for i in infos:
        run = run + 1 if i.get("deviated") else 0
        best = max(best, run)
    return best

_, inf_bounded = episode("chefs_table", baseline_key="handoff", cap=1e6,
                         stall_max=6, seed=0, ticks=300)
run_b = longest_run(inf_bounded)
stalled = sum(1 for i in inf_bounded if i.get("stalled"))
check("<= stall_max consecutive", run_b <= 6,
      "longest deviating run %d, stall fired %d times" % (run_b, stalled))
_, inf_free = episode("chefs_table", baseline_key="handoff", cap=1e6,
                      stall_max=10 ** 9, seed=0, ticks=300)
run_f = longest_run(inf_free)
check("guard III is load-bearing", run_f > 6,
      "with the budget removed the run reaches %d -- the failure mode is real"
      % run_f)

# ---------------------------------------------------------------------------
print("\nNO_TASK_KNOWLEDGE -- nothing downstream of tail_ticks leaked back in")
src = open(FOV_SRC).read()
doc = ast.get_docstring(ast.parse(src)) or ""
body = src.replace(doc, "", 1)            # the DROPPED list lives in the docstring
banned = ["value_tail", "progress", "tail_ticks", "orders_remaining",
          "blind", "no_handoff"]
hits = [w for w in banned if w in body]
check("no banned identifier in code", not hits, "found: %s" % (hits or "none"))
probe = ("import sys, os\n"
         "sys.path.insert(0, %r)\n"
         "sys.path.insert(0, %r)\n"
         "import robot.filter.core.fov_filter as f\n"
         "print([m for m in sys.modules if 'value_tail' in m or "
         "m.endswith('core.progress') or 'qmdp' in m])\n"
         % (os.environ.get("STEAK_ROOT",
                           "/Users/mishafu/Desktop/obs_function/steakhouse"),
            _MISHA))
out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                     text=True)
check("imports nothing it dropped", out.stdout.strip() == "[]",
      "sys.modules after a bare import: %s" % out.stdout.strip())

# ---------------------------------------------------------------------------
print("\nINFO -- every key the harnesses read, on every row of the table")
NEEDED = ["gated", "p_map", "base_subtask", "jobs", "cands", "n_mask",
          "n_states", "plan", "plan_t", "Q", "gain", "subtask", "top3",
          "top3_kind", "theta_entropy", "fov_post", "deviated", "Q_action",
          "action_dist"]
rows = [k for k in METHODS if k.startswith("fov")]
missing, widths = {}, set()
for k in rows:
    _, infos = episode("butchery", kind=k, seed=0, ticks=40)
    # FOUR ways to be inert, not two: the certainty gate, no reachable candidate,
    # and the two refusals -- pi does not price the baseline's own draw, or the
    # draw is not a row of this tick's ranking. All four return before a single
    # plan is scored, so none of them carries the readouts, and they belong in this
    # list for exactly the reason the first two do -- adding a guard without adding
    # it here failed this check on fov-bayes and on nothing else.
    live = [i for i in infos if not i.get("gated") and not i.get("no_reach")
            and not i.get("pi_no_ref") and not i.get("stale_pick")]
    for i in live:
        for key in NEEDED:
            if key not in i:
                missing.setdefault(k, set()).add(key)
        for row in i.get("cands", []):
            widths.add(len(row))
check("all keys present", not missing,
      "%d registry rows, %s" % (len(rows), missing or "nothing missing"))
check("cands rows are 8 wide", widths == {8},
      "widths seen: %s -- watch.py:308/407/465 unpack 8 names" % sorted(widths))

# ---------------------------------------------------------------------------
print("\nDETERMINISM -- same seed, byte-identical traces")
det = True
for k in ("fov", "fov-free", "fov-bayes"):
    a1, _ = episode("butchery", kind=k, seed=0, ticks=120, keep=False)
    a2, _ = episode("butchery", kind=k, seed=0, ticks=120, keep=False)
    det = det and a1 == a2
check("two runs agree", det, "fov / fov-free / fov-bayes, 120 ticks each")

print("\n%s  (%d failures)"
      % ("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS), len(FAILS)))
sys.exit(1 if FAILS else 0)
