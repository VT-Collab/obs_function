"""Checks for the LOOK-FOR half of human/limited_vision_human.py. Run it directly:

    ~/miniconda3/envs/steakhouse-ai/bin/python human/tests/test_look_for_human.py

The file keeps its name because the FEATURE keeps its name; what it no longer
has is a module of its own. The look-for was human/look_for_human.py and a
subclass, and it is now a flag on the one human class -- see ONE FILE, ONE CLASS
in limited_vision_human.py for the forecast bug that split made possible, which
is the reason for the merge and the reason nothing here names a class twice.

Eight of these carry the feature's actual claims. The rest are guard rails.

  CONTROL          enable_look_for=False is the plain ladder, untouched: no
                   sighting is ever armed, no look_for verb is ever chosen, and
                   WATCHING the agent does not steer it. It is first because it
                   is the A/B control every number this feature produces will be
                   measured against. See the section itself for why the old
                   two-class comparison could not survive the merge and what
                   pins that claim now.
  TIER_TABLE       LOOK_TIER and KEEP_TIER are RE-DERIVED by executing
                   legal_subtasks, not restated. A change to the ladder must
                   break this test rather than silently misprice every look-for.
  EXAMPLES         the cases the feature was asked for, DRIVEN THROUGH observe()
                   on real states, including the ones that must produce NO
                   look-for at all.
  GATE_HAS_POWER   with lookahead_gate=False and a cooking pot, example 2 does
                   NOT fire. Without this the relaxation looks like decoration
                   and somebody deletes it.
  MEMORY           the sighting survives views.py:200 nulling the position half
                   of the robot belief -- the datum the whole feature needs and
                   the one the belief throws away.
  PURITY           rank() and decide() are queries. Asking either of them must
                   not change what the agent then does, and must not edit the
                   sighting book. Both were false, and both failures were
                   INVISIBLE to every other test in this file, which is why they
                   get their own section rather than an assert inside one.
  PAYOFF           look-for bouts END IN THE HUMAN ACTUALLY GETTING THE THING at
                   a non-trivial rate. This is the only test that asks whether
                   the feature works rather than whether it is wired up, and it
                   is the one that was missing.
  NO_PICK_BACK_UP  nothing stashed to free a hand for a look-for is picked back
                   up while that same look-for is still being run.

--------------------------------------------------------------------------------
WHY THE EXAMPLES GO THROUGH observe() AND NOT THROUGH h.view
--------------------------------------------------------------------------------
They used to write a belief straight into the human's BeliefView, set
_cleared[item] = set() by hand and call decide() against a stub state. The
argument was that decide() reads only the view and two player fields, so the stub
was exact. It was not, and the way it was wrong is the entire reason defect 1
survived a green suite: observe() is the ONLY thing that populates _cleared, so
handing decide() an empty cleared set tested a state the agent can never occupy.
On a real cone that set arrives at decide() already holding two thirds of the
kitchen -- 7.5 of 11.2 known counters, measured -- and the sighting was retired
before it was ever used. Every example below passed while the feature had a zero
success rate in a rollout.

So the examples now build a REAL SteakHouseGridworld from _GRID, construct real
OvercookedStates, and drive the human tick by tick through observe(). Nothing is
written into the belief by hand; every station the human knows about, it looked
at, and every counter in its cleared set got there through the cone.

_GRID is purpose-built and its shape is load-bearing in three places:

  TWO ROOMS, ONE PASS. Column x=10 is solid counter from y=1 to y=5, so the two
  floors are not connected: the human can SEE the whole of the robot's room
  through the see-through counters and can WALK to exactly one cell of it,
  (10,4). That is the geometry every layout in layout/layouts has, and it is what
  makes "reachable" and "visible" come apart, which several tests below need.
  DISPENSARIES BEHIND WALLS. M/O/D sit in an alcove whose only opening is (7,3),
  so "the human has not found the meat dispenser yet" is produced by the cone
  rather than asserted -- observe from (3,4) and they are undiscovered, observe
  from (7,4) and they are found. That is the discovery channel, which is the base
  agent's strongest, and the examples turn on it.
  A COUNTER WE CAN SEE AND CANNOT REACH. (10,5) touches only the robot's floor.

NOTE ON LAYOUTS. layout/layouts/divide.layout is omitted from the rollout sweeps
because SteakHouseGridworld._assert_valid_grid rejects it as a Ragged grid --
that is true on main and has nothing to do with this change.
"""
import itertools
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import overcooked_ai_py                                              # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "layout", "layouts")

from overcooked_ai_py.mdp.actions import Action                      # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv        # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import (                    # noqa: E402
    SteakHouseGridworld, OvercookedState, PlayerState, ObjectState)
from common.tasks import (legal_subtasks, POT, BOARD, SINK, MEAT,    # noqa: E402
                          ONION, PLATE, SERVE, COUNTER)
# ONE IMPORT, ONE CLASS. The look-for used to live in human/look_for_human.py as
# a subclass and this file imported both; the merge is what makes "the agent in
# the seat" and "the agent the robot's filter simulates" the same object by
# construction rather than by every caller remembering to say so.
from human.limited_vision_human import (LimitedVisionHuman,          # noqa: E402
                                        LOOK_TIER, KEEP_TIER,
                                        LOOK_HORIZON)
from robot.nominal_policy.baselines import BASELINES                 # noqa: E402

LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "pantry"]
FOVS = [30, 60, 90, 180, 360]
ITEMS = ["dish", "steak_dish", "garnish_dish", "washed_plate", "garnish",
         "meat", "onion", "plate"]

FAILS = []


def check(name, ok, detail=""):
    print("  %-4s %-30s %s" % ("ok" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


# =============================================================================
# a rollout harness, shared by every property test below
# =============================================================================
def rollout(layout, fov, seed, ticks=250, spectate=False, **kw):
    """One episode against the handoff baseline. Returns (human, actions, rows).

    The robot is `handoff` rather than a filter because this file is about the
    HUMAN: handoff actually puts things on the pass, which is what gets seen in
    the robot's hands, and it costs nothing per tick.

    NO `human_cls` PARAMETER ANY MORE, and its absence is the point of the merge
    rather than a tidy-up: there is one human class, so `enable_look_for` in **kw
    is the only thing that separates the feature from its control. Every place
    that used to choose a CLASS -- here, both harnesses, FOVPosterior's shadows --
    could get that choice wrong independently, and FOVPosterior's did.

    `spectate` asks the human what it would do BEFORE stepping it, which is what
    a filter shadow or a likelihood term does. It must not change the trace; see
    the CONTROL and PURITY sections.
    """
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=ticks + 5, info_level=0)
    env.reset()
    h = LimitedVisionHuman(mdp, fov, agent_index=1, seed=seed, **kw)
    # beta=inf, rho=1: a predictable, reproducible partner. handoff DRAWS its
    # sub-task by default (baselines.py), and this file's checks are calibrated
    # against the one it always used to make -- see the module docstring above.
    bot = BASELINES["handoff"](mdp, agent_index=0, seed=seed,
                               beta=float("inf"), rho=1.0)
    acts, rows = [], []
    for _ in range(ticks):
        if mdp.is_terminal(env.state):
            break
        a, _ = bot.action(env.state)
        mine = env.state.players[1].held_object
        if spectate:
            h.decide(env.state)              # "what would you do?", pre-observe
        b, info = h.action(env.state)
        acts.append((a, b))
        tier, verb, cell = info["subtask"]
        obj = env.state.objects.get(cell) if cell is not None else None
        # SNAPSHOT PER TICK, not once at the end. The internals below are
        # single-tick facts -- _look_target in particular is cleared the moment
        # no look-for wins -- and reading them after the episode measured the
        # last tick 300 times over, which quietly reported zero drop legs on
        # episodes that had a hundred of them.
        rows.append(dict(
            tier=tier, verb=verb, cell=cell, act=b,
            held=mine.name if mine else None,
            cell_obj=obj.name if obj else None,
            robot=h.view.robot,
            # no getattr() guard: every human HAS a sighting book now, and with
            # enable_look_for=False it is simply always empty. That is a stronger
            # statement than "the base class does not have the attribute", and
            # the CONTROL section below checks it rather than assuming it.
            sight=dict(h.sightings),
            look=h._look_target,
            # what the belief says it knows the location of, which is the event
            # a bout is trying to produce: S2 withdraws the look-for the instant
            # counters_holding goes non-empty.
            known={i: bool(h.view.counters_holding(i)) for i in ITEMS},
        ))
        env.step((a, b))
    return h, acts, rows


def subtasks(rows):
    return [(r["tier"], r["verb"], r["cell"]) for r in rows]


# =============================================================================
print("\nCONTROL -- enable_look_for=False must be the plain ladder, untouched")
# The single most load-bearing configuration here. Every headline number this
# feature produces is a PAIRED difference against exactly this one.
#
# WHAT CHANGED WHEN THE TWO CLASSES BECAME ONE. This used to read as the base
# class against the look-for subclass with the feature off: two classes, so the
# base one WAS the reference and the comparison was live. There is no second
# class now, and the reason is in limited_vision_human.py under ONE FILE, ONE
# CLASS -- the robot's filter builds SHADOW humans to predict the cone with, the
# shadows were built as the base class while the seat held the subclass, and the
# whole forecast was therefore about a different agent from the one playing.
#
# So "identical to the agent that existed before the look-for" became a claim
# about a CODE CHANGE, and it was pinned where such claims belong: at the merge
# itself, by 60 md5 action-trace hashes over {look-for on, off} x 5 layouts x 3
# cones x 2 seeds x 150 ticks, recorded against the two-class code and re-run
# against the merged class. All 60 matched, the 30 "off" ones being exactly this
# claim. A frozen copy of those hashes is NOT kept here on purpose: it would
# turn every legitimate change to the ladder into a failure with no live
# reference to regenerate from, which is a worse test than none.
#
# What is testable forever is the PROPERTY THAT MADE THOSE HASHES MATCH, and
# that is what this section checks: with the feature off, none of the machinery
# the merge brought in can reach the trace. Nothing ever arms, nothing is ever
# committed to, no look_for verb is ever chosen, and the rank() memo -- the one
# addition that also runs in the control -- cannot answer from a stale belief
# even when something is watching. That last clause is the only one with a live
# failure mode, and it is the failure the merge could most plausibly have
# introduced: the memo is keyed on a belief version, and a control that someone
# ASKS is the only way to reach it.
# =============================================================================
bad, leaked = [], []
for lay in LAYOUTS:
    for fov in FOVS:
        for seed in range(2):
            h, a1, r1 = rollout(lay, fov, seed, enable_look_for=False)
            # NOTHING LOOKED FOR, NOTHING REMEMBERED, NOTHING COMMITTED TO -- on
            # every tick, not just at the end. observe() returns before it can
            # write a sighting, so an entry here means the early return moved.
            if (any(r["sight"] or r["look"] is not None for r in r1)
                    or any(r["verb"].startswith("look_for_") for r in r1)
                    or h.sightings or h._cleared or h._drop_target is not None):
                leaked.append("%s/fov%s/seed%s" % (lay, fov, seed))
            # ...and being WATCHED does not steer it. Same episode, one
            # speculative decide() per tick: ask, observe, ask again, which is
            # the only shape in which a wrongly-keyed memo gets HIT rather than
            # merely filled. Keyed on self.t -- which action() bumps at the end
            # of the tick, not at the observation -- the second ask answered
            # from the first ask's pre-observation belief, and the agent acted a
            # tick stale exactly when something was looking at it.
            _, a2, r2 = rollout(lay, fov, seed, enable_look_for=False,
                                spectate=True)
            if a1 != a2 or subtasks(r1) != subtasks(r2):
                bad.append("%s/fov%s/seed%s" % (lay, fov, seed))
n_cfg = len(LAYOUTS) * len(FOVS) * 2
check("CONTROL_IS_INERT", not leaked,
      "%d configs, %d with look-for state in the control %s" % (n_cfg,
                                                                len(leaked),
                                                                leaked[:3]))
check("CONTROL_IS_STABLE_UNDER_WATCHING", not bad,
      "%d configs, %d mismatches %s" % (n_cfg, len(bad), bad[:3]))

# and the control must not be vacuous: with the feature ON the trace has to
# differ somewhere, or "identical to the plain ladder" is a claim about a
# feature that never fires.
diff = 0
for lay in LAYOUTS:
    for fov in FOVS:
        _, a1, _ = rollout(lay, fov, 0, enable_look_for=False)
        _, a2, _ = rollout(lay, fov, 0)
        diff += int(a1 != a2)
check("CONTROL_HAS_POWER", diff > 0,
      "%d of %d configs change when the feature is ON" % (diff,
                                                          len(LAYOUTS) * len(FOVS)))


# =============================================================================
print("\nTIER_TABLE -- re-derive LOOK_TIER and KEEP_TIER by EXECUTING the ladder")
# The tables are the whole pricing model, and they are a transcription of
# tasks.py:375-378 plus four held-branch clauses. Restating them here would test
# nothing. Instead: build a view in which every station is found and reachable
# and ONE counter holds the item, ask legal_subtasks what it offers at that
# counter, and require the answer to be the number the module hardcodes.
# =============================================================================
class _FakeView:
    """Everything found and reachable; counter (9,9) holds `item`.

    The one hand-built view left in this file, and it is legitimate because it is
    not standing in for a belief: nothing here is asking what the AGENT would do,
    only what the LADDER prices an item at, and the ladder cannot tell a view
    from a stub. Every test that asks about the agent goes through observe().

    forget_horizon is present because tasks._counts_composites duck-types on it
    to decide whether composites count toward saturation -- the human hands in a
    BeliefView, so this has to read as one or the surplus rule differs.
    """
    forget_horizon = 30
    _CELLS = {POT: [(1, 1)], BOARD: [(2, 1)], SINK: [(3, 1)], MEAT: [(4, 1)],
              ONION: [(5, 1)], PLATE: [(6, 1)], SERVE: [(7, 1)],
              COUNTER: [(9, 9), (9, 8)]}

    def __init__(self, item, status):
        self.item, self.status = item, status

    def stations(self, ch):
        return list(self._CELLS.get(ch, []))

    def ready(self, ch):
        return self.stations(ch) if self.status == "ready" else []

    def in_progress(self, ch):
        return self.stations(ch) if self.status == "cooking" else []

    def empty(self, ch):
        return self.stations(ch) if self.status == "empty" else []

    def counters_holding(self, name):
        return [(9, 9)] if name == self.item else []

    def free_counters(self):
        return [(9, 8)]


# the station state under which each item is worth having at all. The raw three
# need an EMPTY machine, everything else needs a FINISHED one -- that is
# actionable() read backwards, and getting it wrong is what makes a naive
# derivation of this table come back empty for meat/onion/plate.
_GATING = {"dish": "ready", "steak_dish": "ready", "garnish_dish": "ready",
           "washed_plate": "ready", "garnish": "ready",
           "meat": "empty", "onion": "empty", "plate": "empty"}

drop_derived = {}
for y in ITEMS:
    v = _FakeView(y, _GATING[y])
    tiers = sorted({t for (t, _, c) in legal_subtasks(v, None) if c == (9, 9)})
    if tiers:
        drop_derived[y] = tiers[0]
check("DROP_TIER", drop_derived == LOOK_TIER, str(drop_derived))

keep_derived = {}
for held in ITEMS:
    for y in ITEMS:
        v = _FakeView(y, _GATING[y])
        tiers = sorted({t for (t, _, c) in legal_subtasks(v, held) if c == (9, 9)})
        if tiers:
            keep_derived[(held, y)] = tiers[0]
check("KEEP_TIER", keep_derived == KEEP_TIER, str(sorted(keep_derived.items())))

# The loop-safety argument in the module docstring rests on this and nothing
# else: putting something down must always make it WORSE, never better, or the
# human can stash an item to free a hand and be offered it straight back at a
# tier that beats the look-for that outranked it.
worse = all(LOOK_TIER[y] > keep for (h, y), keep in KEEP_TIER.items())
check("DROP_IS_WORSE_THAN_KEEP", worse,
      "empty-handed take is strictly worse than every in-hand counter verb")


# =============================================================================
print("\nEXAMPLES -- the cases asked for, driven through observe() on real states")
# =============================================================================
#      x: 0  1  2  3  4  5  6  7  8  9 10 11 12 13
_GRID = ["##############",
         "#PBWS#OMD#X X#",
         "#    #   #X  #",
         "#    ## ##X  #",
         "#1        X2 #",
         "#XXXXXXXXXX X#",
         "##############"]
# M sits at (7,1), straight above the alcove's only opening, so it is the
# NEAREST of the three dispensers from anywhere in the main room. That is not
# decoration: EX1_never_hunt_when_you_can_fetch asserts the exact verb, and with
# the meat dispenser further in than the onion one the ladder would answer
# get_onion -- true to the claim being made, but a weaker sentence to read.
# the three stations that can HOLD something, and so are the ones _state has to
# be able to fill. The serving hatch at (4,1) never holds anything; it only has
# to exist, or actionable("dish") is False and EX4 stops meaning what it says.
_STATION = {"pot": (1, 1), "board": (2, 1), "sink": (3, 1)}
_HOME = (3, 4)              # sees the stations, the counters and the robot;
#                             the dispensary alcove is walled off from here
_DISPENSARY = (7, 4)        # ...and from here the alcove IS in the cone
_RPOS = (11, 4)             # the robot, across the pass, visible not reachable
_PASS = (10, 4)             # the one counter BOTH agents can reach: L1 1 from
#                             the robot, so the default look target
_FAR = (10, 5)              # seen through the counters, reachable only by the
#                             robot -- the S2_unreachable case
_NEAR = (9, 5)              # ours, L1 3 from the robot: inside the look radius

EMPTY, COOKING, READY = None, "cooking", "ready"

_MDP = SteakHouseGridworld.from_grid(
    _GRID, base_layout_params=dict(start_order_list=["steak"] * 3,
                                   cook_time=15, chop_time=5, wash_time=5))
_IDS = itertools.count(1)

# what to put on a station to make it read EMPTY / COOKING / READY, as the mdp
# spells it. The ready thresholds are the mdp's own parameters rather than the
# numbers above, so retuning the layout cannot silently make "ready" mean
# "cooking" -- which would take GATE_HAS_POWER with it and leave it green.
_OCCUPANT = {
    "pot": ("steak", lambda m: ("steak", 1, 0), lambda m: ("steak", 1,
                                                           m.steak_cooking_time)),
    "board": ("garnish", lambda m: 0, lambda m: m.chopping_time),
    "sink": ("washed_plate", lambda m: 0, lambda m: m.wash_time),
}


def _obj(name, pos, st=None):
    return ObjectState(next(_IDS), name, pos, st)


def _state(hpos, hheld, rheld, stations=(), counters=(), facing=(0, 1)):
    """A real OvercookedState. players[1] is the human -- agent_index=1.

    `stations` is [(name, COOKING|READY)], `counters` is [(cell, item)]. An
    absent station is EMPTY, which is a genuine observation and not the same
    thing as unknown -- the point of driving this through observe() is that the
    difference survives.
    """
    objs = []
    for name, status in stations:
        obj_name, cooking, ready = _OCCUPANT[name]
        st = (ready if status == READY else cooking)(_MDP)
        objs.append(_obj(obj_name, _STATION[name], st))
    for cell, item in counters:
        objs.append(_obj(item, cell))
    players = [PlayerState(_RPOS, (-1, 0), _obj(rheld, _RPOS) if rheld else None),
               PlayerState(hpos, facing, _obj(hheld, hpos) if hheld else None)]
    return OvercookedState(players, {o.position: o for o in objs},
                           order_list=["steak"] * 3)


def _agent(fov=360, **kw):
    """The human on the example grid, at fov 360 unless asked otherwise.

    360 so that what the examples know is decided by WALLS and by where the human
    has stood, never by which way it happened to be facing -- the alcove is the
    only thing hiding the dispensers, and that is deliberate: it makes the
    discovery channel the one variable the examples turn on. The cleared-set
    tests below want the opposite and ask for a narrow cone explicitly.
    """
    return LimitedVisionHuman(_MDP, fov, agent_index=1, seed=0, **kw)


def _drive(h, frames):
    """observe() every frame in order, one tick each; return the LAST decision.

    This is action()'s own loop with the walking taken out -- observe, decide,
    advance the clock -- so every belief the decision is made on arrived through
    the cone, and the cleared sets are whatever a real sweep would have built.
    """
    out = None
    for i, s in enumerate(frames):
        h.observe(s)
        out = h.decide(s)
        # the commitments action() would have written. decide() is a pure query
        # and deliberately does not, so a multi-frame drive has to do it here or
        # it is testing an agent with no stickiness.
        choice, look, drop = h._choose(s)
        h._look_target, h._drop_target = look, drop
        h.last_subtask = None if choice is None else (None, choice[4], choice[3])
        h.t += 1
    return out


def _verb(h, frames):
    return _drive(h, frames)[4]


def _sight(hheld=None, rheld="meat", hpos=_HOME, **kw):
    """One frame: the human at home, the robot across the pass holding rheld."""
    return _state(hpos, hheld, rheld, **kw)


# EX1. "Robot has meat, human is doing onions -> carry on with the onions; when
# the onions are done and there is nothing else left -> go and look for it."
h = _agent()
check("EX1_carry_on_with_onions",
      _verb(h, [_sight("onion")]) == "load_board",
      "look_for_meat is T7 and load_board is T6, so it cannot fire")

# the onion chain is exhausted and no dispenser has been found -- the alcove has
# never been in the cone -- so the ladder has genuinely nothing: it falls to
# EXPLORE (T9) and the T7 look-for wins.
h = _agent()
check("EX1_then_look_for_meat", _verb(h, [_sight()]) == "look_for_meat",
      "ladder empty -> T9; look_for_meat T7 preempts")

# and the bonus that comes free with STRICT preemption: never hunt for something
# you could simply walk up and fetch. Both are T7, and equal is not better. The
# only difference from the test above is one tick spent standing at (7,4), where
# the alcove is visible -- the dispensers are DISCOVERED, not conjured.
h = _agent()
got = _verb(h, [_sight(hpos=_DISPENSARY), _sight()])
check("EX1_never_hunt_when_you_can_fetch", got == "get_meat",
      "%s -- T7 vs T7, strict comparison keeps the dispenser" % got)

# EX2. "Robot has a garnish_dish, human is fetching an onion or getting a plate
# -> drop it and go and look." Fires on a pot that is merely COOKING, which is
# the whole job of the look-ahead relaxation.
for held in ("onion", "plate"):
    h = _agent()
    first = _verb(h, [_sight(held, "garnish_dish", stations=[("pot", COOKING)])])
    second = _verb(h, [_sight(None, "garnish_dish", stations=[("pot", COOKING)])])
    check("EX2_drop_then_look_holding_%s" % held,
          first == "stash" and second == "look_for_garnish_dish",
          "%s -> %s" % (first, second))

# EX3. "Robot has a garnish_dish but the human wanted to do meat -> do the meat."
# Free, from the gate alone: wanting to do meat MEANS believing a pot is empty,
# and an empty pot is neither ready nor in progress, so the look-ahead gate fails
# and the tier is never even computed.
h = _agent()
a = _verb(h, [_sight("meat", "garnish_dish")])
b = _verb(h, [_sight(None, "garnish_dish")])
check("EX3_do_the_meat", a == "load_pot" and not b.startswith("look_for_"),
      "%s / %s" % (a, b))

# EX4. "Robot has a dish -> drop whatever you are doing, unless it is delivering
# a dish." Read as "drop anything LESS URGENT than the dish itself": T2 does not
# preempt T2, so a human one INTERACT away from making its own dish finishes.
h = _agent()
a = _verb(h, [_sight("washed_plate", "dish", stations=[("pot", READY)])])
b = _verb(h, [_sight(None, "dish", stations=[("pot", READY)])])
check("EX4_divert_for_a_dish", a == "stash" and b == "look_for_dish",
      "%s -> %s" % (a, b))

h = _agent()
check("EX4_carve_out_holding_a_dish",
      _verb(h, [_sight("dish", "dish", stations=[("pot", READY)])]) == "deliver",
      "NEVER_DROP is unconditional")

# the KEEP branch: the four pairs where a held-branch verb already targets a
# counter, so there is nothing to put down and the walk starts immediately.
h = _agent()
check("KEEP_no_drop_needed",
      _verb(h, [_sight("steak_dish", "garnish")]) == "look_for_garnish",
      "steak_dish + garnish is a T2 combine, so no stash leg at all")

# EX5, and it is the one the user stated as a rule rather than a scene: IF YOU
# ALREADY SAW THAT ITEM PLACED DOWN, DO NOT LOOK FOR IT. Two ticks, and the
# second is the robot having let go -- so this also exercises the release path
# that the clearing rule now turns on.
h = _agent()
got = _verb(h, [_sight(rheld="meat"),
                _state(_HOME, None, None, counters=[(_NEAR, "meat")])])
check("EX5_do_not_look_for_what_you_watched_it_put_down", got == "take_meat",
      "%s -- the ordinary ladder takes over at the very tier the look-for "
      "borrowed" % got)

# S2 past the literal brief: the item is on a counter we know about and cannot
# reach -- (10,5) touches only the robot's floor. Default ON suppresses, because
# "I know where it is and cannot get to it" answers the question the look-for
# asks and walking over to re-confirm it is pure waste. suppress_unreachable
# =False is the literal rule and keeps looking.
for flag, want in ((True, "stash"), (False, "look_for_garnish")):
    h = _agent(suppress_unreachable=flag)
    got = _verb(h, [_sight("steak_dish", "garnish",
                           counters=[(_FAR, "garnish")])])
    check("S2_unreachable_suppress_%s" % flag, got == want,
          "%s (wanted %s)" % (got, want))


# =============================================================================
print("\nTHE CLEARED SET -- the defect that hid behind a hand-written belief")
# The failure was silent and total: the sought item is BY DEFINITION in the
# robot's hands, so every counter in the cone truthfully reports "not this", the
# candidate set came back empty on the tick the sighting was written, and the
# sighting was deleted. Two tests, because the two halves of the fix are
# independent and either one alone leaves it broken.
# =============================================================================
# THE HEADLINE. A whole 360 cone on twelve counters, every one of them visibly
# NOT holding a meat, and the sighting must come out of the tick alive with an
# EMPTY cleared set -- because the meat is in the robot's hands and none of those
# twelve counters was ever a candidate to rule out. This is the tick the old code
# died on, and it died harder the better the human could see.
h = _agent()
s = _sight()
h.observe(s)
h.decide(s)
check("SIGHTING_SURVIVES_ITS_OWN_BIRTH_TICK",
      h.sightings.get("meat") is not None and not h._cleared["meat"],
      "sighting %s, %d of %d counters in the cone cleared (the old code cleared "
      "7.5 of 11.2 on average and then deleted the sighting)"
      % (h.sightings.get("meat"), len(h._cleared.get("meat", ())),
         len(h.view.stations(COUNTER))))

# Now the two conditions ONE AT A TIME, which needs a cone narrow enough to look
# at one counter at a time. From (9,4) a 30-degree cone sees exactly _NEAR facing
# down, and exactly _PASS plus the robot facing right -- so each tick below is a
# single, nameable observation and nothing else.
_SPOT = (9, 4)
_DOWN, _RIGHT = (0, 1), (1, 0)
h = _agent(fov=30)
h.observe(_state(_SPOT, None, None, facing=_DOWN))       # t0: _NEAR seen empty
h.t += 1
h.observe(_state(_SPOT, None, "meat", facing=_RIGHT))    # t1: armed, _PASS seen
check("HELD_BLOCKS_CLEARING", not h._cleared.get("meat"),
      "%s was looked at this very tick and is empty, and clears nothing: "
      "the meat is in the robot's hands" % (_PASS,))

h.t += 1
h.observe(_state(_SPOT, None, None, facing=_RIGHT))      # t2: it let go
done = h._cleared.get("meat", set())
check("PRE_SIGHTING_LOOKS_DO_NOT_CLEAR",
      _PASS in done and _NEAR not in done and "meat" in h.sightings,
      "cleared %s -- %s was looked at after the sighting, %s before it, and "
      "both records are still fresh" % (sorted(done), _PASS, _NEAR))

h.t += 1
h.observe(_state(_SPOT, None, None, facing=_DOWN))       # t3: _NEAR re-checked
check("SWEPT_AREA_RETIRES", "meat" not in h.sightings,
      "both counters in reach of the sighting have now been looked at since it "
      "was armed, so it retires -- and only now")


# =============================================================================
print("\nGATE_HAS_POWER -- the look-ahead relaxation is the ONLY thing carrying EX2")
# Without this check the relaxation reads as decoration and the next person
# deletes it. With lookahead_gate=False and a pot that is merely cooking,
# actionable('garnish_dish') is False and example 2 does not happen at all.
# =============================================================================
frame = [_sight("onion", "garnish_dish", stations=[("pot", COOKING)])]
off = _verb(_agent(lookahead_gate=False), frame)
on = _verb(_agent(lookahead_gate=True), frame)
check("GATE_HAS_POWER", off == "load_board" and on == "stash",
      "gate off -> %s, gate on -> %s" % (off, on))

# gate_composites=False is the documented escape hatch for the hole EX2 has when
# nothing at all is cooking. It must fire there, and it must still be loop-safe.
check("GATE_COMPOSITES_ESCAPE",
      _verb(_agent(gate_composites=False),
            [_sight("onion", "garnish_dish")]) == "stash",
      "exempting the composites fires on an empty pot, as documented")


# =============================================================================
print("\nCHAIN_GUARD -- the optional predicate, tested as a predicate")
# chain_guard makes example 3 literal in a MULTI-POT layout, and there is no
# multi-pot layout in this suite, so there is no behavioural case to assert --
# the design says so and this test says so rather than inventing a kitchen to
# hide it. What can be pinned down is the predicate itself and the fact that
# switching the flag on changes nothing in the single-pot examples above, which
# is what makes it safe to leave as an option.
# =============================================================================
g = _agent(chain_guard=True)
cases = [("garnish_dish", "load_pot", False),    # the EX3 divergence itself
         ("garnish_dish", "load_board", True),   # onion chain -> garnish_dish
         ("garnish_dish", "load_sink", True),    # plate chain -> garnish_dish
         ("dish", "load_pot", True),             # a dish is downstream of all
         ("washed_plate", "load_pot", False),    # the legitimate block, named
         ("meat", "deliver", True)]              # unmapped verb: no opinion
check("CHAIN_GUARD_PREDICATE",
      all(g._chain_ok(item, verb) is want for item, verb, want in cases),
      str([(i, v, g._chain_ok(i, v)) for i, v, _ in cases]))

a = _verb(_agent(chain_guard=True),
          [_sight("onion", "garnish_dish", stations=[("pot", COOKING)])])
b = _verb(_agent(chain_guard=True), [_sight("meat", "garnish_dish")])
check("CHAIN_GUARD_IS_INERT_HERE", a == "stash" and b == "load_pot",
      "single pot: EX2 %s, EX3 %s -- unchanged with the guard on" % (a, b))


# =============================================================================
print("\nPURITY -- rank() and decide() are QUERIES, and both used to lie")
# Neither of these is visible in a rollout, because a rollout only ever calls
# action(). They are visible to every OTHER caller -- my_fov_filter.py's shadow
# rollouts, the filter's likelihood terms, any probe -- and that is exactly the
# population this agent is built to be used by.
# =============================================================================
# DEFECT 2. The memo was keyed on self.t, which action() increments at the END of
# a tick while observe() rewrites the belief at the start of the next one. So a
# caller asking decide() before action() filled the memo from last tick's belief
# under this tick's key, and action()'s own decide() hit it and acted stale. The
# THE CONTROL IS THE SAME CLASS WITH THE FEATURE OFF, not a different class any
# more, and it is still an honest control: observe() bumps the belief version
# between the speculative ask and the real one, so on this path the memo can
# never be hit and the count is the number that is REALLY there. If that stops
# being true this test stops meaning anything, which is why it is spelled out.
def _memo_probe(**kw):
    mdp = SteakHouseGridworld.from_layout_name("banquet_pass")
    env = OvercookedEnv.from_mdp(mdp, horizon=250, info_level=0)
    env.reset()
    h = LimitedVisionHuman(mdp, 90, agent_index=1, seed=0, **kw)
    bot = BASELINES["handoff"](mdp, agent_index=0, seed=0,
                               beta=float("inf"), rho=1.0)
    n = 0
    for _ in range(200):
        a, _ = bot.action(env.state)
        pre = h.decide(env.state)          # the speculative ask
        act, info = h.action(env.state)    # observe() then decide() for real
        n += int(info["subtask"][1] != pre[4])
        env.step((a, act))
    return n


base_n, look_n = _memo_probe(enable_look_for=False), _memo_probe()
check("MEMO_IS_INVALIDATED_BY_OBSERVE", look_n >= base_n * 0.5 and look_n > 1,
      "answer changes across observe() on %d of 200 ticks (ladder only: %d). "
      "Keyed on self.t this was 1." % (look_n, base_n))

# DEFECT 3, part one, and it needs its own scene rather than a rollout. Retiring
# a swept-out sighting used to happen inside _scan, so ASKING the human what it
# would do DELETED its memory -- but in a rollout that retirement is idempotent
# (the sighting is already gone by the time anything can look), so hammering
# decide() in a loop cannot see it. What sees it is a sighting that observe()
# deliberately KEEPS and the scan would throw away: look_radius=0 gives the sweep
# nothing to sweep, and the robot is still visibly holding the meat, so the fixed
# observe() refuses to call that a finished search. One decide() must not either.
h = _agent(look_radius=0)
s = _sight()
h.observe(s)
armed = dict(h.sightings)
h.decide(s)
check("ASKING_DOES_NOT_RETIRE_A_SIGHTING",
      "meat" in armed and h.sightings == armed,
      "%s -> %s across one decide()" % (armed, h.sightings))

# ...and the whole book stays byte-identical under a real episode too.
mdp = SteakHouseGridworld.from_layout_name("banquet_pass")
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
env.reset()
h = LimitedVisionHuman(mdp, 360, agent_index=1, seed=0)
bot = BASELINES["handoff"](mdp, agent_index=0, seed=0,
                           beta=float("inf"), rho=1.0)
touched, asked = 0, 0
for _ in range(200):
    a, _ = bot.action(env.state)
    act, _ = h.action(env.state)
    if h.sightings:
        before = (dict(h.sightings), {k: set(v) for k, v in h._cleared.items()})
        for _ in range(3):
            h.decide(env.state)
            h.rank(env.state)
        asked += 1
        after = (dict(h.sightings), {k: set(v) for k, v in h._cleared.items()})
        touched += int(before != after)
    env.step((a, act))
check("ASKING_DOES_NOT_EDIT_THE_SIGHTINGS", touched == 0 and asked > 0,
      "%d of %d ticks with a live sighting were mutated by asking" % (touched,
                                                                      asked))


# THE CONTRACT ITSELF, which is what both defects were really violations of:
# WATCHING THE AGENT MUST NOT CHANGE IT. Two paired episodes, same seed, the
# second with one speculative decide() before every action -- exactly what a
# filter's shadow rollout or a likelihood term does -- and the traces have to
# be identical, action for action. The stale memo broke this on 180 of 200
# ticks. It is the strictest statement of the property and the cheapest to
# keep true, so it is
# worth having even though the commitment write alone happens to slip past it on
# these five layouts.
def _speculative(spectate):
    mdp = SteakHouseGridworld.from_layout_name("banquet_pass")
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    env.reset()
    h = LimitedVisionHuman(mdp, 90, agent_index=1, seed=0)
    bot = BASELINES["handoff"](mdp, agent_index=0, seed=0,
                               beta=float("inf"), rho=1.0)
    out = []
    for _ in range(200):
        a, _ = bot.action(env.state)
        if spectate:
            h.decide(env.state)            # "what would you do?", pre-observe
        act, info = h.action(env.state)
        out.append((act, info["subtask"]))
        env.step((a, act))
    return out


clean, watched = _speculative(False), _speculative(True)
n_diff = sum(int(x != y) for x, y in zip(clean, watched))
check("ASKING_DOES_NOT_STEER_THE_AGENT", clean == watched,
      "%d of %d ticks diverge when something merely WATCHES the agent" % (n_diff,
                                                                          200))


# =============================================================================
print("\nROLLOUT PROPERTIES -- one sweep, and the payoff measurement")
# Everything below is checked on the SAME episodes because each needs a real
# cone, a real robot and a few hundred ticks, and re-running the sweep once per
# invariant is the difference between a test suite you run and one you skip.
#
# A BOUT is a maximal run of consecutive ticks committed to looking for one item.
# It FOUND the thing if the tick after it ended the belief knew of a counter
# holding it -- that is the S2 withdrawal, and it is the event the whole feature
# exists to produce. It PAID OFF if the human then INTERACTed at a counter that
# really held the item, INSIDE THE SEARCHED AREA, within LOOK_HORIZON/2 ticks.
# The area restriction is what stops an unrelated take_plate on the far side of
# the kitchen being counted as a payoff; without it the broken version scored 27%
# on coincidence alone.
# =============================================================================
WINDOW = LOOK_HORIZON // 2
nulled_and_remembered = 0
look_ticks = 0
interacts_while_looking = []
diverted_with_a_dish = []
pick_back_ups = []
longest_run = 0
drop_legs = 0
wasted_drop_legs = 0
n_bouts = found = paid = 0


def bouts(rows):
    """[(item, first_tick, last_tick)] -- maximal runs of one committed item."""
    out, cur = [], None
    for i, r in enumerate(rows):
        item = r["look"][0] if r["look"] else None
        if cur is not None and cur[0] == item:
            cur[2] = i
            continue
        if cur is not None:
            out.append(tuple(cur))
        cur = [item, i, i] if item is not None else None
    if cur is not None:
        out.append(tuple(cur))
    return out


for lay in LAYOUTS:
    for fov in FOVS:
        for seed in range(2):
            h, acts, rows = rollout(lay, fov, seed, ticks=300)
            stashed = {}         # counter cell -> (item, tick, sought item)
            run, prev = 0, None
            for i, r in enumerate(rows):
                verb, cell, act, held = r["verb"], r["cell"], r["act"], r["held"]

                # MEMORY. views.py:200 has nulled the position half of the robot
                # belief while the hands half survives -- exactly the state that
                # destroys the datum this feature needs -- and our own sighting
                # still has a position.
                rob, sight = r["robot"], r["sight"]
                if rob is not None and rob[0] is None and rob[2] is not None:
                    if rob[2] in sight and sight[rob[2]][0] is not None:
                        nulled_and_remembered += 1

                if verb.startswith("look_for_"):
                    look_ticks += 1
                    if act == Action.INTERACT:
                        interacts_while_looking.append((lay, fov, seed, i))
                    if held == "dish":
                        diverted_with_a_dish.append((lay, fov, seed, i))
                    run = run + 1 if prev == verb else 1
                    longest_run = max(longest_run, run)
                    prev = verb
                else:
                    run, prev = 0, None

                # the drop leg: a stash emitted while a look target is committed
                if verb == "stash" and r["look"] is not None:
                    drop_legs += 1
                    if act == Action.INTERACT:
                        stashed[cell] = (held, i, r["look"][0])
                        # A DROP LEG THAT BUYS NOTHING is the waste the
                        # after-drop tier test exists to stop: the hand is free
                        # and the human never goes looking. 30 of 45 before it.
                        tail = rows[i + 1:i + 1 + WINDOW]
                        if not any(x["verb"].startswith("look_for_")
                                   for x in tail):
                            wasted_drop_legs += 1

                # NO_PICK_BACK_UP. The loop the drop leg has to be safe against
                # is "put it down to go and look, take it straight back, repeat"
                # -- so the violation is taking it back WHILE STILL COMMITTED to
                # the very look-for the drop was made for. Scoring any later
                # pickup at that cell as a violation is over-attribution: it
                # flagged a garnish reclaimed 39 ticks later because a sink had
                # finally finished, which is the ladder working.
                # THE TAKE SIDE MUST NOT TEST r["look"]. _look_target is only ever
                # set on a look_for_* tick or the stash that precedes one, so on a
                # take_* tick it is None BY CONSTRUCTION and the predicate becomes
                # unsatisfiable -- measured at 1340 take_* ticks with zero live
                # look targets across the 50-episode sweep, i.e. a guard that can
                # never fire. The over-attribution it was added to cure is already
                # cured on the STASH side, where `stashed` only records a drop that
                # a look target was actually committed for; with that alone the
                # predicate fires 0 times, and with unconditional recording it
                # fires 67. So the honest test is the item-and-cell match.
                if verb.startswith("take_") and act == Action.INTERACT:
                    was = stashed.get(cell)
                    if was and was[0] == verb[len("take_"):]:
                        pick_back_ups.append((lay, fov, seed, i, was[0]))

            for item, s, e in bouts(rows):
                n_bouts += 1
                rpos = rows[s]["sight"].get(item)
                rpos = rpos[0] if rpos else None
                tail = rows[e + 1:e + 1 + WINDOW]
                f = bool(tail) and tail[0]["known"][item]
                found += int(f)
                paid += int(f and any(
                    x["act"] == Action.INTERACT
                    and x["verb"] in ("take_" + item, "combine")
                    and x["cell_obj"] == item and rpos is not None
                    and abs(x["cell"][0] - rpos[0])
                    + abs(x["cell"][1] - rpos[1]) <= h.look_radius
                    for x in tail))

check("FIRES_AT_ALL", look_ticks > 0 and drop_legs > 0,
      "%d look-for ticks, %d drop-leg stashes" % (look_ticks, drop_legs))
check("MEMORY_SURVIVES_NULLING", nulled_and_remembered > 0,
      "%d ticks where the belief lost the position and we kept it"
      % nulled_and_remembered)
check("NEVER_INTERACT_WHILE_LOOKING", not interacts_while_looking,
      str(interacts_while_looking[:3]))
check("NEVER_DIVERT_A_DISH", not diverted_with_a_dish,
      str(diverted_with_a_dish[:3]))
check("NO_PICK_BACK_UP", not pick_back_ups, str(pick_back_ups[:3]))
# TERMINATION. The sweep can only shrink -- the cleared set ignores belief decay
# and is only ever added to by a look taken after the arming tick -- and the two
# clocks bound the rest: the hands belief expires at FORGET_HORIZON, so the human
# cannot wait at a counter longer than that, and the sighting itself expires
# LOOK_HORIZON after arming. The bound is generous on purpose: what it is really
# testing is that this number is not "300".
check("TERMINATION", longest_run < 100,
      "longest unbroken run on one look target: %d ticks" % longest_run)

# =============================================================================
# THE PAYOFF. Everything above says the feature is wired up correctly; this is
# the only thing that says it WORKS, and it is the test whose absence let a
# feature with a zero success rate ship green. Numbers as measured:
#
#                          bouts  found  paid   rate   drop legs  wasted
#   before (both defects)     37      8     7  18.9%          17      10
#   sighting fix only        125     29    26  20.8%          45      30
#   + after-drop tier test    96     32    28  29.2%           5       0
#
# The rate is the headline the user asked for, but on its own it is a bad guard:
# the broken version scored 18.9% because it barely fired at all, and a feature
# that fires four times an episode and pays once is not the same object as one
# that fires once and pays once. So the floors are on BOTH the rate and the
# count, and the count is what actually moved.
# =============================================================================
rate = 100.0 * paid / n_bouts if n_bouts else 0.0
check("BOUTS_PAY_OFF", n_bouts > 0 and paid >= 20 and rate >= 25.0,
      "%d bouts, %d found the item, %d paid off -- %.1f%% (was 18.9%% on 37 "
      "bouts with both defects present)" % (n_bouts, found, paid, rate))
check("NO_WASTED_DROP_LEGS", wasted_drop_legs == 0,
      "%d of %d drop-leg stashes were not followed by a look-for (was 10 of 17)"
      % (wasted_drop_legs, drop_legs))


# =============================================================================
print("\nDETERMINISM -- same seed identical, and the clone is faithful")
# =============================================================================
_, a1, _ = rollout("banquet_pass", 180, 0)
_, a2, _ = rollout("banquet_pass", 180, 0)
check("DETERMINISM", a1 == a2, "%d ticks" % len(a1))

# clone() is the hook my_fov_filter.py's shadow rollouts depend on (qmdp.py's
# clone helpers used to get this wrong -- see the module docstring). Faithful
# means the copy's NEXT action is the original's next action -- a fresh human
# would differ on the first step alone, because decide() is sticky within a
# tier and a fresh one has last_subtask=None.
mdp = SteakHouseGridworld.from_layout_name("butchery")
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
env.reset()
h = LimitedVisionHuman(mdp, 60, agent_index=1, seed=0)
bot = BASELINES["handoff"](mdp, agent_index=0, seed=0,
                           beta=float("inf"), rho=1.0)
for _ in range(120):
    a, _ = bot.action(env.state)
    b, _ = h.action(env.state)
    env.step((a, b))
c = h.clone()
same = all(h.action(env.state)[0] == c.action(env.state)[0] for _ in range(5))
check("CLONE_IS_FAITHFUL",
      same and c.sightings == h.sightings and c.view.contents == h.view.contents,
      "5 steps of a clone against the original")


print("\n%s -- %d failures %s\n" % ("FAILED" if FAILS else "ALL PASS",
                                    len(FAILS), FAILS if FAILS else ""))
sys.exit(1 if FAILS else 0)
