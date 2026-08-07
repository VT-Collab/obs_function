"""
═══════════════════════════════════════════════════════════════════════════
fov/robot/policy/ - the robot's assistance decision, LEARNED rather than
hand-coded, with a Bayesian-FOV module bolted on top.

    baseline/    FOV-BLIND policy (vocabulary, network, env wrapper, PPO)
    module/      the ONLY place FOV information is ever used
    end_to_end/  adapters + comparison scripts

Mirrors minigrid's robot/policy/neural/ package, with ONE structural difference
that changes everything downstream:

    MINIGRID   the robot has no body. Its action space is "wait" + 5 reveal
               types - it never moves, it only SPEAKS. Assistance means telling
               the human a fact.
    STEAKHOUSE the robot is a physical cook. It has no communication channel at
               all. Assistance means DOING things: taking over a subtask the
               human cannot discover, or acting at a station so the resulting
               state change happens where the human can see it.

So the action space here is SUBTASK SELECTION, executed physically by the motion
planner. There is no "tell" action and there must never be one - any help the
robot gives has to be inferable by the human through their own eyes.

THIS FILE - the shared vocabulary every other file agrees on:
    ACTIONS          what the policy may choose each step
    extract_features what the policy is allowed to see

─────────────────────────────────────────────────────────────────────────────
THE OBSERVATION
─────────────────────────────────────────────────────────────────────────────
Minigrid's robot sees the map as 16 "sheets of tracing paper" stacked over a
grid. OUR robot does NOT get a picture. It already has full vision of the kitchen
and its own motion planner reads the map for it, so it does not need the map
inside its observation at all.

Instead the observation is a little DASHBOARD: a single row of 17 numbers, like
17 gauges on a car's dash. Every gauge is between 0 and 1. Each one answers ONE
small question about the kitchen RIGHT NOW. The network reads all 17 gauges at
once and picks one of the 7 ACTIONS.

Here is the whole dashboard, gauge by gauge:

  index   gauge                          reads as (every value is 0..1)
  ──────  ─────────────────────────────  ──────────────────────────────────────
  [0]  ┐                                 "work" is possible          (always 1)
  [1]  │   "WHICH STATIONS LOOK LIKE     pot is empty   (wants meat)
  [2]  │    THEY NEED SOMEONE?"          pot is cooking
  [3]  │    status lights - roughly two  board is empty (wants onion)
  [4]  │    per station (empty? /        board is chopping
  [5]  │    mid-job?). HINTS, not a      sink is empty  (wants plate)
  [6]  ┘    hard gate. They do NOT       "stage_visible" is possible (always 1)
           line up 1:1 with the 7 actions - they are grouped by STATION.

  [7]      HOW DONE IS THE POT?          0 empty · 0.5 cooking · 1 steak ready
  [8]      HOW DONE IS THE BOARD?        0 empty · 0.5 chopping · 1 garnish ready
  [9]      HOW DONE IS THE SINK?         0 empty · 0.5 washing · 1 plate ready

  [10]     HOW LONG HAS THE POT sat like this?    0 just changed .. 1 stuck ages
  [11]     HOW LONG HAS THE BOARD sat like this?  (same idea)
  [12]     HOW LONG HAS THE SINK sat like this?   (same idea)

  [13]     ARE THE HUMAN'S HANDS FULL?            0 empty-handed · 1 holding
  [14]     HOW FAR IS THE HUMAN from the nearest station?  0 right on it .. 1 far
  [15]     HOW MUCH OF THE EPISODE IS GONE?       0 just started .. 1 almost over
  [16]     HOW MANY ORDERS ARE LEFT?              0 none .. 1 lots (4+)

WHY THESE AND NOT OTHERS - THE DELIBERATE OMISSION. Gauges [10]..[14] are the
robot's only INDIRECT clues about how the human is doing: a station stuck for
ages + a human standing far away + empty-handed roughly says "the human probably
can't see this one, maybe I should step in." But the robot NEVER receives the
human's actual field-of-view, their beliefs, or the FOV posterior/entropy. That
absence is the whole experiment: the baseline must be FOV-BLIND so the module's
FOV inference has something real to add. Handing over the beliefs would make the
module trivially correct and the comparison meaningless. (The ablation at the
very bottom, STEAK_MINIMAL_FEAT, deletes even the indirect clues to prove they
were substituting for FOV.)

ORDERING IS LOAD-BEARING. Action index i means the same thing in three places:
ACTIONS[i], the network's i-th output logit, and the module's i-th bias entry
(module/fov_module.py). Reorder ACTIONS and all three silently desync. (The
[0:7] status lights are the one block grouped by STATION rather than by action,
so do not read action-alignment into them - the real per-action alignment lives
in the 7 logits and the 7-long bias vector.)

WHAT THE ROBOT MAY SEE. Full vision of the world (it is not the limited-vision
agent), plus the human's POSITION, ORIENTATION and ACTIONS. It must NEVER read
the human's internal beliefs - those are exactly what the Bayes filter has to
infer from behaviour.
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import numpy as np

# The robot's entire action space: which subtask to pursue this step. All
# physical - each is executed by the motion planner. "work" means "carry on with
# whatever the greedy worker would do", i.e. no assistive deviation.
ACTIONS = [
    "work",             # default: pursue own best subtask, no assistance
    "take_meat",        # take over: fetch meat + load the pot
    "take_onion",       # take over: fetch onion + load the board
    "take_chop",        # take over: chop the board
    "take_plate",       # take over: fetch plate + load the sink
    "take_wash",        # take over: heat the hot plate
    "stage_visible",    # act at the station nearest the human's cone, so the
                        # resulting state change lands where they can see it
]
N_ACTIONS = len(ACTIONS)

# Station kinds the human's decisions branch on (see the human's decide()).
# The three physical work-spots of the steak pipeline: pot (cook meat), board
# (chop onion), sink (wash plate). Gauges [7:10] and [10:13] have one slot each
# in exactly this order.
STATIONS = ["pot", "board", "sink"]

# The observation is 17 numbers - the "dashboard" drawn in full at the top of
# this file. Quick legend:
#   [0:7]   status lights - which stations look like they need someone
#   [7:10]  how DONE each station is        (0 empty / .5 mid-job / 1 ready)
#   [10:13] how LONG each station has been stuck  (0 fresh .. 1 ages)
#   [13]    human's hands full?
#   [14]    human's distance to nearest station   (0 on it .. 1 far)
#   [15]    fraction of the episode elapsed
#   [16]    orders remaining                 (0 .. 1)
OBS_DIM = 17

# Turns a station's word-state into ONE number for gauges [7:10] - basically
# "how done is this spot": nothing there = 0, being worked on = halfway = 0.5,
# finished and ready to grab = 1. ("occupied" = something is there we can't
# classify cleanly, treated as mid-ish = 0.5; "unknown" = 0.)
_STATE_VAL = {"empty": 0.0, "cooking": 0.5, "chopping": 0.5, "washing": 0.5,
              "ready": 1.0, "occupied": 0.5, "unknown": 0.0}


def station_locs(mdp):
    # WHERE the stations physically sit on this layout. Ask the map (mdp) for the
    # grid cells of every pot, every board ('B'), every sink ('W'). A layout can
    # have more than one of each, so each value is a LIST of (x, y) cells.
    # Everything below loops over these cells.
    return {
        "pot": list(mdp.get_pot_locations()),
        "board": list(mdp.terrain_pos_dict.get('B', [])),
        "sink": list(mdp.terrain_pos_dict.get('W', [])),
    }


def _station_state(mdp, state, loc):
    # Look at ONE station cell and say, in a single word, what is going on there.
    # This is the robot's FULL-VISION truth - it really can see this, no guessing.
    obj = state.objects.get(loc)
    if obj is None:
        return "empty"                     # nothing on the cell at all
    name = getattr(obj, "name", "")
    try:
        # For each kind of thing, ask the mdp "is it finished yet?" -> ready,
        # otherwise it is still being worked on (cooking / chopping / washing).
        if name == "steak":
            return "ready" if mdp.steak_ready_at_location(state, loc) else "cooking"
        if name in ("garnish", "onion"):
            return "ready" if mdp.garnish_ready_at_location(state, loc) else "chopping"
        if name in ("washed_plate", "dish", "plate"):
            return "ready" if mdp.plate_washed_at_location(state, loc) else "washing"
    except Exception:
        # Seatbelt: if an mdp helper ever throws on a weird object, fall through
        # to "occupied" instead of crashing the whole feature extraction.
        pass
    return "occupied"                      # something is there we can't classify


def extract_features(mdp, state, human_index=1, t=0, horizon=260, age=None):
    """FOV-BLIND observation. Deliberately contains no posterior and no entropy.

    `age` is an optional {station_kind: steps_in_current_state} dict maintained
    by the env wrapper - it is a fact about the WORLD (how long the pot has been
    cooking), never about what the human knows.
    """
    locs = station_locs(mdp)
    obs = np.zeros(OBS_DIM, dtype=np.float32)   # every gauge starts at 0; switch on the ones that apply

    # ── [0:7] STATUS LIGHTS: "which stations look like they need someone?" ──
    # Hints for the network, NOT a hard gate (policy.py never uses them to forbid
    # an action). They are grouped by STATION - two lights each - which is why
    # they don't line up 1:1 with the 7 action slots. See the header.
    obs[0] = 1.0    # "work" is always on the table
    for i, kind in enumerate(("pot", "board", "sink")):
        # Collapse a station's several cells to one representative word. `min`
        # just picks the alphabetically-first word - an arbitrary but stable pick.
        st = min((_station_state(mdp, state, l) for l in locs[kind]), default="empty")
        # light A: the station is empty and wants loading
        obs[1 + i * 2] = 1.0 if st == "empty" else 0.0
        # light B: the station is mid-job (someone/something is working it)
        obs[2 + i * 2] = 1.0 if st in ("chopping", "washing", "cooking") else 0.0
    # "stage_visible" is always on the table. (This lands on obs[6], overwriting
    # the sink's light-B that the loop just set - harmless, these are only hints.)
    obs[6] = 1.0

    # ── [7:10] HOW DONE IS EACH STATION (full-vision truth) ─────────────────
    # One number per station in STATIONS order: 0 empty, 0.5 mid-job, 1 ready.
    # If a station has several cells we keep the MOST-done one (max), because
    # "a steak is ready SOMEWHERE" is what matters to the robot.
    for i, kind in enumerate(STATIONS):
        vals = [_STATE_VAL.get(_station_state(mdp, state, l), 0.0) for l in locs[kind]]
        obs[7 + i] = max(vals) if vals else 0.0

    # ── [10:13] HOW LONG each station has been stuck in its current state ───
    # `age[kind]` counts env steps since that station last changed (kept by the
    # env wrapper). Squash it: 40+ steps stuck -> 1.0 (capped), fewer -> less.
    # A high number = "this has sat a long while, maybe the human can't see it".
    # One of the INDIRECT human clues the header warns about.
    if age:
        for i, kind in enumerate(STATIONS):
            obs[10 + i] = min(1.0, age.get(kind, 0) / 40.0)

    # ── [13:17] HUMAN + CLOCK context (position/holding only, NEVER beliefs) ─
    hp = state.players[human_index]
    # [13] are the human's hands full? (holding an object vs empty-handed)
    obs[13] = 1.0 if hp.held_object is not None else 0.0
    allst = [l for v in locs.values() for l in v]     # every station cell, flattened into one list
    if allst:
        # [14] Manhattan distance (|dx| + |dy|) to the CLOSEST station, squashed:
        # 0 = human is standing right on a station, ~1 = far away. Far + empty-
        # handed hints the human isn't currently helping at any station.
        d = min(abs(l[0] - hp.position[0]) + abs(l[1] - hp.position[1]) for l in allst)
        obs[14] = min(1.0, d / 20.0)
    # [15] the clock: how far through the episode we are, 0 at the start -> 1 near the end.
    obs[15] = min(1.0, t / max(1, horizon))
    # [16] how many orders are still on the ticket rail, capped/normalised at 4.
    obs[16] = min(1.0, len(state.order_list or []) / 4.0)
    if os.environ.get("STEAK_MINIMAL_FEAT") == "1":
        # ABLATION: zero the human-contribution PROXIES (station-idle age [10:13],
        # human holding [13], human distance [14]) so the FOV-blind baseline can
        # no longer infer "the human isn't contributing" indirectly. If the FOV
        # module then wins big again, those proxies were substituting for FOV.
        obs[10:15] = 0.0
    return obs
