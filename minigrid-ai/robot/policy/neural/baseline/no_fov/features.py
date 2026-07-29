# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/ - the RAW-STATE baseline. No field-of-view reasoning of any
# kind: no shadow agent, no assumed FOV, no CandidateFinder, no belief about
# what the human has or has not seen. The network sees the world and the human's
# body, and maps that straight to a speech action.
#
# Contrast with its sibling:
#   static_fov/  13 hand-crafted dims, computed by running CandidateFinder
#                against a shadow human fixed at ASSUMED_FOV=120. Encodes a
#                BELIEF about what the human is missing.
#   no_fov/      this. Raw grid + human body. Encodes only what is physically
#                true. Any notion of "does the human already know this" has to
#                be learned from the reward, not handed over in the features.
#
# THIS FILE - the observation, and the two things every method here shares.
#
# ── THE OBSERVATION: 16 sheets, each a 19x19 grid of 0s and 1s ─────────────
#
# Picture 16 sheets of tracing paper stacked on top of the map. Every sheet is
# the same 19x19 shape. Every sheet answers ONE yes/no question about every
# cell. 1 = yes, 0 = no.
#
#     the "wall" sheet  ->  1 in each cell that is a wall,   0 everywhere else
#     the "key"  sheet  ->  1 in each cell that has a key,   0 everywhere else
#     ... and so on, 16 times.
#
# That is the entire idea. The list below is just WHICH question each sheet asks.
#
# Why a stack instead of one map: pick any cell, look straight down through all
# 16 sheets, and you get 16 numbers describing that one cell. A conv net slides
# a small window across the map and sees all 16 answers at every position, so
# "a key sitting next to a door" is a local pattern one filter can pick up.
#
# Why one sheet per kind rather than one sheet of type NUMBERS: a conv filter
# multiplies and adds its inputs, so a single channel holding 2 for wall and
# 5 for key would assert that key is "bigger than" door(4) and "next to"
# ball(6). Those orderings are fiction - the numbers are name tags. Separate
# sheets make every kind equidistant. (The cheaper alternative is one integer
# channel plus an nn.Embedding in the network, which is what torch_ac's stock
# MiniGrid model does. That moves the work into actor_critic.py; not done here.)
#
# ── the 16 sheets, by their literal index ──────────────────────────────────
#
#  WHAT KIND OF THING IS IN THIS CELL          sheets 0-4
#  Exactly ONE of these five is 1 for any given cell; the other four are 0.
#      0  empty        2  door        4  goal
#      1  wall         3  key
#  MiniGrid's vocabulary has 11 kinds, but LockedRoom only ever contains these
#  five - see OBJ_TYPES below for the measurement and why the other six are out.
#
#  DOOR DETAIL                                 sheets 5-6
#  "Is a door" is not enough - a door is also open, closed, or locked.
#      5  this cell is a door AND it is locked
#      6  this cell is a door AND it is open
#         (a closed-but-unlocked door is 0 on BOTH of these)
#
#  WHERE THE HUMAN IS                          sheets 7-9
#      7  the single cell the human stands on     (exactly one 1 on this sheet)
#      8  the single cell the human is facing     (exactly one 1)
#      9  every cell holding an object the same COLOUR as what the human
#         carries - i.e. "the door your key opens is one of these".
#         All zeros when they carry nothing.
#
#  THE SAME NUMBER REPEATED IN ALL 361 CELLS    sheets 10-15
#  These are not maps. Each is a single fact smeared across the whole sheet,
#  because a conv net only accepts inputs shaped like a map. Wasteful, standard.
#     10  the clock: step_count / max_steps. 0.0 at the start, 1.0 at the end.
#     11  1 if the robot already said "key"        this episode, else 0
#     12  1 if the robot already said "door"       this episode, else 0
#     13  1 if the robot already said "goal"       this episode, else 0
#     14  1 if the robot already said "dead_room"  this episode, else 0
#     15  1 if the robot already said "empty_room" this episode, else 0
#  Sheets 11-15 are the robot remembering its OWN words. That is self-knowledge,
#  not a belief about what the human has seen, so it is not an FOV leak.
#
# 5 kinds + 2 door + 3 human + 1 clock + 5 told = 16. That is N_CHANNELS.
#
# The clock is deliberately included, and it is the one real asymmetry with
# static_fov (whose 13 dims have no time signal at all). Justification: the
# episode truncates at max_steps paying zero reward, so remaining time is part
# of the true state of the MDP. Leaving it out is what makes the problem
# non-Markov. It is also NOT an FOV leak in this file's sense - it says nothing
# about perception - though note a narrow-FOV human survives longer on average,
# so the clock does correlate with FOV. That correlation is the price of a
# Markov state here, and it is why this is a SEPARATE baseline rather than a
# drop-in replacement for static_fov.
#
# ── The action space ───────────────────────────────────────────────────────
# Same 6 as static_fov, so the two are directly comparable and both plug into
# the same evaluation harness: wait, or name one of 5 things.
#
# ── NO ACTION MASKING. This is the defining choice of this baseline. ───────
# static_fov hard-masks illegal reveals to -1e9 before sampling, so the network
# is structurally unable to propose naming a thing that isn't there. That mask is
# hand-written domain logic - "a key reveal is valid iff a live key exists and
# the human isn't already carrying one" - and it does a large share of the work.
#
# Here there is none. All 6 actions are always available. Speaking always costs
# comm_cost, whether or not it was useful, so the policy has to discover from
# reward alone that naming a thing twice, or naming something that doesn't
# exist, is simply wasted budget. That is what "maps state to action" means:
# no deterministic rule stands between the network's output and the world.
#
# Two consequences worth knowing before you compare the numbers:
#   - this is a strictly HARDER problem than static_fov's, so expect it to start
#     worse; the comparison measures how much the hand-written mask was worth.
#   - it removes the pathology masking creates for entropy-regularised methods.
#     With a mask, 94% of states allow exactly one action, so achievable entropy
#     is log(1)=0 and SAC's auto-tuned temperature chases an impossible target
#     (measured: alpha 1.34 -> 4.48 while return fell 0.46 -> 0.22). With every
#     action always available, log(6) is achievable everywhere and the standard
#     fixed entropy target is correct again.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

# Same vocabulary and ORDER as static_fov/features.py. Index alignment is
# load-bearing: policy.py-style action masking slices the first N_ACTIONS
# entries, and the eval adapter maps action index -> reveal type by this list.
ACTIONS = ["wait", "key", "door", "goal", "dead_room", "empty_room"]
N_ACTIONS = len(ACTIONS)
REVEAL_KEYS = ACTIONS[1:] #all except wait

GRID = 19                       # LockedRoom is 19x19 (max_steps 190 = 10*size)

# The only object kinds LockedRoom ever contains. MiniGrid's OBJECT_TO_IDX has
# 11, but measured over 60 seeds plus 6 full random-policy episodes, six of them
# never occur even once, so their sheets were 361 zeros each, every step:
#
#   unseen  only produced when you pass a vis_mask to grid.encode(). We call it
#           with no mask, so the mask defaults to all-ones and nothing is ever
#           blanked. Not a field-of-view signal, and not one we could use even
#           if we wanted to - this observation is the full omniscient grid.
#   agent   MiniGrid never writes the agent into the grid at all. That is
#           precisely why sheet 7 has to be drawn by hand further down.
#   floor / ball / box / lava   simply do not appear in this environment.
#
# Dropping them takes the observation from 22 sheets to 16 with zero loss. The
# ORDER here defines sheets 0-4 and is load-bearing; append, never reorder.
OBJ_TYPES = ["empty", "wall", "door", "key", "goal"]
OBJ_IDX = [OBJECT_TO_IDX[t] for t in OBJ_TYPES]   # -> [1, 2, 4, 5, 8] in enc
OBJ_IDX_SET = set(OBJ_IDX)                        # for the guard in encode_state

# How the 16 sheets are counted. These names are why the code below writes
# x[N_OBJ + 4] instead of x[9] - same sheet, spelled relative to its block.
N_OBJ = len(OBJ_TYPES)          # 5 -> sheets 0..4, one per kind that can occur
N_EXTRA = 3 + 3 + len(REVEAL_KEYS)
#         |   |   '-- 5 -> sheets 11..15  told key/door/goal/dead_room/empty_room
#         |   '------ 3 -> sheets 8, 9, 10  facing cell, carry-colour, clock
#         '---------- 3 -> sheets 5, 6, 7   door locked, door open, human's cell

N_CHANNELS = N_OBJ + N_EXTRA    # 5 + 11 = 16 sheets in total

# The final shape handed to the network: 16 sheets, each 19x19.
OBS_SHAPE = (N_CHANNELS, GRID, GRID)


def encode_state(state, told: dict, max_steps: int = 190) -> np.ndarray:
    """Raw world state -> (C, 19, 19) float32. No shadow, no belief, no FOV."""
    
    # ── STEP 1: get the raw map out of MiniGrid ─────────────────────────────
    # The grid stores Python OBJECTS - a Wall instance, a Door instance, None
    # for an empty cell. A network cannot eat objects, so encode() rewrites
    # every cell as THREE numbers:
    #
    #        cell (7,9) = (4, 1, 2)
    #                      |  |  '-- 2 = locked   (0 open, 1 closed, 2 locked)
    #                      |  '----- 1 = green
    #                      '-------- 4 = door
    #
    # So `enc` is ONE 19x19 map that is three numbers deep. It is NOT the 16
    # sheets yet - those get built below out of this.
    #
    # The third number only means something for doors; everything else stores 0.
    # Empty cells store colour 0, which decodes as "red" - that is filler, not a
    # real colour, and it is why the carry sheet below has to exclude them.
    #
    # INDEX ORDER IS enc[x][y] - x first. grid.py:252 fills array[i, j] from
    # grid.get(i, j) with i running over the WIDTH. Getting this backwards is
    # invisible on a square grid and silently mirrors the whole observation.
    enc = state.grid.encode()

    # ── STEP 2: split those three numbers into three separate 19x19 maps ────
    #   obj = the "what kind of thing" number of every cell   (2=wall, 5=key, ..)
    #   _   = the colour number of every cell   (not needed under this name; the
    #                                            carry sheet re-reads enc[...,1])
    #   st  = the open/closed/locked number of every cell  (doors only)
    #
    # `...` means "every axis before this one", so enc[..., 0] is exactly the
    # same thing as enc[:, :, 0].
    #
    # CAREFUL: `st` is DOOR STATE. It is not `state`, which is the whole
    # environment and is this function's first argument. Two different words.
    obj, _, st = enc[..., 0], enc[..., 1], enc[..., 2]

    # GUARD: this file commits to the five kinds in OBJ_TYPES. If a different
    # environment (or an edited LockedRoom) ever puts a ball, box or lava on the
    # grid, that object would silently vanish from the observation - it would be
    # 0 on all five kind sheets and the network would see empty floor. Fail loud
    # instead. Cheap: np.unique over 361 uint8s.
    present = set(np.unique(obj).tolist())
    assert present <= OBJ_IDX_SET, (
        f"object kind(s) {sorted(present - OBJ_IDX_SET)} are on the grid but not in "
        f"OBJ_TYPES={OBJ_TYPES}; add them there (append, do not reorder) or the "
        f"network never sees them")

    # ── STEP 3: 16 blank sheets, every cell 0. Everything below switches 1s on.
    x = np.zeros(OBS_SHAPE, dtype=np.float32)

    # ── SHEETS 0-4: one sheet per kind of thing that can occur ──────────────
    # (obj == idx) asks all 361 cells at once "is your number idx?" and hands
    # back a 19x19 grid of True/False, which numpy stores as 1.0/0.0. So on the
    # pass where idx is 5 this writes the key sheet: 1 wherever a key sits, 0
    # everywhere else. Five passes, five sheets.
    #
    # `t` is the SHEET number (0..4) and `idx` is MiniGrid's raw number for that
    # kind (1,2,4,5,8). They are deliberately different: we renumbered to close
    # the gaps left by the six kinds LockedRoom never uses.
    # Exactly one of these five sheets is 1 for any given cell.
    for t, idx in enumerate(OBJ_IDX):
        x[t] = (obj == idx)

    # ── SHEETS 5-6: locked / open doors ─────────────────────────────────────
    # No single number in `enc` means "locked door", so ask two questions and
    # AND them cell by cell: is this a door, AND is its state number 2?
    # Note this compares against MiniGrid's RAW door number, not sheet 2.
    door = OBJECT_TO_IDX["door"]
    x[N_OBJ + 0] = (obj == door) & (st == 2)     # sheet 5: door AND locked
    x[N_OBJ + 1] = (obj == door) & (st == 0)     # sheet 6: door AND open

    # ── SHEET 7: where the human is standing ────────────────────────────────
    # One single 1 on this whole sheet; the other 360 cells stay 0. MiniGrid
    # never puts the agent in the grid, so this is the ONLY place the human's
    # position enters the observation.
    # Written [ax, ay] - x first - to line up with `enc` from step 1. If this
    # were [ay, ax] the human would appear at the mirrored cell, sometimes
    # inside a wall, and nothing would ever raise an error.
    ax, ay = state.agent_pos
    x[N_OBJ + 2, ax, ay] = 1.0

    # ── SHEET 8: the cell the human is looking at ───────────────────────────
    # dir_vec is a one-step arrow in the direction they face, e.g. (1,0) for
    # right, so position + dir_vec is the next cell over. Also a single 1.
    # The bounds check covers a human standing against the outer wall.
    dx, dy = state.dir_vec
    fx, fy = ax + int(dx), ay + int(dy)
    if 0 <= fx < GRID and 0 <= fy < GRID:
        x[N_OBJ + 3, fx, fy] = 1.0

    # ── SHEET 9: things matching the colour the human carries ───────────────
    # Lets the network see "your green key goes with that green door" without
    # being told which door that is. Stays all-zero when they carry nothing.
    # enc[..., 1] is the colour map from step 2. The `!= empty` half is
    # essential: empty cells store colour 0 = red, so a human carrying anything
    # red would otherwise light up every empty cell on the map.
    carrying = getattr(state, "carrying", None)
    if carrying is not None:
        from minigrid.core.constants import COLOR_TO_IDX
        c = COLOR_TO_IDX.get(getattr(carrying, "color", None))
        if c is not None:
            x[N_OBJ + 4] = (enc[..., 1] == c) & (obj != OBJECT_TO_IDX["empty"])

    # ── SHEET 10: the clock ─────────────────────────────────────────────────
    # ONE number - how far through the episode we are - written into all 361
    # cells. 0.0 on the first step, 1.0 at truncation. Not a map; a fact in the
    # shape of a map, because that is the only shape a conv net takes.
    x[N_OBJ + 5] = min(state.step_count, max_steps) / max_steps

    # ── SHEETS 11-15: what the robot has already said ───────────────────────
    # One sheet per phrase, in REVEAL_KEYS order: key, door, goal, dead_room,
    # empty_room. Filled entirely with 1.0 once that phrase has been used this
    # episode, otherwise left blank. Also facts-shaped-like-maps, not maps.
    for i, k in enumerate(REVEAL_KEYS):
        if told.get(k):
            x[N_OBJ + 6 + i] = 1.0
    return x


# Deliberately no legal_reveal_mask() here. See the header: any such function is
# hand-written domain logic, and this baseline exists to do without it.
