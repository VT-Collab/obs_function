"""THE LAYOUTS.  Three topologies, built so a limited FOV actually costs you
something -- which is the one property every layout used so far lacks.

===========================================================================
WHY THE OLD LAYOUTS COULD NEVER WORK
===========================================================================
steak_gc00 is a 3x3 room:

    XPBWX          9 walkable cells, ZERO interior non-floor tiles
    X   M          max Manhattan distance: 4 steps
    X21 O          every station on the perimeter of that one room
    X   D
    XXXSX

Two constants in the human model decide whether a field of view can matter:

    FORGET_HORIZON = 12    a belief goes stale after 12 ticks
    SIGHT_RADIUS   = 8     tiles beyond this are never resolved at all

On gc00 ordinary movement re-observes every station within ~4 ticks, so the
forget horizon can never fire, and every tile is inside the sight radius from
every cell. The human's knowledge base K_t therefore equals the true state s at
essentially all times, for any cone >= 90 degrees. Measured, and it is exactly
what the data says: completion 139.4 at fov90 vs 138.6 at fov360, and
`h_wasted = 0` on every single episode.

`occlude=True` is the right lever and it does real work -- `_clear_los` raycasts
against true terrain -- but gc00 has no interior non-floor tiles, so there is
nothing to raycast against. Turning it on there is a no-op.

So the requirement is not "a bigger kitchen". It is:

    1. INTERIOR OBSTRUCTION   some pair of regions cannot see each other from
                              ANY standing position, at ANY cone width. This is
                              what makes ignorance structural rather than a
                              two-tick attention lag.
    2. LEGS LONGER THAN 12    the walk between station clusters must exceed
                              FORGET_HORIZON, or beliefs never actually decay.
    3. SPLIT STATIONS         no single vantage point sees every station, so
                              there is always something the human is wrong about.

===========================================================================
WHERE THE TIMED STATIONS GO, AND WHY IT IS THE WHOLE DESIGN
===========================================================================
Three stations carry a hidden clock: the grill (P, cook_time 15), the board
(B, chop_time 5) and the sink (W, wash_time 5). Those are the only facts in this
kitchen that CHANGE WITHOUT ANYBODY TOUCHING THEM -- and therefore the only
facts a human can be wrong about through no fault of their own.

All three go on the FAR side of the obstruction from the serve/dispenser work.
That is what manufactures the paper's headline case: the steak finishes cooking,
the robot can see it, and the human provably cannot -- not because they were
careless, but because a wall is in the way and their belief aged out. That is an
unseen-assistance opportunity that exists in the world rather than in a weight.

===========================================================================
THE THREE TOPOLOGIES
===========================================================================
    island      solid block, corridor all the way round. Opposite sides are
                mutually invisible, but both agents share one loop, so
                contention survives AND there are two routes to everything --
                which is what makes a legible detour expressible at all.
    peninsula   a wall down from the top, two lobes joined at the bottom only.
                Strongest blind spot of the three. Risk: the lobes decouple into
                independent workspaces and the agents stop contending.
    tworoom     two chambers joined by one doorway. Maximum ignorance, minimum
                contention -- included as the extreme end of the axis.

Testing all three IS a result: it maps assistance value against layout topology
instead of asserting one.

===========================================================================
TERRAIN AND THE DIGIT CONVENTION
===========================================================================
    P grill   B board   W sink   M meat   O onion   D plate   S serve
    X counter/wall   (space) floor

'1' is player index 0 = THE ROBOT.  '2' is player index 1 = THE HUMAN.
Confirmed against steak_gc00 ("X21 O": '2' at x=1, '1' at x=2) and against
ROBOT_INDEX/HUMAN_INDEX in baseline.py. Getting this backwards silently swaps
who has the cone, and every number afterwards is meaningless.

ASYMMETRIC START: the two digits are placed on OPPOSITE sides of the
obstruction, so an epistemic gap exists at t=0 rather than having to develop.
The human starts where they cannot see the timed stations.

===========================================================================
ROBOT STATION RESTRICTION LIVES ELSEWHERE
===========================================================================
Forcing division of labour is a rule about who may use a station, not a fact
about the grid, so it is enforced in the environment wrapper and not by carving
the map. Keeping it out of here means the same three grids serve both the
restricted and unrestricted conditions, and the comparison stays paired.
"""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
STEAKHOUSE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

#Our own layout folder. Deliberately NOT fov/layouts_final/layouts: that library
#is validated and frozen, and dropping experimental grids into it would put
#unvalidated maps in the same namespace as the 31 that carry the human-model
#results.
LAYOUT_DIR = os.path.join(HERE, "layouts")

#Where overcooked_ai_py actually looks. read_layout_dict only ever reads this
#one folder, so a .layout anywhere else does not exist as far as
#from_layout_name is concerned.
INSTALL_DIR = os.path.join(STEAKHOUSE, "overcooked_ai_py", "data", "layouts")


# =========================================================================
# THE GRIDS
# =========================================================================
# Every row of a grid must be the SAME WIDTH. A ragged grid does not raise --
# the mdp builds a lopsided terrain matrix and the failure surfaces much later
# as an inexplicable pathing bug. probe.py asserts rectangularity for that
# reason; do not remove the check.

#---- ISLAND -------------------------------------------------------------
#Solid 7x4 block, corridor 2-3 wide all the way round. The timed stations
#(P/B/W) sit on the top wall; dispensers on the flanks; plate and serve at the
#bottom. A human working the bottom band has the island between them and all
#three clocks, from every cell down there, at every cone width.
ISLAND = """\
XXXXXPBWXXXXXX
X            X
X   1        X
X   XXXXXXX  X
M   XXXXXXX  O
X   XXXXXXX  X
X   XXXXXXX  X
X            X
X      2     X
D            S
XXXXXXXXXXXXXX"""

#---- PENINSULA ----------------------------------------------------------
#A wall hanging from the top wall, splitting the room into two lobes that meet
#only along the bottom two rows. Crossing between lobes is a long walk, so
#anything learned in one lobe is stale by the time you are in the other.
#A station in the top wall must have FLOOR directly below it. Putting one at the
#same x as the hanging wall makes its only approach cell a wall tile, and the
#station is unreachable -- the mdp builds happily and the human simply never
#completes an order. probe.py's `unreachable` check exists because of this.
PENINSULA = """\
XXXPBWXXXXXXXX
X     X      X
X     X      X
M  1  X      O
X     X      X
X     X      X
X     X      X
X     X   2  X
X     X      X
D            S
XXXXXXXXXXXXXX"""

#---- TWO ROOMS ----------------------------------------------------------
#Two chambers joined by a single doorway. The extreme end of the axis: the
#human in the south room knows nothing whatsoever about the north room until
#they walk through the door.
#One doorway, at (6,5). Every other tile of the x=6 column is wall, so the two
#chambers share exactly one crossing. The first draft sealed the right chamber
#entirely -- probe.py's `components` check caught it, which is what that check
#is for: a disconnected grid does not raise, the human just never finds half the
#kitchen and you read it as a hard layout.
TWOROOM = """\
XXXPBWXXXXXXXX
X     X      X
X     X      X
M  1  X      O
X     X      X
X            X
X     X      X
X     X   2  X
D     X      S
X     X      X
XXXXXXXXXXXXXX"""

GRIDS = {
    "setup_island": ISLAND,
    "setup_peninsula": PENINSULA,
    "setup_tworoom": TWOROOM,
}

#The steak task parameters. Held identical across all three grids so the only
#thing that varies between layouts is the GEOMETRY -- otherwise a difference in
#assistance value could be a difference in cook time wearing a disguise.
#These match fov/layouts_final/layouts/*.layout exactly.
TASK_PARAMS = dict(
    cook_time=15,
    delivery_reward=20,
    num_items_for_steak=1,
    chop_time=5,
    wash_time=5,
)


def layout_text(grid, n_orders=4):
    """One .layout file's contents.

    start_order_list is written as a python LIST here, not the bare string the
    layouts_final files use. That string form is TRAP 1 in baseline.py: len() on
    a string counts characters, so the order list silently has ~24 entries and no
    delivery ever fires. Kitchen overrides it at construction anyway, but a file
    that is wrong on disk will eventually be loaded by something that does not.
    """
    #each entry already carries its own trailing comma, so join on a bare
    #newline. Joining on ",\n" produces "15,," and eval() dies on it.
    body = "\n".join("    %r: %r," % (k, v) for k, v in TASK_PARAMS.items())
    return ('{\n    "grid":\n"""%s""",\n    "start_order_list": %r,\n%s\n'
            '    "rew_shaping_params": None\n}\n'
            % (grid, ["steak"] * n_orders, body))


def write_layouts(n_orders=4):
    """Write every grid to LAYOUT_DIR. Returns the paths written."""
    os.makedirs(LAYOUT_DIR, exist_ok=True)
    out = []
    for name, grid in GRIDS.items():
        path = os.path.join(LAYOUT_DIR, name + ".layout")
        with open(path, "w") as f:
            f.write(layout_text(grid, n_orders))
        out.append(path)
    return out


def stage(force=True):
    """Copy our layouts into overcooked_ai_py/data/layouts so from_layout_name
    can find them.

    force=True by default, which is the OPPOSITE of baseline.stage_layouts.
    That function refuses to overwrite, correctly, because it ships a frozen
    validated library. Ours are under active iteration: edit a grid here, and a
    no-overwrite copy would leave the OLD grid installed and you would spend an
    afternoon debugging a layout you already fixed.
    """
    write_layouts()
    os.makedirs(INSTALL_DIR, exist_ok=True)
    copied = []
    for name in GRIDS:
        src = os.path.join(LAYOUT_DIR, name + ".layout")
        dst = os.path.join(INSTALL_DIR, name + ".layout")
        if force or not os.path.exists(dst):
            shutil.copyfile(src, dst)
            copied.append(name)
    return copied


def parse(grid):
    """-> list of rows, each a list of single characters. Asserts rectangular."""
    rows = [list(r) for r in grid.split("\n")]
    widths = {len(r) for r in rows}
    assert len(widths) == 1, "ragged grid, row widths = %s" % sorted(widths)
    return rows


if __name__ == "__main__":
    for p in write_layouts():
        print("wrote", p)
    print("staged:", stage())
