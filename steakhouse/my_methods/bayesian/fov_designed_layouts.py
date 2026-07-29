"""
MISHA NEW CHANGE - DESIGNED (not random) layouts that guarantee FOV-dependent
subtask choice by construction.

WHY RANDOM SEARCH FAILED
------------------------
768 random wall-embedded layouts, scored with RNG held fixed and timing
artifacts removed:

    0/681 passed
    39/681 showed ANY FOV effect at all
    mean real subtask divergence = 0.1

642 of 681 had literally zero. That is not a threshold problem - relaxing gates
cannot rescue a mean of 0.1. Random placement essentially never creates the
precise condition FOV needs, so hoping for it is the wrong strategy.

THE CONDITION FOV ACTUALLY NEEDS
--------------------------------
The human's subtask choice branches on kb_to_state_info (agent.py:1088):
    (num_item_in_pot, chop_time, wash_time, robot_held_object)
So a FOV difference can only change behaviour if a hypothesis with wide vision
LEARNS one of those facts while a hypothesis with narrow vision does NOT.

Two things make that rare by accident:
  * in_bound() (agent.py:873-911) treats any tile immediately beside the player
    as visible regardless of FOV, and the human stands next to whatever station
    it is acting on. So FOV only ever matters for facts learned AT A DISTANCE.
  * The cone test is  y <= -cos(fov/2) * |x|  in the player's rotated frame.
    Whether a station falls between two candidate cones is pure geometry.

So instead of scattering stations and hoping, this module SOLVES for the
geometry: it computes the set of cells that are inside the WIDE candidate's cone
but outside the NARROW candidate's cone, from the pose the human actually
occupies while working, and puts the pot there. The teammate (full vision) then
changes the pot state; the wide-FOV hypothesis sees it and switches subtask, the
narrow one does not.

Run with: python -m my_methods.bayesian.fov_designed_layouts [n] [out_dir]
"""
import math
import os
import sys

DIRECTIONS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def in_cone(player_pos, orientation, loc, fov_deg):
    """Standalone replica of SteakLimitVisionHumanModel.in_bound (agent.py:873-911).

    Kept deliberately faithful, including the adjacency exemption - that
    exemption is precisely why FOV so rarely matters, so a design-time model
    that omitted it would place stations that look discriminating on paper and
    are not in the simulator.
    """
    px, py = player_pos
    lx, ly = loc
    half = fov_deg / 2.0
    if half == 0:
        return True

    # Adjacency exemption: the two tiles perpendicular to facing are always seen.
    if orientation == (0, -1):      # north
        if ly == py and (lx == px - 1 or lx == px + 1):
            return True
        rot = math.radians(180)
    elif orientation == (1, 0):     # east
        if lx == px and (ly == py - 1 or ly == py + 1):
            return True
        rot = math.radians(270)
    elif orientation == (0, 1):     # south
        if ly == py and (lx == px - 1 or lx == px + 1):
            return True
        rot = math.radians(0)
    elif orientation == (-1, 0):    # west
        if lx == px and (ly == py - 1 or ly == py + 1):
            return True
        rot = math.radians(90)
    else:
        return False

    c, s = math.cos(rot), math.sin(rot)
    sx, sy = lx - px, ly - py
    fx, fy = sx, -sy                      # y flip
    rx = c * fx - s * fy
    ry = s * fx + c * fy
    return -abs(rx * math.cos(math.radians(half))) >= ry


def differential_cells(player_pos, orientation, fov_narrow, fov_wide, width, height):
    """Cells visible to fov_wide but NOT to fov_narrow from this pose.

    This is the whole point: a station placed here is a fact the wide-FOV
    hypothesis can learn at a distance and the narrow one cannot.
    """
    out = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if (x, y) == player_pos:
                continue
            if in_cone(player_pos, orientation, (x, y), fov_wide) and \
               not in_cone(player_pos, orientation, (x, y), fov_narrow):
                out.append((x, y))
    return out


def describe(fov_triple, width=15, height=9):
    """Report, for each anchor pose, how many cells discriminate each FOV pair.

    Used to sanity-check a candidate FOV triple before building anything: if the
    differential region is empty on a grid this size, no placement of stations
    can make those two FOVs disagree, and the triple should be rejected outright
    rather than burning a planner build on it.
    """
    fn, fm, fw = fov_triple
    rows = []
    for name, ori in DIRECTIONS.items():
        for pos in [(width // 2, height // 2), (3, height // 2), (width - 4, height // 2)]:
            nw = len(differential_cells(pos, ori, fn, fw, width, height))
            nm = len(differential_cells(pos, ori, fn, fm, width, height))
            mw = len(differential_cells(pos, ori, fm, fw, width, height))
            rows.append((pos, name, nm, mw, nw))
    return rows


STATIONS = ['P', 'D', 'M', 'O', 'B', 'W', 'S']


def build_designed(idx, fov_triple, width=15, height=9, anchor=None, ori=(0, -1)):
    """Construct a layout where the POT is deliberately placed in the region
    visible to the wide FOV but not the narrow one, from the pose the human
    occupies while working at the onion/board side of the kitchen.

    Returns None if this FOV triple has no usable differential region (some
    triples genuinely cannot be separated on a grid this size - see describe()).

    Rationale: the human's subtask branches on num_item_in_pot among other
    things (agent.py:1088). If the pot sits where only the wide hypothesis can
    see it, then when the full-vision teammate loads the pot, the wide
    hypothesis learns "there is meat cooking" and moves on to plating, while the
    narrow one still believes the pot is empty and goes to fetch meat. That is a
    different subtask, in a different order - the thing random search could not
    find because the target region is 2-9 cells out of ~90.
    """
    fn, fm, fw = fov_triple
    # Human's working pose. Defaults to lower-left facing north; varied by
    # enumerate_designed() so the differential regions land in different places.
    if anchor is None:
        anchor = (3, height - 3)

    # Cells separating narrow from mid, and mid from wide. Require both so all
    # three hypotheses can differ, not just the extreme pair.
    d_nm = set(differential_cells(anchor, ori, fn, fm, width, height))
    d_mw = set(differential_cells(anchor, ori, fm, fw, width, height))
    if not d_nm or not d_mw:
        return None

    cells = [['X'] * width for _ in range(height)]
    x_lo, x_hi, y_lo, y_hi = 2, width - 3, 2, height - 3
    for y in range(y_lo, y_hi + 1):
        for x in range(x_lo, x_hi + 1):
            cells[y][x] = ' '

    def wall_slot_near(target):
        """Nearest wall tile adjacent to open floor - stations must be embedded
        in a wall to be interactable, as in every hand-designed steak layout."""
        best, bd = None, 1e9
        for y in range(height):
            for x in range(width):
                if cells[y][x] != 'X':
                    continue
                if not any(0 <= x + dx < width and 0 <= y + dy < height
                           and cells[y + dy][x + dx] == ' '
                           for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    continue
                d = (x - target[0]) ** 2 + (y - target[1]) ** 2
                if d < bd:
                    best, bd = (x, y), d
        return best

    used = {}
    human_reserved = None
    # Pot into the mid|wide differential region; chopping board into narrow|mid.
    # Those two stations are the ones whose state the teammate actually changes.
    for sym, region in (('P', d_mw), ('B', d_nm)):
        slot = None
        for t in sorted(region, key=lambda c: -((c[0] - anchor[0]) ** 2 + (c[1] - anchor[1]) ** 2)):
            cand = wall_slot_near(t)
            if cand and cand not in used:
                slot = cand
                break
        if slot is None:
            return None
        used[slot] = sym

    # MISHA NEW CHANGE - SPREAD the routine stations around the whole perimeter
    # instead of packing them into the first free wall tiles.
    #
    # The first version filled left-to-right along the top wall, producing
    # "XXDMOWSXXXBXPXX" - every station in one row, the human in the far corner.
    # Every rollout on those ended STUCK with 23 stalls, which fails the
    # full-observability requirement outright: a layout the human cannot work is
    # useless no matter how good its visibility geometry is. Compare the
    # hand-designed steak layouts, which distribute stations across the top wall
    # AND the side walls with open floor between them.
    #
    # Only P and B need to sit in the differential regions - they are the ones
    # whose state the teammate changes. Everything else should be placed for
    # ergonomics.
    perimeter = []
    for x in range(x_lo, x_hi + 1):
        perimeter.append((x, y_lo - 1))          # top wall
        perimeter.append((x, y_hi + 1))          # bottom wall
    for y in range(y_lo, y_hi + 1):
        perimeter.append((x_lo - 1, y))          # left wall
        perimeter.append((x_hi + 1, y))          # right wall

    def far_from_used(c):
        if not used:
            return 1e9
        return min((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 for u in used)

    for sym in STATIONS:
        if sym in used.values():
            continue
        cands = [c for c in perimeter if c not in used and cells[c[1]][c[0]] == 'X'
                 and c != human_reserved]
        if not cands:
            return None
        # farthest-point placement: keeps the kitchen walkable and stops the
        # human from having every station crowded into one corner
        used[max(cands, key=far_from_used)] = sym

    human = anchor
    robot = (width - 4, 2)
    cells[human[1]][human[0]] = '2'
    cells[robot[1]][robot[0]] = '1'
    for (x, y), sym in used.items():
        cells[y][x] = sym
    return dict(name=f"fov_designed_{idx}", grid=[''.join(r) for r in cells],
                fov_triple=tuple(fov_triple), human_pos=human, robot_start=robot,
                pot_pos=[p for p, s in used.items() if s == 'P'][0],
                board_pos=[p for p, s in used.items() if s == 'B'][0],
                n_diff_nm=len(d_nm), n_diff_mw=len(d_mw))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("Differential-visibility feasibility for candidate FOV triples.")
    print("A pair can only ever disagree if it has cells visible to one and not")
    print("the other. narrow|mid = cells separating those two, etc.\n")
    print(f"{'fov triple':<20} {'pose':<22} {'n|m':>5} {'m|w':>5} {'n|w':>5}")
    triples = [(30, 90, 180), (40, 100, 180), (60, 120, 180), (20, 80, 160), (50, 110, 170)]
    for t in triples[:max(1, n)]:
        rows = describe(t)
        # report the best pose for this triple - the one maximising the weakest pair
        best = max(rows, key=lambda r: min(r[2], r[3]))
        pos, ori, nm, mw, nw = best
        print(f"{str(t):<20} {str(pos)+' facing '+ori:<22} {nm:>5} {mw:>5} {nw:>5}")
        if min(nm, mw) == 0:
            print(f"    ^ UNUSABLE: a pair has no discriminating cell anywhere on the grid")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# MISHA NEW CHANGE - enumerate a LARGE designed set.
#
# Validated on the first designed layout (15x9, anchor (3,6) facing north,
# triple (30,90,180)): all three FOVs finish the orders (DONE, 129 steps, 12
# distinct subtasks), the full-vision control stalls 0% of steps, and two of the
# three FOV pairs show real phase-corrected divergence sustained into the late
# half (clean=2, late=2) - narrow FOVs perform pickup_onion, full vision skips
# it for pickup_garnish.
#
# Because the visibility condition is pure geometry there is nothing to search:
# vary room size / anchor pose / FOV triple / which station goes in which
# differential region, and every combination is either feasible by construction
# or rejected instantly by describe(). That is the whole reason this replaces
# the random search, which needed the pot to land in a 2-9 cell target by luck
# and returned mean divergence 0.1 over 681 trials.

ROOM_SIZES = [(15, 9), (17, 9), (17, 11), (19, 11), (15, 11)]
FOV_TRIPLES = [(30, 90, 180), (40, 100, 180), (30, 100, 170), (35, 95, 175),
               (25, 85, 165), (45, 105, 180), (30, 80, 160), (50, 110, 180)]


def enumerate_designed(max_n=200):
    """Yield designed layout configs across room sizes, poses and FOV triples."""
    out = []
    for (w, h) in ROOM_SIZES:
        for triple in FOV_TRIPLES:
            for ax in (3, w // 2, w - 4):
                for ay in (h - 3, h // 2):
                    for ori_name, ori in DIRECTIONS.items():
                        if len(out) >= max_n:
                            return out
                        cfg = build_designed(
                            f"{w}x{h}_{triple[0]}_{triple[1]}_{triple[2]}_{ax}_{ay}_{ori_name}",
                            triple, width=w, height=h, anchor=(ax, ay), ori=ori)
                        if cfg:
                            cfg["room"] = (w, h)
                            cfg["anchor"] = (ax, ay)
                            cfg["ori"] = ori_name
                            out.append(cfg)
    return out
