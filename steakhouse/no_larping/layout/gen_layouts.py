"""Hand-designed coordination layouts for the wall-occlusion FOV experiment.

Legend: # wall (blocks sight AND movement)   X counter (see-through, stages items)
        P grill  B board  W sink  M meat  O onion  D plate  S serve
        1 human  2 robot
"""
import os

LAYOUTS = {}

# 1. Two rooms, one wall, two pass-through counters. The human's side can make a
#    garnish_dish; only the robot's side can cook. Neither can finish alone.
# REMOVED: coop_hatch was a HARD-SPLIT layout (human and robot walled apart,
# recipe only completable across a hatch). Both agents can reach the hatch
# counters, but the pair never completed a single order in 600 ticks at any
# cone. Deleted rather than left in the suite as a silent zero.

# 2. Everything is shared except the grill and the meat, which sit in an annex
#    reachable only through one doorway on the far side. A wide cone finds the
#    door; a narrow one hunts for it.
def _blank(w, h):
    """all-floor interior inside a solid wall border"""
    g = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = " "
    return g


def _meat_annex():
    W, H = 25, 11
    g = _blank(W, H)
    for y in range(1, H - 1):        # wall sealing off the annex ...
        g[y][20] = "#"
    g[5][20] = " "                   # ... with exactly one doorway
    for x, y, c in [(1, 1, "O"), (5, 1, "D"), (1, 5, "W"), (1, 7, "B"), (1, 9, "S"),
                    (23, 1, "M"), (23, 8, "P"), (4, 3, "1"), (10, 6, "2")]:
        g[y][x] = c
    for x in range(13, 16): g[3][x] = "X"
    for x in range(12, 15): g[7][x] = "X"
    return ["".join(r) for r in g]


LAYOUTS["meat_annex"] = _meat_annex()

# 3. One big room broken by free-standing pillars. No hard partition, but no
#    single vantage point sees the whole kitchen either.
LAYOUTS["pillars"] = [
    "#######################",
    "#O    D         M     #",
    "#                     #",
    "#   ##      ##     ## #",
    "#   ##  1   ##     ## #",
    "#           XX        #",
    "#      ##        ##   #",
    "#      ##   2    ##   #",
    "#         XX          #",
    "#W   ##         ##   B#",
    "#    ##         ##   P#",
    "#S                    #",
    "#######################",
]

# 4. Two rooms with hatches at OPPOSITE ends. The robot must choose which hatch
#    to stage on, and only the right choice is inside the human's cone.
LAYOUTS["double_hatch"] = [
    "#########################",
    "#O      X#######        #",
    "#        #      #      M#",
    "#D  1    #      #       #",
    "#        #      #   2   #",
    "#        #      #       #",
    "#W       #      #   P   #",
    "#        #      #       #",
    "#B       #######X      S#",
    "#   XXX             XXX #",
    "#########################",
]

# 5. The serving hatch is around a corner from the human's work area, so the
#    human must leave its alcove (and lose sight of it) to deliver.
# REMOVED: blind_serve was a HARD-SPLIT layout (human and robot walled apart,
# recipe only completable across a hatch). Both agents can reach the hatch
# counters, but the pair never completed a single order in 600 ticks at any
# cone. Deleted rather than left in the suite as a silent zero.

# 6. A long hall. Nothing is hidden, but a 30-degree cone needs many turns to
#    sweep 25 tiles of width.
LAYOUTS["long_hall"] = [
    "#########################",
    "#O  D                  M#",
    "#                       #",
    "#     ####       ####   #",
    "#  1                 2  #",
    "#XXXX            XXXXXXX#",
    "#     ####       ####   #",
    "#                       #",
    "#W B                 P S#",
    "#########################",
]

# 7. Switchback corridors: sightlines are never longer than one leg, so every
#    cone width is forced to walk for its information.
LAYOUTS["switchback"] = [
    "#######################",
    "#O   D#         #    M#",
    "#     #         #     #",
    "#  1  #    X    #  2  #",
    "#     #         #     #",
    "#  ####    #    ####  #",
    "#          #          #",
    "####  ######  ######  #",
    "#W          X        P#",
    "#B                   S#",
    "#######################",
]

# 8. A cross wall splits the kitchen into four quadrants. Each quadrant holds
#    part of the recipe, and the arms are pierced by narrow doorways with a
#    handoff counter beside each - so you can always walk through, but you can
#    almost never SEE through.
def _quad_cross():
    W, H = 23, 13
    g = _blank(W, H)
    cx, cy = 11, 6
    for y in range(1, H - 1): g[y][cx] = "#"       # vertical arm
    for x in range(1, W - 1): g[cy][x] = "#"       # horizontal arm
    for y, x in [(3, cx), (9, cx)]:  g[y][x] = " "   # N/S doorways
    for y, x in [(cy, 5), (cy, 17)]: g[y][x] = " "   # W/E doorways
    for y, x in [(2, cx), (10, cx), (cy, 4), (cy, 18)]:
        g[y][x] = "X"                                # a handoff counter by each door
    for x, y, c in [(1, 1, "O"), (1, 3, "D"), (1, 9, "W"), (1, 11, "B"),
                    (21, 1, "M"), (21, 3, "P"), (21, 11, "S"),
                    (4, 2, "1"), (16, 2, "2")]:
        g[y][x] = c
    return ["".join(r) for r in g]


LAYOUTS["quad_cross"] = _quad_cross()

def _connected(rows, extra_block=()):
    """Every floor cell reachable from the first player start."""
    H, W = len(rows), len(rows[0])
    block = set(extra_block)
    floor = {(x, y) for y in range(H) for x in range(W)
             if rows[y][x] in " 123456789" and (x, y) not in block}
    starts = [(x, y) for y in range(H) for x in range(W) if rows[y][x] in "123456789"]
    if not starts:
        return set()
    seen, q = {starts[0]}, [starts[0]]
    while q:
        x, y = q.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, y + dy)
            if c in floor and c not in seen:
                seen.add(c)
                q.append(c)
    return seen


def _components(rows):
    """All connected floor components (layouts may be split ON PURPOSE)."""
    H, W = len(rows), len(rows[0])
    floor = {(x, y) for y in range(H) for x in range(W)
             if rows[y][x] in " 123456789"}
    comps, seen = [], set()
    for c0 in sorted(floor):
        if c0 in seen:
            continue
        comp, q = {c0}, [c0]
        while q:
            x, y = q.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (x + dx, y + dy)
                if c in floor and c not in comp:
                    comp.add(c)
                    q.append(c)
        seen |= comp
        comps.append(frozenset(comp))
    return comps


def _signature(rows):
    """(number of components, station set reachable by EACH player start).

    Comparing whole-floor counts was wrong twice over: a conversion always
    removes one cell, and the split layouts have two components by design. What
    must be preserved is that nothing gets cut off and no player loses access to
    a station - that is all.
    """
    stations = set("PBWMODS")
    comps = _components(rows)
    H, W = len(rows), len(rows[0])
    starts = [(x, y) for y in range(H) for x in range(W)
              if rows[y][x] in "123456789"]
    per_player = []
    for st in sorted(starts):
        comp = next((c for c in comps if st in c), frozenset())
        kinds = set()
        for (x, y) in comp:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ch = rows[y + dy][x + dx]
                if ch in stations:
                    kinds.add(ch)
        per_player.append(frozenset(kinds))
    return len(comps), tuple(per_player)


def line_with_counters(rows):
    """Turn wall-hugging floor into counters, greedily and safely.

    Stashing is the ladder's only escape hatch for an item with no current use,
    so a layout with a handful of counters deadlocks as soon as they fill -
    measured: switchback had 2, both ended up holding garnish, and after that
    neither agent had ANY legal subtask for the rest of the episode.

    Counters are see-through, so this costs nothing in FOV terms: the walls
    behind them still do all the occluding.
    """
    g = [list(r) for r in rows]
    H, W = len(g), len(g[0])
    keep = _signature(["".join(r) for r in g])
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if g[y][x] != " ":
                continue
            if not any(g[y + dy][x + dx] == "#"
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue                      # hug walls only, keep the middle open
            g[y][x] = "X"
            if _signature(["".join(r) for r in g]) != keep:
                g[y][x] = " "                 # would cut something off: revert
    return ["".join(r) for r in g]


TEMPLATE = '''{{
    "grid":  """{grid}""",
    "start_order_list": ['steak', 'steak', 'steak'],
    "cook_time": 15,
    "delivery_reward": 20,
    'num_items_for_steak': 1,
    'chop_time': 5,
    'wash_time': 5,
    "rew_shaping_params": None
}}
'''


def write(outdir):
    os.makedirs(outdir, exist_ok=True)
    written = []
    for name, rows in LAYOUTS.items():
        w = max(len(r) for r in rows)
        ragged = [(i, len(r)) for i, r in enumerate(rows) if len(r) != w]
        if ragged:
            print(f"  [pad] {name}: rows {ragged} -> width {w}")
            rows = [r.ljust(w - 1, " ") + "#" if len(r) < w else r for r in rows]
            rows = [r if len(r) == w else r.ljust(w, "#") for r in rows]
        rows = line_with_counters(rows)
        indent = " " * 16
        grid = ("\n" + indent).join(rows)
        with open(os.path.join(outdir, name + ".layout"), "w") as f:
            f.write(TEMPLATE.format(grid=grid))
        written.append((name, w, len(rows)))
    return written


if __name__ == "__main__":
    import sys
    for n, w, h in write(sys.argv[1]):
        print(f"  {n:<16} {w}x{h}")
