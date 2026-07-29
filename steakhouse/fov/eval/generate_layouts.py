"""Generate a batch of VALID steak layouts across a contention gradient:
small rooms (cramped -> high robot-influence) to large (spread), with stations
either spread evenly around the room or clustered on one side. Every layout is
valid by construction (single rectangular room: all interior floor connected,
each station on a border cell adjacent to floor, 2 agents on interior floor).
Only layouts that successfully load are written."""
import sys, os
sys.path.insert(0, "/Users/mishafu/Desktop/obs_function/steakhouse")
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld

STATIONS = ['P', 'B', 'W', 'M', 'O', 'D', 'S']   # all 7 the steak task needs
LAYDIR = "/Users/mishafu/Desktop/obs_function/steakhouse/overcooked_ai_py/data/layouts"
CONFIG = ('steak, steak, steak, steak', 15, 5, 5)


def build(rw, rh, cluster):
    W, H = rw + 2, rh + 2
    g = [['X'] * W for _ in range(H)]
    for r in range(1, rh + 1):
        for c in range(1, rw + 1):
            g[r][c] = ' '
    ring = []                                        # border cells adjacent to floor, clockwise
    for c in range(1, rw + 1): ring.append((0, c))
    for r in range(1, rh + 1): ring.append((r, W - 1))
    for c in range(rw, 0, -1): ring.append((H - 1, c))
    for r in range(rh, 0, -1): ring.append((r, 0))
    n = len(ring)
    if n < 7:
        return None
    raw = list(range(7)) if cluster else [round(i * n / 7) for i in range(7)]
    used, pos = set(), []
    for i in raw:
        j = i % n
        while j in used:
            j = (j + 1) % n
        used.add(j); pos.append(j)
    for st, j in zip(STATIONS, pos):
        r, c = ring[j]; g[r][c] = st
    floor = [(r, c) for r in range(1, rh + 1) for c in range(1, rw + 1)]
    if len(floor) < 2:
        return None
    if cluster:
        a1, a2 = floor[len(floor) // 2], floor[len(floor) // 2 - 1]
    else:
        a1, a2 = floor[0], floor[-1]
    g[a1[0]][a1[1]] = '1'; g[a2[0]][a2[1]] = '2'
    return '\n'.join(''.join(row) for row in g)


def write_layout(name, grid):
    orders, cook, chop, wash = CONFIG
    body = ('{\n    "grid":\n"""' + grid + '""",\n'
            f"    \"start_order_list\": '{orders}',\n"
            f"    \"cook_time\": {cook},\n    \"delivery_reward\": 20,\n"
            f"    'num_items_for_steak': 1,\n    'chop_time': {chop},\n    'wash_time': {wash},\n"
            "    \"rew_shaping_params\": None\n}\n")
    with open(os.path.join(LAYDIR, name + ".layout"), "w") as f:
        f.write(body)


SPREAD = [(4, 3), (5, 4), (6, 4), (7, 5), (8, 5), (6, 6), (8, 6), (10, 6), (5, 7), (9, 7)]
CLUSTER = [(3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (6, 5), (3, 5), (7, 4)]


def main():
    made = []
    for i, (rw, rh) in enumerate(SPREAD):
        g = build(rw, rh, False)
        if g is None: continue
        name = f"steak_gs{i:02d}"
        write_layout(name, g)
        try:
            SteakHouseGridworld.from_layout_name(name, start_order_list=['steak'] * 4)
            made.append(name)
        except Exception as e:
            os.remove(os.path.join(LAYDIR, name + ".layout"))
            print(f"  drop {name}: {e}")
    for i, (rw, rh) in enumerate(CLUSTER):
        g = build(rw, rh, True)
        if g is None: continue
        name = f"steak_gc{i:02d}"
        write_layout(name, g)
        try:
            SteakHouseGridworld.from_layout_name(name, start_order_list=['steak'] * 4)
            made.append(name)
        except Exception as e:
            os.remove(os.path.join(LAYDIR, name + ".layout"))
            print(f"  drop {name}: {e}")
    print("GENERATED " + str(len(made)) + ": " + ",".join(made))


if __name__ == "__main__":
    main()
