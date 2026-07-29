"""
Geometry-only scan: for a given layout, how often do 60/120/180 degree FOVs
actually disagree about what's visible?

This is the thing that actually determines whether FOV inference is possible
at all. If two FOVs never disagree about what a human standing at position P
facing direction D can see, then a human's actions can never reveal which of
those two FOVs they have - no filter, however good, can do better than a
coin flip in that case. This is exactly what happened on the layout
evaluate_bayesian.py's live QMDP test uses: 0/30 episodes ever produced any
FOV-distinguishing evidence, even though the filter's Bayesian math is
verified correct (see bayesian_inference.py and evaluate_bayesian.py's
"episodes w/ any evidence" column). So before blaming the inference, check
whether the scenario even contains evidence.

This is cheap (just cone geometry, no QMDP planner, no pickle loading) so we
can sweep many layouts/viewpoints quickly - unlike the full live episode
eval in evaluate_bayesian.py, which is pinned to one already-computed layout
because building the QMDP planner for a new layout would trigger an
expensive (likely multi-hour) recomputation.

Two metrics, both printed by main():
  - "uniform" rate: sample every walkable tile x every facing x every
    station, unweighted. Reflects the raw geometry of the cone angles more
    than the layout's real gameplay.
  - "at-station" rate: only look from tiles adjacent to some station A while
    facing it (the realistic "working at A" pose), and check visibility of
    every OTHER station B. This is the practically relevant question -
    "while doing something at A, would you notice something at B?" - so it's
    a better proxy for whether a given layout will actually produce
    FOV-diagnostic moments during real play.

Run with: python -m my_methods.bayesian.fov_divergence_scan

-----------------------------------------------------------------------------
RESULTS (last measured; re-run main() after editing layouts, these will
drift - "at-station" is the more decision-relevant metric, see above):

  Geometric ranking (this file's static scan):
    1. steak_tshape (existing)              uniform 21.4% / at-station 42.9%
    2. steak_parrallel (existing)           uniform 21.0% / at-station 33.3%
    3. steak_side_4 (existing)              uniform 20.7% / at-station 30.4%
    4. corridor_stations_behind (custom)    uniform 22.6% / at-station 30.0%
       stations bunched at both ends of a narrow room; working at one end
       puts the other end's stations behind/beside you
    5. steak_island (existing)              uniform 21.2% / at-station 29.2%
    6. steak_side_3 (existing)              uniform 20.7% / at-station 28.6%
    ...
    - all_stations_clustered (custom)       uniform  8.3% / at-station 19.0%
      every station crammed directly around a central spawn point -
      deliberately the worst case, kept as a sanity-check control
    - original (real_user_lvls.csv)         uniform 20.7% / at-station 21.4%
      the layout evaluate_bayesian.py's live QMDP test is stuck using, since
      switching its layout would invalidate the already-cached ~6GB QMDP
      planner (see that file's docstring).
    - steak_practice (existing)             uniform 18.2% / at-station 12.5%

  IMPORTANT: this geometric ranking turned out to be a WEAK PREDICTOR of
  real behavioral divergence (see evaluate_bayesian_lightweight.py). We ran
  full live human+filter episodes (see that file) on the top geometric
  candidates and got:
    - steak_tshape (rank #1 geometrically):  0/45 episodes ever informative
    - steak_parrallel (rank #2):              9/24 episodes informative, 33.3% accuracy
    - steak_side_4 (rank #3):                24/24 episodes informative, but only 37.5%
      accuracy (small sample, worth rechecking before trusting)
    - steak_island (rank #5, near the bottom of the "matters" list!):
      60/60 episodes informative, 88.3% final-guess accuracy over 60 episodes
      -> the actual best layout found, despite a mediocre geometric score.

  Why the geometric metric misleads: in_bound() has a special case where any
  tile immediately beside the player is always visible regardless of FOV,
  and the human is always standing right next to whatever station it's
  about to act on - so "can 60/120/180 disagree about seeing station B from
  near station A" doesn't capture whether the human's subtask decision ever
  actually depends on B's freshness at the moments it's near A. There's no
  known cheap proxy for that - only running the actual human+filter loop
  and checking whether belief ever leaves the uniform prior (which
  evaluate_bayesian_lightweight.py does) reveals it.

Takeaway: use this file to rule out obviously-bad layouts and get more
existing-layout candidates to try (no need to invent one - the repo already
has plenty), but always confirm any candidate with a live run in
evaluate_bayesian_lightweight.py before trusting it. "steak_island" is the
current best validated choice (88.3% accuracy, 100% informative episodes).
If evaluate_bayesian.py's paper-faithful QMDP-teammate test ever moves off
the "unnamed_layout" cache to use steak_island instead, that's a real,
possibly large compute cost (see that file's docstring) - confirm with
whoever owns that time/resource budget before starting it.
-----------------------------------------------------------------------------
"""
import math
import itertools
from overcooked_ai_py.helpers import init_steak_env
from overcooked_ai_py.mdp.overcooked_mdp import Direction, SteakHouseGridworld

FOV_CANDIDATES = (60, 120, 180)

# stations/dispensers worth checking visibility of; floor ('X' walls) and
# empty floor (' ') aren't "things a human needs to notice"
POINT_OF_INTEREST_SYMBOLS = {'O', 'M', 'P', 'D', 'B', 'W', 'S'}


def in_bound(player_pos, player_ori, loc, vision_bound):
    """Same forward-facing cone check as SteakLimitVisionHumanModel.in_bound,
    reimplemented standalone so we don't need a full agent/env/state to call it."""
    if vision_bound == 0:
        return True
    px, py = player_pos
    lx, ly = loc
    ori = Direction.DIRECTION_TO_INDEX[player_ori]

    if ori in (0, 1) and ly == py and lx in (px - 1, px + 1):
        return True
    if ori in (2, 3) and lx == px and ly in (py - 1, py + 1):
        return True
    rot_angle = {0: math.radians(180), 2: math.radians(270),
                 1: math.radians(0), 3: math.radians(90)}[ori]

    c, s = math.cos(rot_angle), math.sin(rot_angle)
    shifted_x, shifted_y = lx - px, (ly - py) * -1
    rot_x = c * shifted_x - s * shifted_y
    rot_y = s * shifted_x + c * shifted_y

    y = -abs(rot_x * math.cos(math.radians(vision_bound)))
    return y >= rot_y


def _points_of_interest(mdp):
    return {
        symbol: pos_list for symbol, pos_list in mdp.terrain_pos_dict.items()
        if symbol in POINT_OF_INTEREST_SYMBOLS
    }


def scan_mdp(mdp, fov_candidates=FOV_CANDIDATES):
    """Uniform metric: every walkable tile x every facing x every station.
    Returns (disagreement_rate, n_pairs_checked)."""
    positions = mdp.get_valid_player_positions()
    pois = [pos for pos_list in _points_of_interest(mdp).values() for pos in pos_list]

    total, disagree = 0, 0
    for pos, ori, poi in itertools.product(positions, Direction.ALL_DIRECTIONS, pois):
        if poi == pos:
            continue
        visible = [in_bound(pos, ori, poi, fov / 2) for fov in fov_candidates]
        total += 1
        if len(set(visible)) > 1:
            disagree += 1

    return (disagree / total if total else 0.0), total


def scan_mdp_at_station(mdp, fov_candidates=FOV_CANDIDATES):
    """At-station metric: only from tiles adjacent to some station A, facing
    A (the realistic "working here" pose), checking visibility of every
    other station B. Returns (disagreement_rate, n_pairs_checked)."""
    pois = _points_of_interest(mdp)
    valid = set(mdp.get_valid_player_positions())
    facing_deltas = [(-1, 0, Direction.EAST), (1, 0, Direction.WEST),
                      (0, -1, Direction.SOUTH), (0, 1, Direction.NORTH)]

    total, disagree = 0, 0
    for positions_a in pois.values():
        for a_pos in positions_a:
            for dx, dy, ori in facing_deltas:
                stand = (a_pos[0] + dx, a_pos[1] + dy)
                if stand not in valid:
                    continue
                for positions_b in pois.values():
                    for b_pos in positions_b:
                        if b_pos == a_pos:
                            continue
                        visible = [in_bound(stand, ori, b_pos, fov / 2) for fov in fov_candidates]
                        total += 1
                        if len(set(visible)) > 1:
                            disagree += 1

    return (disagree / total if total else 0.0), total


def scan_layout(lvl_str, fov_candidates=FOV_CANDIDATES):
    """Scan a raw grid string (as used by evaluate_bayesian.py's LVL_STR)."""
    env = init_steak_env(lvl_str, horizon=1)
    return scan_mdp(env.mdp, fov_candidates)


def scan_named_layout(layout_name, fov_candidates=FOV_CANDIDATES):
    """Scan an existing layout file from overcooked_ai_py/data/layouts/."""
    mdp = SteakHouseGridworld.from_layout_name(layout_name)
    return scan_mdp(mdp, fov_candidates)


# ---- custom grids designed to sit at the two extremes ----

# FOV matters A LOT: stations sit at both ends of a long, narrow room.
# Working at one end (e.g. chopping at B/W/S) puts the other end's stations
# (O/M/P/D) far behind and off to the side - a 60 deg cone regularly misses
# them, a 180 deg cone regularly catches them.
CORRIDOR_STATIONS_BEHIND = """XXXXXXXXXXXXXXX
XOMXXXXXXXXXPXX
X             X
X             X
X   1     2   X
X             X
X             X
XDXXXXXXXXXXXBX
XXXXXXXXXXWXSXX
XXXXXXXXXXXXXXX"""

# FOV barely matters: every station is crammed directly around a central
# spawn point, all within a narrow forward cone no matter which way you face.
ALL_STATIONS_CLUSTERED = """XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
XXXXOMPDXXXXXXX
XXXX1 2XXXXXXXX
XXXXBWSSXXXXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX"""

ORIGINAL_LAYOUT = """XXXXXXXXXXXXXXX
XXXOXMXXXPXXDXX
XX           XX
XX 2         XX
XX        1  XX
XX   XBWXX   XX
XX   XXSSX   XX
XX           XX
XX           XX
XX           XX
XXXXXXXXXXXXXXX"""

CUSTOM_LAYOUTS = {
    "corridor_stations_behind (custom, high divergence)": CORRIDOR_STATIONS_BEHIND,
    "all_stations_clustered (custom, low divergence)": ALL_STATIONS_CLUSTERED,
    "original (real_user_lvls.csv, live QMDP eval uses this)": ORIGINAL_LAYOUT,
}

EXISTING_LAYOUT_NAMES = [
    "steak_practice", "steak_island", "steak_island2", "steak_parrallel",
    "steak_none_3", "steak_mid_2", "steak_side_3", "steak_side_4", "steak_tshape",
]


def main():
    results = []
    for name, lvl_str in CUSTOM_LAYOUTS.items():
        env = init_steak_env(lvl_str, horizon=1)
        uniform_rate, uniform_n = scan_mdp(env.mdp)
        station_rate, station_n = scan_mdp_at_station(env.mdp)
        results.append((name, uniform_rate, uniform_n, station_rate, station_n))
    for name in EXISTING_LAYOUT_NAMES:
        mdp = SteakHouseGridworld.from_layout_name(name)
        uniform_rate, uniform_n = scan_mdp(mdp)
        station_rate, station_n = scan_mdp_at_station(mdp)
        results.append((f"{name} (existing)", uniform_rate, uniform_n, station_rate, station_n))

    results.sort(key=lambda r: -r[3])
    print(f"{'layout':>55} | {'uniform':>9} | {'at-station':>10}")
    for name, u_rate, u_n, s_rate, s_n in results:
        print(f"{name:>55} | {u_rate * 100:>8.1f}% | {s_rate * 100:>9.1f}%  (n={u_n},{s_n})")


if __name__ == "__main__":
    main()
