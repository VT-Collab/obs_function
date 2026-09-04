"""Crossing turn: a single 4-way intersection where the human turns LEFT
across oncoming traffic -- built entirely from build_scene.py's
add_four_way, unlike mega_scene.py's right-turn-only drive. A right turn
never crosses an opposing lane's path; a left turn does, right at the box
center, which is exactly the classic real-world "did you see the oncoming
car before you turned" FOV scenario mega_scene.py doesn't actually create
(its own turns are a right turn and a converging merge, never a straight
crossing conflict).

HUMAN_ROUTE: south arm, straight in, LEFT turn onto the west arm.
ROBOT_ROUTE: north arm, straight through to the south arm -- the exact
oncoming traffic the human's left turn must cross. Full vision (no FOV
limit on the robot; see highway_env/human/limit_vision_human.py), so this
is a clean test of whether a robot that can infer the human is FOV-limited
should be more cautious approaching this same box than one that assumes
full mutual visibility.

Kept a 2-lane intersection (unlike mega_scene.py's fully single-laned
network) for background-traffic realism; only the two specific fork nodes
the human and robot actually use are pruned down to their one intended
movement (see _layout_utils.prune_to_route's own docstring for why that's
necessary at all: add_four_way's own turn/straight options at one corner
share a single stop-line node, which the route-follower's fork-ranking
logic can't break a tie on by chance without this).
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import add_four_way
from _layout_utils import polyline, prune_to_route, route_adjacent_lane_indexes as _route_adjacent

CENTER = (0.0, 0.0)
ACCESS = 70.0
N_LANES = 2


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=CENTER, n_vertical=N_LANES, n_horizontal=N_LANES, access_length=ACCESS, prefix="fw_")
    prune_to_route(net, _HUMAN_ROUTE_LANES, _ROBOT_ROUTE_LANES)
    return net


def build_road() -> Road:
    return Road(network=_build_network())


# South -> straight in -> LEFT turn -> west arm. add_four_way's own
# convention: from corner c, left turn lands on next_c=(c+1)%4; south=0,
# so left lands on corner 1 (west).
_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il1_0", 0), ("fw_il1_0", "fw_o1_0", 0),
]

# North -> straight through -> south arm: opp_c=(2+2)%4=0. The oncoming
# traffic the human's own left turn has to cross.
_ROBOT_ROUTE_LANES = [
    ("fw_o2_0", "fw_ir2_0", 0), ("fw_ir2_0", "fw_il0_0", 0), ("fw_il0_0", "fw_o0_0", 0),
]

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
