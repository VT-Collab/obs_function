"""EXPERIMENT, not a validated layout: real_010_rebuilt's own network,
unchanged (imported directly, never rebuilt or edited), with its Road
swapped for highway_env's own RegulatedRoad -- see
highway_env/road/regulation.py. See real_001_regulated_road.py's own
docstring for the full reasoning (why this should be a rendering no-op,
why this lives in a separate file rather than editing real_010_rebuilt.py
directly) -- identical here, just for a different layout number. Verified
by layout/layouts/_compare_regulated_render.py across all ten real_NNN
layouts at once.
"""
from highway_env.road.regulation import RegulatedRoad

from real_010_rebuilt import (  # noqa: F401 -- re-exposed for any tooling that expects these names
    _build_network, _HUMAN_ROUTE_LANES, _ROBOT_ROUTE_LANES,
    HUMAN_ROUTE, ROBOT_ROUTE, route_adjacent_lane_indexes,
)


def build_road() -> RegulatedRoad:
    return RegulatedRoad(network=_build_network())
