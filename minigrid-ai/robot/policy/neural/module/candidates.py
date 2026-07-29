# ═══════════════════════════════════════════════════════════════════════════
# robot/policy/neural/baseline/no_fov/candidates.py
#
# CandidateFinder - answers "what COULD be revealed right now, for a human with
# THIS knowledge base?" It is AssistBase split in half: it KEEPS the world-
# geometry facts the robot is entitled to know (where the solution's keys/doors/
# goal are, what is reachable without passing a locked door, which room lies past
# which door), and DROPS the phase/patience rule that decides which fact to say
# and when.
#
# all_candidates() returns all 5 reveal types at once, unprioritised - each is a
# concrete (color, loc) or None. None means revealing it would be meaningless or
# harmful (already known / dead-end room / unreachable goal) - a validity check,
# not a preference.
#
# The `kb` argument is a SHADOW knowledge base - a hypothesis about what a human
# with some FOV has seen - never the human's real KB. The FOV module
# (module/fov_module.py) runs this against each of its 3 shadow agents and combines
# the results by the posterior to build the NEED vector. That is the only caller.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

from robot.policy.deterministic._base import AssistBase


class CandidateFinder(AssistBase):
    """All currently-valid reveal candidates for one KB, with no priority ordering
    and no decision about whether to act on any of them. Reuses AssistBase's
    world-fact helpers (self.solution, _accessible_zone, _room_zone_past_door,
    _doorless_zone, _hallway_zone).

    Never reads self.shadow - all_candidates() is handed a KB by the caller.
    _get_assumed_fov only exists because AssistBase.reset() calls it to build a
    shadow; that shadow is never advanced or read here."""

    def _get_assumed_fov(self) -> int:
        return 120  # dummy: the shadow AssistBase.reset builds from this is unused

    def all_candidates(self, state, kb: dict) -> dict:
        carrying = getattr(state, "carrying", None)
        held_type = getattr(carrying, "type", None) if carrying else None
        dead = kb.get("dead_door_colors", set())

        out = {"key": None, "door": None, "goal": None, "dead_room": None, "empty_room": None}

        if held_type == "key":
            held_color = carrying.color

            if held_color not in dead:
                if held_color not in kb.get("seen_doors", {}):
                    loc = self.solution["doors"].get(held_color)
                    if loc:
                        goal_loc = self.solution.get("goal")
                        room_zone = self._room_zone_past_door(state, loc)
                        if goal_loc is not None and goal_loc in room_zone:
                            out["door"] = (held_color, loc)

                door_open = held_color in kb.get("door_open_colors", set())
                if door_open and kb.get("goal_loc") is None:
                    loc = self.solution.get("goal")
                    if loc and loc in self._accessible_zone(state):
                        out["goal"] = (None, loc)

                room_cells = kb.get("goal_room_cells", {}).get(held_color, set())
                if (door_open and room_cells
                        and tuple(state.agent_pos) in room_cells
                        and not room_cells.issubset(kb.get("explored_cells", set()))
                        and self.solution.get("goal") not in room_cells):
                    out["dead_room"] = (held_color, tuple(state.agent_pos))

        if held_type != "key":
            seen_keys = kb.get("seen_keys", {})
            live_seen = {c for c in seen_keys if c not in dead}
            if not live_seen:
                accessible = self._accessible_zone(state)
                ax, ay = state.agent_pos
                key_candidates = [
                    (color, loc) for color, loc in self.solution["keys"].items()
                    if color not in dead and loc in accessible
                    and getattr(state.grid.get(*loc), "type", None) == "key"
                ]
                if key_candidates:
                    out["key"] = min(key_candidates, key=lambda c: abs(c[1][0]-ax) + abs(c[1][1]-ay))

            pos = tuple(state.agent_pos)
            if pos not in self._hallway_zone:
                room_cells = self._doorless_zone(state)
                has_key = any(loc in room_cells for loc in self.solution["keys"].values())
                if (not has_key
                        and not room_cells.issubset(kb.get("explored_cells", set()))
                        and frozenset(room_cells) not in kb.get("told_empty_rooms", set())):
                    out["empty_room"] = (None, pos)

        return out
