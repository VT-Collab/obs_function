"""
gateway_assist.py — Region-based assistance: reveal ENTRY DOOR, not exact object.

Communication mode comparison
------------------------------
LocationAssist (static/dynamic_assist.py):
    Injects the exact (x, y) of the missing key/door/goal into the human's KB.
    Human navigates directly via BFS.  Maximum information, fastest help.

GatewayAssist (this file):
    For Phase 1 (key search): instead of revealing the key's (x, y), reveals the
    ENTRY DOOR to the room that contains the correct key.  The human's exploration
    BFS then navigates toward that door, enters the room, and finds the key through
    natural exploration (FOV scanning).  This is coarser guidance — the robot says
    "go through that door" not "the key is at (x, y)."

    Phase 2 (door for held key) and Phase 3 (goal) behave identically to
    LocationAssist because at those stages exact location is critical.

Motivation
----------
The contrast between LocationAssist and GatewayAssist isolates the value of
exact-location communication.  For narrow-FOV agents, GatewayAssist may still
miss the key inside the room; for wide-FOV agents, once guided to the correct
room they find the key in very few steps.  The per-FOV performance gap is
therefore WIDER for GatewayAssist than for LocationAssist, revealing how the
communication granularity interacts with the human's perceptual capability.

This maps to the "implicit cue" vs "explicit cue" distinction in human–robot
teaming (Mutlu et al. 2013; Jiang et al. 2018): partial guidance leverages
the human's own capability while full revelation removes it entirely.
"""

from __future__ import annotations
import os, sys
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from robot.policy.deterministic._base import AssistBase


class GatewayAssist(AssistBase):
    """
    Assistance that reveals the *entry door* to the room containing the
    target object, rather than the object's exact location.

    Phase 1 (key): reveals the unlocked entry door to the correct-key's room.
    Phase 2 (locked door) and Phase 3 (goal): same as LocationAssist.

    Parameters
    ----------
    assumed_fov : int or None
        Fixed assumed FOV.  Pass None to subclass with dynamic inference.
    patience : int
        Intervention patience K.
    """

    def __init__(self, assumed_fov: int = 120, patience: int = 5):
        super().__init__(patience=patience)
        self._assumed_fov = assumed_fov

    def _get_assumed_fov(self) -> int:
        return self._assumed_fov

    # ── Override _next_needed for Phase 1 ────────────────────────────────────

    def _next_needed(self, state, kb: dict) -> Optional[Tuple]:
        """
        Phase 1 returns the ENTRY DOOR to the correct-key's room, not the key.
        Phases 2 and 3 defer to the parent (exact location reveals).
        kb is the shadow's belief KB, never the real human's.
        """
        carrying = getattr(state, "carrying", None)
        held_type = getattr(carrying, "type", None) if carrying else None
        dead = kb.get("dead_door_colors", set())

        # Phases 2 and 3: delegate to parent (holding a key)
        if held_type == "key":
            return super()._next_needed(state, kb)

        # Phase 1: not holding a useful key
        seen_keys = kb.get("seen_keys", {})
        live_seen = {c for c in seen_keys if c not in dead}
        if live_seen:
            return None  # human already knows a useful key

        goal_loc = self.solution.get("goal")
        if goal_loc is None:
            return None

        accessible = self._accessible_zone(state)

        for door_color, door_loc in self.solution["doors"].items():
            if door_color in dead:
                continue
            obj = state.grid.get(*door_loc)
            if not getattr(obj, "is_locked", False):
                continue
            room_zone = self._room_zone_past_door(state, door_loc)
            if goal_loc not in room_zone:
                continue

            # Found goal door — find its key and the KEY'S ENTRY DOOR
            key_loc = self.solution["keys"].get(door_color)
            if key_loc is None or key_loc not in accessible:
                break

            # Find the unlocked entry door to the room that contains the key
            entry_door = self._entry_door_for_location(state, key_loc)
            if entry_door is not None:
                edoor_color, edoor_loc = entry_door
                if edoor_color not in kb.get("seen_doors", {}):
                    return ("entry_door", edoor_color, edoor_loc)
            break  # goal door found; either we revealed or nothing to do

        return None

    def _entry_door_for_location(self, state, target_loc: Tuple):
        """
        Find the unlocked door that serves as the entry to the room containing
        target_loc.  Returns (door_color, door_loc) or None.

        Strategy: for each unlocked door in the solution, check if target_loc
        lies in the room on the far side of that door.
        """
        W, H = state.width, state.height
        agent_side = self._accessible_zone(state)

        for door_color, door_loc in self.solution["doors"].items():
            obj = state.grid.get(*door_loc)
            if getattr(obj, "is_locked", False):
                continue  # locked — not an entry door
            # Compute the zone on the far side of this door
            dx, dy = door_loc
            for vx, vy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nx, ny = dx + vx, dy + vy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if (nx, ny) in agent_side:
                    continue
                cell = state.grid.get(nx, ny)
                t = getattr(cell, "type", None) if cell else None
                if t in ("wall", "door"):
                    continue
                # BFS from this room-side entry to see if target_loc is in there
                from collections import deque
                zone = {(nx, ny)}
                q = deque([(nx, ny)])
                while q:
                    x, y = q.popleft()
                    for ddx, ddy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                        nnx, nny = x + ddx, y + ddy
                        if (nnx, nny) in zone:
                            continue
                        if not (0 <= nnx < W and 0 <= nny < H):
                            continue
                        obj2 = state.grid.get(nnx, nny)
                        t2 = getattr(obj2, "type", None) if obj2 else None
                        if t2 == "wall":
                            continue
                        if t2 == "door" and getattr(obj2, "is_locked", False):
                            continue
                        zone.add((nnx, nny))
                        q.append((nnx, nny))
                if target_loc in zone:
                    return (door_color, door_loc)
                break  # checked the one room side of this door
        return None

    # ── Override _reveal to handle entry_door kind ───────────────────────────

    def _reveal(self, human_kb: dict, target: Tuple, state) -> None:
        kind, color, loc = target
        if kind != "entry_door":
            super()._reveal(human_kb, target, state)
            return

        # Inject the entry door into the human's KB (and mirror into the
        # shadow's KB, same as the base class does for other kinds) so
        # exploration navigates toward it. We add to grid_cache and
        # explored_cells so _explore_all includes this door cell as a
        # frontier target and routes through it.
        obj = state.grid.get(*loc)
        if obj is None:
            return

        kbs = [human_kb] + ([self.shadow.knowledge_base] if self.shadow is not None else [])
        for kb in kbs:
            kb.setdefault("grid_cache", {})[loc] = obj
            kb.setdefault("explored_cells", set()).add(loc)
            # Mark as seen door (even though human hasn't physically visited)
            kb.setdefault("seen_doors", {})[color] = loc
