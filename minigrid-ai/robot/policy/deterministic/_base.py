"""
_base.py — AssistBase: shared logic for all three assistance conditions.

Intervention model
------------------
At each step (before the human acts) the robot:
  1. Runs a "shadow agent" at the assumed FOV θ̂ from the real agent's position,
     accumulating KB entries only for cells visible under that cone.
     Shadow simulation follows Baker et al. (2009) "Bayesian Theory of Mind":
     simulate a rational agent with the inferred parameters to predict their
     information state.
  2. Identifies the *next blocker* using the shadow's OWN simulated KB — the
     piece of information the shadow (at the assumed/inferred FOV) doesn't
     yet know. The robot never reads what the real human has actually
     discovered; every decision is driven purely by its belief.
  3. If the shadow doesn't know it, the robot starts/continues a patience
     counter K. Once K consecutive steps pass with the shadow still missing
     it, the robot reveals the object's location into the real human's KB —
     the only point where the real KB is touched, and only as a write.

This patience-threshold intervention rule approximates the cost-reduction
criterion in Javdani et al. (2015) "Shared Autonomy via Hindsight
Optimization" (IJRR): intervene when the expected remaining cost with
assistance < expected remaining cost without, discounted by intervention cost.
In our discrete setting, K ~ intervention_cost / (cost_per_step × FOV_mismatch).

The static vs. adaptive model comparison follows the design of Nikolaidis
et al. (2017) "Human-Robot Mutual Adaptation in Collaborative Tasks" (HRI),
where a fixed-model baseline is compared against an online-adapting model.
"""

from __future__ import annotations
import os, sys
from collections import deque
from typing import Dict, Optional, Set, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from human.agents.bayes_agent import BayesHumanAgent


def solve_full_map(state) -> dict:
    """
    Omniscient robot scan — reads the full grid unconditionally.
    Returns {'keys': {color: (x,y)}, 'doors': {color: (x,y)}, 'goal': (x,y)|None}.
    """
    W, H = state.width, state.height
    keys: Dict[str, Tuple] = {}
    doors: Dict[str, Tuple] = {}
    goal: Optional[Tuple] = None

    for x in range(W):
        for y in range(H):
            obj = state.grid.get(x, y)
            if obj is None:
                continue
            color = getattr(obj, "color", None)
            t = getattr(obj, "type", None)
            if t == "key":
                keys[color] = (x, y)
            elif t == "door":
                doors[color] = (x, y)
            elif t == "goal":
                goal = (x, y)

    return {"keys": keys, "doors": doors, "goal": goal}


class AssistBase:
    """
    Abstract base for patience-threshold information-assistance policies.

    Subclasses must implement _get_assumed_fov() → int.
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.shadow: Optional[BayesHumanAgent] = None
        self.solution: Optional[dict] = None
        self.timer: int = 0
        self.n_assists: int = 0

    # ── Interface for subclasses ──────────────────────────────────────────────

    def _get_assumed_fov(self) -> int:
        raise NotImplementedError

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset(self, state) -> None:
        self.solution = solve_full_map(state)
        theta = self._get_assumed_fov()
        self.shadow = BayesHumanAgent(fov=theta)
        self.shadow.init_knowledge_base(state)
        self.timer = 0
        self.n_assists = 0
        # Agent always starts in the hallway in LockedRoom-v0 — capture that
        # zone once so later steps can tell "inside a side room" from
        # "still in the hallway" (used by the empty_room hint).
        self._hallway_zone = self._doorless_zone(state)

    # ── Patience ─────────────────────────────────────────────────────────────

    def _effective_patience(self) -> int:
        """
        MISHA NEW CHANGE — the first intervention in an episode still waits
        the full patience window (don't jump in before it's clearly needed).
        But once the robot has already had to step in once, treat it as
        "actively helping" and respond faster to further gaps instead of
        re-running the full wait from scratch every single time — e.g.
        recovering from a decoy key currently means a fresh full-patience
        wait for the dead_room hint, then another fresh full wait for the
        next key reveal, then another for its door, etc., stacking up delay
        that a human's own uninterrupted exploration doesn't pay.
        """
        if self.n_assists == 0:
            return self.patience
        return max(1, self.patience // 2)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, state, human_kb: dict) -> Optional[Tuple]:
        """
        Call BEFORE the human acts each step.
        human_kb is only ever a WRITE target (the reveal destination) — the
        decision of what's missing and whether to intervene is driven
        entirely by the shadow's own belief KB (skb), never by human_kb.

        If θ̂ changed since last step (dynamic policy), rebuilds the shadow.
        Note: rebuilding discards the shadow's accumulated KB. A production
        implementation would replay the episode history under the new FOV;
        here we accept the approximation since MAP converges in ~20-30 steps.
        """
        theta = self._get_assumed_fov()
        if self.shadow is None or theta != self.shadow.fov:
            self.shadow = BayesHumanAgent(fov=theta)
            self.shadow.init_knowledge_base(state)

        # Advance shadow's KB by one step from the current agent position
        self.shadow.select_subtask(state)
        skb = self.shadow.knowledge_base

        target = self._next_needed(state, skb)
        if target is None:
            self.timer = 0
            return None

        self.timer += 1
        if self.timer >= self._effective_patience():
            self._reveal(human_kb, target, state)
            self.n_assists += 1
            self.timer = 0
            return target

        return None

    # ── What the human is missing ─────────────────────────────────────────────

    def _next_needed(self, state, kb: dict) -> Optional[Tuple]:
        """
        Return (kind, color, loc) for the single most critical missing object,
        judged against kb — the shadow's belief KB, never the real human's.
        Priority: door location > goal location > key location.
        Returns None when the shadow believes the human has all they need.
        """
        carrying = getattr(state, "carrying", None)
        held_type = getattr(carrying, "type", None) if carrying else None
        dead = kb.get("dead_door_colors", set())

        if held_type == "key":
            held_color = carrying.color

            if held_color not in dead:
                # Phase 2: need door location?
                # Guard: only reveal the matching locked door when the robot can
                # confirm the goal is accessible from the room on the other side.
                # Revealing a dead-end door (no goal behind it) accelerates the
                # wrong detour and may exhaust the step budget before the agent
                # can reach the correct room.
                if held_color not in kb.get("seen_doors", {}):
                    loc = self.solution["doors"].get(held_color)
                    if loc:
                        goal_loc = self.solution.get("goal")
                        room_zone = self._room_zone_past_door(state, loc)
                        if goal_loc is not None and goal_loc in room_zone:
                            return ("door", held_color, loc)

                # Phase 3: door is open, need goal?
                # Guard: only reveal the goal when it's actually reachable from
                # the agent's current position without crossing any locked door
                # (robot has full-grid vision). This prevents a premature reveal
                # when the agent opened the WRONG locked door first — the goal
                # may still require another locked door the agent can't open yet,
                # and goto_goal's BFS would get stuck on unknown walls/locked doors.
                door_open = held_color in kb.get("door_open_colors", set())
                if door_open and kb.get("goal_loc") is None:
                    loc = self.solution.get("goal")
                    if loc and loc in self._accessible_zone(state):
                        return ("goal", None, loc)

                # NEW HINT: message-only, no KB write (_reveal has no branch
                # for this kind). Robot's full-grid map already confirms this
                # room has no goal, even though the shadow hasn't explored
                # every cell yet (so it can't reach that conclusion itself
                # the way _decide()'s dead_door_colors logic eventually
                # would) — tell the human straight out it's a dead end so
                # they stop searching it, instead of waiting for them to
                # finish exploring it the slow way.
                room_cells = kb.get("goal_room_cells", {}).get(held_color, set())
                if (door_open and room_cells
                        and tuple(state.agent_pos) in room_cells
                        and not room_cells.issubset(kb.get("explored_cells", set()))
                        and self.solution.get("goal") not in room_cells):
                    return ("dead_room", held_color, tuple(state.agent_pos))

                return None  # shadow believes human is on track with this key

        # Phase 1: not holding a useful key — reveal the CLOSEST accessible
        # key by Manhattan distance, regardless of whether it's the one that
        # matches the goal-room door. May be a decoy that leads to a dead
        # end; the dead_room hint above is what recovers from that, not this
        # phase avoiding it.
        seen_keys = kb.get("seen_keys", {})
        live_seen = {c for c in seen_keys if c not in dead}
        if not live_seen:
            accessible = self._accessible_zone(state)
            ax, ay = state.agent_pos
            # MISHA NEW CHANGE (fix) — self.solution["keys"] is a one-time
            # snapshot taken at reset(); if a key has since been picked up
            # and dropped elsewhere (or is currently being carried), that
            # snapshot location is stale. Confirm the key is still actually
            # there on the real grid before proposing it, otherwise this
            # phantom-reveals forever once the original cell is empty.
            candidates = [
                (color, loc) for color, loc in self.solution["keys"].items()
                if color not in dead and loc in accessible
                and getattr(state.grid.get(*loc), "type", None) == "key"
            ]
            if candidates:
                color, loc = min(
                    candidates,
                    key=lambda c: abs(c[1][0] - ax) + abs(c[1][1] - ay),
                )
                return ("key", color, loc)

        # NEW HINT: message-only. Mirrors the dead_room hint above but for
        # key-hunting instead of goal-hunting — not holding any key at all,
        # standing inside a room (not the hallway) the robot's full-grid map
        # confirms has no key, but the shadow hasn't finished exploring it
        # yet. "Told" tracked per-room in told_empty_rooms (see
        # _write_reveal) instead of dead_door_colors, since there's no
        # color/key involved here to key that set off of.
        if held_type != "key":
            pos = tuple(state.agent_pos)
            if pos not in self._hallway_zone:
                room_cells = self._doorless_zone(state)
                has_key = any(loc in room_cells for loc in self.solution["keys"].values())
                if (not has_key
                        and not room_cells.issubset(kb.get("explored_cells", set()))
                        and frozenset(room_cells) not in kb.get("told_empty_rooms", set())):
                    return ("empty_room", None, pos)

        return None

    # ── Reachability (robot uses full grid) ──────────────────────────────────

    def _accessible_zone(self, state) -> Set[Tuple]:
        """
        BFS from agent through the FULL GRID, treating only LOCKED doors as
        barriers (unlocked/open doors are passable). Robot is omniscient so
        this reflects the true locked state of every door.

        Used for Phase 3 (goal reveal): ensures the goal is reachable via the
        currently-opened door chain before we tell the human where it is.
        If the goal requires another locked door the agent can't open yet,
        this check prevents a premature reveal that would send the planner
        into an unsolvable BFS detour.
        """
        ax, ay = state.agent_pos
        W, H   = state.width, state.height
        zone: Set[Tuple] = {(ax, ay)}
        q: deque = deque([(ax, ay)])

        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in zone:
                    continue
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                obj = state.grid.get(nx, ny)
                t   = getattr(obj, "type", None) if obj else None
                if t == "wall":
                    continue
                if t == "door" and getattr(obj, "is_locked", False):
                    continue  # locked door — stop expansion
                zone.add((nx, ny))
                q.append((nx, ny))

        return zone

    def _room_zone_past_door(self, state, door_loc: Tuple) -> Set[Tuple]:
        """
        Zone accessible on the ROOM SIDE of door_loc, using the full grid.
        Starts from the entry cell just inside the room (the neighbor of door_loc
        that is NOT in the agent's current accessible zone) and BFS-expands
        avoiding locked doors.

        Used for Phase 2: ensures we only reveal a locked door whose room actually
        leads to the goal (directly or via unchained unlocked paths). Dead-end
        rooms (no goal reachable without another locked key) are not revealed.
        """
        dx, dy = door_loc
        W, H   = state.width, state.height
        agent_side = self._accessible_zone(state)

        for vx, vy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = dx + vx, dy + vy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in agent_side:
                continue  # this neighbour is on the agent's current side
            obj = state.grid.get(nx, ny)
            t   = getattr(obj, "type", None) if obj else None
            if t in ("wall", "door"):
                continue  # not a passable room-interior cell
            # BFS from this room-side entry
            zone: Set[Tuple] = {(nx, ny)}
            q: deque = deque([(nx, ny)])
            while q:
                x, y = q.popleft()
                for ddx, ddy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    nnx, nny = x + ddx, y + ddy
                    if (nnx, nny) in zone:
                        continue
                    if not (0 <= nnx < W and 0 <= nny < H):
                        continue
                    obj2 = state.grid.get(nnx, nny)
                    t2   = getattr(obj2, "type", None) if obj2 else None
                    if t2 == "wall":
                        continue
                    if t2 == "door" and getattr(obj2, "is_locked", False):
                        continue
                    zone.add((nnx, nny))
                    q.append((nnx, nny))
            return zone

        return set()  # no room-side cell found (door is embedded in outer wall)

    def _doorless_zone(self, state) -> Set[Tuple]:
        """
        BFS from agent treating ALL doors as barriers.
        Returns cells reachable without entering any room through a door.
        Used to filter key reveal candidates: only surfaces keys the planner
        can navigate to without needing to toggle any door.
        """
        ax, ay = state.agent_pos
        W, H   = state.width, state.height
        zone: Set[Tuple] = {(ax, ay)}
        q: deque = deque([(ax, ay)])

        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in zone:
                    continue
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                obj = state.grid.get(nx, ny)
                t   = getattr(obj, "type", None) if obj else None
                if t in ("wall", "door"):  # ALL doors treated as barriers
                    continue
                zone.add((nx, ny))
                q.append((nx, ny))

        return zone

    # ── Knowledge injection ───────────────────────────────────────────────────

    def _reveal(self, human_kb: dict, target: Tuple, state) -> None:
        """
        Inject the target into the human's KB, and mirror the same write
        into the shadow's KB — once a hint is given, the robot's own belief
        should treat it as known too, not just the real human's. DynamicAssist
        overrides this to mirror into all 3 hypothesis agents' KBs instead of
        self.shadow, which it doesn't keep in sync (see dynamic_assist.py).
        """
        self._write_reveal(human_kb, target, state)
        if self.shadow is not None:
            self._write_reveal(self.shadow.knowledge_base, target, state)

    def _write_reveal(self, kb: dict, target: Tuple, state) -> None:
        """
        Apply one (kind, color, loc) reveal to a single KB dict.
        For doors: also populate grid_cache and locked_door_locs so the
        planner's BFS correctly recognises the door tile.

        MISHA NEW CHANGE — also stamps kb["seen_at"], the same forgetfulness
        timestamp BayesHumanAgent._update_kb uses for FOV sightings (see
        bayes_agent.py), so a hint decays on the same clock as something
        seen firsthand and can eventually need re-telling.
        """
        kind, color, loc = target
        seen_at = kb.setdefault("seen_at", {})
        current = state.step_count

        if kind == "key":
            kb.setdefault("seen_keys", {})[color] = loc
            seen_at[("key", color)] = current

        elif kind == "door":
            kb.setdefault("seen_doors", {})[color] = loc
            seen_at[("door", color)] = current
            obj = state.grid.get(*loc)
            if obj is not None:
                kb.setdefault("grid_cache", {})[loc] = obj
                if getattr(obj, "is_locked", False):
                    kb.setdefault("locked_door_locs", set()).add(loc)

        elif kind == "goal":
            kb["goal_loc"] = loc
            seen_at[("goal",)] = current

        elif kind == "dead_room":
            kb.setdefault("dead_door_colors", set()).add(color)
            seen_at[("dead", color)] = current

        elif kind == "empty_room":
            room_key = frozenset(self._doorless_zone(state))
            kb.setdefault("told_empty_rooms", set()).add(room_key)
            seen_at[("empty_room", room_key)] = current
