"""
bayes_planner.py
Implements the human action likelihood for Bayesian FOV inference — NO HINTS version.

Handles component (3) of the human generative model:
    π^H(a^H_t | s_{1:t}, θ, τ_t) — action probability distribution given subtask + FOV

Subtask names match bayes_agent.py SUBTASK_LIST (no-hint, no color-specific strings):
    find_key, pickup_key, find_locked_door, goto_locked_door, find_goal, goto_goal

  action_probs(subtask, state, kb)
    Fully simulated: calls next_action (real BFS from hypothesis KB) to get the
    deterministic predicted action, then wraps it in epsilon-greedy noise.
    kb must be the hypothesis agent's KB — reconstructed under hypothesis FOV θ.

  next_action(subtask, state, kb)
    Stateful: uses BFS exploration and navigation helpers.
    Use this to actually run the agent through an episode.

NOTE on MiniGrid env: No changes needed.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Dict, Optional, Tuple

from minigrid.core.constants import COLOR_NAMES

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

LEFT, RIGHT, FWD, PICKUP, DROP, TOGGLE, DONE = 0, 1, 2, 3, 4, 5, 6
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 0=right, 1=down, 2=left, 3=up for computing tiles

# ── Action probability hyperparameters ──────────────────────────────────────
_EPSILON    = 0.10   # noise: mass spread uniformly over non-optimal actions
_N_ACTIONS  = 7      # LEFT RIGHT FWD PICKUP DROP TOGGLE DONE


def _color_str(c) -> Optional[str]:
    if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
        return COLOR_NAMES[c]
    if isinstance(c, str):
        return c
    return None


class BayesianPlanner:
    """
    Computes π^H(a^H_t | s_{1:t}, θ, τ_t) — the human action likelihood.

    action_probs / sample_action: fully simulated per hypothesis KB.
      Calls next_action (real BFS from hypothesis KB) to get the deterministic
      predicted action, then wraps in epsilon-greedy noise.
    next_action: stateful BFS nav + exploration, for running the agent.

    KB must be the hypothesis agent's KB (reconstructed under hypothesis FOV θ).
    """

    def __init__(self):
        pass

    # ── Public API ──────────────────────────────────────────────────────────

    def action_probs(self, subtask: str, state, kb: Dict) -> Dict[int, float]:
        """
        π^H(a^H_t | s_{1:t}, θ, τ_t=subtask)
        Predicted action from next_action on hypothesis KB + epsilon noise.
        """
        opt = self.next_action(subtask, state, kb)
        return self._epsilon_dist(opt)

    def sample_action(self, subtask: str, state, kb: Dict) -> int:
        """Sample from π^H for stochastic simulation."""
        probs = self.action_probs(subtask, state, kb)
        actions, weights = zip(*sorted(probs.items()))
        return random.choices(list(actions), weights=list(weights))[0]

    def next_action(self, subtask: str, state, kb: Dict) -> int:
        """MAP action for running the agent. All navigation uses KB + BFS only."""
        if subtask == "find_key":
            # If holding a dead-color key, drop it into the dead room so it
            # doesn't land in the hallway and block BFS pathfinding
            # (key.can_overlap()=False breaks _bfs_passable navigation).
            carrying = getattr(state, "carrying", None)
            if carrying and getattr(carrying, "type", None) == "key":
                held_color = _color_str(getattr(carrying, "color", None))
                if held_color in kb.get("dead_door_colors", set()):
                    ax, ay = state.agent_pos
                    room_cells = kb.get("goal_room_cells", {}).get(held_color, set())
                    # Best case: adjacent to dead room — face it and drop inside.
                    for vx, vy in DIR_TO_VEC:
                        nx, ny = ax + vx, ay + vy
                        if (nx, ny) in room_cells:
                            desired = self._dir_to_face((ax, ay), (nx, ny))
                            if state.agent_dir != desired:
                                return self._rotate_towards(state.agent_dir, desired)
                            return DROP
                    # Not adjacent — navigate to dead room door tile first.
                    # Once there, the next call will find room_cells adjacent and drop.
                    dead_door_loc = kb.get("seen_doors", {}).get(held_color)
                    if dead_door_loc and room_cells:
                        path = self._bfs(kb, state, (ax, ay), dead_door_loc)
                        if path and len(path) >= 2:
                            return self._turn_or_fwd(state, path[1], kb)
                    return DROP  # absolute last resort
            return self._explore_all(state, kb)

        if subtask == "find_locked_door":
            return self._explore_for_locked_door(state, kb)

        if subtask == "find_goal":
            door_loc = self._open_door_for_held_key(state, kb)
            carrying = getattr(state, "carrying", None)
            held_color = _color_str(getattr(carrying, "color", None)) if carrying else None
            room_cells = kb.get("goal_room_cells", {}).get(held_color, set()) if held_color else set()
            ax, ay = state.agent_pos
            # Agent not yet inside the room — navigate straight to the door.
            # Avoids _explore_room wandering to unvisited hallway cells first.
            if room_cells and (ax, ay) not in room_cells:
                if door_loc:
                    if (ax, ay) == door_loc:
                        return FWD  # on door tile — step through into room
                    return self._bfs_step_onto(kb, state, door_loc)
                return FWD
            # Agent is inside the room — explore it (room_cells restricts targets
            # to room interior so BFS doesn't escape back through the open door).
            action = self._explore_room(state, kb, open_door_loc=door_loc,
                                        room_cells=room_cells if room_cells else None)
            if action is not None:
                return action
            return self._bfs_step_onto(kb, state, door_loc) if door_loc else FWD

        if subtask == "pickup_key":
            target = self._any_key_loc(kb)
            return self._bfs_nav(kb, state, target, PICKUP) if target else self._explore_all(state, kb)

        if subtask == "goto_locked_door":
            target = self._matching_door_loc(state, kb)
            return self._bfs_nav(kb, state, target, TOGGLE) if target else self._explore_for_locked_door(state, kb)

        if subtask == "leave_room":
            target = self._open_door_for_held_key(state, kb)
            if target:
                ax, ay = state.agent_pos
                if (ax, ay) == target:
                    return FWD
                return self._bfs_step_onto(kb, state, target)
            carrying = getattr(state, "carrying", None)
            held_color = _color_str(getattr(carrying, "color", None)) if carrying else None
            door_loc = kb.get("seen_doors", {}).get(held_color) if held_color else None
            return self._bfs_step_onto(kb, state, door_loc) if door_loc else FWD

        if subtask == "goto_goal":
            target = kb.get("goal_loc")
            return self._bfs_step_onto(kb, state, target) if target else FWD

        return FWD

    # ── Action distribution builder ─────────────────────────────────────────

    def _epsilon_dist(self, opt: int) -> Dict[int, float]:
        """
        Epsilon-greedy noise around the deterministic predicted action.
        P(opt) = 1 - _EPSILON, remaining mass split uniformly over all other actions.
        """
        other = _EPSILON / (_N_ACTIONS - 1)
        probs = {a: other for a in range(_N_ACTIONS)}
        probs[opt] = 1.0 - _EPSILON
        return probs

    # ── KB-aware target helpers ─────────────────────────────────────────────

    def _any_key_loc(self, kb: Dict) -> Optional[Tuple]:
        """Return any known live (non-dead) key location from KB."""
        dead = kb.get("dead_door_colors", set())
        seen = kb.get("seen_keys", {})
        for color, loc in seen.items():
            if color not in dead:
                return loc
        return None

    def _matching_door_loc(self, state, kb: Dict) -> Optional[Tuple]:
        """Return location of door whose color matches the key being held."""
        carrying = getattr(state, "carrying", None)
        if carrying is None or getattr(carrying, "type", None) != "key":
            return None
        held_color = _color_str(getattr(carrying, "color", None))
        return kb.get("seen_doors", {}).get(held_color)

    def _open_door_for_held_key(self, state, kb: Dict) -> Optional[Tuple]:
        """
        Return the location of the open door matching the held key's color.
        Used by leave_room to navigate back to the exit.
        Returns None if not holding a key or door not known/not open.
        """
        carrying = getattr(state, "carrying", None)
        if carrying is None or getattr(carrying, "type", None) != "key":
            return None
        held_color = _color_str(getattr(carrying, "color", None))
        loc = kb.get("seen_doors", {}).get(held_color)
        if loc is None:
            return None
        if held_color in kb.get("door_open_colors", set()):
            return loc
        return None

    # ── Exploration helpers ─────────────────────────────────────────────────

    def _explore_all(self, state, kb: Dict) -> int:
        """
        Navigate toward the nearest unvisited cell reachable through any
        passable or unlocked-door tile. Used for find_key / find_locked_door.

        Two-phase approach:
          Phase 1: BFS using explored_cells (FOV-based). Fast — covers large
            areas in few steps because FOV scans many cells per position.
          Phase 2: When Phase 1 finds no target (all FOV-scanned cells exhausted),
            fall back to visited_cells (physical presence). This handles LOS blind
            spots where a locked door is adjacent to a cell the agent has seen from
            far away but never physically stood next to — physically visiting that
            cell may reveal the door.

        Dead-room doors are permanently excluded from the BFS.
        """
        ax, ay = state.agent_pos
        dead_colors = kb.get("dead_door_colors", set())
        dead_door_locs = {loc for color, loc in kb.get("seen_doors", {}).items()
                          if color in dead_colors}
        dead_room_cells: set = set()
        for color in dead_colors:
            dead_room_cells.update(kb.get("goal_room_cells", {}).get(color, set()))
        W, H = state.width, state.height
        explored = kb.get("explored_cells", set())
        locked_locs = kb.get("locked_door_locs", set())
        cache = kb.get("grid_cache", {})

        def _bfs_for_unvisited(unseen_set, add_frontier: bool):
            """
            BFS toward nearest cell not in unseen_set.
            Only expands through cells we have seen (in explored_cells) and know
            are passable — no reads from state.grid for unseen cells.
            If add_frontier=True (Phase 1), adjacent unseen cells are added as
            targets; if False (Phase 2), only already-explored cells are targeted.
            """
            queue: deque = deque([(ax, ay)])
            prev: Dict = {(ax, ay): None}
            found = None

            while queue:
                x, y = queue.popleft()

                # Target check: cell not in the "seen" set and not a dead-room cell.
                if (x, y) not in unseen_set and (x, y) not in dead_room_cells:
                    found = (x, y)
                    break

                # Don't expand from unseen frontier cells — we have no info beyond them.
                if (x, y) not in explored:
                    continue

                for dx, dy in DIR_TO_VEC:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in prev:
                        continue
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    if (nx, ny) in dead_door_locs or (nx, ny) in dead_room_cells:
                        continue

                    if (nx, ny) not in explored:
                        # Unseen frontier cell — add as potential target (Phase 1 only).
                        if add_frontier:
                            prev[(nx, ny)] = (x, y)
                            queue.append((nx, ny))
                        continue

                    # Seen cell — use KB cache for passability; never read state.grid.
                    obj = cache.get((nx, ny))
                    t = getattr(obj, "type", None) if obj else None
                    if t == "wall":
                        continue
                    if t == "door":
                        if (nx, ny) in locked_locs:
                            continue  # seen-locked door — treat as barrier
                        # seen open/unlocked door — passable
                    # passable
                    prev[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

            if found is None:
                return None
            path = []
            cur: Optional[Tuple] = found
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path[1] if len(path) >= 2 else None

        # Phase 1: navigate toward any unseen frontier cell.
        next_step = _bfs_for_unvisited(explored, add_frontier=True)
        if next_step is not None:
            return self._turn_or_fwd(state, next_step, kb)

        # Phase 2: all frontier cells exhausted — physically visit explored cells
        # not yet stood on (catches LOS blind spots).
        next_step = _bfs_for_unvisited(kb.get("visited_cells", set()), add_frontier=False)
        if next_step is not None:
            return self._turn_or_fwd(state, next_step, kb)

        return FWD

    def _explore_room(self, state, kb: Dict,
                      open_door_loc: Optional[Tuple] = None,
                      room_cells: Optional[set] = None) -> Optional[int]:
        """
        Navigate toward the nearest physically unvisited cell reachable by crossing
        only the specific open goal door (open_door_loc). Stops at all other door
        tiles so the agent stays within the goal-room / hallway cluster.

        room_cells: when provided, only target cells inside the room (or unseen cells,
        which are likely room interior). Prevents the BFS from escaping through the
        open door tile and targeting unvisited hallway cells.

        Returns None when all reachable cells are already visited (caller should
        navigate through the door directly or fall back to explore_step).
        """
        ax, ay = state.agent_pos
        visited = kb.get("visited_cells", set())
        explored = kb.get("explored_cells", set())
        cache = kb.get("grid_cache", {})
        W, H = state.width, state.height

        queue: deque = deque([(ax, ay)])
        prev: Dict = {(ax, ay): None}
        target = None

        while queue:
            x, y = queue.popleft()
            if (x, y) not in visited:
                # If room_cells given: only target cells in the room or unseen cells
                # (unseen = likely room interior or wall to discover).
                # Skip seen cells outside the room (hallway) as targets.
                if room_cells is None or (x, y) not in explored or (x, y) in room_cells:
                    target = (x, y)
                    break
                # Seen non-room cell — not a valid target; fall through to expand
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if (nx, ny) in prev:
                    continue
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if (nx, ny) not in explored:
                    # Unseen adjacent cell — certainly unvisited, add as target.
                    prev[(nx, ny)] = (x, y)
                    queue.append((nx, ny))
                    continue
                obj = cache.get((nx, ny))
                t = getattr(obj, "type", None) if obj else None
                if t == "door":
                    # Allow crossing only the specific known-open goal door.
                    if open_door_loc and (nx, ny) == open_door_loc:
                        prev[(nx, ny)] = (x, y)
                        queue.append((nx, ny))
                    continue
                if obj is None or obj.can_overlap():
                    prev[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

        if target is None:
            # All cells reachable without crossing a door are already visited.
            # Return None so the caller knows to navigate through the open door.
            return None

        # Trace back to find the first step toward target
        path = []
        cur: Optional[Tuple] = target
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if len(path) < 2:
            return FWD
        return self._turn_or_fwd(state, path[1], kb)

    def _explore_for_locked_door(self, state, kb: Dict) -> int:
        """
        Search for the locked door without entering other rooms.

        Two cases:
          - Agent in a room (just picked up a key): exactly 1 door tile borders
            the zone → navigate to it and step through to the hallway.
          - Agent in hallway: BFS toward unvisited hallway cells only
            (stops at ALL door tiles, so the agent never enters a room).

        All locked doors are visible from the hallway, so exploring the hallway
        is sufficient to find the matching locked door.
        """
        ax, ay = state.agent_pos
        W, H = state.width, state.height
        explored = kb.get("explored_cells", set())
        cache = kb.get("grid_cache", {})

        # Standing on a door tile already — step through to the other side.
        obj_here = cache.get((ax, ay))
        if getattr(obj_here, "type", None) == "door":
            return FWD

        # Compute agent's zone using only KB-cached cells (no state.grid reads).
        # Only expand through cells we've seen and know are passable.
        zone: set = {(ax, ay)}
        bq: deque = deque([(ax, ay)])
        while bq:
            x, y = bq.popleft()
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if (nx, ny) in zone or not (0 <= nx < W and 0 <= ny < H):
                    continue
                if (nx, ny) not in explored:
                    continue  # unseen — don't expand through unknown cells
                obj = cache.get((nx, ny))
                t = getattr(obj, "type", None) if obj else None
                if t in ("wall", "door"):
                    continue
                if obj is None or obj.can_overlap():
                    zone.add((nx, ny))
                    bq.append((nx, ny))

        # Count seen door tiles adjacent to this zone.
        adj_doors: set = set()
        for x, y in zone:
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if (nx, ny) not in explored:
                    continue  # only count doors we've actually seen
                obj = cache.get((nx, ny))
                if getattr(obj, "type", None) == "door":
                    adj_doors.add((nx, ny))

        # Exactly 1 adjacent seen door → agent is inside a room → exit immediately.
        if len(adj_doors) == 1:
            door_loc = next(iter(adj_doors))
            if (ax, ay) == door_loc:
                return FWD
            return self._bfs_step_onto(kb, state, door_loc)

        # In hallway — BFS toward unvisited zone cells and unexplored frontier cells.
        # Zone is KB-only (explored cells), so zone_bfs is naturally honest.
        def zone_bfs(unseen: set, add_frontier: bool) -> Optional[Tuple]:
            q: deque = deque([(ax, ay)])
            prev: Dict = {(ax, ay): None}
            found = None
            while q:
                x, y = q.popleft()
                if (x, y) not in unseen:
                    found = (x, y)
                    break
                if (x, y) not in zone:
                    continue  # frontier cell — don't expand beyond it
                for dx, dy in DIR_TO_VEC:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in prev:
                        continue
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    if (nx, ny) in zone:
                        prev[(nx, ny)] = (x, y)
                        q.append((nx, ny))
                    elif add_frontier and (nx, ny) not in explored:
                        # Unseen cell adjacent to explored zone — frontier target.
                        prev[(nx, ny)] = (x, y)
                        q.append((nx, ny))
            if found is None:
                return None
            path: list = []
            cur: Optional[Tuple] = found
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path[1] if len(path) >= 2 else None

        # Phase 1: navigate toward unseen frontier cells adjacent to explored zone.
        step = zone_bfs(explored, add_frontier=True)
        if step:
            return self._turn_or_fwd(state, step, kb)

        # Phase 2: physically visit explored-but-unvisited zone cells (LOS blind spots).
        step = zone_bfs(kb.get("visited_cells", set()), add_frontier=False)
        if step:
            return self._turn_or_fwd(state, step, kb)

        return FWD

    # ── Navigation helpers (stateless BFS) ─────────────────────────────────

    def _bfs_nav(self, kb: Dict, state, target_xy: Tuple, interact_action: int) -> int:
        """Navigate adjacent to target_xy, face it, then interact."""
        if target_xy is None:
            return FWD
        ax, ay = state.agent_pos
        tx, ty = target_xy

        if abs(ax - tx) + abs(ay - ty) == 1:
            desired = self._dir_to_face((ax, ay), (tx, ty))
            if state.agent_dir != desired:
                return self._rotate_towards(state.agent_dir, desired)
            return interact_action

        adj = [(tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]
        best: Optional[list] = None
        for cell in adj:
            gx, gy = cell
            if not (0 <= gx < state.width and 0 <= gy < state.height):
                continue
            # Skip if KB says this approach cell is a wall.
            obj = kb.get("grid_cache", {}).get((gx, gy))
            if getattr(obj, "type", None) == "wall":
                continue
            path = self._bfs(kb, state, (ax, ay), cell)
            if path and (best is None or len(path) < len(best)):
                best = path

        if not best or len(best) < 2:
            return FWD
        return self._turn_or_fwd(state, best[1], kb)

    def _bfs_step_onto(self, kb: Dict, state, target_xy: Tuple) -> int:
        """BFS to step ONTO the target cell (goal tile)."""
        if target_xy is None:
            return FWD
        ax, ay = state.agent_pos
        tx, ty = target_xy
        if (ax, ay) == (tx, ty):
            return DONE
        path = self._bfs(kb, state, (ax, ay), (tx, ty))
        if not path or len(path) < 2:
            return FWD
        return self._turn_or_fwd(state, path[1], kb)

    def _bfs(self, kb: Dict, state, start: Tuple, goal: Tuple) -> Optional[list]:
        """Stateless BFS. Routes through closed doors (execution will toggle them).
        Uses KB cache for passability — unseen cells treated as passable."""
        W, H = state.width, state.height
        sx, sy = start
        gx, gy = goal
        if not (0 <= gx < W and 0 <= gy < H):
            return None

        q: deque = deque([(sx, sy)])
        prev: Dict = {(sx, sy): None}

        while q:
            x, y = q.popleft()
            if (x, y) == (gx, gy):
                path = []
                cur: Optional[Tuple] = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return list(reversed(path))
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in prev and self._bfs_passable(kb, nx, ny, W, H, (gx, gy)):
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
        return None

    def _bfs_passable(self, kb: Dict, x: int, y: int, W: int, H: int, goal: Tuple) -> bool:
        if not (0 <= x < W and 0 <= y < H):
            return False
        if (x, y) == goal:
            return True
        cache = kb.get("grid_cache", {})
        if (x, y) not in cache:
            return True  # unseen → assume passable for navigation routing
        obj = cache[(x, y)]
        if obj is None:
            return True
        t = getattr(obj, "type", None)
        if t == "wall":
            return False
        if t == "door":
            # Known locked doors are impassable mid-path; the goal-check above
            # still lets BFS reach an adjacent cell whose *neighbor* is the locked
            # door, so _turn_or_fwd can issue TOGGLE from there.
            if (x, y) in kb.get("locked_door_locs", set()):
                return False
            return True  # unlocked/open door — _turn_or_fwd will toggle if closed
        return bool(obj.can_overlap())

    def _is_walkable(self, kb: Dict, state, x: int, y: int) -> bool:
        if not (0 <= x < state.width and 0 <= y < state.height):
            return False
        cache = kb.get("grid_cache", {})
        if (x, y) not in cache:
            return True  # unseen → assume walkable
        obj = cache[(x, y)]
        if obj is None:
            return True
        t = getattr(obj, "type", None)
        if t == "wall":
            return False
        if t == "door":
            color = _color_str(getattr(obj, "color", None))
            return color in kb.get("door_open_colors", set())
        return bool(obj.can_overlap())

    # ── Low-level steering ──────────────────────────────────────────────────

    def _turn_or_fwd(self, state, next_xy: Tuple, kb: Dict = None) -> int:
        ax, ay = state.agent_pos
        desired = self._dir_to_face((ax, ay), next_xy)
        if state.agent_dir != desired:
            return self._rotate_towards(state.agent_dir, desired)
        nx, ny = next_xy
        if kb is not None:
            obj = kb.get("grid_cache", {}).get((nx, ny))
            t = getattr(obj, "type", None) if obj else None
            if t == "door":
                color = _color_str(getattr(obj, "color", None))
                if color not in kb.get("door_open_colors", set()):
                    return TOGGLE
        return FWD

    def _rotate_towards(self, cur: int, desired: int) -> int:
        diff = (desired - cur) % 4
        if diff == 1: return RIGHT
        if diff == 3: return LEFT
        return RIGHT  # 180°: bias right

    def _dir_to_face(self, src: Tuple, dst: Tuple) -> int:
        vx, vy = dst[0] - src[0], dst[1] - src[1]
        if (vx, vy) == (1,  0): return 0
        if (vx, vy) == (0,  1): return 1
        if (vx, vy) == (-1, 0): return 2
        if (vx, vy) == (0, -1): return 3
        return 0
