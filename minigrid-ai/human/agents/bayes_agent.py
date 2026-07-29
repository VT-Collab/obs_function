"""
bayes_agent.py
Bayesian human agent for MiniGrid-LockedRoom — NO HINTS version.

From the skeleton assumption (3): "no hints". The agent does NOT read the
mission string. It explores, sees keys and doors, and figures out what to do
from accumulated observations alone.

Implements components (1) and (2) of the human generative model:

    θ       ~ P(θ)                              ← fov_prior() / pick_fov()
    τ_t     ~ P_τ(τ_t | τ_{t-1}, s_{1:t}, θ)   ← BayesHumanAgent.subtask_transition_probs()
    a^H_t   ~ π^H(a^H_t | s_{1:t}, θ, τ_t)     ← planning/bayes_planner.py

KB (no hints):
  seen_keys:       {color_str → (x, y)}   most recent grid location of each seen key
  seen_doors:      {color_str → (x, y)}   most recent grid location of each seen door
  goal_loc:        (x, y) or None
  explored_cells:  set of (x, y) ever seen through the FOV cone
  dead_door_colors: set of door-color strings confirmed to have no goal behind them

Human pipeline (from skeleton):
  1. Explore → see key → pick it up.
  2. Find locked door whose color matches held key → unlock it.
  3. Explore inside for goal → go to goal.
  4. If room fully explored and no goal → leave_room → mark dead → find another key.

NOTE on MiniGrid env: No changes required. Uses existing
  estimated_world_cone_vis_mask(fov) (tagged "new misha edit" in minigrid_env.py).
"""

from __future__ import annotations
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from minigrid.core.constants import COLOR_NAMES

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from human.planning.bayes_planner import BayesianPlanner

LEFT, RIGHT, FWD, PICKUP, DROP, TOGGLE, DONE = 0, 1, 2, 3, 4, 5, 6
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 0=right, 1=down, 2=left, 3=up


# ── Subtask list ──────────────────────────────────────────────────────────────
# Fixed strings, no color-specific names (no hints).
# leave_room is triggered once the current room is confirmed goalless.

SUBTASK_LIST = [
    'find_key',           # explore until any key is spotted
    'pickup_key',         # navigate to and pick up the spotted key
    'find_locked_door',   # explore to find a locked door matching held key color
    'goto_locked_door',   # navigate to matching door and unlock it
    'find_goal',          # explore inside the unlocked room for the goal
    'leave_room',         # room fully explored, no goal — exit back through door
    'goto_goal',          # navigate onto the goal tile
]

_N = len(SUBTASK_LIST)

# ── Subtask transition hyperparameters ────────────────────────────────────────
_P_STAY    = 0.88   # self-loop mass when decision tree says stay
_P_NEXT    = 0.85   # mass on target subtask when decision tree says transition
_P_PERSIST = 0.08   # small persistence on previous subtask (humans are slow)
_TAU_FLOOR = 0.005  # minimum per-subtask probability (no zero posteriors)

# MISHA NEW CHANGE — forgetfulness: a fact (seen through FOV, or told via a
# robot hint) is dropped from the KB once this many steps have passed since
# it was last confirmed. Applies uniformly to the real human's KB and every
# shadow/hypothesis KB, since they're all built by this same class.
FORGET_HORIZON = 20


def _color_str(c) -> Optional[str]:
    if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
        return COLOR_NAMES[c]
    if isinstance(c, str):
        return c
    return None


# ── Stand-alone prior functions ───────────────────────────────────────────────

def fov_prior(fov_list: List[int]) -> Dict[int, float]:
    """P(θ) — uniform prior over candidate FOVs."""
    p = 1.0 / len(fov_list)
    return {f: p for f in fov_list}

def pick_fov(fov_list: List[int]) -> int:
    """Sample θ ~ P(θ). True FOV is drawn here in the generative model."""
    return random.choice(fov_list)

def initial_subtask_prior() -> Dict[str, float]:
    """
    P(τ_1) — distribution over subtasks at episode start.
    Agent always starts in hallway with empty KB → always find_key.
    All non-floor mass goes to find_key; every other subtask gets just the floor
    so the observer's posterior never permanently zeroes them out.
    """
    probs = {t: _TAU_FLOOR for t in SUBTASK_LIST}
    probs["find_key"] += 1.0 - _TAU_FLOOR * _N  # all remaining mass
    return probs

def pick_first_subtask() -> str:
    """Sample τ_1 ~ P(τ_1). Almost always 'find_key'."""
    #Dict[str, float] — e.g. {'find_key': 0.97, 'pickup_key': 0.005, ...}
    prior    = initial_subtask_prior()      
        
    #['find_key', 'pickup_key', ...] — ordered list of subtask names
    subtasks = list(prior.keys())   
                 
    #[0.97, 0.005, ...] — matching probability for each subtask
    weights  = [prior[t] for t in subtasks]  
        
    #weighted random draw → almost always 'find_key'
    return random.choices(subtasks, weights=weights)[0] 


# ── Main agent class ──────────────────────────────────────────────────────────

class BayesHumanAgent:
    """
    Functional Bayesian human agent — NO HINTS.

    Never reads the mission string. Accumulates observations through its
    FOV-limited KB and follows a hint-free decision tree.

    Exploration exits:
      Once the agent enters a locked room, finds no goal, and has seen all
      reachable cells in that room, _decide() returns 'leave_room'.
      The door's color is added to dead_door_colors so it is never re-entered.

    Same class can serve as a candidate hypothesis in inference (different fov).
    """

    def __init__(self, fov: int):
        self.fov = fov
        self.knowledge_base: Dict = {}
        self.current_subtask: str = pick_first_subtask()  # τ_1 ~ P(τ_1)

    # ── KB management ─────────────────────────────────────────────────────────

    def init_knowledge_base(self, state=None) -> None:
        """
        Start with an empty KB — no mission string parsed (no hints).
        state accepted for API compatibility but not used.
        """
        self.knowledge_base = {
            "seen_keys":          {},    # color_str → (x, y)
            "seen_doors":         {},    # color_str → (x, y)
            "door_open_colors":   set(), # door colors seen-as-open through FOV
            "goal_loc":           None,
            "explored_cells":     set(), # all (x,y) ever seen through FOV (for _room_fully_explored)
            "visited_cells":      set(), # cells agent has physically stood on (for exploration BFS)
            "dead_door_colors":   set(), # door colors confirmed no goal behind them
            "pending_leave_color": None, # held key color mid-exit (not yet finalized)
            "pending_room_cells":  set(), # room snapshot used to detect when agent exits
            "goal_room_cells":    {},    # color → frozenset: cached room cells computed from
                                         # hallway side (correct even when agent is inside room)
            "grid_cache":         {},    # (x,y) → obj: ground-truth obj stored ONLY for seen cells
            "locked_door_locs":   set(), # (x,y) of doors seen-as-locked through FOV
            # MISHA NEW CHANGE — forgetfulness: fact_key -> step last confirmed
            # (seen through FOV, or told via a robot hint). See _forget_stale.
            "seen_at":            {},
        }
        self.current_subtask = pick_first_subtask()  # τ_1 ~ P(τ_1)

    def _update_kb(self, state) -> None:
        """
        Scan all cells visible under self.fov. Update seen objects and
        add every visible cell to explored_cells.

        Keys that were recorded in seen_keys but are now visibly absent
        from their location (picked up) are removed.

        Uses estimated_world_cone_vis_mask(fov) — safe on hypothesis instances.
        """
        kb = self.knowledge_base
        mask = state.estimated_world_cone_vis_mask(self.fov)

        # MISHA NEW CHANGE — drop anything not re-confirmed within
        # FORGET_HORIZON steps before re-scanning, so something still in
        # view right now gets immediately re-stamped rather than flickering.
        self._forget_stale(state)

        current = state.step_count

        # Track physical presence for exploration BFS (separate from FOV-based explored_cells)
        kb["visited_cells"].add(tuple(state.agent_pos))

        for x in range(state.width):
            for y in range(state.height):
                if not mask[x, y]:
                    continue

                # All visible cells count as explored
                kb["explored_cells"].add((x, y))

                obj      = state.grid.get(x, y)
                obj_type = getattr(obj, "type", None) if obj else None
                color    = _color_str(getattr(obj, "color", None)) if obj else None

                # Cache every visible cell's object so the planner never reads
                # state.grid for unseen cells.
                kb["grid_cache"][(x, y)] = obj

                if obj_type == "key":
                    kb["seen_keys"][color] = (x, y)
                    kb["seen_at"][("key", color)] = current  # MISHA NEW CHANGE
                elif obj_type == "door":
                    kb["seen_doors"][color] = (x, y)
                    kb["seen_at"][("door", color)] = current  # MISHA NEW CHANGE
                    if getattr(obj, "is_open", False):
                        kb["door_open_colors"].add(color)
                    if getattr(obj, "is_locked", False):
                        kb["locked_door_locs"].add((x, y))
                    else:
                        kb["locked_door_locs"].discard((x, y))
                elif obj_type == "goal":
                    kb["goal_loc"] = (x, y)
                    kb["seen_at"][("goal",)] = current  # MISHA NEW CHANGE

        # Remove keys whose location is now visible but empty (key was picked up)
        for kc in list(kb["seen_keys"].keys()):
            kx, ky = kb["seen_keys"][kc]
            if mask[kx, ky]:
                obj_there = state.grid.get(kx, ky)
                if obj_there is None or getattr(obj_there, "type", None) != "key":
                    del kb["seen_keys"][kc]
                    kb["seen_at"].pop(("key", kc), None)  # MISHA NEW CHANGE

    # ── Forgetfulness ────────────────────────────────────────────────────────

    def _forget_stale(self, state) -> None:
        """
        MISHA NEW CHANGE — drop any fact (seen through FOV, or told via a
        robot hint) that hasn't been reconfirmed within FORGET_HORIZON steps.
        Models imperfect human memory, and lets a hint become relevant again
        (e.g. a previously-revealed key location, or a room marked dead) once
        it's old enough to plausibly have been forgotten.

        Deliberately does NOT touch explored_cells/visited_cells — raw
        cell-level exploration memory isn't covered by this pass, only the
        higher-level facts (key/door/goal locations, dead-room conclusions).
        """
        kb = self.knowledge_base
        current = state.step_count
        seen_at = kb["seen_at"]

        for fact in list(seen_at.keys()):
            if current - seen_at[fact] <= FORGET_HORIZON:
                continue
            del seen_at[fact]
            kind = fact[0]

            if kind == "key":
                kb["seen_keys"].pop(fact[1], None)

            elif kind == "door":
                color = fact[1]
                loc = kb["seen_doors"].pop(color, None)
                kb["door_open_colors"].discard(color)
                if loc is not None:
                    kb["grid_cache"].pop(loc, None)
                    kb["locked_door_locs"].discard(loc)

            elif kind == "goal":
                kb["goal_loc"] = None

            elif kind == "dead":
                kb["dead_door_colors"].discard(fact[1])

            elif kind == "empty_room":
                kb.get("told_empty_rooms", set()).discard(fact[1])

    # ── Room exploration check ─────────────────────────────────────────────────
    
    def _reachable_no_doors_from(self, state, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        BFS from `start` through walkable cells, stopping at all door tiles.
        Gives the connected region on start's side of any door boundary.
        
        BFS from a starting cell, expanding to neighboring cells 
        that are walkable — but stops at any door tile 
        (doesn't cross doors, doesn't add them to the visited set)
        Returns the set of all cells reachable from start without crossing any door
    
        So if the agent is in the hallway, it gets all hallway cells.
        If the agent is inside a room, it gets all cells in that room. 
        The door tile itself acts as a wall boundary.
        
        """
        sx, sy = start
        visited: Set[Tuple[int, int]] = {(sx, sy)}
        q: deque = deque([(sx, sy)])

        while q:
            x, y = q.popleft()
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < state.width and 0 <= ny < state.height):
                    continue
                cell = state.grid.get(nx, ny)
                t = getattr(cell, "type", None) if cell else None
                if t == "door":
                    continue
                if cell is None or cell.can_overlap():
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return visited

    def _room_cells_past_door(self, state, door_loc: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        First, assumes the agent is in the hallway, not a room
        First BFS from the agent position to get all hallways cells
        
        Then figures out for the given door location (from seen_doors):
        The door tile has 4 neighbors (up, down, left, right).

        One of those neighbors is the hallway. One is the room. The other two are walls.

        The code loops through all 4 neighbors and asks: "is this neighbor on the room side?"
        and if it is, pass that cell to the reachable_no_doors_from() BFS to get the full room interior cell.
        
        AKA, 
        Return cells on the ROOM SIDE of door_loc (the side the agent is NOT on).

        BFS from the hallway (agent side) gives hallway cells. Any walkable
        neighbor of door_loc that is NOT in the hallway cells is the room interior.
        Returns all cells reachable from that interior neighbor without crossing doors.

        Returns empty set if no room side cell exists (door embedded in wall).
        """
        ax, ay = state.agent_pos
        dx, dy = door_loc
        W, H = state.width, state.height

        hallway_cells = self._reachable_no_doors_from(state, (ax, ay))

        for vx, vy in DIR_TO_VEC:
            nx, ny = dx + vx, dy + vy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in hallway_cells:
                continue  # this is the hallway side
            cell = state.grid.get(nx, ny)
            t = getattr(cell, "type", None) if cell else None
            if t in ("wall", "door"):
                continue
            if cell is None or cell.can_overlap():
                #from the tile that is inside room
                return self._reachable_no_doors_from(state, (nx, ny))

        return set()

    def _room_fully_explored(self, state, door_loc: Tuple[int, int], kb,
                              held_color: Optional[str] = None) -> bool:
        """
        True iff every cell in the room behind door_loc has been seen through FOV.

        Uses the cached goal_room_cells[held_color] when available (computed from
        the hallway side, so it stays correct even when the agent is inside the room).
        Falls back to _room_cells_past_door only if no cache exists yet.
        """
        if held_color and held_color in kb.get("goal_room_cells", {}):
            room_cells = kb["goal_room_cells"][held_color]
        else:
            room_cells = self._room_cells_past_door(state, door_loc)
        if not room_cells:
            return False
        #if all the room cells are in explored_cells, meaning you have explored all the room cells
        return room_cells.issubset(kb["explored_cells"])

    # ── Deterministic decision tree ───────────────────────────────────────────

    def _decide(self, state) -> str:
        """
        Hint-free subtask selection from current KB.

        Priority:
          1. Goal visible → goto_goal
          2. Mid-exit (pending_leave_color set): still inside → leave_room;
             crossed door → finalize dead_door_colors, fall through
          3. Holding key, room confirmed dead → find_key (look for different key)
          4. Holding key, matching door locked → goto_locked_door
          5. Holding key, door open, room fully explored → leave_room (arm pending)
          6. Holding key, door open, room not fully explored → find_goal
          7. Holding key, no matching door seen → find_locked_door
          8. Not holding key, key location known → pickup_key
          9. Not holding key, no key seen → find_key

        leave_room uses a two-phase mechanism:
          Phase 1 (arm): _room_fully_explored fires → set pending_leave_color +
            pending_room_cells (snapshot of current room), return leave_room.
          Phase 2 (persist): while agent_pos is still inside pending_room_cells
            keep returning leave_room.
          Finalize: once agent_pos steps outside pending_room_cells (crossed the
            door tile), add color to dead_door_colors and clear pending state.
        This prevents the one-step collapse where dead_door_colors is set
        immediately and the next _decide call returns find_key prematurely.
        """
        kb = self.knowledge_base

        if kb["goal_loc"] is not None:
            return "goto_goal"

        # ── Two-phase leave_room persistence ─────────────────────────────────
        pending_color = kb["pending_leave_color"]
        if pending_color is not None:
            ax, ay = state.agent_pos
            if (ax, ay) in kb["pending_room_cells"]:
                # Agent still inside the snapshot room — keep driving toward door
                return "leave_room"
            else:
                # Agent has crossed the door tile — finalize
                kb["dead_door_colors"].add(pending_color)
                kb["seen_at"][("dead", pending_color)] = state.step_count  # MISHA NEW CHANGE
                kb["pending_leave_color"] = None
                kb["pending_room_cells"]  = set()
                # Fall through to normal decision logic below

        carrying    = getattr(state, "carrying", None)
        held_type   = getattr(carrying, "type", None) if carrying else None
        holding_key = (held_type == "key")

        if holding_key:
            held_color = _color_str(getattr(carrying, "color", None))

            # Dead room: confirmed no goal behind this key's door
            if held_color in kb["dead_door_colors"]:
                return "find_key"

            door_loc = kb["seen_doors"].get(held_color)

            if door_loc:
                # Use KB-tracked door state (populated in _update_kb from FOV only).
                # Never read state.grid for a door that may be outside current FOV.
                door_is_open = held_color in kb["door_open_colors"]
                if not door_is_open:
                    return "goto_locked_door"

                # Door is open.
                # Cache room cells the first time we see this door open.
                # _room_cells_past_door is only correct from the hallway side;
                # once the agent is inside, it returns hallway cells instead.
                # Caching on the first call (agent just toggled → still in hallway)
                # gives us the correct room cells for all subsequent checks.
                if held_color not in kb["goal_room_cells"]:
                    fresh = self._room_cells_past_door(state, door_loc)
                    if fresh:
                        kb["goal_room_cells"][held_color] = fresh

                room_cells = kb["goal_room_cells"].get(held_color, set())
                ax_now, ay_now = state.agent_pos
                agent_inside = (ax_now, ay_now) in room_cells

                if agent_inside:
                    if self._room_fully_explored(state, door_loc, kb, held_color):
                        kb["pending_leave_color"] = held_color
                        kb["pending_room_cells"]  = room_cells
                        return "leave_room"
                    return "find_goal"
                else:
                    # Agent is in the hallway — check if FOV already covered the room.
                    if room_cells and self._room_fully_explored(state, door_loc, kb, held_color):
                        kb["dead_door_colors"].add(held_color)
                        kb["seen_at"][("dead", held_color)] = state.step_count  # MISHA NEW CHANGE
                        return "find_key"
                    return "find_goal"

            return "find_locked_door"

        # Not holding a key — pick up any known key whose door isn't dead
        live_keys = {c: loc for c, loc in kb["seen_keys"].items()
                     if c not in kb["dead_door_colors"]}
        if live_keys:
            return "pickup_key"
        return "find_key"

    # ── Subtask transition distribution ───────────────────────────────────────

    def subtask_transition_probs(self, tau_prev: str, state) -> Dict[str, float]:
        """
        P_τ(τ_t | τ_{t-1}=tau_prev, s_{1:t}, θ=self.fov)

        Updates KB under self.fov, gets deterministic target from _decide(),
        then builds a soft distribution:

          target == tau_prev (stay): P(tau_prev) = _P_STAY, residual uniform
          target != tau_prev (go):   P(target) = _P_NEXT, P(tau_prev) = _P_PERSIST,
                                     residual uniform over others

        find_goal → leave_room transition captures the "room exhausted" case:
        high P(leave_room | find_goal) once _room_fully_explored is True.
        This is FOV-dependent: a narrower FOV takes longer to explore all cells,
        so the transition fires later for small-FOV agents.

        All subtasks guaranteed ≥ _TAU_FLOOR (no zero posteriors).
        """
        self._update_kb(state)
        target = self._decide(state)

        probs  = {t: _TAU_FLOOR for t in SUBTASK_LIST}
        budget = 1.0 - _TAU_FLOOR * _N

        if target == tau_prev:
            probs[target] += budget * _P_STAY
            others = [t for t in SUBTASK_LIST if t != target]
            per = budget * (1.0 - _P_STAY) / len(others)
            for t in others:
                probs[t] += per
        else:
            probs[target]   += budget * _P_NEXT
            probs[tau_prev] += budget * _P_PERSIST
            others = [t for t in SUBTASK_LIST if t not in (target, tau_prev)]
            per = budget * (1.0 - _P_NEXT - _P_PERSIST) / max(len(others), 1)
            for t in others:
                probs[t] += per

        total = sum(probs.values())
        return {t: p / total for t, p in probs.items()}

    # ── Subtask selection (MAP for running the agent) ─────────────────────────

    def select_subtask(self, state) -> str:
        """Return MAP subtask (argmax of transition dist). Updates self.current_subtask."""
        self._update_kb(state)
        self.current_subtask = self._decide(state)
        return self.current_subtask

    def step(self, state) -> str:
        """Convenience wrapper. Pair with BayesianPlanner.next_action()."""
        return self.select_subtask(state)

