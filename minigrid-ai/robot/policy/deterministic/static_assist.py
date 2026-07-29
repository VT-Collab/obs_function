"""
static_assist.py — Baseline: assistance assuming a fixed, known human FOV.

The robot permanently assumes the human's FOV is `assumed_fov` degrees.
This mirrors the common HRI baseline of a "fixed human model" — the robot
never updates its belief about the human's perceptual capability.

In practice this over-assists narrow-FOV humans (shadow with FOV=120 sees
more than a real FOV=60 human, so it often already "knows" the target and
the patience counter never fires) and under-assists wide-FOV humans (the
reverse). It sets the performance floor that the adaptive policy must beat.

Baseline design follows Nikolaidis et al. (2017) "Human-Robot Mutual
Adaptation in Collaborative Tasks" (IJRR) where fixed-model agents are
the standard comparison condition against adaptive policies.
"""

from __future__ import annotations
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from robot.policy.deterministic._base import AssistBase


class StaticAssist(AssistBase):
    """
    Assistance policy with a fixed assumed FOV.

    Parameters
    ----------
    assumed_fov : int
        The FOV (degrees) the robot always assumes the human has.
        Default 120 — the median candidate (between narrow 60 and wide 180).
    patience : int
        Steps to wait before revealing a missing object (K in the paper).
    """

    def __init__(self, assumed_fov: int = 120, patience: int = 5):
        super().__init__(patience=patience)
        self._assumed_fov = assumed_fov

    def _get_assumed_fov(self) -> int:
        return self._assumed_fov
    
    

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

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, state, human_kb: dict) -> Optional[Tuple]:
        """
        Call BEFORE the human acts each step.
        Optionally injects info into human_kb. Returns revealed target or None.

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

        target = self._next_needed(state, human_kb)
        if target is None:
            self.timer = 0
            return None

        if self._shadow_knows(skb, target):
            self.timer = 0
            return None

        self.timer += 1
        if self.timer >= self.patience:
            self._reveal(human_kb, target, state)
            self.n_assists += 1
            self.timer = 0
            return target

        return None

    # ── What the human is missing ─────────────────────────────────────────────

    def _next_needed(self, state, human_kb: dict) -> Optional[Tuple]:
        """
        Return (kind, color, loc) for the single most critical missing object.
        Priority: door location > goal location > key location.
        Returns None when the human already has all they need.
        """
        carrying = getattr(state, "carrying", None)
        held_type = getattr(carrying, "type", None) if carrying else None
        dead = human_kb.get("dead_door_colors", set())

        if held_type == "key":
            held_color = carrying.color

            if held_color not in dead:
                # Phase 2: need door location?
                # Guard: only reveal the matching locked door when the robot can
                # confirm the goal is accessible from the room on the other side.
                # Revealing a dead-end door (no goal behind it) accelerates the
                # wrong detour and may exhaust the step budget before the agent
                # can reach the correct room.
                if held_color not in human_kb.get("seen_doors", {}):
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
                door_open = held_color in human_kb.get("door_open_colors", set())
                if door_open and human_kb.get("goal_loc") is None:
                    loc = self.solution.get("goal")
                    if loc and loc in self._accessible_zone(state):
                        return ("goal", None, loc)

                return None  # human is on track with this key

        # Phase 1: not holding a useful key — identify and reveal the CORRECT key
        # (the key whose color matches the locked door that leads to the goal room).
        # We skip keys that are inaccessible (behind another locked door) since
        # those can't be reached. Keys in rooms behind unlocked doors are safe to
        # reveal now that _bfs_passable treats locked doors as barriers: the
        # pickup_key BFS will route through unlocked doors only.
        seen_keys = human_kb.get("seen_keys", {})
        live_seen = {c for c in seen_keys if c not in dead}
        if not live_seen:
            goal_loc = self.solution.get("goal")
            accessible = self._accessible_zone(state)
            if goal_loc:
                for door_color, door_loc in self.solution["doors"].items():
                    if door_color in dead:
                        continue
                    obj = state.grid.get(*door_loc)
                    if not getattr(obj, "is_locked", False):
                        continue  # not locked — not the goal-room door
                    room_zone = self._room_zone_past_door(state, door_loc)
                    if goal_loc not in room_zone:
                        continue  # goal not behind this door
                    # Found the goal door — reveal its key if accessible
                    key_loc = self.solution["keys"].get(door_color)
                    if key_loc and key_loc in accessible:
                        return ("key", door_color, key_loc)
                    break  # goal door found but key unreachable — stop

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

    # ── Shadow knowledge check ────────────────────────────────────────────────

    def _shadow_knows(self, skb: dict, target: Tuple) -> bool:
        """True if shadow KB already has this object — human will find it soon."""
        kind, color, _ = target
        if kind == "key":
            return color in skb.get("seen_keys", {})
        elif kind == "door":
            return color in skb.get("seen_doors", {})
        elif kind == "goal":
            return skb.get("goal_loc") is not None
        return False

    # ── Knowledge injection ───────────────────────────────────────────────────

    def _reveal(self, human_kb: dict, target: Tuple, state) -> None:
        """
        Inject the target's location into the human's KB.
        For doors: also populate grid_cache and locked_door_locs so the
        planner's BFS correctly recognises the door tile.
        """
        kind, color, loc = target
        if kind == "key":
            human_kb.setdefault("seen_keys", {})[color] = loc

        elif kind == "door":
            human_kb.setdefault("seen_doors", {})[color] = loc
            obj = state.grid.get(*loc)
            if obj is not None:
                human_kb.setdefault("grid_cache", {})[loc] = obj
                if getattr(obj, "is_locked", False):
                    human_kb.setdefault("locked_door_locs", set()).add(loc)

        elif kind == "goal":
            human_kb["goal_loc"] = loc

