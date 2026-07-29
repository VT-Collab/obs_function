from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)


class LockedRoomEnv(MiniGridEnv):
    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
 
    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        random_walls: bool = False,
        **kwargs,
    ):
        self.size = size
        # MISHA NEW CHANGE — toggle: when True, each room gets one random
        # 2-cell wall segment dropped inside it for extra exploration
        # complexity. Off by default so existing callers are unaffected;
        # pass random_walls=True to gym.make(ENV_ID, ...) to turn it on,
        # from either the user-study interface or a simulation script.
        self.random_walls = random_walls

        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES] * 3,
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(lockedroom_color: str, keyroom_color: str, door_color: str):
        return (
            f"get the {lockedroom_color} key from the {keyroom_color} room,"
            f" unlock the {door_color} door and go to the goal"
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        self._lWallIdx, self._rWallIdx = lWallIdx, rWallIdx  # MISHA NEW CHANGE — reused by _hallway_bounds
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 3)))
            self.rooms.append(
                LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 3))
            )

        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        lockedRoom.locked = True
        goalPos = lockedRoom.rand_pos(self)
        self.grid.set(*goalPos, Goal())

        # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color
            if room.locked:
                self.grid.set(*room.doorPos, Door(color, is_locked=True))
            else:
                self.grid.set(*room.doorPos, Door(color))
                
        # --- Randomly Initialize doors so that some are open when starting each episode. 50% chance for each unlocked door ---
        opened_any = False
        open_candidates = [r for r in self.rooms if not r.locked]

        for r in open_candidates:
            # ~50% chance to start open; tweak as you like
            if self.np_random.random() < 0.3:
                d = self.grid.get(*r.doorPos)
                if isinstance(d, Door) and not d.is_locked:
                    d.is_open = True
                    opened_any = True

        # Guarantee at least one open door if RNG didn't open any
        if not opened_any and open_candidates:
            r = self._rand_elem(open_candidates)
            d = self.grid.get(*r.doorPos)
            if isinstance(d, Door):
                d.is_open = True
        # --- END Randomly Initialize doors so that some are open when starting each episode. 30% chance for each unlocked door ---
                
        # Map each door color to its room for quick lookup
        color_to_room = {room.color: room for room in self.rooms}

        # Select a random room to contain the REAL key to the goal (not the locked goal room itself)
        while True:
            keyRoom = self._rand_elem(self.rooms)
            if keyRoom != lockedRoom:
                break
        keyPos = keyRoom.rand_pos(self)
        self.grid.set(*keyPos, Key(lockedRoom.color))

        # --- PLACE DECOY KEYS so that using any decoy key always opens an EMPTY room. ---
        # Strategy: use disjoint pairs (door_room, key_room) from the remaining rooms.
        # Each decoy key color == door_room.color; the key is placed in key_room.
        # We exclude lockedRoom (goal) and keyRoom from pairing, guaranteeing:
        #   - no key ever sits inside its own door room,
        #   - no decoy door room contains any key,
        #   - decoy rooms unlock to empty interiors.
        decoys_to_place = 2
        # candidate rooms for pairing (must not be the goal room or the real-key room)
        remaining_rooms = [r for r in self.rooms if (r is not lockedRoom and r is not keyRoom)]
        # shuffle by drawing in random order using our RNG
        # (sample without replacement by repeatedly popping random indices)
        pairs = []
        while len(pairs) < decoys_to_place and len(remaining_rooms) >= 2:
            # choose a door_room from remaining
            door_idx = self._rand_int(0, len(remaining_rooms))
            door_room = remaining_rooms.pop(door_idx)
            # choose a key_room from remaining (disjoint from door_room)
            key_idx = self._rand_int(0, len(remaining_rooms))
            key_room = remaining_rooms.pop(key_idx)
            pairs.append((door_room, key_room))

        for door_room, key_room in pairs:
            decoy_color = door_room.color
            # lock the decoy door (overwrite existing door state if needed)
            self.grid.set(*door_room.doorPos, Door(decoy_color, is_locked=True))
            # place the decoy key in a DIFFERENT room (key_room), never inside door_room
            tries = 0
            while tries < 100:
                tries += 1
                px, py = key_room.rand_pos(self)
                if self.grid.get(px, py) is None and (px, py) != tuple(self.agent_pos):
                    self.grid.set(px, py, Key(decoy_color))
                    break
            # If placement fails after 100 tries, we silently skip; room remains without decoy key.
        # --- END DECOYS ---

        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )

        # Generate the mission string
        self.mission = (
            "get the %s key from the %s room, "
            "unlock the %s door and "
            "go to the goal"
        ) % (lockedRoom.color, keyRoom.color, lockedRoom.color)

        # MISHA NEW CHANGE — optional random 2-cell wall segments, tried
        # throughout every room AND the hallway. Runs last, after every
        # object (keys, doors, goal, agent) is placed, so the safety check
        # can verify the whole level stays navigable — including that a
        # wall doesn't combine with a key (which, like a door, you can only
        # approach, not walk through) to seal something off.
        if self.random_walls:
            self._place_random_walls()

    # MISHA NEW CHANGE — random 2-cell wall segments, rooms + hallway (see random_walls toggle)

    def _place_random_walls(self) -> None:
        """
        One attempt per room's interior, plus a few in the hallway corridor.
        Each attempt places a wall, checks _level_fully_navigable(), and
        reverts if it broke anything — so an unlucky placement never makes
        the level unsolvable or strands a key/door/room.
        """
        for room in self.rooms:
            topX, topY = room.top
            sizeX, sizeY = room.size
            self._try_wall_segment(topX + 1, topX + sizeX - 2, topY + 1, topY + sizeY - 2)

        hx0, hx1 = self._lWallIdx + 1, self._rWallIdx - 1
        hy0, hy1 = 1, self.height - 2
        for _ in range(3):
            self._try_wall_segment(hx0, hx1, hy0, hy1)

    def _try_wall_segment(self, x0: int, x1: int, y0: int, y1: int) -> None:
        """Try one random 2-cell wall segment within [x0,x1]x[y0,y1]; revert if unsafe."""
        can_h = x1 > x0  # need 2 distinct x's for a horizontal segment
        can_v = y1 > y0  # need 2 distinct y's for a vertical segment
        if not can_h and not can_v:
            return
        horizontal = self._rand_bool() if (can_h and can_v) else can_h

        if horizontal:
            x = self._rand_int(x0, x1)
            y = self._rand_int(y0, y1 + 1)
            cells = [(x, y), (x + 1, y)]
        else:
            x = self._rand_int(x0, x1 + 1)
            y = self._rand_int(y0, y1)
            cells = [(x, y), (x, y + 1)]

        # Never overwrite an existing object (door/key/goal) or the agent's cell.
        for cx, cy in cells:
            if (cx, cy) == tuple(self.agent_pos) or self.grid.get(cx, cy) is not None:
                return

        for cx, cy in cells:
            self.grid.set(cx, cy, Wall())

        if not self._level_fully_navigable():
            for cx, cy in cells:
                self.grid.set(cx, cy, None)  # revert — would have blocked something

    def _level_fully_navigable(self) -> bool:
        """
        Whole-level reachability check: every key and every door must have
        at least one reachable neighboring cell (you approach and interact
        with them, same as a real player — you can't walk through a key any
        more than you can walk through a closed door), and the goal itself
        must be directly reachable. Doors are treated as passable regardless
        of lock state here, since this checks the level's underlying
        navigability, not whether the human currently holds the right key.
        """
        def passable(x, y):
            obj = self.grid.get(x, y)
            if obj is None:
                return True
            if isinstance(obj, Door):
                return True
            return obj.can_overlap()

        start = tuple(self.agent_pos)
        seen = {start}
        stack = [start]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in seen or not passable(nx, ny):
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))

        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                t = getattr(obj, "type", None)
                if t in ("key", "door"):
                    neighbors = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                    if not any(n in seen for n in neighbors):
                        return False
                elif t == "goal":
                    if (x, y) not in seen:
                        return False

        return True
