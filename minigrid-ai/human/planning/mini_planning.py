import random
from collections import deque
from typing import Optional, Tuple, Dict, Set
from minigrid.core.constants import COLOR_NAMES

# MiniGrid primitive actions
LEFT, RIGHT, FWD, PICKUP, DROP, TOGGLE, DONE = 0, 1, 2, 3, 4, 5, 6
DIR_TO_VEC = [(1,0), (0,1), (0,-1), (-1,0)]  # 0=right,1=down,3=up,2=left (unchanged behaviorally if you rely on indices)
# If you prefer the original order you had, swap back:
DIR_TO_VEC = [(1,0), (0,1), (-1,0), (0,-1)]  # 0=right,1=down,2=left,3=up
OPP_DIR   = [2, 3, 0, 1]


class MotionPlanner(object):
    
    """
    Stateless planner that returns the next primitive action given:
      - subtask (string)
      - state (MiniGrid env, read-only)
      - kb (your knowledge_base dictionary)
    You can add a small 'memory' dict if you want to keep frontier/visited.
    """

    def __init__(self):
        
        # memory of visited (position, orientation) for exploration
        self.visited_pos_or = set()

        # Added: position visits, recent positions, recent edges, and a tiny door cooldown
        self.visited_pos = set()
        self.recent = deque(maxlen=10)
        self.recent_edges = deque(maxlen=20)           # list of ((x1,y1),(x2,y2))
        self.door_cooldown = {}                        # (x,y) -> steps remaining

        # New: very long cooldown for doors we just exited through
        self.LONG_DOOR_COOLDOWN = 200

        self._forbidden_doors = set()  # NEW: {(x,y)} doors we must never enter

    """
    SUBTASK_LIST = [
    
    #1st phase (hallway)
    'find_key_room_door', #random explore
    'goto_key_room', #A* + open motion
    
    #2nd phase (inside key room)
    'find_key', #random explore
    'pickup_key', #A* + pick up motion
    
    #3rd phase (hallway) 
    'find_locked_room', #random explore 
    'goto_locked_room', #A* + use key to unlock motion
    
    #4th phase (inside locked room)
    #'unlock_door',
    'find_goal', #random explore
    'goto_goal', #A* + interact motion
    
    ]
    """

    # NEW: tiny helper to read a door's color as a string
    def _door_color_name(self, cell):
        c = getattr(cell, "color", None)
        if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
            return COLOR_NAMES[c]
        return c

    # NEW: recompute which doors are forbidden each step (colors not in mission)
    def _refresh_forbidden_doors(self, state, kb):
        allowed = {kb.get("keyRoom_color"), kb.get("lockedRoom_color")}
        forb = set()
        for x in range(state.width):
            for y in range(state.height):
                cell = state.grid.get(x, y)
                if cell is None:
                    continue
                if getattr(cell, "type", None) == "door":
                    if self._door_color_name(cell) not in allowed:
                        forb.add((x, y))
        self._forbidden_doors = forb

    def next_action(self, subtask, state, kb):
        """
        Depending on the subtask, call the correct function that would give you the correct action
        """
        self._refresh_forbidden_doors(state, kb)  # NEW

        if subtask == "find_key_room_door":
            return self._explore_step(state)
        
        if subtask == "goto_" + kb['key_color'] + "_room":
        #if subtask == "goto_key_room":
            # get key room door position
            door_xy = kb.get("keyRoom_loc")
            
            # if it is not door, explore more
            if not door_xy:
                return self._explore_step(state)
            
            # go to the key_room door and open
            return self._goto_and_face_then_toggle(state, door_xy)

        if subtask == "find_key":
            return self._explore_step(state)
        
        if subtask == "pickup_key":
            key_xy = kb.get("key_loc")
            if not key_xy:
                return self._explore_step(state)
            # go to key and pickup
            return self._goto_and_pickup(state, key_xy)

        if subtask == "find_locked_room":
            return self._explore_step(state)

        if subtask == "goto_locked_room":
        #if subtask == "goto_" + self.knowledge_base['lockedRoom_color'] + "_room":
            door_xy = kb.get("lockedRoom_loc")
            if not door_xy:
                return self._explore_step(state)
            # same as keyroom
            return self._goto_and_face_then_toggle(state, door_xy)

        if subtask == "find_goal":
            return self._explore_step(state)

        if subtask == "goto_goal":
            goal_xy = kb.get("goal_loc")
            if not goal_xy:
                return self._explore_step(state)
            # For goal you typically step ONTO the goal tile (no interact)
            return self._goto_cell(state, goal_xy)

        # default: keep exploring
        return self._explore_step(state)
     
    # helper for going to door and key
    def _goto_adjacent_and_face(self, state, target_xy):
        """
        Ensure we are adjacent to target_xy and oriented to face it.
        Returns:
          - a primitive action to keep moving/rotating, or
          - None if we are already facing the target (caller will interact/pickup).
        """
        ax, ay = state.agent_pos
        
        # If adjacent, rotate to face
        if self._manhattan((ax, ay), target_xy) == 1:
            desired_dir = self._dir_to_face((ax, ay), target_xy)
            if state.agent_dir != desired_dir:
                return self._rotate_towards(state.agent_dir, desired_dir)
            return None  # ready to interact
        
        # else move to any adjacent cell of target
        adj_cells = self._adjacent_cells(target_xy, state)
        
        # pick closest reachable adjacent cell
        best_path = None
        for cell in adj_cells:
            path = self._bfs_positions(state, start=(ax, ay), goal=cell)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
        if not best_path or len(best_path) < 2:
            return self._explore_step(state)
        next_xy = best_path[1]
        return self._turn_or_forward_towards(state, next_xy)
    
    # for going to and opening key room and locked room 
    def _goto_and_face_then_toggle(self, state, target_xy):
        """Move to a cell adjacent to target and face it; then TOGGLE (open/unlock)."""
        act = self._goto_adjacent_and_face(state, target_xy)
        if act is not None:
            return act
        return TOGGLE
    
    # going to and picking up key
    def _goto_and_pickup(self, state, target_xy):
        """Move to a cell adjacent to target and face it; then PICKUP."""
        act = self._goto_adjacent_and_face(state, target_xy)
        if act is not None:
            return act
        return PICKUP
    
    # for going to the goal   
    def _goto_cell(self, state, goal_xy):
        """Plan to step ONTO goal cell (no face requirement)."""
        ax, ay = state.agent_pos
        
        if (ax, ay) == tuple(goal_xy):
            # Already on goal—try DONE or small nudge; env will terminate on forward if goal ahead.
            return DONE
        
        # get shortest path of positions (including current, including goal)
        path = self._bfs_positions(state, start=(ax, ay), goal=tuple(goal_xy))
        if not path or len(path) < 2:
            # no path, explore
            return self._explore_step(state)
        
        # get the first returned list of positions from start to goal (inclusive)
        next_xy = path[1]
        return self._turn_or_forward_towards(state, next_xy)
    
    # ---------- Exploration (frontier-ish & simple) ----------
    
    # determine if the cell contains an open door
    def _is_open_door_cell(self, state, x, y):
        if (x, y) in self._forbidden_doors:  # NEW
            return False
        cell = state.grid.get(x, y)
        return (
            cell is not None
            and getattr(cell, "type", None) == "door"
            and bool(getattr(cell, "is_open", False))
        )

    # helpers for exploration scoring and memory
    def _in_bounds(self, state, x, y):
        return 0 <= x < state.width and 0 <= y < state.height

    def _neighbors4(self, x, y):
        for dx, dy in DIR_TO_VEC:
            yield x + dx, y + dy

    def _passable_or_door(self, state, x, y):
        if not self._in_bounds(state, x, y):
            return False
        if (x, y) in self._forbidden_doors:  # NEW
            return False
        cell = state.grid.get(x, y)
        if cell is None:
            return True
        t = getattr(cell, "type", None)
        if t == "door":
            return bool(getattr(cell, "is_open", False))
        return bool(cell.can_overlap())

    def _tick_cooldown(self):
        if not self.door_cooldown:
            return
        to_del = []
        for k, v in self.door_cooldown.items():
            nv = v - 1
            if nv <= 0:
                to_del.append(k)
            else:
                self.door_cooldown[k] = nv
        for k in to_del:
            del self.door_cooldown[k]
            
    def _explore_step(self, state):
        """
        Slightly smarter random explorer:
        - if an open door is in front, walk through it
        - else, if an open door is immediately left/right, turn toward it
        - epsilon-greedy random turn sometimes
        - prefer FWD if passable and the (next_pos, same_dir) isn't visited
        - otherwise turn toward an unseen orientation at the same tile
        """
        ax, ay = state.agent_pos
        ad = state.agent_dir
        key = (ax, ay, ad)
        self.visited_pos_or.add(key)

        # Track visits & short-term memory to avoid ping-pong
        self.visited_pos.add((ax, ay))
        self.recent.append((ax, ay))
        self._tick_cooldown()

        # --- Beeline priority: nearest in-bounds OPEN door that is NOT on cooldown ---
        best_path = None
        best_door = None
        for x in range(state.width):
            for y in range(state.height):
                if self._is_open_door_cell(state, x, y):
                    # respect cooldown: skip doors we recently used
                    if self.door_cooldown.get((x, y), 0) > 0:
                        continue
                    path = self._bfs_positions(state, start=(ax, ay), goal=(x, y))
                    if path and (best_path is None or len(path) < len(best_path)):
                        best_path = path
                        best_door = (x, y)

        if best_path and len(best_path) >= 2:
            # take one step along the door path
            next_xy = best_path[1]
            desired = self._dir_to_face((ax, ay), next_xy)
            if ad != desired:
                diff = (desired - ad) % 4
                return RIGHT if diff in (1, 2) else LEFT  # bias right on 180

            # aligned -> step; remember edge
            self.recent_edges.append(((ax, ay), next_xy))
            # If next step is the door tile itself, set long cooldown so we don't pop back in
            if next_xy == best_door:
                self.door_cooldown[next_xy] = max(self.door_cooldown.get(next_xy, 0), self.LONG_DOOR_COOLDOWN)
            return FWD

        # vectors & neighbor coords
        fx, fy = DIR_TO_VEC[ad]
        nx, ny = ax + fx, ay + fy

        # 0) if open door directly ahead -> go through it
        if self._in_bounds(state, nx, ny) and self._is_open_door_cell(state, nx, ny):
            # record edge and set a very long cooldown on that doorway tile (don't re-enter soon)
            self.recent_edges.append(((ax, ay), (nx, ny)))
            self.door_cooldown[(nx, ny)] = max(self.door_cooldown.get((nx, ny), 0), self.LONG_DOOR_COOLDOWN)
            return FWD

        # 0.25) Prefer forward if passable AND UNVISITED (don’t step onto seen tiles if we can avoid it)
        if self._passable_or_door(state, nx, ny) and (nx, ny) not in self.visited_pos:
            # avoid stepping onto a cooled doorway tile
            if not (self._is_open_door_cell(state, nx, ny) and self.door_cooldown.get((nx, ny), 0) > 0):
                self.recent_edges.append(((ax, ay), (nx, ny)))
                return FWD

        # 0.5) if open door is immediately left or right -> turn toward it
        left_dir  = (ad - 1) % 4
        right_dir = (ad + 1) % 4
        lx, ly = ax + DIR_TO_VEC[left_dir][0],  ay + DIR_TO_VEC[left_dir][1]
        rx, ry = ax + DIR_TO_VEC[right_dir][0], ay + DIR_TO_VEC[right_dir][1]

        if self._in_bounds(state, lx, ly) and self._is_open_door_cell(state, lx, ly):
            # don't turn toward a cooled doorway
            if self.door_cooldown.get((lx, ly), 0) == 0:
                return LEFT
        if self._in_bounds(state, rx, ry) and self._is_open_door_cell(state, rx, ry):
            if self.door_cooldown.get((rx, ry), 0) == 0:
                return RIGHT

        # Build candidate steps with frontier score, anti-reentry penalties
        candidates = []
        for dir_idx, (dx, dy) in enumerate(DIR_TO_VEC):
            tx, ty = ax + dx, ay + dy
            if not self._passable_or_door(state, tx, ty):
                continue

            # skip stepping onto door tiles that are on cooldown (prevents re-entry)
            if self._is_open_door_cell(state, tx, ty) and self.door_cooldown.get((tx, ty), 0) > 0:
                continue

            # penalties
            recent_pen = -3 if (tx, ty) in self.recent else 0
            if (ax, ay) != (tx, ty):
                rev_edge = ((tx, ty), (ax, ay))
            else:
                rev_edge = None
            edge_pen = -6 if rev_edge and rev_edge in self.recent_edges else 0

            # frontier reward (prefer tiles with many unseen neighbors)
            frontier = 0
            for qx, qy in self._neighbors4(tx, ty):
                if self._in_bounds(state, qx, qy) and (qx, qy) not in self.visited_pos:
                    frontier += 1

            jitter = random.random() * 0.1
            score = frontier + recent_pen + edge_pen + jitter
            candidates.append((score, dir_idx, (tx, ty)))

        if not candidates:
            # dead-end: rotate randomly
            return random.choice([LEFT, RIGHT])

        # ***** New: hard-prefer UNVISITED targets *****
        unvisited = [c for c in candidates if c[2] not in self.visited_pos]
        pool = unvisited if unvisited else candidates

        pool.sort(reverse=True)
        _score, best_dir, (tx, ty) = pool[0]

        # Turn toward best direction, then step; record edge only on forward
        if ad != best_dir:
            diff = (best_dir - ad) % 4
            return RIGHT if diff in (1, 2) else LEFT  # bias right on 180

        # aligned -> step forward; remember edge & apply long cooldown if stepping onto a door
        self.recent_edges.append(((ax, ay), (tx, ty)))
        if self._is_open_door_cell(state, tx, ty):
            self.door_cooldown[(tx, ty)] = max(self.door_cooldown.get((tx, ty), 0), self.LONG_DOOR_COOLDOWN)
        return FWD

    # ---------- Pathing ----------
    def _bfs_positions(self, state, start, goal):
        """
        BFS over positions (not orientations). Treats a cell as walkable iff:
          - empty or can_overlap(), OR
          - a DOOR that is_open or closed and then open it, OR
          - the goal cell itself (so we can path to goal).
        Returns a list of positions from start..goal (inclusive), or None.
        """
        W, H = state.width, state.height
        sx, sy = start
        gx, gy = goal
        if not (0 <= gx < W and 0 <= gy < H):
            return None

        q = deque([(sx, sy)])
        prev = { (sx, sy): None }

        def walkable(x, y):
            if not (0 <= x < W and 0 <= y < H): return False
            if (x, y) == (gx, gy): return True
            if (x, y) in self._forbidden_doors:  # NEW
                return False
            cell = state.grid.get(x, y)
            if cell is None: return True
            t = getattr(cell, "type", None)
            if t == "door":
                # allow routing through closed doors if allowed color; we'll toggle at execution time
                return True
            return bool(cell.can_overlap())

        while q:
            x, y = q.popleft()
            if (x, y) == (gx, gy):
                # rebuild path
                path = []
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return list(reversed(path))
            for dx, dy in DIR_TO_VEC:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in prev and walkable(nx, ny):
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
        return None

    # ---------- Low-level steering ----------
    def _turn_or_forward_towards(self, state, next_xy):
        ax, ay = state.agent_pos
        desired = self._dir_to_face((ax, ay), next_xy)
        if state.agent_dir != desired:
            return self._rotate_towards(state.agent_dir, desired)

        # aligned: check if next cell is a closed door -> open it
        nx, ny = next_xy
        cell = state.grid.get(nx, ny)
        if cell is not None and getattr(cell, "type", None) == "door":
            if (nx, ny) in self._forbidden_doors:  # NEW
                return LEFT  # or RIGHT; just don't interact with forbidden door
            if not bool(getattr(cell, "is_open", False)):
                return TOGGLE

        return FWD

    def _rotate_towards(self, cur_dir, desired_dir):
        # choose shorter turn (ties -> RIGHT)
        diff = (desired_dir - cur_dir) % 4
        if diff == 1: return RIGHT
        if diff == 3: return LEFT
        if diff == 2: return RIGHT  # 180: pick a side
        return DONE  # already aligned

    # ---------- Tiny utilities ----------
    def _dir_to_face(self, src_xy, dst_xy):
        (sx, sy), (dx, dy) = src_xy, dst_xy
        vx, vy = dx - sx, dy - sy
        if   (vx, vy) == (1, 0):  return 0  # right
        elif (vx, vy) == (0, 1):  return 1  # down
        elif (vx, vy) == (-1, 0): return 2  # left
        elif (vx, vy) == (0, -1): return 3  # up
        # not adjacent: pick any (we only call this when adj or stepping)
        return 0

    def _adjacent_cells(self, xy, state):
        x, y = xy
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        W, H = state.width, state.height
        return [(i,j) for (i,j) in cand if 0 <= i < W and 0 <= j < H and self._is_passable(state, i, j)]

    def _is_passable(self, state, x, y):
        if not (0 <= x < state.width and 0 <= y < state.height):
            return False
        if (x, y) in self._forbidden_doors:  # NEW
            return False
        cell = state.grid.get(x, y)
        if cell is None:
            return True
        t = getattr(cell, "type", None)
        if t == "door":
            return bool(getattr(cell, "is_open", False))
        return bool(cell.can_overlap())

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
