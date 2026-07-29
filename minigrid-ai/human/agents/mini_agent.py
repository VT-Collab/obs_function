
"""
Not a runnable standaline; to be called/used in another file. 

Implement Minigrid human models, with input including
    1. the type of minigrid envrionment (lockedroom)
    2. fov (integer)
    
take in the string hints at the very beginning and as soon as see the color door goes there

take in a string
select subtask from go to key door, pick up key, go to goal room, drop key into goal

STILL NEEDS WORK = not implemented yet

Next things:
Relax:
second thing: we make the human don't know what it is looking for
Breaking it into 2 pieces

collect human that has hints
collect human that has keys in different rooms
collect human that has no hints

then train the lstm on all of those to figure that out 

"""


from __future__ import annotations
import re
from minigrid.core.constants import COLOR_NAMES
from typing import Optional, Tuple, Dict, Set
from collections import deque

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

LEFT, RIGHT, FWD, PICKUP, DROP, TOGGLE, DONE = 0, 1, 2, 3, 4, 5, 6
DIR_TO_VEC = [(1,0), (0,1), (-1,0), (0,-1)]  # 0=right,1=down,2=left,3=up
OPP_DIR   = [2, 3, 0, 1]


#human subtask or intent list 
"""
Note on random exploring:
    Intead of complete random like before it was action a = env.action_space.sample() 
    use graph and dont revisit orientation and grid position that we were in 
    DFS probs graph search (in motion planner) to do smarter way of exploring 
    or random sampleing potential neighbors 
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


# Accepts the standard mission string and tiny formatting differences
_LOCKEDROOM_RE = re.compile(
    r"""
    ^\s*get\s+the\s+(?P<key_color>\w+)\s+key\s+from\s+the\s+(?P<keyRoom_color>\w+)\s+room\s*,?\s*
    unlock\s+the\s+(?P<lockedRoom_color>\w+)\s+door\s+and\s+go\s+to\s+the\s+goal\s*$
    """,
    re.IGNORECASE | re.VERBOSE
)

#returns a dictionary that contains keys key_color, keyRoom_color, and lockedRoom_color with values of their respective colors
def parse_lockedroom_colors(mission: str):
    
    #check if if the mission string is in the standard format
    m = _LOCKEDROOM_RE.match(mission)
    if not m:
        raise ValueError(f"Unrecognized mission: {mission!r}")
    #the colors
    colors = {k: m.group(k).lower() for k in ("key_color", "keyRoom_color", "lockedRoom_color")}
    
    bad = [c for c in colors.values() if c not in COLOR_NAMES]
    if bad:
        raise ValueError(f"Unknown color(s): {bad}. Allowed: {', '.join(COLOR_NAMES)}")
    return colors



class limitVisionHumanModel():
    def __init__(self, fov):
                
        self.fov = fov
        #instantaneos kb update no delay        
        #for kb, as soon as it sees something in the environment, you can keep it there until it updates when u see it again
        self.knowledge_base = {}
        
        self.estimated_fov = 0
        self.estimated_knowledge_base = {}
    
    
    # === NEW: update the estimated FOV each timestep ===
    def set_estimated_fov(self, new_fov: int):
        """
        Setter function called each timestep to update the agent’s estimated field of view.
        """
        self.estimated_fov = new_fov

         
    #initialize knowledge_base, fill in the key_room color and locked_room color, and dont know any location yet
    def init_knowledge_base(self, start_state):
        self.knowledge_base = {}
        
        colors = parse_lockedroom_colors(start_state.mission)
        
        self.knowledge_base['key_color']         = colors['key_color']          # color of the key (== locked room/door color)
        self.knowledge_base['keyRoom_color']     = colors['keyRoom_color']      # color of the room that contains the key
        self.knowledge_base['lockedRoom_color']  = colors['lockedRoom_color']   # color of the locked door/room
        
        #keycolor and keyroom_color should be the same by default, but we can change it later

        self.knowledge_base["lockedRoom_loc"] = None
        self.knowledge_base["key_loc"] = None
        self.knowledge_base["keyRoom_loc"] = None
        self.knowledge_base["goal_loc"] = None
        
        
        #estimated section
        self.estimated_knowledge_base = {}
        self.estimated_knowledge_base['key_color']         = colors['key_color']          # color of the key (== locked room/door color)
        self.estimated_knowledge_base['keyRoom_color']     = colors['keyRoom_color']      # color of the room that contains the key
        self.estimated_knowledge_base['lockedRoom_color']  = colors['lockedRoom_color']   # color of the locked door/room
        
        self.estimated_knowledge_base["lockedRoom_loc"] = None
        self.estimated_knowledge_base["key_loc"] = None
        self.estimated_knowledge_base["keyRoom_loc"] = None
        self.estimated_knowledge_base["goal_loc"] = None
        
        
    

    #Determins if a locaiton is within the agent's filed of view
    #return True if yes, False if no
    #TODO: STILL NEEDS WORK!!!!! currently can see every single tile; so currently it should see everything, aka go straight to first door, then key, then locked door, then goal
    #TODO: currently recompute the mask many times per step
    def in_bound(self, state, loc, fov):
        """
        True iff world cell `loc=(x,y)` is within the agent's cone FOV and not occluded.
        Uses the env's _world_cone_vis_mask (angle + line of sight which together constitutes the cone).
        """
        x, y = loc
        
        #if outside of world dimension
        if not (0 <= x < state.width and 0 <= y < state.height):
            return False

        # # (Optional) let caller override FOV per check
        # if fov is not None and getattr(state, "agent_fov", None) != fov:
        #     state.agent_fov = fov

        mask = state._world_cone_vis_mask()  # shape (W, H), True = visible
        return bool(mask[x, y])
    
    
    def estimated_in_bound(self, state, loc, fov):
        """
        True iff world cell `loc=(x,y)` is within the estimated fov
        """
        x, y = loc
        
        #if outside of world dimension
        if not (0 <= x < state.width and 0 <= y < state.height):
            return False

        # # (Optional) let caller override FOV per check
        # if fov is not None and getattr(state, "agent_fov", None) != fov:
        #     state.agent_fov = fov

        mask = state.estimated_world_cone_vis_mask(self.estimated_fov)  # shape (W, H), True = visible
        return bool(mask[x, y])
    
    
    #if visible, update knowledge_base
    #DONE: scan everything in the grid, if visible, and is door/key, and the door color matches hint, update kb
    def update_knowledge_base(self, state):
        
        new_knowledge_base = self.knowledge_base.copy()
        
        
        
        #scan everything in the grid
        width, height = state.width, state.height
        for x in range(width):
            for y in range(height):
                if not self.in_bound(state, (x,y), self.fov):
                    continue
                
                #get the coordinate location's object
                obj = state.grid.get(x,y)
                
                if obj is None:
                    continue
                
                #get object type and color
                obj_type = getattr(obj, "type", None)   # e.g., 'door', 'key', 'goal', ...
                obj_color_idx = getattr(obj, "color", None) # e.g., 'red', 'green', ...
                
                obj_color = None
                if isinstance(obj_color_idx, int) and 0 <= obj_color_idx < len(COLOR_NAMES):
                    obj_color = COLOR_NAMES[obj_color_idx]
                elif isinstance(obj_color_idx, str):
                    obj_color = obj_color_idx  # MiniGrid often stores color as a string
                
                # doors: remember positions by color
                #in Python, my_dict.get("key_name") returns the value for that key if it exists; otherwise it returns None by default (or a default you provide).
                if obj_type == "door":
                    #if color matches
                    if obj_color == new_knowledge_base.get("keyRoom_color"):
                        new_knowledge_base["keyRoom_loc"] = (x, y)
                    if obj_color == new_knowledge_base.get("lockedRoom_color"):
                        new_knowledge_base["lockedRoom_loc"] = (x, y)

                # key: remember its position by color
                elif obj_type == "key":
                    if obj_color == new_knowledge_base.get("key_color"):
                        new_knowledge_base["key_loc"] = (x, y)
                        
                elif obj_type == "goal":
                    new_knowledge_base["goal_loc"] = (x, y)
                    
        
                
        self.knowledge_base = new_knowledge_base
        return new_knowledge_base

    # === NEW: estimated KB update using self.estimated_fov ===
    def update_estimated_knowledge_base(self, state):
        """
        Update the estimated KB exactly like update_knowledge_base(), but
        using self.estimated_fov instead of the ground truth fov.
        Called each timestep after self.set_estimated_fov() updates the bound.
        """
        new_estimated_kb = self.estimated_knowledge_base.copy()

        width, height = state.width, state.height
        for x in range(width):
            for y in range(height):
                if not self.estimated_in_bound(state, (x, y), self.estimated_fov):
                    continue

                obj = state.grid.get(x, y)
                if obj is None:
                    continue

                obj_type = getattr(obj, "type", None)
                obj_color_idx = getattr(obj, "color", None)

                obj_color = None
                if isinstance(obj_color_idx, int) and 0 <= obj_color_idx < len(COLOR_NAMES):
                    obj_color = COLOR_NAMES[obj_color_idx]
                elif isinstance(obj_color_idx, str):
                    obj_color = obj_color_idx

                # mirror exact same logic as real KB
                if obj_type == "door":
                    if obj_color == new_estimated_kb.get("keyRoom_color"):
                        new_estimated_kb["keyRoom_loc"] = (x, y)
                    if obj_color == new_estimated_kb.get("lockedRoom_color"):
                        new_estimated_kb["lockedRoom_loc"] = (x, y)
                elif obj_type == "key":
                    if obj_color == new_estimated_kb.get("key_color"):
                        new_estimated_kb["key_loc"] = (x, y)
                elif obj_type == "goal":
                    new_estimated_kb["goal_loc"] = (x, y)

        self.estimated_knowledge_base = new_estimated_kb
        return new_estimated_kb

    
    
    ##--- helpers for choosing the right subtask ---
    
    #returns the type of the object right in front of us
    def _front_obj(self, state):
        # MiniGrid: dir 0=right, 1=down, 2=left, 3=up
        DIR_TO_VEC = [(1,0), (0,1), (-1,0), (0,-1)]
        ax, ay = state.agent_pos
        dx, dy = DIR_TO_VEC[state.agent_dir]
        x, y = ax + dx, ay + dy
        if 0 <= x < state.width and 0 <= y < state.height:
            return state.grid.get(x, y)
        return None
    
    #returns true if the given object is the door/color that we want
    def _is_door_of_color(self, obj, color_name: str):
        if getattr(obj, "type", None) != "door":
            return False
        c = getattr(obj, "color", None)
        
        # support both int color index and str color
        if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
            return COLOR_NAMES[c] == color_name
        return c == color_name
    
    
    #main subtask selection policy
    #selects a subtask and passes that subtask along with necessary info such as human agent's current location and facing direction to motion planner to get a concrete action 
    #TODO: add hard code logic of choosing one subtask based on current state:
    #TODO: DOUBLE CHECK THIS LOGIC
    """
    ok so like i want to code ml_action now, of like choosing a subtask correctly based on current state. 
    I am thinking: so first, if havnt seen the key door aka havnt seen that door and that color, then subtask would be find_key_room_door. 
    Then if we have seen that door, but havnt seen key yet, then we need to choose subtask goto_key_room. 
    Then if we are right in front of the key room like facing it, aka the door in front is the keyroom door and it is open, and we havn't found key yet, subtask is find_key. 
    then if key is found, but we are not holding the key, subtask needs to be pickup_key. 
    then if we are holding the key, and havnt found locked room yet aka the locked room location is None, then subtask needs to be find_locked_room. 
    otherwise if we havn't found goal yet but have found the locked room door and holding the key, subtask needs to be goto_locked_room. 
    then if we havn't found goal yet, but we are right in fron tof locked room door and it is open, then we ned to select subtask find_goal. 
    then if we do have goal location, then subtask needs to be goto_goal.
    
    okay so it seems without partial observability it won't work. also seems like the sequence of subtasks will be purely sequential like 
    the way that i set it up, one subtask follows another, and one subtask CANNOT COME before another subtask
    """
    def ml_action(self, state):
        """
        Choose one subtask string from SUBTASK_LIST based on current knowledge & agent context.
        Assumes `state` is the MiniGrid env (has agent_pos/dir, grid, carrying, etc.)
        """
        # keep KB fresh (no-ops if you've already called it elsewhere)
        self.update_knowledge_base(state)
        kb = self.knowledge_base

        #boolean indicators of our current knowledge
        key_room_seen     = kb.get("keyRoom_loc")     is not None
        key_seen          = kb.get("key_loc")         is not None
        locked_room_seen  = kb.get("lockedRoom_loc")  is not None
        goal_seen         = kb.get("goal_loc")        is not None

        # holding key?
        carrying = getattr(state, "carrying", None)
        holding_key = False
        
        #if carrying key, is the key the right color of the given key_color?
        if carrying is not None and getattr(carrying, "type", None) == "key":
            c = getattr(carrying, "color", None)
            if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
                holding_key = COLOR_NAMES[c] == kb.get("key_color")
            else:
                holding_key = (c == kb.get("key_color"))

        # what's in front?
        front = self._front_obj(state)
        
        #is front an open door
        front_is_open_key_door = (
            front is not None
            and self._is_door_of_color(front, kb.get("keyRoom_color"))
            and bool(getattr(front, "is_open", False))
        )
        
        #is whats in front an opened lock door
        front_is_open_locked_door = (
            front is not None
            and self._is_door_of_color(front, kb.get("lockedRoom_color"))
            and bool(getattr(front, "is_open", False))
        )

        # --- decision tree as you specified ---
        # 1) haven't seen key-room door yet
        if not key_room_seen:
            return "find_key_room_door"

        # 2) saw key-room door but not the key yet
        if key_room_seen and not key_seen:
            if front_is_open_key_door:
                return "find_key"
            else:
                return "goto_" + self.knowledge_base['key_color'] + "_room"
                #return "goto_key_room"
            

        # 3) key known but not holding it
        if key_seen and not holding_key:
            return "pickup_key"

        # 4) holding key, but locked room not seen
        if holding_key and not locked_room_seen:
            return "find_locked_room"

        # 5) holding key, locked room seen, but goal not seen
        if holding_key and locked_room_seen and not goal_seen:
            if front_is_open_locked_door:
                return "find_goal"
            else:
                return "goto_locked_room"
                #return "goto_" + self.knowledge_base['lockedRoom_color'] + "_room"

        # 6) goal known -> go to it
        if goal_seen:
            return "goto_goal"

        # 7) conservative fallback (should rarely happen)
        return "find_goal" if holding_key else "find_key"
   

    """
    Returns the function the subtask that is selectedf from the estimated knowledgebase
    """
    def estimated_ml_action(self, state):
        """
        Choose one subtask string from SUBTASK_LIST based on current knowledge & agent context.
        Assumes `state` is the MiniGrid env (has agent_pos/dir, grid, carrying, etc.)
        """
        # keep KB fresh (no-ops if you've already called it elsewhere)
        self.update_estimated_knowledge_base(state)
        kb = self.estimated_knowledge_base

        #boolean indicators of our current knowledge
        key_room_seen     = kb.get("keyRoom_loc")     is not None
        key_seen          = kb.get("key_loc")         is not None
        locked_room_seen  = kb.get("lockedRoom_loc")  is not None
        goal_seen         = kb.get("goal_loc")        is not None

        # holding key?
        carrying = getattr(state, "carrying", None)
        holding_key = False
        
        #if carrying key, is the key the right color of the given key_color?
        if carrying is not None and getattr(carrying, "type", None) == "key":
            c = getattr(carrying, "color", None)
            if isinstance(c, int) and 0 <= c < len(COLOR_NAMES):
                holding_key = COLOR_NAMES[c] == kb.get("key_color")
            else:
                holding_key = (c == kb.get("key_color"))

        # what's in front?
        front = self._front_obj(state)
        
        #is front an open door
        front_is_open_key_door = (
            front is not None
            and self._is_door_of_color(front, kb.get("keyRoom_color"))
            and bool(getattr(front, "is_open", False))
        )
        
        #is whats in front an opened lock door
        front_is_open_locked_door = (
            front is not None
            and self._is_door_of_color(front, kb.get("lockedRoom_color"))
            and bool(getattr(front, "is_open", False))
        )

        # --- decision tree as you specified ---
        # 1) haven't seen key-room door yet
        if not key_room_seen:
            return "find_key_room_door"

        # 2) saw key-room door but not the key yet
        if key_room_seen and not key_seen:
            if front_is_open_key_door:
                return "find_key"
            else:
                return "goto_" + self.estimated_knowledge_base['key_color'] + "_room"
                #return "goto_key_room"
            

        # 3) key known but not holding it
        if key_seen and not holding_key:
            return "pickup_key"

        # 4) holding key, but locked room not seen
        if holding_key and not locked_room_seen:
            return "find_locked_room"

        # 5) holding key, locked room seen, but goal not seen
        if holding_key and locked_room_seen and not goal_seen:
            if front_is_open_locked_door:
                return "find_goal"
            else:
                return "goto_locked_room"
                #return "goto_" + self.knowledge_base['lockedRoom_color'] + "_room"

        # 6) goal known -> go to it
        if goal_seen:
            return "goto_goal"

        # 7) conservative fallback (should rarely happen)
        return "find_goal" if holding_key else "find_key"
    


    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])