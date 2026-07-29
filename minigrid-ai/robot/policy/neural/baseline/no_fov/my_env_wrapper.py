# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/env_wrapper.py 
# still may need to fix the problem of training a mute robot
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
from collections import deque

import gymnasium as gym
import numpy as np
from robot.policy.neural.baseline.no_fov.features import (
    ACTIONS, N_ACTIONS, REVEAL_KEYS, OBS_SHAPE, encode_state,
)
#Used for setting up the training episode only to make the human agent; not cheating from robot side
HUMAN_FOV_POOL = [60, 120, 180]   # the HUMAN's true eyes, redrawn each episode
from human.agents.bayes_agent import BayesHumanAgent
from human.planning.bayes_planner import BayesianPlanner


ENV_ID = "MiniGrid-LockedRoom-v0"


#parameters for the reward function are in the function parameters passed into NoFovAssistEnv


    

class NoFovAssistEnv(gym.Env):
    """Raw-state assistance env. obs = (C,19,19) float32, action = one of 6."""

    def __init__(self, human_fovs=HUMAN_FOV_POOL, seeds=range(10_000),
                     comm_cost: float = 0.005, max_steps: int = 190,
                     random_walls: bool = True,
                     time_cost: float = 0.005,
                     key_bonus: float = 0.15):   # NEW; 0.0 disables

        self.human_fovs = list(human_fovs)
        self.seeds = list(seeds)
        self.comm_cost = comm_cost
        self.max_steps = max_steps
        self.random_walls = random_walls
        self.time_cost = time_cost
        self.key_bonus = key_bonus

        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=OBS_SHAPE, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def reset(self, seed=None, options=None):
        
        #seed and human true fov
        super().reset(seed=seed)
        ep_seed = int(self.np_random.choice(self.seeds))
        human_fov = int(self.np_random.choice(self.human_fovs))
        
        #env and state
        self.env = gym.make(ENV_ID, random_walls=self.random_walls, max_steps=self.max_steps)
        self.env.reset(seed=ep_seed)
        self.state = self.env.unwrapped
        
        #true human agent itself
        self.human = BayesHumanAgent(fov=human_fov)
        self.human.init_knowledge_base(self.state)
        self.planner = BayesianPlanner()
        
        #which of the hints already told
        self.told = {k: False for k in REVEAL_KEYS}

        #NEW: milestone bookkeeping, one payout per episode
        self._correct_color = self._correct_key_color()
        self._key_bonus_paid = False

        return self._observe(), {}

    #return the state as layers of layout
    def _observe(self):
        return encode_state(self.state, self.told, self.max_steps)

    # NEW: which key opens the room the goal is in. Pure grid geometry - flood
    # the goal's room treating every door as a wall, then read the colour of the
    # door on its edge. Nothing perceptual; the robot already has the full map.
    def _correct_key_color(self):
        st = self.state
        _, doors, goal = self._scan()
        if goal is None:
            return None
        zone, q = {goal}, deque([goal])
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in zone or not (0 <= nx < st.width and 0 <= ny < st.height):
                    continue
                if getattr(st.grid.get(nx, ny), "type", None) in ("wall", "door"):
                    continue
                zone.add((nx, ny))
                q.append((nx, ny))
        for color, (dx, dy) in doors.items():
            if any((nx, ny) in zone for nx, ny in
                   ((dx + 1, dy), (dx - 1, dy), (dx, dy + 1), (dx, dy - 1))):
                return color
        return None

    # ── world facts the robot is entitled to know ────────────────────────────
    # Live omniscient scan, recomputed every call. Not a reset-time snapshot,
    # so a key that was picked up and dropped elsewhere is always current.
    # A carried key is absent from the grid, which is exactly what we want.
    def _scan(self):
        st = self.state
        keys, doors, goal = {}, {}, None
        for x in range(st.width):
            for y in range(st.height):
                o = st.grid.get(x, y)
                if o is None:
                    continue
                if o.type == "key":
                    keys[o.color] = (x, y)
                elif o.type == "door":
                    doors[o.color] = (x, y)
                elif o.type == "goal":
                    goal = (x, y)
        return keys, doors, goal

    # ── given the kind of message, build the actual human injection ──────────
    def _resolve(self, kind: str):
        st = self.state
        ax, ay = st.agent_pos
        held = getattr(getattr(st, "carrying", None), "color", None)
        keys, doors, goal = self._scan()

        if kind == "key":
            # nearest key on the ground. No check that its door leads to the
            # goal - naming a decoy is a mistake the policy has to learn to
            # avoid, not one the env prevents for it.
            if not keys:
                return None
            return min(keys.items(),
                       key=lambda kv: abs(kv[1][0] - ax) + abs(kv[1][1] - ay))
        if kind == "door":
            loc = doors.get(held)
            return (held, loc) if loc else None
        if kind == "goal":
            return (None, goal) if goal else None
        # dead_room and empty_room are the only reveals that can state something
        # FALSE, and a false one poisons the human: dead_room writes into
        # dead_door_colors, bayes_agent.py:435 makes them drop that key and :479
        # filters it out for good, so a wrong dead_room on the correct colour
        # makes the episode unwinnable. DELIBERATELY UNGUARDED - the policy has
        # to learn that from reward, the same reason there is no action mask.
        # Measured cost of leaving it open: a speaking policy scored 25.3% vs
        # 46.7% for silence, so "say nothing" is a rational local optimum the
        # training schedule has to get it out of.
        if kind == "dead_room":
            return (held, (ax, ay)) if held else None
        if kind == "empty_room":
            return (None, (ax, ay))
        return None

    
    #write into human knowledge base logic
    def room_without_doors(self, state) -> frozenset:
        """Cells reachable from the human's position without crossing any door —
        i.e. 'the room they are standing in'. Uses state.grid.get(x, y) directly,
        so no encode()/indexing concerns."""
        ax, ay = state.agent_pos
        W, H = state.width, state.height
        zone = {(ax, ay)}
        q = deque([(ax, ay)])
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in zone or not (0 <= nx < W and 0 <= ny < H):
                    continue
                obj = state.grid.get(nx, ny)
                if getattr(obj, "type", None) in ("wall", "door"):
                    continue          # ALL doors are barriers here, locked or not
                zone.add((nx, ny))
                q.append((nx, ny))
        return frozenset(zone)


    def inject(self, kb, kind, color, loc, state) -> None:
        """Speak one fact into the real human's knowledge base.

        This is a SCHEMA CONTRACT with BayesHumanAgent/BayesianPlanner, not
        assistance logic. Every key name and tuple shape below is read back
        somewhere in human/ — if one drifts, the reveal is silently ignored and
        you get a mute-robot result with no error anywhere.
        """
        seen_at = kb.setdefault("seen_at", {})
        now = state.step_count

        if kind == "key":
            kb.setdefault("seen_keys", {})[color] = loc
            seen_at[("key", color)] = now

        elif kind == "door":
            kb.setdefault("seen_doors", {})[color] = loc
            seen_at[("door", color)] = now
            # The planner's BFS needs the actual door object to recognise the tile.
            obj = state.grid.get(*loc)
            if obj is not None:
                kb.setdefault("grid_cache", {})[loc] = obj
                if getattr(obj, "is_locked", False):
                    kb.setdefault("locked_door_locs", set()).add(loc)

        elif kind == "goal":
            kb["goal_loc"] = loc
            seen_at[("goal",)] = now              # 1-tuple — trailing comma matters

        elif kind == "dead_room":
            kb.setdefault("dead_door_colors", set()).add(color)
            seen_at[("dead", color)] = now        # "dead", NOT "dead_room"

        elif kind == "empty_room":
            room = self.room_without_doors(state)      # frozenset — must be hashable
            kb.setdefault("told_empty_rooms", set()).add(room)
            seen_at[("empty_room", room)] = now

    
    
    def step(self, action):
        kind = ACTIONS[int(action)]
        
        #reward for current timestep only
        reward = 0.0 
        
        spoke = kind != "wait"
        
        
        
        if spoke:
            target = self._resolve(kind)
            if target is not None:
                color, loc = target
                self.inject(self.human.knowledge_base, kind, color, loc, self.state)
                self.told[kind] = True
            reward -= self.comm_cost
            
        #get actual human subtask, action, etc that moves the actual game forward
        subtask = self.human.select_subtask(self.state)
        human_action = self.planner.next_action(subtask, self.state, self.human.knowledge_base)
        if human_action is None:
            human_action = 2  # forward fallback, matches eval_three_way.py

        _, _, terminated, truncated, _ = self.env.step(human_action)
        if terminated:
            reward += 1.0

        #NEW: one-shot bonus the moment they pick up the key that opens the goal
        #room. Paid once per episode, so a drop/pickup loop cannot farm it.
        if not self._key_bonus_paid and self.key_bonus:
            held = getattr(self.state, "carrying", None)
            if (getattr(held, "type", None) == "key"
                    and getattr(held, "color", None) == self._correct_color):
                self._key_bonus_paid = True
                reward += self.key_bonus

        reward -= self.time_cost
        
        return self._observe(), reward, terminated, truncated, {}
    
    
    

