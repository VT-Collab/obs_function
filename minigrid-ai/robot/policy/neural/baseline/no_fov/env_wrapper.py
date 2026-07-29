# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/ - the RAW-STATE baseline. See features.py for the design.
#
# THIS FILE - the Gym env every method in this folder trains against.
#
# Structurally identical to static_fov/env_wrapper.py: the RL agent is NOT the
# MiniGrid agent. The human moves itself (BayesHumanAgent picks a subtask,
# BayesianPlanner picks the action); the policy's only action is speech.
#
# REWARD  +1.0 when the human reaches the goal
#         -comm_cost per INFORMATIVE word (see charge_effective_only)
#         + potential-based shaping toward the goal (see shaping)
# Truncation at max_steps pays nothing. The human's true FOV is redrawn from
# [60,120,180] every episode and is never observable - training must be robust
# to all three, which is what makes this a fair no-FOV baseline.
#
# The ONE difference, and it is the whole point of this folder: there is no
# shadow agent. static_fov runs a BayesHumanAgent fixed at ASSUMED_FOV=120 and
# asks CandidateFinder "what is that human missing?". Here nothing models the
# human's perception at all. Observations come from the raw grid, and a reveal
# resolves to a concrete target from world geometry alone.
#
# WHAT REVEAL TARGET GETS SPOKEN. static_fov reads the (color, loc) out of
# CandidateFinder's belief-filtered candidate dict. With no belief there is no
# such dict, so targets resolve from self.solution - the world facts the robot
# is entitled to know (where the keys/doors/goal are) - with no filtering for
# whether the human already knows. The policy has to learn from reward that
# saying a thing the human already knows just costs comm_cost.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import gymnasium as gym
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX
from human.agents.bayes_agent import BayesHumanAgent
from human.planning.bayes_planner import BayesianPlanner
from robot.policy.deterministic._base import AssistBase
from robot.policy.neural.baseline.no_fov.features import (
    ACTIONS, N_ACTIONS, REVEAL_KEYS, OBS_SHAPE, encode_state,
)

ENV_ID = "MiniGrid-LockedRoom-v0"
HUMAN_FOV_POOL = [60, 120, 180]   # the HUMAN's true eyes, redrawn each episode


class _Geometry(AssistBase):
    """AssistBase purely for its world-fact helpers (self.solution, zones) and
    _write_reveal. It never runs a shadow: _get_assumed_fov exists only because
    AssistBase.reset() calls it, and the shadow it builds is never read."""

    def _get_assumed_fov(self) -> int:
        return 120  # unused - nothing here reads self.shadow


class NoFovAssistEnv(gym.Env):
    """Raw-state assistance env. obs = (C,19,19) float32, action = one of 6."""

    def __init__(self, human_fovs=HUMAN_FOV_POOL, seeds=range(10_000),
                 comm_cost: float = 0.02, max_steps: int = 190,
                 random_walls: bool = True,
                 charge_effective_only: bool = True, shaping: float = 0.0,
                 reveal_bonus: float = 0.15):
        # charge_effective_only - bill a word ONLY if it actually changed the
        #   human's knowledge base. Measured: a random policy utters 157 times per
        #   episode but only 44 of those transfer any information; the other 72%
        #   were being charged in full. That is a 3.15 exploration tax against a
        #   +1.00 prize, so every method learns "shut up" long before it can learn
        #   "speak here". Billing only effective words drops the tax to 0.88 and
        #   leaves the incentive to be concise exactly where it belongs - on words
        #   that actually say something.
        #
        # shaping - potential-based reward shaping (Ng, Harada & Russell 1999,
        #   "Policy Invariance Under Reward Transformations"), scale factor.
        #   F(s,s') = gamma*PHI(s') - PHI(s) with PHI = -(BFS distance from the
        #   human to the goal), normalised. Provably leaves the optimal policy
        #   unchanged while turning one +1 delivered ~130 steps late into a dense
        #   per-step signal. Uses only world geometry - never the human's FOV or
        #   knowledge - so it cannot leak.
        self.human_fovs = list(human_fovs)
        self.seeds = list(seeds)
        self.comm_cost = comm_cost
        self.max_steps = max_steps
        self.random_walls = random_walls
        self.charge_effective_only = charge_effective_only
        self.gamma = 0.99   # must match the learner's discount
        # TWO-PHASE MILESTONE REWARD - the credit-assignment fix. The bare +1.0 is
        #   ~130 steps and ~0.5 Bernoulli noise away from any single decision, and
        #   PPO from scratch stayed flat on it for 1M frames. Decompose the task
        #   into the milestone that actually gates success and reward progress
        #   toward the RIGHT subgoal in each phase:
        #
        #     phase 0 (no correct key): the human must reach and pick up the ONE
        #        key whose door leads to the goal. "Closer to the goal" is the
        #        wrong signal here - the key is usually AWAY from the goal - so we
        #        reward progress toward the correct KEY instead.
        #     milestone: +key_bonus the moment they pick up the correct key.
        #     phase 1 (holding correct key): now reward progress toward the GOAL.
        #
        #   The correct key/door is pure solution geometry (which door's room
        #   contains the goal) - facts the robot is entitled to know. Nothing here
        #   reads the human's FOV or belief. The robot can only ever REVEAL the
        #   nearest key, not necessarily the correct one, so it must learn from
        #   this signal which reveals actually move the human toward the right key
        #   (and to say "dead_room" to peel them off wrong rooms).
        self.progress_scale = shaping if shaping else 1.0   # weight on per-step progress
        self.key_bonus = reveal_bonus if reveal_bonus else 0.3

        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=OBS_SHAPE, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    # ── episode ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = int(self.np_random.choice(self.seeds))
        human_fov = int(self.np_random.choice(self.human_fovs))

        self.env = gym.make(ENV_ID, random_walls=self.random_walls, max_steps=self.max_steps)
        self.env.reset(seed=ep_seed)
        self.state = self.env.unwrapped

        self.human = BayesHumanAgent(fov=human_fov)
        self.human.init_knowledge_base(self.state)
        self.planner = BayesianPlanner()

        self.geo = _Geometry(patience=1)
        self.geo.reset(self.state)

        self.told = {k: False for k in REVEAL_KEYS}

        # Two-phase reward setup. correct_color = the door whose room holds the
        # goal; its key is the one the human actually needs. Pure geometry.
        sol = self.geo.solution
        goal = sol.get("goal")
        self._correct_color = None
        for color, dloc in sol.get("doors", {}).items():
            if goal is not None and goal in self.geo._room_zone_past_door(self.state, dloc):
                self._correct_color = color
                break
        key_loc = sol.get("keys", {}).get(self._correct_color)

        self._dist_goal = self._bfs_from(goal)
        self._dist_key = self._bfs_from(key_loc) if key_loc else self._dist_goal
        self._got_key = False
        ax, ay = self.state.agent_pos
        self._prev_target = float(self._dist_key[ax, ay])   # phase-0 target: the key
        self._norm = float(self._cap)                       # set in _bfs_from, same cap
        return self._observe(), {}

    def _has_correct_key(self) -> bool:
        c = getattr(self.state, "carrying", None)
        return (c is not None and getattr(c, "type", None) == "key"
                and getattr(c, "color", None) == self._correct_color)

    # ── potential-based shaping ──────────────────────────────────────────────

    def _bfs_from(self, cell):
        """BFS distance from every cell to `cell`, once per episode.

        Doors count as passable: the human CAN get through them with the right
        key, so a door is a detour, not a wall. Uses only the grid - world
        geometry the robot is entitled to - never anything about perception.
        """
        from collections import deque
        enc = self.state.grid.encode()[..., 0]
        H, W = enc.shape
        WALL = OBJECT_TO_IDX["wall"]
        # CAP, not infinity: an unreachable or far cell reads as "maximally far",
        # never a 1e6 that would blow up (prev_target - d). Any real BFS distance
        # in a 19x19 grid is well under this cap. self._norm divides by the same
        # cap, so one step of progress is bounded to <= 1.
        self._cap = 4 * (W + H)
        d = np.full((W, H), self._cap, dtype=np.int32)
        if cell is None:
            return d
        cx, cy = cell
        d[cx, cy] = 0
        q = deque([(cx, cy)])
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < W and 0 <= ny < H and d[nx, ny] == self._cap and enc[ny, nx] != WALL:
                    d[nx, ny] = min(d[x, y] + 1, self._cap)
                    q.append((nx, ny))
        return d

    def _observe(self):
        return encode_state(self.state, self.told, self.max_steps)

    # NOTE: there is deliberately no action_mask(). All 6 actions are always
    # available; an utterance that resolves to nothing simply does nothing, and
    # under charge_effective_only it is free. See features.py's header.

    # ── the reveal target, resolved from world geometry only ─────────────────

    def _resolve(self, kind: str):
        sol, st = self.geo.solution, self.state
        ax, ay = st.agent_pos
        carrying = getattr(st, "carrying", None)
        held_color = getattr(carrying, "color", None)

        if kind == "key":
            live = [(c, loc) for c, loc in sol["keys"].items()
                    if getattr(st.grid.get(*loc), "type", None) == "key"]
            if not live:
                return None
            # nearest live key. NOTE: no check that this key's door leads to the
            # goal - the same decoy behaviour AssistBase has (~43% of key
            # candidates are decoys, measured). Kept identical on purpose so the
            # two baselines differ only in FOV reasoning.
            return min(live, key=lambda c: abs(c[1][0] - ax) + abs(c[1][1] - ay))
        if kind == "door":
            loc = sol["doors"].get(held_color)
            return (held_color, loc) if loc else None
        if kind == "goal":
            loc = sol.get("goal")
            return (None, loc) if loc else None
        if kind == "dead_room":
            return (held_color, (ax, ay)) if held_color else None
        if kind == "empty_room":
            return (None, (ax, ay))
        return None

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        kind = ACTIONS[int(action)]
        reward = 0.0

        informative = False
        if kind != "wait":
            # No action mask - any utterance is allowed. What it costs depends on
            # whether it actually told the human something (see __init__).
            target = self._resolve(kind)
            before = self._kb_signature()
            if target is not None:
                color, loc = target
                self.geo._write_reveal(self.human.knowledge_base, (kind, color, loc), self.state)
                self.told[kind] = True
            informative = self._kb_signature() != before
            if informative or not self.charge_effective_only:
                reward -= self.comm_cost

        subtask = self.human.select_subtask(self.state)
        human_action = self.planner.next_action(subtask, self.state, self.human.knowledge_base)
        if human_action is None:
            human_action = 2  # forward fallback, matches eval_three_way.py

        _, _, terminated, truncated, _ = self.env.step(human_action)
        if terminated:
            reward += 1.0

        # ── two-phase milestone reward (see __init__) ────────────────────────
        # Milestone: the human just picked up the correct key -> phase 0 -> 1.
        if not self._got_key and self._has_correct_key():
            self._got_key = True
            reward += self.key_bonus
            ax, ay = self.state.agent_pos
            self._prev_target = float(self._dist_goal[ax, ay])  # retarget, no spurious jump

        # Dense progress toward THIS phase's subgoal: the correct key while they
        # lack it, the goal once they hold it. Summed over an episode this
        # telescopes to net distance closed, so it cannot be farmed by pacing.
        field = self._dist_goal if self._got_key else self._dist_key
        ax, ay = self.state.agent_pos
        d = float(field[ax, ay])
        reward += self.progress_scale * (self._prev_target - d) / self._norm
        self._prev_target = d

        return self._observe(), reward, terminated, truncated, {}

    def _kb_signature(self):
        """What the human KNOWS, ignoring the seen_at timestamps _write_reveal
        always stamps. Two signatures differ iff a real fact was transferred."""
        kb = self.human.knowledge_base
        return (tuple(sorted(kb.get("seen_keys", {}).items())),
                tuple(sorted(kb.get("seen_doors", {}).items())),
                kb.get("goal_loc"),
                tuple(sorted(kb.get("dead_door_colors", set()))),
                len(kb.get("told_empty_rooms", set())),
                len(kb.get("explored_cells", set())))
