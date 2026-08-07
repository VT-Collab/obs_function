"""
MISHA NEW CHANGE - RL environment for the robot, paired with the limited-vision
human. One robot, one human, SHARED team reward.

DESIGN NOTES

* The human's FOV is RESAMPLED EVERY EPISODE from CANDIDATE_FOVS. The baseline
  policy is FOV-blind (features.py carries no posterior, no entropy), so training
  against a random FOV forces it to learn the best SINGLE strategy under
  uncertainty. That is the fair floor for the module to beat - training against
  one fixed FOV would produce an artificially narrow baseline that the module
  could beat for the wrong reason.

* The reward is shared: deliveries by EITHER agent count. The robot is not
  scored on its own throughput, so "help the human" and "cook it yourself" are
  both legitimate strategies and the policy has to work out which is better
  against a partner whose vision it cannot see.

* The robot's action is a SUBTASK CHOICE, executed physically by the motion
  planner. There is no communication action - any influence on the human has to
  travel through the world, and only lands if the robot is inside the human's
  cone (see the human's robot_belief).

* The robot may see the world fully, plus the human's position/orientation/
  actions. It may NEVER read the human's beliefs - those are what the Bayes
  module has to infer.
"""
from __future__ import annotations

import random

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner
from steakhouse.fov.robot.policy.old.baseline.features import (
    ACTIONS, N_ACTIONS, OBS_DIM, extract_features, station_locs, _station_state)

CANDIDATE_FOVS = [30, 60, 90, 120, 180, 360]
N_ORDERS = 4
HORIZON = 260

# Which station each take_* action services, and what it needs.
TAKE = {
    "take_meat": ("pot", "meat"), "take_onion": ("board", "onion"),
    "take_chop": ("board", None), "take_plate": ("sink", "dish"),
    "take_wash": ("sink", None),
}


class RobotAssistEnv:
    """Minimal gym-style env: reset() -> obs, step(a) -> obs, r, done, info."""

    def __init__(self, layout="steak_side_2", fovs=None, horizon=HORIZON, seed=None):
        self.layout = layout
        self.fovs = list(fovs or CANDIDATE_FOVS)
        self.horizon = horizon
        self.mdp = SteakHouseGridworld.from_layout_name(layout,
                                                        start_order_list=['steak'] * N_ORDERS)
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, BASE_PARAMS,
                                                             force_compute=False)
        self.locs = station_locs(self.mdp)
        self._seed = seed
        self.observation_dim = OBS_DIM
        self.n_actions = N_ACTIONS

    def reset(self, seed=None):
        s = self._seed if seed is None else seed
        if s is not None:
            np.random.seed(s)
            random.seed(s)
        # FOV resampled per episode - the baseline never learns which one it got
        self.true_fov = random.choice(self.fovs)
        self.env = OvercookedEnv.from_mdp(self.mdp, info_level=0, horizon=self.horizon + 10)
        self.human = LimitedVisionSteakHuman(self.mdp, self.true_fov,
                                             SteakMotionPlanner(self.mdp, self.mlp),
                                             agent_index=1)
        self.planner = SteakMotionPlanner(self.mdp, self.mlp)
        self.t = 0
        self.age = {k: 0 for k in ("pot", "board", "sink")}
        self._prev_state = {k: None for k in self.age}
        self.n_orders0 = len(self.env.state.order_list or [])
        self.last_human_action = None
        return self._obs()

    def _obs(self):
        return extract_features(self.mdp, self.env.state, human_index=1,
                                t=self.t, horizon=self.horizon, age=self.age)

    def _robot_action(self, a_idx):
        """Translate a subtask choice into a physical action."""
        name = ACTIONS[a_idx]
        state = self.env.state
        if name == "work":
            return self._greedy_work(state)
        if name == "stage_visible":
            # Act at whichever station is nearest the human's facing direction,
            # so the state change lands where they can see it. Still purely
            # physical - the human only benefits if it actually looks.
            hp = state.players[1]
            ahead = (hp.position[0] + hp.orientation[0] * 2,
                     hp.position[1] + hp.orientation[1] * 2)
            best, bd = None, 1e9
            for k, v in self.locs.items():
                for l in v:
                    d = abs(l[0] - ahead[0]) + abs(l[1] - ahead[1])
                    if d < bd:
                        best, bd = (k, l), d
            if best is None:
                return self._greedy_work(state)
            return self._go_interact(state, [best[1]])
        kind, _ = TAKE.get(name, (None, None))
        if kind is None:
            return self._greedy_work(state)
        return self._go_interact(state, self.locs[kind])

    def _greedy_work(self, state):
        try:
            from overcooked_ai_py.agents.agent import GreedySteakHumanModel
            if not hasattr(self, "_worker"):
                self._worker = GreedySteakHumanModel(self.mlp)
                self._worker.set_agent_index(0)
            a, _ = self._worker.action(state)
            return a
        except Exception:
            return Action.STAY

    def _go_interact(self, state, targets):
        p = state.players[0]
        faced = (p.position[0] + p.orientation[0], p.position[1] + p.orientation[1])
        if faced in targets:
            return Action.INTERACT
        best, bc = None, 1e9
        for g in targets:
            for d in Direction.ALL_DIRECTIONS:
                stand = (g[0] - d[0], g[1] - d[1])
                if stand not in self.mdp.get_valid_player_positions():
                    continue
                try:
                    plan, _, c = self.mlp.mp.get_plan(p.pos_and_or, (stand, d))
                except Exception:
                    continue
                if plan and c < bc:
                    best, bc = plan[0], c
        return best if best is not None else Action.STAY

    def step(self, a_idx):
        state = self.env.state                       # PRE-step state (what the human acts on)
        a_r = self._robot_action(a_idx)
        try:
            a_h, h_info = self.human.action(state)
        except Exception:
            a_h, h_info = Action.STAY, {}
        self.last_human_action = a_h
        # the human's chosen SUBTASK (not the raw move) - this is what the
        # SamplingBayesFOVInference scores, so the module must receive it.
        h_subtask = h_info.get("subtask") if isinstance(h_info, dict) else None

        before = len(self.env.state.order_list or [])
        _, _, done, _ = self.env.step((a_r, a_h))
        after = len(self.env.state.order_list or [])

        # SHARED reward: a delivery by either agent counts.
        r = 20.0 * (before - after)
        # small shaping so early learning has gradient at all
        r -= 0.01

        for k in self.age:
            cur = min((_station_state(self.mdp, self.env.state, l)
                       for l in self.locs[k]), default="empty")
            self.age[k] = 0 if cur != self._prev_state[k] else self.age[k] + 1
            self._prev_state[k] = cur

        self.t += 1
        done = done or self.t >= self.horizon
        info = dict(true_fov=self.true_fov, human_action=a_h,
                    human_subtask=h_subtask, obs_state=state,
                    delivered=self.n_orders0 - after,
                    human_delivered=self.human.n_delivered)
        return self._obs(), r, done, info
