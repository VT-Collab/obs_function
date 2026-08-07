"""The evaluation kitchen: ONE self-play robot + ONE limited-vision human.

The self-play checkpoints in official_baselines/SP were trained with BOTH chefs
driven by the same network. Here chef 0 stays that network and chef 1 is
replaced by `LimitedVisionSteakHuman` at some FOV. That swap is the whole
experiment -- the robot is now playing with a partner it has never met, whose
behaviour depends on a hidden parameter (the cone) it cannot observe.

===========================================================================
WHAT IS REPRODUCED FROM TRAINING, EXACTLY
===========================================================================
The checkpoint is bound to the layout's grid size (CNNBase ends in
Linear(32*W*H, 64)), and it is bound to the observation encoding. Both have to
match or the numbers mean nothing:

  * `start_order_list=["steak"] * n_orders` as a real LIST. The .layout files
    declare it as a string, which silently breaks order consumption -- see the
    long note in official_baselines/utils/env_wrapper.py.
  * `rew_shaping_params=BASE_REW_SHAPING_PARAMS`. Not used for any decision
    here (nothing in this package reads reward), but the mdp's shaped-reward
    bookkeeping is part of the state transition, so it is kept identical.
  * `horizon=400`, `n_orders=4`, matching SP/run_specialists.sbatch.
  * `OvercookedEnv.from_mdp(..., horizon=horizon + 10)` and our own horizon
    check, exactly as SteakSelfPlayEnv does.
  * observation = `features.build_full_state(mdp, state, agent_index=0, t,
    horizon)` with NO padding, because the specialists trained unpadded.

===========================================================================
WHAT THE ROBOT IS AND IS NOT ALLOWED TO SEE
===========================================================================
ALLOWED   the full world state (it is a fully-observable agent -- that is what
          build_full_state hands the network), and the human's emitted ACTION
          each tick, which is a physical event anyone in the kitchen can watch.
NOT       the human's FOV, the human's subtask label, the human's beliefs, and
          the environment's reward. `step()` returns sparse reward and delivery
          counts for the METRIC only; test_no_cheating.py asserts that no
          decision-making code path ever reads them.
"""
import random

import numpy as np

import _paths  # noqa: F401  (sys.path side effect)

from overcooked_ai_py.mdp.overcooked_mdp import (
    SteakHouseGridworld, Action, BASE_REW_SHAPING_PARAMS)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from utils.features import build_full_state, N_LAYERS

from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner

ROBOT_INDEX = 0
HUMAN_INDEX = 1


def make_human(mdp, fov, seed, temperature=0.5, agent_index=HUMAN_INDEX):
    """A fresh limited-vision human with a reproducible sampler.

    `LimitedVisionSteakHuman.reset()` seeds its private RNG from the GLOBAL
    random stream (`random.Random(random.random())`), so seeding `random` here
    is what makes an episode replayable. It also means two conditions that see
    the same seed start from the same draw sequence and diverge only because
    the world diverged -- which is what we want when comparing baseline against
    baseline+module.
    """
    random.seed(seed)
    planner = SteakMotionPlanner(mdp, None)
    return LimitedVisionSteakHuman(mdp, fov, planner, agent_index=agent_index,
                                   temperature=temperature)


class SteakHumanRobotEnv:
    """One kitchen. Chef 0 = the policy under test, chef 1 = the human.

    Deliberately thin: it owns the mdp and the clock and nothing else. The
    episode loop lives in rollout.py, because the robot has to be able to
    interleave "let the shadows perceive s_t", "decide a_t", "observe h_t",
    "feed the filter" in a specific order, and an env that hides the human
    inside step() cannot express that order.
    """

    def __init__(self, layout, n_orders=4, horizon=400):
        self.layout = layout
        self.n_orders = n_orders
        self.horizon = horizon
        self.mdp = SteakHouseGridworld.from_layout_name(
            layout,
            start_order_list=["steak"] * n_orders,
            rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
        assert not isinstance(self.mdp.start_order_list, str)
        self.obs_shape = (N_LAYERS, self.mdp.shape[0], self.mdp.shape[1])
        self.env = None

    # -- episode ------------------------------------------------------------

    def reset(self):
        #+10 so is_done() is driven by our own horizon check, matching
        #SteakSelfPlayEnv. The mdp's own terminal (order list run down) still
        #fires normally and is the "finished early" case we are measuring.
        self.env = OvercookedEnv.from_mdp(self.mdp, info_level=0,
                                          horizon=self.horizon + 10)
        return self.state

    @property
    def state(self):
        return self.env.state

    @property
    def t(self):
        return self.env.t

    def robot_obs(self):
        """(1, 23, W, H) float32 -- one row, the robot's own point of view."""
        obs = build_full_state(self.mdp, self.env.state, agent_index=ROBOT_INDEX,
                               t=self.env.t, horizon=self.horizon)
        return obs.astype(np.float32)[None, ...]

    def step(self, robot_action, human_action):
        """Apply the joint action. Returns (sparse_reward, done, terminal).

        sparse_reward is METRIC-ONLY. Nothing that chooses an action reads it.
        """
        joint = (robot_action, human_action)
        _, sparse, done, _info = self.env.step(joint)
        hit_horizon = self.env.t >= self.horizon
        terminal = bool(done) and not hit_horizon   # the order list ran out
        return float(sparse), bool(done or hit_horizon), terminal

    def orders_left(self):
        return self.env.state.num_orders_remaining


def action_index(action):
    return Action.ACTION_TO_INDEX[action]


def index_action(i):
    return Action.INDEX_TO_ACTION[int(i)]
