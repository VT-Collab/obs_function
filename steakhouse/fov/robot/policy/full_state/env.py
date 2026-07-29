"""
Env wrappers that swap in the full-state observation. Both subclass the validated
RobotAssistEnv (dynamics, reward, fair planner unchanged) and only override what
the robot SEES. Everything else - the human, the reward, the motion - is identical
to the baseline, so the ONLY variable is the observation.
"""
from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.full_state.features_flat import extract_full_flat, OBS_DIM_FULL
from fov.robot.policy.full_state.features_grid import (
    extract_full_grid, N_CHANNELS, grid_dims)


class FlatEnv(RobotAssistEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.observation_dim = OBS_DIM_FULL

    def _obs(self):
        return extract_full_flat(self.mdp, self.env.state, human_index=1,
                                 t=self.t, horizon=self.horizon, age=self.age,
                                 last_human_action=self.last_human_action)


class GridEnv(RobotAssistEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        w, h = grid_dims(self.mdp)
        self.obs_shape = (N_CHANNELS, h, w)

    def _obs(self):
        return extract_full_grid(self.mdp, self.env.state, human_index=1,
                                 t=self.t, horizon=self.horizon, age=self.age,
                                 last_human_action=self.last_human_action)
