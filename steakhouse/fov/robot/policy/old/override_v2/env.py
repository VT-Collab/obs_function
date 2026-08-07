"""
OverrideEnv - a THIN HOST for the FOV module. It subclasses the validated
RobotAssistEnv (dynamics/reward/motion unchanged) and does exactly two things:

  1. computes the baseline's suggested primitive robot action (super()._robot_action)
     and hands it to the module, applying whatever the module returns
  2. feeds the human's observed subtask into the module each step

NO decision logic, NO greedy, NO FOV reasoning lives here - all of that is in
AssistModule (assist_module.py). The base RobotAssistEnv is never modified.

`mode` selects the module's decision strategy ("subtask" / "assist" / else defer).
`knob` is accepted for backward-compat with old callers and ignored.
"""
from steakhouse.fov.robot.policy.old.baseline.env_wrapper import RobotAssistEnv, CANDIDATE_FOVS
from steakhouse.fov.robot.policy.old.override_v2.assist_module import AssistModule


class OverrideEnv(RobotAssistEnv):
    def __init__(self, *a, mode="subtask", candidate_fovs=None, conf_min=0.4,
                 blind_fov_min=45, blind_fov_max=90, knob=0.0, **k):
        super().__init__(*a, **k)
        self._mode = mode
        self._cand = list(candidate_fovs or CANDIDATE_FOVS)
        self._mkw = dict(conf_min=conf_min, blind_fov_min=blind_fov_min,
                         blind_fov_max=blind_fov_max)
        self.module = None
        self._last = "defer"

    def reset(self, seed=None):
        o = super().reset(seed)
        self.module = AssistModule(self.mdp, self.mlp, self._cand, human_index=1,
                                   robot_index=0, mode=self._mode, **self._mkw)
        self._last = "defer"
        return o

    def _robot_action(self, a_idx):
        baseline_primitive = super()._robot_action(a_idx)   # what the baseline would do
        action, tag = self.module.robot_action(self.env.state, baseline_primitive)
        self._last = tag
        return action

    def step(self, a_idx):
        obs, r, done, info = super().step(a_idx)             # calls my _robot_action
        # feed the human's observed subtask + the PRE-step state to the module
        self.module.observe(info.get("obs_state"), info.get("human_subtask"))
        info["final_kind"] = self._last
        return obs, r, done, info

    # ---- diagnostics passthrough (for the eval scripts) --------------------
    @property
    def inf(self):
        return self.module.inf

    @property
    def n_step(self):
        return self.module.n_step

    @property
    def n_override(self):
        return self.module.n_override

    @property
    def n_takeover(self):
        return self.module.n_takeover

    @property
    def n_in_cone(self):
        return 0     # reroute removed in the refactor; kept for eval-script compat
