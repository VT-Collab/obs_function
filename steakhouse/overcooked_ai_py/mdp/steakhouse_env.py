import numpy as np
from overcooked_ai_py.mdp.overcooked_env import MAX_HORIZON, OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from gymnasium import spaces
from .steakhouse_mdp import EVENT_TYPES, SteakhouseGridworld
import time
from overcooked_ai_py.mdp.actions import Action

class SteakhouseEnv(OvercookedEnv):
    def __init__(
        self,
        mdp_generator_fn,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=1,
        initial_info={},
        **kwargs,
    ):
        super().__init__(
            mdp_generator_fn,
            start_state_fn,
            horizon,
            mlam_params,
            info_level,
            num_mdp,
            initial_info,
        )

        self.metadata = None 

    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=None,
        **kwargs,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, SteakhouseGridworld)
        if num_mdp is not None:
            assert num_mdp == 1
        mdp_generator_fn = lambda _ignored: mdp
        return SteakhouseEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
            **kwargs,
        )

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return SteakhouseEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp,
        )

    ###################
    # BASIC ENV LOGIC #
    ###################

    def step(self, joint_action, joint_agent_action_info=None, display_phi=False):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()

        reward = 0

        if joint_agent_action_info is None:
            joint_agent_action_info = [{}, {}, {}]
        next_state, mdp_infos = self.mdp.get_state_transition(
            self.state, joint_action, display_phi, self.mp
        )

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done:
            self._add_episode_info(env_info)
            if self.mdp.is_terminal(self.state):
                reward += 100

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        reward += timestep_sparse_reward
        return (next_state, timestep_sparse_reward, done, env_info)

    def lossless_state_encoding_mdp(self, state):
        """
        Wrapper of the mdp's lossless_encoding
        """
        return self.mdp.lossless_state_encoding(state, self.horizon)

    def featurize_state_mdp(self, state, num_pots=2):
        """
        Wrapper of the mdp's featurize_state
        """
        return self.mdp.featurize_state(state, self.mlam, num_pots=num_pots)

    def reset(self, regen_mdp=True, outside_info={}, rand_start=False):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if rand_start:
            self.state = self.mdp.rand_pos_start_state_fn()
        else:
            if self.start_state_fn is None:
                self.state = self.mdp.get_standard_start_state()
            else:
                self.state = self.start_state_fn()

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
        }
        self.game_stats = {**events_dict, **rewards_dict}

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)

    ####################
    # TRAJECTORY LOGIC #
    ####################

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset(False)
        return successor_state, done

    ##################
    #   RENDERING    #
    ##################

    def render(self, mode="human"):
        time_step_left = self.horizon - self.t if self.horizon != MAX_HORIZON else None
        time_passed = (
            time.time() - self.start_time if self.start_time is not None else 0
        )
        self.mdp.render(
            self.state,
            mode,
            time_step_left=time_step_left,
            time_passed=time_passed,
        )

class CommsSteakhouseEnv(SteakhouseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mlam = kwargs.get("mlam", None)
        self.discretization = kwargs.get("discretization", "short")
        self.steakhouse_one_dim_obs_dim = kwargs.get("steakhouse_one_dim_obs_dim", 13)
        if kwargs.get("one_dim_obs", False):
            self.single_observation_space = (self.steakhouse_one_dim_obs_dim,)
        else:
            self.single_observation_space = tuple(list(self.mdp.shape) + [23])
        self.single_action_space = self.space()
        self.noti_action_length = len(self.single_action_space.nvec)-1
        self.human_action_idx = len(self.single_action_space.nvec)-1
        self.noti_history = []
        self.curr_agent_action = Action.STAY
        self.overwrite_flag = False
        self.max_history_size = 10
        env_info = {}
        env_info["utterance"] = np.array([0]*self.noti_action_length)
        self.t = 0
        self.start_time = 0
    
    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=None,
        **kwargs,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, SteakhouseGridworld)
        if num_mdp is not None:
            assert num_mdp == 1
        mdp_generator_fn = lambda _ignored: mdp
        return CommsSteakhouseEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
            **kwargs,
        )

    def space(self) -> spaces.Tuple:
        # the plus 1 is for the long notification type where you directly udpate the full knowledge base of the human, and 4 for the 4 directions
        if self.discretization == "short":
            return spaces.MultiDiscrete([3, 4, 1, Action.NUM_ACTIONS])
        elif self.discretization == "simple":
            return spaces.MultiDiscrete([3, 4, 2, Action.NUM_ACTIONS])
        elif self.discretization == "shortvlong":
            return spaces.MultiDiscrete([3, 5, Action.NUM_ACTIONS]) # addional 5th action that is a long notification, 1-4 are all short manuever notifications
        elif self.discretization == "complex":
            # Additional to the notify new, continue, and idle, notification type, notification length, and we add in notification key word position (short or full). Finally, we add the original action space (action actually taken by the human agent).
            return spaces.MultiDiscrete([3, 4, 2, 2, Action.NUM_ACTIONS])
        
    def step(self, joint_action):
        human_action = joint_action[-2]
        noti_action = np.array(joint_action[:-2])[0]
        overwrite_flag = joint_action[-1]
        
        # Store notification in history
        if noti_action[0] == 0: # no notification
            noti_action = np.array([0]*self.noti_action_length)
        elif noti_action[0] == 1: # continue previous notification
            noti_action = np.array([1] + [0]*(self.noti_action_length-1))
        elif noti_action[0] == 2 and self.discretization != "shortvlong":
            # process notification length: 2 or 5
            noti_action[2] = (noti_action[2]*3) + 2

            if self.noti_action_length > 3:
                noti_action[3] = min(noti_action[2], (noti_action[3]*3)+2)
        
        self.noti_history.append(noti_action)
        self.curr_agent_action = human_action
        self.overwrite_flag = overwrite_flag

        if len(self.noti_history) > self.max_history_size:
            self.noti_history.pop(0)

        next_state, timestep_sparse_reward, done, env_info = super().step([human_action, Action.STAY])

        env_info["utterance"] = noti_action

        return next_state, timestep_sparse_reward, done, env_info
