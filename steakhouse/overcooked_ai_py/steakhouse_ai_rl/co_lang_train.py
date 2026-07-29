from __future__ import annotations

import os
import random
import time
import copy
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import shutil
import math
from ppo_eval import evaluate
from agents.steak_agent import HRLModel, SteakGreedyHumanModel
from agents.reactive_agents import ReactiveSteakLimitVisionHumanModel, ReactiveHRLModel, ReactiveStaySteakLimitVisionHumanModel
from agents.notifier_agent import LangStayAgent, HierarchicalNotifierAgent
from overcooked_ai_py.agents.agent import StayAgent
from overcooked_ai_py.mdp.actions import Action
from planners.steak_planner import SteakMediumLevelActionManager
from mdp.steakhouse_env import SteakhouseEnv
from mdp.steakhouse_mdp import SteakhouseGridworld, SteakhouseState
from utils import OvercookedPygame, Logger, StudyConfig
from util import LSTM_Agent, BASE_REW_SHAPING_PARAMS

import ray
import sys
import os

# Get the absolute path to the project root (adjust if necessary)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure this path is available in Ray workers
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pass this environment variable to Ray workers
os.environ["PYTHONPATH"] = project_root


@dataclass
class Args:
    exp_name: str = "subtask_ppo_lstm"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "human_subtask"
    wandb_entity: str = "yachuanh"
    capture_video: bool = False
    evaluate: bool = False

    # Algorithm specific arguments
    env_id: str = "steakhouse"
    total_timesteps: int = 1000000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1  # number of parallel environments
    num_steps: int = 350
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 1
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    lstm_size: int = 128
    max_steps: int = 200

    layout: str = "steak"
    total_time: int = 400
    rand_start: bool = True
    fov: int = 120
    participant_id: int = 0
    log_file_name: str = ""
    record_video: bool = True
    order_list: list = None
    agent: str = "lang"
    other_agent: str = "reactive_stay"
    other_agent_subtask_path: str = None # should set path if other_agent is "reactive_RL"
    condition_length: int = 2
    ml_action_n: int = 12 # not counting chicken
    utter_id_offset: int = 2
    hist_utter_size: int = 10
    lstm_layers_n: int = 1
    encoded_latent_size: int = 128
    obs_seq_encoder_path: str = "models/lstm_obs_seq_encoder.pth"
    optimal_policy_path: str = "models/optimal_policy.pth"
    hrl_training_mode: str = "cotrain"
    VISION_LIMIT: bool = False
    VISION_BOUND: int = 120
    EXPLORE: bool = False
    KB_UPDATE_DELAY: int = 0
    KB_ACKN_PROB: bool = False

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ---------------------------------------------------
# Ray actor that encapsulates a rollout worker.
# Each worker creates its own environment, a dedicated agent copy,
# and a single StayAgent (for the other agent).
# ---------------------------------------------------
@ray.remote
class RolloutWorker:
    def __init__(self, layout_name, args):
        self.layout_name = layout_name
        self.world_mdp = SteakhouseGridworld.from_layout_name(layout_name)
        self.args = args

        # Ensure device selection is safe
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.env = SteakhouseEnv.from_mdp(self.world_mdp, horizon=self.args.total_time)
        self.env.reset(rand_start=self.args.rand_start)
        self.single_observation_space = tuple(list(self.env.mdp.shape) + [39])
        self.agent = None  # will be set via set_agent
        self.mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
            self.world_mdp,
            {
                'start_orientations': True,
                'wait_allowed': True,
                'counter_goals': [],
                'counter_drop': self.world_mdp.terrain_pos_dict['X'],
                'counter_pickup': self.world_mdp.terrain_pos_dict['X'],
                'same_motion_goals': True,
                "enable_same_cell": True,
            },
            custom_filename=None,
            force_compute=False,
            info=False,
        )

        # Instantiate StayAgent only once.
        if args.other_agent == "greedy":
            self.other_agent = SteakGreedyHumanModel(self.mlam, auto_unstuck=False)
        elif args.other_agent == "reactive_greedy":
            self.other_agent = ReactiveSteakLimitVisionHumanModel(self.mlam, self.env.state, vision_limit=self.args.VISION_LIMIT, vision_bound=self.args.VISION_BOUND,kb_update_delay=self.args.KB_UPDATE_DELAY, kb_ackn_prob=self.args.KB_ACKN_PROB, auto_unstuck=False)
            self.other_agent.set_agent_index(0)
            self.other_agent.init_knowledge_base(self.env.state)
        elif args.other_agent == "reactive_stay":
            self.other_agent = ReactiveStaySteakLimitVisionHumanModel(self.mlam, self.env.state, vision_limit=self.args.VISION_LIMIT, vision_bound=self.args.VISION_BOUND,kb_update_delay=self.args.KB_UPDATE_DELAY, kb_ackn_prob=self.args.KB_ACKN_PROB, auto_unstuck=False)
            self.other_agent.set_agent_index(0)
            self.other_agent.init_knowledge_base(self.env.state)
        elif args.other_agent == "reactive_RL":
            self.other_agent = ReactiveHRLModel(
                mlam=self.mlam,
                start_state=self.env.state,
                auto_unstuck=True,
                explore=self.args.EXPLORE,
                vision_limit=self.args.VISION_LIMIT,
                vision_bound=self.args.VISION_BOUND,
                kb_update_delay=self.args.KB_UPDATE_DELAY,
                kb_ackn_prob=self.args.KB_ACKN_PROB,
                subtask_planner_model="LSTM",
                obs_size=math.prod(self.single_observation_space),
                debug=False,
                device=self.device,
                pretrained_subtask_path=args.other_agent_subtask_path,
                lstm_size=self.args.lstm_size
            )
            self.other_agent.set_agent_index(0)
            self.other_agent.init_knowledge_base(self.env.state)
        else:
            self.other_agent = StayAgent()

        self.other_agent.set_agent_index(0)
        self.other_agent.set_mdp(self.env.mdp)

    def set_agent(self, notification_policy_state_dict, manager_policy_state_dict=None):
        # Build a new subtask planner and HRLModel using the provided parameters.

        self.agent = HierarchicalNotifierAgent(
            mlam=self.mlam,
            start_state=self.env.state,
            auto_unstuck=True,
            explore=self.args.EXPLORE,
            vision_limit=self.args.VISION_LIMIT,
            vision_bound=self.args.VISION_BOUND,
            kb_update_delay=self.args.KB_UPDATE_DELAY,
            kb_ackn_prob=self.args.KB_ACKN_PROB,
            debug=False,
            device=self.device,
            encoded_latent_size=self.args.encoded_latent_size,
            hist_utter_size=self.args.hist_utter_size,
            hrl_training_mode=self.args.hrl_training_mode,
            obs_seq_encoder_path=self.args.obs_seq_encoder_path,
            optimal_policy_path=self.args.optimal_policy_path,
            notification_policy_state_dict=notification_policy_state_dict,
            manager_policy_state_dict=manager_policy_state_dict
        )
        self.agent.set_agent_index(1)
        self.agent.init_knowledge_base(self.env.state)
        self.agent.set_mdp(self.env.mdp)

    def utter_hist_analysis(self):
        interrupt_flag = False
        exceeded_utter = False
        for i in range(len(self.agent.hist_utter)-1, -1, -1):
            if self.agent.hist_utter[i] != (-1,0,0): # there is an utterance
                # check if the utterance is complete
                if self.agent.hist_utter[i] != (0,0,0): # it's an actionable utterance
                    utter_length = self.agent.hist_utter[i][2]
                    if utter_length == (len(self.agent.hist_utter) - i):
                        return interrupt_flag, exceeded_utter
                    elif utter_length < (len(self.agent.hist_utter) - i): # if the utterance is longer than the remaining utterances
                        exceeded_utter = True # return continue as the agent would decide when it is completed
                        return interrupt_flag, exceeded_utter
                else:
                    interrupt_flag = True
                    return interrupt_flag, exceeded_utter
                break
        
        exceeded_utter = True # if the for loop ends, it means there is only 'continue' in the history
        return interrupt_flag, exceeded_utter

    def utter_reward_wrapper(self, utter):
        reward = 0

        if utter not in [(0,0,0), (-1,0,0)]:
            reward -= 0.1

        # check if the utterance is completed
        interrupt_flag, exceeded_utter = self.utter_hist_analysis()
        if interrupt_flag:
            reward -= 0.5
        else:
            if exceeded_utter:
                reward -= 0.5
            else:
                reward += 0.2

        return reward
        
    def rollout(self):
        """
        Run a rollout for args.num_steps steps and return rollout data.
        Uses the agent's internal recurrent state.
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent first.")

        args = self.args
        device = self.device

        next_obs_history = []

        # Initialize observation using agent's get_obs; also reset the agent's recurrent states.
        vstates = self.agent.get_obs(self.env.state)
        next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)

        next_obs_history = next_obs.expand(args.hist_utter_size, -1, -1, -1).to(device)
        noti_mode_history = torch.zeros(args.hist_utter_size, device=device)
        subtask_history = torch.zeros(args.hist_utter_size, device=device)
        length_history = torch.zeros(args.hist_utter_size, device=device)

        next_obs_encoded = self.agent.obs_seq_encoder(next_obs_history)
        next_target_encoded = self.agent.optimal_policy(next_obs_encoded)
        next_latent_input = torch.cat((next_obs_encoded, next_target_encoded), dim=1)
        next_manager_input = torch.cat((next_obs_encoded, noti_mode_history.unsqueeze(0), subtask_history.unsqueeze(0), length_history.unsqueeze(0)), dim=1)
        next_done = torch.zeros(1).to(device)

        total_reward = 0
        first_flag = True

        # Preallocate storage for rollout data.
        obs = torch.zeros((args.num_steps,) + self.single_observation_space, device=device)
        latent_inputs = torch.zeros((args.num_steps,) + (self.agent.obs_seq_encoder.latent_dim*2,), device=device)
        manager_inputs = torch.zeros((args.num_steps,) + (self.agent.obs_seq_encoder.latent_dim + args.hist_utter_size*3,), device=device)
        noti_modes = torch.zeros((args.num_steps,), device=device)
        id_actions = torch.zeros((args.num_steps,), device=device)
        length_actions = torch.zeros((args.num_steps,), device=device)
        logprobs = torch.zeros((args.num_steps,), device=device)
        rewards = torch.zeros((args.num_steps,), device=device)
        dones = torch.zeros((args.num_steps,), device=device)
        policy_values = torch.zeros((args.num_steps,), device=device)
        manager_values = torch.zeros((args.num_steps,), device=device)
        cosine_sims = torch.zeros((args.num_steps,), device=device)

        for step in range(args.num_steps):
            obs[step] = next_obs
            latent_inputs[step] = next_latent_input
            manager_inputs[step] = next_manager_input
            dones[step] = next_done

            if next_done[0].item() or first_flag:
                self.env.reset(rand_start=args.rand_start)
                self.env = SteakhouseEnv.from_mdp(self.world_mdp, horizon=args.total_time)
                self.agent.reset()
                self.agent.set_agent_index(1)
                self.agent.init_knowledge_base(self.env.state)
                self.agent.set_mdp(self.env.mdp)
                if self.args.other_agent in ["reactive_greedy", "reactive_RL", "reactive_stay"]:
                    self.other_agent.init_knowledge_base(self.env.state)
                    self.other_agent.set_mdp(self.env.mdp)
                vstates = self.agent.get_obs(self.env.state)

                next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
                next_obs_history = next_obs.expand(args.hist_utter_size, -1, -1, -1).to(device)
                noti_mode_history = torch.zeros(args.hist_utter_size, device=device)
                subtask_history = torch.zeros(args.hist_utter_size, device=device)
                length_history = torch.zeros(args.hist_utter_size, device=device)
                next_obs_encoded = self.agent.obs_seq_encoder(next_obs_history)
                next_target_encoded = self.agent.optimal_policy(next_obs_encoded)
                next_latent_input = torch.cat((next_obs_encoded, next_target_encoded), dim=1)
                next_manager_input = torch.cat((next_obs_encoded, noti_mode_history.unsqueeze(0), subtask_history.unsqueeze(0), length_history.unsqueeze(0)), dim=1)
                next_done = torch.Tensor([False]).to(device)

                first_flag = False
                
            # ALGO LOGIC: action logic
            with torch.no_grad():
                noti_mode, noti_mode_logprob, _, manager_value = self.agent.notifier_manager.get_action_and_value(next_manager_input)
                (id_action, length_action), id_logprob, length_logprob, _, _, policy_value = self.agent.notification_policy.get_action_and_value(next_latent_input)
                manager_values[step] = manager_value.flatten()
                policy_values[step] = policy_value.flatten()
            noti_modes[step] = noti_mode.item()
            id_actions[step] = id_action
            length_actions[step] = length_action
            logprobs[step] = id_logprob + length_logprob + noti_mode_logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            self.agent.update_utter(notify_mode=noti_mode.item(), condition=0, subtask=id_action.item(), length=length_action.item()+2)

            if isinstance(self.agent, HierarchicalNotifierAgent):
                low_level_action = Action.STAY
            else:
                low_level_action = self.agent.subtask_to_action(self.env.state)[0]
            other_player_action = self.other_agent.action(self.env.state)[0]
            joint_action = (other_player_action, low_level_action)
            
            # Update the environment and agent utterances in the observation based on the action
            next_obs, reward, next_done, infos = self.env.step(joint_action)
            if isinstance(self.agent, HierarchicalNotifierAgent):
                next_obs.players[self.agent.agent_index].utter = self.agent.curr_utter
            # update knowledge base based on the masked next observation given the vision limitation
            next_obs = self.agent.get_obs(next_obs)

            # Update the reward based on domain knowledge
            reward = self.utter_reward_wrapper(self.agent.curr_utter)
            reward -= 0.1 # cost for each step
            reward += (sum(infos["sparse_r_by_agent"]) + sum(infos["shaped_r_by_agent"]))
            total_reward += reward
        
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)
            next_obs_history = torch.cat((next_obs_history[1:], next_obs), dim=0)
            noti_mode_history = torch.cat((noti_mode_history[1:], noti_mode), dim=0)
            subtask_history = torch.cat((subtask_history[1:], id_action), dim=0)
            length_history = torch.cat((length_history[1:], length_action), dim=0)

            next_obs_encoded = self.agent.obs_seq_encoder(next_obs_history)

            # Compute the reward fot latent vector distance
            cosine_sim = F.cosine_similarity(next_obs_encoded, next_target_encoded, dim=-1)
            # reward += cosine_sim * 10

            next_target_encoded = self.agent.optimal_policy(next_obs_encoded)
            next_latent_input = torch.cat((next_obs_encoded, next_target_encoded), dim=1)
            next_manager_input = torch.cat((next_obs_encoded, noti_mode_history.unsqueeze(0), subtask_history.unsqueeze(0), length_history.unsqueeze(0)), dim=1)
            cosine_sims[step] = cosine_sim.item()
            rewards[step] = torch.tensor(reward).to(device).view(-1)


        # bootstrap value if not done
        with torch.no_grad():
            next_policy_value = self.agent.notification_policy.get_value(
                next_latent_input,
            ).reshape(1, -1)
            policy_advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_policy_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = policy_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - policy_values[t]
                policy_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            policy_returns = policy_advantages + policy_values


            next_manager_value = self.agent.notifier_manager.get_value(
                next_manager_input,
            ).reshape(1, -1)
            manager_advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_manager_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = manager_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - manager_values[t]
                manager_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            manager_returns = manager_advantages + manager_values

        return {
            "obs": obs,
            "manager_inputs": manager_inputs,
            "latent_inputs": latent_inputs,
            "noti_modes": noti_modes,
            "id_actions": id_actions,
            "length_actions": length_actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "policy_values": policy_values,
            "manager_values": manager_values,
            "policy_advantages": policy_advantages,
            "manager_advantages": manager_advantages,
            "policy_returns": policy_returns,
            "manager_returns": manager_returns,
            "cosine_sims": cosine_sims,
            "total_reward": total_reward,
        }

# ---------------------------------------------------
# Main training loop using Ray.
# ---------------------------------------------------
if __name__ == "__main__":
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": project_root}})  # Start Ray

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.single_action_space_n = args.ml_action_n
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir="runs")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.run.log_code(".")
        wandb.config.update({"filepath": wandb.run.dir})

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Layout and environment setup.
    layout_name = args.layout
    layout_file_name = layout_name + ".layout"
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
    path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
    path_to = os.path.join(base_folder, "overcooked_ai", "src", "overcooked_ai_py", "data", "layouts", layout_file_name)
    shutil.copy(path_from, path_to)
    world_mdp = SteakhouseGridworld.from_layout_name(layout_name)
    env = SteakhouseEnv.from_mdp(world_mdp, horizon=args.total_time)
    env.reset(rand_start=args.rand_start)
    single_observation_space = tuple(list(env.mdp.shape) + [39])
    encoded_latent_size = args.encoded_latent_size

    mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
        world_mdp,
        {
            'start_orientations': True,
            'wait_allowed': True,
            'counter_goals': [],
            'counter_drop': world_mdp.terrain_pos_dict['X'],
            'counter_pickup': world_mdp.terrain_pos_dict['X'],
            'same_motion_goals': True,
            "enable_same_cell": True,
        },
        custom_filename=None,
        force_compute=False,
        info=False,
    )
    human_agent = HierarchicalNotifierAgent(
        mlam=mlam,
        start_state=env.state,
        auto_unstuck=True,
        explore=args.EXPLORE,
        vision_limit=args.VISION_LIMIT,
        vision_bound=args.VISION_BOUND,
        kb_update_delay=args.KB_UPDATE_DELAY,
        kb_ackn_prob=args.KB_ACKN_PROB,
        debug=False,
        device=device,
        encoded_latent_size=args.encoded_latent_size,
        hist_utter_size=args.hist_utter_size,
        obs_seq_encoder_path=args.obs_seq_encoder_path,
        optimal_policy_path=args.optimal_policy_path,
        hrl_training_mode=args.hrl_training_mode,
    )
    human_agent.set_agent_index(1)
    human_agent.init_knowledge_base(env.state)
    human_agent.set_mdp(env.mdp)

    optimizer = optim.Adam(list(human_agent.notification_policy.parameters()) + list(human_agent.notifier_manager.parameters()), lr=2.5e-4, eps=1e-5)

    global_step = 0
    start_time = time.time()

    vstates = human_agent.get_obs(env.state)
    next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    total_reward = 0

    layout_names = ["Overcooked1_1-4"]
    # layout_names = ["steak", "Overcooked1_1-4", "Overcooked2_1-2", "Overcooked2_2-4", "Overcooked2_2-5"]
    # Layout and environment setup.
    for i in layout_names:
        layout_name = i # args.layout_name
        layout_file_name = layout_name + ".layout"
        base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
        path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
        path_to = os.path.join(base_folder, "overcooked_ai", "src", "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)

    # Create a pool of RolloutWorker actors.
    if args.num_envs > len(layout_names):
        layout_names += [random.choice(layout_names) for _ in range(args.num_envs - len(layout_names))]
    workers = [RolloutWorker.remote(layout_names[i], args) for i in range(args.num_envs)]
    # w = RolloutWorker(layout_names[0], args)

    # Main training loop.
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # agent_state_dict = {k: v.cpu().clone() for k, v in human_agent.notification_policy.state_dict().items()}
        policy_agent_state_dict = {k: v.cpu().clone() for k, v in human_agent.notification_policy.state_dict().items()}
        manager_agent_state_dict = {k: v.cpu().clone() for k, v in human_agent.notifier_manager.state_dict().items()}
        ray.get([w.set_agent.remote(policy_agent_state_dict, manager_agent_state_dict) for w in workers])
        rollout_ids = [w.rollout.remote() for w in workers]
        rollouts = ray.get(rollout_ids)

        # w.set_agent(policy_agent_state_dict)
        # rollouts = [w.rollout()]

        # Log rollout rewards and steps for each worker.
        for i, rollout_data in enumerate(rollouts):
            writer.add_scalar(f"rollout/worker_reward", rollout_data["total_reward"], global_step)
            writer.add_scalar(f"rollout/worker_steps", args.num_steps, global_step)

        rollout_obs = torch.stack([r["obs"] for r in rollouts], dim=1)   
        rollout_manager_inputs = torch.stack([r["manager_inputs"] for r in rollouts], dim=1)
        rollout_latent_inputs = torch.stack([r["latent_inputs"] for r in rollouts], dim=1)
        rollout_noti_modes = torch.stack([r["noti_modes"] for r in rollouts], dim=1)
        rollout_id_actions = torch.stack([r["id_actions"] for r in rollouts], dim=1)       
        rollout_length_actions = torch.stack([r["length_actions"] for r in rollouts], dim=1)
        rollout_logprobs = torch.stack([r["logprobs"] for r in rollouts], dim=1)
        rollout_rewards = torch.stack([r["rewards"] for r in rollouts], dim=1)
        rollout_dones = torch.stack([r["dones"] for r in rollouts], dim=1)
        rollout_policy_values = torch.stack([r["policy_values"] for r in rollouts], dim=1)
        rollout_manager_values = torch.stack([r["manager_values"] for r in rollouts], dim=1)
        rollout_policy_advantages = torch.stack([r["policy_advantages"] for r in rollouts], dim=1)
        rollout_manager_advantages = torch.stack([r["manager_advantages"] for r in rollouts], dim=1)
        # rollout_returns = torch.stack([r["returns"] for r in rollouts], dim=1)
        rollout_policy_returns = torch.stack([r["policy_returns"] for r in rollouts], dim=1)
        rollout_manager_returns = torch.stack([r["manager_returns"] for r in rollouts], dim=1)
        rollout_cosine_sims = torch.stack([r["cosine_sims"] for r in rollouts], dim=1)

        b_obs = rollout_obs.reshape((-1,) + single_observation_space).to(device).detach()
        b_manager_inputs = rollout_manager_inputs.reshape((-1,) + (encoded_latent_size + args.hist_utter_size*3,)).to(device).detach()
        b_latent_inputs = rollout_latent_inputs.reshape((-1,) + (encoded_latent_size*2,)).to(device).detach()
        b_noti_modes = rollout_noti_modes.reshape(-1).to(device).detach()
        b_logprobs = rollout_logprobs.reshape(-1).to(device).detach()
        b_id_actions = rollout_id_actions.reshape(-1).to(device).detach()
        b_length_actions = rollout_length_actions.reshape(-1).to(device).detach()
        b_dones = rollout_dones.reshape(-1).to(device).detach()
        b_policy_advantages = rollout_policy_advantages.reshape(-1).to(device).detach()
        b_manager_advantages = rollout_manager_advantages.reshape(-1).to(device).detach()
        # b_returns = rollout_returns.reshape(-1).to(device).detach()
        b_policy_values = rollout_policy_values.reshape(-1).to(device).detach()
        b_manager_values = rollout_manager_values.reshape(-1).to(device).detach()
        b_policy_returns = rollout_policy_returns.reshape(-1).to(device).detach()
        b_manager_returns = rollout_manager_returns.reshape(-1).to(device).detach()
        b_cosine_sims = rollout_cosine_sims.reshape(-1).to(device).detach()

        # frequency of noti modes
        noti_mode_freq = dict(zip(np.unique(b_noti_modes.cpu().numpy(), return_counts=True)[0].astype(str), np.unique(b_noti_modes.cpu().numpy(), return_counts=True)[1]/len(b_noti_modes)))
        writer.add_scalars("rollout/noti_mode_freq", noti_mode_freq, global_step)

        # frequency of id actions
        id_action_freq = dict(zip(np.unique(b_id_actions.cpu().numpy(), return_counts=True)[0].astype(str), np.unique(b_id_actions.cpu().numpy(), return_counts=True)[1]/len(b_id_actions)))
        writer.add_scalars("rollout/id_action_freq", id_action_freq, global_step)

        # frequency of length actions
        length_action_freq = dict(zip(np.unique(b_length_actions.cpu().numpy(), return_counts=True)[0].astype(str), np.unique(b_length_actions.cpu().numpy(), return_counts=True)[1]/len(b_length_actions)))
        writer.add_scalars("rollout/length_action_freq", length_action_freq, global_step)
        
        writer.add_scalar("rollout/cosine_sim", b_cosine_sims.mean().item(), global_step)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                
                _, newnotimodelogprob, noti_mode_entropy, new_manager_value = human_agent.notifier_manager.get_action_and_value(
                    b_manager_inputs[mb_inds],
                    b_noti_modes.long()[mb_inds],
                )

                _, newidlogprob, newlengthlogprob, id_entropy, length_entropy, new_policy_value = human_agent.notification_policy.get_action_and_value(
                    b_latent_inputs[mb_inds],
                    b_id_actions.long()[mb_inds],
                    b_length_actions.long()[mb_inds],
                )
                newlogprob = newidlogprob + newlengthlogprob + newnotimodelogprob
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_policy_advantages[mb_inds] + b_manager_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_policy_value = new_policy_value.view(-1)
                new_manager_value = new_manager_value.view(-1)
                if args.clip_vloss:
                    policy_v_loss_unclipped = (new_policy_value - b_policy_returns[mb_inds]) ** 2
                    policy_v_clipped = b_policy_values[mb_inds] + torch.clamp(
                        new_policy_value - b_policy_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    policy_v_loss_clipped = (policy_v_clipped - b_policy_returns[mb_inds]) ** 2
                    policy_v_loss_max = torch.max(policy_v_loss_unclipped, policy_v_loss_clipped)
                    policy_v_loss = 0.5 * policy_v_loss_max.mean()

                    manager_v_loss_unclipped = (new_manager_value - b_manager_returns[mb_inds]) ** 2
                    manager_v_clipped = b_manager_values[mb_inds] + torch.clamp(
                        new_manager_value - b_manager_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    manager_v_loss_clipped = (manager_v_clipped - b_manager_returns[mb_inds]) ** 2
                    manager_v_loss_max = torch.max(manager_v_loss_unclipped, manager_v_loss_clipped)
                    manager_v_loss = 0.5 * manager_v_loss_max.mean()
                    v_loss = (policy_v_loss + manager_v_loss)/2
                else:
                    policy_v_loss = 0.5 * ((new_policy_value - b_policy_returns[mb_inds]) ** 2).mean()
                    manager_v_loss = 0.5 * ((new_manager_value - b_manager_returns[mb_inds]) ** 2).mean()
                    v_loss = (policy_v_loss + manager_v_loss)/2

                entropy_loss = id_entropy.mean() + length_entropy.mean() + noti_mode_entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(human_agent.notification_policy.parameters()) + list(human_agent.notifier_manager.parameters()), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_policy_values.cpu().numpy(), b_policy_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and iteration % 100 == 0:
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(human_agent.notification_policy.state_dict(), f"{wandb.run.dir}/policy.pt")
            torch.save(human_agent.notifier_manager.state_dict(), f"{wandb.run.dir}/manager.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/policy.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/manager.pt", base_path=wandb.run.dir, policy="now")

        global_step += args.num_envs * args.num_steps
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/policy_v_loss", policy_v_loss.item(), global_step)
        writer.add_scalar("losses/manager_v_loss", manager_v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        explained_var = 1 - np.var(b_policy_returns.cpu().numpy() - b_policy_values.cpu().numpy()) / np.var(b_policy_returns.cpu().numpy())
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()
    ray.shutdown()
