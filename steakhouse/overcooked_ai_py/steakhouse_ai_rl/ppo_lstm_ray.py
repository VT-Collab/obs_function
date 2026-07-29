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
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import shutil
import math
from ppo_eval import evaluate
from agents.steak_agent import HRLModel, SteakGreedyHumanModel
from overcooked_ai_py.agents.agent import StayAgent
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
    wandb_project_name: str = "steakhouse-plzz" #MISHA CHANGE: changed from human_subtask to steakhouse-plzz
    wandb_entity: str = "steakteam" #MISHA CHANGE: changed from yachuanh to steakteam (mishafu is personal so it doesn't work)
    capture_video: bool = False
    evaluate: bool = False

    # Algorithm specific arguments
    env_id: str = "steakhouse"
    total_timesteps: int = 1000000000
    learning_rate: float = 2.5e-4
    num_envs: int = 8  # number of parallel environments
    num_steps: int = 350
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
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

    lstm_size: int = 16
    max_steps: int = 200

    layout: str = "steak"
    total_time: int = 400
    rand_start: bool = True
    fov: int = 120
    participant_id: int = 0
    log_file_name: str = ""
    record_video: bool = True
    order_list: list = None
    agent: str = "lstm"
    other_agent: str = "greedy"
    layout_names: list = None
    condition_length: int = 1
    ml_action_n: int = 12 #MISHA CHANGE: CHANGED FROM 15 to 12!!!!!
    single_action_space_n: int = None

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
        else:
            self.other_agent = StayAgent()
        self.other_agent.set_agent_index(0)
        self.other_agent.set_mdp(self.env.mdp)

    def set_agent(self, agent_state_dict):
        # Build a new subtask planner and HRLModel using the provided parameters.
        
        self.agent = HRLModel(
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
            action_size=self.args.single_action_space_n,
            pretrained_subtask_state_dict=agent_state_dict,
            debug=False,
            device=self.device,
            lstm_size=self.args.lstm_size,
        )
        self.agent.set_agent_index(1)
        self.agent.init_knowledge_base(self.env.state)
        self.agent.set_mdp(self.env.mdp)

    def rollout(self):
        """
        Run a rollout for args.num_steps steps and return rollout data.
        Uses the agent's internal recurrent state.
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent first.")

        args = self.args
        device = self.device

        # Initialize observation using agent's get_obs; also reset the agent's recurrent states.
        vstates = self.agent.get_obs(self.env.state)
        next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
        next_done = torch.zeros(1).to(device)
        next_lstm_state = (
            torch.zeros(self.agent.subtask_planner.lstm.num_layers, 1, self.agent.subtask_planner.lstm.hidden_size).to(device),
            torch.zeros(self.agent.subtask_planner.lstm.num_layers, 1, self.agent.subtask_planner.lstm.hidden_size).to(device),
        )

        total_reward = 0
        first_flag = True

        # Preallocate storage for rollout data.
        obs = torch.zeros((args.num_steps,) + self.single_observation_space, device=device)
        actions = torch.zeros((args.num_steps,), device=device)
        logprobs = torch.zeros((args.num_steps,), device=device)
        rewards = torch.zeros((args.num_steps,), device=device)
        dones = torch.zeros((args.num_steps,), device=device)
        values = torch.zeros((args.num_steps,), device=device)

        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            if next_done[0].item() or first_flag:
                self.env.reset(rand_start=args.rand_start)
                self.env = SteakhouseEnv.from_mdp(self.world_mdp, horizon=args.total_time)
                self.agent.init_knowledge_base(self.env.state)
                self.agent.set_mdp(self.env.mdp)
                vstates = self.agent.get_obs(self.env.state)

                next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
                next_done = torch.Tensor([False]).to(device)

                first_flag = False
                
            # ALGO LOGIC: action logic
            with torch.no_grad():
                subtask, logprob, _, value, next_lstm_state = self.agent.subtask_planner.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = subtask
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            low_level_action = self.agent.subtask_to_action(self.env.state, subtask.cpu().numpy()[0])[0]
            other_player_action = self.other_agent.action(self.env.state)[0]
            joint_action = (other_player_action, low_level_action)
            next_obs, reward, next_done, infos = self.env.step(joint_action)
            next_obs = self.agent.get_obs(next_obs)
            reward -= 0.1
            reward += (sum(infos["sparse_r_by_agent"]) + sum(infos["shaped_r_by_agent"]))
            total_reward += reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.subtask_planner.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        return {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "total_reward": total_reward,
            "next_lstm_state": next_lstm_state,
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

    layout_names = ["steak"]
    # layout_names = ["steak", "Overcooked1_1-4", "Overcooked2_1-2", "Overcooked2_2-4", "Overcooked2_2-5"]
    # Layout and environment setup.
    for i in layout_names:
        layout_name = i # args.layout_name
        layout_file_name = layout_name + ".layout"
        base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
        path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
        path_to = os.path.join(base_folder, "overcooked_ai", "src", "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)
    
    #MISHA CHANGE ADD AN ENV LINE CUZ ITS THROWING ERRORS
    env = SteakhouseEnv.from_mdp(SteakhouseGridworld.from_layout_name(layout_names[0]), horizon=args.total_time)
    #END MISHA CHANGE
    
    single_observation_space = tuple(list(env.mdp.shape) + [39])

    subtask_planner_model = LSTM_Agent(
        obs_size=math.prod(single_observation_space),
        action_size=args.single_action_space_n,
        lstm_size=args.lstm_size,
    ).to(device)
    optimizer = optim.Adam(subtask_planner_model.parameters(), lr=2.5e-4, eps=1e-5)

    global_step = 0
    start_time = time.time()

    next_lstm_state = (
        torch.zeros(subtask_planner_model.lstm.num_layers, args.num_envs, subtask_planner_model.lstm.hidden_size).to(device),
        torch.zeros(subtask_planner_model.lstm.num_layers, args.num_envs, subtask_planner_model.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    total_reward = 0

    # Create a pool of RolloutWorker actors.
    if args.num_envs > len(layout_names):
        layout_names += [random.choice(layout_names) for _ in range(args.num_envs - len(layout_names))]
    workers = [RolloutWorker.remote(layout_names[i], args) for i in range(args.num_envs)]
    # w = RolloutWorker(layout_name, args)

    # Main training loop.
    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent_state_dict = {k: v.cpu().clone() for k, v in subtask_planner_model.state_dict().items()}
        ray.get([w.set_agent.remote(agent_state_dict) for w in workers])
        rollout_ids = [w.rollout.remote() for w in workers]
        rollouts = ray.get(rollout_ids)

        # w.set_agent(agent_state_dict)
        # rollouts = [w.rollout()]

        # Log rollout rewards and steps for each worker.
        for i, rollout_data in enumerate(rollouts):
            writer.add_scalar(f"rollout/worker_reward", rollout_data["total_reward"], global_step)
            writer.add_scalar(f"rollout/worker_steps", args.num_steps, global_step)

        rollout_obs = torch.stack([r["obs"] for r in rollouts], dim=1)          
        rollout_actions = torch.stack([r["actions"] for r in rollouts], dim=1)    
        rollout_logprobs = torch.stack([r["logprobs"] for r in rollouts], dim=1)
        rollout_rewards = torch.stack([r["rewards"] for r in rollouts], dim=1)
        rollout_dones = torch.stack([r["dones"] for r in rollouts], dim=1)
        rollout_values = torch.stack([r["values"] for r in rollouts], dim=1)
        rollout_advantages = torch.stack([r["advantages"] for r in rollouts], dim=1)
        rollout_returns = torch.stack([r["returns"] for r in rollouts], dim=1)
        next_lstm_state = (torch.stack([r["next_lstm_state"][0][0] for r in rollouts], dim=1).to(device),
                           torch.stack([r["next_lstm_state"][1][0] for r in rollouts], dim=1).to(device))

        b_obs = rollout_obs.reshape((-1,) + single_observation_space).to(device)
        b_logprobs = rollout_logprobs.reshape(-1).to(device)
        b_actions = rollout_actions.reshape(-1).to(device)
        b_dones = rollout_dones.reshape(-1).to(device)
        b_advantages = rollout_advantages.reshape(-1).to(device)
        b_returns = rollout_returns.reshape(-1).to(device)
        b_values = rollout_values.reshape(-1).to(device)

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

                _, newlogprob, entropy, newvalue, _ = subtask_planner_model.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(subtask_planner_model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and iteration % 100 == 0:
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(subtask_planner_model.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")

            if args.evaluate:

                world_mdp = SteakhouseGridworld.from_layout_name(args.layout_name)

                evaluate(
                    f"{wandb.run.dir}/agent.pt",
                    LSTM_Agent(
                        obs_size=math.prod(single_observation_space),
                        action_size=args.single_action_space_n,
                        args=args
                    ),
                    world_mdp,
                    eval_episodes=5,
                    device=device,
                    args=args,
                    params={
                        'start_orientations': True,
                        'wait_allowed': True,
                        'counter_goals': [],
                        'counter_drop': world_mdp.terrain_pos_dict['X'],
                        'counter_pickup': world_mdp.terrain_pos_dict['X'],
                        'same_motion_goals': True,
                        "enable_same_cell": True,
                    },
                    single_observation_space=single_observation_space,
                    iter=iteration,
                    wandb=wandb,
                )

        global_step += args.num_envs * args.num_steps
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        explained_var = 1 - np.var(b_returns.cpu().numpy() - b_values.cpu().numpy()) / np.var(b_returns.cpu().numpy())
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()
    ray.shutdown()
