from __future__ import annotations

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
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
from agents.steak_agent import HRLModel
from overcooked_ai_py.agents.agent import StayAgent
from planners.steak_planner import SteakMediumLevelActionManager
from mdp.steakhouse_env import SteakhouseEnv
from mdp.steakhouse_mdp import SteakhouseGridworld, SteakhouseState
from utils import OvercookedPygame, Logger, StudyConfig
from util import LSTM_Agent, BASE_REW_SHAPING_PARAMS


@dataclass
class Args:
    exp_name: str = "subtask_ppo_lstm"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "human_subtask"
    """the wandb's project name"""
    wandb_entity: str = "yachuanh"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "steakhouse"
    """the id of the environment"""
    total_timesteps: int = 1000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 350
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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
    
    VISION_LIMIT: bool = False # degree of vision
    VISION_BOUND: int = 120 # whether other agent knows visionlimitagent is vision limited
    EXPLORE: bool = False # low level exploaration depth, 
    KB_UPDATE_DELAY: int = 0
    KB_ACKN_PROB: bool = False

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_steps=args.max_steps)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_steps=args.max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStackObservation(env, 1)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    layout_name = args.layout
    layout_file_name = layout_name + ".layout"

    # Copy layout to Overcooked AI code base
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
    path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
    path_to = os.path.join(base_folder, "overcooked_ai", "src",
                            "overcooked_ai_py", "data", "layouts", layout_file_name)
    shutil.copy(path_from, path_to)
    world_mdp = SteakhouseGridworld.from_layout_name(layout_name)

    def rand_pos_start_state_fn():
        valid_positions = world_mdp.get_valid_joint_player_positions()
        start_pos = valid_positions[
            np.random.choice(len(valid_positions))
        ]
        start_state = SteakhouseState.from_player_positions(
            start_pos,
            bonus_orders=world_mdp.start_bonus_orders,
            all_orders=world_mdp.start_all_orders,
            order_list=world_mdp.order_list,
            order_display_list=world_mdp.order_display_list,
        )
        return start_state

    if args.rand_start:
        start_state_fn = rand_pos_start_state_fn
    else:
        start_state_fn = None
        
    env = SteakhouseEnv.from_mdp(world_mdp, start_state_fn=start_state_fn, horizon=args.total_time)
    env.reset()
    
    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': world_mdp.terrain_pos_dict['X'],
        'counter_pickup': world_mdp.terrain_pos_dict['X'],
        'same_motion_goals': True,
        "enable_same_cell": True,
    }

    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    single_observation_space = tuple(list(env.mdp.shape) + [39])
    single_action_space_n = 15

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + single_observation_space).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    # Initialize two human agent
    # mlam = SteakMediumLevelActionManager(world_mdp, COUNTERS_PARAMS)
    mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
        world_mdp, COUNTERS_PARAMS, custom_filename=None, force_compute=False, info=False
    )
    subtask_planner_model = LSTM_Agent(obs_size=math.prod(single_observation_space), action_size=single_action_space_n, args=args).to(device)
    human_agent = HRLModel(mlam, env.state, auto_unstuck=True, explore=args.EXPLORE, vision_limit=args.VISION_LIMIT, vision_bound=args.VISION_BOUND, kb_update_delay=args.KB_UPDATE_DELAY, kb_ackn_prob=args.KB_ACKN_PROB, subtask_planner_model=subtask_planner_model, obs_size=math.prod(single_observation_space), debug=False)
    other_agent = StayAgent()
    human_agent.set_agent_index(1)
    other_agent.set_agent_index(0)
    human_agent.init_knowledge_base(env.state)
    human_agent.set_mdp(env.mdp)
    other_agent.set_mdp(env.mdp)

    # Initialize states
    vstates = human_agent.get_obs(env.state)
    next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(human_agent.subtask_planner.lstm.num_layers, args.num_envs, human_agent.subtask_planner.lstm.hidden_size).to(device),
        torch.zeros(human_agent.subtask_planner.lstm.num_layers, args.num_envs, human_agent.subtask_planner.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    total_reward = 0

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            human_agent.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if next_done[0].item():
                writer.add_scalar("charts/episodic_reward", total_reward, global_step)
                writer.add_scalar("charts/episodic_return", (sum(infos["sparse_r_by_agent"]) + sum(infos["shaped_r_by_agent"])), global_step)
                writer.add_scalar("charts/episodic_length", env.state.timestep, global_step)

                env.reset()
                env = SteakhouseEnv.from_mdp(world_mdp, start_state_fn=start_state_fn, horizon=args.total_time)
                human_agent.init_knowledge_base(env.state)
                human_agent.set_mdp(env.mdp)
                vstates = human_agent.get_obs(env.state)

                next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
                next_done = torch.Tensor([False]).to(device)

                total_reward = 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                subtask, logprob, _, value, next_lstm_state = human_agent.subtask_planner.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = subtask
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            low_level_action = human_agent.subtask_to_action(env.state, subtask.cpu().numpy()[0])[0]
            other_player_action = other_agent.action(env.state)[0]
            joint_action = (low_level_action, other_player_action)
            next_obs, reward, next_done, infos = env.step(joint_action)
            next_obs = human_agent.get_obs(next_obs)
            reward += (sum(infos["sparse_r_by_agent"]) + sum(infos["shaped_r_by_agent"]))
            total_reward += reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = human_agent.subtask_planner.get_value(
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + single_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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

                _, newlogprob, entropy, newvalue, _ = human_agent.subtask_planner.get_action_and_value(
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

                human_agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(human_agent.subtask_planner.parameters(), args.max_grad_norm)
                human_agent.optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and iteration % 100 == 0:
            torch.save(human_agent.optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(human_agent.subtask_planner.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")

            # evaluate(
            #     f"{wandb.run.dir}/agent.pt",
            #     LSTM_Agent(obs_size=math.prod(single_observation_space), action_size=single_action_space_n, args=args).to(device),
            #     world_mdp,
            #     eval_episodes=3,
            #     device=torch.device("cpu"),
            #     args=args,
            #     params=COUNTERS_PARAMS,
            #     single_observation_space=single_observation_space,
            #     iter=iteration,
            #     wandb=wandb
            # )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", human_agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()