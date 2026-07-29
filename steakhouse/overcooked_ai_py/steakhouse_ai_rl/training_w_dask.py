# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import hydra
import random
import time
from omegaconf import DictConfig, OmegaConf
import math
import os
import shutil
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from util import make_env, init_env, NNAgent, setup_logging
from agents.steak_agent import HRLModel, ML_ACTION_LIST
from overcooked_ai_py.agents.agent import StayAgent
from planners.steak_planner import SteakMediumLevelActionManager
from mdp.steakhouse_env import SteakhouseEnv
from mdp.steakhouse_mdp import SteakhouseGridworld
from utils import OvercookedPygame, Logger, StudyConfig
from dask.distributed import Client, LocalCluster
from functools import partial

def evaluate(
        subtask_policy_path,
        world_mdp,
        eval_episodes=5,
        device=torch.device("cpu"),
        cfg=None,
        params=None,
        single_observation_space=None,
        iter=0,
        wandb=None,
    ):

    for i in range(eval_episodes):
        cfg.env.participant_id = i
        cfg.env.log_file_name = iter
        study_config = StudyConfig(cfg.env)

        if 'rand_start' in cfg.env:
            empty_cell = study_config.world_mdp.terrain_pos_dict[' ']
            p1, p2 = random.choices(np.arange(0, len(empty_cell)), k=2)
            study_config.world_mdp.start_player_positions = [empty_cell[p1], empty_cell[p2]]
            
        mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
            world_mdp, params, custom_filename=None, force_compute=False, info=False
        )
        human_agent = HRLModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=cfg.agent.EXPLORE, vision_limit=cfg.agent.VISION_LIMIT, vision_bound=cfg.agent.VISION_BOUND, kb_update_delay=cfg.agent.KB_UPDATE_DELAY, kb_ackn_prob=cfg.agent.KB_ACKN_PROB, obs_size=math.prod(single_observation_space), pretrained_subtask_planner=f"{wandb.run.dir}/agent.pt", debug=True)
        human_agent.subtask_planner.load_state_dict(torch.load(subtask_policy_path, map_location=device, weights_only=True))
        human_agent.subtask_planner.eval()

        other_agent = StayAgent()
        human_agent.set_agent_index(1)
        other_agent.set_agent_index(0)
        human_agent.init_knowledge_base(study_config.base_env.state)
        human_agent.set_mdp(study_config.base_env.mdp)
        other_agent.set_mdp(study_config.base_env.mdp)

        # Initialize logging
        logger = Logger(study_config, study_config.log_file_name,
                        agent1=other_agent, agent2=human_agent)
        gametime = 10000
        gameapp = OvercookedPygame(study_config.base_env, other_agent, human_agent, logger,gameTime=gametime)
        score = gameapp.on_execute() 
        wandb.log({'eval reward': score})

# Define the function each Dask worker will run
def worker_task(args, cfg, pretrained_subtask_planner=None):
    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize world_mdp, env, mlam, human_agent, and other_agent
    world_mdp = SteakhouseGridworld.from_layout_name(cfg.env.layout)

    # Randomize start positions if required
    if 'rand_start' in cfg.env:
        empty_cell = world_mdp.terrain_pos_dict[' ']
        p1, p2 = random.choices(np.arange(0, len(empty_cell)), k=2)
        world_mdp.start_player_positions = [empty_cell[p1], empty_cell[p2]]
    
    env = SteakhouseEnv.from_mdp(world_mdp, horizon=cfg.env.total_time)
    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': world_mdp.terrain_pos_dict['X'],
        'counter_pickup': world_mdp.terrain_pos_dict['X'],
        'same_motion_goals': True,
        "enable_same_cell": True,
    }
    
    mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
        world_mdp, COUNTERS_PARAMS, custom_filename=None, force_compute=False, info=False
    )
    
    single_observation_space = tuple(list(env.mdp.shape) + [39])

    human_agent = HRLModel(
        mlam, env.state, auto_unstuck=True, explore=cfg.agent.EXPLORE,
        vision_limit=cfg.agent.VISION_LIMIT, vision_bound=cfg.agent.VISION_BOUND,
        kb_update_delay=cfg.agent.KB_UPDATE_DELAY, kb_ackn_prob=cfg.agent.KB_ACKN_PROB,
        obs_size=math.prod(single_observation_space), 
        pretrained_subtask_planner=pretrained_subtask_planner,
        device=device, debug=True
    )
    other_agent = StayAgent()

    # Set agent indices and initialize states
    human_agent.set_agent_index(1)
    other_agent.set_agent_index(0)
    human_agent.init_knowledge_base(env.state)
    human_agent.set_mdp(env.mdp)
    other_agent.set_mdp(env.mdp)

    # Initialize storage for a single environment per worker
    obs = torch.zeros((args.num_steps,) + single_observation_space).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)
    actions = torch.zeros(args.num_steps).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)

    # Initial observation and done flag
    vstates = human_agent.get_obs(env.state)
    next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
    next_done = torch.zeros(1, dtype=torch.bool).to(device)

    for step in range(0, args.num_steps):
        global_step += 1
        obs[step] = next_obs.squeeze(0)  # Store single environment obs
        dones[step] = next_done

        if next_done.item():
            if 'rand_start' in cfg.env:
                empty_cell = world_mdp.terrain_pos_dict[' ']
                p1, p2 = random.choices(np.arange(0, len(empty_cell)), k=2)
                world_mdp.start_player_positions = [empty_cell[p1], empty_cell[p2]]

            env = SteakhouseEnv.from_mdp(world_mdp, horizon=cfg.env.total_time)
            human_agent.init_knowledge_base(env.state)
            human_agent.set_mdp(env.mdp)
            vstates = human_agent.get_obs(env.state)

            next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
            next_done = torch.Tensor([False]).to(device)

        # ALGO LOGIC: action logic
        with torch.no_grad():
            subtask, logprob, _, value = human_agent.subtask_planner.get_action_and_value(next_obs.flatten())
            values[step] = value.item()
        actions[step] = subtask
        logprobs[step] = logprob

        # Execute the game and log data
        low_level_action = human_agent.subtask_to_action(env.state, subtask.cpu().numpy())[0]
        other_player_action = other_agent.action(env.state)[0]
        joint_action = (low_level_action, other_player_action)
        next_obs, reward, next_done, infos = env.step(joint_action)
        next_obs = human_agent.get_obs(next_obs)
        rewards[step] = torch.tensor(reward).to(device)
        next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)

    # Bootstrap value if not done
    with torch.no_grad():
        next_value = human_agent.subtask_planner.get_value(next_obs.flatten()).reshape(1, -1)
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

    return obs, dones, values, actions, logprobs, rewards, advantages, returns, single_observation_space  # Return the required data

@hydra.main(config_path=".", config_name="ppo_training")
def main(cfg: DictConfig):
    args = cfg.ppo
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Initialize Dask
    cluster = LocalCluster()
    client = Client(cluster)
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    layout_name = cfg.env.layout
    layout_file_name = layout_name + ".layout"

    # Copy layout to Overcooked AI code base
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
    path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
    path_to = os.path.join(base_folder, "overcooked_ai", "src",
                            "overcooked_ai_py", "data", "layouts", layout_file_name)
    shutil.copy(path_from, path_to)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    for iteration in range(1, args.num_iterations + 1):

        if os.path.exists(f"{wandb.run.dir}/agent.pt"):
            pretrained_subtask_planner = f"{wandb.run.dir}/agent.pt"
        else:
            pretrained_subtask_planner = None
            
        # Configure Dask cluster
        client = Client(**cfg.worker)
        print(f"Started a dask local cluster: {client}")

        # Run worker_task on each worker and gather all tensors from each worker
        futures = [client.submit(worker_task, args, cfg, pretrained_subtask_planner) for _ in range(len(client.ncores()))]

        # Collect results from all workers
        results = client.gather(futures)

        client.close()

        # Extract and concatenate individual components from results
        obs = torch.cat([result[0].unsqueeze(1) for result in results], dim=1)
        dones = torch.cat([result[1].unsqueeze(1) for result in results], dim=1)
        values = torch.cat([result[2].unsqueeze(1) for result in results], dim=1)
        actions = torch.cat([result[3].unsqueeze(1) for result in results], dim=1)
        logprobs = torch.cat([result[4].unsqueeze(1) for result in results], dim=1)
        rewards = torch.cat([result[5].unsqueeze(1) for result in results], dim=1)
        advantages = torch.cat([result[6].unsqueeze(1) for result in results], dim=1)
        returns = torch.cat([result[7].unsqueeze(1) for result in results], dim=1)
        single_observation_space = results[0][8]

        print("All worker results collected")

        # Close client
        client.close()

        subtask_planner = NNAgent(obs_size=math.prod(single_observation_space), action_size=len(ML_ACTION_LIST)).to(device)
        optimizer = optim.Adam(subtask_planner.parameters(), lr=2.5e-4, eps=1e-5)

        if os.path.exists(f"{wandb.run.dir}/agent.pt"):
            subtask_planner.load_state_dict(torch.load(f"{wandb.run.dir}/agent.pt"))
            optimizer.load_state_dict(torch.load(f"{wandb.run.dir}/optimizer.pt"))

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # flatten the batch
        b_obs = obs.reshape((-1,) + (math.prod(single_observation_space),))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = subtask_planner.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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
                nn.utils.clip_grad_norm_(subtask_planner.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and iteration % 100 == 0:
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(subtask_planner.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")

            # evaluation()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # env.close()
    writer.close()

if __name__ == "__main__":
    main()