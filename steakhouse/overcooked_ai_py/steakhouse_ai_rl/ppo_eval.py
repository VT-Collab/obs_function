from typing import Callable
import argparse
import gymnasium as gym
import torch
import numpy as np
import sys
import os
import math
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from agents.steak_agent import HRLModel, SteakGreedyHumanModel
from overcooked_ai_py.agents.agent import StayAgent
from agents.notifier_agent import LangStayAgent
from agents.reactive_agents import ReactiveSteakLimitVisionHumanModel, ReactiveHRLModel, ReactiveStaySteakLimitVisionHumanModel
from planners.steak_planner import SteakMediumLevelActionManager
from utils import OvercookedPygame, Logger, StudyConfig, LangOvercookedPygame
from mdp.steakhouse_mdp import SteakhouseGridworld, SteakhouseState
from mdp.steakhouse_env import SteakhouseEnv
from util import LSTM_Agent

class Args:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

def evaluate(
        subtask_policy_path,
        subtask_planner_model,
        world_mdp,
        eval_episodes=5,
        device=torch.device("cpu"),
        args=None,
        params=None,
        obs_size=None,
        action_size=None,
        iter=0,
        wandb=None,
    ):

    for i in range(eval_episodes):
        args.participant_id = i
        args.log_file_name = i
        study_config = StudyConfig(args, layout_name=world_mdp.layout_name)

        mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
            world_mdp, params, custom_filename=None, force_compute=False, info=False
        )
        if args.agent == "lstm":
            human_agent = HRLModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=args.EXPLORE, vision_limit=args.VISION_LIMIT, vision_bound=args.VISION_BOUND, kb_update_delay=args.KB_UPDATE_DELAY, kb_ackn_prob=args.KB_ACKN_PROB, subtask_planner_model=subtask_planner_model, obs_size=obs_size, action_size=action_size, pretrained_subtask_path=subtask_policy_path, debug=True, lstm_size=args.lstm_size)
            human_agent.set_agent_index(1)
            human_agent.init_knowledge_base(study_config.base_env.state)
        elif args.agent == "lang":
            human_agent = LangStayAgent(mlam, study_config.base_env.state, utter_model="LSTM", auto_unstuck=True, explore=args.EXPLORE, vision_limit=args.VISION_LIMIT, vision_bound=args.VISION_BOUND, kb_update_delay=args.KB_UPDATE_DELAY, kb_ackn_prob=args.KB_ACKN_PROB, subtask_planner_model=subtask_planner_model, obs_size=obs_size, action_size=action_size, utter_model_path=args.filepath+'/agent.pt', debug=True, lstm_size=args.lstm_size)
            human_agent.set_agent_index(1)
            human_agent.init_knowledge_base(study_config.base_env.state)
        elif args.agent == "no-op":
            human_agent = StayAgent()
        else:
            human_agent = SteakGreedyHumanModel(mlam, auto_unstuck=True)
            human_agent.set_agent_index(1)
            human_agent.init_knowledge_base(study_config.base_env.state)

        if args.other_agent == "greedy":
            other_agent = SteakGreedyHumanModel(mlam, auto_unstuck=False)
        elif args.other_agent == "reactive_greedy":
            other_agent = ReactiveSteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=False)
            other_agent.set_agent_index(0)
            other_agent.init_knowledge_base(study_config.base_env.state)
        elif args.other_agent == "reactive_stay":
            other_agent = ReactiveStaySteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=False)
            other_agent.set_agent_index(0)
            other_agent.init_knowledge_base(study_config.base_env.state)
        elif args.other_agent == "reactive_RL":
            other_agent = ReactiveHRLModel(
                mlam=mlam,
                start_state=study_config.base_env.state,
                auto_unstuck=True,
                explore=args.EXPLORE,
                vision_limit=args.VISION_LIMIT,
                vision_bound=args.VISION_BOUND,
                kb_update_delay=args.KB_UPDATE_DELAY,
                kb_ackn_prob=args.KB_ACKN_PROB,
                subtask_planner_model="LSTM",
                obs_size=obs_size,
                action_size=action_size,
                debug=False,
                device=device,
                pretrained_subtask_path=args.other_agent_subtask_path,
                lstm_size=args.lstm_size
            )
            other_agent.set_agent_index(0)
            other_agent.init_knowledge_base(study_config.base_env.state)
        else:
            other_agent = StayAgent()
            other_agent.set_agent_index(0)

        human_agent.set_agent_index(1)
        other_agent.set_agent_index(0)
        human_agent.set_mdp(study_config.base_env.mdp)
        other_agent.set_mdp(study_config.base_env.mdp)

        # Initialize logging
        logger = Logger(study_config, study_config.log_file_name,
                        agent1=other_agent, agent2=human_agent)
        gametime = 10000
        if args.agent == "lang":
            gameapp = LangOvercookedPygame(study_config.base_env, other_agent, human_agent, logger, gameTime=gametime)
        else:
            gameapp = OvercookedPygame(study_config.base_env, other_agent, human_agent, logger, gameTime=gametime)
        score = gameapp.on_execute() 
        if wandb:
            wandb.log({'eval reward': score})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--subtask_policy_path", type=str, default="/home/sophie.hsu.pi/Steakhouse-AI/wandb/run-20250311_063328-jvmxkuw2/files/agent.pt")
    # parser.add_argument("--subtask_policy_path", type=str, default="/home/sophie.hsu.pi/Steakhouse-AI/wandb/run-20250311_063337-hcc24cge/files/agent.pt")
    parser.add_argument("--run_id", type=str, default="d1q97dcv")
    # parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--layout_name", type=str, default="Overcooked2_1-2")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb", type=str, required=False)
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"yachuanh/human_subtask/{args.run_id}")
    # run.config['agent'] = 'lang'#'no-op'
    run.config['single_action_space_n'] = run.config['ml_action_n'] * run.config['condition_length'] + run.config['utter_id_offset']
    config = Args(**run.config)
    
    world_mdp = SteakhouseGridworld.from_layout_name(args.layout_name)

    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': world_mdp.terrain_pos_dict['X'],
        'counter_pickup': world_mdp.terrain_pos_dict['X'],
        'same_motion_goals': True,
        "enable_same_cell": True,
    }

    env = SteakhouseEnv.from_mdp(world_mdp, horizon=config.total_time)
    env.reset(rand_start=config.rand_start)
    single_observation_space = tuple(list(env.mdp.shape) + [39])
    

    evaluate(
        f"{config.filepath}/agent.pt",
        "LSTM",
        world_mdp,
        eval_episodes=args.eval_episodes,
        device=args.device,
        args=config,
        params=COUNTERS_PARAMS,
        obs_size=math.prod(single_observation_space),
        action_size=config.single_action_space_n,
    )