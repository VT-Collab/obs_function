#SET THE VISION BOUND EVERY STEP AND THEN GET THE SUBTASK BY CALLING GET ATTR prev_chosen_subtask

"""
run from Steakhouse-AI/src/bayesian/
python infer_bayesian.py

make sure to delete the user_study/log folder each run
run every permutation episode with that layout and populate Steakhouse-AI/user_study/log/0,1,2,3 with json files of the episodes, each episode is a different order of orderlist
navigate to scripts/combine_json, which will turn the json in csv and store in data/fov_traj
    
"""


from __future__ import annotations
import re, os, sys, pickle
from typing import List, Any

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pygame
from pygame.locals import *
import copy
import argparse
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, PlayerState, ObjectState
from steakhouse_ai_py.agents.steak_agent import SteakLimitVisionHumanModel, SteakGreedyHumanModel, ML_ACTION_LIST
from  steakhouse_ai_py.planners.steak_planner import SteakMediumLevelActionManager
from utils import OvercookedPygame, Logger, StudyConfig, obs_to_data, flatten_obs_data, process_collected_data
import time
import itertools, random
import numpy as np
from visualization.state_visualizer import SteakhouseStateVisualizer
from overcooked_ai_py.utils import generate_temporary_file_path, load_from_json

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 200

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False
MAX_STEPS = 20000
HUD_HEIGHT = 140
USER_STUDY_LOG = os.path.join(os.getcwd(), '../user_study/log')
TIMER, t = pygame.USEREVENT+1, 1000
VIDEO_FPS = 5
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../data')


# Add project root (one level up) so we can import your dataset helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Your project-specific vectorizer:
#   obs_to_list -> obs_list_to_1D_vec

from dataset import obs_to_list, obs_list_to_1D_vec
def obs_to_vec_default(obs: Any) -> np.ndarray:
    return np.asarray(obs_list_to_1D_vec(obs_to_list(obs)), dtype=float)


#TODO: import the correct functions
from bayes_inference import (
    select_diff_layout, loadPickle, estimate_only_three_prior_counts, add_vec_to_combined_data, process_bayes, bayes_inference, process_bayes_knn
    )


# ---------- Pre-wired data paths ----------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj/baseline"

TRAIN_FILES_DEFAULT = [
    
    os.path.join(PKL_DIR, "log_90_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5_dedup.pkl"),
    
]

#end new misha edit

def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description='Initialize configurations for a human study.')

    ### Args for the game setup ###
    parser.add_argument('--layout', type=str, default='Overcooked2_2-4',
                         help='List of tasks to be performed in the study')
    
    #TODO: make this order list automticlaly permute
    parser.add_argument('--order_list', type=str, nargs='+',
                        help='List of dishes (steak_dish, chicken_dish, steak_onion_dish, boiled_chicken_onion_dish) to serve')
    
    parser.add_argument('--total_time', type=int, default=MAX_STEPS,
                        help='Total time to given to complete the game')
    
    #new misha edit
    parser.add_argument('--no-headless', dest='headless', action='store_false', default=False, help='Run with display (not headless)')
    #end new misha edit

    # The following game config options are still undergoing construction
    # parser.add_argument('--served_in_order', type=bool, help='Complete the order list in order')
    # parser.add_argument('--single_player', type=bool, help='Single player mode: one human controlled agent collaborating with a modeled greedy agent')

    ### Args for the study ###
    parser.add_argument('--participant_id', type=int,
                        help='ID of participants in the study', default=0)
    parser.add_argument('--log_file_name', type=str,
                        default='infer_fov', help='Log file name')
    parser.add_argument('--record_video', dest='record_video',
                        action='store_true', help='Record video during replay')
    parser.add_argument('--no-record_video', dest='record_video',
                        action='store_false', help='Do not record video during replay')
    
    parser.add_argument('--rand_start', dest='rand_start',
                    action='store_true', help='Randomize staring point')
    
    parser.add_argument('--fov', type=int, default=120, help='The FOV range of the limited FOV agent')

    args = parser.parse_args()

    if args.log_file_name == '':
        args.log_file_name = '-'.join([str(args.participant_id), args.layout])

    return StudyConfig(args)


def render(state_visualizer, screen, env, max_time, score):
    kitchen = state_visualizer.render_state(
        env.state, env.mdp.terrain_mtx, hud_data=state_visualizer.default_hud_data(
            env.state, time_left=round(max(max_time - (time.time() - start_time), 0)), score=score)
    )
    
    # on top of the kitchen, render fog
    for agent in [agent1, agent2]:
        state_visualizer.render_fog(kitchen, env, agent)

    screen.blit(kitchen, (0, 0))
    pygame.display.flip()


#function that goes through the current folders and save json to pkl to ANOTHER FOLDER, including the layout name and ground truth fov number in the file title
#should be one file per (layout+fov)
#TODO: call this function after done with running the episode



if __name__ == "__main__":
    
    MANUAL_LAYOUT = "Overcooked2_2-5"   # layout to use for all episodes
    MANUAL_FOV = 179                    # ground-truth FOV for all episodes
    MAX_EPISODES = 11                   # or however many you want

    #run it for this number of episodes
    #TODO: set order list to number of permuataions, copy collect_soa.py
    #TODO: change to number of permutation episodes    
    config = initialize_config_from_args()
    config.layout_name = MANUAL_LAYOUT
    config.fov = MANUAL_FOV

    orders = list(config.world_mdp.order_list)
    
    #permute the order list using itertools.permutations, drop duplocates using dict.fromkeys as keys have to be unique
    unique_perms = list(dict.fromkeys(itertools.permutations(orders)))
    #randomly shuffle the order permutations
    random.shuffle(unique_perms)
    
    #get number of total permutations, which will be the number of episodes, but if num permutations > 11 or some number, stop it early
    #eg. change to 10 or something for less running time        
    num_eps = min(11, len(unique_perms))
    
    print(f"[orders] base: {orders} | unique perms: {len(unique_perms)} | episodes to run: {num_eps}", flush=True)
    #end new Misha edit
    
    
    for participant_id in range(num_eps):
        print(f"\n================ EPISODE {participant_id} ================\n")

        #make the folder if it didnt exist
        participant_dir = os.path.join(USER_STUDY_LOG, f"{participant_id}")
        os.makedirs(participant_dir, exist_ok=True)

        #initialize configurations
        study_config = initialize_config_from_args()
        study_config.layout_name = MANUAL_LAYOUT
        study_config.fov = MANUAL_FOV
        
        study_config.participant_id = participant_id
        study_config.log_file_name = f"infer_fov_ep{participant_id}"

        #correctly apply permutations
        study_config.world_mdp.order_list = list(unique_perms[participant_id])
        study_config.base_env.mdp.order_list = study_config.world_mdp.order_list
        
        #print statements
        print("Study Configuration Initialized:")
        print(f"Participant ID: {study_config.participant_id}")
        print(f"Layout: {study_config.layout_name}")
        print(study_config.base_env.mdp.terrain_mtx)
        print("Orders:")
        for i, task in enumerate(study_config.base_env.mdp.order_list, start=1):
            print(f"{i}. {task}")
        print(study_config.world_mdp.order_list)
        
        
        # switch to limit vision
        VISION_LIMIT = True
        #degree of vision
        VISION_BOUND = study_config.fov #study_config.fov
        # whether other agent knows visionlimitagent is vision limited
        VISION_LIMIT_AWARE = True
        EXPLORE = False
        # medium level A* search depth,
        SEARCH_DEPTH = 5
        # low level exploaration depth, 
        KB_SEARCH_DEPTH = 1
        KB_UPDATE_DELAY = 3
        KB_ACKN_PROB = False

        COUNTERS_PARAMS = {
            'start_orientations': True,
            'wait_allowed': True,
            'counter_goals': [],
            'counter_drop': study_config.world_mdp.terrain_pos_dict['X'],
            'counter_pickup': study_config.world_mdp.terrain_pos_dict['X'],
            'same_motion_goals': True,
            "enable_same_cell": True,
        }

        env = study_config.base_env
    
        # Initialize two human agent
        mlam = SteakMediumLevelActionManager(study_config.world_mdp, COUNTERS_PARAMS)

        agent1 = SteakLimitVisionHumanModel(mlam, env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, kb_ackn_prob=KB_ACKN_PROB, debug=True)
        
        # agent1 = SteakGreedyHumanModel(mlam)
        agent2 = SteakGreedyHumanModel(mlam)
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)

        agent1.init_knowledge_base(env.state)
        # agent2.init_knowledge_base(env.state)


        # Initialize logging
        logger = Logger(study_config, study_config.log_file_name,
                        agent1=agent1, agent2=agent2)
        
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode(
            (env.mdp.width * 30, env.mdp.height * 30 + HUD_HEIGHT), pygame.RESIZABLE)
        print(pygame.display.get_surface().get_size())
        # Initialize agents
        # agent1.set_agent_index(agent_idx)
        agent1.set_mdp(env.mdp)
        # agent2.set_agent_index(agent_idx+1)
        agent2.set_mdp(env.mdp)

        start_time = time.time()
        pygame.time.set_timer(TIMER, t)
        
        #----new misha edit------
        # Get the directory where this kitchen_config.json script (infer_fov.py) is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one level to reach the 'src' directory, then navigate to the config
        ds = load_from_json(os.path.join(script_dir, os.path.pardir, 
            "data", "config", "kitchen_config.json"))
        #----end new misha edit------

        # ds = load_from_json(os.path.join(os.getcwd(), "src",
        #     "data", "config", "kitchen_config.json"))
        
        test_dict = copy.deepcopy(ds)
        print(test_dict["config"])
        state_visualizer = SteakhouseStateVisualizer(
            headless=False, **test_dict["config"])
        

        #new misha edit
        diff_count = 0
        
        step_counts = 0
        logger.env = env
        score = 0
        max_time = 200
        init_time = time.time()
        done = False
        
        #done: change this to our collected data
        layout = study_config.layout_name
        files =  select_diff_layout(layout, TRAIN_FILES_DEFAULT)
        
        collected_data = loadPickle(files)
        collected_data = process_collected_data(collected_data)
        
        #turn collected data obs to like 1d vector

        num_fov_types = 3
        #done: change prior to be from our updated list
        num90, num120, num179, total = estimate_only_three_prior_counts(files)
        
        #see which prior performs better
        prior = np.array([num90, num120, num179]) / total
        #prior = np.ones(num_fov_types)*(1/num_fov_types)

        print("!!!!!!!!!!!!!!!!!", prior)
        #end 
        
        while not done and env.state.timestep < MAX_GAME_TIME:
            #study_config = initialize_config_from_args()
            
            time_now_in_milisecond = round(time.time() * 1000 - init_time *1000)

            render(state_visualizer, screen, env, max_time, score)
            time.sleep(0.1)

            logger.env = env

            player_1_action = agent1.action(env.state)[0] # limited vision human
            player_2_action = agent2.action(env.state)[0] # Greedy agent
            joint_action = (player_1_action, player_2_action)
            prev_state = env.state
            state, info = env.mdp.get_state_transition(prev_state, joint_action)

            curr_reward = sum(info["sparse_reward_by_agent"])
            score += curr_reward
            
            next_state, timestep_sparse_reward, done, info = env.step(joint_action, joint_agent_action_info =[{"1"},{"2"}])
            
            #New misha edit
            #obs_data = obs_to_data(env, joint_action, score, done)
            # pull latest medium-level intent each agent chose
            p0_subtask = getattr(agent1, "prev_chosen_subtask", None)
            p1_subtask = getattr(agent2, "prev_chosen_subtask", None)


            #['90','120','179'][1] → '120'
            #select the esimtated fov every timestep
            estimated_fov = int(['90','120','179'][np.argmax(prior)])
            agent1.set_estimated_vision_bound(estimated_fov) 
            
            
            estimated_subtask = getattr(agent1, "prev_estimated_chosen_subtask", None)
            print("estimated_subtask", estimated_subtask)
            if p0_subtask != estimated_subtask:
                diff_count += 1
            print("ESTIMATED VS REAL DIFFERENCE: ", diff_count)
            print("TOTAL STEP COUNT: ", step_counts)

            def to_id(name):
                return ML_ACTION_LIST.index(name) if (name in ML_ACTION_LIST) else None

            #get the correct current env data and turn into 1d vector
            obs_data = obs_to_data(
                env, joint_action, score, done, agent1, agent2
                
                #-----new misha edit-----
                #p0_subtask=p0_subtask,
                #p1_subtask=p1_subtask,
                #p0_intent_id=to_id(p0_subtask),
                #p1_intent_id=to_id(p1_subtask),
                #-----end misha edit-----
            )

            #end new misha edit
            
            #TAKE THE COLLECTED DATA AND SEND TO BAYES. ONLY ENV. NO SUBTASK HERE 
            obs = flatten_obs_data(obs_data)
            obs_vec = obs_list_to_1D_vec(obs_to_list(obs))
            #full_collected_data = add_vec_to_combined_data(collected_data)
            
            #add and call function to add a column to combined_data called obs_vec that containst the vec equivalient version of obs
            
            #evidence, likelihood = process_bayes(obs_vec, Action.ACTION_TO_INDEX[player_1_action], collected_data, num_fov_types, False)
            evidence, likelihood = process_bayes_knn(obs_vec, Action.ACTION_TO_INDEX[player_1_action], collected_data, num_fov_types, 40, False)
            
            #so If process_bayes_knn() sometimes returns low evidence or zero similarities, bayes DOES NOT RUN
            if evidence > 0 and likelihood.any() > 0:
                posterior = bayes_inference(prior, evidence, likelihood)
                prior = posterior
            
            
            print('prior prob =', prior)
            print('max prob id =', ['90','120', '179'][np.argmax(prior)])
            print('gt =', VISION_BOUND)
            
            step_counts += 1

            # --- New new Misha addition ---
            obs_data["estimated_subtask"] = estimated_subtask
            obs_data["estimated_fov"] = estimated_fov
            
            obs_data["bayes_prob_90"] = float(prior[0])
            obs_data["bayes_prob_120"] = float(prior[1])
            obs_data["bayes_prob_179"] = float(prior[2])
            # --- End New new Misha addition ---

            logger.episode.append(obs_data)

            if logger.video_record:
                frame_name = logger.img_name(time_now_in_milisecond/1000)
                pygame.image.save(screen, frame_name)
        
        
        #gameapp = OvercookedPygame(env, agent1, agent2, logger, gameTime=MAX_GAME_TIME)
        print("Episode complete — saving log to pickle...")
        logger.save_log_as_pickle()
        
        
        
        
        
        