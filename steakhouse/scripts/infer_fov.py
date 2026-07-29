#SET THE VISION BOUND EVERY STEP AND THEN GET THE SUBTASK BY CALLING GET ATTR prev_chosen_subtask

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
from utils import Logger, StudyConfig, obs_to_data, flatten_obs_data, process_collected_data
import time
import numpy as np
from visualization.state_visualizer import SteakhouseStateVisualizer
from overcooked_ai_py.utils import generate_temporary_file_path, load_from_json
from bayes_inference import process_bayes, loadPickle, bayes_inference

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

#new misha edit
diff_count = 0
step_count = 0
#end new misha edit

def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description='Initialize configurations for a human study.')

    ### Args for the game setup ###
    parser.add_argument('--layout', type=str, default='Overcooked2_2-4',
                        help='List of tasks to be performed in the study')
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

if __name__ == "__main__":

    study_config = initialize_config_from_args()
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
    VISION_BOUND = 90#study_config.fov
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
    # Get the directory where this script (infer_fov.py) is located
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
    
    step_counts = 0
    logger.env = env
    score = 0
    max_time = 200
    init_time = time.time()
    done = False
    
    
    collected_data = loadPickle(os.path.join(DATA_FOLDER, "fov_traj_data.pkl"))
    collected_data = process_collected_data(collected_data)
    num_fov_types = 3
    prior = np.ones(num_fov_types)*(1/num_fov_types)

    while not done:
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
        
        #new misha edit
        agent1.set_estimated_vision_bound(1)
        estimated_subtask = getattr(agent1, "prev_estimated_chosen_subtask", None)
        print("estimated_subtask", estimated_subtask)
        if p0_subtask != estimated_subtask:
            diff_count += 1
        print("ESTIMATED VS REAL DIFFERENCE: ", diff_count)
        step_count += 1
        print("TOTAL STEP COUNT: ", step_count)
        #end new misha edit

        def to_id(name):
            return ML_ACTION_LIST.index(name) if (name in ML_ACTION_LIST) else None

        obs_data = obs_to_data(
            env, joint_action, score, done,
            #-----new misha edit-----
            #p0_subtask=p0_subtask,
            #p1_subtask=p1_subtask,
            #p0_intent_id=to_id(p0_subtask),
            #p1_intent_id=to_id(p1_subtask),
            #-----end misha edit-----
        )
        #end new misha edit
        
        
        obs = flatten_obs_data(obs_data)
        evidence, likelihood = process_bayes(obs, Action.ACTION_TO_INDEX[player_1_action], collected_data, num_fov_types)
        
        if evidence > 0 and likelihood.any() > 0:
            posterior = bayes_inference(prior, evidence, likelihood)
            prior = posterior
        
        print('prior prob =', prior)
        print('max prob id =', ['90','120', '179'][np.argmax(prior)])
        print('gt =', VISION_BOUND)
        
        
    

        step_counts += 1

        logger.episode.append(obs_data)

        if logger.video_record:
            frame_name = logger.img_name(time_now_in_milisecond/1000)
            pygame.image.save(screen, frame_name)
