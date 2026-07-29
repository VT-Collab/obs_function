import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import OvercookedPygame, Logger, initialize_config_from_args
from planners.steak_planner import SteakMediumLevelActionManager
from agents.steak_agent import SteakLimitVisionHumanModel, SteakGreedyHumanModel, StaySteakLimitVisionHumanModel
#from overcooked_ai_py.agents.agent import StayAgent


import itertools
import random

#first run 90 on as many environments as possible 
#combine into a huge file
#filter file to delete all redundencies 
#predict on another filtered file with one totally unseen layout

if __name__ == "__main__":
    
    #new Misha edit to make sure order list is different every time (# episode = # permutations of the order list)
    #get the order list
    config = initialize_config_from_args()
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
    
    
    
    for i in range(num_eps): #changed from a number
        study_config = initialize_config_from_args()

        #misha edit
        #pick a unique permutation based on the current episode
        study_config.world_mdp.order_list = list(unique_perms[i])
        #explicitly set the env's mdp
        study_config.base_env.mdp.order_list = study_config.world_mdp.order_list
        print(f"[orders][ep {i}] {study_config.world_mdp.order_list}", flush=True)


        # #spawn point check
        # # (Heads-up: changing rand_start AFTER this line won't change the env created inside StudyConfig)
        # # study_config.rand_start = True  # <- this does NOT affect the already-created env
        # # 1) How many valid joint spawns does this layout have?
        # vj = study_config.world_mdp.get_valid_joint_player_positions()
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("[debug] valid joint spawns:", len(vj), flush=True)
        # # 2) Which spawn did this episode actually start at?
        # print("[debug] chosen start positions:",
        #     tuple(p.position for p in study_config.base_env.state.players),
        #     "orientations:",
        #     tuple(p.orientation for p in study_config.base_env.state.players),
        #     flush=True)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # #end spawn point num check
        #end misha edit

        
        study_config.rand_start = True
        study_config.participant_id = i
        # if study_config.log_file_name == '':
        #     study_config.log_file_name = '-'.join([str(study_config.participant_id), study_config.layout_name])
        
        #new misha edit
        # AFTER (minimal)
        if study_config.log_file_name == '':
            # Ensure JSON filename encodes layout and fov so we can split later without re-running
            # Example: Overcooked2_2-4-fov90-ep3.json
            study_config.log_file_name = f"{study_config.layout_name}-fov{study_config.fov}-ep{i}"
        #end new Misha edit

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
        VISION_BOUND = study_config.fov
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
    
        # Initialize two human agent
        mlam = SteakMediumLevelActionManager(study_config.world_mdp, COUNTERS_PARAMS)

        #agent1 = SteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, kb_ackn_prob=KB_ACKN_PROB, debug=True)
        agent1 = StaySteakLimitVisionHumanModel(
            mlam,
            study_config.base_env.state,
            auto_unstuck=True,
            explore=EXPLORE,
            vision_limit=VISION_LIMIT,
            vision_bound=VISION_BOUND,        # your --fov (e.g. 90, 120, 179)
            #vision_mode="cone",
            kb_update_delay=KB_UPDATE_DELAY,  # 3 in your config
            kb_ackn_prob=KB_ACKN_PROB,        # False in your config
            debug=True,
        )
        
        # agent1 = SteakGreedyHumanModel(mlam)
        agent2 = SteakGreedyHumanModel(mlam)
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)

        agent1.init_knowledge_base(study_config.base_env.state)
        # agent2.init_knowledge_base(study_config.base_env.state)

        # Initialize logging
        logger = Logger(study_config, study_config.log_file_name,
                        agent1=agent1, agent2=agent2)
        gametime = 200
        gameapp = OvercookedPygame(study_config.base_env, agent1, agent2, logger, gameTime=gametime)
        gameapp.on_execute()


"""


/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 90  --layout Overcooked1_1-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 90  --layout Overcooked2_1-2 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 90  --layout Overcooked2_2-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 90  --layout Overcooked2_2-5 --rand_start

/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 120 --layout Overcooked1_1-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 120 --layout Overcooked2_1-2 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 120 --layout Overcooked2_2-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 120 --layout Overcooked2_2-5 --rand_start

/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 179 --layout Overcooked1_1-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 179 --layout Overcooked2_1-2 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 179 --layout Overcooked2_2-4 --rand_start
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/collect_soa_data.py --fov 179 --layout Overcooked2_2-5 --rand_start


This script runs a batch of simulated Overcooked/Steakhouse games for 150 “participants,” 
where one agent is a limited-vision model and the other is a greedy model, and logs the runs.
It sets up the environment from command-line arguments, configures agent parameters, 
and executes the game loop with OvercookedPygame while saving logs via Logger.
executes with --fov # to set fov, otherwise default is 120
--layout ...  to set layout, otherwise default is Overcooked2_2-4
how to run:
cd Steakhouse-AI/src/scripts

conda activate steakhouse-ai

first;  default is fov 120 and layout Overcooked2_2-4 (done)
python collect_soa_data.py  --fov 90 --layout Overcooked2_1-2 --rand_start

second is fov 120 but  layout is (done)
python collect_soa_data.py --fov 90 --layout Overcooked2_2-4 --rand_start


third is fov 120 but layout is (to-do)
python collect_soa_data.py --fov 90 --layout Overcooked2_2-5 --rand_start


120 FOV
1 (done)
python collect_soa_data.py  --fov 120 --layout Overcooked2_1-2 --rand_start

2 (done)
python collect_soa_data.py --fov 120 --layout Overcooked2_2-4 --rand_start

3
python collect_soa_data.py --fov 120 --layout Overcooked2_2-5 --rand_start


179 FOV
1
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python collect_soa_data.py  --fov 179 --layout Overcooked2_1-2 --rand_start

2
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python collect_soa_data.py --fov 179 --layout Overcooked2_2-4 --rand_start

3
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python collect_soa_data.py --fov 179 --layout Overcooked2_2-5 --rand_start

Trying:
python collect_soa_data.py --fov 120 --layout small1 --rand_start


=

"""