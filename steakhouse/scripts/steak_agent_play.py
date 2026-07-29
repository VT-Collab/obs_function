from pygame.locals import *
from steakhouse_ai_py.agents.steak_agent import SteakLimitVisionHumanModel, SteakGreedyHumanModel
from overcooked_ai_py.agents.agent import StayAgent
from steakhouse_ai_py.planners.steak_planner import SteakMediumLevelActionManager
from utils import OvercookedPygame, Logger, initialize_config_from_args

if __name__ == "__main__":
    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    if study_config.log_file_name == '':
        study_config.log_file_name = '-'.join([str(study_config.participant_id), study_config.layout_name])
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
    VISION_BOUND = 120
    # whether other agent knows visionlimitagent is vision limited
    VISION_LIMIT_AWARE = True
    VISION_MODE = "grid" #"cone"
    EXPLORE = False
    # medium level A* search depth,
    SEARCH_DEPTH = 5
    # low level exploaration depth, 
    KB_SEARCH_DEPTH = 1
    KB_UPDATE_DELAY = 1
    KB_ACKN_PROB = True

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

    agent1 = SteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, vision_mode=VISION_MODE, kb_update_delay=KB_UPDATE_DELAY, kb_ackn_prob=KB_ACKN_PROB, debug=True)
    
    # agent2 = SteakGreedyHumanModel(mlam)
    agent2 = StayAgent(mlam)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)

    agent1.init_knowledge_base(study_config.base_env.state)

    # Initialize logging
    logger = Logger(study_config, study_config.log_file_name,
                    agent1=agent1, agent2=agent2)
    gametime = 10000
    gameapp = OvercookedPygame(study_config.base_env, agent1, agent2, logger, gameTime=gametime, headless=False)
    gameapp.on_execute()
