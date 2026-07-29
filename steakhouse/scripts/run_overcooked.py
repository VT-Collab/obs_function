#how to run: 
#run from: 
# cd /Users/mishafu/Desktop/steakhouse/Steakhouse-AI
#script to run: 
#NOTE: the layout aa is the EXACT same for Overcooked2_2-4 EXCEPT I ADDED POT AT [3,0] and [4,0]
# /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python run_overcooked.py --layout aa --csv "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov-90.csv" --episode 0 --record_video

import os, sys, shutil, csv

#current file (run_overcooked.py)'s repo; __file__ is current file's path, and os.path.dirname is the directory containing the file
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
#src's directory 
SRC_DIR   = os.path.join(REPO_ROOT, "src")
#add src/ folder to sys.path so python can import src/utils.py; sys.path = list of directories Python interpreter search through
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# --- Imports that rely on /src being on the path ---
from utils import OvercookedPygame, initialize_config_from_args, Logger
# we'll use this to validate dish names; it is
# def dishname2ingradient(dish_name):
#     # map dish_name to its ingredient, for example, steak_onion_dish to {"ingredients" : ["meat","onion"]},
#     if dish_name == "steak_dish":
#         return {"ingredients": ["meat"]}
#     elif dish_name == "boiled_chicken_dish":
#         return {"ingredients": ["chicken"]}
#     elif dish_name == "steak_onion_dish":
#         return {"ingredients": ["meat", "onion"]}
#     elif dish_name == "boiled_chicken_onion_dish":
#         return {"ingredients": ["chicken", "onion"]}
from mdp.steakhouse_mdp import dishname2ingradient

# to map indices -> Action enums
from overcooked_ai_py.mdp.overcooked_mdp import Action  
#from overcooked_ai_py.agents.agent import StayAgent


#those are the helpers to parse the -- flags things that we pass into the argument, like layout and file path
#flag: the thing you’re looking for, e.g. "--csv"
#default: what to return if the flag isn’t provided
#take_value:
    #if True (normal): return the value after the flag
    #if False: return True/False depending on whether the flag is present
def _argv_get(flag, default=None, take_value=True):
    #get the system arguments
    #basically if the system arguments were python run.py --layout aa --csv /path/file.csv --episode 0 --record_video
    # then sys.argv would be 
    #  [
    #   "run.py",
    #   "--layout", "aa",
    #   "--csv", "/path/file.csv",
    #   "--episode", "0",
    #   "--record_video"
    # ]
    
    # 1. If the flag isn’t in sys.argv:
    #     Return default (if take_value=True)
    #     Return False (if take_value=False)

    # 2. If the flag is present:
    #     If take_value=False, return True
    #     If take_value=True, look at the next token:
    #         If it exists and doesn’t start with "--", treat it as the value and return it
    #         Otherwise, return default (flag present but no value given)
    argv = sys.argv
    if flag not in argv:
        return False if not take_value else default
    #if flag is present only
    if not take_value:
        return True
    
    #index of the flag
    i = argv.index(flag)
    
    #If there’s a token after the flag and it does not start with "--", treat that next token as the value and return it.
    if i + 1 < len(argv) and not argv[i+1].startswith("--"):
        return argv[i+1]
    return default

#remove custom flags and its value from the system argument 
def _strip_flags_with_values(flags):
    """Remove each flag in `flags` from sys.argv along with its value token (if present)."""
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in flags:
            i += 1
            if i < len(sys.argv) and not sys.argv[i].startswith("--"):
                i += 1
        else:
            new_argv.append(tok)
            i += 1
    sys.argv = new_argv

#wrappers for layout, csv, episode
def get_layout_arg(default="Overcooked2_2-4"): 
    return _argv_get("--layout", default)
def get_csv_arg(): 
    return _argv_get("--csv", None)
#gets valye after --episode (defaults to 0), then converts to int safely (if can't, do 0)
def get_episode_arg():
    val = _argv_get("--episode", "0")
    try: return int(val)
    except: return 0
    
"""
Minimum expected columns: episode, timestep, p0_action, p1_action
Returns two ordered lists of Action enums: a0_seq, a1_seq, ordered by timestep.
1. reads csv and only keep rows of the chosen episode
2. sorts these rows by timestep
3. converts p0_action and p1_action values into ACTUAL overcooked Action/Direction the env expects
4. Repturn 2 ordered lists: one for player 0's actions, one for player 1's
"""
def load_actions_from_csv(csv_path, episode_id=0):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    #open csv and make a dict per row: {'episode': '0', 'timestep': '1', ...}, etc. 
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True) #skip initial space just tolerates leading space
    
        needed = {"episode", "timestep", "p0_action", "p1_action"}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(needed)}; found {reader.fieldnames}")

        #data processing, converts episode & timestep to int, and store the p0_action and p1_action as given
        #so now it is like 
        # [
        #     {"t": 1, "a0": "2",     "a1": "0"},
        #     {"t": 2, "a0": "SPACE", "a1": "RIGHT"},
        #     {"t": 3, "a0": "5",     "a1": "3"},
        # ]
        for r in reader:
            try:
                if int(float(r["episode"])) == int(episode_id):
                    rows.append({
                        "t": int(float(r["timestep"])),  # handle '2' or '2.0'
                        "a0": r["p0_action"],
                        "a1": r["p1_action"],
                    })
            except Exception:
                continue
            
    if not rows:
        raise ValueError(f"No rows found for episode={episode_id} in {csv_path}")
    
    #sort rows, not really needed cuz csv is alr sorted
    rows.sort(key=lambda x: x["t"])

    #flips mapping from like STAY: 0 to 0: STAY, creating new dictionary
    #syntax: new_dict = {  new_key: new_value   for var1, var2 in some_dict.items()}
    #but like new_key new_value must be defined in the old dict. their names has to be defined 
    index_to_action = {idx: act for act, idx in Action.ACTION_TO_INDEX.items()}
    
    #this converts one token or row in our csv into an enum that the Overcooked environment can accept
    def to_action(token):
        try:
            #convert to int
            i = int(token)
            #look up integer in the dictionary and see its corresponding value: either stay (the default), north, etc. 
            return index_to_action.get(i, Action.STAY)
        except (ValueError, TypeError):
            pass
        
        return Action.STAY

    #prepare and eventually return the list of action enums
    a0_seq, a1_seq = [], []
    for r in rows:
        a0_seq.append(to_action(r["a0"]))
        a1_seq.append(to_action(r["a1"]))

    print(f"[runner] Loaded {len(a0_seq)} steps from episode {episode_id}")
    # Debug mapping if you want:
    # print("[debug] ACTION_TO_INDEX:", Action.ACTION_TO_INDEX)
    return a0_seq, a1_seq

#class so that the overcooked visualization script can accept as one agent
#takes in a list of actions into itself
class ScriptedAgent:
    """
    Minimal agent compatible with OvercookedPygame.
    .action(state) -> (Action, info)
    """
    def __init__(self, actions):
        #stores actions given into its own actions list
        self.actions = list(actions)
        #index pointer of the action
        self.i = 0
    def set_mdp(self, mdp):  # game calls this???????
        self.mdp = mdp
    def action(self, state): #the script also implicitely calls this at every timestep with the current state?
        #this is for if we still have scripted actions left
        if self.i < len(self.actions):
            a = self.actions[self.i]
            self.i += 1
            return (a, {})
        return (Action.STAY, {})  # after script ends, do nothing cuz we finished and just wait for overall env to return 

                
# -------------- Main --------------
if __name__ == "__main__":
    # If you get display errors in headless runs, uncomment:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    layout = get_layout_arg()
    #ensure_layout_for_studyconfig(layout)

    csv_path   = get_csv_arg()
    episode_id = get_episode_arg()
    if not csv_path:
        raise SystemExit("Please pass your data file: --csv /ABS/PATH/to/your.csv")

    # Load scripted actions first (so we can fail fast if CSV/episode is wrong)
    a0_seq, a1_seq = load_actions_from_csv(csv_path, episode_id=episode_id)

    # Remove custom flags so utils.py's argparse doesn't choke
    _strip_flags_with_values({"--csv", "--episode"})

    # Build config from remaining CLI flags (e.g., --layout, --record_video, etc.)
    config = initialize_config_from_args()

    # Use scripted agents
    agent1 = ScriptedAgent(a0_seq)
    agent2 = ScriptedAgent(a1_seq)

    # Logger that is prvided
    out_name = f"replay_ep{episode_id}"
    logger = Logger(config, filename=out_name, agent1=agent1, agent2=agent2, video_record=True)

    # Run
    game = OvercookedPygame(config.base_env, agent1, agent2, logger, gameTime=300)
    # Ensure we don't stop early
    game.max_steps = max(game.max_steps, len(a0_seq), len(a1_seq)) + 5
    game.on_execute()

    print("\nDone. Check outputs under:")
    print(f"  user_study/log/{config.participant_id}/img_{config.layout_name}/")
    print(f"  user_study/log/{config.participant_id}/{out_name}.mp4")

