# Initializes Steakhouse environments for chosen layouts, runs a greedy human model (vs. a stay agent) to roll out episodes, 
# logging per-step state features, actions/subtasks, and rewards. 
# Saves each trajectory plus metadata into a compressed .npz and copies required .layout files into 
# Overcooked-AI so the MDP can load them.

import os
import sys
import shutil
import numpy as np
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mdp.steakhouse_env import SteakhouseEnv
from mdp.steakhouse_mdp import SteakhouseGridworld
from planners.steak_planner import SteakMediumLevelActionManager
from agents.steak_agent import SteakGreedyHumanModel, ML_ACTION_LIST
from overcooked_ai_py.agents.agent import StayAgent
from overcooked_ai_py.mdp.actions import Action

class TrajectoryCollector:
    def __init__(self, layout_names=["steak"], trajectories_per_layout=20, max_steps=400):
        self.layout_names = layout_names
        self.trajectories_per_layout = trajectories_per_layout
        self.max_steps = max_steps
        
        # Initialize environments and agents for each layout
        self.environments = {}
        self.mlams = {}
        self.greedy_agents = {}
        self.other_agents = {}
        self.agent_index = 1
        for layout_name in layout_names:
            # Initialize environment
            world_mdp = SteakhouseGridworld.from_layout_name(layout_name)
            env = SteakhouseEnv.from_mdp(world_mdp, horizon=max_steps)
            
            # Setup MLAM
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
            
            # Initialize agents
            greedy_agent = SteakGreedyHumanModel(mlam)
            other_agent = StayAgent()
            
            # Set agent indices
            greedy_agent.set_agent_index(self.agent_index)
            other_agent.set_agent_index(1-self.agent_index)
            
            # Set MDP for both agents
            greedy_agent.set_mdp(env.mdp)
            other_agent.set_mdp(env.mdp)
            
            # Store everything
            self.environments[layout_name] = {
                'mdp': world_mdp,
                'env': env
            }
            self.mlams[layout_name] = mlam
            self.greedy_agents[layout_name] = greedy_agent
            self.other_agents[layout_name] = other_agent

    def get_state_features(self, state, layout_name):
        """Convert state to feature vector"""
        state_mask_dict = self.environments[layout_name]['mdp'].lossless_state_encoding(state, self.max_steps)[self.agent_index]
        return state_mask_dict.flatten()

    def collect_trajectories(self):
        """Collect trajectories using the greedy agent for each layout"""
        all_trajectories = {}
        
        for layout_name in self.layout_names:
            print(f"\nCollecting trajectories for layout: {layout_name}")
            trajectories = []
            env = self.environments[layout_name]['env']
            greedy_agent = self.greedy_agents[layout_name]
            other_agent = self.other_agents[layout_name]
            
            for traj_idx in range(self.trajectories_per_layout):
                # Reset environment
                env.reset(rand_start=True)
                trajectory = defaultdict(list)
                done = False
                episode_reward = 0
                
                while not done:
                    # Get current state
                    current_state = env.state
                    state_features = self.get_state_features(current_state, layout_name)
                    
                    # Get actions from both agents
                    greedy_action, _ = greedy_agent.action(current_state)
                    other_action = other_agent.action(current_state)[0]
                    joint_action = (other_action, greedy_action)
                    joint_action_ids = (Action.ACTION_TO_INDEX[other_action], Action.ACTION_TO_INDEX[greedy_action])
                    
                    # Store state and action
                    trajectory['states'].append(state_features)
                    trajectory['action_ids'].append(Action.ACTION_TO_INDEX[greedy_action])
                    trajectory['human_ml_actions'].append(ML_ACTION_LIST.index(greedy_agent.prev_chosen_subtask))
                    # trajectory['actions'].append(greedy_action)
                    trajectory['joint_action_ids'].append(joint_action_ids)
                    # trajectory['joint_actions'].append(joint_action)
                    trajectory['timesteps'].append(current_state.timestep)

                    if hasattr(greedy_agent, 'prev_chosen_subtask'):
                        trajectory['subtasks'].append(greedy_agent.prev_chosen_subtask)
                    
                    # Take step in environment
                    next_state, reward, done, info = env.step(joint_action)
                    episode_reward += reward
                    
                    # Store reward and info
                    trajectory['rewards'].append(reward)
                    trajectory['sparse_rewards'].append(sum(info["sparse_r_by_agent"]))
                    trajectory['shaped_rewards'].append(sum(info["shaped_r_by_agent"]))
                    
                    if done:
                        break
                
                # Convert lists to numpy arrays
                for key in trajectory:
                    trajectory[key] = np.array(trajectory[key])
                
                trajectory['episode_reward'] = episode_reward
                trajectory['layout_name'] = layout_name
                trajectories.append(trajectory)
                
                print(f"Trajectory {traj_idx + 1}/{self.trajectories_per_layout} collected. "
                      f"Episode Reward: {episode_reward:.2f}, "
                      f"Length: {len(trajectory['states'])}")
            
            all_trajectories[layout_name] = trajectories

        return all_trajectories

    def save_trajectories(self, all_trajectories, save_path="expert_trajectories.npz"):
        """Save trajectories to file"""
        # Convert trajectories to a format suitable for saving
        save_dict = {}
        
        for layout_name, trajectories in all_trajectories.items():
            for i, traj in enumerate(trajectories):
                # Convert state features from tuple to flat array
                states = [np.array(state).flatten() for state in traj['states']]
                save_dict[f"{layout_name}_trajectory_{i}"] = {k: v for k, v in traj.items()}
        
        # Add metadata
        save_dict['metadata'] = {
            'layout_names': self.layout_names,
            'trajectories_per_layout': self.trajectories_per_layout,
            'max_steps': self.max_steps
        }
        
        np.savez_compressed(save_path, **save_dict)
        print(f"Trajectories saved to {save_path}")

def setup_layouts(layout_names):
    """Setup layout files"""
    for layout_name in layout_names:
        layout_file_name = layout_name + ".layout"
        base_folder = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir))
        path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
        path_to = os.path.join(base_folder, "overcooked_ai", "src", "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)

if __name__ == "__main__":
    # Setup parameters
    layout_names = ["Overcooked1_1-4"]
    # layout_names = ["steak", "Overcooked1_1-4"]#, "Overcooked2_1-2", "Overcooked2_2-4", "Overcooked2_2-5"]
    trajectories_per_layout = 20
    max_steps = 400
    save_path = "expert_trajectories_layout.npz"
    
    # Setup layouts
    setup_layouts(layout_names)
    
    # Create collector and collect trajectories
    collector = TrajectoryCollector(
        layout_names=layout_names,
        trajectories_per_layout=trajectories_per_layout,
        max_steps=max_steps
    )
    
    all_trajectories = collector.collect_trajectories()
    collector.save_trajectories(all_trajectories, save_path)

    # Print statistics for each layout
    print("\nTrajectory Collection Statistics:")
    for layout_name, trajectories in all_trajectories.items():
        total_rewards = [t['episode_reward'] for t in trajectories]
        trajectory_lengths = [len(t['states']) for t in trajectories]
        
        print(f"\nLayout: {layout_name}")
        print(f"Average Episode Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average Trajectory Length: {np.mean(trajectory_lengths):.2f} ± {np.std(trajectory_lengths):.2f}")
        print(f"Max Episode Reward: {np.max(total_rewards):.2f}")
        print(f"Min Episode Reward: {np.min(total_rewards):.2f}")