from typing import Callable
import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import math
import wandb
import torch.optim as optim
from torch.distributions import Categorical
from overcooked_ai_py.mdp.actions import Direction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# from rl.e_obs_d_subtask import Encoder
# from rl.optimal_policy import OptimalPolicy

from overcooked_ai_py.agents.agent import StayAgent
from overcooked_ai_py.mdp.actions import Action
from steakhouse_ai_py.agents.steak_agent import HRLModel, SteakLimitVisionHumanModel, ML_ACTION_LIST
from steakhouse_ai_rl.util import LSTM_Agent, Utter_Agent

class BaseNotifierAgent():
    def __init__(self, utter_id_offset=2, condition_length=3, ml_action_list=ML_ACTION_LIST, hist_utter_size=10, device="cpu"):
        self.utter_id_offset = utter_id_offset
        self.condition_length = condition_length
        self.ml_action_list = ml_action_list
        self.hist_utter_size = hist_utter_size
        self.device = device
        self.reset()

    def reset(self):
        self.curr_utter = (0,0,0)
        self.hist_utter = torch.tensor(np.array([(0,0,0)]*self.hist_utter_size)).to(self.device)
        self.hist_utter_id = torch.zeros(1, self.hist_utter_size).to(self.device)

    def action(self, state):
        pass

    def utter_id_to_utter(self, utter_id):
        if utter_id == 0:
            return (0,0,0)
        if utter_id == 1:
            return (-1,0,0)
        
        remove_no_op_utter_id = utter_id - self.utter_id_offset
        
        condition = remove_no_op_utter_id // len(self.ml_action_list) # 0: now, 1: next
        subtask = remove_no_op_utter_id % len(self.ml_action_list)
        length = len(self.ml_action_list[subtask].split('_')) + (condition * self.condition_length)

        return (condition, subtask, length)
    
    def utter_to_lang(self, utter):
        if utter == (0,0,0):
            return 'no_op'

        if utter == (-1,0,0):
            return 'continue' 

        condition, subtask, length = utter
        if condition == 0:
            return self.ml_action_list[subtask]
        else:
            return 'next_' + self.ml_action_list[subtask]

    def utter_id_to_lang(self, utter_id):
        return self.utter_to_lang(self.utter_id_to_utter(utter_id))

    def update_utter(self, utter_id):
        # Update the current and previous utterances tracking
        self.curr_utter = self.utter_id_to_utter(utter_id)

        # Update the history of utterances
        self.hist_utter = torch.cat((self.hist_utter[1:], torch.tensor([self.utter_id_to_utter(utter_id)]).to(self.device)))
        self.hist_utter_id = torch.cat((self.hist_utter_id[1:], torch.tensor([utter_id]).to(self.device)))

class LangStayAgent(SteakLimitVisionHumanModel):
    def __init__(self, mlam, start_state, notifier_model, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, one_dim_obs=False, debug=False, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.notifier_model = notifier_model
        self.optimal_subtask = None
        self.one_dim_obs = one_dim_obs
        self.human_model = None
        self.prev_human_held_object = None

        super().__init__(mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, debug=debug)

    def reset(self):
        super().reset()

    def get_obs(self, state, horizon: int = 400):
        """
        """
        if self.one_dim_obs:
            return self.get_one_dim_obs(state)
        
        self.update(state)
        # self.update_kb_log()
        
        other_players = self.knowledge_base['other_player']

        counter_objects = {}
        all_objects = []
        # counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        for k, o in self.knowledge_base.items():
            if k not in ['pot_states', 'sink_states', 'chop_states', 'grill_states', 'other_player']:
                if o.position in self.mlam.mdp.get_counter_locations():
                    counter_objects[o.name] = [o.position]
                all_objects.append(o)

        base_map_features = [
            "counter_loc",
            "dirty_plate_disp_loc",
            "onion_disp_loc",
            "serve_loc",
            "grill_loc",
            "sink_loc",
            "meat_disp_loc",
            "chopping_board_loc",
        ]
        variable_map_features = [
            "onions",
            "meats",
            "dirty_plates",
            "steak_onions",
            "steak_cook_time_remaining",
            "steak_done",
            "plate_clean_time_remaining",
            "plate_cleaned",
            "garnish_chop_time_remaining",
            "garnish_chopped",
        ]
        urgency_features = ["urgency"]

        # all_objects = steakhouse_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.mlam.mdp.shape)
            layer[position] = value
            return layer

        # Ensure that primary_agent_idx layers are ordered before other_agent_idx
        # layers
        ordered_player_features = [
            f"other_player_visual",
        ] + [
            # f"human_loc",
            f"other_player_loc",
        # ] + [
            # f"human_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            # for d in Direction.ALL_DIRECTIONS
        # ] + [
        #     f"other_player_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
        #     for d in Direction.ALL_DIRECTIONS
        ] 

        DISH_TYPES = [
            "steak_dish",
            "steak_onion_dish",
        ]

        LAYERS = (
            ordered_player_features
            + base_map_features
            + variable_map_features
            + urgency_features
            + DISH_TYPES
        )
        state_mask_dict = {k: np.zeros(self.mlam.mdp.shape) for k in LAYERS}

        # MAP LAYERS
        if horizon - state.timestep < 40:
            state_mask_dict["urgency"] = np.ones(self.mlam.mdp.shape)

        for loc in self.mlam.mdp.get_counter_locations():
            state_mask_dict["counter_loc"][loc] = 1

        for loc in self.mlam.mdp.get_dirty_plate_locations():
            state_mask_dict["dirty_plate_disp_loc"][loc] = 1

        for loc in self.mlam.mdp.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1

        for loc in self.mlam.mdp.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1

        for loc in self.mlam.mdp.get_grill_locations():
            state_mask_dict["grill_loc"][loc] = 1

        for loc in self.mlam.mdp.get_sink_locations():
            state_mask_dict["sink_loc"][loc] = 1

        for loc in self.mlam.mdp.get_meat_dispenser_locations():
            state_mask_dict["meat_disp_loc"][loc] = 1

        for loc in self.mlam.mdp.get_chopping_board_locations():
            state_mask_dict["chopping_board_loc"][loc] = 1

        # Current order layers
        if state.order_list:
            # Order list is not None and there is at least one order remaining
            cur_order = state.order_list[0]
            state_mask_dict[cur_order] = np.ones(self.mlam.mdp.shape)

        # PLAYER LAYERS
        # human_orientation_idx = Direction.DIRECTION_TO_INDEX[
        #     player.orientation
        # ]
        # state_mask_dict[f"human_loc"] = make_layer(player.position, 1)
        # state_mask_dict[f"human_orientation_{human_orientation_idx}"] = (
        #     make_layer(player.position, 1)
        # )

        other_player_orientation_idx = Direction.DIRECTION_TO_INDEX[
            other_players.orientation
        ]
        state_mask_dict[f"other_player_loc"] = make_layer(other_players.position, 1)
        state_mask_dict[f"other_player_orientation_{other_player_orientation_idx}"] = (
            make_layer(other_players.position, 1)
        )

        if other_player_orientation_idx == Direction.DIRECTION_TO_INDEX[Direction.NORTH]:
            list_of_positions = [
                (max(other_players.position[0]-1, 0), other_players.position[1]),
                (other_players.position[0], other_players.position[1]),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), other_players.position[1]),
                (max(other_players.position[0]-1, 0), max(other_players.position[1]-1, 0)),
                (other_players.position[0], max(other_players.position[1]-1, 0)),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), max(other_players.position[1]-1, 0)),
            ]
        elif other_player_orientation_idx == Direction.DIRECTION_TO_INDEX[Direction.SOUTH]:
            list_of_positions = [
                (max(other_players.position[0]-1, 0), min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
                (other_players.position[0], min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
                (max(other_players.position[0]-1, 0), other_players.position[1]),
                (other_players.position[0], other_players.position[1]),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), other_players.position[1]),
            ]
        elif other_player_orientation_idx == Direction.DIRECTION_TO_INDEX[Direction.WEST]:
            list_of_positions = [
                (other_players.position[0], max(other_players.position[1]-1, 0)),
                (other_players.position[0], other_players.position[1]),
                (other_players.position[0], min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
                (max(other_players.position[0]-1, 0), max(other_players.position[1]-1, 0)),
                (max(other_players.position[0]-1, 0), other_players.position[1]),
                (max(other_players.position[0]-1, 0), min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
            ]
        elif other_player_orientation_idx == Direction.DIRECTION_TO_INDEX[Direction.EAST]:
            list_of_positions = [
                (other_players.position[0], max(other_players.position[1]-1, 0)),
                (other_players.position[0], other_players.position[1]),
                (other_players.position[0], min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), max(other_players.position[1]-1, 0)),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), other_players.position[1]),
                (min(other_players.position[0]+1, self.mlam.mdp.shape[0]-1), min(other_players.position[1]+1, self.mlam.mdp.shape[1]-1)),
            ]
        for pos in list_of_positions:
            state_mask_dict[f"other_player_visual"][pos] = 1

        # OBJECT & STATE LAYERS
        for obj in all_objects:
            if obj.name == "steak":
                # Steak is similar to boiled chicken.
                if obj.position in self.mlam.mdp.get_grill_locations():
                    state_mask_dict["steak_cook_time_remaining"] += make_layer(
                        obj.position, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["steak_done"] += make_layer(obj.position, 1)
                else:
                    state_mask_dict["steak_done"] += make_layer(obj.position, 1)

            elif obj.name == "clean_plate":
                # Cleaning plate is similar to steak and boiled chicken for
                # observation (but interact action is required to move the cleaning
                # forward unlike cooking which happens automatically).
                if obj.position in self.mlam.mdp.get_sink_locations():
                    state_mask_dict["plate_clean_time_remaining"] += make_layer(
                        obj.position, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["plate_cleaned"] += make_layer(
                            obj.position, 1
                        )
                else:
                    state_mask_dict["plate_cleaned"] += make_layer(obj.position, 1)

            elif obj.name == "garnish":
                # Cutting for garnish is similar to cleaning plate.
                if obj.position in self.mlam.mdp.get_chopping_board_locations():
                    state_mask_dict["garnish_chop_time_remaining"] += make_layer(
                        obj.position, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["garnish_chopped"] += make_layer(
                            obj.position, 1
                        )
                else:
                    state_mask_dict["garnish_chopped"] += make_layer(
                        obj.position, 1
                    )

            elif obj.name == "onion":
                state_mask_dict["onions"] += make_layer(obj.position, 1)
            elif obj.name == "meat":
                state_mask_dict["meats"] += make_layer(obj.position, 1)
            elif obj.name == "dirty_plate":
                state_mask_dict["dirty_plates"] += make_layer(obj.position, 1)
            elif obj.name == "steak_onion":
                # Garnished steak doesn't need cooking, so treated as a regular
                # object.
                state_mask_dict["steak_onions"] += make_layer(obj.position, 1)
            else:
                raise ValueError("Unrecognized object")

        # print("terrain----")
        # print(np.array(self.mlam.mdp.terrain_mtx))
        # print("-----------")
        # print(len(LAYERS))
        # print(len(state_mask_dict))
        # for k, v in state_mask_dict.items():
        #     print(k)
        #     print(np.transpose(v, (1, 0)))

        # Stack of all the state masks, order decided by order of LAYERS
        state_mask_stack = np.array(
            [state_mask_dict[layer_id] for layer_id in LAYERS]
        )
        state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
        assert state_mask_stack.shape[:2] == self.mlam.mdp.shape
        assert state_mask_stack.shape[2] == len(LAYERS)
        # NOTE: currently not including time left or order_list in featurization

        return np.array(state_mask_stack).astype(int)

    def get_one_dim_obs(self, state):
        """
        Creates a one-dimensional observation vector containing:
        - Player[0]'s location, orientation, holding object
        - Base map features' relative positions to player[0]
        - Sink, grill, and chopping board status
        - Remaining order list
        """
        self.update(state)
        self.human_model.update(state)
        
        # Get human player (player[0]) information
        human = state.players[0]
        human_pos = human.position
        human_orientation = Direction.DIRECTION_TO_INDEX[human.orientation]
        human_held_object = 0
        object_mapping = {
            "onion": 1, 
            "meat": 2, 
            "dirty_plate": 3, 
            "clean_plate": 4, 
            "garnish": 5,
            "steak": 6,
            "steak_onion": 7
        }
        if human.held_object is not None:
            # Encode held object with a number (1-9)
            human_held_object = object_mapping.get(human.held_object.name, 8)  # 8 for any other object
            
        # Get relative positions of base map features
        feature_positions = {}
        
        # Counters (just compute distance to nearest counter)
        counter_positions = self.mlam.mdp.get_counter_locations()
        nearest_counter_dist = float('inf')
        for pos in counter_positions:
            dist = abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1])  # Manhattan distance
            if dist < nearest_counter_dist:
                nearest_counter_dist = dist
        feature_positions["counter_dist"] = min(nearest_counter_dist, 7)  # Cap at 7
        
        feature_vec = []
        # For other features, find distances and directions
        # object_features = {
        #     "meat_disp": self.mlam.mdp.get_meat_dispenser_locations(),
        #     "dirty_plate_disp": self.mlam.mdp.get_dirty_plate_locations(),
        #     "onion_disp": self.mlam.mdp.get_onion_dispenser_locations(),
        #     "serve": self.mlam.mdp.get_serving_locations()
        # }

        # # For object stations, find distances and directions
        # for feature_name, positions in object_features.items():
        #     if positions:
        #         # Find nearest position of this feature
        #         nearest_pos = None
        #         nearest_dist = float('inf')
        #         for pos in positions:
        #             dist = abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1])
        #             if dist < nearest_dist:
        #                 nearest_dist = dist
        #                 nearest_pos = pos
                
        #         # Compute relative position
        #         if nearest_pos:
        #             rel_x = nearest_pos[0] - human_pos[0]
        #             rel_y = nearest_pos[1] - human_pos[1]
        #             # Add normalized x, y to feature vector
        #             feature_vec.extend([min(max(rel_x, -6), 6) / 6, min(max(rel_y, -7), 7) / 7])
        #         else:
        #             feature_vec.extend([0, 0])  # No position found
        #     else:
        #         feature_vec.extend([0, 0])  # No position found

        obj_stations_features = {
            "grill": self.mlam.mdp.get_grill_locations(),
            "chopping_board": self.mlam.mdp.get_chopping_board_locations(),
            "sink": self.mlam.mdp.get_sink_locations()
        }
        
        for feature_name, positions in obj_stations_features.items():
            if positions:
                # Find nearest position of this feature
                nearest_pos = None
                nearest_dist = float('inf')
                for pos in positions:
                    dist = abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1])
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_pos = pos
                
                # # Compute relative position
                # if nearest_pos:
                #     rel_x = nearest_pos[0] - human_pos[0]
                #     rel_y = nearest_pos[1] - human_pos[1]
                #     # Add normalized x, y to feature vector
                #     feature_vec.extend([min(max(rel_x, -6), 6) / 6, min(max(rel_y, -7), 7) / 7])
                # else:
                #     feature_vec.extend([0, 0])  # No position found

                # Add object status 
                if state.has_object(nearest_pos):
                    obj = state.get_object(nearest_pos)
                    if obj.name in ['steak', 'garnish', 'clean_plate', 'dirty_plate']:
                        cooking_tick = obj._cooking_tick / obj.cook_time
                        feature_vec.extend([1, cooking_tick])
                    else:
                        feature_vec.extend([1, 0])

                    # Add human model awareness
                    if obj.id in self.human_model.knowledge_base.keys() and self.human_model.knowledge_base[obj.id].position == nearest_pos:
                        feature_vec.extend([1])
                    else:
                        feature_vec.extend([0])
                else:
                    feature_vec.extend([0, 0, 0])

            else:
                feature_vec.extend([0, 0, 0])  # object existence, cooking tick, human awareness
        
        # # Get all objects directly from state
        # all_objects = state.all_objects_list
        
        # # Status indicators - cooking, cleaning, chopping
        # steak_cooking = 0
        # steak_done = 0
        # plate_cleaning = 0
        # plate_cleaned = 0
        # garnish_chopping = 0
        # garnish_chopped = 0
        
        # # Find items and states
        # onions_count = 0
        # meats_count = 0
        # dirty_plates_count = 0
        # steak_onions_count = 0
        
        # for obj in all_objects:
        #     if obj.name == "steak":
        #         if obj.position in self.mlam.mdp.get_grill_locations():
        #             if obj.is_ready:
        #                 steak_done = 1
        #             else:
        #                 steak_cooking = min((obj.cook_time - obj._cooking_tick) / obj.cook_time, 1.0)
        #         else:
        #             steak_done = 1
        #     elif obj.name == "clean_plate":
        #         if obj.position in self.mlam.mdp.get_sink_locations():
        #             if obj.is_ready:
        #                 plate_cleaned = 1
        #             else:
        #                 plate_cleaning = min((obj.cook_time - obj._cooking_tick) / obj.cook_time, 1.0)
        #         else:
        #             plate_cleaned = 1
        #     elif obj.name == "garnish":
        #         if obj.position in self.mlam.mdp.get_chopping_board_locations():
        #             if obj.is_ready:
        #                 garnish_chopped = 1
        #             else:
        #                 garnish_chopping = min((obj.cook_time - obj._cooking_tick) / obj.cook_time, 1.0)
        #         else:
        #             garnish_chopped = 1
        #     elif obj.name == "onion":
        #         onions_count += 1
        #     elif obj.name == "meat":
        #         meats_count += 1
        #     elif obj.name == "dirty_plate":
        #         dirty_plates_count += 1
        #     elif obj.name == "steak_onion":
        #         steak_onions_count += 1
        
        # # Add status indicators to feature vector
        # feature_vec.extend([
        #     steak_cooking, steak_done, 
        #     plate_cleaning, plate_cleaned,
        #     garnish_chopping, garnish_chopped,
        #     min(onions_count, 3) / 3,
        #     min(meats_count, 3) / 3,
        #     min(dirty_plates_count, 3) / 3,
        #     min(steak_onions_count, 3) / 3
        # ])
        
        # Encode order information
        order_encoding = 0
        if state.order_list:
            order_encoding = 1 if state.order_list[0] == 'steak' else 2  # 1 for steak, 2 for steak_onion
        
        # Combine all features
        obs_vector = [
            human_pos[0] / self.mlam.mdp.shape[0],  # Normalize position
            human_pos[1] / self.mlam.mdp.shape[1], 
            human_orientation / 4,  # Normalize orientation (0-3)
            human_held_object / 8,  # Normalize held object
            # feature_positions["counter_dist"] / 7,  # Normalized counter distance
            # order_encoding / 2  # Normalized order encoding
        ] + feature_vec
        
        return np.array(obs_vector, dtype=np.float32)

    def notification_reward(self, state, utterance=None, dense_reward=False):
        # Base case - no optimal subtask identified
        wrong_item = False
        if self.optimal_subtask is None:
            return 0, wrong_item
        
        reward = 0
        
        # Part 1: Existing penalties for wrong object handling
        human_held_object = None if state.players[0].held_object is None else state.players[0].held_object.name
        if 'pickup' in self.optimal_subtask and human_held_object is not None:
            target_object = '_'.join(self.optimal_subtask.split('_')[1:])
            if target_object == 'steak' and human_held_object not in ['clean_plate'] and self.prev_human_held_object is None:
                reward -= 5
                wrong_item = True
            elif target_object != 'steak' and target_object != human_held_object and self.prev_human_held_object is None:
                reward -= 5
                wrong_item = True
        elif self.optimal_subtask in ['chop_onion', 'rinse_plate'] and human_held_object is not None and self.prev_human_held_object is None:
            reward -= 5
            wrong_item = True
        elif self.optimal_subtask == 'add_garnish' and human_held_object not in ['steak'] and self.prev_human_held_object is None:
            reward -= 5
            wrong_item = True
        
        if reward == 0:
            # Part 2: Reward for correct object handling
            if 'pickup' in self.optimal_subtask and human_held_object is not None:
                target_object = '_'.join(self.optimal_subtask.split('_')[1:])
                if target_object == 'steak' and human_held_object in ['steak']:
                    reward += 2
                elif target_object != 'steak' and target_object == human_held_object:
                    reward += 2
            elif self.optimal_subtask == 'add_garnish' and human_held_object in ['steak_onion']:
                reward += 2
        delta_reward = 0
        if dense_reward:
            # Part 3: Distance-based rewards
            human_pos = state.players[0].position
            target_location = None
            
            # Determine target location based on subtask
            if 'pickup' in self.optimal_subtask:
                target_object = '_'.join(self.optimal_subtask.split('_')[1:])
                if target_object == 'steak':
                    # Find nearest grill location
                    grill_locations = self.mlam.mdp.get_grill_locations()
                    if grill_locations:
                        target_location = min(grill_locations, 
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                elif target_object == 'onion':
                    # Find nearest onion dispenser
                    onion_locations = self.mlam.mdp.get_onion_dispenser_locations()
                    if onion_locations:
                        target_location = min(onion_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                elif target_object == 'meat':
                    # Find nearest meat dispenser
                    meat_locations = self.mlam.mdp.get_meat_dispenser_locations()
                    if meat_locations:
                        target_location = min(meat_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                elif target_object == 'dirty_plate':
                    # Find nearest dirty plate location
                    plate_locations = self.mlam.mdp.get_dirty_plate_locations()
                    if plate_locations:
                        target_location = min(plate_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))

            elif 'drop' in self.optimal_subtask:
                target_object = '_'.join(self.optimal_subtask.split('_')[1:])
                if target_object == 'meat':
                    # Find nearest grill location
                    grill_locations = self.mlam.mdp.get_grill_locations()
                    if grill_locations:
                        target_location = min(grill_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                elif target_object == 'onion':
                    # Find nearest onion dispenser
                    chop_locations = self.mlam.mdp.get_chopping_board_locations()
                    if chop_locations:
                        target_location = min(chop_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                elif target_object == 'dirty_plate':
                    # Find nearest dirty plate location
                    sink_locations = self.mlam.mdp.get_sink_locations()
                    if sink_locations:
                        target_location = min(sink_locations,
                            key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                        
            elif self.optimal_subtask == 'chop_onion':
                # Find nearest chopping board
                chop_locations = self.mlam.mdp.get_chopping_board_locations()
                if chop_locations:
                    target_location = min(chop_locations,
                        key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))
                    
            elif self.optimal_subtask == 'rinse_plate':
                # Find nearest sink
                sink_locations = self.mlam.mdp.get_sink_locations()
                if sink_locations:
                    target_location = min(sink_locations,
                        key=lambda pos: abs(pos[0] - human_pos[0]) + abs(pos[1] - human_pos[1]))

            # Calculate distance-based reward if target location exists
            if target_location:
                # Calculate current Manhattan distance
                current_distance = abs(target_location[0] - human_pos[0]) + abs(target_location[1] - human_pos[1])
                
                # Calculate previous distance if available, otherwise use current distance
                if hasattr(self, 'prev_distance'):
                    distance_delta = self.prev_distance - current_distance
                else:
                    distance_delta = 0
                
                # Store current distance for next step
                self.prev_distance = current_distance
                
                # Normalize delta by map size for consistent rewards
                max_distance = self.mlam.mdp.shape[0] + self.mlam.mdp.shape[1]
                normalized_delta = distance_delta / max_distance
                
                # Add delta-based reward (positive delta = moving closer = better)
                delta_reward = normalized_delta

        self.prev_human_held_object = human_held_object

        return reward, wrong_item, delta_reward
    
    def tmp_notification_reward(self, state, utterance=None):
        """
        Reward function for notification that considers the risk of imminent human errors
        due to knowledge gaps, and how HumanChefAgent processes notifications.
        
        Key insights:
        1. Urgency is based on imminent risk of human picking up wrong items
        2. Short notifications when mistakes are imminent (no time for knowledge reset)
        3. Longer notifications when there's time/distance to reset knowledge
        4. HumanChefAgent has fixed reaction time but benefits from knowledge reset
        
        Args:
            state: Current game state
            utterance: Current utterance tuple (condition, subtask, length), if provided
                      
        Returns:
            float: Reward value
        """
        # Base case - no optimal subtask identified
        if self.optimal_subtask is None:
            return 0
        
        reward = 0
        
        # Part 1: Existing penalties for wrong object handling
        human_held_object = None if state.players[0].held_object is None else state.players[0].held_object.name
        if 'pickup' in self.optimal_subtask and human_held_object is not None:
            target_object = '_'.join(self.optimal_subtask.split('_')[1:])
            if target_object == 'steak' and human_held_object not in ['clean_plate'] and self.prev_human_held_object is None:
                reward -= 1
            elif target_object != 'steak' and target_object != human_held_object and self.prev_human_held_object is None:
                reward -= 1
        elif self.optimal_subtask in ['chop_onion', 'rinse_plate'] and human_held_object is not None and self.prev_human_held_object is None:
            reward -= 1
        elif self.optimal_subtask == 'add_garnish' and human_held_object not in ['steak', 'steak_onion'] and self.prev_human_held_object is None:
            reward -= 1
        
        # Get full observation to analyze game state
        obs = self.get_one_dim_obs(state)
        
        # Get notification length if provided
        notification_length = 0
        if utterance is not None and len(utterance) == 3:
            _, _, notification_length = utterance
        elif hasattr(self, 'curr_utter') and len(self.curr_utter) == 3:
            _, _, notification_length = self.curr_utter
        
        # Get information about human agent
        human = state.players[0]
        human_pos = human.position
        human_orientation = human.orientation
        
        # Analyze risk of human making wrong pickup decisions
        
        # Part 2: Calculate urgency based on imminent wrong actions
        # Assess the human's proximity to potentially wrong pickup items
        urgency = 0
        
        # First, determine what items the human should NOT interact with based on optimal subtask
        wrong_items = []
        target_item = None
        
        if 'pickup' in self.optimal_subtask:
            target_item = '_'.join(self.optimal_subtask.split('_')[1:])
            # All items except the target are wrong
            possible_items = ['onion', 'meat', 'dirty_plate', 'clean_plate', 'garnish', 'steak', 'steak_onion']
            wrong_items = [item for item in possible_items if item != target_item]
            # Special case for steak which can be picked up with a clean plate
            if target_item == 'steak' and human_held_object == 'clean_plate':
                wrong_items = [item for item in wrong_items if item != 'steak']
        elif self.optimal_subtask in ['chop_onion', 'rinse_plate']:
            # For these tasks, human shouldn't pick up anything if not already holding the right item
            if human_held_object is None:
                wrong_items = ['onion', 'meat', 'dirty_plate', 'clean_plate', 'garnish', 'steak', 'steak_onion']
        elif self.optimal_subtask == 'add_garnish':
            # Human should only pick up steak or already have steak/steak_onion
            if human_held_object not in ['steak', 'steak_onion']:
                wrong_items = ['onion', 'meat', 'dirty_plate', 'clean_plate', 'garnish']
        
        # Find closest wrong item and distance to it
        min_distance_to_wrong_item = float('inf')
        closest_wrong_item_pos = None
        
        for obj in state.all_objects_list:
            if obj.name in wrong_items:
                # Check if the object is accessible (on counter)
                if obj.position in self.mlam.mdp.get_counter_locations() or obj.position in self.mlam.mdp.get_serving_locations():
                    distance = abs(obj.position[0] - human_pos[0]) + abs(obj.position[1] - human_pos[1])
                    if distance < min_distance_to_wrong_item:
                        min_distance_to_wrong_item = distance
                        closest_wrong_item_pos = obj.position
        
        # Calculate steps to reach the wrong item (Manhattan distance + orientation change)
        steps_to_wrong_item = float('inf')
        if closest_wrong_item_pos:
            # Manhattan distance
            steps_to_wrong_item = min_distance_to_wrong_item
            
            # Add step for orientation change if needed
            if steps_to_wrong_item > 0:
                dx = closest_wrong_item_pos[0] - human_pos[0]
                dy = closest_wrong_item_pos[1] - human_pos[1]
                
                needed_orientation = None
                if abs(dx) > abs(dy):
                    needed_orientation = 'EAST' if dx > 0 else 'WEST'
                else:
                    needed_orientation = 'SOUTH' if dy > 0 else 'NORTH'
                
                if human_orientation != needed_orientation:
                    steps_to_wrong_item += 1
        
        # Set urgency based on steps to wrong item
        if steps_to_wrong_item <= 2:
            urgency = 3  # Critical - human will make wrong move very soon
        elif steps_to_wrong_item <= 4:
            urgency = 2  # High - human will make wrong move soon
        elif steps_to_wrong_item <= 6:
            urgency = 1  # Medium - human may make wrong move if not guided
        
        # # Part 3: Cooking status urgency (secondary to human mistake risk)
        # steak_cooking = obs[14]  # Progress of steak cooking (0-1)
        # steak_done = obs[15]     # Whether steak is done
        # plate_cleaning = obs[16] # Progress of plate cleaning
        # plate_cleaned = obs[17]  # Whether plate is cleaned
        # garnish_chopping = obs[18] # Progress of garnish chopping
        # garnish_chopped = obs[19]  # Whether garnish is chopped
        
        # # Boost urgency if cooking items need attention
        # if urgency < 3 and (
        #     (0.7 < steak_cooking < 1.0 and steak_done == 0) or
        #     (0.7 < plate_cleaning < 1.0 and plate_cleaned == 0) or
        #     (0.7 < garnish_chopping < 1.0 and garnish_chopped == 0)):
        #     urgency = max(urgency, 2)
            
        # Task complexity assessment - does the task require full knowledge of environment?
        knowledge_need = 0
        high_knowledge_tasks = ['pickup_clean_plate', 'pickup_steak', 'deliver_dish']
        medium_knowledge_tasks = ['drop_meat', 'drop_onion', 'add_garnish', 'rinse_plate']
        
        if self.optimal_subtask in high_knowledge_tasks:
            knowledge_need = 2  # Task that greatly benefits from knowledge reset
        elif self.optimal_subtask in medium_knowledge_tasks:
            knowledge_need = 1  # Task that moderately benefits from knowledge
        
        # Determine optimal notification length based on urgency and knowledge needs
        optimal_length = 2  # Default minimal length
        
        if urgency >= 3:
            # Imminent mistake - use shortest notification regardless of task
            optimal_length = 2
        elif urgency == 2:
            # High risk of mistake soon - shorter notifications but consider knowledge need
            if knowledge_need >= 2:
                optimal_length = 3  # Balance between speed and knowledge
            else:
                optimal_length = 2  # Prioritize speed for simple tasks
        elif urgency == 1:
            # Medium risk - balance between knowledge and speed
            if knowledge_need == 2:
                optimal_length = 4  # More knowledge for complex tasks
            elif knowledge_need == 1:
                optimal_length = 3  # Moderate knowledge for medium tasks
            else:
                optimal_length = 2  # Simple tasks need minimal info
        else:
            # Low risk - prioritize knowledge
            if knowledge_need == 2:
                optimal_length = 5  # Maximum knowledge for complex tasks
            elif knowledge_need == 1:
                optimal_length = 4  # Good knowledge for medium tasks
            else:
                optimal_length = 3  # Even simple tasks benefit from some context
        
        # Adjust reward based on notification length if there is one
        if notification_length > 0:
            # Reward for matching optimal length exactly
            if notification_length == optimal_length:
                reward += 0.5
            # Small reward for being close to optimal
            elif abs(notification_length - optimal_length) == 1:
                reward += 0.2
            # Penalty for being far from optimal length
            elif abs(notification_length - optimal_length) >= 2:
                reward -= 0.2 * abs(notification_length - optimal_length)
        
        # Additional rewards based on situation
        
        # Reward for short notifications when mistakes are imminent
        if urgency >= 2 and notification_length <= 2:
            reward += 0.4  # Strong reward for short messages in urgent situations
            
        # Reward for longer notifications for high knowledge tasks when time permits
        if knowledge_need == 2 and urgency <= 1 and notification_length >= 4:
            reward += 0.4
            
        # Penalty for very long notifications when mistakes are imminent
        if urgency >= 2 and notification_length > 3:
            reward -= 0.5  # Significant penalty for verbose messages in urgent situations
            
        # Penalty for very short notifications for complex tasks with no urgency
        if knowledge_need == 2 and urgency == 0 and notification_length < 4:
            reward -= 0.3

        self.prev_human_held_object = human_held_object
            
        return reward
    
    def optimal_humans_ml_action(self, state, vision_bound=0, vision_mode=None):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        self.update(state, vision_bound=vision_bound, vision_mode=vision_mode)
        # self.update_kb_log()
        
        # Initialize state variables
        human = state.players[1-self.agent_index] # flip to consider the human as the player
        other_players = [] # leave empty as the human does not consider other players (or there is no other players/ is a stay player)
        
        # Get kitchen state information
        kitchen_state = self._get_kitchen_state_info(state)
        
        # Get player state information
        player_state = self._get_player_state_info(human, other_players)
        
        # Determine the best subtask based on current state
        motion_goals, chosen_subtask = self._get_motion_goals_for_current_state(
            state, kitchen_state, player_state
        )
        
        # Update state and return
        return motion_goals, chosen_subtask

    def action(self, state):
    ### Action function decides to utter a language command or not. The function will directly update the utterance to self.curr_utter such that later on the rollout could register it to the next state's player's utter property.

        # Get the observation and pass into the utterance model to output the next utterance
        _, self.optimal_subtask = self.optimal_humans_ml_action(state, vision_bound=0)

        # utter_id, logprob, _, value, self.next_lstm_state = self.utter_model.get_action_and_value(torch.Tensor(obs).unsqueeze(0), self.next_lstm_state, self.hist_utter_id.unsqueeze(0), self.next_done)

        # self.update_utter(utter_id.item())

        return Action.STAY, {'optimal_subtask': self.optimal_subtask}
    
    
class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.output_dim = output_dim

        # encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True).to(device)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3).to(device)

        self.obs_seq_encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.obs_seq_encoder(x)
    

class NotifierManager(nn.Module):
    def __init__(self, latent_dim, seq_length=10, hidden_dim=128):
        super().__init__()
        self.input_dim = latent_dim + (3 * seq_length)

        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 3)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class NotifierPolicy(nn.Module):
    """
    Notifier Policy that predicts a notification action consisting of:
      - Notification ID (one of 12 categories)
      - Notification Length (integer in {2,3,4,5,6})
      
    Inputs:
      - encoder_latent: Tensor of shape (batch_size, latent_dim)
      - target_latent:  Tensor of shape (batch_size, target_latent_dim)
      
    The two latent vectors are concatenated and processed by a shared layer,
    then split into two heads:
      - Head 1 outputs logits for the notification ID (12 classes).
      - Head 2 outputs logits for the notification length (5 classes, corresponding to lengths 2–6).
    
    This network is intended to be used when the Manager has decided to "notify with new notification."
    """
    def __init__(self, latent_dim, hidden_dim=32):
        super(NotifierPolicy, self).__init__()
        self.input_dim = latent_dim*2
        # Shared hidden layer to combine both latent inputs.
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
        # Head 1: Classification for notification ID (12 possible notifications).
        self.fc_id = nn.Linear(hidden_dim, 12)
        # Head 2: Classification for notification length (5 classes representing lengths 2–6).
        self.fc_length = nn.Linear(hidden_dim, 5)

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, encoder_latent, target_latent):
        """
        Forward pass that computes logits for both notification ID and length.
        
        Args:
            encoder_latent: Tensor of shape (batch_size, latent_dim)
            target_latent:  Tensor of shape (batch_size, target_latent_dim)
            
        Returns:
            id_logits: Tensor of shape (batch_size, 12) with logits for notification ID.
            length_logits: Tensor of shape (batch_size, 5) with logits for notification length.
        """
        # Concatenate the latent vectors along the feature dimension.
        x = torch.cat([encoder_latent, target_latent], dim=-1)
        x = self.fc1(x)
        
        id_logits = self.fc_id(x)
        length_logits = self.fc_length(x)
        
        return id_logits, length_logits

    def get_notification(self, encoder_latent, target_latent):
        """
        Returns the chosen notification ID and length using greedy decoding.
        
        Args:
            encoder_latent: Tensor of shape (batch_size, latent_dim)
            target_latent:  Tensor of shape (batch_size, target_latent_dim)
            
        Returns:
            notification_id: Tensor of shape (batch_size,) with predicted notification IDs (0–11).
            notification_length: Tensor of shape (batch_size,) with predicted lengths mapped to {2,3,4,5,6}.
        """
        id_logits, length_logits = self.forward(encoder_latent, target_latent)
        
        # For each head, choose the class with the highest logit.
        notification_id = torch.argmax(id_logits, dim=-1)
        length_class = torch.argmax(length_logits, dim=-1)
        # Map the length class indices (0..4) to actual lengths (2..6).
        notification_length = length_class + 2
        
        return notification_id, notification_length

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, latent_input, id_action=None, length_action=None):
        x = self.fc1(latent_input)
        id_logits, length_logits = self.fc_id(x), self.fc_length(x)
        id_probs = Categorical(logits=id_logits)
        length_probs = Categorical(logits=length_logits)
        if id_action is None:
            id_action = id_probs.sample()
        if length_action is None:
            length_action = length_probs.sample()

        return (id_action, length_action), id_probs.log_prob(id_action), length_probs.log_prob(length_action), id_probs.entropy(), length_probs.entropy(), self.critic(latent_input)
   
    
# class HierarchicalNotifierAgent(HRLModel, BaseNotifierAgent):
#     def __init__(self, mlam, start_state, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, device="cpu", debug=False, utter_id_offset=2, condition_length=3, ml_action_list=ML_ACTION_LIST, hist_utter_size=10, hrl_training_mode="policy", **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#         self.device = device
#         self.hist_utter_size = hist_utter_size

#         super().__init__(mlam, start_state, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, device=device, debug=debug, utter_id_offset=utter_id_offset, condition_length=condition_length, ml_action_list=ml_action_list, hist_utter_size=hist_utter_size)

#         self.obs_seq_encoder = Encoder(latent_dim=self.encoded_latent_size).to(device)
#         self.optimal_policy = OptimalPolicy(latent_dim=self.encoded_latent_size).to(device)
#         self.notifier_manager = NotifierManager(latent_dim=self.encoded_latent_size, hidden_dim=32).to(device)
#         self.notification_policy = NotifierPolicy(latent_dim=self.encoded_latent_size, hidden_dim=32).to(device)

#         if hrl_training_mode == "policy":
#             self.optimizer = optim.Adam(self.notification_policy.parameters(), lr=2.5e-4, eps=1e-5)
#         elif hrl_training_mode == "cotrain":
#             self.optimizer = optim.Adam(list(self.notifier_manager.parameters()) + list(self.notification_policy.parameters()), lr=2.5e-4, eps=1e-5)


#         # Load trained models if provided
#         self._load_model_weights(device)

#         self.utter_id_offset = utter_id_offset
#         self.condition_length = condition_length
#         self.ml_action_list = ml_action_list

#         self.reset()

#     def reset(self):
#         super().reset()
#         self.curr_utter = (0,0,0)
#         self.hist_utter = torch.tensor(np.array([(0,0,0)]*self.hist_utter_size)).to(self.device)
#         self.hist_notify_mode = torch.zeros(self.hist_utter_size).to(self.device)
#         self.hist_condition = torch.zeros(self.hist_utter_size).to(self.device)
#         self.hist_length = torch.zeros(self.hist_utter_size).to(self.device)
#         self.hist_ml_action_id = torch.zeros(self.hist_utter_size).to(self.device)

#     def _load_model_weights(self, device):
#         """Helper method to load model weights from paths or state dicts"""
#         models = {
#             'obs_seq_encoder': self.obs_seq_encoder,
#             'optimal_policy': self.optimal_policy,
#             'notifier_manager': self.notifier_manager,
#             'notification_policy': self.notification_policy
#         }

#         for model_name, model in models.items():
#             if hasattr(self, f'{model_name}_path') and getattr(self, f'{model_name}_path') is not None:
#                 model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../..',getattr(self, f'{model_name}_path')), map_location=device, weights_only=True))
#                 model.eval()

#             if hasattr(self, f'{model_name}_state_dict') and getattr(self, f'{model_name}_state_dict') is not None:
#                 model.load_state_dict(getattr(self, f'{model_name}_state_dict'))
#                 model.eval()

#     def update_utter(self, notify_mode=0, condition=0, subtask=0, length=0):
#         if notify_mode == 0:
#             self.curr_utter = (0,0,0)
#         elif notify_mode == 1:
#             self.curr_utter = (condition, subtask, length)
#         elif notify_mode == 2:
#             self.curr_utter = (-1,0,0)

#         # Update the history of utterances
#         self.hist_utter = torch.cat((self.hist_utter[1:], torch.tensor([self.curr_utter]).to(self.device)))
#         self.hist_notify_mode = torch.cat((self.hist_notify_mode[1:], torch.tensor([notify_mode]).to(self.device)))
#         self.hist_condition = torch.cat((self.hist_condition[1:], torch.tensor([condition]).to(self.device)))
#         self.hist_length = torch.cat((self.hist_length[1:], torch.tensor([length]).to(self.device)))
#         self.hist_ml_action_id = torch.cat((self.hist_ml_action_id[1:], torch.tensor([subtask]).to(self.device)))

#     def action(self, state, notify_mode=None):
#         obs_encoded = self.obs_seq_encoder(state)
#         target_encoded = self.optimal_policy(obs_encoded)

#         if notify_mode is None:
#             noti_state = torch.cat((self.hist_notify_mode, self.hist_condition, self.hist_length, self.hist_ml_action_id), dim=1)
#             notify_mode, _, _, _ = self.notifier_manager.get_action_and_value(torch.cat((obs_encoded, noti_state), dim=1))

#         if notify_mode == 0: # no notification
#             condition = 0
#             length = 0
#             ml_action_id = 0
#         elif notify_mode == 1: # notify
#             condition = 0
#             notify_action = self.notification_policy(torch.cat((obs_encoded, target_encoded), dim=1))
#             length, ml_action_id = notify_action.detach().numpy().astype(int)
#         elif notify_mode == 2: # continue
#             condition = -1
#             length = 0
#             ml_action_id = 0

#         self.update_utter(notify_mode, condition, ml_action_id, length)

#         return Action.STAY, {'notify_mode': notify_mode, 'condition': condition, 'length': length, 'ml_action_id': ml_action_id}

