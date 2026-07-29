import os
import torch
import random
import numpy as np
import itertools, copy
import torch.optim as optim
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import GreedyHumanModel, AgentFromPolicy
from steakhouse_ai_rl.util import NNAgent, LSTM_Agent

ML_ACTION_LIST = [
    'pickup_meat',
    'pickup_onion',
    # 'pickup_chicken',
    'pickup_dirty_plate',
    'drop_meat',
    'drop_onion',
    # 'drop_chicken',
    'drop_dirty_plate',
    'chop_onion',
    'rinse_plate',
    'pickup_clean_plate',
    'pickup_steak',
    # 'pickup_boiled_chicken',
    'add_garnish',
    'deliver_dish',
]

# class SteakGreedyHumanModel(GreedyHumanModel):
#     def __init__(self, mlam, start_state=None, drop_on_counter=False,hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
#         GreedyHumanModel.__init__(self, mlam, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)
#         self.prev_chosen_subtask = None

#         if drop_on_counter:
#             self.mlam.counter_drop = set(self.mlam.mdp.get_empty_counter_locations(start_state))
#             self.mlam.motion_planner.counter_goals = set(self.mlam.mdp.get_empty_counter_locations(start_state))

#     def _others_have_object(self, others, object_name):
#         """Returns True if any of the other players is holding the specified object"""
#         for player in others:
#             if player.has_object() and player.get_object().name == object_name:
#                 return True
#         return False

#     def _get_kitchen_state(self, state):
#         """Returns a dictionary with all relevant kitchen state information."""
#         # Get basic state information
#         counter_objects = self.mlam.mdp.get_counter_objects_dict(state)
#         sink_states = self.mlam.mdp.get_sink_states(state)
#         chopping_board_states = self.mlam.mdp.get_chopping_board_states(state)
#         pot_states_dict = self.mlam.mdp.get_pot_states(state)
#         grill_states_dict = self.mlam.mdp.get_grill_states(state)
        
#         # Get other players
#         other_players = [
#             state.players[i] for i in range(len(state.players)) if i != self.agent_index
#         ]
        
#         # Create a comprehensive state dictionary
#         kitchen_state = {
#             "counter_objects": counter_objects,
#             "sink_states": sink_states,
#             "chopping_board_states": chopping_board_states,
#             "pot_states_dict": pot_states_dict,
#             "grill_states_dict": grill_states_dict,
#             "other_players": other_players,
#             "num_orders_remaining": state.num_orders_remaining,
#             "order_list": state.order_list,
            
#             # Derived state
#             "steak_nearly_ready": len(grill_states_dict["ready"]) > 0 or len(grill_states_dict["cooking"]) > 0,
#             "chicken_nearly_ready": len(pot_states_dict["ready"]) > 0 or len(pot_states_dict["cooking"]) > 0,
#             "garnish_ready": len(chopping_board_states["ready"]) > 0,
#             "steak_ready": len(grill_states_dict["ready"]) > 0,
#             "boiled_chicken_ready": len(pot_states_dict["ready"]) > 0,
#             "chopping": len(chopping_board_states["full"]) > 0,
#             "board_empty": len(chopping_board_states["empty"]) > 0,
#             "clean_plate_ready": len(sink_states["ready"]) > 0,
#             "rinsing": len(sink_states["full"]) > 0,
#             "sink_empty": len(sink_states["empty"]) > 0,
            
#             # Other player state
#             "other_has_dirty_plate": self._others_have_object(other_players, "dirty_plate"),
#             "other_has_clean_plate": self._others_have_object(other_players, "clean_plate"),
#             "other_has_steak": self._others_have_object(other_players, "steak"),
#             "other_has_meat": self._others_have_object(other_players, "meat"),
#             "other_has_onion": self._others_have_object(other_players, "onion"),
#             "other_has_chicken": self._others_have_object(other_players, "chicken"),
#             "other_has_boiled_chicken": self._others_have_object(other_players, "boiled_chicken"),
#         }
        
#         return kitchen_state

#     def _get_available_tasks(self, kitchen_state, player, order_idx):
#         """Returns a list of available tasks with their priorities."""
#         available_tasks = []
        
#         # Define task templates with conditions and priorities
#         task_templates = [
#             # Steak dish tasks
#             {
#                 "name": "pickup_meat",
#                 "condition": lambda: (
#                     not kitchen_state["steak_nearly_ready"] and 
#                     kitchen_state["num_orders_remaining"] > 0 and 
#                     (not kitchen_state["other_has_meat"] or order_idx > 0) and
#                     not (player.has_object() and 
#                     player.get_object().name == "meat")
#                 ),
#                 "priority": 10,
#                 "get_motion_goals": lambda: self.mlam.pickup_meat_actions(kitchen_state["counter_objects"])
#             },
#             {
#                 "name": "pickup_dirty_plate",
#                 "condition": lambda: (
#                     not kitchen_state["rinsing"] and 
#                     not kitchen_state["clean_plate_ready"] and 
#                     (not kitchen_state["other_has_dirty_plate"] or order_idx > 0) and 
#                     (not kitchen_state["other_has_clean_plate"] or order_idx > 0) and
#                     not (player.has_object() and 
#                     player.get_object().name == "dirty_plate")
#                 ),
#                 "priority": 9,
#                 "get_motion_goals": lambda: self.mlam.pickup_dirty_plate_actions(kitchen_state["counter_objects"])
#             },
#             {
#                 "name": "rinse_plate",
#                 "condition": lambda: (
#                     kitchen_state["rinsing"] and 
#                     not kitchen_state["clean_plate_ready"]
#                 ),
#                 "priority": 8,
#                 "get_motion_goals": lambda: self.mlam.rinse_plate_in_sink_actions(state=None, other_players=kitchen_state["other_players"])
#             },
#             {
#                 "name": "pickup_clean_plate",
#                 "condition": lambda: (
#                     kitchen_state["steak_nearly_ready"] and 
#                     kitchen_state["clean_plate_ready"]
#                 ),
#                 "priority": 7,
#                 "get_motion_goals": lambda: self.mlam.pickup_clean_plate_from_sink_actions(
#                     kitchen_state["counter_objects"], 
#                     state=None
#                 )
#             },
#             {
#                 "name": "drop_meat",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "meat"
#                 ),
#                 "priority": 6,
#                 "get_motion_goals": lambda: self.mlam.put_meat_in_grill_actions(kitchen_state["grill_states_dict"])
#             },
#             {
#                 "name": "drop_dirty_plate",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "dirty_plate"
#                 ),
#                 "priority": 5,
#                 "get_motion_goals": lambda: self.mlam.put_dirty_plate_in_sink_actions(
#                     kitchen_state["counter_objects"], 
#                     state=None
#                 )
#             },
#             {
#                 "name": "pickup_steak",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "clean_plate" and
#                     kitchen_state["steak_ready"]
#                 ),
#                 "priority": 4,
#                 "get_motion_goals": lambda: self.mlam.pickup_steak_with_clean_plate_actions(
#                     kitchen_state["grill_states_dict"], 
#                     only_nearly_ready=True
#                 )
#             },
#             {
#                 "name": "deliver_dish",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "steak"
#                 ),
#                 "priority": 3,
#                 "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
#             },
            
#             # Steak onion dish tasks
#             {
#                 "name": "pickup_onion",
#                 "condition": lambda: (
#                     not kitchen_state["chopping"] and 
#                     not kitchen_state["garnish_ready"] and 
#                     (not kitchen_state["other_has_onion"] or order_idx > 0) and
#                     not (player.has_object() and 
#                     player.get_object().name == "onion")
#                 ),
#                 "priority": 10,
#                 "get_motion_goals": lambda: self.mlam.pickup_onion_actions(kitchen_state["counter_objects"])
#             },
#             {
#                 "name": "drop_onion",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "onion"
#                 ),
#                 "priority": 9,
#                 "get_motion_goals": lambda: self.mlam.put_onion_on_board_actions(state=None)
#             },
#             {
#                 "name": "chop_onion",
#                 "condition": lambda: (
#                     kitchen_state["chopping"] and 
#                     not kitchen_state["garnish_ready"]
#                 ),
#                 "priority": 8,
#                 "get_motion_goals": lambda: self.mlam.chop_onion_on_board_actions(state=None, other_players=kitchen_state["other_players"])
#             },
#             {
#                 "name": "add_garnish",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "steak" and
#                     kitchen_state["garnish_ready"]
#                 ),
#                 "priority": 7,
#                 "get_motion_goals": lambda: self.mlam.add_garnish_to_dish_actions(state=None)
#             },
#             {
#                 "name": "deliver_garnished_dish",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "steak_onion"
#                 ),
#                 "priority": 3,
#                 "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
#             },
            
#             # Boiled chicken dish tasks
#             {
#                 "name": "pickup_chicken",
#                 "condition": lambda: (
#                     not kitchen_state["chicken_nearly_ready"] and 
#                     kitchen_state["num_orders_remaining"] > 0 and 
#                     (not kitchen_state["other_has_chicken"] or order_idx > 0) and
#                     not (player.has_object() and 
#                     player.get_object().name == "chicken")
#                 ),
#                 "priority": 10,
#                 "get_motion_goals": lambda: self.mlam.pickup_chicken_actions(kitchen_state["counter_objects"])
#             },
#             {
#                 "name": "drop_chicken",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "chicken"
#                 ),
#                 "priority": 6,
#                 "get_motion_goals": lambda: self.mlam.put_chicken_in_pot_actions(kitchen_state["pot_states_dict"])
#             },
#             {
#                 "name": "pickup_boiled_chicken",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "clean_plate" and
#                     kitchen_state["boiled_chicken_ready"]
#                 ),
#                 "priority": 4,
#                 "get_motion_goals": lambda: self.mlam.pickup_boiled_chicken_with_clean_plate_actions(
#                     kitchen_state["pot_states_dict"], 
#                     only_nearly_ready=True
#                 )
#             },
#             {
#                 "name": "deliver_boiled_chicken",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "boiled_chicken"
#                 ),
#                 "priority": 3,
#                 "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
#             },
            
#             # Boiled chicken onion dish tasks
#             {
#                 "name": "add_garnish_to_chicken",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "boiled_chicken" and
#                     kitchen_state["garnish_ready"]
#                 ),
#                 "priority": 7,
#                 "get_motion_goals": lambda: self.mlam.add_garnish_to_dish_actions(state=None)
#             },
#             {
#                 "name": "deliver_garnished_chicken",
#                 "condition": lambda: (
#                     player.has_object() and 
#                     player.get_object().name == "boiled_chicken_onion"
#                 ),
#                 "priority": 3,
#                 "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
#             },
#         ]
        
#         # Check each task template and add available ones to the list
#         for template in task_templates:
#             if template["condition"]():
#                 available_tasks.append(template)
        
#         return available_tasks

#     def _select_task(self, available_tasks, kitchen_state, order_idx):
#         """Selects the highest priority task from available tasks."""
#         if not available_tasks:
#             return None
        
#         # Sort tasks by priority (highest first)
#         sorted_tasks = sorted(available_tasks, key=lambda x: x["priority"], reverse=True)
        
#         # Return the highest priority task
#         return sorted_tasks[0]["name"]

#     def _get_motion_goals_for_task(self, task_name, kitchen_state, player):
#         """Returns motion goals for the specified task."""
#         if task_name is None:
#             return []
        
#         # Define task-specific motion goal generation
#         task_handlers = {
#             "pickup_meat": lambda: self.mlam.pickup_meat_actions(kitchen_state["counter_objects"]),
#             "pickup_onion": lambda: self.mlam.pickup_onion_actions(kitchen_state["counter_objects"]),
#             "pickup_dirty_plate": lambda: self.mlam.pickup_dirty_plate_actions(kitchen_state["counter_objects"]),
#             "pickup_clean_plate": lambda: self.mlam.pickup_clean_plate_from_sink_actions(
#                 kitchen_state["counter_objects"], 
#                 state=None
#             ),
#             "drop_meat": lambda: self.mlam.put_meat_in_grill_actions(kitchen_state["grill_states_dict"]),
#             "drop_onion": lambda: self.mlam.put_onion_on_board_actions(state=None),
#             "drop_dirty_plate": lambda: self.mlam.put_dirty_plate_in_sink_actions(
#                 kitchen_state["counter_objects"], 
#                 state=None
#             ),
#             "rinse_plate": lambda: self.mlam.rinse_plate_in_sink_actions(state=None, other_players=kitchen_state["other_players"]),
#             "pickup_steak": lambda: self.mlam.pickup_steak_with_clean_plate_actions(
#                 kitchen_state["grill_states_dict"], 
#                 only_nearly_ready=True
#             ),
#             "deliver_dish": lambda: self.mlam.deliver_dish_actions(),
#             "add_garnish": lambda: self.mlam.add_garnish_to_dish_actions(state=None),
#             "chop_onion": lambda: self.mlam.chop_onion_on_board_actions(state=None, other_players=kitchen_state["other_players"]),
#             "pickup_chicken": lambda: self.mlam.pickup_chicken_actions(kitchen_state["counter_objects"]),
#             "drop_chicken": lambda: self.mlam.put_chicken_in_pot_actions(kitchen_state["pot_states_dict"]),
#             "pickup_boiled_chicken": lambda: self.mlam.pickup_boiled_chicken_with_clean_plate_actions(
#                 kitchen_state["pot_states_dict"], 
#                 only_nearly_ready=True
#             ),
#             "deliver_boiled_chicken": lambda: self.mlam.deliver_dish_actions(),
#             "add_garnish_to_chicken": lambda: self.mlam.add_garnish_to_dish_actions(state=None),
#             "deliver_garnished_chicken": lambda: self.mlam.deliver_dish_actions(),
#             "deliver_garnished_dish": lambda: self.mlam.deliver_dish_actions(),
#         }
        
#         if task_name in task_handlers:
#             return task_handlers[task_name]()
        
#         return []

#     def _get_fallback_action(self, state, kitchen_state, player):
#         """Returns a fallback action when no valid tasks are available."""
#         # Try to place any held object on a counter
#         if player.has_object():
#             return self.mlam.place_obj_on_counter_actions(state=state)
        
#         # Otherwise, go to the closest feature
#         return self.mlam.go_to_closest_feature_actions(player)

#     def action(self, state, return_ml_action=False):
#         if return_ml_action:
#             possible_motion_goals, ml_action = self.ml_action(state, return_ml_action)
#         else:
#             possible_motion_goals = self.ml_action(state)

#         # Once we have identified the motion goals for the medium
#         # level action we want to perform, select the one with lowest cost
#         start_pos_and_or = state.players_pos_and_or[self.agent_index]

#         chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
#             start_pos_and_or, possible_motion_goals
#         )

#         if (
#             self.ll_boltzmann_rational
#             and chosen_goal[0] == start_pos_and_or[0]
#         ):
#             chosen_action, action_probs = self.boltzmann_rational_ll_action(
#                 start_pos_and_or, chosen_goal
#             )

#         if self.auto_unstuck and self.prev_state is not None:
#             # HACK: if two agents get stuck, select an action at random that would
#             # change the player positions if the other player were not to move
#             interact_flag = (chosen_action == 'interact')
#             for p0, p1 in zip(state.players, self.prev_state.players):
#                 if p0.held_object != p1.held_object:
#                     interact_flag = True
#             if (
#                 (self.prev_state is not None)
#                 and (state.players_pos_and_or
#                 == self.prev_state.players_pos_and_or) and (not interact_flag)
#             ):
#                 if self.agent_index == 0:
#                     joint_actions = list(
#                         itertools.product(Action.ALL_ACTIONS, [Action.STAY])
#                     )
#                 elif self.agent_index == 1:
#                     joint_actions = list(
#                         itertools.product([Action.STAY], Action.ALL_ACTIONS)
#                     )
#                 else:
#                     raise ValueError("Player index not recognized")

#                 unblocking_joint_actions = []
#                 for j_a in joint_actions:
#                     new_state, _ = self.mlam.mdp.get_state_transition(
#                         state, j_a
#                     )
#                     if (
#                         new_state.player_positions
#                         != self.prev_state.player_positions
#                     ):
#                         unblocking_joint_actions.append(j_a)
#                 # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
#                 if len(unblocking_joint_actions) == 0:
#                     unblocking_joint_actions.append([Action.STAY, Action.STAY])
#                 chosen_action = unblocking_joint_actions[
#                     np.random.choice(len(unblocking_joint_actions))
#                 ][self.agent_index]
#                 action_probs = self.a_probs_from_action(chosen_action)

#         # NOTE: Assumes that calls to the action method are sequential
#         self.prev_state = state

#         if return_ml_action:
#             return chosen_action, {"action_probs": action_probs, 'ml_action': ml_action}
#         else:
#             return chosen_action, {"action_probs": action_probs}
    
#     def ml_action(self, state, return_ml_action=False):
#         """Selects a medium level action for the current state."""
#         # Get current state information
#         player = state.players[self.agent_index]
#         am = self.mlam
        
#         # Get kitchen state
#         kitchen_state = self._get_kitchen_state(state)
        
#         # Try each order in the order list
#         motion_goals = []
#         ml_action = None
        
#         for order_idx in range(len(state.order_list)):
#             # Get available tasks based on current state
#             available_tasks = self._get_available_tasks(kitchen_state, player, order_idx)
            
#             # Select the highest priority task
#             chosen_task = self._select_task(available_tasks, kitchen_state, order_idx)
            
#             # Get motion goals for the chosen task
#             task_motion_goals = self._get_motion_goals_for_task(chosen_task, kitchen_state, player)
            
#             # If we found valid motion goals, use them and break the loop
#             if len(task_motion_goals) > 0:
#                 motion_goals = task_motion_goals
#                 ml_action = chosen_task
#                 break
        
#         # If no valid motion goals were found, use a fallback action
#         if len(motion_goals) == 0:
#             motion_goals = self._get_fallback_action(state,kitchen_state, player)
#             ml_action = None
        
#         # Filter motion goals to ensure they are valid
#         motion_goals = [
#             mg
#             for mg in motion_goals
#             if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
#                 player.pos_and_or, mg
#             )
#         ]
        
#         # If still no valid motion goals, use a fallback action
#         if len(motion_goals) == 0:
#             motion_goals = self.mlam.go_to_closest_feature_actions(player)
#             ml_action = None
        
#         if return_ml_action:
#             return motion_goals, ml_action
        
#         self.prev_chosen_subtask = ml_action if ml_action is not None else self.prev_chosen_subtask
        
#         return motion_goals


class SteakGreedyHumanModel(GreedyHumanModel):
    def __init__(self, mlam, start_state=None, drop_on_counter=False,hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
        GreedyHumanModel.__init__(self, mlam, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)
        if drop_on_counter:
            self.mlam.counter_drop = set(self.mlam.mdp.get_empty_counter_locations(start_state))
            self.mlam.motion_planner.counter_goals = set(self.mlam.mdp.get_empty_counter_locations(start_state))

    
    def _others_have_object(self, others, object_name):
        """Returns True if any of the other players is holding the specified object"""
        for player in others:
            if player.has_object() and player.get_object().name == object_name:
                return True
        return False

    def _get_kitchen_state(self, state):
        """Returns a dictionary with all relevant kitchen state information."""
        # Get basic state information
        counter_objects = self.mlam.mdp.get_counter_objects_dict(state)
        sink_states = self.mlam.mdp.get_sink_states(state)
        chopping_board_states = self.mlam.mdp.get_chopping_board_states(state)
        pot_states_dict = self.mlam.mdp.get_pot_states(state)
        grill_states_dict = self.mlam.mdp.get_grill_states(state)
        
        # Get other players
        other_players = [
            state.players[i] for i in range(len(state.players)) if i != self.agent_index
        ]
        
        # Create a comprehensive state dictionary
        kitchen_state = {
            "counter_objects": counter_objects,
            "sink_states": sink_states,
            "chopping_board_states": chopping_board_states,
            "pot_states_dict": pot_states_dict,
            "grill_states_dict": grill_states_dict,
            "other_players": other_players,
            "num_orders_remaining": state.num_orders_remaining,
            "order_list": state.order_list,
            
            # Derived state
            "steak_nearly_ready": len(grill_states_dict["ready"]) > 0 or len(grill_states_dict["cooking"]) > 0,
            "chicken_nearly_ready": len(pot_states_dict["ready"]) > 0 or len(pot_states_dict["cooking"]) > 0,
            "garnish_ready": len(chopping_board_states["ready"]) > 0,
            "steak_ready": len(grill_states_dict["ready"]) > 0,
            "boiled_chicken_ready": len(pot_states_dict["ready"]) > 0,
            "chopping": len(chopping_board_states["full"]) > 0,
            "board_empty": len(chopping_board_states["empty"]) > 0,
            "clean_plate_ready": len(sink_states["ready"]) > 0,
            "rinsing": len(sink_states["full"]) > 0,
            "sink_empty": len(sink_states["empty"]) > 0,
            
            # Other player state
            "other_has_dirty_plate": self._others_have_object(other_players, "dirty_plate"),
            "other_has_clean_plate": self._others_have_object(other_players, "clean_plate"),
            "other_has_steak": self._others_have_object(other_players, "steak"),
            "other_has_meat": self._others_have_object(other_players, "meat"),
            "other_has_onion": self._others_have_object(other_players, "onion"),
            "other_has_chicken": self._others_have_object(other_players, "chicken"),
            "other_has_boiled_chicken": self._others_have_object(other_players, "boiled_chicken"),
        }
        kitchen_state["overcooked_state"] = state #MISHA CHANGE TO SATISFY LIKE FALL BACK ACTION BELOW
        return kitchen_state

    def _get_available_tasks(self, kitchen_state, player, order_idx):
        """Returns a list of available tasks with their priorities."""
        available_tasks = []
        
        # Define task templates with conditions and priorities
        task_templates = [
            # Steak dish tasks
            {
                "name": "pickup_meat",
                "condition": lambda: (
                    not kitchen_state["steak_nearly_ready"] and 
                    kitchen_state["num_orders_remaining"] > 0 and 
                    (not kitchen_state["other_has_meat"] or order_idx > 0)
                ),
                "priority": 10,
                "get_motion_goals": lambda: self.mlam.pickup_meat_actions(kitchen_state["counter_objects"])
            },
            {
                "name": "pickup_dirty_plate",
                "condition": lambda: (
                    not kitchen_state["rinsing"] and 
                    not kitchen_state["clean_plate_ready"] and 
                    (not kitchen_state["other_has_dirty_plate"] or order_idx > 0) and 
                    (not kitchen_state["other_has_clean_plate"] or order_idx > 0)
                ),
                "priority": 9,
                "get_motion_goals": lambda: self.mlam.pickup_dirty_plate_actions(kitchen_state["counter_objects"])
            },
            {
                "name": "rinse_plate",
                "condition": lambda: (
                    kitchen_state["rinsing"] and 
                    not kitchen_state["clean_plate_ready"]
                ),
                "priority": 8,
                "get_motion_goals": lambda: self.mlam.rinse_plate_in_sink_actions(state=None, other_players=kitchen_state["other_players"])
            },
            {
                "name": "pickup_clean_plate",
                "condition": lambda: (
                    kitchen_state["steak_nearly_ready"] and 
                    kitchen_state["clean_plate_ready"]
                ),
                "priority": 7,
                "get_motion_goals": lambda: self.mlam.pickup_clean_plate_from_sink_actions(
                    kitchen_state["counter_objects"], 
                    state=None
                )
            },
            {
                "name": "drop_meat",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "meat"
                ),
                "priority": 6,
                "get_motion_goals": lambda: self.mlam.put_meat_in_grill_actions(kitchen_state["grill_states_dict"])
            },
            {
                "name": "drop_dirty_plate",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "dirty_plate"
                ),
                "priority": 5,
                "get_motion_goals": lambda: self.mlam.put_dirty_plate_in_sink_actions(
                    kitchen_state["counter_objects"], 
                    state=None
                )
            },
            {
                "name": "pickup_steak",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "clean_plate" and
                    kitchen_state["steak_ready"]
                ),
                "priority": 4,
                "get_motion_goals": lambda: self.mlam.pickup_steak_with_clean_plate_actions(
                    kitchen_state["grill_states_dict"], 
                    only_nearly_ready=True
                )
            },
            {
                "name": "deliver_dish",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "steak"
                ),
                "priority": 3,
                "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
            },
            
            # Steak onion dish tasks
            {
                "name": "pickup_onion",
                "condition": lambda: (
                    not kitchen_state["chopping"] and 
                    not kitchen_state["garnish_ready"] and 
                    (not kitchen_state["other_has_onion"] or order_idx > 0)
                ),
                "priority": 10,
                "get_motion_goals": lambda: self.mlam.pickup_onion_actions(kitchen_state["counter_objects"])
            },
            {
                "name": "drop_onion",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "onion"
                ),
                "priority": 9,
                "get_motion_goals": lambda: self.mlam.put_onion_on_board_actions(state=None)
            },
            {
                "name": "chop_onion",
                "condition": lambda: (
                    kitchen_state["chopping"] and 
                    not kitchen_state["garnish_ready"]
                ),
                "priority": 8,
                "get_motion_goals": lambda: self.mlam.chop_onion_on_board_actions(state=None, other_players=kitchen_state["other_players"])
            },
            {
                "name": "add_garnish",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "steak" and
                    kitchen_state["garnish_ready"]
                ),
                "priority": 7,
                "get_motion_goals": lambda: self.mlam.add_garnish_to_dish_actions(state=None)
            },
            {
                "name": "deliver_garnished_dish",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "steak_onion"
                ),
                "priority": 3,
                "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
            },
            
            # Boiled chicken dish tasks
            {
                "name": "pickup_chicken",
                "condition": lambda: (
                    not kitchen_state["chicken_nearly_ready"] and 
                    kitchen_state["num_orders_remaining"] > 0 and 
                    (not kitchen_state["other_has_chicken"] or order_idx > 0)
                ),
                "priority": 10,
                "get_motion_goals": lambda: self.mlam.pickup_chicken_actions(kitchen_state["counter_objects"])
            },
            {
                "name": "drop_chicken",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "chicken"
                ),
                "priority": 6,
                "get_motion_goals": lambda: self.mlam.put_chicken_in_pot_actions(kitchen_state["pot_states_dict"])
            },
            {
                "name": "pickup_boiled_chicken",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "clean_plate" and
                    kitchen_state["boiled_chicken_ready"]
                ),
                "priority": 4,
                "get_motion_goals": lambda: self.mlam.pickup_boiled_chicken_with_clean_plate_actions(
                    kitchen_state["pot_states_dict"], 
                    only_nearly_ready=True
                )
            },
            {
                "name": "deliver_boiled_chicken",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "boiled_chicken"
                ),
                "priority": 3,
                "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
            },
            
            # Boiled chicken onion dish tasks
            {
                "name": "add_garnish_to_chicken",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "boiled_chicken" and
                    kitchen_state["garnish_ready"]
                ),
                "priority": 7,
                "get_motion_goals": lambda: self.mlam.add_garnish_to_dish_actions(state=None)
            },
            {
                "name": "deliver_garnished_chicken",
                "condition": lambda: (
                    player.has_object() and 
                    player.get_object().name == "boiled_chicken_onion"
                ),
                "priority": 3,
                "get_motion_goals": lambda: self.mlam.deliver_dish_actions()
            },
        ]
        
        # Check each task template and add available ones to the list
        for template in task_templates:
            if template["condition"]():
                available_tasks.append(template)
        
        return available_tasks

    def _select_task(self, available_tasks, kitchen_state, order_idx):
        """Selects the highest priority task from available tasks."""
        if not available_tasks:
            return None
        
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(available_tasks, key=lambda x: x["priority"], reverse=True)
        
        # Return the highest priority task
        return sorted_tasks[0]["name"]

    def _get_motion_goals_for_task(self, task_name, kitchen_state, player):
        """Returns motion goals for the specified task."""
        if task_name is None:
            return []
        
        # Define task-specific motion goal generation
        task_handlers = {
            "pickup_meat": lambda: self.mlam.pickup_meat_actions(kitchen_state["counter_objects"]),
            "pickup_onion": lambda: self.mlam.pickup_onion_actions(kitchen_state["counter_objects"]),
            "pickup_dirty_plate": lambda: self.mlam.pickup_dirty_plate_actions(kitchen_state["counter_objects"]),
            "pickup_clean_plate": lambda: self.mlam.pickup_clean_plate_from_sink_actions(
                kitchen_state["counter_objects"], 
                state=None
            ),
            "drop_meat": lambda: self.mlam.put_meat_in_grill_actions(kitchen_state["grill_states_dict"]),
            "drop_onion": lambda: self.mlam.put_onion_on_board_actions(state=None),
            "drop_dirty_plate": lambda: self.mlam.put_dirty_plate_in_sink_actions(
                kitchen_state["counter_objects"], 
                state=None
            ),
            "rinse_plate": lambda: self.mlam.rinse_plate_in_sink_actions(state=None, other_players=kitchen_state["other_players"]),
            "pickup_steak": lambda: self.mlam.pickup_steak_with_clean_plate_actions(
                kitchen_state["grill_states_dict"], 
                only_nearly_ready=True
            ),
            "deliver_dish": lambda: self.mlam.deliver_dish_actions(),
            "add_garnish": lambda: self.mlam.add_garnish_to_dish_actions(state=None),
            "chop_onion": lambda: self.mlam.chop_onion_on_board_actions(state=None, other_players=kitchen_state["other_players"]),
            "pickup_chicken": lambda: self.mlam.pickup_chicken_actions(kitchen_state["counter_objects"]),
            "drop_chicken": lambda: self.mlam.put_chicken_in_pot_actions(kitchen_state["pot_states_dict"]),
            "pickup_boiled_chicken": lambda: self.mlam.pickup_boiled_chicken_with_clean_plate_actions(
                kitchen_state["pot_states_dict"], 
                only_nearly_ready=True
            ),
            "deliver_boiled_chicken": lambda: self.mlam.deliver_dish_actions(),
            "add_garnish_to_chicken": lambda: self.mlam.add_garnish_to_dish_actions(state=None),
            "deliver_garnished_chicken": lambda: self.mlam.deliver_dish_actions(),
            "deliver_garnished_dish": lambda: self.mlam.deliver_dish_actions(),
        }
        
        if task_name in task_handlers:
            return task_handlers[task_name]()
        
        return []

    def _get_fallback_action(self, kitchen_state, player):
        """Returns a fallback action when no valid tasks are available."""
        # Try to place any held object on a counter
        #print("1 kitchen_state keys:", kitchen_state.keys())
        if player.has_object():
            #print("2 kitchen_state keys:", kitchen_state.keys())
            return self.mlam.place_obj_on_counter_actions(state=kitchen_state["overcooked_state"]) #MISHA CHANGE: from (state=None), which expects a dictionary to (state=kitchen_state["overcooked_state"]) after adding that key to kitchen_state
        
        # Otherwise, go to the closest feature
        return self.mlam.go_to_closest_feature_actions(player)

    def action(self, state, return_ml_action=False, vision_bound=None):
        if return_ml_action:
            possible_motion_goals, ml_action = self.ml_action(state, return_ml_action=return_ml_action, vision_bound=vision_bound)
        else:
            possible_motion_goals = self.ml_action(state, vision_bound=vision_bound)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, possible_motion_goals
        )

        if (
            self.ll_boltzmann_rational
            and chosen_goal[0] == start_pos_and_or[0]
        ):
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal
            )

        if self.auto_unstuck and self.prev_state is not None:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            interact_flag = (chosen_action == 'interact')
            for p0, p1 in zip(state.players, self.prev_state.players):
                if p0.held_object != p1.held_object:
                    interact_flag = True
            if (
                (self.prev_state is not None)
                and (state.players_pos_and_or
                == self.prev_state.players_pos_and_or) and (not interact_flag)
            ):
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(
                        state, j_a
                    )
                    if (
                        new_state.player_positions
                        != self.prev_state.player_positions
                    ):
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

        # NOTE: Assumes that calls to the action method are sequential
        self.prev_state = state
        return chosen_action, {"action_probs": action_probs}
    
    def ml_action(self, state, return_ml_action=False, vision_bound=None):
        """Selects a medium level action for the current state."""
        # Get current state information
        player = state.players[self.agent_index]
        other_players = [
            state.players[i] for i in range(len(state.players)) if i != self.agent_index
        ]

        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(state)

        sink_states = self.mlam.mdp.get_sink_states(state)
        chopping_board_states = self.mlam.mdp.get_chopping_board_states(state)
        grill_states_dict = self.mlam.mdp.get_grill_states(state)
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        ready_grill = grill_states_dict["ready"]
        cooking_grill = grill_states_dict["cooking"]
        ready_pot = pot_states_dict["ready"]
        cooking_pot = pot_states_dict["cooking"]

        # ======= World state =======
        steak_nearly_ready = len(ready_grill) > 0 or len(cooking_grill) > 0
        chicken_nearly_ready = len(ready_pot) > 0 or len(cooking_pot) > 0
        garnish_ready = len(chopping_board_states["ready"]) > 0
        steak_ready = len(grill_states_dict["ready"]) > 0
        boiled_chicken_ready = len(pot_states_dict["ready"]) > 0
        chopping = len(chopping_board_states["full"]) > 0
        board_empty = len(chopping_board_states["empty"]) > 0
        clean_plate_ready = len(sink_states["ready"]) > 0
        rinsing = len(sink_states["full"]) > 0
        sink_empty = len(sink_states["empty"]) > 0

        # ======= Considering what other players are holding =======
        other_has_dirty_plate = self._others_have_object(other_players, "dirty_plate")
        other_has_clean_plate = self._others_have_object(other_players, "clean_plate")
        other_has_steak = self._others_have_object(other_players, "steak")
        other_has_meat = self._others_have_object(other_players, "meat")
        other_has_onion = self._others_have_object(other_players, "onion")
        other_has_chicken = self._others_have_object(other_players, "chicken")
        other_has_boiled_chicken = self._others_have_object(
            other_players, "boiled_chicken"
        )

        motion_goals = []
        order_idx = 0
        while(len(motion_goals) == 0):
            curr_order = state.order_list[order_idx]
            # Naive assumption: once we start prepping for the next dish, we assume that the other agent's action no longer matter since we assume they are working on the current dish. Hence we have if conditions like "(not other_has_meat or order_idx > 0)". Also every dish needs a plate, so we assume we need the number amount of clean plates ready based on the number needed.
            clean_plate_ready = len(sink_states["ready"]) > order_idx
            rinsing = len(sink_states["full"]) > order_idx

            if curr_order == "steak_dish" and not other_has_steak:
                if not player.has_object():
                    # Grab meat
                    if (
                        not steak_nearly_ready
                        and state.num_orders_remaining > 0
                        and (not other_has_meat or order_idx > 0)
                    ):
                        motion_goals = am.pickup_meat_actions(counter_objects)
                        
                    # Grab dirty plate
                    elif (
                        not rinsing
                        and not clean_plate_ready
                        and (not other_has_dirty_plate or order_idx > 0)
                        and (not other_has_clean_plate or order_idx > 0)
                    ):
                        motion_goals = am.pickup_dirty_plate_actions(counter_objects)
                        
                    # Wash dirty plate
                    elif rinsing and not clean_plate_ready:
                        motion_goals = am.rinse_plate_in_sink_actions(state, other_players=other_players)
                        
                    # Grab clean plate
                    elif steak_nearly_ready and clean_plate_ready:
                        motion_goals = am.pickup_clean_plate_from_sink_actions(
                            counter_objects, state
                        )
                        
                    elif not steak_ready:
                        motion_goals = []
                    else:  # TODO: These situations should be handled properly. E.g., if the other agent has the plate and the steak is ready.
                        motion_goals = []
                        # raise ValueError(
                        #     f"Unexpected situation happened to Greedy Agent. DEBUG THE CODE!\n steak ready: {steak_ready}, other has plate {other_has_clean_plate or other_has_dirty_plate}"
                        # )
                else:
                    player_obj = player.get_object()

                    if player_obj.name == "meat":
                        motion_goals = am.put_meat_in_grill_actions(grill_states_dict)
                        
                    elif player_obj.name == "dirty_plate":
                        motion_goals = am.put_dirty_plate_in_sink_actions(
                            counter_objects, state
                        )
                        
                    elif player_obj.name == "clean_plate":
                        motion_goals = am.pickup_steak_with_clean_plate_actions(
                            grill_states_dict, only_nearly_ready=True
                        )
                        
                    elif player_obj.name == "steak":
                        motion_goals = am.deliver_dish_actions()
                        
                    else:
                        motion_goals = []
                        # motion_goals = am.go_to_closest_feature_actions(player)
            elif curr_order == "steak_onion_dish":
                if not player.has_object():
                    if chopping and not garnish_ready:
                        motion_goals = am.chop_onion_on_board_actions(state, other_players=other_players)
                        
                    elif not chopping and not garnish_ready and (not other_has_onion or order_idx > 0):
                        motion_goals = am.pickup_onion_actions(counter_objects)
                        
                    elif (
                        not steak_nearly_ready
                        and state.num_orders_remaining > 0
                        and (not other_has_meat or order_idx > 0)
                    ):
                        motion_goals = am.pickup_meat_actions(counter_objects)
                        
                    elif (
                        not rinsing
                        and not clean_plate_ready
                        and (not other_has_dirty_plate or order_idx > 0)
                        and (not other_has_clean_plate or order_idx > 0)
                    ):
                        motion_goals = am.pickup_dirty_plate_actions(counter_objects)
                        
                    elif rinsing and not clean_plate_ready:
                        motion_goals = am.rinse_plate_in_sink_actions(state, other_players=other_players)
                        
                    elif steak_nearly_ready and clean_plate_ready and (not other_has_clean_plate or order_idx > 0):
                        motion_goals = am.pickup_clean_plate_from_sink_actions(
                            counter_objects, state
                        )
                    elif not steak_ready:
                        motion_goals = []
                    else:  # TODO: These situations should be handled properly. E.g., if the other agent has the plate and the steak is ready.
                        motion_goals = []

                else:
                    player_obj = player.get_object()

                    if player_obj.name == "onion":
                        motion_goals = am.put_onion_on_board_actions(state)
                        
                    elif player_obj.name == "meat":
                        motion_goals = am.put_meat_in_grill_actions(grill_states_dict)
                        
                    elif player_obj.name == "dirty_plate":
                        motion_goals = am.put_dirty_plate_in_sink_actions(
                            counter_objects, state
                        )
                        
                    elif player_obj.name == "clean_plate":
                        motion_goals = am.pickup_steak_with_clean_plate_actions(
                            grill_states_dict, only_nearly_ready=True
                        )
                        
                    elif player_obj.name == "steak" and garnish_ready:
                        motion_goals = am.add_garnish_to_dish_actions(state)
                        
                    elif player_obj.name == "steak_onion":
                        motion_goals = am.deliver_dish_actions()
                        
                    else:
                        motion_goals = []
            elif curr_order == "boiled_chicken_dish" and not other_has_boiled_chicken:
                if not player.has_object():
                    if (
                        not chicken_nearly_ready
                        and state.num_orders_remaining > 0
                        and (not other_has_chicken or order_idx > 0)
                    ):
                        motion_goals = am.pickup_chicken_actions(counter_objects)
                        
                    elif (
                        not rinsing
                        and not clean_plate_ready
                        and (not other_has_dirty_plate or order_idx > 0)
                        and (not other_has_clean_plate or order_idx > 0)
                    ):
                        motion_goals = am.pickup_dirty_plate_actions(counter_objects)
                        
                    elif rinsing and not clean_plate_ready:
                        motion_goals = am.rinse_plate_in_sink_actions(state, other_players=other_players)
                        
                    elif chicken_nearly_ready and clean_plate_ready:
                        motion_goals = am.pickup_clean_plate_from_sink_actions(
                            counter_objects, state
                        )
                        
                    elif not boiled_chicken_ready:
                        motion_goals = []

                    else:  # TODO: These situations should be handled properly. E.g., if the other agent has the plate and the steak is ready.
                        motion_goals = []

                else:
                    player_obj = player.get_object()

                    if player_obj.name == "chicken":
                        motion_goals = am.put_chicken_in_pot_actions(pot_states_dict)
                        
                    elif player_obj.name == "dirty_plate":
                        motion_goals = am.put_dirty_plate_in_sink_actions(
                            counter_objects, state
                        )
                        
                    elif player_obj.name == "clean_plate":
                        motion_goals = am.pickup_boiled_chicken_with_clean_plate_actions(
                            pot_states_dict, only_nearly_ready=True
                        )
                        
                    elif player_obj.name == "boiled_chicken":
                        motion_goals = am.deliver_dish_actions()
                        
                    else:
                        motion_goals = []
            elif curr_order == "boiled_chicken_onion_dish":
                if not player.has_object():
                    if chopping and not garnish_ready:
                        motion_goals = am.chop_onion_on_board_actions(state, other_players=other_players)
                        
                    elif not chopping and not garnish_ready and (not other_has_onion or order_idx > 0):
                        motion_goals = am.pickup_onion_actions(counter_objects)
                        
                    elif (
                        not chicken_nearly_ready
                        and state.num_orders_remaining > 0
                        and (not other_has_chicken or order_idx > 0)
                    ):
                        motion_goals = am.pickup_chicken_actions(counter_objects)
                        
                    elif (
                        not rinsing
                        and not clean_plate_ready
                        and (not other_has_dirty_plate or order_idx > 0)
                        and (not other_has_clean_plate or order_idx > 0)
                    ):
                        motion_goals = am.pickup_dirty_plate_actions(counter_objects)
                        
                    elif rinsing and not clean_plate_ready:
                        motion_goals = am.rinse_plate_in_sink_actions(state, other_players=other_players)
                        
                    elif (
                        chicken_nearly_ready
                        and clean_plate_ready
                        and (not other_has_clean_plate or order_idx > 0)
                    ):
                        motion_goals = am.pickup_clean_plate_from_sink_actions(
                            counter_objects, state
                        )
                        
                    elif not boiled_chicken_ready:
                        motion_goals = []
                        
                    else:  # TODO: These situations should be handled properly. E.g., if the other agent has the plate and the steak is ready.
                        motion_goals = []
                else:
                    player_obj = player.get_object()

                    if player_obj.name == "onion":
                        motion_goals = am.put_onion_on_board_actions(state)
                        
                    elif player_obj.name == "chicken":
                        motion_goals = am.put_chicken_in_pot_actions(pot_states_dict)
                        
                    elif player_obj.name == "dirty_plate":
                        motion_goals = am.put_dirty_plate_in_sink_actions(
                            counter_objects, state
                        )
                        
                    elif player_obj.name == "clean_plate":
                        motion_goals = am.pickup_boiled_chicken_with_clean_plate_actions(
                            pot_states_dict, only_nearly_ready=True
                        )
                        
                    elif player_obj.name == "boiled_chicken" and garnish_ready:
                        motion_goals = am.add_garnish_to_dish_actions(state)
                        
                    elif player_obj.name == "boiled_chicken_onion":
                        motion_goals = am.deliver_dish_actions()
                        
                    else:
                        motion_goals = []
            else:
                motion_goals = []
            
            # Switch to complete the next order and break loop if no orders that can be complete
            if order_idx >= state.num_orders_remaining - 1:# or order_idx > 0:
                break
            order_idx += 1

        # Stay in place
        if len(motion_goals) == 0:
            # motion_goals = am.go_to_closest_feature_actions(player)
            # motion_goals = am.place_obj_on_any_empty_counter_actions(state)
            motion_goals += am.place_obj_on_counter_actions(state)

        
        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]
        
        if len(motion_goals) == 0:
            motion_goals += am.go_to_closest_feature_actions(player)

        return motion_goals

class limitVisionHumanModel(GreedyHumanModel):
    def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True, explore=False, vision_limit=True, vision_mode="cone", vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, debug=False):
        GreedyHumanModel.__init__(self, mlam, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp,
                 auto_unstuck)
        self.explore = explore
        self.vision_limit = vision_limit
        self.vision_bound = vision_bound
        self.vision_mode = vision_mode
        
        
        self.kb_update_delay_track = [{}, {}, {}]
        self.kb_update_delay = kb_update_delay
        self.kb_ackn_prob = kb_ackn_prob
        self.knowledge_base = {}
        self.debug = debug
        
        self.estimated_kb_update_delay_track = [{}, {}, {}]
        
        
        

    def init_knowledge_base(self, start_state):
        self.knowledge_base = {}
        for obj in start_state.objects.values():
            key = self.knowledge_base_key(obj)
            self.knowledge_base[key] = obj
        self.knowledge_base['pot_states'] = self.mlam.mdp.get_pot_states(start_state)
        self.knowledge_base['other_player'] = start_state.players[1 - self.agent_index]
        
        self.kb_update_delay_track[0]['human_holding'] = [None, 0, False]
        self.kb_update_delay_track[1]['other_player'] = start_state.players[1 - self.agent_index].deepcopy()
        self.kb_update_delay_track[2] = {}
        
        self.estimated_kb_update_delay_track[0]['human_holding'] = [None, 0, False]
        self.estimated_kb_update_delay_track[1]['other_player'] = start_state.players[1 - self.agent_index].deepcopy()
        self.estimated_kb_update_delay_track[2] = {}
        

    def knowledge_base_key(self, object):
        key = '_'.join((str(object.position[0]), str(object.position[1]), str(object.name)))
        return key
    
    def in_bound(self, state, loc, vision_bound=120/2, move_back=False, vision_mode="cone"):
        """
        Determines if a location is within the agent's field of view.

        vision_mode:
            - "cone" (default): uses the original cone-based vision
            - "grid": uses a 2x3 rectangle vision in front of the agent, offset as in the visualization
        """
        player = state.players[self.agent_index]
        px, py = player.position
        lx, ly = loc
        ori = Direction.DIRECTION_TO_INDEX[player.orientation]
        if vision_bound == 0:
            return True
        
        if vision_mode == "grid":
            # The agent always sees the tile it is on
            if (lx, ly) == (px, py):
                return True

            # 2x3 rectangle in front of the agent, offset as in the visualization
            if ori == 0:  # North (y decreases)
                # Covers (px-1, py), (px, py), (px+1, py) and (px-1, py-1), (px, py-1), (px+1, py-1)
                if (py - 1 <= ly <= py) and (px - 1 <= lx <= px + 1):
                    return True
            elif ori == 1:  # South (x increases)
                # Covers (px, py-1), (px, py), (px, py+1) and (px+1, py-1), (px+1, py), (px+1, py+1)
                if (px - 1 <= lx <= px + 1) and (py - 1 <= ly <= py + 1):
                    return True
            elif ori == 2:  # East (y increases)
                # Covers (px-1, py), (px, py), (px+1, py) and (px-1, py+1), (px, py+1), (px+1, py+1)
                if (py - 1 <= ly <= py + 1) and (px - 1 <= lx <= px + 1):
                    return True
            elif ori == 3:  # West (x decreases)
                # Covers (px, py-1), (px, py), (px, py+1) and (px-1, py-1), (px-1, py), (px-1, py+1)
                if (px - 1 <= lx <= px) and (py - 1 <= ly <= py + 1):
                    return True

            return False

        # --- Existing cone-based logic ---

        center_pt = [px, py]

        if ori == 0: # north
            if move_back: center_pt[1] += 1
            if ly == py and (lx == px-1 or lx == px+1): return True
            rot_angel = np.radians(180)
        elif ori == 2: # east
            if move_back: center_pt[0] -= 1
            if lx == px and (ly == py-1 or ly == py+1): return True
            rot_angel = np.radians(270)
        elif ori == 1: # south
            if move_back: center_pt[1] -= 1
            if ly == py and (lx == px-1 or lx == px+1): return True
            rot_angel = np.radians(0)
        elif ori == 3: # west
            if move_back: center_pt[0] += 1
            if lx == px and (ly == py-1 or ly == py+1): return True
            rot_angel = np.radians(90)

        c, s = np.cos(rot_angel), np.sin(rot_angel)
        R = np.array(((c, -s), (s, c)))

        shifted_loc = np.array([lx, ly]) - np.array(center_pt)
        y_flip_loc = [shifted_loc[0], shifted_loc[1]*-1]

        rot_loc = np.matmul(R, y_flip_loc)
        
        y = -np.abs(rot_loc[0]*np.cos(np.radians(vision_bound)))
        if y >= rot_loc[1]:
            return True
        
        return False

    def update(self, state):
        pass

    def get_knowledge_base(self, state):
        # right_pt, left_pt = self.get_vision_bound(state, half_bound=self.vision_bound/2)
        valid_pot_pos = []
        new_knowledge_base = self.knowledge_base.copy()
        for obj in state.objects.values():
            if self.in_bound(state, obj.position, vision_bound=self.vision_bound/2, vision_mode=self.vision_mode):
                key = self.knowledge_base_key(obj)
                new_knowledge_base[key] = obj

                # update the pot states based on the knowledge base
                if obj.name == 'soup':
                    valid_pot_pos.append(obj.position)
                    new_knowledge_base['pot_states'] = self.mlam.mdp.get_pot_states(state, pots_states_dict=self.pot_states, valid_pos=valid_pot_pos)

        # check if other player is in vision
        other_player = state.players[1 - self.agent_index]
        if self.in_bound(state, other_player.position, vision_bound=self.vision_bound/2, vision_mode=self.vision_mode):
            # print('Other agent in bound')
            new_knowledge_base['other_player'] = other_player

        return new_knowledge_base

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """

        self.update(state)
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.knowledge_base['pot_states']
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order == 'steak':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
            
            elif player_obj.name == 'meat':
                motion_goals = am.put_meat_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = am.deliver_dish_actions()

            elif player_obj.name == 'hot_plate':
                motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            if self.explore: # explore to expand the vision.
                # get four directions to explore
                for o in Direction.ALL_DIRECTIONS:
                    motion_goals.append((player.position, o))
                motion_goals.remove(player.pos_and_or)
                random.shuffle(motion_goals)
                motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)][0] # directly return on specific motion goal as the interact plan will always cost
                assert len(motion_goals) != 0
            else: # get to the closest key object location
                motion_goals = am.go_to_closest_feature_actions(player)
                motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
                assert len(motion_goals) != 0

        return motion_goals

class SteakLimitVisionHumanModel(limitVisionHumanModel, SteakGreedyHumanModel):
    
    def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, vision_mode="cone", robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, debug=False):
        
        limitVisionHumanModel.__init__(self, mlam, start_state, hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck, explore, vision_limit=vision_limit, vision_bound=vision_bound, vision_mode=vision_mode, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, debug=debug)
        
        self.robot_aware = robot_aware
        self.prev_chosen_subtask = None
        self.kb_log = []
        self.knowledge_base = {}
        self.drop_on_counter = drop_on_counter
        SteakGreedyHumanModel.__init__(self, mlam)
        if drop_on_counter:
            self.mlam.counter_drop = set(self.mlam.mdp.get_empty_counter_locations(start_state))
            self.mlam.motion_planner.counter_goals = set(self.mlam.mdp.get_empty_counter_locations(start_state))
            
        #----new misha edit----------
        self.estimated_vision_bound = 0
        self.prev_estimated_chosen_subtask = None
        self.estimated_knowledge_base = {}
        #----end new misha edit----------
        
     #----new misha edit----------
    
    # --- PASTE THIS INSIDE SteakLimitVisionHumanModel CLASS ---
    #START MISHA NEW EDIT
    def get_ground_truth_obs(self, state, horizon: int = 400):
        """
        Returns FULL (God-view) observation tensor.
        Shape: (H, W, C)
        This MUST match get_obs channel ordering.
        """

        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]

        # ---------- DEFINE LAYERS (MUST MATCH get_obs EXACTLY) ----------
        ordered_player_features = [
            "human_loc",
            "other_player_loc",
        ] + [
            f"human_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            for d in Direction.ALL_DIRECTIONS
        ] + [
            f"other_player_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            for d in Direction.ALL_DIRECTIONS
        ]

        base_map_features = [
            "counter_loc",
            "pot_loc",
            "dirty_plate_disp_loc",
            "onion_disp_loc",
            "serve_loc",
            "grill_loc",
            "chicken_disp_loc",
            "sink_loc",
            "meat_disp_loc",
            "chopping_board_loc",
        ]

        variable_map_features = [
            "onions",
            "chickens",
            "meats",
            "dirty_plates",
            "steak_onions",
            "boiled_chicken_onions",
            "chicken_cook_time_remaining",
            "chicken_done",
            "steak_cook_time_remaining",
            "steak_done",
            "plate_clean_time_remaining",
            "plate_cleaned",
            "garnish_chop_time_remaining",
            "garnish_chopped",
        ]

        urgency_features = ["urgency"]

        DISH_TYPES = [
            "steak_dish",
            "boiled_chicken_dish",
            "steak_onion_dish",
            "boiled_chicken_onion_dish",
        ]

        LAYERS = (
            ordered_player_features
            + base_map_features
            + variable_map_features
            + urgency_features
            + DISH_TYPES
        )

        H, W = self.mlam.mdp.shape
        state_mask_dict = {k: np.zeros((H, W), dtype=np.float32) for k in LAYERS}

        def make_layer(pos, value):
            layer = np.zeros((H, W), dtype=np.float32)
            layer[pos] = value
            return layer

        # ---------- STATIC MAP ----------
        for loc in self.mlam.mdp.get_counter_locations():
            state_mask_dict["counter_loc"][loc] = 1
        for loc in self.mlam.mdp.get_pot_locations():
            state_mask_dict["pot_loc"][loc] = 1
        for loc in self.mlam.mdp.get_dirty_plate_locations():
            state_mask_dict["dirty_plate_disp_loc"][loc] = 1
        for loc in self.mlam.mdp.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1
        for loc in self.mlam.mdp.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1
        for loc in self.mlam.mdp.get_grill_locations():
            state_mask_dict["grill_loc"][loc] = 1
        for loc in self.mlam.mdp.get_chicken_dispenser_locations():
            state_mask_dict["chicken_disp_loc"][loc] = 1
        for loc in self.mlam.mdp.get_sink_locations():
            state_mask_dict["sink_loc"][loc] = 1
        for loc in self.mlam.mdp.get_meat_dispenser_locations():
            state_mask_dict["meat_disp_loc"][loc] = 1
        for loc in self.mlam.mdp.get_chopping_board_locations():
            state_mask_dict["chopping_board_loc"][loc] = 1

        # ---------- ORDER ----------
        if state.order_list:
            state_mask_dict[state.order_list[0]][:] = 1

        # ---------- URGENCY ----------
        if horizon - state.timestep < 40:
            state_mask_dict["urgency"][:] = 1

        # ---------- PLAYERS ----------
        h_ori = Direction.DIRECTION_TO_INDEX[player.orientation]
        o_ori = Direction.DIRECTION_TO_INDEX[other_player.orientation]

        state_mask_dict["human_loc"] = make_layer(player.position, 1)
        state_mask_dict[f"human_orientation_{h_ori}"] = make_layer(player.position, 1)

        state_mask_dict["other_player_loc"] = make_layer(other_player.position, 1)
        state_mask_dict[f"other_player_orientation_{o_ori}"] = make_layer(other_player.position, 1)

        # ---------- OBJECTS (GROUND TRUTH) ----------
        for obj in state.all_objects_list:
            pos = obj.position

            if obj.name == "onion":
                state_mask_dict["onions"] += make_layer(pos, 1)
            elif obj.name == "chicken":
                state_mask_dict["chickens"] += make_layer(pos, 1)
            elif obj.name == "meat":
                state_mask_dict["meats"] += make_layer(pos, 1)
            elif obj.name == "dirty_plate":
                state_mask_dict["dirty_plates"] += make_layer(pos, 1)
            elif obj.name == "steak_onion":
                state_mask_dict["steak_onions"] += make_layer(pos, 1)
            elif obj.name == "boiled_chicken_onion":
                state_mask_dict["boiled_chicken_onions"] += make_layer(pos, 1)

            elif obj.name == "steak":
                if pos in self.mlam.mdp.get_grill_locations():
                    state_mask_dict["steak_cook_time_remaining"] += make_layer(
                        pos, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["steak_done"] += make_layer(pos, 1)
                else:
                    state_mask_dict["steak_done"] += make_layer(pos, 1)

            elif obj.name == "boiled_chicken":
                if pos in self.mlam.mdp.get_pot_locations():
                    state_mask_dict["chicken_cook_time_remaining"] += make_layer(
                        pos, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["chicken_done"] += make_layer(pos, 1)
                else:
                    state_mask_dict["chicken_done"] += make_layer(pos, 1)

            elif obj.name == "clean_plate":
                if pos in self.mlam.mdp.get_sink_locations():
                    state_mask_dict["plate_clean_time_remaining"] += make_layer(
                        pos, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["plate_cleaned"] += make_layer(pos, 1)
                else:
                    state_mask_dict["plate_cleaned"] += make_layer(pos, 1)

            elif obj.name == "garnish":
                if pos in self.mlam.mdp.get_chopping_board_locations():
                    state_mask_dict["garnish_chop_time_remaining"] += make_layer(
                        pos, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["garnish_chopped"] += make_layer(pos, 1)
                else:
                    state_mask_dict["garnish_chopped"] += make_layer(pos, 1)

        # ---------- STACK ----------
        stacked = np.stack([state_mask_dict[k] for k in LAYERS], axis=-1)
        return stacked.astype(np.float32)

    #END MISHA NEW EDIT
    
    
    
    
    def get_obs(self, state, horizon: int = 400):
        """
        Partial (limited-vision) 3D observation.
        This reuses the HRLModel implementation.
        """
        # Use HRLModel's get_obs logic
        return HRLModel.get_obs(self, state, horizon)

    
    #call this to modify every step with bayes estimate fov 
    def set_estimated_vision_bound(self, fov):
        self.estimated_vision_bound = fov
    #----end new misha edit----------
        
    def deepcopy(self, world_state):
        new_human_model = SteakLimitVisionHumanModel(self.mlam, world_state, auto_unstuck=self.auto_unstuck, explore=self.explore, vision_limit=self.vision_limit, vision_mode=self.vision_mode)
        
        for k, v in self.knowledge_base.items():
            new_human_model.knowledge_base[k] = v.deepcopy()
            
        for k, v in self.estimated_knowledge_base.items():
            new_human_model.estimated_knowledge_base[k] = v.deepcopy()

        return new_human_model
    
    def kb_track_deepcopy(self, kb_track):
        new_kb_track = {}
        for k, v in kb_track.items():
            if type(v) == list:
                if v[0] is not None and v[0].position is not None:
                    new_kb_track[k] = [v[0].deepcopy(), v[1], v[2]]
                else:
                    new_kb_track[k] = [None, v[1], v[2]]
            else:
                new_kb_track[k] = v.deepcopy()
        return new_kb_track

    def _others_have_object(self, others, object_name):
        """Returns True if any of the other players is holding the specified object"""
        for player in others:
            if player.has_object() and player.get_object().name == object_name:
                return True
        return False
    
    def init_knowledge_base(self, start_state):
        self.knowledge_base = {}
        
        # First, add all objects from the start state to the knowledge base
        for obj in start_state.all_objects_list:
            self.knowledge_base[obj.id] = obj
        
        # Then, get the states of different objects and add them to the knowledge base
        self.knowledge_base['pot_states'] = self.mlam.mdp.get_pot_states(start_state, update_knowledge_base=True)
        self.knowledge_base['sink_states'] = self.mlam.mdp.get_sink_states(start_state, update_knowledge_base=True)
        self.knowledge_base['chop_states'] = self.mlam.mdp.get_chopping_board_states(start_state, update_knowledge_base=True)
        self.knowledge_base['grill_states'] = self.mlam.mdp.get_grill_states(start_state, update_knowledge_base=True)
        self.knowledge_base['other_player'] = start_state.players[1 - self.agent_index]

        # Initialize the knowledge base update delay tracking
        self.kb_update_delay_track[0]['human_holding'] = [None, self.kb_update_delay, False]
        self.kb_update_delay_track[1]['other_player'] = start_state.players[1 - self.agent_index].deepcopy()
        self.kb_update_delay_track[2] = {}
        
        # Verify that all objects referenced in chop_states, grill_states, etc. exist in the knowledge base
        for state_type in ['chop_states', 'grill_states', 'pot_states', 'sink_states']:
            if state_type in self.knowledge_base:
                for category in self.knowledge_base[state_type]:
                    for obj_id in self.knowledge_base[state_type][category]:
                        if obj_id not in self.knowledge_base and isinstance(obj_id, int):
                            # If the object ID is not in the knowledge base, try to find it in the state
                            for obj in start_state.all_objects_list:
                                if obj.id == obj_id:
                                    self.knowledge_base[obj_id] = obj
                                    break
                                
        #----new misha edit----------
        self.estimated_knowledge_base = {}
        
        # First, add all objects from the start state to the knowledge base
        for obj in start_state.all_objects_list:
            self.estimated_knowledge_base[obj.id] = obj
        
        # Then, get the states of different objects and add them to the knowledge base
        self.estimated_knowledge_base['pot_states'] = self.mlam.mdp.get_pot_states(start_state, update_knowledge_base=True)
        self.estimated_knowledge_base['sink_states'] = self.mlam.mdp.get_sink_states(start_state, update_knowledge_base=True)
        self.estimated_knowledge_base['chop_states'] = self.mlam.mdp.get_chopping_board_states(start_state, update_knowledge_base=True)
        self.estimated_knowledge_base['grill_states'] = self.mlam.mdp.get_grill_states(start_state, update_knowledge_base=True)
        self.estimated_knowledge_base['other_player'] = start_state.players[1 - self.agent_index]

        self.estimated_kb_update_delay_track[0]['human_holding'] = [None, self.kb_update_delay, False]
        self.estimated_kb_update_delay_track[1]['other_player'] = start_state.players[1 - self.agent_index].deepcopy()
        self.estimated_kb_update_delay_track[2] = {}
        
        # Verify that all objects referenced in chop_states, grill_states, etc. exist in the knowledge base
        for state_type in ['chop_states', 'grill_states', 'pot_states', 'sink_states']:
            if state_type in self.estimated_knowledge_base:
                for category in self.estimated_knowledge_base[state_type]:
                    for obj_id in self.estimated_knowledge_base[state_type][category]:
                        if obj_id not in self.estimated_knowledge_base and isinstance(obj_id, int):
                            # If the object ID is not in the knowledge base, try to find it in the state
                            for obj in start_state.all_objects_list:
                                if obj.id == obj_id:
                                    self.estimated_knowledge_base[obj_id] = obj
                                    break
        
        #----end new misha edit----------



    def kb_to_state_info(self, kb):
        num_item_in_pot = 0
        pots = kb['pot_states']['steak']
        non_emtpy_pots = pots['cooking'] + pots['ready']
        if len(non_emtpy_pots) > 0:
            num_item_in_pot = 1
        
        chop_time = -1
        non_empty_boards = kb['chop_states']['ready'] + kb['chop_states']['full']
        if len(non_empty_boards) > 0:
            if kb[non_empty_boards[0]] is not None:
                chop_time = kb[non_empty_boards[0]].state
            else:
                raise ValueError()
        
        wash_time = -1
        non_empty_sink = kb['sink_states']['ready'] + kb['sink_states']['full']
        if len(non_empty_sink) > 0:
            if kb[non_empty_sink[0]] is not None:
                wash_time = kb[non_empty_sink[0]].state
            else:
                raise ValueError()

        robot_obj = kb['other_player'].held_object.name if kb['other_player'].held_object is not None else 'None'

        return num_item_in_pot, chop_time, wash_time, robot_obj

    # def update_kb_log(self):
    #     self.kb_log += [self.get_kb_key(self.knowledge_base)]
        
        
        
    def get_kb_key(self, kb):
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        kb_key = '.'.join([str(num_item_in_pot), str(chop_time), str(wash_time), str(robot_obj)])
        return kb_key
    
    def get_knowledge_base(self, state, rollout_kb=None, rollout_info=[]):
        return self.update(state, rollout_kb=rollout_kb, rollout_info=rollout_info)

    def update_kb_key_object(self, obj, tmp_kb, prev_kb):
        if 'steak' in obj.name:
            if obj.position in self.mlam.mdp.get_grill_locations():
                cooking_time = obj._cooking_tick
                if cooking_time < obj._cook_time:
                    if obj.position in tmp_kb['grill_states']['empty']:
                        tmp_kb['grill_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['grill_states']['cooking']:
                        tmp_kb['grill_states']['cooking'].append(obj.id)
                    if obj.id in tmp_kb['grill_states']['ready']:
                        tmp_kb['grill_states']['ready'].remove(obj.id)
                else:
                    if obj.position in tmp_kb['grill_states']['empty']:
                        tmp_kb['grill_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['grill_states']['cooking']:
                        tmp_kb['grill_states']['cooking'].remove(obj.id)
                    if obj.id not in tmp_kb['grill_states']['ready']:
                        tmp_kb['grill_states']['ready'].append(obj.id)
            else:
                if obj.id in prev_kb.keys():
                    # NOTE: this only works if we have one pot (pre ICRA submission)
                    # empty_grill_loc = self.mlam.mdp.get_grill_locations()[0] # prev_kb[obj.id].position
                    
                    ## ATTEMPT 1 (Oct 1, 2024): if the previous clean plate location is in the sink, we first check if it is currently empty, if so, assign that sink position to empty.
                    if obj.id in prev_kb['grill_states']['ready'] or obj.id in prev_kb['grill_states']['cooking']:
                        empty_grill_loc = prev_kb[obj.id].position
                        #if obj.id in tmp_kb['grill_states']['ready'] or obj.id in tmp_kb['grill_states']['cooking']:
                        if empty_grill_loc not in tmp_kb['grill_states']['empty']:
                            tmp_kb['grill_states']['empty'].append(empty_grill_loc)
                        if obj.id in tmp_kb['grill_states']['ready']:
                            tmp_kb['grill_states']['ready'].remove(obj.id)
                        if obj.id in tmp_kb['grill_states']['cooking']:
                            tmp_kb['grill_states']['cooking'].remove(obj.id)

        elif 'boiled_chicken' in obj.name:
            if obj.position in self.mlam.mdp.get_pot_locations():
                cooking_time = obj._cooking_tick
                if cooking_time < obj._cook_time:
                    if obj.position in tmp_kb['pot_states']['empty']:
                        tmp_kb['pot_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['pot_states']['cooking']:
                        tmp_kb['pot_states']['cooking'].append(obj.id)
                    if obj.id in tmp_kb['pot_states']['ready']:
                        tmp_kb['pot_states']['ready'].remove(obj.id)
                else:
                    if obj.position in tmp_kb['pot_states']['empty']:
                        tmp_kb['pot_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['pot_states']['cooking']:
                        tmp_kb['pot_states']['cooking'].remove(obj.id)
                    if obj.id not in tmp_kb['pot_states']['ready']:
                        tmp_kb['pot_states']['ready'].append(obj.id)
            else:
                if obj.id in prev_kb.keys():
                    # NOTE: this only works if we have one pot (pre ICRA submission)
                    # empty_pot_loc = self.mlam.mdp.get_pot_locations()[0] # prev_kb[obj.id].position
                    
                    ## ATTEMPT 1 (Oct 1, 2024): if the previous clean plate location is in the sink, we first check if it is currently empty, if so, assign that sink position to empty.
                    if obj.id in prev_kb['pot_states']['ready'] or obj.id in prev_kb['pot_states']['cooking']:
                        empty_pot_loc = prev_kb[obj.id].position
                        #if obj.id in tmp_kb['pot_states']['ready'] or obj.id in tmp_kb['pot_states']['cooking']:
                        if empty_pot_loc not in tmp_kb['pot_states']['empty']:
                            tmp_kb['pot_states']['empty'].append(empty_pot_loc)
                        if obj.id in tmp_kb['pot_states']['ready']:
                            tmp_kb['pot_states']['ready'].remove(obj.id)
                        if obj.id in tmp_kb['pot_states']['cooking']:
                            tmp_kb['pot_states']['cooking'].remove(obj.id)

        elif 'garnish' in obj.name:
            chop_time = obj._cooking_tick
            if obj.position in self.mlam.mdp.get_chopping_board_locations():
                if chop_time >= 0 and chop_time < obj.cook_time:
                    if obj.position in tmp_kb['chop_states']['empty']:
                        tmp_kb['chop_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['chop_states']['full']:
                        tmp_kb['chop_states']['full'].append(obj.id)
                    if obj.id in tmp_kb['chop_states']['ready']:
                        tmp_kb['chop_states']['ready'].remove(obj.id)
                elif chop_time >= obj.cook_time:
                    if obj.position in tmp_kb['chop_states']['empty']:
                        tmp_kb['chop_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['chop_states']['full']:
                        tmp_kb['chop_states']['full'].remove(obj.id)
                    if obj.id not in tmp_kb['chop_states']['ready']:
                        tmp_kb['chop_states']['ready'].append(obj.id)
                else:
                    print(tmp_kb['chop_states'])
                    print(obj)
                    raise ValueError()
            else:
                if obj.id in prev_kb.keys():
                    # NOTE: this only works if we have one chopping board (pre ICRA submission)
                    # empty_board_loc = self.mlam.mdp.get_chopping_board_locations()[0] #prev_kb[obj.id].position
                    
                    ## ATTEMPT 1 (Oct 1, 2024): if the previous clean plate location is in the sink, we first check if it is currently empty, if so, assign that sink position to empty.
                    if obj.id in prev_kb['chop_states']['ready'] or obj.id in tmp_kb['chop_states']['full']:
                        empty_board_loc = prev_kb[obj.id].position
                        # if obj.id in tmp_kb['chop_states']['ready'] or obj.id in tmp_kb['chop_states']['full']:
                        if empty_board_loc not in tmp_kb['chop_states']['empty']:
                            tmp_kb['chop_states']['empty'].append(empty_board_loc)
                        if obj.id in tmp_kb['chop_states']['ready']:
                            tmp_kb['chop_states']['ready'].remove(obj.id)
                        if obj.id in tmp_kb['chop_states']['full']:
                            tmp_kb['chop_states']['full'].remove(obj.id)

        elif 'clean_plate' in obj.name:
            wash_time = obj._cooking_tick
            if obj.position in self.mlam.mdp.get_sink_locations():
                if wash_time >= 0 and wash_time < obj.cook_time:
                    if obj.position in tmp_kb['sink_states']['empty']:
                        tmp_kb['sink_states']['empty'].remove(obj.position)
                    if obj.id not in tmp_kb['sink_states']['full']:
                        tmp_kb['sink_states']['full'].append(obj.id)
                    if obj.id in tmp_kb['sink_states']['ready']:
                        tmp_kb['sink_states']['ready'].remove(obj.id)
                elif wash_time >= obj.cook_time:
                    if obj.position in tmp_kb['sink_states']['empty']:
                        tmp_kb['sink_states']['empty'].remove(obj.position)
                    if obj.id in tmp_kb['sink_states']['full']:
                        tmp_kb['sink_states']['full'].remove(obj.id)
                    if obj.id not in tmp_kb['sink_states']['ready']:
                        tmp_kb['sink_states']['ready'].append(obj.id)
                else:
                    print(tmp_kb['sink_states'])
                    print(obj)
                    raise ValueError()
            else:
                if obj.id in prev_kb.keys():
                    ## NOTE: this only works when we have one sink (pre ICRA submission)
                    # empty_sink_loc = self.mlam.mdp.get_sink_locations()[0] 
                    
                    ## ATTEMPT 1 (Oct 1, 2024): if the previous clean plate location is in the sink, we first check if it is currently empty, if so, assign that sink position to empty.
                    if obj.id in prev_kb['sink_states']['ready'] or obj.id in tmp_kb['sink_states']['full']:
                        empty_sink_loc = prev_kb[obj.id].position
                        # if obj.id in tmp_kb['sink_states']['ready'] or obj.id in tmp_kb['sink_states']['full']:
                        if empty_sink_loc not in tmp_kb['sink_states']['empty']:
                            tmp_kb['sink_states']['empty'].append(empty_sink_loc)
                        if obj.id in tmp_kb['sink_states']['ready']:
                            tmp_kb['sink_states']['ready'].remove(obj.id)
                        if obj.id in tmp_kb['sink_states']['full']:
                            tmp_kb['sink_states']['full'].remove(obj.id)
        return tmp_kb

    def update(self, state, rollout_kb=None, rollout_info=None, vision_bound=None, vision_mode=None):
        
        vision_bound = self.vision_bound/2 if vision_bound is None else vision_bound
        vision_mode = self.vision_mode if vision_mode is None else vision_mode
        
        #estimated_vision_bound = self.estimated_vision_bound/2 if estimated_vision_bound is None else estimated_vision_bound
        
        
        # right_pt, left_pt = self.get_vision_bound(state, half_bound=self.vision_bound/2)
        if self.debug:
            print('Before updating:')
            for key, value in self.knowledge_base.items():
                print(f'{key}={value}')
        other_player = state.players[1 - self.agent_index]

        if rollout_kb is not None:
            [rollout_track, rollout_untrack, rollout_remove] = rollout_info
            prev_kb = copy.deepcopy(rollout_kb)
            prev_track = self.kb_track_deepcopy(rollout_track)
            prev_untrack = self.kb_track_deepcopy(rollout_untrack)
            tmp_kb = copy.deepcopy(rollout_kb)
            tmp_track = self.kb_track_deepcopy(rollout_track)
            tmp_untrack = self.kb_track_deepcopy(rollout_untrack)
            prev_remove_obj_id_list = self.kb_track_deepcopy(rollout_remove)
            tmp_remove_obj_id_list = self.kb_track_deepcopy(rollout_remove)
            
            #new misha edit
            prev_estimated_kb = None
            prev_estimated_track = self.kb_track_deepcopy(rollout_track)
            prev_estimated_untrack = self.kb_track_deepcopy(rollout_untrack)

            tmp_estimated_kb = None
            tmp_estimated_track = self.kb_track_deepcopy(rollout_track)
            tmp_estimated_untrack = self.kb_track_deepcopy(rollout_untrack)
            tmp_estimated_remove_obj_id_list = self.kb_track_deepcopy(rollout_remove)
            #end new misha edit
        else:
            prev_kb = copy.deepcopy(self.knowledge_base) # copy.deepcopy(self.knowledge_base)
            
            #new misha edit
            prev_estimated_kb = copy.deepcopy(self.estimated_knowledge_base)
            prev_estimated_track = self.kb_track_deepcopy(self.estimated_kb_update_delay_track[0])
            prev_estimated_untrack = self.kb_track_deepcopy(self.estimated_kb_update_delay_track[1])
            #end new misha edit
            
            prev_track = self.kb_track_deepcopy(self.kb_update_delay_track[0])
            prev_untrack = self.kb_track_deepcopy(self.kb_update_delay_track[1])
            tmp_kb = self.knowledge_base
            
            #new misha edit
            
            tmp_estimated_kb = self.estimated_knowledge_base
            tmp_estimated_track = self.estimated_kb_update_delay_track[0]
            tmp_estimated_untrack = self.estimated_kb_update_delay_track[1]
            tmp_estimated_remove_obj_id_list = self.estimated_kb_update_delay_track[2]
            #end new misha edit
            
            tmp_track = self.kb_update_delay_track[0]
            tmp_untrack = self.kb_update_delay_track[1]
            prev_remove_obj_id_list = self.kb_track_deepcopy(self.kb_update_delay_track[2])
            tmp_remove_obj_id_list = self.kb_update_delay_track[2]

        ### Update the tracking
        new_track, new_untrack = {}, {}
        other_player_dropped_obj_id = -1
        human_pickup_obj_id = -1
        other_player_dropped_obj_count = -1
        all_object_lists_id = {o.id:o for o in state.all_objects_list}
        # prev_track is a dictionary with object id as key and a list of [obj, consecutive seen count]
        for obj_key in set([o2 for o2 in all_object_lists_id] + [o1 for o1 in prev_track.keys()]):
            if obj_key in all_object_lists_id: obj = all_object_lists_id[obj_key]
            else: obj = prev_track[obj_key][0]
            
            if obj_key == 'human_holding':
                obj = state.players[self.agent_index].held_object

            if obj is not None:
                pos = other_player.position if obj_key == 'other_player' else obj.position
                obj_in_bound = (self.in_bound(state, pos, vision_bound=vision_bound, vision_mode=vision_mode))
                if obj_in_bound:
                    if 'other_player' == obj_key: #in prev_track.keys() and obj == prev_track['other_player'][0]:
                        obj_count = min(self.kb_update_delay, prev_track['other_player'][1] + 1)
                        if not self.kb_ackn_prob:
                            seen = obj_count >= self.kb_update_delay
                        else:
                            seen = prev_track['other_player'][2] if prev_track['other_player'][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
                        tmp_track['other_player'] = [other_player.deepcopy(), obj_count, seen]

                        # if the agent picked up an item, and that item was in untrack, we move it to tmp_remove_obj_id_list
                        if prev_track['other_player'][0].held_object == None and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_untrack.keys():
                            tmp_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), obj_count, seen]                  
                        
                        # if the agent drops item, remove the item from player's hand
                        if prev_track['other_player'][0].held_object != None and state.players[1-self.agent_index].held_object == None:
                            # if obj.id not in tmp_remove_obj_id_list.keys():
                            
                            new_obj = None
                            # set the newly dropped inbound object to have the same count as the threshold
                            if prev_track['other_player'][0].held_object.name == 'meat' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'chicken' and state.all_objects_by_type['boiled_chicken']:
                                new_obj = state.all_objects_by_type['boiled_chicken'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'onion' and state.all_objects_by_type['garnish']:
                                new_obj = state.all_objects_by_type['garnish'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'dirty_plate' and state.all_objects_by_type['clean_plate']:
                                new_obj = state.all_objects_by_type['clean_plate'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'clean_plate' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'steak' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                                # for tmp_o_untrack_key, tmp_o_untrack in prev_track.items():
                                #     if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish' and tmp_o_untrack[0].id not in tmp_remove_obj_id_list.keys():
                                #         tmp_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), obj_count, seen]
                            elif prev_track['other_player'][0].held_object.name == 'boiled_chicken' and state.all_objects_by_type['boiled_chicken']:
                                new_obj = state.all_objects_by_type['boiled_chicken'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'steak_onion' and state.all_objects_by_type['steak_onion']:
                                new_obj = state.all_objects_by_type['steak_onion'][-1]
                            elif prev_track['other_player'][0].held_object.name == 'boiled_chicken_onion' and state.all_objects_by_type['boiled_chicken_onion']:
                                new_obj = state.all_objects_by_type['boiled_chicken_onion'][-1]

                            if new_obj != None:
                                tmp_remove_obj_id_list[prev_track['other_player'][0].held_object.id] = [prev_track['other_player'][0].held_object.deepcopy(), obj_count, seen]
                                # flag the new_object and make sure the obj_count is not reset when dropped
                                other_player_dropped_obj_id = new_obj.id
                                other_player_dropped_obj_count = obj_count
                                other_player_dropped_seen = seen
                                tmp_track[new_obj.id] = [new_obj, obj_count, seen]

                    elif human_pickup_obj_id == obj.id:
                        tmp_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay, True]

                    # is player object
                    elif (obj_key in all_object_lists_id) or (obj_key == 'human_holding' and obj.id in all_object_lists_id):
                        obj_count = 1
                        seen = False
                        if obj.id in prev_track.keys():
                            obj_count = min(self.kb_update_delay, prev_track[obj.id][1] + 1)
                            if not self.kb_ackn_prob:
                                seen = obj_count >= self.kb_update_delay
                            else:
                                seen = prev_track[obj.id][2] if prev_track[obj.id][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
                        elif obj.id in tmp_track.keys():
                            obj_count = tmp_track[obj.id][1]
                            seen = tmp_track[obj.id][2]
                        elif other_player_dropped_obj_id == obj.id:
                            obj_count = other_player_dropped_obj_count
                            seen = other_player_dropped_seen
                            
                        if obj.position == state.players[self.agent_index].position:
                            if state.players[self.agent_index].held_object is not None:
                                obj_count = self.kb_update_delay
                                seen = True
                                tmp_track['human_holding'] = [state.players[self.agent_index].held_object.deepcopy(), self.kb_update_delay, seen]
                        tmp_track[obj.id] = [all_object_lists_id[obj.id].deepcopy(), obj_count, seen]

                    # previous human held now dropped and changed object name
                    elif prev_track['human_holding'][0] != None and obj.id == prev_track['human_holding'][0].id and obj != None:
                        human_pos_and_or = state.players[self.agent_index].pos_and_or
                        # if obj.id not in tmp_remove_obj_id_list.keys():
                        tmp_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay, True]

                        if obj.name in ['steak_onion', 'boiled_chicken_onion']:
                            tmp_track['human_holding'] = [None, self.kb_update_delay, True]
                        elif obj.name == 'steak':
                            new_obj = None
                            # If delivered
                            tmp_track['human_holding'] = [new_obj, self.kb_update_delay, True]

                            # If in front of chopped onions and pick them up
                            for tmp_steak_onion in state.all_objects_by_type['steak_onion']:
                                in_front = (human_pos_and_or == (tmp_steak_onion.position, state.players[self.agent_index].orientation))
                                if self.in_bound(state, tmp_steak_onion.position, vision_bound=vision_bound, vision_mode=vision_mode) and in_front:
                                    new_obj = tmp_steak_onion # can overwrite since the last item is the newest object
                                    tmp_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                    for tmp_o_untrack_key, tmp_o_untrack in prev_track.items():
                                        if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish':
                                            human_pickup_obj_id = tmp_o_untrack[0].id
                                            tmp_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), self.kb_update_delay, True]
                                    tmp_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                        elif obj.name == 'boiled_chicken':
                                new_obj = None
                                # If delivered
                                tmp_track['human_holding'] = [new_obj, self.kb_update_delay, True]

                                # if in front of chopped onions and pick them up
                                for tmp_boilded_chicken_onion in state.all_objects_by_type['boiled_chicken_onion']:
                                    in_front = (human_pos_and_or == (tmp_boilded_chicken_onion.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_boilded_chicken_onion.position, vision_bound=vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_boilded_chicken_onion
                                        tmp_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                        for tmp_o_untrack_key, tmp_o_untrack in prev_track.items():
                                            if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish':
                                                human_pickup_obj_id = tmp_o_untrack[0].id
                                                tmp_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), self.kb_update_delay, True]
                                    tmp_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                        else:
                            # set the newly dropped inbound object to have the same count as the threshold
                            if obj.name == 'meat':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['steak']) == 0 else state.all_objects_by_type['steak'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'chicken':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['boiled_chicken']) == 0 else state.all_objects_by_type['boiled_chicken'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'onion':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['garnish']) == 0 else state.all_objects_by_type['garnish'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'dirty_plate':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['clean_plate']) == 0 else state.all_objects_by_type['clean_plate'][-1]
                                tmp_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'clean_plate':
                                new_obj = None
                                for tmp_steak in state.all_objects_by_type['steak']:
                                    in_front = (human_pos_and_or == (tmp_steak.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_steak.position, vision_bound=vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_steak # can overwrite since the last item is the newest object
                                for tmp_boiled_chicken in state.all_objects_by_type['boiled_chicken']:
                                    in_front = (human_pos_and_or == (tmp_boiled_chicken.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_boiled_chicken.position, vision_bound=vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_boiled_chicken # can overwrite since the last item is the newest object

                                if new_obj != None:
                                    tmp_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                    
                            tmp_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                    else:
                        # if obj_key != 'human_holding':
                        if obj_key not in tmp_remove_obj_id_list.keys():
                            tmp_remove_obj_id_list[obj_key] = [obj.deepcopy(), 0, False]

                else:
                    if 'other_player' == obj_key: # and obj_id == prev_track['other_player'][0].id:
                        tmp_untrack['other_player'] = prev_track['other_player'][0].deepcopy()
                        del tmp_track['other_player']
                    elif obj_key in prev_track.keys():
                        tmp_untrack[obj_key] = prev_track[obj_key][0].deepcopy()
                        del tmp_track[obj_key]

        if 'other_player' not in tmp_track.keys() and self.in_bound(state, other_player.position, vision_bound=vision_bound, vision_mode=vision_mode):
            tmp_track['other_player'] = [other_player.deepcopy(), 1, False if not self.kb_ackn_prob else random.choices([True, False], weights=[0.33, 0.67])[0]]

            # if the agent picked up an item
            if 'other_player' in prev_untrack.keys() and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_untrack.keys() and prev_untrack['other_player'].held_object == None and state.players[1-self.agent_index].held_object != None:
                tmp_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), 0, False]

            ## if the agent dropped an item
            # if 'other_player' in prev_track.keys() and prev_track['other_player'][0].held_object != None and prev_track['other_player'][0].held_object == None:
            #     # if obj.id not in tmp_remove_obj_id_list.keys():
            #     tmp_remove_obj_id_list[prev_track['other_player'][0].held_object.id] = [prev_track['other_player'][0].held_object.deepcopy(), 0]

        for obj_id, obj in prev_untrack.items():
            obj_in_bound = self.in_bound(state, obj.position, vision_bound=vision_bound, vision_mode=vision_mode)
            if obj_in_bound:
                if obj_id in all_object_lists_id.keys() and obj_id in tmp_track.keys():
                    # means is not in sight but still exists
                    del tmp_untrack[obj_id]
                elif obj_id == 'other_player' and obj_id in tmp_track.keys():
                    # new position inbound
                    del tmp_untrack['other_player']
                else:
                    if obj_id not in tmp_remove_obj_id_list.keys():
                        tmp_remove_obj_id_list[obj_id] = [obj.deepcopy(), 0, False]
        
        ### Update the KB with the removed object list
        del_remove_list = []
        for obj_id in tmp_remove_obj_id_list.keys():
            if obj_id in tmp_track.keys(): del tmp_track[obj_id]
            if obj_id in tmp_untrack.keys(): del tmp_untrack[obj_id]

            if self.in_bound(state, tmp_remove_obj_id_list[obj_id][0].position, vision_bound=vision_bound, vision_mode=vision_mode):
                tmp_remove_obj_id_list[obj_id][1] += 1
                if not self.kb_ackn_prob:
                    tmp_remove_obj_id_list[obj_id][2] = (tmp_remove_obj_id_list[obj_id][1] >= self.kb_update_delay)
                else:
                    tmp_remove_obj_id_list[obj_id][2] = tmp_remove_obj_id_list[obj_id][2] if tmp_remove_obj_id_list[obj_id][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
            
            ## update tiles and then remove an object due to dropped on top of key tiles (ex: sink)
            if tmp_remove_obj_id_list[obj_id][2]:
                if obj_id != 'other_player' and obj_id in tmp_kb.keys():
                    tmp_obj = tmp_kb[obj_id].deepcopy()
                    tmp_obj.position = None
                    tmp_kb = self.update_kb_key_object(tmp_obj, tmp_kb, prev_kb)
                else:
                    # do nothing
                    # tmp_kb['other_player'].position = None
                    # tmp_kb['other_player'].held_object = None
                    pass
                ## Remove the object from all tracking
                if obj_id != 'other_player' and obj_id in tmp_kb.keys(): del tmp_kb[obj_id]

                del_remove_list.append(obj_id)
        
        for obj_id in del_remove_list:
            del tmp_remove_obj_id_list[obj_id]
            
        ### Update the KB with the updated tracking list
        for obj_id, [obj, obj_count, obj_seen] in tmp_track.items():
            seen = (obj_count >= self.kb_update_delay and not self.kb_ackn_prob) or (self.kb_ackn_prob and obj_seen)
            # if obj_count >= self.kb_update_delay and obj is not None and obj_id != 'human_holding':
            if seen and obj is not None and obj_id != 'human_holding':
                ## Update the object in KB
                tmp_kb[obj_id] = obj.deepcopy()
                if obj_id != 'other_player':
                    tmp_kb = self.update_kb_key_object(obj.deepcopy(), tmp_kb, prev_kb)
                else:
                    tmp_kb['other_player'] = obj.deepcopy()
        
        ## print out knowledge base
        if rollout_kb is None and self.debug:
            print('\nAfter update:\ntmp_kb:')
            for k, v in tmp_kb.items():
                print(k, ':', v)
            print('\ntmp_track:')
            for k, v in tmp_track.items():
                print(k, ':', v)
            print('\ntmp_untrack:')
            for k, v in tmp_untrack.items():
                print(k, ':', v)
            print('')
            for k, v in tmp_remove_obj_id_list.items():
                print(k, ':', v)
            print('')
            


#------------new misha edit--------------------------------------------------------------------------------------------------------------------------
        ### Update the tracking
        new_track, new_untrack = {}, {}
        other_player_dropped_obj_id = -1
        human_pickup_obj_id = -1
        other_player_dropped_obj_count = -1
        all_object_lists_id = {o.id:o for o in state.all_objects_list}
        # prev_estimated_track is a dictionary with object id as key and a list of [obj, consecutive seen count]
        for obj_key in set([o2 for o2 in all_object_lists_id] + [o1 for o1 in prev_estimated_track.keys()]):
            if obj_key in all_object_lists_id: obj = all_object_lists_id[obj_key]
            else: obj = prev_estimated_track[obj_key][0]
            
            if obj_key == 'human_holding':
                obj = state.players[self.agent_index].held_object

            if obj is not None:
                pos = other_player.position if obj_key == 'other_player' else obj.position
                obj_in_bound = (self.in_bound(state, pos, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode))
                if obj_in_bound:
                    if 'other_player' == obj_key: #in prev_estimated_track.keys() and obj == prev_estimated_track['other_player'][0]:
                        obj_count = min(self.kb_update_delay, prev_estimated_track['other_player'][1] + 1)
                        if not self.kb_ackn_prob:
                            seen = obj_count >= self.kb_update_delay
                        else:
                            seen = prev_estimated_track['other_player'][2] if prev_estimated_track['other_player'][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
                        tmp_estimated_track['other_player'] = [other_player.deepcopy(), obj_count, seen]

                        # if the agent picked up an item, and that item was in untrack, we move it to tmp_estimated_remove_obj_id_list
                        if prev_estimated_track['other_player'][0].held_object == None and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_estimated_untrack.keys():
                            tmp_estimated_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_estimated_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), obj_count, seen]                  
                        
                        # if the agent drops item, remove the item from player's hand
                        if prev_estimated_track['other_player'][0].held_object != None and state.players[1-self.agent_index].held_object == None:
                            # if obj.id not in tmp_estimated_remove_obj_id_list.keys():
                            
                            new_obj = None
                            # set the newly dropped inbound object to have the same count as the threshold
                            if prev_estimated_track['other_player'][0].held_object.name == 'meat' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'chicken' and state.all_objects_by_type['boiled_chicken']:
                                new_obj = state.all_objects_by_type['boiled_chicken'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'onion' and state.all_objects_by_type['garnish']:
                                new_obj = state.all_objects_by_type['garnish'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'dirty_plate' and state.all_objects_by_type['clean_plate']:
                                new_obj = state.all_objects_by_type['clean_plate'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'clean_plate' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'steak' and state.all_objects_by_type['steak']:
                                new_obj = state.all_objects_by_type['steak'][-1]
                                # for tmp_o_untrack_key, tmp_o_untrack in prev_estimated_track.items():
                                #     if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish' and tmp_o_untrack[0].id not in tmp_estimated_remove_obj_id_list.keys():
                                #         tmp_estimated_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), obj_count, seen]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'boiled_chicken' and state.all_objects_by_type['boiled_chicken']:
                                new_obj = state.all_objects_by_type['boiled_chicken'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'steak_onion' and state.all_objects_by_type['steak_onion']:
                                new_obj = state.all_objects_by_type['steak_onion'][-1]
                            elif prev_estimated_track['other_player'][0].held_object.name == 'boiled_chicken_onion' and state.all_objects_by_type['boiled_chicken_onion']:
                                new_obj = state.all_objects_by_type['boiled_chicken_onion'][-1]

                            if new_obj != None:
                                tmp_estimated_remove_obj_id_list[prev_estimated_track['other_player'][0].held_object.id] = [prev_estimated_track['other_player'][0].held_object.deepcopy(), obj_count, seen]
                                # flag the new_object and make sure the obj_count is not reset when dropped
                                other_player_dropped_obj_id = new_obj.id
                                other_player_dropped_obj_count = obj_count
                                other_player_dropped_seen = seen
                                tmp_estimated_track[new_obj.id] = [new_obj, obj_count, seen]

                    elif human_pickup_obj_id == obj.id:
                        tmp_estimated_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay, True]

                    # is player object
                    elif (obj_key in all_object_lists_id) or (obj_key == 'human_holding' and obj.id in all_object_lists_id):
                        obj_count = 1
                        seen = False
                        if obj.id in prev_estimated_track.keys():
                            obj_count = min(self.kb_update_delay, prev_estimated_track[obj.id][1] + 1)
                            if not self.kb_ackn_prob:
                                seen = obj_count >= self.kb_update_delay
                            else:
                                seen = prev_estimated_track[obj.id][2] if prev_estimated_track[obj.id][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
                        elif obj.id in tmp_estimated_track.keys():
                            obj_count = tmp_estimated_track[obj.id][1]
                            seen = tmp_estimated_track[obj.id][2]
                        elif other_player_dropped_obj_id == obj.id:
                            obj_count = other_player_dropped_obj_count
                            seen = other_player_dropped_seen
                            
                        if obj.position == state.players[self.agent_index].position:
                            if state.players[self.agent_index].held_object is not None:
                                obj_count = self.kb_update_delay
                                seen = True
                                tmp_estimated_track['human_holding'] = [state.players[self.agent_index].held_object.deepcopy(), self.kb_update_delay, seen]
                        tmp_estimated_track[obj.id] = [all_object_lists_id[obj.id].deepcopy(), obj_count, seen]

                    # previous human held now dropped and changed object name
                    elif prev_estimated_track['human_holding'][0] != None and obj.id == prev_estimated_track['human_holding'][0].id and obj != None:
                        human_pos_and_or = state.players[self.agent_index].pos_and_or
                        # if obj.id not in tmp_estimated_remove_obj_id_list.keys():
                        tmp_estimated_remove_obj_id_list[obj.id] = [obj.deepcopy(), self.kb_update_delay, True]

                        if obj.name in ['steak_onion', 'boiled_chicken_onion']:
                            tmp_estimated_track['human_holding'] = [None, self.kb_update_delay, True]
                        elif obj.name == 'steak':
                            new_obj = None
                            # If delivered
                            tmp_estimated_track['human_holding'] = [new_obj, self.kb_update_delay, True]

                            # If in front of chopped onions and pick them up
                            for tmp_steak_onion in state.all_objects_by_type['steak_onion']:
                                in_front = (human_pos_and_or == (tmp_steak_onion.position, state.players[self.agent_index].orientation))
                                if self.in_bound(state, tmp_steak_onion.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode) and in_front:
                                    new_obj = tmp_steak_onion # can overwrite since the last item is the newest object
                                    tmp_estimated_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                    for tmp_o_untrack_key, tmp_o_untrack in prev_estimated_track.items():
                                        if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish':
                                            human_pickup_obj_id = tmp_o_untrack[0].id
                                            tmp_estimated_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), self.kb_update_delay, True]
                                    tmp_estimated_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                        elif obj.name == 'boiled_chicken':
                                new_obj = None
                                # If delivered
                                tmp_estimated_track['human_holding'] = [new_obj, self.kb_update_delay, True]

                                # if in front of chopped onions and pick them up
                                for tmp_boilded_chicken_onion in state.all_objects_by_type['boiled_chicken_onion']:
                                    in_front = (human_pos_and_or == (tmp_boilded_chicken_onion.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_boilded_chicken_onion.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_boilded_chicken_onion
                                        tmp_estimated_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                        for tmp_o_untrack_key, tmp_o_untrack in prev_estimated_track.items():
                                            if tmp_o_untrack_key not in ['human_holding', 'other_player'] and tmp_o_untrack[0].name == 'garnish':
                                                human_pickup_obj_id = tmp_o_untrack[0].id
                                                tmp_estimated_remove_obj_id_list[tmp_o_untrack[0].id] = [tmp_o_untrack[0].deepcopy(), self.kb_update_delay, True]
                                    tmp_estimated_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                        else:
                            # set the newly dropped inbound object to have the same count as the threshold
                            if obj.name == 'meat':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['steak']) == 0 else state.all_objects_by_type['steak'][-1]
                                tmp_estimated_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'chicken':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['boiled_chicken']) == 0 else state.all_objects_by_type['boiled_chicken'][-1]
                                tmp_estimated_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'onion':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['garnish']) == 0 else state.all_objects_by_type['garnish'][-1]
                                tmp_estimated_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'dirty_plate':
                                new_obj = obj.deepcopy() if len(state.all_objects_by_type['clean_plate']) == 0 else state.all_objects_by_type['clean_plate'][-1]
                                tmp_estimated_track['human_holding'] = [None, self.kb_update_delay, True]
                            elif obj.name == 'clean_plate':
                                new_obj = None
                                for tmp_steak in state.all_objects_by_type['steak']:
                                    in_front = (human_pos_and_or == (tmp_steak.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_steak.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_steak # can overwrite since the last item is the newest object
                                for tmp_boiled_chicken in state.all_objects_by_type['boiled_chicken']:
                                    in_front = (human_pos_and_or == (tmp_boiled_chicken.position, state.players[self.agent_index].orientation))
                                    if self.in_bound(state, tmp_boiled_chicken.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode) and in_front:
                                        new_obj = tmp_boiled_chicken # can overwrite since the last item is the newest object

                                if new_obj != None:
                                    tmp_estimated_track['human_holding'] = [new_obj.deepcopy(), self.kb_update_delay, True]
                                    
                            tmp_estimated_track[new_obj.id] = [new_obj.deepcopy(), self.kb_update_delay, True]

                    else:
                        # if obj_key != 'human_holding':
                        if obj_key not in tmp_estimated_remove_obj_id_list.keys():
                            tmp_estimated_remove_obj_id_list[obj_key] = [obj.deepcopy(), 0, False]

                else:
                    if 'other_player' == obj_key: # and obj_id == prev_estimated_track['other_player'][0].id:
                        tmp_estimated_untrack['other_player'] = prev_estimated_track['other_player'][0].deepcopy()
                        del tmp_estimated_track['other_player']
                    elif obj_key in prev_estimated_track.keys():
                        tmp_estimated_untrack[obj_key] = prev_estimated_track[obj_key][0].deepcopy()
                        del tmp_estimated_track[obj_key]

        if 'other_player' not in tmp_estimated_track.keys() and self.in_bound(state, other_player.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode):
            tmp_estimated_track['other_player'] = [other_player.deepcopy(), 1, False if not self.kb_ackn_prob else random.choices([True, False], weights=[0.33, 0.67])[0]]

            # if the agent picked up an item
            if 'other_player' in prev_estimated_untrack.keys() and state.players[1-self.agent_index].held_object != None and state.players[1-self.agent_index].held_object.id in prev_estimated_untrack.keys() and prev_estimated_untrack['other_player'].held_object == None and state.players[1-self.agent_index].held_object != None:
                tmp_estimated_remove_obj_id_list[state.players[1-self.agent_index].held_object.id] = [prev_estimated_untrack[state.players[1-self.agent_index].held_object.id].deepcopy(), 0, False]

            ## if the agent dropped an item
            # if 'other_player' in prev_estimated_track.keys() and prev_estimated_track['other_player'][0].held_object != None and prev_estimated_track['other_player'][0].held_object == None:
            #     # if obj.id not in tmp_estimated_remove_obj_id_list.keys():
            #     tmp_estimated_remove_obj_id_list[prev_estimated_track['other_player'][0].held_object.id] = [prev_estimated_track['other_player'][0].held_object.deepcopy(), 0]

        for obj_id, obj in prev_estimated_untrack.items():
            obj_in_bound = self.in_bound(state, obj.position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode)
            if obj_in_bound:
                if obj_id in all_object_lists_id.keys() and obj_id in tmp_estimated_track.keys():
                    # means is not in sight but still exists
                    del tmp_estimated_untrack[obj_id]
                elif obj_id == 'other_player' and obj_id in tmp_estimated_track.keys():
                    # new position inbound
                    del tmp_estimated_untrack['other_player']
                else:
                    if obj_id not in tmp_estimated_remove_obj_id_list.keys():
                        tmp_estimated_remove_obj_id_list[obj_id] = [obj.deepcopy(), 0, False]
        
        ### Update the KB with the removed object list
        del_remove_list = []
        for obj_id in tmp_estimated_remove_obj_id_list.keys():
            if obj_id in tmp_estimated_track.keys(): del tmp_estimated_track[obj_id]
            if obj_id in tmp_estimated_untrack.keys(): del tmp_estimated_untrack[obj_id]

            if self.in_bound(state, tmp_estimated_remove_obj_id_list[obj_id][0].position, vision_bound=self.estimated_vision_bound, vision_mode=vision_mode):
                tmp_estimated_remove_obj_id_list[obj_id][1] += 1
                if not self.kb_ackn_prob:
                    tmp_estimated_remove_obj_id_list[obj_id][2] = (tmp_estimated_remove_obj_id_list[obj_id][1] >= self.kb_update_delay)
                else:
                    tmp_estimated_remove_obj_id_list[obj_id][2] = tmp_estimated_remove_obj_id_list[obj_id][2] if tmp_estimated_remove_obj_id_list[obj_id][2] else random.choices([True, False], weights=[0.33, 0.67])[0]
            
            ## update tiles and then remove an object due to dropped on top of key tiles (ex: sink)
            if tmp_estimated_remove_obj_id_list[obj_id][2]:
                if obj_id != 'other_player' and obj_id in tmp_estimated_kb.keys():
                    tmp_obj = tmp_estimated_kb[obj_id].deepcopy()
                    tmp_obj.position = None
                    tmp_estimated_kb = self.update_kb_key_object(tmp_obj, tmp_estimated_kb, prev_estimated_kb)
                else:
                    # do nothing
                    # tmp_estimated_kb['other_player'].position = None
                    # tmp_estimated_kb['other_player'].held_object = None
                    pass
                ## Remove the object from all tracking
                if obj_id != 'other_player' and obj_id in tmp_estimated_kb.keys(): del tmp_estimated_kb[obj_id]

                del_remove_list.append(obj_id)
        
        for obj_id in del_remove_list:
            del tmp_estimated_remove_obj_id_list[obj_id]
            
        ### Update the KB with the updated tracking list
        for obj_id, [obj, obj_count, obj_seen] in tmp_estimated_track.items():
            seen = (obj_count >= self.kb_update_delay and not self.kb_ackn_prob) or (self.kb_ackn_prob and obj_seen)
            # if obj_count >= self.kb_update_delay and obj is not None and obj_id != 'human_holding':
            if seen and obj is not None and obj_id != 'human_holding':
                ## Update the object in KB
                tmp_estimated_kb[obj_id] = obj.deepcopy()
                if obj_id != 'other_player':
                    tmp_estimated_kb = self.update_kb_key_object(obj.deepcopy(), tmp_estimated_kb, prev_estimated_kb)
                else:
                    tmp_estimated_kb['other_player'] = obj.deepcopy()
        
        # ## print out knowledge base
        # if rollout_kb is None and self.debug:
        #     print('\nAfter update:\ntmp_kb:')
        #     for k, v in tmp_kb.items():
        #         print(k, ':', v)
        #     print('\ntmp_estimated_track:')
        #     for k, v in tmp_estimated_track.items():
        #         print(k, ':', v)
        #     print('\ntmp_estimated_untrack:')
        #     for k, v in tmp_estimated_untrack.items():
        #         print(k, ':', v)
        #     print('')
        #     for k, v in tmp_estimated_remove_obj_id_list.items():
        #         print(k, ':', v)
        #     print('')
        #------------end new misha edit --------

        return tmp_kb, [tmp_track, tmp_untrack, tmp_remove_obj_id_list]
    
    def get_human_traj(self, world_state, human_subtask_obj):
        # get human holding object name
        human_obj = 'None' if world_state.players[1-self.agent_index].held_object == None else world_state.players[1-self.agent_index].held_object.name

        # limit the human to take the optimal action to complete its subtask (robot's belief)
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(world_state, human_subtask_obj[0], human_subtask_obj[1], p0_obj=human_obj, player_idx=abs(1-self.agent_index)) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)

        human_cost, human_feature_pos = self.mp.min_cost_to_feature(world_state.players[abs(1-self.agent_index)].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else world_state.players[(1-self.agent_index)].get_pos_and_or()
        agent_pos = world_state.players[self.agent_index].get_pos_and_or()

        # shift by one grid if goal position overlappes with the robot agent
        if agent_pos == new_human_pos:
            _, new_human_pos = self._shift_same_goal_pos([agent_pos, new_human_pos], np.argmax(np.array([0, human_cost])))

        # get grid path from human's original position to goal
        ori_human_pos = world_state.players[(1-self.agent_index)].get_pos_and_or()
        next_las, _, _ = self.mp.get_plan(ori_human_pos, new_human_pos)

        return next_las

    def ml_action(self, state, chosen_subtask = None, vision_bound=None, vision_mode=None):
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
        player = state.players[self.agent_index]
        other_players = [self.knowledge_base['other_player']] #this obvi has self.knowledge_base
        
        # Get kitchen state information
        kitchen_state = self._get_kitchen_state_info(state) #this calls self.knowledge_base
        
        # Get player state information
        player_state = self._get_player_state_info(player, other_players)
        
        # If a specific subtask is chosen, handle it directly
        if chosen_subtask is not None:
            motion_goals = self._get_motion_goals_for_chosen_subtask(
                chosen_subtask, state, kitchen_state, player_state
            )
        else:
            # Otherwise, determine the best subtask based on current state
            motion_goals, chosen_subtask = self._get_motion_goals_for_current_state(
                state, kitchen_state, player_state
            )
        
        # Filter motion goals to ensure they're valid
        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        
        # If no valid motion goals, use fallback strategy
        if len(motion_goals) == 0:
            motion_goals = self._get_fallback_motion_goals(state, player, player_state)
        
        # Update state and return
        self.prev_chosen_subtask = chosen_subtask
        
        #-----------new misha edit HERE--------------------------------------------
        # Initialize state variables
        player = state.players[self.agent_index]
        estimated_other_players = [self.estimated_knowledge_base['other_player']] #this obvi has self.knowledge_base
        
        # Get kitchen state information
        estimated_kitchen_state = self._get_estimated_kitchen_state_info(state) #this calls self.knowledge_base
        
        # Get player state information
        estimated_player_state = self._get_player_state_info(player, estimated_other_players)
        
        estimated_motion_goals, estimated_chosen_subtask = self._get_motion_goals_for_current_state(
                state, estimated_kitchen_state, estimated_player_state
                )
                
        # If a specific subtask is chosen, handle it directly
        # if chosen_subtask is not None:
        #     estimated_motion_goals = self._get_motion_goals_for_chosen_subtask(
        #         chosen_subtask, state, estimated_kitchen_state, estimated_player_state
        #     )
        # else:
        #     # Otherwise, determine the best subtask based on current state
        #     estimated_motion_goals, estimated_chosen_subtask = self._get_motion_goals_for_current_state(
        #         state, estimated_kitchen_state, estimated_player_state
        #     )
        
        # Filter motion goals to ensure they're valid
        #estimated_motion_goals = [mg for mg in estimated_motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        
        # # If no valid motion goals, use fallback strategy
        # if len(estimated_motion_goals) == 0:
        #     estimated_motion_goals = self._get_fallback_motion_goals(state, player, estimated_player_state)
        
        # Update state and return
        self.prev_estimated_chosen_subtask = estimated_chosen_subtask
        #-----------end new misha edit----------------------------------------
        
        
        
        state.players[self.agent_index].subtask_log += [chosen_subtask]
        return motion_goals
    
    
    def _get_estimated_kitchen_state_info(self, state):
        """Extract and organize kitchen state information."""
        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        sink_states = self.estimated_knowledge_base['sink_states']
        chopping_board_states = self.estimated_knowledge_base['chop_states']
        pot_states_dict = self.estimated_knowledge_base['pot_states']
        grill_states_dict = self.estimated_knowledge_base['grill_states']
        
        # Calculate derived state information
        ready_grill = grill_states_dict["ready"]
        cooking_grill = grill_states_dict["cooking"]
        ready_pot = pot_states_dict["ready"]
        cooking_pot = pot_states_dict["cooking"]
        
        # World state flags
        steak_nearly_ready = len(ready_grill) > 0 or len(cooking_grill) > 0
        chicken_nearly_ready = len(ready_pot) > 0 or len(cooking_pot) > 0
        garnish_ready = len(chopping_board_states["ready"]) > 0
        steak_ready = len(grill_states_dict["ready"]) > 0
        boiled_chicken_ready = len(pot_states_dict["ready"]) > 0
        chopping = len(chopping_board_states["full"]) > 0
        board_empty = len(chopping_board_states["empty"]) > 0
        clean_plate_ready = len(sink_states["ready"]) > 0
        rinsing = len(sink_states["full"]) > 0
        sink_empty = len(sink_states["empty"]) > 0
        
        return {
            'counter_objects': counter_objects,
            'sink_states': sink_states,
            'chopping_board_states': chopping_board_states,
            'pot_states_dict': pot_states_dict,
            'grill_states_dict': grill_states_dict,
            'steak_nearly_ready': steak_nearly_ready,
            'chicken_nearly_ready': chicken_nearly_ready,
            'garnish_ready': garnish_ready,
            'steak_ready': steak_ready,
            'boiled_chicken_ready': boiled_chicken_ready,
            'chopping': chopping,
            'board_empty': board_empty,
            'clean_plate_ready': clean_plate_ready,
            'rinsing': rinsing,
            'sink_empty': sink_empty
        }
    
    
    def _get_kitchen_state_info(self, state):
        """Extract and organize kitchen state information."""
        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        sink_states = self.knowledge_base['sink_states']
        chopping_board_states = self.knowledge_base['chop_states']
        pot_states_dict = self.knowledge_base['pot_states']
        grill_states_dict = self.knowledge_base['grill_states']
        
        # Calculate derived state information
        ready_grill = grill_states_dict["ready"]
        cooking_grill = grill_states_dict["cooking"]
        ready_pot = pot_states_dict["ready"]
        cooking_pot = pot_states_dict["cooking"]
        
        # World state flags
        steak_nearly_ready = len(ready_grill) > 0 or len(cooking_grill) > 0
        chicken_nearly_ready = len(ready_pot) > 0 or len(cooking_pot) > 0
        garnish_ready = len(chopping_board_states["ready"]) > 0
        steak_ready = len(grill_states_dict["ready"]) > 0
        boiled_chicken_ready = len(pot_states_dict["ready"]) > 0
        chopping = len(chopping_board_states["full"]) > 0
        board_empty = len(chopping_board_states["empty"]) > 0
        clean_plate_ready = len(sink_states["ready"]) > 0
        rinsing = len(sink_states["full"]) > 0
        sink_empty = len(sink_states["empty"]) > 0
        
        return {
            'counter_objects': counter_objects,
            'sink_states': sink_states,
            'chopping_board_states': chopping_board_states,
            'pot_states_dict': pot_states_dict,
            'grill_states_dict': grill_states_dict,
            'steak_nearly_ready': steak_nearly_ready,
            'chicken_nearly_ready': chicken_nearly_ready,
            'garnish_ready': garnish_ready,
            'steak_ready': steak_ready,
            'boiled_chicken_ready': boiled_chicken_ready,
            'chopping': chopping,
            'board_empty': board_empty,
            'clean_plate_ready': clean_plate_ready,
            'rinsing': rinsing,
            'sink_empty': sink_empty
        }
    
    def _get_player_state_info(self, player, other_players):
        """Extract and organize player state information."""
        other_has_dirty_plate = self._others_have_object(other_players, "dirty_plate")
        other_has_clean_plate = self._others_have_object(other_players, "clean_plate")
        other_has_steak = self._others_have_object(other_players, "steak")
        other_has_meat = self._others_have_object(other_players, "meat")
        other_has_onion = self._others_have_object(other_players, "onion")
        other_has_chicken = self._others_have_object(other_players, "chicken")
        other_has_boiled_chicken = self._others_have_object(
            other_players, "boiled_chicken"
        )
        
        return {
            'other_has_dirty_plate': other_has_dirty_plate,
            'other_has_clean_plate': other_has_clean_plate,
            'other_has_steak': other_has_steak,
            'other_has_meat': other_has_meat,
            'other_has_onion': other_has_onion,
            'other_has_chicken': other_has_chicken,
            'other_has_boiled_chicken': other_has_boiled_chicken,
            'has_object': player.has_object(),
            'object_name': player.get_object().name if player.has_object() else None
        }
    
    def _get_motion_goals_for_chosen_subtask(self, chosen_subtask, state, kitchen_state, player_state):
        """Get motion goals for a specific chosen subtask."""
        am = self.mlam
        
        # Map of subtask names to their corresponding action methods
        subtask_action_map = {
            'pickup_meat': lambda: am.pickup_meat_actions(kitchen_state['counter_objects']),
            'pickup_onion': lambda: am.pickup_onion_actions(kitchen_state['counter_objects']),
            'pickup_dirty_plate': lambda: am.pickup_dirty_plate_actions(kitchen_state['counter_objects'], state),
            'pickup_clean_plate': lambda: self._get_clean_plate_actions(state, kitchen_state, player_state),
            'pickup_steak': lambda: self._get_steak_pickup_actions(state, kitchen_state, player_state),
            'add_garnish': lambda: self._get_garnish_actions(state, kitchen_state, player_state),
            'drop_meat': lambda: am.put_meat_in_grill_actions(kitchen_state['grill_states_dict'], knowledge_base=self.knowledge_base, only_empty=True),
            'drop_onion': lambda: am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base),
            'drop_dirty_plate': lambda: am.put_dirty_plate_in_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base),
            'chop_onion': lambda: am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base),
            'rinse_plate': lambda: am.rinse_plate_in_sink_actions(state, knowledge_base=self.knowledge_base, other_players=[self.knowledge_base['other_player']]),
            'deliver_dish': lambda: am.deliver_dish_actions(),
            'drop_chicken': lambda: am.put_chicken_in_pot_actions(kitchen_state['pot_states_dict'], knowledge_base=self.knowledge_base),
            'pickup_boiled_chicken': lambda: self._get_boiled_chicken_pickup_actions(state, kitchen_state, player_state),
            'pickup_chicken': lambda: am.pickup_chicken_actions(kitchen_state['counter_objects'])
        }
        
        # Execute the appropriate action method for the chosen subtask
        if chosen_subtask in subtask_action_map:
            return subtask_action_map[chosen_subtask]()
        else:
            if player_state['has_object'] and self.drop_on_counter:
                return am.place_obj_on_counter_actions(state)
            else:
                return []
    
    def _get_clean_plate_actions(self, state, kitchen_state, player_state):
        """Get actions for picking up a clean plate."""
        am = self.mlam
        motion_goals = am.pickup_clean_plate_from_sink_actions(
            kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base
        )
        
        if len(motion_goals) == 0 and player_state['other_has_dirty_plate']:
            motion_goals += self.mlam.mdp.get_sink_states(state)['full']
            if len(motion_goals) == 0:
                tmp_goal = self.mlam.mdp.get_sink_states(state)['empty'][0]
                motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(
                    self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal
                )
        
        return motion_goals
    
    def _get_steak_pickup_actions(self, state, kitchen_state, player_state):
        """Get actions for picking up a steak."""
        am = self.mlam
        motion_goals = am.pickup_steak_with_clean_plate_actions(
            self.mlam.mdp.get_grill_states(state), knowledge_base=self.knowledge_base
        )
        
        if len(motion_goals) == 0 and player_state['other_has_meat']:
            motion_goals = self.mlam.mdp.get_grill_states(state)['steak']['cooking'] + self.mlam.mdp.get_grill_states(state)['steak']['partially_full']
            if len(motion_goals) == 0:
                tmp_goal = self.mlam.mdp.get_grill_states(state)['empty'][0]
                motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(
                    self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal
                )
        
        return motion_goals
    
    def _get_garnish_actions(self, state, kitchen_state, player_state):
        """Get actions for adding garnish to a dish."""
        am = self.mlam
        motion_goals = am.add_garnish_to_dish_actions(state, knowledge_base=self.knowledge_base)
        
        if len(motion_goals) == 0 and player_state['other_has_onion']:
            motion_goals = self.mlam.mdp.get_chopping_board_states(state)['full']
            if len(motion_goals) == 0:
                tmp_goal = self.mlam.mdp.get_chopping_board_states(state)['empty'][0]
                motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(
                    self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal
                )
        
        return motion_goals
    
    def _get_boiled_chicken_pickup_actions(self, state, kitchen_state, player_state):
        """Get actions for picking up boiled chicken."""
        am = self.mlam
        motion_goals = am.pickup_boiled_chicken_with_clean_plate_actions(
            kitchen_state['pot_states_dict'], only_nearly_ready=True, knowledge_base=self.knowledge_base
        )
        
        if len(motion_goals) == 0 and player_state['other_has_chicken']:
            motion_goals = self.mlam.mdp.get_pot_states(state)['chicken']['cooking'] + self.mlam.mdp.get_pot_states(state)['chicken']['partially_full']
            if len(motion_goals) == 0:
                tmp_goal = self.mlam.mdp.get_pot_states(state)['empty'][0]
                motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(
                    self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal
                )
        
        return motion_goals
    
    def _get_motion_goals_for_current_state(self, state, kitchen_state, player_state):
        """Determine the best motion goals based on the current state."""
        am = self.mlam
        motion_goals = []
        chosen_subtask = None
        order_idx = 0
        
        # Try to find motion goals for each order in the queue
        while len(motion_goals) == 0 and order_idx < state.num_orders_remaining:
            curr_order = state.order_list[order_idx]
            
            # Adjust plate readiness based on order index
            clean_plate_ready = len(kitchen_state['sink_states']["ready"]) > order_idx
            rinsing = len(kitchen_state['sink_states']["full"]) > order_idx
            
            # Handle different order types
            if curr_order == "steak_dish" and not player_state['other_has_steak']:
                motion_goals, chosen_subtask = self._handle_steak_dish(
                    state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing
                )
            elif curr_order == "steak_onion_dish":
                motion_goals, chosen_subtask = self._handle_steak_onion_dish(
                    state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing
                )
            elif curr_order == "boiled_chicken_dish" and not player_state['other_has_boiled_chicken']:
                motion_goals, chosen_subtask = self._handle_boiled_chicken_dish(
                    state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing
                )
            elif curr_order == "boiled_chicken_onion_dish":
                motion_goals, chosen_subtask = self._handle_boiled_chicken_onion_dish(
                    state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing
                )
            
            # Move to next order if no motion goals found
            if len(motion_goals) == 0:
                order_idx += 1
        
        return motion_goals, chosen_subtask
    
    def _handle_steak_dish(self, state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing):
        """
        Handle actions for steak dish orders using a priority-based approach.
        
        This method determines the next action to take for a steak dish order
        based on the current state of the kitchen and the player. It uses a priority-based
        approach to select the most appropriate action.
        """
        am = self.mlam
        motion_goals = []
        chosen_subtask = None
        
        # If the player is holding an object, handle that first
        if player_state['has_object']:

            # Define actions based on the object being held
            object_actions = {
                "meat": (lambda: am.put_meat_in_grill_actions(kitchen_state['grill_states_dict'], knowledge_base=self.knowledge_base, only_empty=True), 'drop_meat'),
                "dirty_plate": (lambda: am.put_dirty_plate_in_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base), 'drop_dirty_plate'),
                "clean_plate": (lambda: am.pickup_steak_with_clean_plate_actions(kitchen_state['grill_states_dict'], only_nearly_ready=True, knowledge_base=self.knowledge_base), 'pickup_steak'),
                "steak": (am.deliver_dish_actions, 'deliver_dish')
            }
            
            # Get the appropriate action for the held object
            if player_state['object_name'] in object_actions:
                action_func, subtask = object_actions[player_state['object_name']]
                motion_goals = action_func() if callable(action_func) else action_func
                chosen_subtask = subtask
            
            if len(motion_goals) == 0 and self.drop_on_counter:
                motion_goals = am.place_obj_on_counter_actions(state)
                chosen_subtask = 'drop_'+player_state['object_name']
        
        # If no action was determined based on held object, determine next action based on kitchen state
        if not motion_goals:
            # Define possible actions with their conditions and priorities
            possible_actions = [
                # Pick up meat if steak not nearly ready
                {
                    'condition': not kitchen_state['steak_nearly_ready'] and state.num_orders_remaining > 0 and (not player_state['other_has_meat'] or order_idx > 0),
                    'action': lambda: am.pickup_meat_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_meat',
                    'priority': 1
                },
                # Pick up dirty plate if needed
                {
                    'condition': not rinsing and not clean_plate_ready and (not player_state['other_has_dirty_plate'] or order_idx > 0) and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_dirty_plate_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_dirty_plate',
                    'priority': 2
                },
                # Rinse plate if needed
                {
                    'condition': rinsing and not clean_plate_ready,
                    'action': lambda: am.rinse_plate_in_sink_actions(state, knowledge_base=self.knowledge_base, other_players=[self.knowledge_base['other_player']]),
                    'subtask': 'rinse_plate',
                    'priority': 3
                },
                # Pick up clean plate if steak is nearly ready
                {
                    'condition': kitchen_state['steak_nearly_ready'] and clean_plate_ready,
                    'action': lambda: am.pickup_clean_plate_from_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base),
                    'subtask': 'pickup_clean_plate',
                    'priority': 4
                }
            ]
            
            # Sort actions by priority and find the first one whose condition is met
            possible_actions.sort(key=lambda x: x['priority'])
            for action in possible_actions:
                if action['condition']:
                    motion_goals = action['action']()
                    chosen_subtask = action['subtask']
                    break
        
        if self.debug: print(f'chosen_subtask: {chosen_subtask}')
        return motion_goals, chosen_subtask
    
    def _handle_steak_onion_dish(self, state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing):
        """
        Handle actions for steak with onion dish orders using a priority-based approach.
        
        This method determines the next action to take for a steak with onion dish order
        based on the current state of the kitchen and the player. It uses a priority-based
        approach to select the most appropriate action.
        """
        am = self.mlam
        motion_goals = []
        chosen_subtask = None
        
        # Define action priorities based on the current state
        action_priorities = []
        
        # If the player is holding an object, handle that first
        if player_state['has_object']:
            # Define actions based on the object being held
            object_actions = {
                "onion": (lambda: am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base), 'drop_onion'),
                "meat": (lambda: am.put_meat_in_grill_actions(kitchen_state['grill_states_dict'], knowledge_base=self.knowledge_base, only_empty=True), 'drop_meat'),
                "dirty_plate": (lambda: am.put_dirty_plate_in_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base), 'drop_dirty_plate'),
                "clean_plate": (lambda: am.pickup_steak_with_clean_plate_actions(kitchen_state['grill_states_dict'], only_nearly_ready=True, knowledge_base=self.knowledge_base), 'pickup_steak'),
                "steak": (lambda: am.add_garnish_to_dish_actions(state, knowledge_base=self.knowledge_base) if kitchen_state['garnish_ready'] else [], 'add_garnish'),
                "steak_onion": (am.deliver_dish_actions, 'deliver_dish')
            }
            
            # Get the appropriate action for the held object
            if player_state['object_name'] in object_actions:
                action_func, subtask = object_actions[player_state['object_name']]
                motion_goals = action_func() if callable(action_func) else action_func
                chosen_subtask = subtask
                
                # Special case for steak - only add garnish if it's ready
                if player_state['object_name'] == "steak" and not kitchen_state['garnish_ready']:
                    motion_goals = []
                    chosen_subtask = None

            if len(motion_goals) == 0 and self.drop_on_counter:
                motion_goals = am.place_obj_on_counter_actions(state)
                chosen_subtask = 'drop_'+player_state['object_name']
            
        
        # If no action was determined based on held object, determine next action based on kitchen state
        if not motion_goals:
            # Define possible actions with their conditions and priorities
            possible_actions = [
                # Chop onion if it's on the board and not ready
                {
                    'condition': kitchen_state['chopping'] and not kitchen_state['garnish_ready'],
                    'action': lambda: am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base, other_players=[self.knowledge_base['other_player']]),
                    'subtask': 'chop_onion',
                    'priority': 1
                },
                # Pick up onion if not on board and not ready
                {
                    'condition': not kitchen_state['chopping'] and not kitchen_state['garnish_ready'] and (not player_state['other_has_onion'] or order_idx > 0),
                    'action': lambda: am.pickup_onion_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_onion',
                    'priority': 2
                },
                # Pick up meat if steak not nearly ready
                {
                    'condition': not kitchen_state['steak_nearly_ready'] and state.num_orders_remaining > 0 and (not player_state['other_has_meat'] or order_idx > 0),
                    'action': lambda: am.pickup_meat_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_meat',
                    'priority': 3
                },
                # Pick up dirty plate if needed
                {
                    'condition': not rinsing and not clean_plate_ready and (not player_state['other_has_dirty_plate'] or order_idx > 0) and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_dirty_plate_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_dirty_plate',
                    'priority': 4
                },
                # Rinse plate if needed
                {
                    'condition': rinsing and not clean_plate_ready,
                    'action': lambda: am.rinse_plate_in_sink_actions(state, other_players=[self.knowledge_base['other_player']], knowledge_base=self.knowledge_base),
                    'subtask': 'rinse_plate',
                    'priority': 5
                },
                # Pick up clean plate if steak is ready
                {
                    'condition': kitchen_state['steak_nearly_ready'] and clean_plate_ready and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_clean_plate_from_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base),
                    'subtask': 'pickup_clean_plate',
                    'priority': 6
                }
            ]
            
            # Sort actions by priority and find the first one whose condition is met
            possible_actions.sort(key=lambda x: x['priority'])
            for action in possible_actions:
                if action['condition']:
                    motion_goals = action['action']()
                    chosen_subtask = action['subtask']
                    break
        
        if self.debug: print(f'chosen_subtask: {chosen_subtask}')
        
        return motion_goals, chosen_subtask
    
    def _handle_boiled_chicken_dish(self, state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing):
        """
        Handle actions for boiled chicken dish orders using a priority-based approach.
        
        This method determines the next action to take for a boiled chicken dish order
        based on the current state of the kitchen and the player. It uses a priority-based
        approach to select the most appropriate action.
        """
        am = self.mlam
        motion_goals = []
        chosen_subtask = None
        
        # If the player is holding an object, handle that first
        if player_state['has_object']:
            
            # Define actions based on the object being held
            object_actions = {
                "chicken": (lambda: am.put_chicken_in_pot_actions(kitchen_state['pot_states_dict'], knowledge_base=self.knowledge_base), 'drop_chicken'),
                "dirty_plate": (lambda: am.put_dirty_plate_in_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base), 'drop_dirty_plate'),
                "clean_plate": (lambda: am.pickup_boiled_chicken_with_clean_plate_actions(kitchen_state['pot_states_dict'], only_nearly_ready=True, knowledge_base=self.knowledge_base), 'pickup_boiled_chicken'),
                "boiled_chicken": (am.deliver_dish_actions, 'deliver_dish')
            }
            
            # Get the appropriate action for the held object
            if player_state['object_name'] in object_actions:
                action_func, subtask = object_actions[player_state['object_name']]
                motion_goals = action_func() if callable(action_func) else action_func
                chosen_subtask = subtask
        
        # If no action was determined based on held object, determine next action based on kitchen state
        if not motion_goals:
            # Define possible actions with their conditions and priorities
            possible_actions = [
                # Pick up chicken if not nearly ready
                {
                    'condition': not kitchen_state['chicken_nearly_ready'] and state.num_orders_remaining > 0 and (not player_state['other_has_chicken'] or order_idx > 0),
                    'action': lambda: am.pickup_chicken_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_chicken',
                    'priority': 1
                },
                # Pick up dirty plate if needed
                {
                    'condition': not rinsing and not clean_plate_ready and (not player_state['other_has_dirty_plate'] or order_idx > 0) and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_dirty_plate_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_dirty_plate',
                    'priority': 2
                },
                # Rinse plate if needed
                {
                    'condition': rinsing and not clean_plate_ready,
                    'action': lambda: am.rinse_plate_in_sink_actions(state, other_players=[self.knowledge_base['other_player']], knowledge_base=self.knowledge_base),
                    'subtask': 'rinse_plate',
                    'priority': 3
                },
                # Pick up clean plate if chicken is nearly ready
                {
                    'condition': kitchen_state['chicken_nearly_ready'] and clean_plate_ready,
                    'action': lambda: am.pickup_clean_plate_from_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base),
                    'subtask': 'pickup_clean_plate',
                    'priority': 4
                }
            ]
            
            # Sort actions by priority and find the first one whose condition is met
            possible_actions.sort(key=lambda x: x['priority'])
            for action in possible_actions:
                if action['condition']:
                    motion_goals = action['action']()
                    chosen_subtask = action['subtask']
                    break
        
        return motion_goals, chosen_subtask
    
    def _handle_boiled_chicken_onion_dish(self, state, kitchen_state, player_state, order_idx, clean_plate_ready, rinsing):
        """
        Handle actions for boiled chicken with onion dish orders using a priority-based approach.
        
        This method determines the next action to take for a boiled chicken with onion dish order
        based on the current state of the kitchen and the player. It uses a priority-based
        approach to select the most appropriate action.
        """
        am = self.mlam
        motion_goals = []
        chosen_subtask = None
        
        # If the player is holding an object, handle that first
        if player_state['has_object']:
            
            # Define actions based on the object being held
            object_actions = {
                "onion": (lambda: am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base), 'drop_onion'),
                "chicken": (lambda: am.put_chicken_in_pot_actions(kitchen_state['pot_states_dict'], knowledge_base=self.knowledge_base), 'drop_chicken'),
                "dirty_plate": (lambda: am.put_dirty_plate_in_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base), 'drop_dirty_plate'),
                "clean_plate": (lambda: am.pickup_boiled_chicken_with_clean_plate_actions(kitchen_state['pot_states_dict'], only_nearly_ready=True, knowledge_base=self.knowledge_base), 'pickup_boiled_chicken'),
                "boiled_chicken": (lambda: am.add_garnish_to_dish_actions(state, knowledge_base=self.knowledge_base) if kitchen_state['garnish_ready'] else [], 'add_garnish'),
                "boiled_chicken_onion": (am.deliver_dish_actions, 'deliver_dish')
            }
            
            # Get the appropriate action for the held object
            if player_state['object_name'] in object_actions:
                action_func, subtask = object_actions[player_state['object_name']]
                motion_goals = action_func() if callable(action_func) else action_func
                chosen_subtask = subtask
                
                # Special case for boiled_chicken - only add garnish if it's ready
                if player_state['object_name'] == "boiled_chicken" and not kitchen_state['garnish_ready']:
                    motion_goals = []
                    chosen_subtask = None
        
        # If no action was determined based on held object, determine next action based on kitchen state
        if not motion_goals:
            # Define possible actions with their conditions and priorities
            possible_actions = [
                # Chop onion if it's on the board and not ready
                {
                    'condition': kitchen_state['chopping'] and not kitchen_state['garnish_ready'],
                    'action': lambda: am.chop_onion_on_board_actions(state, other_players=[self.knowledge_base['other_player']], knowledge_base=self.knowledge_base),
                    'subtask': 'chop_onion',
                    'priority': 1
                },
                # Pick up onion if not on board and not ready
                {
                    'condition': not kitchen_state['chopping'] and not kitchen_state['garnish_ready'] and (not player_state['other_has_onion'] or order_idx > 0),
                    'action': lambda: am.pickup_onion_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_onion',
                    'priority': 2
                },
                # Pick up chicken if not nearly ready
                {
                    'condition': not kitchen_state['chicken_nearly_ready'] and state.num_orders_remaining > 0 and (not player_state['other_has_chicken'] or order_idx > 0),
                    'action': lambda: am.pickup_chicken_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_chicken',
                    'priority': 3
                },
                # Pick up dirty plate if needed
                {
                    'condition': not rinsing and not clean_plate_ready and (not player_state['other_has_dirty_plate'] or order_idx > 0) and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_dirty_plate_actions(kitchen_state['counter_objects']),
                    'subtask': 'pickup_dirty_plate',
                    'priority': 4
                },
                # Rinse plate if needed
                {
                    'condition': rinsing and not clean_plate_ready,
                    'action': lambda: am.rinse_plate_in_sink_actions(state, other_players=[self.knowledge_base['other_player']], knowledge_base=self.knowledge_base),
                    'subtask': 'rinse_plate',
                    'priority': 5
                },
                # Pick up clean plate if boiled chicken is ready
                {
                    'condition': kitchen_state['chicken_nearly_ready'] and clean_plate_ready and (not player_state['other_has_clean_plate'] or order_idx > 0),
                    'action': lambda: am.pickup_clean_plate_from_sink_actions(kitchen_state['counter_objects'], state, knowledge_base=self.knowledge_base),
                    'subtask': 'pickup_clean_plate',
                    'priority': 6
                }
            ]
            
            # Sort actions by priority and find the first one whose condition is met
            possible_actions.sort(key=lambda x: x['priority'])
            for action in possible_actions:
                if action['condition']:
                    motion_goals = action['action']()
                    chosen_subtask = action['subtask']
                    break
        
        return motion_goals, chosen_subtask
    
    def _get_fallback_motion_goals(self, state, player, player_state):
        """Get fallback motion goals when no specific task is available."""
        am = self.mlam
        motion_goals = []
        
        if self.explore:  # explore to expand the vision
            if player_state['has_object'] and self.drop_on_counter:
                motion_goals += am.place_obj_on_counter_actions(state)
            else:
                motion_goals += am.pickup_dirty_plate_actions(self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X'])))
                
                # get four directions to explore
                for o in Direction.ALL_DIRECTIONS:
                    if o != player.orientation:
                        motion_goals.append(self.mdp._move_if_direction(player.position, player.orientation, o))
                if player.pos_and_or in motion_goals:
                    motion_goals.remove(player.pos_and_or)
                    
            random.shuffle(motion_goals)
            motion_goals = [[mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)][0]]
        else:  # get to the closest key object location
            if player_state['has_object'] and self.drop_on_counter:
                motion_goals += am.place_obj_on_counter_actions(state)
            else:
                motion_goals += am.go_to_closest_feature_actions(player)
                
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            
            if len(motion_goals) == 0:
                motion_goals += am.go_to_closest_feature_actions(player)
        
        return motion_goals


#new custom class misha edit
class StaySteakLimitVisionHumanModel(SteakLimitVisionHumanModel):
    """
    Identical to SteakLimitVisionHumanModel in every way that matters
    for data collection — full knowledge base, FOV/vision cone tracking,
    delayed acknowledgment, estimated KB — but the agent never moves.
    Every tick we refresh the KB based on what's inside the vision cone,
    then return STAY.
    """

    def action(self, state):
        # ---- 1) Refresh perception ----
        # This is the same call that ml_action() would have made.
        # It respects:
        #   - self.vision_bound   (your --fov argument, halved internally)
        #   - self.vision_mode    (e.g. "cone")
        #   - self.kb_update_delay (how many consecutive ticks an object
        #                           must be seen before it enters the KB)
        #   - self.kb_ackn_prob   (stochastic acknowledgment, if enabled)
        # It mutates:
        #   - self.knowledge_base            (true-FOV KB)
        #   - self.estimated_knowledge_base  (estimated-FOV KB for Bayes)
        #   - self.kb_update_delay_track     (the [track, untrack, remove] triple)
        #   - self.estimated_kb_update_delay_track
        self.update(state)

        # ---- 2) Keep subtask_log consistent ----
        # Normally ml_action appends a chosen subtask to the player's
        # subtask_log each tick. If your Logger reads that list, it expects
        # one entry per timestep, so we append a placeholder to keep lengths aligned.
        state.players[self.agent_index].subtask_log += [None]

        # ---- 3) No decision-making, no motion goal, just STAY ----
        return Action.STAY, {}

    def actions(self, states, agent_indices):
        # Batched version — only used by some planners/rollouts. Safe to mirror.
        return [self.action(s) for s in states]






class HRLModel(SteakLimitVisionHumanModel):
    """
    This agent contains an NN for subtask planning and a motion planner for subtask to motion action mapping.
    """
    def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, vision_mode="cone", robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, subtask_planner_model=None, obs_size=None, action_size=None, device="cpu", pretrained_subtask_state_dict=None, pretrained_subtask_path=None, pretrained_optimizer=None, debug=False, **kwargs):
        SteakLimitVisionHumanModel.__init__(self, mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, vision_mode=vision_mode, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, debug=debug)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.ml_action_list = ML_ACTION_LIST

        if subtask_planner_model is not None:
            if subtask_planner_model == "NNAgent":
                self.subtask_planner = NNAgent(obs_size=obs_size, action_size=len(self.ml_action_list)).to(device) 
            elif subtask_planner_model == "LSTM":
                self.subtask_planner = LSTM_Agent(obs_size=obs_size, action_size=action_size, lstm_size=self.lstm_size).to(device)
            
            self.optimizer = optim.Adam(self.subtask_planner.parameters(), lr=2.5e-4, eps=1e-5)
            self.next_done = torch.zeros(1).to(device)
            self.next_lstm_state = (
                torch.zeros(self.subtask_planner.lstm.num_layers, 1, self.subtask_planner.lstm.hidden_size).to(device),
                torch.zeros(self.subtask_planner.lstm.num_layers, 1, self.subtask_planner.lstm.hidden_size).to(device),
            )

            if pretrained_subtask_path is not None and os.path.exists(pretrained_subtask_path):
                self.subtask_planner.load_state_dict(torch.load(pretrained_subtask_path, map_location=device, weights_only=True))
                self.subtask_planner.eval()

            if pretrained_subtask_state_dict is not None:
                self.subtask_planner.load_state_dict(pretrained_subtask_state_dict)
                self.subtask_planner.eval()
        else:
            self.subtask_planner = None

    def get_obs(self, state, horizon: int = 400):
        """
        """
        self.update(state)
        # self.update_kb_log()
        
        player = state.players[self.agent_index]
        other_players = self.knowledge_base['other_player']
        am = self.mlam

        counter_objects = {}
        all_objects = []
        # counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        sink_states = self.knowledge_base['sink_states']
        chopping_board_states = self.knowledge_base['chop_states']
        pot_states_dict = self.knowledge_base['pot_states']
        grill_states_dict = self.knowledge_base['grill_states']
        for k, o in self.knowledge_base.items():
            if k not in ['pot_states', 'sink_states', 'chop_states', 'grill_states', 'other_player']:
                if o.position in self.mlam.mdp.get_counter_locations():
                    counter_objects[o.name] = [o.position]
                all_objects.append(o)

        base_map_features = [
            "counter_loc",
            "pot_loc",
            "dirty_plate_disp_loc",
            "onion_disp_loc",
            "serve_loc",
            "grill_loc",
            "chicken_disp_loc",
            "sink_loc",
            "meat_disp_loc",
            "chopping_board_loc",
        ]
        variable_map_features = [
            "onions",
            "chickens",
            "meats",
            "dirty_plates",
            "steak_onions",
            "boiled_chicken_onions",
            "chicken_cook_time_remaining",
            "chicken_done",
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
            f"human_loc",
            f"other_player_loc",
        ] + [
            f"human_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            for d in Direction.ALL_DIRECTIONS
        ] + [
            f"other_player_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
            for d in Direction.ALL_DIRECTIONS
        ]

        DISH_TYPES = [
            "steak_dish",
            "boiled_chicken_dish",
            "steak_onion_dish",
            "boiled_chicken_onion_dish",
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

        for loc in self.mlam.mdp.get_pot_locations():
            state_mask_dict["pot_loc"][loc] = 1

        for loc in self.mlam.mdp.get_dirty_plate_locations():
            state_mask_dict["dirty_plate_disp_loc"][loc] = 1

        for loc in self.mlam.mdp.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1

        for loc in self.mlam.mdp.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1

        for loc in self.mlam.mdp.get_grill_locations():
            state_mask_dict["grill_loc"][loc] = 1

        for loc in self.mlam.mdp.get_chicken_dispenser_locations():
            state_mask_dict["chicken_disp_loc"][loc] = 1

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
        human_orientation_idx = Direction.DIRECTION_TO_INDEX[
            player.orientation
        ]
        state_mask_dict[f"human_loc"] = make_layer(player.position, 1)
        state_mask_dict[f"human_orientation_{human_orientation_idx}"] = (
            make_layer(player.position, 1)
        )

        other_player_orientation_idx = Direction.DIRECTION_TO_INDEX[
            other_players.orientation
        ]
        state_mask_dict[f"other_player_loc"] = make_layer(other_players.position, 1)
        state_mask_dict[f"other_player_orientation_{other_player_orientation_idx}"] = (
            make_layer(other_players.position, 1)
        )


        # OBJECT & STATE LAYERS
        for obj in all_objects:
            if obj.name == "boiled_chicken":
                # Boiled chicken is similar to soup except that it immediately
                # starts cooking and only needs 1 chicken.
                if obj.position in self.mlam.mdp.get_pot_locations():
                    # Only one chicken can be in pot and it is never idle. When
                    # player interacts with pot holding chicken, a `ChickenState` is
                    # created, chicken is added, and cooking starts.
                    state_mask_dict["chicken_cook_time_remaining"] += make_layer(
                        obj.position, obj.cook_time - obj._cooking_tick
                    )
                    if obj.is_ready:
                        state_mask_dict["chicken_done"] += make_layer(
                            obj.position, 1
                        )
                else:
                    # If boiled chicken is not in a pot, treat it like a soup that
                    # is cooked with remaining time 0
                    state_mask_dict["chicken_done"] += make_layer(obj.position, 1)

            elif obj.name == "steak":
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
            elif obj.name == "chicken":
                state_mask_dict["chickens"] += make_layer(obj.position, 1)
            elif obj.name == "meat":
                state_mask_dict["meats"] += make_layer(obj.position, 1)
            elif obj.name == "dirty_plate":
                state_mask_dict["dirty_plates"] += make_layer(obj.position, 1)
            elif obj.name == "steak_onion":
                # Garnished steak doesn't need cooking, so treated as a regular
                # object.
                state_mask_dict["steak_onions"] += make_layer(obj.position, 1)
            elif obj.name == "boiled_chicken_onion":
                # Garnished chicken doesn't need cooking, so treated as a regular
                # object.
                state_mask_dict["boiled_chicken_onions"] += make_layer(
                    obj.position, 1
                )
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

    def ml_action(self, state, subtask_id, return_motion_goals=False):
        # self.update_kb_log()
        
        player = state.players[self.agent_index]
        other_players = [self.knowledge_base['other_player']]
        am = self.mlam

        counter_objects = {}
        all_objects = []
        # counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        sink_states = self.knowledge_base['sink_states']
        chopping_board_states = self.knowledge_base['chop_states']
        pot_states_dict = self.knowledge_base['pot_states']
        grill_states_dict = self.knowledge_base['grill_states']
        for k, o in self.knowledge_base.items():
            # NOTE: this most likely will fail in some tomato scenarios
            if k not in ['pot_states', 'sink_states', 'chop_states', 'grill_states', 'other_player']:
                if o.position in self.mlam.mdp.get_counter_locations():
                    counter_objects[o.name] = [o.position]
                all_objects.append(o)

        ready_grill = grill_states_dict["ready"]
        cooking_grill = grill_states_dict["cooking"]
        ready_pot = pot_states_dict["ready"]
        cooking_pot = pot_states_dict["cooking"]

        # ======= World state =======
        steak_nearly_ready = len(ready_grill) > 0 or len(cooking_grill) > 0
        chicken_nearly_ready = len(ready_pot) > 0 or len(cooking_pot) > 0
        garnish_ready = len(chopping_board_states["ready"]) > 0
        steak_ready = len(grill_states_dict["ready"]) > 0
        boiled_chicken_ready = len(pot_states_dict["ready"]) > 0
        chopping = len(chopping_board_states["full"]) > 0
        board_empty = len(chopping_board_states["empty"]) > 0
        clean_plate_ready = len(sink_states["ready"]) > 0
        rinsing = len(sink_states["full"]) > 0
        sink_empty = len(sink_states["empty"]) > 0

        # ======= Considering what other players are holding =======
        other_has_dirty_plate = self._others_have_object(other_players, "dirty_plate")
        other_has_clean_plate = self._others_have_object(other_players, "clean_plate")
        other_has_steak = self._others_have_object(other_players, "steak")
        other_has_meat = self._others_have_object(other_players, "meat")
        other_has_onion = self._others_have_object(other_players, "onion")
        other_has_chicken = self._others_have_object(other_players, "chicken")
        other_has_boiled_chicken = self._others_have_object(
            other_players, "boiled_chicken"
        )

        motion_goals = []
        chosen_subtask = self.ml_action_list[subtask_id]
        if chosen_subtask is not None:
            if chosen_subtask == 'pickup_meat':
                motion_goals += am.pickup_meat_actions(counter_objects)
                
            elif chosen_subtask == 'pickup_onion':
                motion_goals += am.pickup_onion_actions(counter_objects)
                
            elif chosen_subtask == 'pickup_dirty_plate':
                motion_goals += am.pickup_dirty_plate_actions(counter_objects, state)
                
            elif chosen_subtask == 'pickup_clean_plate':
                motion_goals += am.pickup_clean_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_dirty_plate:
                        motion_goals += self.mlam.mdp.get_sink_states(state)['full']
                        if len(motion_goals) == 0 and len(self.mlam.mdp.get_sink_states(state)['empty']) > 0:
                            tmp_goal = self.mlam.mdp.get_sink_states(state)['empty'][0]
                            motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal)
                            
            elif chosen_subtask == 'pickup_steak':
                motion_goals = am.pickup_steak_with_clean_plate_actions(self.mlam.mdp.get_grill_states(state), knowledge_base=self.knowledge_base)#, only_nearly_ready=True
                if len(motion_goals) == 0:
                    if other_has_meat:
                        motion_goals = self.mlam.mdp.get_grill_states(state)['steak']['cooking'] + self.mlam.mdp.get_grill_states(state)['steak']['partially_full']
                        if len(motion_goals) == 0 and len(self.mlam.mdp.get_grill_states(state)['empty']) > 0:
                            tmp_goal = self.mlam.mdp.get_grill_states(state)['empty'][0]
                            motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal)
                            
            elif chosen_subtask == 'add_garnish':
                motion_goals = am.add_garnish_to_dish_actions(state, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_onion:
                        motion_goals = self.mlam.mdp.get_chopping_board_states(state)['full']
                        if len(motion_goals) == 0 and len(self.mlam.mdp.get_chopping_board_states(state)['empty']) > 0:
                            tmp_goal = self.mlam.mdp.get_chopping_board_states(state)['empty'][0]
                            motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal)
                            
            elif chosen_subtask == 'drop_meat':
                motion_goals = am.put_meat_in_grill_actions(self.mlam.mdp.get_grill_states(state), knowledge_base=self.knowledge_base)
                
            elif chosen_subtask == 'drop_onion':
                motion_goals = am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
                
            elif chosen_subtask == 'drop_dirty_plate':
                motion_goals = am.put_dirty_plate_in_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                
            elif chosen_subtask == 'chop_onion':
                motion_goals += am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base, other_players=other_players)
                
            elif chosen_subtask == 'rinse_plate':
                motion_goals += am.rinse_plate_in_sink_actions(state, knowledge_base=self.knowledge_base, other_players=other_players)
                
            elif chosen_subtask == 'deliver_dish':
                motion_goals = am.deliver_dish_actions()
                
            elif chosen_subtask == 'drop_chicken':
                motion_goals = am.put_chicken_in_pot_actions(pot_states_dict, knowledge_base=self.knowledge_base)
                
            elif chosen_subtask == 'pickup_boiled_chicken':
                motion_goals = am.pickup_boiled_chicken_with_clean_plate_actions(pot_states_dict, only_nearly_ready=True, knowledge_base=self.knowledge_base)
                if len(motion_goals) == 0:
                    if other_has_chicken:
                        motion_goals = self.mlam.mdp.get_pot_states(state)['chicken']['cooking'] + self.mlam.mdp.get_pot_states(state)['chicken']['partially_full']
                        if len(motion_goals) == 0 and len(self.mlam.mdp.get_pot_states(state)['empty']) > 0:
                            tmp_goal = self.mlam.mdp.get_pot_states(state)['empty'][0]
                            motion_goals += self.mlam.go_to_closest_feature_or_counter_to_goal(self.mlam.motion_planner.motion_goals_for_pos[tmp_goal][0], tmp_goal)
            
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            
            if len(motion_goals) == 0:
                motion_goals += am.go_to_closest_feature_actions(player)

        self.prev_chosen_subtask = chosen_subtask
        # print('SteakLimitVisionHumanModel\'s motion_goals:', motion_goals)
        state.players[self.agent_index].subtask_log += [chosen_subtask]

        if return_motion_goals:
            return chosen_subtask, motion_goals

        return chosen_subtask
    
    def subtask_to_action(self, state, subtask_id):
        # obs = self.get_obs(state)
        # with torch.no_grad():
        #     subtask_id, _, _, _ = self.subtask_planner.get_action_and_value(obs)
        possible_motion_goals = self.ml_action(state, subtask_id)

        if len(possible_motion_goals) == 0:
            return (0,0), {}
        
        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, possible_motion_goals
        )

        if (
            self.ll_boltzmann_rational
            and chosen_goal[0] == start_pos_and_or[0]
        ):
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal
            )

        # if self.auto_unstuck and self.prev_state is not None:
        #     # HACK: if two agents get stuck, select an action at random that would
        #     # change the player positions if the other player were not to move
        #     interact_flag = (chosen_action == 'interact')
        #     for p0, p1 in zip(state.players, self.prev_state.players):
        #         if p0.held_object != p1.held_object:
        #             interact_flag = True
        #     if (
        #         (self.prev_state is not None)
        #         and (state.players_pos_and_or
        #         == self.prev_state.players_pos_and_or) and (not interact_flag)
        #     ):
        #         if self.agent_index == 0:
        #             joint_actions = list(
        #                 itertools.product(Action.ALL_ACTIONS, [Action.STAY])
        #             )
        #         elif self.agent_index == 1:
        #             joint_actions = list(
        #                 itertools.product([Action.STAY], Action.ALL_ACTIONS)
        #             )
        #         else:
        #             raise ValueError("Player index not recognized")

        #         unblocking_joint_actions = []
        #         for j_a in joint_actions:
        #             new_state, _ = self.mlam.mdp.get_state_transition(
        #                 state, j_a
        #             )
        #             if (
        #                 new_state.player_positions
        #                 != self.prev_state.player_positions
        #             ):
        #                 unblocking_joint_actions.append(j_a)
        #         # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
        #         if len(unblocking_joint_actions) == 0:
        #             unblocking_joint_actions.append([Action.STAY, Action.STAY])
        #         chosen_action = unblocking_joint_actions[
        #             np.random.choice(len(unblocking_joint_actions))
        #         ][self.agent_index]
        #         action_probs = self.a_probs_from_action(chosen_action)

        # NOTE: Assumes that calls to the action method are sequential
        self.prev_state = state
        return chosen_action, {"action_probs": action_probs}
    
    def action(self, state):
        obs = self.get_obs(state)
        if self.subtask_planner is not None:
            with torch.no_grad():
                subtask_id, logprob, _, value, self.next_lstm_state = self.subtask_planner.get_action_and_value(torch.Tensor(obs).unsqueeze(0), self.next_lstm_state, self.next_done)

            return self.subtask_to_action(state, subtask_id[0])
        return (0,0), {}


#limit vision human model that has perfect dynamics for bayesian optimization
