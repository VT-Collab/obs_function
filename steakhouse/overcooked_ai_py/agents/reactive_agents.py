import os
import torch
import random
import numpy as np
import itertools, copy
import torch.optim as optim
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import GreedyHumanModel, AgentFromPolicy
from rl.util import NNAgent, LSTM_Agent
from agents.steak_agent import SteakGreedyHumanModel, HRLModel, SteakLimitVisionHumanModel

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

class ReactiveBaseModel():
    def __init__(self):
        self.reset()

    def reset(self):
        self.past_utters_tuple = [(0,0,0)]*10
        self.ml_action_pipeline = {'immediate': None, 'next': None, 'conditional': []}

    def utter_complete(self):
        for i in range(len(self.past_utters_tuple)-1, -1, -1):
            if self.past_utters_tuple[i] != (-1,0,0): # there is an utterance
                # check if the utterance is complete
                if self.past_utters_tuple[i] != (0,0,0): # it's an actionable utterance
                    utter_length = self.past_utters_tuple[i][2]
                    if utter_length == (len(self.past_utters_tuple) - i):
                        return True, self.past_utters_tuple[i]
                    elif utter_length < (len(self.past_utters_tuple) - i): # if the utterance is longer than the remaining utterances
                        return True, (-1,0,0) # return continue as the agent would decide when it is completed
                else:
                    return True, (0,0,0)
                break
        return False, (0,0,0)

    def process_utters(self):
        # check if the last utterance is complete
        if self.past_utters_tuple[-1] != None:
            if isinstance(self.past_utters_tuple[-1], tuple):
                complete, (condition_id, subtask_id, length) = self.utter_complete()
                if complete:
                    self.utter_to_ml_action_pipeline((condition_id, subtask_id, length))
            else:
                raise ValueError("Unrecognized utterance format")
    
    def condition_id_to_str(self, condition_id):
        if condition_id == 0:
            return 'immediate'
        elif condition_id == 1:
            return 'next'
        else:
            return 'conditional'

    def subtask_id_to_str(self, subtask_id):
        return ML_ACTION_LIST[subtask_id]

    def utter_to_ml_action_pipeline(self, utter_tuple):
        if utter_tuple == (0,0,0):
            self.ml_action_pipeline['immediate'] = None
            return

        if utter_tuple == (-1,0,0):
            return
            
        (condition_id, subtask_id, length) = utter_tuple
        subtask = self.subtask_id_to_str(subtask_id)
        condition = self.condition_id_to_str(condition_id)

        if condition == 'immediate':
            self.ml_action_pipeline['immediate'] = subtask
        elif condition == 'next':
            self.ml_action_pipeline['next'] = subtask
        else:
            self.ml_action_pipeline['conditional'].append((condition, subtask))

# class ReactiveSteakLimitVisionHumanModel(SteakLimitVisionHumanModel, ReactiveBaseModel):
#     def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, debug=False):
#         super().__init__(mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, debug=debug)
#         ReactiveBaseModel.__init__(self)

#         self.prev_chosen_goal = None

#     def reset(self):
#         super().reset()
#         ReactiveBaseModel.reset(self)
 
    
#     def action(self, state):
#         # Given the observed utterance, recieve, process and update the past utterances
#         if state.players[1-self.agent_index].utter is not None:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [state.players[1-self.agent_index].utter]
#         else:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [(0,0,0)]
#         # print('past_utters_tuple:', self.past_utters_tuple)
#         self.process_utters()

#         # Determine the medium level action
#         possible_motion_goals = []

#         if self.ml_action_pipeline['immediate'] is not None:
#             possible_motion_goals = self.ml_action(state, self.ml_action_pipeline['immediate'])

#         if len(possible_motion_goals) == 0:
#             chosen_action = Action.STAY
#             action_probs = []
#             # # completed ml_action_pipeline's ml_action, reset to None
#             # if self.ml_action_pipeline['immediate'] is not None:
#             #     self.ml_action_pipeline['immediate'] = None

#         else:
#             # Once we have identified the motion goals for the medium
#             # level action we want to perform, select the one with lowest cost
#             start_pos_and_or = state.players_pos_and_or[self.agent_index]

#             chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
#                 start_pos_and_or, possible_motion_goals
#             )

#             if (
#                 self.ll_boltzmann_rational
#                 and chosen_goal[0] == start_pos_and_or[0]
#             ):
#                 chosen_action, action_probs = self.boltzmann_rational_ll_action(
#                     start_pos_and_or, chosen_goal
#                 )

#             if chosen_action == 'interact':
#                 self.ml_action_pipeline['immediate'] = self.ml_action_pipeline['next']
#                 self.ml_action_pipeline['next'] = None
                
#             self.prev_chosen_goal = chosen_goal

#         self.prev_state = state

#         return chosen_action, {"action_probs": action_probs}


# class ReactiveStaySteakLimitVisionHumanModel(ReactiveSteakLimitVisionHumanModel):
#     def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, debug=False):
#         super().__init__(mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, debug=debug)

#         self.prev_chosen_goal = None

#     def action(self, state):

#         # Given the observed utterance, recieve, process and update the past utterances
#         if state.players[1-self.agent_index].utter is not None:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [state.players[1-self.agent_index].utter]
#         else:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [(0,0,0)]
#         # print('past_utters_tuple:', self.past_utters_tuple)
#         self.process_utters()

#         # Determine the medium level action
#         possible_motion_goals = []

#         if self.ml_action_pipeline['immediate'] is not None:
#             possible_motion_goals = self.ml_action(state, self.ml_action_pipeline['immediate'])

#         if len(possible_motion_goals) == 0:
#             chosen_action = Action.STAY
#             action_probs = []
#             # # completed ml_action_pipeline's ml_action, reset to None
#             # if self.ml_action_pipeline['immediate'] is not None:
#             #     self.ml_action_pipeline['immediate'] = None

#         else:
#             # Once we have identified the motion goals for the medium
#             # level action we want to perform, select the one with lowest cost
#             start_pos_and_or = state.players_pos_and_or[self.agent_index]

#             chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
#                 start_pos_and_or, possible_motion_goals
#             )

#             if (
#                 self.ll_boltzmann_rational
#                 and chosen_goal[0] == start_pos_and_or[0]
#             ):
#                 chosen_action, action_probs = self.boltzmann_rational_ll_action(
#                     start_pos_and_or, chosen_goal
#                 )

#             if chosen_action == 'interact':
#                 self.ml_action_pipeline['immediate'] = self.ml_action_pipeline['next']
#                 self.ml_action_pipeline['next'] = None

#             self.prev_chosen_goal = chosen_goal

#         self.prev_state = state

#         return chosen_action, {"action_probs": action_probs}


# class ReactiveHRLModel(HRLModel, ReactiveBaseModel):
#     def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, subtask_planner_model=None, obs_size=None, action_size=None, device="cpu", pretrained_subtask_state_dict=None, pretrained_subtask_path=None, pretrained_optimizer=None, debug=False, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#         super().__init__(mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, subtask_planner_model=subtask_planner_model, obs_size=obs_size, action_size=action_size, device=device, pretrained_subtask_state_dict=pretrained_subtask_state_dict, pretrained_subtask_path=pretrained_subtask_path, pretrained_optimizer=pretrained_optimizer, debug=debug)

#         self.ml_action_pipeline = {'immediate': None, 'next': None, 'conditional': []}
#         self.past_utters_tuple = [(0,0,0)]*10

#     def reset(self):
#         super().reset()
 
#     def action(self, state):

#         obs = self.get_obs(state)
#         if state.players[1-self.agent_index].utter is not None:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [state.players[1-self.agent_index].utter]
#         else:
#             self.past_utters_tuple = self.past_utters_tuple[1:] + [(0,0,0)]
#         # print('past_utters_tuple:', self.past_utters_tuple)
#         self.process_utters()

#         if self.ml_action_pipeline['immediate'] is not None:
#             self.ml_action_pipeline['immediate'] = None
#             return self.subtask_to_action(state, self.ml_action_pipeline['immediate'])

#         else:
#             with torch.no_grad():
#                 subtask_id, logprob, _, value, self.next_lstm_state = self.subtask_planner.get_action_and_value(torch.Tensor(obs).unsqueeze(0), self.next_lstm_state, self.next_done)

#             if (self.ml_action_list[subtask_id[0]] != self.prev_chosen_subtask) and (self.ml_action_pipeline['next'] is not None):
#                 self.ml_action_pipeline['next'] = None
#                 return self.subtask_to_action(state, self.ml_action_pipeline['next'])
            
#             return self.subtask_to_action(state, subtask_id[0])

class ReactiveSteakLimitVisionHumanModel(SteakLimitVisionHumanModel):
    def __init__(self, mlam, start_state, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True, explore=False, vision_limit=True, robot_aware=False, vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=False, debug=False):
        super().__init__(mlam, start_state, hl_boltzmann_rational=hl_boltzmann_rational, ll_boltzmann_rational=ll_boltzmann_rational, hl_temp=hl_temp, ll_temp=ll_temp, auto_unstuck=auto_unstuck, explore=explore, vision_limit=vision_limit, robot_aware=robot_aware, vision_bound=vision_bound, kb_update_delay=kb_update_delay, kb_ackn_prob=kb_ackn_prob, drop_on_counter=drop_on_counter, debug=debug)

    def action(self, state, vision_bound=None, vision_mode=None):
        possible_motion_goals = self.ml_action(state, vision_bound=vision_bound, vision_mode=vision_mode)

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

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if (
                self.prev_state is not None
                and state.players_pos_and_or
                == self.prev_state.players_pos_and_or
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