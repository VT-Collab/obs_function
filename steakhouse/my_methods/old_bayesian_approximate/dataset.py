#exact same as baseline_implicit_obs
#just one 1D vector 
#Like from ['[11, 5]', '[0, 1]', 'None', '[5, 3]', '[0, 1]', 'None', '1', '1', '{...}', '{...}', '{...}', '{...}']
#to [11, 5, 0, 1, <p0_held_vector_size10>, 5, 3, 0, 1, <p1_held_vector_size10>, 1, 1, <chopping_vec_size10>, <grill_vec_size10>, <sink_vec_size5>, <pot_vec_size10>]

#each held object will be one hot vector that wil be blended into the 1D vector, total size 10
# held_objects = {
#     'None':                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'meat':                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     'chicken':                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     'dirty_plate':            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     'clean_plate':            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     'onion':                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     'steak':                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     'boiled_chicken':         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#     'steak_onion':            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     'boiled_chicken_onion':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
# }

# chop_sink_states = {
#     'empty':                  [1, 0, 0],
#     'ready':                  [0, 1, 0],
#     'full':                   [0, 0, 1],
# }

# grill_pots_states = {
#     'empty':                   [1, 0, 0],
#     'ready':                   [0, 1, 0],
#     'cooking':                 [0, 0, 1],
# }

#CHANGE TO THIS FOR ALL 4 COOKING EQUIPMENTS
# x1 y1 one-hot-encoding of full empty cooking, etc.  2 3 1 0 0 
# x2 y2 one-hot-encoding

#2 chopping boards of 3 states: empty, full, or ready: size 10
#2 grills of 3 states: empty cooking ready: size 10
#1 sink of 3 states: emptuy, full, or ready: size 5
#2 pots of 3 states: empty cooking ready: size 10

#for grill it will be the same as above except its empty full COOKING 
#for sink it will be only one sink, empty full ready  
#for pot its empty full COOKING 

#total 1D vector dimention: 65


# Goal: take an obs list input that looks like this:
# [[11, 5], [0, 1], None, [5, 3], [0, 1], None, 1, 1, {'empty': [[8, 5], [8, 6]], 'full': [], 'ready': []}, {'empty': [[13, 4], [13, 5]]}, {'empty': [[13, 2]], 'full': [], 'ready': []}, {'empty': [[3, 0], [4, 0]]}]
# and turn into to the corresponding 1D vector that is this
# [11, 5, 0, 1, <p0_held_vector_size10>, 5, 3, 0, 1, <p1_held_vector_size10>, 1, 1, <chopping_vec_size10>, <grill_vec_size10>, <sink_vec_size5>, <pot_vec_size10>]

# where <p0_held_vector_size10> and <p1_held_vector_size10>  will be [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] b/c it's None,

# and <chopping_vec_size12> will be 
# [x1, y1, chop_sink_states_corresponding_one_hot_vector, x2, y2, chop_sink_states_corresponding_one_hot_vector]
# or [8, 5, 1, 0, 0, 8, 6, 1, 0, 0]

# and <grill_vec_size12> will be
# [x1, y1, grill_pots_states_corresponding_one_hot_vector, x2, y2, grill_pots_states_corresponding_one_hot_vector]
# or in this case it's all empty (later if it its cooking it will say 'cooking': )
# [13, 4, 1, 0, 0, 13, 5, 1, 0, 0] 

# and <sink_vec_size6> will be
# [x1, y1, chop_sink_states_corresponding_one_hot_vector]
# [13, 2, 1, 0, 0] 

# and finally <pot_ve_size12> will be
# [x1, y1, grill_pots_states_corresponding_one_hot_vector, x2, y2, grill_pots_states_corresponding_one_hot_vector]
# [3, 0, 1, 0, 0, 4, 0, 1, 0, 0] 


import pickle
import torch
import ast
import pandas as pd


#one-hot-encoding for all possible held_objects
held_objects = {
    'None':                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'meat':                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'chicken':                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'dirty_plate':            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'clean_plate':            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'onion':                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'steak':                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'boiled_chicken':         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'steak_onion':            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'boiled_chicken_onion':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

chop_sink_states = {
    'empty':                  [1, 0, 0],
    'ready':                  [0, 1, 0],
    'full':                   [0, 0, 1],
}

grill_pots_states = {
    'empty':                   [1, 0, 0],
    'ready':                   [0, 1, 0],
    'cooking':                 [0, 0, 1],
}


def obs_to_list(obs_raw):
    #first strip obs of the uncessary "", ', {}, etc. using ast, python tool to turn string into list/dictionary
    #you basically have to get object twice or call ast twice 
    outer = ast.literal_eval(obs_raw)
    #print("ONE USE OF AST\n", outer)
    
    obs = []
    for x in outer:
        try:
            value = ast.literal_eval(x)
            obs.append(value)
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, check if it's a known food item, cuz ast.literal_eval can't do strings
            if x in held_objects:
                obs.append(x)  # keep as string
            else:
                raise ValueError(f"Unknown value: {x}")
    
    return obs


#extract positions for chopping board, grill, sink, and pots 
#inputs are obs[8], ['empty', 'full', 'ready'], 2

#takes dictionary of keys 'empty', 'full', and values [0, 1] or coordinates
#equipment_type: a string such as 'chop', 'sink', or 'grill' that determines the encoding logic.
#num_spots: how many equipment spots to encode. Defaults to 2.
#returns coordinates and one hot encoding of the state
def encode_station(state_dict, equipment_type, num_spots=2):
    output = []
    state_map = chop_sink_states if equipment_type in ['chop', 'sink'] else grill_pots_states
    coords_used = 0
    for state_name in ['empty', 'full', 'ready'] if equipment_type in ['chop', 'sink'] else ['empty', 'cooking', 'ready']:
        coords = state_dict.get(state_name, [])
        for x, y in coords:
            if coords_used >= num_spots:
                break
            #eg. if (3, 4) is 'ready', and 'ready' one hot encoding = [0, 0, 1], you append [3, 4, 0, 0, 1].
            output += [x, y] + state_map[state_name]
            coords_used += 1
        if coords_used >= num_spots:
            break
    while coords_used < num_spots:
        output += [-1, -1] + [0, 0, 0]
        coords_used += 1
    return output if equipment_type != 'sink' else output[:5]

#combine everything into a 65 dim 1D vector. 
# Example:
# -- OBS INPUT --
# [[11, 5], [0, 1], None, [5, 3], [0, 1], None, 1, 1, {'empty': [[8, 5], [8, 6]], 'full': [], 'ready': []}, {'empty': [[13, 4], [13, 5]]}, {'empty': [[13, 2]], 'full': [], 'ready': []}, {'empty': [[3, 0], [4, 0]]}]

# -- VECTOR OUTPUT --
# [11, 5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  5, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  1, 1, 8, 5, 1, 0, 0, 8, 6, 1, 0, 0,
#  13, 4, 1, 0, 0, 13, 5, 1, 0, 0,
#  13, 2, 1, 0, 0,
#  3, 0, 1, 0, 0, 4, 0, 1, 0, 0]

#make the 65-dim vector 
def obs_list_to_1D_vec(obs):
    flat = []
    flat.extend(obs[0])  # p0 pos
    flat.extend(obs[1])  # p0 ori
    
    #print("did it get here 1")

    flat.extend(held_objects[str(obs[2])])  # p0 held
    flat.extend(obs[3])  # p1 pos
    flat.extend(obs[4])  # p1 ori
    flat.extend(held_objects[str(obs[5])])  # p1 held
    flat.append(int(obs[6]))  # p0 action
    flat.append(int(obs[7]))  # p1 action
    
    #print("did it get here 2")
    flat += encode_station(obs[8], 'chop')
    flat += encode_station(obs[9], 'grill')
    flat += encode_station(obs[10], 'sink', num_spots=1)
    flat += encode_station(obs[11], 'pots')
    return flat
