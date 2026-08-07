"""
Building the observation state to be passed into the baseline networks

Work for all baselines

Output is a (N_LAYERS, width, height) float32 stack.

Layers are x-major: indexed [x][y] with a raw (x, y) tuple, matching
mdp.shape == (width, height). NOTE mdp.terrain_mtx is the opposite ([y][x]) --
never mix the two.

Channel map (N_LAYERS = 23). The order below IS the order the layers are
appended in build_full_state; keep the two in sync.
[0]     ego agent position          <- ego-centric: agent_index is always first
[1]     ego agent facing cell
[2]     other agent position
[3]     other agent facing cell
[4-10]  object planes, one per HELD_VOCAB entry except "none":
        meat, onion, plate, washed_plate, steak, garnish, dish
        (covers BOTH placed objects and held ones -- a held object sits at
         its holder's cell, so the two merge naturally)
[11-18] terrain masks, one per TERRAIN entry: P B W M O D S X (static)
[19]    station status: 0 empty / 0.5 in progress / 1 ready
[20]    station timer: elapsed / total, normalized to [0, 1]
[21]    time left in the episode, normalized to [0, 1]
[22]    orders remaining, normalized to [0, 1]
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

#Definitions for the steakhouse environment 
#number of layers:
N_LAYERS = 23

#--------------------TERRAIN-------------------
TERRAIN = ['P', 'B', 'W', 'M', 'O', 'D', 'S', 'X']

#--------------------STATIONS-------------------
STATIONS = ["pot", "board", "sink"]
_STATE_VAL = {"empty": 0.0, "cooking": 0.5, "chopping": 0.5, "washing": 0.5, "ready": 1.0, "occupied": 0.5}

#location
def station_locs(mdp):
    return {
        "pot": list(mdp.get_pot_locations()),
        "board": list(mdp.get_chopping_board_locations()),
        "sink": list(mdp.get_sink_locations()),
    }

#state of station given at location loc
def _station_state(mdp, state, loc):
    obj = state.objects.get(loc)
    
    if obj is None:
        return "empty"
    name = getattr(obj, "name", "")

    #whether it is ready. deliberately NOT wrapped in try/except: each of these
    #three mdp methods starts with a has_object check and then asserts on the
    #object name, and we only reach them once the name already matched -- so a
    #raise here means a real bug, and swallowing it would silently return
    #"occupied" (0.5), which is indistinguishable from "in progress".
    if name == "steak":
        return "ready" if mdp.steak_ready_at_location(state, loc) else "cooking"
    if name == "garnish":
        return "ready" if mdp.garnish_ready_at_location(state, loc) else "chopping"
    if name == "washed_plate":
        return "ready" if mdp.plate_washed_at_location(state, loc) else "washing"

    #some other object is parked on the station cell (meat, onion, plate, dish)
    return "occupied"

#timer of the station: elapsed / total, in [0, 1]. MONOTONIC -- a finished
#station reads 1.0 and stays there, it does not drop back to 0.
#Every counter counts UP and is capped at its own total by overcooked_mdp
#(steak :2419 and :2528, garnish :2450, washed_plate :2357), so elapsed == total is
#exactly the "ready" condition that steak_ready_at_location /
#garnish_ready_at_location / plate_washed_at_location test.
#Totals are SteakHouseGridworld constructor args -- cook_time=20, chop_time=3,
#wash_time=3 by default but settable per layout -- which is why we divide by the
#mdp attribute instead of hardcoding a number.
#obj.state shape differs by object and is fixed at construction: a steak is
#built as ('steak', 1, 0) so the tick lives at [2]; garnish and washed_plate are
#built as a bare int 0.
#NOTE the two clocks are not the same kind. Steak cooking is automatic --
#step_environment_effects ticks it every timestep, and only while the steak sits
#on a 'P' tile -- while chopping and washing advance only when a player
#interacts. So for board/sink this channel means "interactions done", not
#"time elapsed".
#An empty cell -> obj is None -> name is "" -> falls through to 0.0.
def _timer_if_occupied(mdp, state, loc):
    obj = state.objects.get(loc)
    name = getattr(obj, "name", "")

    if name == "steak":
        return float(np.clip(obj.state[2] / mdp.steak_cooking_time, 0.0, 1.0))
    if name == "garnish":
        return float(np.clip(obj.state / mdp.chopping_time, 0.0, 1.0))
    if name == "washed_plate":
        return float(np.clip(obj.state / mdp.wash_time, 0.0, 1.0))

    return 0.0



#--------------------HELD OBJECT----------------------------
#"steak_dish" is plate + steak. The bare name "steak" belongs to the object that
#lives ON the grill and is never carried, so it is not in this vocabulary.
HELD_VOCAB = ["none", "meat", "onion", "plate", "washed_plate", "steak_dish", "garnish", "dish"]

#--------------------HELPER FUNCTIONS----------------------------
#layout shape
def layout_shape(mdp):
    return tuple(mdp.shape)
    #I guess terrain_mtx is y, x
    # mtx = mdp.terrain_mtx 
    # return len(mtx[0]), len(mtx)

#make an all 0 layer
def make_layer(position, value, shape):
    layer = np.zeros(shape, dtype=np.float32)
    layer[position] = value
    return layer


#--------------------Build the layered layouts----------------------------
# Building depending on the human index
# in order workds, you switch channels depening on who it is
# this is because self play have both agents yse the same encoder
# so you need to differentiate between the two otherwise the 2 agents would
# always do the EXACT same thing
def build_full_state(mdp, state, agent_index=0, t=0, horizon=260):

    #the .layout files declare start_order_list as a STRING:
    #    "start_order_list": 'steak, steak, steak',
    #so from_layout_name(name) with no override hands back that raw string, and
    #len() then counts CHARACTERS (19, not 3). channel [22] would compute
    #19/19 = 1.0 and stay pinned there for the whole episode.
    #it also breaks the mdp itself: deliver_dish does order_list[0], which on a
    #string is the character 's', so 'steak' == 's' is False -- orders are never
    #consumed and the delivery reward never fires. Flat learning curve, no error.
    #every caller in this repo already passes start_order_list=['steak'] * N;
    #this assert is here so a caller that forgets finds out immediately.
    assert not isinstance(mdp.start_order_list, str), (
        "build the mdp with start_order_list=['steak'] * N -- the .layout file "
        "gives a raw string, which silently breaks orders_left and deliveries"
    )

    x = []
    w, h = layout_shape(mdp)
    shape = (w, h)

    #agents; note that this already switch channel depending on index
    p = state.players[agent_index]
    op = state.players[1-agent_index]

    #add the agent position and the cell the agent are looking at (orientation)
    #also a layer of held object of the agent (but the held object can be like 5 different things....)
    for i in [p, op]:
        px = i.position[0]
        py = i.position[1]
        orn_x = px + i.orientation[0]
        orn_y = py + i.orientation[1]
        x.append(make_layer((px,py), 1, shape))
        x.append(make_layer((orn_x,orn_y), 1, shape))
        
    #add the held object list loop
    #make a big plane for it first cuz we may have to set it twice
    for i in HELD_VOCAB:
        if i != "none":
            plane = np.zeros(shape, dtype=np.float32)
            for loc, obj in state.objects.items():
                if obj.name == i:
                    plane[loc] = 1.0
            if p.held_object is not None and p.held_object.name==i:
                plane[p.position] = 1.0
            if op.held_object is not None and op.held_object.name==i:
                plane[op.position] = 1.0
            x.append(plane)
            
    
    #make terrain those are set things not moving at all
    #WALLS ('#') ride in the 'X' plane rather than getting a channel of their own.
    #All-zeros across the 8 terrain channels means FLOOR in this encoding, so a
    #wall with no channel would read as walkable -- the same trap the padding in
    #env_wrapper avoids by writing padded cells as 'X'. Both are solid and
    #impassable, so 'X' is the honest bucket. The cost is that the policy cannot
    #tell "counter I can place on" from "wall I cannot"; fixing that means a 9th
    #channel (N_LAYERS 23 -> 24), which changes the obs shape and invalidates
    #existing checkpoints, so it waits for the next retrain.
    for c in TERRAIN:
        plane = np.zeros(shape, dtype=np.float32)
        for loc in mdp.terrain_pos_dict.get(c, []):
            plane[loc] = 1.0
        if c == 'X':
            for loc in mdp.terrain_pos_dict.get('#', []):
                plane[loc] = 1.0
        x.append(plane)
        
    #dynamic status of the stations: ready/in progress/empty and the timer of the station
    status = np.zeros(shape, dtype=np.float32)
    timer = np.zeros(shape, dtype=np.float32)
    locs = station_locs(mdp)
    for kind in STATIONS:
        for loc in locs[kind]:
            status[loc] = _STATE_VAL.get(_station_state(mdp, state, loc), 0.0)
            timer[loc] = _timer_if_occupied(mdp, state, loc)
    x.append(status)
    x.append(timer)
    
    #current time left, and number of orders left
    time_left = np.clip((horizon - t) / horizon, 0.0, 1.0) if horizon else 0.0
    n0 = len(mdp.start_order_list) if mdp.start_order_list else 0
    orders_left = state.num_orders_remaining / n0 if n0 else 0.0
    x.append(np.full(shape, time_left, dtype=np.float32))
    x.append(np.full(shape, orders_left, dtype=np.float32))
    
    #assert <condition>, <message shown if it fails>
    #check the COUNT before stacking: np.stack happily stacks 22 or 24 planes,
    #so a dropped/duplicated append only shows up here
    assert len(x) == N_LAYERS, (len(x), N_LAYERS)
    #axis=0 is the channel axis -> (N_LAYERS, w, h)
    obs = np.stack(x, axis=0)
    assert obs.shape == (N_LAYERS, w, h)
    return obs
