"""
From best_human_belief_cvae.pth, rebuild:
1. images on the validation set
2. Pinpoint important weights on tiles and highlight them 
3. Recalculate loss based on weighted tiles
"""
import os
import torch
import numpy as np
import pygame

# Import your custom VAE stuff
from vae_dataset import SteakhouseVAEDataset
from vae_class import HumanBeliefCVAE

# Import standard Overcooked state objects
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, ObjectState, Direction, Recipe
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.visualization.pygame_utils import MultiFramePygameImage

# --- CONFIG ---
MODEL_WEIGHTS = "best_human_belief_cvae.pth"
VAL_FILE = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/full_179_2_2-5.pkl"
OUTPUT_DIR = "new_vae_renders"
CUSTOM_GRAPHICS_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/graphics"

# --- EXACT GRID FROM YOUR LAYOUT ---
RAW_GRID = """
XXXXXXXXXXSXXXX
XX     X     XX
XW 1   X     XX
XX     X  DXDXX
XXOMM        XX
XXXDX  X     XX
XX     X     XX
XX     X   2 XX
XX     X     XX
XXXBXBXXXGXGXXX
"""
# Clean up the string into a list of strings
ACTUAL_GRID = [line.strip() for line in RAW_GRID.strip().split('\n')]

# --- MONKEY PATCH THE VISUALIZER ---
# 1. Force Python to reload the specific Steakhouse graphics into memory
StateVisualizer.TERRAINS_IMG = MultiFramePygameImage(
    os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.png"),
    os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.json")
)
StateVisualizer.OBJECTS_IMG = MultiFramePygameImage(
    os.path.join(CUSTOM_GRAPHICS_DIR, "objects.png"),
    os.path.join(CUSTOM_GRAPHICS_DIR, "objects.json")
)
StateVisualizer.SOUPS_IMG = MultiFramePygameImage(
    os.path.join(CUSTOM_GRAPHICS_DIR, "soups.png"),
    os.path.join(CUSTOM_GRAPHICS_DIR, "soups.json")
)
StateVisualizer.CHEFS_IMG = MultiFramePygameImage(
    os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.png"),
    os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.json")
)

# We inject your custom layout letters directly into the visualizer's dictionary 
# so it doesn't crash when it sees a 'W' (Sink) or 'G' (Grill).
# *NOTE*: Your terrain.json/terrain.png must have these frame names!
StateVisualizer.TILE_TO_FRAME_NAME.update({
    'W': 'sink',          # JSON: "sink.png" -> stripped to 'sink'
    'M': 'steaks',        # JSON: "steaks.png" -> stripped to 'steaks'
    'B': 'board_knife',   # JSON: "board_knife.png" -> stripped to 'board_knife'
    'G': 'grill',         # JSON: "grill.png" -> stripped to 'grill'
    'C': 'chickens',      # JSON: "chickens.png" -> stripped to 'chickens'
    'D': 'dishes',        
    'S': 'serve',         
    'O': 'onions',        
    'X': 'counter',       
    ' ': 'floor',        
    '1': 'floor',         # <-- ADD THIS
    '2': 'floor'          # <-- ADD THIS  
})

def tensor_to_state(obs_tensor, threshold=0.5):
    """
    Decodes the VAE's predicted tensor back into a Python OvercookedState.
    """
    width, height, channels = obs_tensor.shape
    
    players = []
    objects = {}

    # 1. EXTRACT PLAYERS (Layers 0 & 1 for location)
    for p_idx, loc_layer in enumerate([0, 1]): 
        flat_idx = np.argmax(obs_tensor[:, :, loc_layer])
        x, y = np.unravel_index(flat_idx, (width, height))
        
        if obs_tensor[x, y, loc_layer] > threshold:
            # Orientation (Layers 2-5 for P1, 6-9 for P2)
            dir_start = 2 if p_idx == 0 else 6
            dir_idx = np.argmax(obs_tensor[x, y, dir_start:dir_start+4])
            orientation = Direction.ALL_DIRECTIONS[dir_idx]
            players.append(PlayerState((x, y), orientation))

    # 2. EXTRACT OBJECTS
    layer_to_obj_name = {
        20: "onion", 
        21: "chicken", 
        22: "meat", 
        23: "dirty_plate", 
        31: "clean_plate"
    }

    for layer, name in layer_to_obj_name.items():
        xs, ys = np.where(obs_tensor[:, :, layer] > threshold)
        for x, y in zip(xs, ys):
            pos = (x, y)
            objects[pos] = ObjectState(name, pos)

    # 3. EXTRACT CHICKEN POTS (Layer 26=time, 27=done, cook_time=40)
    xs, ys = np.where((obs_tensor[:, :, 26] > threshold) | (obs_tensor[:, :, 27] > threshold))
    for x, y in zip(xs, ys):
        pos = (x, y)
        is_done = obs_tensor[x, y, 27] > threshold
        
        # 👇 Use the exact JSON keys for chicken
        obj_name = "chicken_cooked" if is_done else "chicken"
        obj = ObjectState(obj_name, pos)
        
        time_left = int(round(obs_tensor[x, y, 26]))
        obj._cooking_tick = 40 - time_left if not is_done else 40
        obj.cook_time = 40
        obj.is_ready = is_done
        objects[pos] = obj

    # 4. EXTRACT STEAK GRILLS (Layer 28=time, 29=done, cook_time=30)
    xs, ys = np.where((obs_tensor[:, :, 28] > threshold) | (obs_tensor[:, :, 29] > threshold))
    for x, y in zip(xs, ys):
        pos = (x, y)
        is_done = obs_tensor[x, y, 29] > threshold
        
        # 👇 Use the exact JSON keys for steak/meat
        obj_name = "meat_cooked" if is_done else "meat"
        obj = ObjectState(obj_name, pos)
        
        time_left = int(round(obs_tensor[x, y, 28]))
        obj._cooking_tick = 30 - time_left if not is_done else 30
        obj.cook_time = 30
        obj.is_ready = is_done
        objects[pos] = obj

    fake_state = OvercookedState(players=players, objects=objects)
    fake_state.timestep = 0 
    return fake_state    

#vae results
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # This tricks the backend into thinking the game recipes are set up.
    Recipe.configure({})
    
    print("Loading validation data to generate images...")
    val_dataset = SteakhouseVAEDataset(VAL_FILE, human_player_idx=0)
    
    # Init Model
    sample_obs = val_dataset[0]['X']
    model = HumanBeliefCVAE(sample_obs=sample_obs, latent_dim=128, action_emb_dim=16).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    
    # Init Visualizer
    visualizer = StateVisualizer(tile_size=75, is_rendering_hud=False)

    print("Generating renders from VAE output...")
    
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            batch = val_dataset[i]
            
            # Target (Only used to see if we were right)
            X       = batch['X'].unsqueeze(0).to(device) 
            
            # Context (The only things the model should know)
            r_s_t   = batch['r_s_t'].unsqueeze(0).to(device)
            h_s_t_1 = batch['h_s_t_1'].unsqueeze(0).to(device)
            h_a_t   = batch['h_a_t'].unsqueeze(0).to(device)
            
            # 1. INSTEAD OF ENCODING X: Generate a z from the prior (Mean=0)
            # In a VAE, during inference, z should follow N(0,1)
            z = torch.zeros(1, 128).to(device) # Use the mean of the distribution
            
            # 2. DECODE ONLY: This is the "Forward Dynamics" prediction
            recon_batch = model.decode(z, r_s_t, h_s_t_1, h_a_t)
            
            # 1. Grab the raw tensor from GPU to CPU
            predicted_tensor = recon_batch[0].cpu().numpy()
            
            # 🚨 THE FIX: Transpose from (Channels, X, Y) to (X, Y, Channels)
            # This turns your (39, 15, 10) tensor into a (15, 10, 39) tensor!
            predicted_tensor = np.transpose(predicted_tensor, (1, 2, 0))
            
            # 🚨 SAFETY NET: If your dataset saved them as (Channels, Y, X) instead,
            # this will flip X and Y back so the visualizer doesn't draw sideways.
            if predicted_tensor.shape[0] == 10 and predicted_tensor.shape[1] == 15:
                predicted_tensor = np.transpose(predicted_tensor, (1, 0, 2))

            # DEBUG: Check VAE confidence for players
            p1_max = np.max(predicted_tensor[:, :, 0])
            p2_max = np.max(predicted_tensor[:, :, 1])
            print(f"Frame {i} Confidence - P1: {p1_max:.2f}, P2: {p2_max:.2f}")

            # Now decode it!
            fake_state = tensor_to_state(predicted_tensor, threshold=0.2) # Try lowering to 0.2
            
            surface = visualizer.render_state(fake_state, grid=ACTUAL_GRID)
            
            filepath = os.path.join(OUTPUT_DIR, f"vae_belief_frame_{i:04d}.png")
            pygame.image.save(surface, filepath)
            
        print(f"✅ Saved renders to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
    
    
    
    
    # def get_obs(self, state, horizon: int = 400):
    #     """
    #     """
    #     self.update(state)
    #     # self.update_kb_log()
        
    #     player = state.players[self.agent_index]
    #     other_players = self.knowledge_base['other_player']
    #     am = self.mlam

    #     counter_objects = {}
    #     all_objects = []
    #     # counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
    #     sink_states = self.knowledge_base['sink_states']
    #     chopping_board_states = self.knowledge_base['chop_states']
    #     pot_states_dict = self.knowledge_base['pot_states']
    #     grill_states_dict = self.knowledge_base['grill_states']
    #     for k, o in self.knowledge_base.items():
    #         if k not in ['pot_states', 'sink_states', 'chop_states', 'grill_states', 'other_player']:
    #             if o.position in self.mlam.mdp.get_counter_locations():
    #                 counter_objects[o.name] = [o.position]
    #             all_objects.append(o)

    #     base_map_features = [
    #         "counter_loc",
    #         "pot_loc",
    #         "dirty_plate_disp_loc",
    #         "onion_disp_loc",
    #         "serve_loc",
    #         "grill_loc",
    #         "chicken_disp_loc",
    #         "sink_loc",
    #         "meat_disp_loc",
    #         "chopping_board_loc",
    #     ]
    #     variable_map_features = [
    #         "onions",
    #         "chickens",
    #         "meats",
    #         "dirty_plates",
    #         "steak_onions",
    #         "boiled_chicken_onions",
    #         "chicken_cook_time_remaining",
    #         "chicken_done",
    #         "steak_cook_time_remaining",
    #         "steak_done",
    #         "plate_clean_time_remaining",
    #         "plate_cleaned",
    #         "garnish_chop_time_remaining",
    #         "garnish_chopped",
    #     ]
    #     urgency_features = ["urgency"]

    #     # all_objects = steakhouse_state.all_objects_list

    #     def make_layer(position, value):
    #         layer = np.zeros(self.mlam.mdp.shape)
    #         layer[position] = value
    #         return layer

    #     # Ensure that primary_agent_idx layers are ordered before other_agent_idx
    #     # layers
    #     ordered_player_features = [
    #         f"human_loc",
    #         f"other_player_loc",
    #     ] + [
    #         f"human_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
    #         for d in Direction.ALL_DIRECTIONS
    #     ] + [
    #         f"other_player_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
    #         for d in Direction.ALL_DIRECTIONS
    #     ]

    #     DISH_TYPES = [
    #         "steak_dish",
    #         "boiled_chicken_dish",
    #         "steak_onion_dish",
    #         "boiled_chicken_onion_dish",
    #     ]

    #     LAYERS = (
    #         ordered_player_features
    #         + base_map_features
    #         + variable_map_features
    #         + urgency_features
    #         + DISH_TYPES
    #     )
    #     state_mask_dict = {k: np.zeros(self.mlam.mdp.shape) for k in LAYERS}

    #     # MAP LAYERS
    #     if horizon - state.timestep < 40:
    #         state_mask_dict["urgency"] = np.ones(self.mlam.mdp.shape)

    #     for loc in self.mlam.mdp.get_counter_locations():
    #         state_mask_dict["counter_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_pot_locations():
    #         state_mask_dict["pot_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_dirty_plate_locations():
    #         state_mask_dict["dirty_plate_disp_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_onion_dispenser_locations():
    #         state_mask_dict["onion_disp_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_serving_locations():
    #         state_mask_dict["serve_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_grill_locations():
    #         state_mask_dict["grill_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_chicken_dispenser_locations():
    #         state_mask_dict["chicken_disp_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_sink_locations():
    #         state_mask_dict["sink_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_meat_dispenser_locations():
    #         state_mask_dict["meat_disp_loc"][loc] = 1

    #     for loc in self.mlam.mdp.get_chopping_board_locations():
    #         state_mask_dict["chopping_board_loc"][loc] = 1

    #     # Current order layers
    #     if state.order_list:
    #         # Order list is not None and there is at least one order remaining
    #         cur_order = state.order_list[0]
    #         state_mask_dict[cur_order] = np.ones(self.mlam.mdp.shape)

    #     # PLAYER LAYERS
    #     human_orientation_idx = Direction.DIRECTION_TO_INDEX[
    #         player.orientation
    #     ]
    #     state_mask_dict[f"human_loc"] = make_layer(player.position, 1)
    #     state_mask_dict[f"human_orientation_{human_orientation_idx}"] = (
    #         make_layer(player.position, 1)
    #     )

    #     other_player_orientation_idx = Direction.DIRECTION_TO_INDEX[
    #         other_players.orientation
    #     ]
    #     state_mask_dict[f"other_player_loc"] = make_layer(other_players.position, 1)
    #     state_mask_dict[f"other_player_orientation_{other_player_orientation_idx}"] = (
    #         make_layer(other_players.position, 1)
    #     )


    #     # OBJECT & STATE LAYERS
    #     for obj in all_objects:
    #         if obj.name == "boiled_chicken":
    #             # Boiled chicken is similar to soup except that it immediately
    #             # starts cooking and only needs 1 chicken.
    #             if obj.position in self.mlam.mdp.get_pot_locations():
    #                 # Only one chicken can be in pot and it is never idle. When
    #                 # player interacts with pot holding chicken, a `ChickenState` is
    #                 # created, chicken is added, and cooking starts.
    #                 state_mask_dict["chicken_cook_time_remaining"] += make_layer(
    #                     obj.position, obj.cook_time - obj._cooking_tick
    #                 )
    #                 if obj.is_ready:
    #                     state_mask_dict["chicken_done"] += make_layer(
    #                         obj.position, 1
    #                     )
    #             else:
    #                 # If boiled chicken is not in a pot, treat it like a soup that
    #                 # is cooked with remaining time 0
    #                 state_mask_dict["chicken_done"] += make_layer(obj.position, 1)

    #         elif obj.name == "steak":
    #             # Steak is similar to boiled chicken.
    #             if obj.position in self.mlam.mdp.get_grill_locations():
    #                 state_mask_dict["steak_cook_time_remaining"] += make_layer(
    #                     obj.position, obj.cook_time - obj._cooking_tick
    #                 )
    #                 if obj.is_ready:
    #                     state_mask_dict["steak_done"] += make_layer(obj.position, 1)
    #             else:
    #                 state_mask_dict["steak_done"] += make_layer(obj.position, 1)

    #         elif obj.name == "clean_plate":
    #             # Cleaning plate is similar to steak and boiled chicken for
    #             # observation (but interact action is required to move the cleaning
    #             # forward unlike cooking which happens automatically).
    #             if obj.position in self.mlam.mdp.get_sink_locations():
    #                 state_mask_dict["plate_clean_time_remaining"] += make_layer(
    #                     obj.position, obj.cook_time - obj._cooking_tick
    #                 )
    #                 if obj.is_ready:
    #                     state_mask_dict["plate_cleaned"] += make_layer(
    #                         obj.position, 1
    #                     )
    #             else:
    #                 state_mask_dict["plate_cleaned"] += make_layer(obj.position, 1)

    #         elif obj.name == "garnish":
    #             # Cutting for garnish is similar to cleaning plate.
    #             if obj.position in self.mlam.mdp.get_chopping_board_locations():
    #                 state_mask_dict["garnish_chop_time_remaining"] += make_layer(
    #                     obj.position, obj.cook_time - obj._cooking_tick
    #                 )
    #                 if obj.is_ready:
    #                     state_mask_dict["garnish_chopped"] += make_layer(
    #                         obj.position, 1
    #                     )
    #             else:
    #                 state_mask_dict["garnish_chopped"] += make_layer(
    #                     obj.position, 1
    #                 )

    #         elif obj.name == "onion":
    #             state_mask_dict["onions"] += make_layer(obj.position, 1)
    #         elif obj.name == "chicken":
    #             state_mask_dict["chickens"] += make_layer(obj.position, 1)
    #         elif obj.name == "meat":
    #             state_mask_dict["meats"] += make_layer(obj.position, 1)
    #         elif obj.name == "dirty_plate":
    #             state_mask_dict["dirty_plates"] += make_layer(obj.position, 1)
    #         elif obj.name == "steak_onion":
    #             # Garnished steak doesn't need cooking, so treated as a regular
    #             # object.
    #             state_mask_dict["steak_onions"] += make_layer(obj.position, 1)
    #         elif obj.name == "boiled_chicken_onion":
    #             # Garnished chicken doesn't need cooking, so treated as a regular
    #             # object.
    #             state_mask_dict["boiled_chicken_onions"] += make_layer(
    #                 obj.position, 1
    #             )
    #         else:
    #             raise ValueError("Unrecognized object")

    #     # print("terrain----")
    #     # print(np.array(self.mlam.mdp.terrain_mtx))
    #     # print("-----------")
    #     # print(len(LAYERS))
    #     # print(len(state_mask_dict))
    #     # for k, v in state_mask_dict.items():
    #     #     print(k)
    #     #     print(np.transpose(v, (1, 0)))

    #     # Stack of all the state masks, order decided by order of LAYERS
    #     state_mask_stack = np.array(
    #         [state_mask_dict[layer_id] for layer_id in LAYERS]
    #     )
    #     state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
    #     assert state_mask_stack.shape[:2] == self.mlam.mdp.shape
    #     assert state_mask_stack.shape[2] == len(LAYERS)
    #     # NOTE: currently not including time left or order_list in featurization

    #     return np.array(state_mask_stack).astype(int)