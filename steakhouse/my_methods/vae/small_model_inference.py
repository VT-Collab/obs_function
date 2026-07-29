"""
small_model_inference.py
"""
import os
import torch
import numpy as np
import pygame

# Import your CUSTOM VAE stuff
from small_vae_dataset import SteakhouseVAEDataset
from small_vae_class import HumanBeliefCVAE

# Import standard Overcooked state objects
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, ObjectState, Direction, Recipe
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.visualization.pygame_utils import MultiFramePygameImage

# --- CONFIG ---
MODEL_WEIGHTS = "best_small_human_belief_cvae.pth"
VAL_FILE = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/90_small_small5.pkl"
OUTPUT_DIR = "vae_renders_small5"
CUSTOM_GRAPHICS_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/graphics"

# --- EXACT GRID ---
RAW_GRID = """
XMXBX
X   O
G   X
W21 D
XXSXX
"""
ACTUAL_GRID = [line.strip() for line in RAW_GRID.strip().split('\n') if line.strip()]

# --- MONKEY PATCH THE VISUALIZER ---
StateVisualizer.TERRAINS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.json"))
StateVisualizer.OBJECTS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "objects.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "objects.json"))
StateVisualizer.SOUPS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "soups.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "soups.json"))
StateVisualizer.CHEFS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.json"))

StateVisualizer.TILE_TO_FRAME_NAME.update({
    'W': 'sink', 'M': 'steaks', 'B': 'board_knife', 'G': 'grill', 
    'C': 'chickens', 'D': 'dishes', 'S': 'serve', 'O': 'onions', 
    'X': 'counter', ' ': 'floor', '1': 'floor', '2': 'floor'
})

def tensor_to_state(obs_tensor, threshold=0.5):
    width, height, channels = obs_tensor.shape
    players = []
    objects = {}

    # 1. EXTRACT PLAYERS
    for p_idx, loc_layer in enumerate([0, 1]): 
        flat_idx = np.argmax(obs_tensor[:, :, loc_layer])
        x, y = np.unravel_index(flat_idx, (width, height))
        
        if obs_tensor[x, y, loc_layer] > threshold:
            dir_start = 2 if p_idx == 0 else 6
            dir_idx = np.argmax(obs_tensor[x, y, dir_start:dir_start+4])
            orientation = Direction.ALL_DIRECTIONS[dir_idx]
            players.append(PlayerState((x, y), orientation))

    # 2. EXTRACT STATIC OBJECTS
    layer_to_obj_name = {20: "onion", 21: "chicken", 22: "meat", 23: "dirty_plate", 31: "clean_plate"}
    for layer, name in layer_to_obj_name.items():
        xs, ys = np.where(obs_tensor[:, :, layer] > threshold)
        for x, y in zip(xs, ys):
            objects[(x, y)] = ObjectState(name, (x, y))

    # 3. EXTRACT STEAK GRILLS
    xs, ys = np.where((obs_tensor[:, :, 28] > threshold) | (obs_tensor[:, :, 29] > threshold))
    for x, y in zip(xs, ys):
        is_done = obs_tensor[x, y, 29] > threshold
        obj = ObjectState("meat_cooked" if is_done else "meat", (x, y))
        
        time_left = int(round(obs_tensor[x, y, 28]))
        obj.cook_time = 10 
        obj._cooking_tick = 10 - time_left if not is_done else 10
        obj.is_ready = is_done
        objects[(x, y)] = obj

    fake_state = OvercookedState(players=players, objects=objects)
    fake_state.timestep = 0 
    return fake_state    

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Recipe.configure({})
    
    print("Loading validation data...")
    val_dataset = SteakhouseVAEDataset(VAL_FILE, human_player_idx=0)
    
    sample_obs = val_dataset[0]['X']
    model = HumanBeliefCVAE(sample_obs=sample_obs, latent_dim=128, action_emb_dim=16).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    
    visualizer = StateVisualizer(tile_size=75, is_rendering_hud=False)

    print("Generating renders from VAE output...")
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            batch = val_dataset[i]
            
            r_s_t   = batch['r_s_t'].unsqueeze(0).to(device)
            h_s_t_1 = batch['h_s_t_1'].unsqueeze(0).to(device)
            h_a_t   = batch['h_a_t'].unsqueeze(0).to(device)
            
            # Predict from N(0,1)
            z = torch.zeros(1, 128).to(device) 
            recon_batch = model.decode(z, r_s_t, h_s_t_1, h_a_t)
            
            # Grab raw tensor and convert to (5, 5, 21)
            predicted_tensor = recon_batch[0].cpu().numpy()
            predicted_tensor = np.transpose(predicted_tensor, (1, 2, 0))
            
            # RE-INFLATE to 39 layers
            inflated_tensor = np.zeros((5, 5, 39), dtype=np.float32)
            ACTIVE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 37]
            for new_idx, original_idx in enumerate(ACTIVE_LAYERS):
                inflated_tensor[:, :, original_idx] = predicted_tensor[:, :, new_idx]

            # UN-NORMALIZE TIMERS
            inflated_tensor[:, :, 28] *= 9.0
            inflated_tensor[:, :, 30] *= 3.0
            inflated_tensor[:, :, 32] *= 2.0

            # Decode to State
            fake_state = tensor_to_state(inflated_tensor, threshold=0.5) 
            surface = visualizer.render_state(fake_state, grid=ACTUAL_GRID)
            
            filepath = os.path.join(OUTPUT_DIR, f"small_vae_belief_frame_{i:04d}.png")
            pygame.image.save(surface, filepath)
            
        print(f"✅ Saved VAE predictions to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()