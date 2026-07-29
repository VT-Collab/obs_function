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
VAL_FILE = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/full_179_2_2-5.pkl"
GT_OUTPUT_DIR = "gt_renders"
CUSTOM_GRAPHICS_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/graphics"

# --- EXACT GRID ---
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
ACTUAL_GRID = [line.strip() for line in RAW_GRID.strip().split('\n')]

# --- MONKEY PATCH ---
StateVisualizer.TERRAINS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "terrain.json"))
StateVisualizer.OBJECTS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "objects.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "objects.json"))
StateVisualizer.SOUPS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "soups.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "soups.json"))
StateVisualizer.CHEFS_IMG = MultiFramePygameImage(os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.png"), os.path.join(CUSTOM_GRAPHICS_DIR, "chefs.json"))

StateVisualizer.TILE_TO_FRAME_NAME.update({
    'W': 'sink', 'M': 'steaks', 'B': 'board_knife', 'G': 'grill', 
    'C': 'chickens', 'D': 'dishes', 'S': 'serve', 'O': 'onions', 
    'X': 'counter', ' ': 'floor', '1': 'floor', '2': 'floor'
})

def tensor_to_state(obs_tensor, threshold=0.3): # Lowered threshold for safety
    width, height, channels = obs_tensor.shape
    players = []
    objects = {}

    # 1. EXTRACT PLAYERS (Layers 0 & 1)
    for p_idx, loc_layer in enumerate([0, 1]): 
        flat_idx = np.argmax(obs_tensor[:, :, loc_layer])
        x, y = np.unravel_index(flat_idx, (width, height))
        
        # If the data is clean, the max should be ~1.0
        if obs_tensor[x, y, loc_layer] > threshold:
            dir_start = 2 if p_idx == 0 else 6
            dir_idx = np.argmax(obs_tensor[x, y, dir_start:dir_start+4])
            orientation = Direction.ALL_DIRECTIONS[dir_idx]
            players.append(PlayerState((x, y), orientation))

    # 2. EXTRACT OBJECTS
    layer_to_obj_name = {20: "onion", 21: "chicken", 22: "meat", 23: "dirty_plate", 31: "clean_plate"}
    for layer, name in layer_to_obj_name.items():
        xs, ys = np.where(obs_tensor[:, :, layer] > threshold)
        for x, y in zip(xs, ys):
            objects[(x, y)] = ObjectState(name, (x, y))

    # 3. EXTRACTION FOR COOKING ITEMS
    # Chicken
    xs, ys = np.where((obs_tensor[:, :, 26] > threshold) | (obs_tensor[:, :, 27] > threshold))
    for x, y in zip(xs, ys):
        is_done = obs_tensor[x, y, 27] > threshold
        obj = ObjectState("chicken_cooked" if is_done else "chicken", (x, y))
        
        # ADD THESE TWO LINES BACK:
        time_left = int(round(obs_tensor[x, y, 26]))
        obj._cooking_tick = 40 - time_left if not is_done else 40
        
        obj.is_ready = is_done
        objects[(x, y)] = obj

    # Steak
    xs, ys = np.where((obs_tensor[:, :, 28] > threshold) | (obs_tensor[:, :, 29] > threshold))
    for x, y in zip(xs, ys):
        is_done = obs_tensor[x, y, 29] > threshold
        obj = ObjectState("meat_cooked" if is_done else "meat", (x, y))
        
        # ADD THESE TWO LINES BACK:
        time_left = int(round(obs_tensor[x, y, 28]))
        obj._cooking_tick = 30 - time_left if not is_done else 30
        
        obj.is_ready = is_done
        objects[(x, y)] = obj

    return OvercookedState(players=players, objects=objects, timestep=0)

def main():
    os.makedirs(GT_OUTPUT_DIR, exist_ok=True)
    Recipe.configure({})
    
    print(f"Loading data from {VAL_FILE}...")
    val_dataset = SteakhouseVAEDataset(VAL_FILE, human_player_idx=0)
    visualizer = StateVisualizer(tile_size=75, is_rendering_hud=False)

    for i in range(min(50, len(val_dataset))):
        batch = val_dataset[i]
        
        # We try to grab 'full_obs_3d' specifically. 
        # If your Dataset class renames it to 'X', we'll use that.
        gt_tensor = batch.get('full_obs_3d', batch.get('X'))
        
        if gt_tensor is None:
            print(f"Error: Could not find 'full_obs_3d' or 'X' in batch keys: {batch.keys()}")
            return

        gt_tensor = gt_tensor.numpy()
        
        # Ensure shape is (X, Y, Channels)
        # Your data seems to be (39, 15, 10) -> (Channels, H, W)
        if gt_tensor.shape[0] == 39:
            gt_tensor = np.transpose(gt_tensor, (1, 2, 0)) # Now (15, 10, 39)
        
        # Match visualizer expectations
        if gt_tensor.shape[0] == 10 and gt_tensor.shape[1] == 15:
            gt_tensor = np.transpose(gt_tensor, (1, 0, 2))

        fake_state = tensor_to_state(gt_tensor)
        
        if len(fake_state.players) == 0:
            print(f"Frame {i}: No players found in tensor! Max values in P1 layer: {np.max(gt_tensor[:,:,0])}")

        surface = visualizer.render_state(fake_state, grid=ACTUAL_GRID)
        pygame.image.save(surface, os.path.join(GT_OUTPUT_DIR, f"gt_frame_{i:04d}.png"))
        
    print(f"✅ Done! Check {GT_OUTPUT_DIR} for the truth.")

if __name__ == "__main__":
    main()