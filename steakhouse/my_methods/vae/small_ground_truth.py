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
VAL_FILE = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/90_small_small5.pkl"
GT_OUTPUT_DIR = "gt_renders_small5"
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

def tensor_to_state(obs_tensor, threshold=0.3): 
    width, height, channels = obs_tensor.shape
    players = []
    objects = {}

    # 1. EXTRACT PLAYERS (Layers 0 & 1)
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

    # 3. EXTRACTION FOR STEAK (Layer 28=time, 29=done)
    xs, ys = np.where((obs_tensor[:, :, 28] > threshold) | (obs_tensor[:, :, 29] > threshold))
    for x, y in zip(xs, ys):
        is_done = obs_tensor[x, y, 29] > threshold
        obj = ObjectState("meat_cooked" if is_done else "meat", (x, y))
        
        time_left = int(round(obs_tensor[x, y, 28]))
        # Small5 config states steak_time is 10
        obj.cook_time = 10 
        obj._cooking_tick = 10 - time_left if not is_done else 10
        obj.is_ready = is_done
        
        # --- NEW: explicitly attach time_left so we can easily read it later ---
        obj.time_left = time_left 
        
        objects[(x, y)] = obj

    return OvercookedState(players=players, objects=objects, timestep=0)

def main():
    os.makedirs(GT_OUTPUT_DIR, exist_ok=True)
    Recipe.configure({})
    
    # --- NEW: Initialize Pygame font for timer overlay ---
    pygame.font.init()
    timer_font = pygame.font.SysFont("arial", 22, bold=True)
    
    print(f"Loading data from {VAL_FILE}...")
    val_dataset = SteakhouseVAEDataset(VAL_FILE, human_player_idx=0)
    visualizer = StateVisualizer(tile_size=75, is_rendering_hud=False)

    for i in range(min(50, len(val_dataset))):
        batch = val_dataset[i]
        
        # Pull the 21-layer ground truth out of the dataset
        gt_tensor = batch['X'].numpy() # Shape: (21, 5, 5)
        gt_tensor = np.transpose(gt_tensor, (1, 2, 0)) # Shape: (5, 5, 21)

        # RE-INFLATE: Put the 21 layers into a 39-layer format for the decoder
        inflated_tensor = np.zeros((5, 5, 39), dtype=np.float32)
        ACTIVE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 37]
        for new_idx, original_idx in enumerate(ACTIVE_LAYERS):
            inflated_tensor[:, :, original_idx] = gt_tensor[:, :, new_idx]

        # UN-NORMALIZE TIMERS (Steak max=9, Plate max=3, Garnish max=2)
        inflated_tensor[:, :, 28] *= 9.0
        inflated_tensor[:, :, 30] *= 3.0
        inflated_tensor[:, :, 32] *= 2.0

        fake_state = tensor_to_state(inflated_tensor)
        
        # Render the base state
        surface = visualizer.render_state(fake_state, grid=ACTUAL_GRID)
        
        # --- NEW: Overlay the timers directly onto the generated Pygame surface ---
        for obj in fake_state.objects.values():
            if hasattr(obj, 'time_left') and obj.time_left > 0 and not obj.is_ready:
                # Calculate pixel coordinates based on tile_size=75
                px = obj.position[0] * 75
                py = obj.position[1] * 75
                
                # Render the text
                text_surface = timer_font.render(str(obj.time_left), True, (255, 255, 255))
                
                # Center the text slightly near the bottom of the tile
                text_rect = text_surface.get_rect(center=(px + 37, py + 55))
                
                # Draw a tiny dark background rounded rect so the text is legible against meat
                bg_rect = text_rect.inflate(8, 4)
                pygame.draw.rect(surface, (40, 40, 40), bg_rect, border_radius=4)
                
                # Blit (draw) the text on top
                surface.blit(text_surface, text_rect)
        
        pygame.image.save(surface, os.path.join(GT_OUTPUT_DIR, f"gt_frame_{i:04d}.png"))
        
    print(f"✅ Done! Check {GT_OUTPUT_DIR} for the truth.")

if __name__ == "__main__":
    main()