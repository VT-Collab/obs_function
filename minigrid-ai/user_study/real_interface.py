"""
Playable keyboard interface for MiniGrid-LockedRoom

Added features:
- Make sure the description is not cut off 
- play twice to familiarize with the env before collecting data
- More action space for the robot
- Different modes
- More walls in the environment
- longer hints
- allow forgetfulness 

real_interface.py — Playable keyboard interface for MiniGrid-LockedRoom.

Same as interface.py, plus two additions still being validated before they
get folded back in:

  1. A live [debug] readout of DynamicAssist's inferred FOV + posterior
     entropy in the header, so convergence can be eyeballed during play
     instead of trusted blindly.
  2. Landmark-based hint text instead of raw (x, y) coordinates — real
     humans don't navigate by grid coordinates (Denis, 1997, "The
     Description of Routes"). Hints are phrased as a compass direction +
     rough distance, grounded in what the player is already holding/has
     already opened (common ground, Clark & Wilkes-Gibbs, 1986) rather than
     an absolute location — the minimal distinguishing description, per
     Dale & Reiter (1995).

A real human plays the game directly with the keyboard (no simulated agent).
On launch, an instructions screen is shown FIRST — the game description,
controls, and FOV/assistance choices are all on that opening screen, and
play only starts once the participant presses ENTER.

FOV
---
The participant sees only what falls inside a forward-facing vision cone of
60, 120, or 180 degrees; every square outside that cone is rendered solid
black. This mirrors the simulated BayesHumanAgent's `estimated_world_cone_vis_mask`
so a human participant is exposed to the same partial observability the
research pipeline models. The blackout is drawn entirely in this file (a
post-process on top of the normal render) — it does not touch minigrid_env.py,
so every other script's rendering is unaffected.

Assistance mode
----------------
If enabled, the same robot policies used in the simulated evaluation
(no_assist / static / dynamic / gateway) watch the participant's accumulated
knowledge base and, when they'd otherwise be stuck too long, interrupt with
a full-screen hint message that must be dismissed with a key press before
play resumes.

Controls
--------
  arrow keys   move up / down / left / right on screen (auto-turns for you —
               MiniGrid's native action set only has turn-left/turn-right/
               forward, so a directional keypress silently issues the turns
               needed to face that way before stepping)
  SPACE        toggle / open a door
  P            pick up
  O            drop
  BACKSPACE    restart the episode
  ESC          quit
"""

from __future__ import annotations


import sys
from pathlib import Path

#walks 2 directory levels above, aka, the minigrid-ai/ root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import pygame


from minigrid.core.actions import Actions

from human.agents.bayes_agent import BayesHumanAgent
from robot.policy.deterministic.static_assist import StaticAssist
from robot.policy.deterministic.dynamic_assist import DynamicAssist
from robot.policy.deterministic.gateway_assist import GatewayAssist


ENV_ID = "MiniGrid-LockedRoom-v0"
TILE_SIZE = 32
HEADER_H = 120 #room for the header — controls line wraps to 2 rows, plus the debug row
PATIENCE = 4
RANDOM_WALLS = True  # MISHA NEW CHANGE — toggle random 2-cell wall segments in rooms

FOV_CHOICES = [60, 120, 180]

#lambda is python syntax for inline, unmaed functions (without def statement); form: lambda: expression; can be called later
ASSIST_CHOICES = [
    ("N", "No assist", None),
    ("S", "Static", lambda: StaticAssist(assumed_fov=120, patience=PATIENCE)),
    ("D", "Dynamic", lambda: DynamicAssist(patience=PATIENCE)),
    ("G", "Gateway", lambda: GatewayAssist(assumed_fov=120, patience=PATIENCE)),
]

ASSIST_LABELS = {
    "NoneType":       "No assist",
    "NoAssist":       "No assist",
    "StaticAssist":   "Static",
    "DynamicAssist":  "Dynamic",
    "GatewayAssist":  "Gateway",
}

#specify to the corresponding minigrid number for direction and to minigrid action
ARROW_TO_DIR = {
    pygame.K_RIGHT: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_UP: 3,
}

IMMEDIATE_KEY_TO_ACTION = {
    pygame.K_SPACE: Actions.toggle,
    pygame.K_p:     Actions.pickup,
    pygame.K_o:     Actions.drop,
}


def turn_and_move_actions(agent_dir: int, target_dir: int)-> list:
    """
    Action key turn and move at the same time always
    b/c Actions.forward different depending on the agent's current facing direction
    """
    
    diff = (target_dir - agent_dir) % 4
    if diff == 0:
        return [Actions.forward]
    elif diff == 1:
        return [Actions.right, Actions.forward]
    elif diff == 3:
        return [Actions.left, Actions.forward]
    
    return [Actions.right, Actions.right, Actions.forward]  # diff == 2, turn around

DESCRIPTION = """\
You are standing in a hallway lined with locked rooms. 
Somewhere in the side rooms is a key; the matching locked door 
leads to a room with the goal tile. Find the right key, 
unlock the matching door, and reach the
goal before the step limit runs out."""
 
COLORS = {
    "bg":       (18, 18, 20),
    "text":     (230, 230, 230),
    "dim":      (150, 150, 150),
    "accent":   (120, 200, 255),
    "selected": (255, 210, 90),
    "hint_bg":  (10, 10, 10),
    "hint_text": (255, 230, 140),
}

def _wrap_text_to_width(font, text: str, max_width: int) -> list:
    """Greedy word-wrap so text fits within max_width pixels for the given font."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ── FOV-masked rendering (local to this file only) ─────────────────────────

def render_masked_frame(state, tile_size: int=TILE_SIZE) -> np.ndarray:
    """
    black out all cells outher than agent's own FOV
    """
    img = state.get_full_render(False, tile_size) #render as image so pixels
    mask = state._world_cone_vis_mask() #get whether each x,y is 0 or 1
    for x in range(state.width):
        for y in range(state.height):
            if not mask[x,y]:
                #compute pixel bound
                y0, y1 = y * tile_size, (y+1)* tile_size #row
                x0, x1 = x * tile_size, (x+1)* tile_size#column
                img[y0:y1, x0:x1, :] = 0 #row, column, all channels
                
    return img

def blit_frame(screen, state, tile_size: int=TILE_SIZE) -> None:
    """draws masked grid image into pygame"""
    img = render_masked_frame(state, tile_size)
    img = np.ascontiguousarray(np.transpose(img, axes = (1, 0, 2))) #swap channels with .transpose
    #convert into pygame Surface object that can be drawn to the screen
    surf = pygame.surfarray.make_surface(img)
    screen.blit(surf, (0, HEADER_H)) #(blits) that surface onto the main screen, positioned at x=0, y=HEADER_H — i.e., flush left, starting just below the 96px header bar so it doesn't overlap the status text.
    

def draw_header(screen, font, fov: int, assist_label: str, state, robot=None):
    pygame.draw.rect(screen, COLORS["bg"], (0, 0, screen.get_width(), HEADER_H)) #draw background rectangle for header
    
    lines = [
        (f"FOV {fov}°   |   Assist: {assist_label}   |   "
         f"Step {state.step_count}/{state.max_steps}", COLORS["accent"]),
        ("arrows move   SPACE toggle   P pickup   "
         "O drop   BACKSPACE restart   ESC quit", COLORS["dim"]),
    ]
    
    if robot is not None and hasattr(robot, "entropy"):
        est_fov = robot.inf.map_fov()
        match = "✓" if est_fov == fov else "…"
        lines.append((f"[debug] FOV estimate: {est_fov}°  {match}   "
                       f"posterior entropy: {robot.entropy():.2f}", COLORS["selected"]))    
        
    max_w = screen.get_width() - 20
    y = 8
    for text, color in lines:
        for wrapped in _wrap_text_to_width(font, text, max_w):
            screen.blit(font.render(wrapped, True, color), (10, y))
            y += 26
        

# --------------- HINT SYSTEM: Landmark-based hint text (no raw coordinates) ---------------------
_COMPASS = {
    (0, -1): "top", (0, 1): "down", (1, 0): "right", (-1, 0): "left",
    (1, -1): "top-right", (1, 1): "down-right",
    (-1, -1): "top-left", (-1, 1): "down-left",
}

def _compass_direction(agent_pos, target_pos) -> str:
    """
    compare a pair of coordinates and return one relative to the other, eg. northest
    """
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    if dx == 0 and dy == 0:
        return "right where you're standing"
    dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    return _COMPASS[(dx, dy)]


def _distance_phrase(agent_pos, target_pos) -> str:
    """
    converts actual distances to a human readable phrase
    """
    dist = abs(target_pos[0] - agent_pos[0]) + abs(target_pos[1] - agent_pos[1])
    if dist <=4:
        return "very close by"
    if dist <=9: 
        return "a short walk away"
    return "a long way off"






def build_hint_text(reveal: tuple, state) -> str:
    """Turn a reveal thing (kind, color, loc) into an actual. landmark-based description"""
    kind, color, loc = reveal
    direction = _compass_direction(state.agent_pos, loc)
    distance = _distance_phrase(state.agent_pos, loc)

    if kind == "key":
        return f"The {color} key is {distance}, to your {direction}."

    if kind == "door":
        return (f"The {color} door — the one that matches the key you're "
                f"holding — is {direction} of you, {distance}.")

    # NEW HINT: message-only, no KB write — see _next_needed in _base.py.
    # Robot's full-grid map already confirms this room is a dead end.
    if kind == "dead_room":
        return "This room is a dead end — no goal in here. No need to keep exploring it."

    # NEW HINT: message-only — see _next_needed in _base.py. Same idea as
    # dead_room, but for key-hunting: robot's map confirms no key is here.
    if kind == "empty_room":
        return "This room is empty — no key in here. No need to keep exploring it."

    # kind == "goal"
    held_color = getattr(getattr(state, "carrying", None), "color", None)
    if held_color:
        return (f"The goal is {direction} of you, {distance}, through the "
                f"{held_color} door you just unlocked.")
    return f"The goal is {direction} of you, {distance}."

        
    
# ---------------END HINT BUILDING SYSTEM ------------------------------




# ---------------Instructions / setup screen -----------------------------
def show_instructions_screen(screen, font, big_font) -> tuple[int, str]:
    """
    Show the instructions screen and let the user select FOV and assist mode.
    Returns the selected FOV and assist mode.
    
    font.render(text, antialias, color)
        1. text (string) — here, f"[{key_char}] {label}", e.g. "[S] Static". This is the actual text to render into pixels.
        2. antialias (bool, True/False) — whether to smooth the edges of the glyphs. True gives smoother-looking text (blended edges); False gives blocky/jagged edges but is very slightly cheaper to render. Almost always True for UI text.
        3. color — an RGB(A) color for the text itself, here the color variable computed just above (COLORS["selected"] or COLORS["dim"], depending on whether this is the currently chosen option)
    
    screen.blit(source, dest) — this is where positioning actually happens. 
        blit = "block transfer," i.e. copy pixels from one surface onto another.
        1. source — the surface to copy from. Here, the little text-image we just got from font.render(...).
        2. dest — where to place it on screen, as an (x, y) pixel coordinate — specifically, the top-left corner of where source gets pasted onto screen. Here, (50, y): fixed x=50 (indent), and y is the running cursor variable that's been incrementing throughout the loop.
    """
    
    fov_idx = 1 #default 120 degree
    assist_idx = 0 #default no assist
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    fov_idx = event.key - pygame.K_1
                for i, (key_char, _, _) in enumerate(ASSIST_CHOICES):
                    if event.unicode.lower() == key_char.lower():
                        assist_idx = i
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return FOV_CHOICES[fov_idx], ASSIST_CHOICES[assist_idx][2]
                
        screen.fill(COLORS["bg"]) #fill entire screen the bg color
        
        #build title
        y = 30
        big = big_font.render("MiniGrid Locked Room — User Study", True, COLORS["text"])
        screen.blit(big, (30, y)) #font, x, y
        y += 50
        
        #build description
        for line in _wrap_text_to_width(font, DESCRIPTION, screen.get_width() - 60):
            screen.blit(font.render(line, True, COLORS["text"]), (30, y))
            y += 24
        y += 20
        
        screen.blit(font.render("Choose your field of view:", True, COLORS["accent"]), (30, y))
        y += 28
        
        for i, fov in enumerate(FOV_CHOICES):
            color = COLORS["selected"] if i == fov_idx else COLORS["dim"]
            screen.blit(font.render(f"[{i + 1}] {fov}°", True, color), (50, y)) #draw [1] 60 etc.
            y += 24
        y += 20            
        
        screen.blit(font.render("Choose assistance mode:", True, COLORS["accent"]), (30, y))
        y += 28
        for i, (key_char, label, _) in enumerate(ASSIST_CHOICES):
            color = COLORS["selected"] if i == assist_idx else COLORS["dim"]
            screen.blit(font.render(f"[{key_char}] {label}", True, color), (50, y))
            y += 24
        y+= 30
        
        screen.blit(font.render("Press ENTER to start.", True, COLORS["text"]), (30, y))

        pygame.display.flip() #make visible
        clock.tick(30)
        
# ── Full-screen hint overlay ───────────────────────────────────────────────

HINT_MIN_SECONDS = 4.0

def show_hint_overlay(screen, font, big_font, message: str,
                       min_seconds: float = HINT_MIN_SECONDS) -> None:
    """
    Blocks for at least `min_seconds`, no matter what the player does —
    key presses during this window are read (so the window stays responsive
    and the queue doesn't pile up) but never dismiss the hint early, so a
    hurried player can't blink and miss it.
    """

    hint_font = pygame.font.SysFont("monospace", 30, bold=True)
    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    cx, cy = screen.get_width() // 2, screen.get_height() // 2

    body_lines = _wrap_text_to_width(hint_font, message, screen.get_width() - 80)
    body_surfs = [hint_font.render(line, True, COLORS["text"]) for line in body_lines]
    line_h = hint_font.get_linesize()

    while True:
        remaining = min_seconds - (pygame.time.get_ticks() - start_ticks) / 1000.0
        if remaining <= 0:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        overlay = pygame.Surface(screen.get_size())
        overlay.fill(COLORS["hint_bg"])
        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, COLORS["hint_text"], screen.get_rect(), width=8)

        title = big_font.render("ASSISTANCE HINT", True, COLORS["hint_text"])
        count = font.render(f"resuming in {remaining:0.1f}s", True, COLORS["dim"])

        gap = 20
        content_h = title.get_height() + gap + line_h * len(body_surfs) + gap + count.get_height()
        y = cy - content_h // 2

        screen.blit(title, (cx - title.get_width() // 2, y))
        y += title.get_height() + gap

        for surf in body_surfs:
            screen.blit(surf, (cx - surf.get_width() // 2, y))
            y += line_h

        y += gap
        screen.blit(count, (cx - count.get_width() // 2, y))

        pygame.display.flip()
        clock.tick(30)
    
    


def show_end_screen(screen, font, big_font, success: bool, steps: int, n_assists: int) -> None:
    screen.fill(COLORS["bg"])
    title = "Goal reached!" if success else "Out of steps."
    lines = [
        (title, COLORS["accent"]),
        (f"Steps taken: {steps}   |   Hints used: {n_assists}", COLORS["text"]),
        ("Press ENTER to play again, ESC to quit.", COLORS["dim"]),
    ]
    y = screen.get_height() // 2 - 40
    for text, color in lines:
        surf = big_font.render(text, True, color) if text == title else font.render(text, True, color)
        screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, y))
        y += 40
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return

# ------------- Episode loop ---------------------------------------------
def run_episode(screen, font, big_font, fov:int, make_robot) -> tuple[bool, int, int]:
    
    env = gym.make(ENV_ID, agent_fov=fov, random_walls=RANDOM_WALLS, render_mode="rgb_array")  # MISHA NEW CHANGE
    env.reset()
    state = env.unwrapped
    
    human = BayesHumanAgent(fov=fov)
    human.init_knowledge_base(state)
    robot = make_robot() if make_robot else None #select which assist robot; initialize aka DyanmicAssist() or StaticAssist, etc.

    if robot:
        robot.reset(state)
    
    clock = pygame.time.Clock()
    terminated = truncated = False
    
    while not (terminated or truncated):
        human.select_subtask(state) #advances kb with the simulated real human kb
        
        if robot is not None:
            reveal = robot.step(state, human.knowledge_base)
            if reveal is not None:
                message = build_hint_text(reveal, state)
                show_hint_overlay(screen, font, big_font, message)

        def redraw():
            screen.fill(COLORS["bg"])
            blit_frame(screen, state)
            draw_header(screen, font, fov, ASSIST_LABELS.get(type(robot).__name__, "No assist"), state, robot)
            pygame.display.flip()

        redraw()
        
        actions = None
        while actions is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_BACKSPACE:
                        env.close()
                        return run_episode(screen, font, big_font, fov, make_robot)
                    if event.key in ARROW_TO_DIR:
                        actions = turn_and_move_actions(state.agent_dir, ARROW_TO_DIR[event.key])
                    elif event.key in IMMEDIATE_KEY_TO_ACTION:
                        actions = [IMMEDIATE_KEY_TO_ACTION[event.key]]
            clock.tick(30)

        # Turning + forward is issued as one keypress, but replayed as separate
        # env steps with a redraw between them so the rotation is visible
        # instead of the player appearing to teleport.
        for action in actions:
            if robot is not None and hasattr(robot, "observe"):
                robot.observe(state, action)  # feeds DynamicAssist's FOV posterior
            _, _, terminated, truncated, _ = env.step(action)
            redraw()
            if terminated or truncated:
                break
            pygame.time.delay(60)

    n_assists = robot.n_assists if robot is not None else 0
    steps = state.step_count
    env.close()
    return terminated, steps, n_assists

            
    
def main() -> None:
    pygame.init()
    pygame.display.set_caption("MiniGrid Locked Room — User Study")
    
    grid_px = 19 * TILE_SIZE # default size = 19
    screen = pygame.display.set_mode((grid_px, grid_px + HEADER_H))
    font = pygame.font.SysFont("monospace", 16)
    big_font = pygame.font.SysFont("monospace", 26, bold=True)
    
    while True:
        fov, make_robot = show_instructions_screen(screen, font, big_font)
        success, steps, n_assists = run_episode(screen, font, big_font, fov, make_robot)
        show_end_screen(screen, font, big_font, success, steps, n_assists)

    


if __name__ == "__main__":
    main()