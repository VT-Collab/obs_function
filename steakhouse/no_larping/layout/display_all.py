"""
Render every steak layout in no_larping/layouts at once, tiled in one window.

    python play.py                # all layouts, auto grid
    python play.py --cols 6       # force 6 columns

Keys:  q / ESC / close window  ->  quit

How it works: SteakHouseGridworld.render() normally calls pygame.display.set_mode()
itself and caches the window as mdp.viewer. Pre-assigning mdp.viewer to our own
off-screen Surface makes it draw there instead, so each kitchen can be scaled down
and blitted into a tile.
"""
import argparse
import math
import os
import sys

import pygame

# repo root -- STEAK_ROOT on CARC, else the local checkout.
# must run before anything imports overcooked_ai_py.
sys.path.insert(0, os.environ.get("STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))

import overcooked_ai_py

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS_DIR = os.path.join(HERE, "layouts")
overcooked_ai_py.LAYOUTS_DIR = LAYOUTS_DIR  # read_layout_dict reads this at call time

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld

SPRITE = 50  # graphics.SPRITE_LENGTH -- pixels per tile, fixed by the renderer
BG = (28, 28, 32)
LABEL = (235, 235, 235)
LABEL_H = 22


def layout_names():
    """Every .layout in our layouts dir, sorted."""
    return sorted(f[:-len(".layout")] for f in os.listdir(LAYOUTS_DIR)
                  if f.endswith(".layout"))


def render_offscreen(name):
    """Render a layout's start state to its own Surface. None if it won't load."""
    try:
        mdp = SteakHouseGridworld.from_layout_name(name)
        env = OvercookedEnv.from_mdp(mdp, horizon=400)
    except Exception as e:
        print(f"  skipping {name}: {type(e).__name__}: {e}")
        return None

    surface = pygame.Surface((mdp.width * SPRITE, mdp.height * SPRITE))
    surface.fill((255, 255, 255))
    mdp.viewer = surface  # non-None -> render() skips set_mode and draws here
    env.render()
    return surface


def draw_grid(window, tiles, cols):
    """Scale each kitchen to fit its cell and blit it with a caption."""
    font = pygame.font.SysFont(None, 20)
    win_w, win_h = window.get_size()
    rows = math.ceil(len(tiles) / cols)
    cell_w, cell_h = win_w / cols, win_h / rows

    window.fill(BG)
    for i, (name, surface) in enumerate(tiles):
        col, row = i % cols, i // cols
        avail_w, avail_h = cell_w - 8, cell_h - LABEL_H - 8
        scale = min(avail_w / surface.get_width(), avail_h / surface.get_height())
        w, h = int(surface.get_width() * scale), int(surface.get_height() * scale)

        x = int(col * cell_w + (cell_w - w) / 2)
        y = int(row * cell_h + LABEL_H + (avail_h - h) / 2)
        window.blit(pygame.transform.smoothscale(surface, (w, h)), (x, y))

        caption = font.render(name, True, LABEL)
        window.blit(caption, (int(col * cell_w + (cell_w - caption.get_width()) / 2),
                              int(row * cell_h + 4)))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", type=int, default=None,
                        help="columns in the grid (default: roughly square)")
    args = parser.parse_args()

    names = layout_names()
    if not names:
        sys.exit(f"no .layout files in {LAYOUTS_DIR}")
    cols = args.cols or math.ceil(math.sqrt(len(names)))

    pygame.init()
    # a display must exist before the renderer can .convert() its sprites
    desktop = pygame.display.Info()
    window = pygame.display.set_mode((int(desktop.current_w * 0.9),
                                      int(desktop.current_h * 0.85)))
    pygame.display.set_caption(f"steak layouts ({len(names)})")

    print(f"rendering {len(names)} layouts from {LAYOUTS_DIR}")
    tiles = [(n, s) for n in names for s in [render_offscreen(n)] if s is not None]
    print(f"showing {len(tiles)} in a {cols}x{math.ceil(len(tiles) / cols)} grid")

    draw_grid(window, tiles, cols)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
