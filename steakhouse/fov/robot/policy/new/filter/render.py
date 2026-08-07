"""
RENDER an episode with the real pygame graphics -- baseline and baseline+module
side by side, the same checkpoint and the same human in both.

    # watch it live, side by side
    python render.py --layout steak_gc00 --fov 30 --seed 0 --tau 0

    # save every frame to PNGs (then make a gif/mp4 from them)
    python render.py --layout steak_gc07 --fov 30 --seed 0 --tau 0 --save frames/

    # one arm only, bigger
    python render.py --layout steak_cram --fov 30 --arm module --scale 60

Keys while it plays:  SPACE pause/resume   .  step one frame   q  quit

Uses SteakhouseStateVisualizer from overcooked_ai_py.visualization -- the same
renderer my_methods/old_bayesian_approximate/infer_bayesian.py uses, so the
sprites are the ones you already know.

The two arms are played to completion FIRST and then replayed together. They
diverge the moment the module overrides, and after that the same frame number is
a different situation in each kitchen -- that is expected, and the HUD marks the
tick it happened.
"""
import argparse
import os
import sys

import numpy as np
import torch

import baseline  # noqa: F401   sys.path shim
from baseline import Kitchen, Actor, make_human, stage_layouts
from inference import SamplingBayesFOVInference                 # noqa: E402
from qmdp import QMDPModule, sync_shadows                       # noqa: E402
from overcooked_ai_py.mdp.actions import Action                 # noqa: E402


def play(layout, fov, seed, use_module, k, depth, alpha, horizon, ckpt=None):
    """Play one episode; return (frames, info). A frame is a full state object."""
    kit = Kitchen(layout, horizon=horizon)
    act = Actor(layout, kit.obs_shape, ckpt_path=ckpt, device=torch.device("cpu"))
    kit.reset()
    hum = make_human(kit.mdp, fov, seed)
    act.reset()
    filt = SamplingBayesFOVInference(kit.mdp, None, [30, 60, 90, 120, 180, 360],
                                     human_agent_index=1)
    mod = QMDPModule(kit.mdp, act, k=k, depth=depth, alpha=alpha,
                     horizon=kit.horizon)
    frames, meta, deliveries = [], [], 0
    while kit.t < kit.horizon and not kit.mdp.is_terminal(kit.state):
        s = kit.state
        p = act.probs(kit.robot_obs())
        want = int(np.argmax(p))
        if use_module:
            idx = mod.choose(s, p, filt, seed=seed)
        else:
            sync_shadows(filt, s)
            idx = want
        frames.append(s.deepcopy())
        meta.append({"t": kit.t, "override": idx != want,
                     "map_fov": filt.map_fov(), "deliveries": deliveries})
        h, _info = hum.action(s)             # _info["subtask"] never read
        filt.update(s, h)
        sparse, done = kit.step(Action.INDEX_TO_ACTION[int(idx)], h)
        if sparse > 0:
            deliveries += int(round(sparse / float(kit.mdp.delivery_reward))) or 1
        if done:
            break
    return kit, frames, meta, deliveries


def main(argv=None):
    ap = argparse.ArgumentParser("render baseline vs baseline+module")
    ap.add_argument("--layout", default="steak_gc00")
    ap.add_argument("--fov", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None,
                    help="evidence gate; 0 = ungated so the module always acts")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arm", choices=["both", "base", "module"], default="both")
    ap.add_argument("--scale", type=int, default=40, help="pixels per tile")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames per second. 2 = comfortable, 1 = slow, 8 = fast")
    ap.add_argument("--save", default=None, help="write PNGs to this directory")
    args = ap.parse_args(argv)

    stage_layouts()
    if args.tau is not None:
        import qmdp
        qmdp.TAU_EFFORT = args.tau
    torch.set_num_threads(1)

    print("playing both arms...", flush=True)
    runs = {}
    for name, use_mod in (("base", False), ("module", True)):
        if args.arm != "both" and args.arm != name:
            continue
        kit, fr, meta, dl = play(args.layout, args.fov, args.seed, use_mod,
                                 args.k, args.depth, args.alpha, args.horizon,
                                 args.ckpt)
        runs[name] = (kit, fr, meta, dl)
        print("  %-7s %3d ticks, %d deliveries, %d overrides"
              % (name, len(fr), dl, sum(1 for m in meta if m["override"])),
              flush=True)

    import pygame
    from overcooked_ai_py.mdp.graphics import SPRITE_LENGTH

    #SteakHouseGridworld.render() is the fork's OWN renderer (overcooked_mdp.py
    #:2749) and it takes view_angle -- it draws the human's FOV CONE. It writes
    #to self.viewer, and only creates a display if that is still None. So we
    #hand each kitchen its own off-screen Surface and both can be drawn side by
    #side without fighting over the one display.
    pygame.init()
    names = [n for n in ("base", "module") if n in runs]
    k0 = runs[names[0]][0]
    tile = SPRITE_LENGTH
    gw, gh = k0.mdp.width * tile, k0.mdp.height * tile
    W = gw * len(names) + 12 * (len(names) - 1)
    H = gh + 52

    for n in names:
        runs[n][0].mdp.viewer = pygame.Surface((gw, gh))

    #ALWAYS create a display. The mdp's render() ends with pygame.display.update(),
    #which raises "No video mode has been set" against a bare Surface. When
    #saving we just run it under SDL_VIDEODRIVER=dummy, where set_mode works and
    #update() is a no-op.
    if args.save:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.makedirs(args.save, exist_ok=True)
        pygame.display.quit(); pygame.display.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "%s  fov=%d  seed=%d   |   left BASELINE   right +MODULE"
        % (args.layout, args.fov, args.seed))
    font = pygame.font.SysFont("couriernew", 16, bold=True)
    small = pygame.font.SysFont("couriernew", 13)
    clock = pygame.time.Clock()

    n_frames = max(len(runs[n][1]) for n in names)
    i, paused, running = 0, False, True
    while running and i < n_frames:
        if not args.save:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_PERIOD:
                        paused, i = True, min(i + 1, n_frames - 1)
                    elif e.key == pygame.K_COMMA:
                        paused, i = True, max(i - 1, 0)
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        args.fps = min(args.fps * 1.5, 30)
                    elif e.key == pygame.K_MINUS:
                        args.fps = max(args.fps / 1.5, 0.25)

        screen.fill((20, 20, 24))
        #GREEN HAT = ROBOT (player 0), BLUE HAT = HUMAN (player 1).
        #graphics.py PLAYER_HAT_COLOR = {0:'greenhat', 1:'bluehat'} and the
        #Kitchen seats the robot at index 0. The black area is everything
        #OUTSIDE the human's vision cone.
        #lay the legend out from the ACTUAL window width -- a 5x5 kitchen gives
        #a ~512px window and fixed pixel offsets ran the labels into each other
        leg = [("GREEN = ROBOT", (120, 230, 120)),
               ("BLUE = HUMAN", (120, 170, 255)),
               ("black = unseen", (150, 150, 150))]
        step = max(110, W // len(leg))
        for c, (txt, col) in enumerate(leg):
            screen.blit(small.render(txt, True, col), (6 + c * step, 28))
        x = 0
        for n in names:
            kit, fr, meta, dl = runs[n]
            j = min(i, len(fr) - 1)
            m = meta[j]
            #view_angle draws the HUMAN's cone -- the whole point of watching this
            kit.mdp.viewer.fill((255, 255, 255))
            try:
                kit.mdp.render(fr[j], "human", view_angle=args.fov)
            except Exception as ex:
                if i == 0:
                    print("render fallback (%s): %s" % (n, ex))
            screen.blit(kit.mdp.viewer, (x, 48))
            tag = "%-7s t=%-3d del=%d%s" % (n.upper(), m["t"], m["deliveries"],
                                            "  << OVERRIDE" if m["override"] else "")
            screen.blit(font.render(tag, True,
                        (255, 205, 60) if m["override"] else (225, 225, 225)),
                        (x + 4, 6))
            x += gw + 12

        if args.save:
            pygame.image.save(screen, os.path.join(args.save, "f%04d.png" % i))
            i += 1
        else:
            pygame.display.flip()
            clock.tick(max(args.fps, 0.25))
            if not paused:
                i += 1

    if args.save:
        print("wrote %d PNGs to %s" % (i, args.save))
        print("  ffmpeg -framerate %d -i %s/f%%04d.png -pix_fmt yuv420p out.mp4"
              % (args.fps, args.save))
    pygame.quit()
    for n in names:
        print("%-7s %d ticks, %d deliveries" % (n, len(runs[n][1]), runs[n][3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
