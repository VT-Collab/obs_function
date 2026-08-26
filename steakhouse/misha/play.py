"""Playable keyboard interface for the no_larping steakhouse.

    python play.py --layout back_bar --robot fov     the FoV-only filter
    python play.py --list                        the layouts you can name
    python play.py --list-robots                 every policy name

ONE FILTER: `fov*` knows no task knowledge at all -- it scores a plan by how
many of the ticks that job takes the human would see the robot on, capped at
`cap` ticks of baseline cost. Its predecessor, qmdp.py, scored a plan by
estimated ticks to FINISH the order and knew the recipe; it's deleted -- see
robot/methods.py's module docstring and RESULTS.md for what it did and why.
See watch.py's header for the full list and the cap sweep.

THE SCRIPTED HUMAN LOOKS FOR THINGS. Besides the ordinary ladder it will go and
LOOK for whatever it last saw the robot carrying, when it has not seen that item
put down and the item outranks its current job. `--no-look-for` is the control
and reproduces the plain ladder human action for action. It is one class
(human/limited_vision_human.py) with a flag, not two classes -- so the robot's
shadow humans are necessarily the same agent as the one in the seat.

You sit in ONE of the two seats and the code plays the other:

    HUMAN seat   limited field of view. Everything outside your cone is BLACK.
                 Ground you have walked past stays as a dark silhouette (terrain
                 is permanent knowledge), but what SITS on a counter, pot, board
                 or sink fades to '?' after --forget ticks -- a station you have
                 not looked at recently is UNKNOWN, not "empty". That is exactly
                 BeliefView's rule, so you play under the same observation model
                 as human/limited_vision_human.py. Your teammate is one of the
                 robot baselines.

    ROBOT seat   full view of the kitchen, because the robot observes s directly.
                 The area the human CANNOT see is greyed out -- not hidden from
                 you, just marked, so you can watch their blind spot swing around
                 as they turn and decide where to put a handoff. Your teammate is
                 LimitedVisionHuman on the chosen cone.

Which seat you take is the whole experiment in miniature. Play the human and you
feel what the cone costs. Play the robot and you have to guess what they can see
-- press TAB and the same Bayesian filter the robot policy uses will show you its
running estimate of their cone, inferred from nothing but the actions they take.

The teammate policies live in robot/methods.py and the setup screen prints that
registry verbatim -- `--list-robots` prints the same thing without opening a
window. It is a BASELINE COLUMN and ONE FILTER laid over it, nothing else.

BASELINES, in order of how much of you each one models. All theta-blind: none
reads your cone, so whatever an FOV-aware teammate beats them by is attributable
to the cone and nothing else. Each adds a little to the one above.

    greedy   nearest job first, every tick. No partner model of any kind.
    solo     greedy plus a contention penalty: any candidate CELL your
                   shortest path reaches sooner is demoted within its tier, so
                   it is ceded only when that tier holds something else. It is
                   not limited to stations -- it hits stash targets too, and
                   mostly does, pushing its drops away from the counters you are
                   near. Those counters are the only channel between the two
                   rooms, so this can backfire: over 54 matched episodes it
                   delivers fewer dishes than greedy (22 vs 35).
    handoff  solo plus staging, and it is two changes rather than one.
                   Holding something shareable with nothing better than a stash
                   on offer, it prepends every reachable free counter as a
                   target, ordered by distance from ITSELF -- it cannot see what
                   you can. It also biases the `stash` verb within its tier,
                   which leaves its own ranking unchanged but does shift the
                   distribution the fov layer reads. The default teammate.
    bayes     infers WHICH JOB you are doing from your moves and takes
                   another one. It models your intent, never your cone.
    bayes-noip    CONTROL: bayes with inverse planning switched off, so your
                   observed actions are never folded in. Not frozen at the prior
                   -- the belief is still carried and re-projected each tick --
                   but EVIDENCE is the only knob that differs, so the gap
                   between bayes-noip and bayes is what inference buys.

THE SAME FILTER OVER EACH ONE. fov-solo is not a cousin of fov; it is the same
class over a different floor. There is one row per baseline because the layer
degrades to the baseline it wraps, so that baseline is both its floor and the
thing it has to beat. Each infers your CONE from your actions, then searches over
HOW to do its baseline's job: which of the legal target cells, which first step,
whether to wait a tick. Watch the "it DEVIATES" line in the footer -- most ticks
it reads "follows its baseline" and the teammate simply IS that baseline; the
ticks it flips are the ones where the rollouts found a placement you will
actually see.

Press TAB and the "weighs" line shows the three sub-tasks your teammate was
choosing between this tick, with `>` on the one it took. Every teammate prints
it, but the number differs: a baseline shows the PROBABILITY it drew from -- so
the marked row is often NOT the top one, which is the draw showing itself -- and
a fov teammate shows a cost in ticks, lower better, where the marked row always
is the top one.

THE FoV-ONLY FILTER is the one to pick if what you want to
watch is a robot trying to stay where you can see it rather than a robot
rescheduling your kitchen. Cost is C = regret + detour - bonus, all in ticks of
the BASELINE's own cost, where the bonus pays `cap` ticks for the first tick of
the job your cone is on the robot and `decay` times as much for each one after
it. So the whole plan is worth at most cap/(1-decay) -- 8 ticks at the default
cap 4, decay 0.5 -- and THAT is the budget, not cap. No recipe, no orders left,
no handoff model.

    fov            over handoff, cap = 4, budget 8 ticks -- the headline row
    fov-greedy     over greedy
    fov-solo       over solo
    fov-bayes      over bayes
    fov-base       PARITY CONTROL: cap = 0, plays the baseline tick for tick.
                   Exact, not empirical -- it falls out of the cost being zero
                   for the baseline's own action.
    fov-fixed      THETA-BLIND CONTROL: cone frozen at 90.
    fov-free       UNBOUNDED CONTROL: cap = 1e6, no budget at all.
    fov-c1         the sweep, in ticks per sighting: 1, 2, 8, 10.7 -- budgets of
    fov-c2         2, 4, 16 and 21.4. One rung of the sub-task ladder is 21.4
    fov-c8         ticks, so every row at or below it provably cannot change
    fov-c21        which TIER of job the robot does -- only which cell it picks
                   and how it walks there. fov-c21 sits exactly on the rung.
    fov-d02        the decay sweep at a FIXED budget of 8 ticks: decay 0.2 spends
    fov-d08        nearly all of it on being seen once, decay 0.8 spreads it over
                   many sightings. Against `fov` (decay 0.5) this is the shape of
                   the spend with the size of it held still.

WHY EVERY BASELINE DRAWS. They take their sub-task from the distribution
the policy really induces (robot/nominal_policy/baselines.py) instead of
taking the argmax, and that is not a flavour -- it is what makes the comparison
mean anything. The filter consumes a DISTRIBUTION over sub-tasks; a
deterministic policy has none, so it could only be fed a Boltzmann
reconstruction of what a policy that never draws anything might have drawn, and
then scored against a baseline that never held those preferences. Given the
sub-task the low-level action is a pure function for every policy here, so the
sub-task choice is the only place the variation lives. There is no deterministic
variant to pick instead -- there used to be, and removing it is what made the
comparison honest.

--subtask-beta sets how sharp that distribution is and --subtask-rho how long a
draw is held. rho does not change the distribution: the sticky kernel leaves it
exactly stationary, which is asserted in test_baselines.py.

An instructions screen comes first: seat, cone width, teammate policy and layout
are all chosen there, and play starts on ENTER. --skip-setup goes straight in
with the flags given on the command line.

Controls
    arrows / WASD   walk (walking into a station just turns you to face it)
    SPACE           interact  (pick up, drop, chop, wash, plate, deliver)
    .               wait one tick
    TAB             show/hide what your teammate is thinking
    m               show/hide your memory (human seat: off = pure black)
    p               pause (real-time mode only)
    BACKSPACE       restart the episode
    ESC / q         quit

THE CLOCK. Your keys act the INSTANT you press them; the world does not
rate-limit you and a tick only happens because you took an action. The single
exception is the GRILL, which runs on a wall clock: put meat on it and it cooks
at --grill-fps ticks a second whether or not anybody does anything, so a steak
you started is getting done while you walk. The board and the sink do NOT work
that way, and that is the env's own rule rather than a choice made here --
chopping and washing only advance inside an INTERACT, so somebody has to stand
there and press. Everything else, teammate and horizon included, moves one tick
per action you take.

Two flags change that, and only these two:
    --fps N        puts the WHOLE world on a clock -- your teammate walks while
                   you idle and your keys are rate-limited to N/s. The grill's
                   own clock switches off, or env.step would cook it twice.
    --grill-fps 0  stops even the grill, which makes the game strictly
                   turn-based. Use it with --autopilot for a headless run.

Every flag
    --layout NAME     kitchen to run, from layout/layouts/ (default divide)
    --list            print the layout names and exit
    --list-robots     print the teammate policies above, grouped, and exit
    --seat human|robot  which seat you play (default human). Also on the setup
                      screen, which is where most people will pick it
    --fov DEG         the human seat's cone: 30, 60, 90, 180 or 360 (default 90).
                      Yours if you take that seat, the code's if you do not
    --partner NAME    teammate policy when you play the human (default
                      handoff).
                      --robot is accepted as a synonym so a command line copies
                      straight over from watch.py. The list is robot/methods.py
                      and nothing else -- both scripts read that one registry
    --horizon N       episode length in ticks (default 400)
    --forget N        ticks before a station's contents fade to '?' in the human
                      seat's memory. Terrain is never forgotten, only contents
    --no-memory       human seat: no memory at all, everything outside the cone
                      is black. Harder than the model the agents actually run
    --no-infer        robot seat: skip the FOV filter readout under TAB. The
                      readout is only ever a readout -- nothing acts on it when
                      you are in the seat
    --scale PX        pixels per tile; 0 (default) fits the screen
    --grill-fps N     how fast the grill cooks in real time (default 2). It is
                      the only station on a clock; board and sink move only
                      under an INTERACT. 0 makes the game strictly turn-based
    --fps N           put the WHOLE world on a clock at N ticks/s. 0 (default)
                      means only the grill self-ticks. See THE CLOCK above
    --seed N          seeds the teammate and the autopilot (default 0)
    --human-index I   which player index wears the cone: 0 or 1 (default 1, the
                      blue hat, as everywhere else in this package)
    --top-k N         legacy -- does not reach fov-*, see robot/methods.py's
                      `_fov()` and the DESIGN.md knob table for what's real
    --depth N         legacy -- does not reach fov-*, same as --top-k above
    --subtask-beta B  stochastic methods: sharpness of the sub-task
                      distribution, pi ~ value**B (default 8). B->inf is the
                      deterministic argmax, B=1 is bayes's own team-value prior,
                      B=0 is uniform over legal sub-tasks
    --subtask-rho R   stochastic methods: probability of HOLDING the current
                      sub-task each tick (default 0.95, ~20 ticks). R=0 re-draws
                      every tick, R=1 commits until the job is done. It does not
                      change the distribution, only how long a draw lasts
    --skip-setup      straight into the game on the flags given here, no setup
                      screen
    --autopilot       let LimitedVisionHuman drive YOUR seat too. Pair it with
                      --fps to pace it, or use watch.py to watch properly
    --max-ticks N     stop after N ticks; 0 (default) means run to --horizon
    --log FILE        write the per-tick trajectory to this json
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import overcooked_ai_py                                          # noqa: E402

LAYOUTS_DIR = os.path.join(HERE, "layout", "layouts")
overcooked_ai_py.LAYOUTS_DIR = LAYOUTS_DIR   # read_layout_dict reads this at call time

import pygame                                                    # noqa: E402
from overcooked_ai_py.mdp.actions import Action, Direction        # noqa: E402
from overcooked_ai_py.mdp.graphics import (SPRITE_LENGTH,         # noqa: E402
                                           blit_terrain)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv     # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402

from common import geometry as geo                                # noqa: E402
# ONE HUMAN CLASS, and the look-for is a FLAG on it rather than a subclass.
# enable_look_for defaults to True and =False reproduces the plain ladder agent
# action for action, checked over 5 layouts x 5 cones x 2 seeds -- so
# --no-look-for is a control on one agent rather than a second agent, and the
# robot's shadows cannot end up being a different class from the seat.
from human.limited_vision_human import (LimitedVisionHuman,       # noqa: E402
                                        FORGET_HORIZON)
from robot.filter.core.fov_posterior import FOVPosterior              # noqa: E402
from robot.methods import (GROUPS, HOTKEYS, METHOD_KEYS,          # noqa: E402
                           in_group, listing, make_robot, resolve)

S = SPRITE_LENGTH                 # native px per tile, fixed by the renderer
DYNAMIC = set("XPBW")             # cells whose CONTENTS change, so beliefs expire
FOV_CHOICES = [30, 60, 90, 180, 360]

BG = (18, 18, 22)
FG = (232, 232, 236)
DIM = (150, 150, 158)
HOT = (255, 205, 60)
ACCENT = (120, 200, 255)
GREEN = (120, 230, 120)
BLUE = (120, 170, 255)

MEM_DIM_NEAR = 105                # a just-seen tile, redrawn from memory
MEM_DIM_FAR = 195                 # a tile whose contents are about to expire
STALE_DIM = 215                   # expired: terrain silhouette and a '?'
BLIND_GREY = 120                  # robot seat: the shade over the human's blind area

KEY_TO_ACTION = {
    pygame.K_UP: Direction.NORTH, pygame.K_w: Direction.NORTH,
    pygame.K_DOWN: Direction.SOUTH, pygame.K_s: Direction.SOUTH,
    pygame.K_LEFT: Direction.WEST, pygame.K_a: Direction.WEST,
    pygame.K_RIGHT: Direction.EAST, pygame.K_d: Direction.EAST,
    pygame.K_SPACE: Action.INTERACT,
    pygame.K_PERIOD: Action.STAY,
}

RECIPE = [
    "A steak dish = washed plate + cooked steak + chopped garnish.",
    "  meat M -> grill P          the grill then cooks on its own",
    "  onion O -> board B         stand there, INTERACT again and again to chop",
    "  plate D -> sink W          stand there, INTERACT again and again to wash",
    "  washed plate + ready grill -> steak dish;  + ready board -> the dish",
    "  carry the finished dish to the serving hatch S",
    "Counters X hold anything, and are the only way to hand something over.",
]


def layout_names():
    return sorted(f[:-len(".layout")] for f in os.listdir(LAYOUTS_DIR)
                  if f.endswith(".layout"))


def wrap(font, text, width):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.size(trial)[0] <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
class View:
    """Draws the kitchen from one seat's point of view.

    Holds the human seat's MEMORY, because a cone without memory is not a game:
    you would forget the kitchen every time you turned your head. Terrain is
    remembered forever, contents expire after `forget` ticks, and nothing is
    remembered that was not inside the cone when it happened.
    """

    def __init__(self, mdp, human_idx, fov, forget, memory=True):
        self.mdp, self.human_idx, self.fov = mdp, human_idx, fov
        self.robot_idx = 1 - human_idx
        self.forget, self.memory = forget, memory
        self.w, self.h = mdp.width, mdp.height
        self.px = (self.w * S, self.h * S)

        self.truth = pygame.Surface(self.px)
        self.terrain = pygame.Surface(self.px)      # furniture only, never changes
        for y in range(self.h):
            for x in range(self.w):
                blit_terrain(x, y, mdp.terrain_mtx, self.terrain, "human")

        self._shades = {}
        self._qmark = None
        self.reset()

    def reset(self):
        self.mem = {}          # cell -> (tile, tick last seen)
        self.ghost = None      # (tile, cell, tick) -- WHERE you last saw the robot
        self.hands = None      # (held, tick)       -- WHAT they were carrying
        self.seen = set()

    # -- helpers ------------------------------------------------------------
    def _shade(self, alpha, colour=(0, 0, 0)):
        key = (alpha, colour)
        if key not in self._shades:
            s = pygame.Surface((S, S))
            s.fill(colour)
            s.set_alpha(alpha)
            self._shades[key] = s
        return self._shades[key]

    def _tile(self, src, cell):
        t = pygame.Surface((S, S))
        t.blit(src, (0, 0), pygame.Rect(cell[0] * S, cell[1] * S, S, S))
        return t

    def _question(self):
        if self._qmark is None:
            self._qmark = pygame.font.SysFont("couriernew", 26, bold=True) \
                                .render("?", True, (175, 175, 185))
        return self._qmark

    # -- per-tick -----------------------------------------------------------
    def observe(self, state, t):
        """Redraw ground truth, then fold whatever the cone caught into memory."""
        self.mdp.viewer = self.truth
        self.truth.fill((255, 255, 255))
        self.mdp.render(state, "human")

        h = state.players[self.human_idx]
        self.seen = geo.visible_cells(self.mdp.terrain_mtx, tuple(h.position),
                                      tuple(h.orientation), self.fov)
        # a cell with a chef standing on it is remembered as bare terrain: you
        # remember the kitchen, not where somebody happened to be standing.
        occupied = {tuple(p.position) for p in state.players}
        for c in self.seen:
            self.mem[c] = (self._tile(self.terrain if c in occupied else self.truth, c), t)

        # Same split as BeliefView: WHERE they are is falsified the moment you
        # look at the cell and they are not in it; WHAT they were carrying is
        # not -- an empty tile says nothing about their hands -- so it lives on
        # its own clock and outlives the position.
        rp = tuple(state.players[self.robot_idx].position)
        if rp in self.seen:
            obj = state.players[self.robot_idx].held_object
            self.ghost = (self._tile(self.truth, rp), rp, t)
            self.hands = (obj.name if obj else None, t)
        elif self.ghost and (self.ghost[1] in self.seen
                             or t - self.ghost[2] > self.forget):
            self.ghost = None
        if self.hands and t - self.hands[1] > self.forget:
            self.hands = None

    # -- the two seats ------------------------------------------------------
    def human_seat(self, t):
        """What the cone allows. Outside it: black, or memory if you kept it."""
        surf = pygame.Surface(self.px)
        surf.fill((0, 0, 0))
        for c in self.seen:
            # straight from ground truth, NOT from the memory tile: memory stores
            # a chef's cell as bare terrain, so drawing memory here would erase
            # both chefs from the one place you can actually see them.
            at = (c[0] * S, c[1] * S)
            surf.blit(self.truth, at, pygame.Rect(at, (S, S)))
        for c, (tile, seen_t) in self.mem.items():
            at = (c[0] * S, c[1] * S)
            if c in self.seen or not self.memory:
                continue
            age = t - seen_t
            ch = self.mdp.terrain_mtx[c[1]][c[0]]
            if ch in DYNAMIC and age > self.forget:
                # belief expired. The terrain is still known -- what is ON it is not.
                surf.blit(self.terrain, at, pygame.Rect(at, (S, S)))
                surf.blit(self._shade(STALE_DIM), at)
                q = self._question()
                surf.blit(q, (at[0] + (S - q.get_width()) // 2,
                              at[1] + (S - q.get_height()) // 2))
            else:
                # dynamic cells fade as their contents go stale; walls, floor and
                # dispensers sit at a fixed dim, because terrain is permanent
                # knowledge and there is nothing about it left to forget.
                frac = (min(age / float(max(self.forget, 1)), 1.0)
                        if ch in DYNAMIC else 0.6)
                surf.blit(tile, at)
                surf.blit(self._shade(int(MEM_DIM_NEAR +
                                          frac * (MEM_DIM_FAR - MEM_DIM_NEAR))), at)

        if self.memory and self.ghost and self.ghost[1] not in self.seen:
            tile, cell, _ = self.ghost
            at = (cell[0] * S, cell[1] * S)
            surf.blit(tile, at)
            surf.blit(self._shade(MEM_DIM_FAR), at)
            pygame.draw.rect(surf, (70, 130, 70), pygame.Rect(at, (S, S)), 2)
        return surf

    def robot_seat(self):
        """Everything -- with the human's blind area greyed, not hidden."""
        surf = self.truth.copy()
        grey = self._shade(BLIND_GREY, (70, 70, 82))
        for y in range(self.h):
            for x in range(self.w):
                if (x, y) not in self.seen:
                    surf.blit(grey, (x * S, y * S))
        for (x, y) in self.seen:            # outline the cone so the edge is legible
            at = (x * S, y * S)
            for dx, dy, a, b in ((0, -1, (0, 0), (S, 0)), (0, 1, (0, S), (S, S)),
                                 (-1, 0, (0, 0), (0, S)), (1, 0, (S, 0), (S, S))):
                if (x + dx, y + dy) not in self.seen:
                    pygame.draw.line(surf, (215, 185, 90),
                                     (at[0] + a[0], at[1] + a[1]),
                                     (at[0] + b[0], at[1] + b[1]), 2)
        return surf


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
class Episode:
    """The env, both seats, and everything the HUD needs to describe them."""

    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.human_idx = args.human_index
        self.robot_idx = 1 - self.human_idx
        self.you_idx = self.human_idx if cfg["seat"] == "human" else self.robot_idx
        self.mate_idx = 1 - self.you_idx

        self.mdp = SteakHouseGridworld.from_layout_name(cfg["layout"])
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=args.horizon,
                                          info_level=0)
        self.view = View(self.mdp, self.human_idx, cfg["fov"], args.forget,
                         memory=not args.no_memory)
        self.orders_total = len(self.env.state.order_list or [])
        self.reset()

    def reset(self):
        cfg, args = self.cfg, self.args
        self.env.reset()
        self.view.reset()
        self.post = None                  # the filter's belief about the cone

        if cfg["seat"] == "human":
            self.mate, self.post = make_robot(cfg["partner"], self.mdp,
                                              self.robot_idx, self.human_idx,
                                              args.seed, args.top_k, args.depth,
                                              beta=args.subtask_beta,
                                              rho=args.subtask_rho,
                                              human_kw={
                                                  "forget_horizon": args.forget,
                                                  "enable_look_for":
                                                      not args.no_look_for})
            self.mate_name = cfg["partner"]
        else:
            self.mate = LimitedVisionHuman(self.mdp, cfg["fov"],
                                           agent_index=self.human_idx,
                                           forget_horizon=args.forget,
                                           seed=args.seed,
                                           enable_look_for=not args.no_look_for)
            self.mate_name = ("fov%d human" % cfg["fov"]
                              + ("" if not args.no_look_for else " (no look-for)"))
            # you are the robot, so nobody is running the filter for you. Run it
            # anyway, purely as a readout: this is the estimate the FOV-aware
            # policy would be acting on, from the same evidence you have.
            # BUILT FROM THE SAME KWARGS AS self.mate above, for the same reason
            # the policy's own posterior is: the shadows ARE the model, and a
            # shadow that forgets at a different rate or does not look for what
            # it saw you carrying is scoring cones against a human who is not in
            # the seat. This readout was left on the defaults while the mate took
            # --forget and --no-look-for, so the TAB panel could disagree with
            # the very human on screen.
            if not args.no_infer:
                self.post = FOVPosterior(self.mdp, human_index=self.human_idx,
                                         seed=args.seed,
                                         human_kw={
                                             "forget_horizon": args.forget,
                                             "enable_look_for":
                                                 not args.no_look_for})

        # The autopilot is the SAME agent as the teammate, on the other seat, and
        # BUILT THE SAME WAY. It was left on the default flags once while the
        # teammate was configured, and --autopilot then silently demonstrated a
        # different human from the one the robot is being scored against.
        self.autopilot = LimitedVisionHuman(self.mdp, cfg["fov"],
                                            agent_index=self.you_idx,
                                            forget_horizon=args.forget,
                                            seed=args.seed + 1,
                                            enable_look_for=not args.no_look_for)
        self.events = ["press a direction key to start"]
        self.reward, self.done, self.mate_info, self.trace = 0, False, {}, []
        self.view.observe(self.env.state, self.env.t)

    # -- the tick -----------------------------------------------------------
    def step(self, your_action):
        state = self.env.state
        held_before = self.held(self.you_idx)

        mate_action, self.mate_info = self.mate.action(state)
        human_action = your_action if self.you_idx == self.human_idx else mate_action
        if self.post is not None:
            # the filter learns theta from the human seat's actions and nothing else
            self.post.update(state, human_action)

        joint = [None, None]
        joint[self.you_idx] = your_action
        joint[self.mate_idx] = mate_action
        nxt, rew, done, _ = self.env.step(tuple(joint))

        self.reward += rew
        self.done = done or self.mdp.is_terminal(nxt)
        if rew > 0:
            self._event("DELIVERED  +%d" % rew)
        elif self.held(self.you_idx) != held_before:
            self._event("you now hold %s" % (self.held(self.you_idx) or "nothing"))

        if self.args.log:
            self.trace.append({
                "t": self.env.t - 1,
                "you": _action_name(your_action),
                "mate": _action_name(mate_action),
                "mate_subtask": _subtask_str(self.mate_info.get("subtask")),
                "fov_post": self.posterior_str(),
                "reward": rew,
            })
        self.view.observe(nxt, self.env.t)

    def tick_grill(self):
        """Advance the ONE clock that runs whether or not anybody acts.

        mdp.step_environment_effects only touches steaks sitting on a 'P', so
        this cooks and nothing else. The board and the sink deliberately do NOT
        move here -- the env only advances those inside resolve_interacts, i.e.
        somebody has to stand there and press INTERACT. Same thing the study
        harness does on its 1s TIMER (multi_player_user_study.py:on_event).

        Keeping this OFF the action loop is what stops the world clock from
        rate-limiting your keys: you act as fast as you can press, and the grill
        cooks in real time underneath you.
        """
        self.mdp.step_environment_effects(self.env.state)
        self.view.observe(self.env.state, self.env.t)

    def held(self, idx):
        obj = self.env.state.players[idx].held_object
        return obj.name if obj else None

    def _event(self, msg):
        self.events.append("t%-4d %s" % (self.env.t, msg))
        del self.events[:-4]

    @property
    def delivered(self):
        return self.orders_total - len(self.env.state.order_list or [])

    def posterior(self):
        if self.post is not None:
            return dict(self.post.p)
        return self.mate_info.get("fov_post")

    def posterior_str(self):
        p = self.posterior()
        return {} if not p else {str(k): round(v, 4) for k, v in p.items()}

    def summary(self):
        return {"layout": self.cfg["layout"], "seat": self.cfg["seat"],
                "fov": self.cfg["fov"], "partner": self.mate_name,
                "ticks": self.env.t, "delivered": self.delivered,
                "orders": self.orders_total, "reward": self.reward}

    def write_log(self):
        if not self.args.log:
            return
        out = self.summary()
        out["trace"] = self.trace
        with open(self.args.log, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote %s" % self.args.log)


def _action_name(a):
    return "INTERACT" if a == Action.INTERACT else str(tuple(a))


def _subtask_str(sub):
    return "-" if not sub else "%s %s %s" % sub


# ---------------------------------------------------------------------------
# screens
# ---------------------------------------------------------------------------
class App:
    # FOOT fits the tallest footer at 16px a line: events, teammate recall,
    # their sub-task, the top-3 it was weighing, the execution-layer line, the
    # partner-intent guess and the cone posterior. An eighth line means raising
    # this. WIDTH is the constraint that is easier to miss -- an over-long line
    # is cut off at MIN_W with nothing on screen to say so, which is why
    # baselines.short_subtask exists and carries the measured numbers.
    HEAD, FOOT, PAD = 62, 128, 12
    MIN_W = 980                       # the HUD lines must not get clipped
    # Tall enough for the whole method table at one line each. _draw_setup
    # asserts it still fits, so adding a method to robot/methods.py fails loudly
    # here rather than quietly rendering the last row off the bottom edge.
    SETUP_SIZE = (1000, 880)

    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.display.set_caption("no_larping -- steakhouse with a limited-FOV teammate")
        self.screen = pygame.display.set_mode(self.SETUP_SIZE)
        self.font = pygame.font.SysFont("couriernew", 16)
        self.bold = pygame.font.SysFont("couriernew", 16, bold=True)
        self.small = pygame.font.SysFont("couriernew", 13)
        self.smallbold = pygame.font.SysFont("couriernew", 13, bold=True)
        self.big = pygame.font.SysFont("couriernew", 26, bold=True)
        self.clock = pygame.time.Clock()
        self.layouts = layout_names()
        self.cfg = {"seat": args.seat, "fov": args.fov,
                    "partner": args.partner, "layout": args.layout}

    def run(self):
        while True:
            if not self.args.skip_setup:
                self.setup_screen()
            ep = self.play()
            if ep is None:
                break
            ep.write_log()
            print("%s  seat=%s fov=%d partner=%s  ->  %d/%d orders in %d ticks"
                  % (ep.cfg["layout"], ep.cfg["seat"], ep.cfg["fov"],
                     ep.mate_name, ep.delivered, ep.orders_total, ep.env.t))
            if self.args.skip_setup or not self.end_screen(ep):
                break
        pygame.quit()

    # -- instructions -------------------------------------------------------
    def setup_screen(self):
        cfg = self.cfg
        self.screen = pygame.display.set_mode(self.SETUP_SIZE)
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if e.type != pygame.KEYDOWN:
                    continue
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if e.key == pygame.K_h:
                    cfg["seat"] = "human"
                elif e.key == pygame.K_r:
                    cfg["seat"] = "robot"
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    cfg["fov"] = FOV_CHOICES[e.key - pygame.K_1]
                elif e.key in (pygame.K_UP, pygame.K_DOWN):
                    # There are more methods than there are free letters, so the
                    # arrows walk the whole table and the five original letters
                    # survive as shortcuts into it.
                    i = METHOD_KEYS.index(cfg["partner"])
                    cfg["partner"] = METHOD_KEYS[
                        (i + (1 if e.key == pygame.K_DOWN else -1)) % len(METHOD_KEYS)]
                elif e.unicode.lower() in HOTKEYS:
                    cfg["partner"] = HOTKEYS[e.unicode.lower()]
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    i = self.layouts.index(cfg["layout"])
                    cfg["layout"] = self.layouts[
                        (i + (1 if e.key == pygame.K_RIGHTBRACKET else -1)) % len(self.layouts)]
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return cfg
            self._draw_setup()
            self.clock.tick(30)

    def _draw_setup(self):
        cfg = self.cfg
        self.screen.fill(BG)
        w = self.screen.get_width()
        y = 24

        def line(text, colour=FG, font=None, dy=21, x=30):
            nonlocal y
            self.screen.blit((font or self.font).render(text, True, colour), (x, y))
            y += dy

        line("NO_LARPING -- steakhouse with a limited-FOV teammate", ACCENT,
             self.big, 42)
        for t in wrap(self.font,
                      "Two cooks, one kitchen, three steak orders. One seat sees "
                      "the whole kitchen; the other only sees what falls inside a "
                      "forward vision cone. Pick a seat and the code plays the "
                      "other one.", w - 60):
            line(t, FG)
        y += 10

        line("YOUR SEAT", ACCENT, self.bold, 24)
        for key, name, blurb in (("H", "human",
                                  "limited cone -- outside it is BLACK. Teammate "
                                  "is a robot baseline."),
                                 ("R", "robot",
                                  "full view -- the human's blind area is GREYED. "
                                  "Teammate is the fov human.")):
            sel = cfg["seat"] == name
            line("[%s] %-6s %s" % (key, name.upper(), blurb),
                 HOT if sel else DIM, self.bold if sel else self.font, 20, 50)
        y += 10

        line("VISION CONE  (the human seat's, whichever of you is in it)",
             ACCENT, self.bold, 24)
        x = 50
        for i, f in enumerate(FOV_CHOICES):
            sel = cfg["fov"] == f
            img = (self.bold if sel else self.font).render(
                "[%d] %d deg" % (i + 1, f), True, HOT if sel else DIM)
            self.screen.blit(img, (x, y))
            x += img.get_width() + 18
        y += 32

        line("TEAMMATE POLICY  (used when you take the human seat -- UP/DOWN "
             "moves, or press a letter)", ACCENT, self.bold, 22)
        for group in GROUPS:
            line(group, ACCENT, self.small, 16, 50)
            for m in in_group(group):
                sel = cfg["partner"] == m.key
                line("%s %-12s %s" % ("[%s]" % m.hotkey.upper() if m.hotkey
                                      else "   ", m.key, m.blurb),
                     HOT if sel else DIM,
                     self.smallbold if sel else self.small, 16, 62)
        y += 8

        line("LAYOUT   %s   ( [ and ] cycle through all %d )"
             % (cfg["layout"], len(self.layouts)), ACCENT, self.bold, 26)

        for t in RECIPE:
            line(t, DIM, self.small, 16, 30)
        y += 6
        line("ENTER to start        ESC to quit", FG, self.bold)
        # The method table grows every time robot/methods.py does. Fail here, on
        # the first frame, rather than silently clipping the last row.
        assert y <= self.screen.get_height(), (
            "setup screen overflows SETUP_SIZE by %dpx -- raise it"
            % (y - self.screen.get_height()))
        pygame.display.flip()

    # -- the game -----------------------------------------------------------
    def play(self):
        args = self.args
        ep = Episode(self.cfg, args)
        gw, gh = ep.mdp.width, ep.mdp.height
        info = pygame.display.Info()
        fit = min((info.current_w * 0.92 - 2 * self.PAD) / gw,
                  (info.current_h * 0.82 - self.HEAD - self.FOOT) / gh)
        scale = args.scale or max(14, min(S, int(fit)))
        pw, ph = gw * scale, gh * scale
        self.screen = pygame.display.set_mode(
            (max(pw + 2 * self.PAD, self.MIN_W), ph + self.HEAD + self.FOOT))

        show_mate, paused, dirty = False, False, True
        # you will press keys faster than the world ticks, so the last few are
        # queued rather than dropped -- but only a few, or the controls start
        # feeling like they belong to somebody else.
        queue = collections.deque(maxlen=3)
        interval = 1.0 / args.fps if args.fps > 0 else 0.0
        # the grill's own clock. Off when the whole world is already self-ticking,
        # or env.step would cook the steak twice per second.
        grill = (1.0 / args.grill_fps
                 if args.grill_fps > 0 and args.fps <= 0 else 0.0)
        next_tick = time.time() + interval
        next_grill = time.time() + grill
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type != pygame.KEYDOWN:
                    continue
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None
                if e.key == pygame.K_BACKSPACE:
                    ep.reset()
                    queue.clear()
                    dirty = True
                elif e.key == pygame.K_TAB:
                    show_mate, dirty = not show_mate, True
                elif e.key == pygame.K_m:
                    ep.view.memory, dirty = not ep.view.memory, True
                elif e.key == pygame.K_p:
                    paused, dirty = not paused, True
                elif e.key in KEY_TO_ACTION and not ep.done:
                    queue.append(KEY_TO_ACTION[e.key])

            if args.autopilot and not ep.done and not paused:
                queue.clear()
                queue.append(ep.autopilot.action(ep.env.state)[0])

            if paused or ep.done:
                next_tick = time.time() + interval    # don't bank up ticks
                next_grill = time.time() + grill
            else:
                if queue and (args.fps <= 0 or time.time() >= next_tick):
                    # your key acts NOW. Only --fps > 0 makes the world clock
                    # gate it, and that is not the default any more.
                    ep.step(queue.popleft())
                    next_tick = time.time() + interval
                    dirty = True
                elif args.fps > 0 and time.time() >= next_tick:
                    ep.step(Action.STAY)              # full world clock: idling costs
                    next_tick = time.time() + interval
                    dirty = True
                if grill and time.time() >= next_grill:
                    ep.tick_grill()                   # the grill cooks regardless
                    next_grill = time.time() + grill
                    dirty = True

            if dirty:
                self._draw_game(ep, scale, (pw, ph), show_mate, paused)
                dirty = False
            if ep.done:
                return ep
            if args.max_ticks and ep.env.t >= args.max_ticks:
                return ep
            self.clock.tick(60 if args.fps <= 0 else max(30, int(args.fps * 4)))

    def _draw_game(self, ep, scale, size, show_mate, paused):
        self.screen.fill(BG)
        seat_human = ep.cfg["seat"] == "human"
        panel = ep.view.human_seat(ep.env.t) if seat_human else ep.view.robot_seat()
        self.screen.blit(pygame.transform.smoothscale(panel, size),
                         (self.PAD, self.HEAD))

        you_blue = ep.you_idx == 1
        head = ("t %d/%d   orders %d/%d   you: %s (%s hat)   holding: %s   "
                "teammate: %s"
                % (ep.env.t, self.args.horizon, ep.delivered, ep.orders_total,
                   ep.cfg["seat"].upper(), "BLUE" if you_blue else "GREEN",
                   ep.held(ep.you_idx) or "nothing", ep.mate_name))
        self.screen.blit(self.bold.render(head, True, BLUE if you_blue else GREEN),
                         (self.PAD, 8))
        sub = ("fov %d -- black is what you cannot see" % ep.cfg["fov"]
               if seat_human else
               "fov %d -- grey is what the HUMAN cannot see (you see it all)"
               % ep.cfg["fov"])
        clock = ("PAUSED" if paused else
                 "whole world on a %.1f/s clock" % self.args.fps
                 if self.args.fps > 0 else
                 "grill cooking at %.1f/s, the rest waits for you"
                 % self.args.grill_fps if self.args.grill_fps > 0 else
                 "turn-based")
        self.screen.blit(self.small.render("%s   |   %s" % (sub, clock), True,
                                           HOT if paused else DIM), (self.PAD, 29))
        self.screen.blit(self.small.render(
            "arrows/WASD move   SPACE interact   . wait   p pause   TAB teammate"
            "   m memory   BACKSPACE restart   ESC quit", True, DIM), (self.PAD, 44))

        y = self.HEAD + size[1] + 6
        for text, colour in self._footer(ep, show_mate):
            self.screen.blit(self.small.render(text, True, colour), (self.PAD, y))
            y += 16

        if ep.done:
            self._banner("EPISODE OVER -- %d/%d orders in %d ticks"
                         % (ep.delivered, ep.orders_total, ep.env.t))
        pygame.display.flip()

    def _recall(self, ep):
        """What you actually know about your teammate right now. Position dies
        the moment you look and they are gone; their hands outlive it."""
        v = ep.view
        held = (v.hands[0] or "nothing") if v.hands else None
        if tuple(ep.env.state.players[ep.mate_idx].position) in v.seen:
            return ("you can see them -- holding %s" % held, DIM)
        if v.ghost:
            return ("last saw them %d ticks ago at %s, holding %s"
                    % (ep.env.t - v.ghost[2], v.ghost[1], held), DIM)
        if v.hands:
            return ("WHERE they are: unknown.  last saw them holding %s, "
                    "%d ticks ago" % (held, ep.env.t - v.hands[1]), DIM)
        return ("no idea where they are or what they are carrying", DIM)

    def _footer(self, ep, show_mate):
        out = [(" | ".join(ep.events[-2:]), FG)]
        if ep.cfg["seat"] == "human":
            out.append(self._recall(ep))
        if not show_mate:
            out.append(("TAB shows what your teammate is doing", DIM))
            return out
        out.append(("teammate subtask: %s" % _subtask_str(ep.mate_info.get("subtask")),
                    HOT))
        if ep.mate_info.get("top3"):
            # THE THREE THINGS IT WAS CHOOSING BETWEEN, with the one it took
            # marked `>`. Every teammate emits this, baseline and filter alike, so
            # the line is comparable across the whole method table -- but the
            # NUMBER is not the same quantity in the two cases and the label says
            # which it is: a baseline shows the PROBABILITY it drew from, the
            # filter shows a cost in ticks, where lower wins.
            #
            # Watching a baseline, the marked row is often not the top one: that
            # is the draw, and it is the fastest way to see a sampled policy
            # behaving unlike a greedy one. Watching the filter, the marked row IS
            # the cheapest, and the interesting comparison is against the
            # baseline's own ordering -- when they disagree, the layer has
            # re-ranked the job.
            kind = ep.mate_info.get("top3_kind", "p")
            fmt = (lambda s: "%.2f" % s) if kind == "p" else (lambda s: "%.0ft" % s)
            out.append(("weighs %-5s  %s%s"
                        % ("p" if kind == "p" else "ticks",
                           "   ".join("%s%s %s" % (">" if take else " ", lab, fmt(sc))
                                      for lab, sc, take in ep.mate_info["top3"]),
                           ("   [%s]" % ep.mate_info["subtask_redraw"]
                            if ep.mate_info.get("subtask_redraw") else "")), ACCENT))
        if "theta_entropy" in ep.mate_info:
            # THE EXECUTION LAYER. Almost every tick this reads "follows
            # baseline" and the teammate is exactly its nominal policy; the ticks
            # where it flips to DEVIATES are the ones where the rollouts found a
            # placement worth leaving that policy for. Sitting in the human seat
            # with this on screen is the only way to watch it happen live.
            dev = ep.mate_info.get("deviated")
            plan = ep.mate_info.get("plan")
            bits = ["it %-16s" % ("DEVIATES" if dev else "follows its baseline")]
            if plan:
                bits.append("plan %s@%s" % (plan[0], plan[1]))
            if dev and ep.mate_info.get("gain") is not None:
                bits.append("gain %+.2f" % ep.mate_info["gain"])
            bits.append("H(theta) %.2f over %d cone%s"
                        % (ep.mate_info["theta_entropy"],
                           ep.mate_info.get("n_cones", 1),
                           "" if ep.mate_info.get("n_cones", 1) == 1 else "s"))
            out.append(("   ".join(bits), HOT if dev else DIM))
        if "partner_subtask" in ep.mate_info:
            # a teammate that infers YOUR INTENT rather than your cone
            out.append(("it thinks YOUR job is  %s   p=%.2f  H=%.2f bits"
                        % (_subtask_str(ep.mate_info["partner_subtask"]),
                           ep.mate_info.get("partner_p") or 0.0,
                           ep.mate_info.get("partner_entropy") or 0.0), ACCENT))
        post = ep.posterior()
        if post:
            top = max(post, key=post.get)
            bars = "  ".join("%d:%s%.2f" % (f, "*" if f == top else " ", p)
                             for f, p in sorted(post.items()))
            who = "YOUR" if ep.cfg["seat"] == "human" else "the human's"
            out.append(("filter's belief about %s cone   %s" % (who, bars), HOT))
        elif ep.cfg["seat"] == "robot":
            out.append(("the fov filter is off (drop --no-infer to see its "
                        "estimate of the human's cone)", DIM))
        else:
            out.append(("%s is theta-blind by construction -- the fov-* methods "
                        "are the ones that infer your cone" % ep.mate_name, DIM))
        return out

    def _banner(self, text):
        img = self.bold.render(text, True, (20, 20, 20))
        box = pygame.Surface((img.get_width() + 24, img.get_height() + 16))
        box.fill(HOT)
        box.blit(img, (12, 8))
        self.screen.blit(box, ((self.screen.get_width() - box.get_width()) // 2,
                               self.screen.get_height() // 2))

    # -- end ----------------------------------------------------------------
    def end_screen(self, ep):
        """True to go round again, False to quit."""
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False
                    if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return True
            self.screen.fill(BG)
            lines = [
                ("all three orders served!" if ep.delivered == ep.orders_total
                 else "time up.", ACCENT, self.big),
                ("%d/%d orders in %d ticks   (reward %d)"
                 % (ep.delivered, ep.orders_total, ep.env.t, ep.reward), FG, self.font),
                ("seat %s   fov %d   teammate %s   layout %s"
                 % (ep.cfg["seat"], ep.cfg["fov"], ep.mate_name, ep.cfg["layout"]),
                 DIM, self.font),
                ("ENTER to set up another run, ESC to quit", FG, self.font),
            ]
            y = self.screen.get_height() // 2 - 70
            for text, colour, font in lines:
                img = font.render(text, True, colour)
                self.screen.blit(img, ((self.screen.get_width() - img.get_width()) // 2, y))
                y += 44
            pygame.display.flip()
            self.clock.tick(30)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # `pillars` was the default until the layout library was regenerated and it
    # stopped existing, which made both scripts fail on a bare invocation.
    p.add_argument("--layout", default="divide", help="name in layout/layouts/")
    p.add_argument("--list", action="store_true", help="print the layouts and exit")
    p.add_argument("--list-robots", action="store_true",
                   help="print the robot methods and exit")
    p.add_argument("--seat", choices=["human", "robot"], default="human",
                   help="which seat you play (also choosable on the setup screen)")
    p.add_argument("--fov", type=int, default=90, choices=FOV_CHOICES,
                   help="the human seat's cone, in degrees")
    # --robot is watch.py's name for the same thing; accept both so a command
    # line copies straight between the two scripts.
    p.add_argument("--partner", "--robot", default="handoff",
                   help="robot policy when you play the human: %s"
                        % ", ".join(METHOD_KEYS))
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--forget", type=int, default=FORGET_HORIZON,
                   help="ticks before a station's contents fade to '?'")
    p.add_argument("--no-look-for", action="store_true",
                   help="scripted human: turn OFF 'go and find what I last saw "
                        "the robot carrying'. This is the CONTROL -- with it the "
                        "agent reproduces the plain ladder human action for "
                        "action, so any difference you see is the new sub-task")
    p.add_argument("--no-memory", action="store_true",
                   help="human seat: no memory at all, everything outside the "
                        "cone is black")
    p.add_argument("--no-infer", action="store_true",
                   help="robot seat: skip the FOV filter readout under TAB")
    p.add_argument("--scale", type=int, default=0, help="px per tile (0 = fit screen)")
    p.add_argument("--grill-fps", type=float, default=2.0,
                   help="how fast the grill cooks in real time, ticks/s "
                        "(default 2). It is the only station on a clock; the "
                        "board and sink only move under an INTERACT. 0 = the "
                        "grill waits for you too")
    p.add_argument("--fps", type=float, default=0.0,
                   help="put the WHOLE world on a clock at this rate -- your "
                        "teammate walks while you idle, and your keys are rate "
                        "limited to it. 0 (default) = only the grill self-ticks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human-index", type=int, default=1, choices=[0, 1],
                   help="which player index is the limited-vision seat; 1 (blue "
                        "hat) everywhere else in this package")
    p.add_argument("--top-k", type=int, default=3,
                   help="legacy -- does not reach fov-*, see the flag list above")
    p.add_argument("--depth", type=int, default=40,
                   help="legacy -- does not reach fov-*, see the flag list above")
    p.add_argument("--subtask-beta", type=float, default=None,
                   help="stochastic methods: sharpness of pi ~ value**beta "
                        "(default 8; inf = argmax, 1 = bayes's prior, 0 = uniform)")
    p.add_argument("--subtask-rho", type=float, default=None,
                   help="stochastic methods: probability of holding the current "
                        "sub-task each tick (default 0.95; 0 re-draws every tick, "
                        "1 commits until done). Does not change pi")
    p.add_argument("--skip-setup", action="store_true",
                   help="straight into the game with the flags given here")
    p.add_argument("--autopilot", action="store_true",
                   help="let LimitedVisionHuman drive YOUR seat too. Pair it "
                        "with --fps to pace it, or watch.py to watch properly")
    p.add_argument("--max-ticks", type=int, default=0, help="stop after N ticks")
    p.add_argument("--log", default=None, help="write the trajectory to this json")
    args = p.parse_args(argv)
    if args.list:
        print("\n".join(layout_names()))
        raise SystemExit(0)
    if args.list_robots:
        print(listing())
        raise SystemExit(0)
    if args.layout not in layout_names():
        raise SystemExit("no layout %r. Available:\n  %s"
                         % (args.layout, "\n  ".join(layout_names())))
    # canonicalise, so the setup screen's index-into-METHOD_KEYS and every
    # equality test downstream see one spelling per method.
    key = resolve(args.partner)
    if key is None:
        raise SystemExit("no partner %r. Available:\n%s"
                         % (args.partner, listing()))
    args.partner = key
    return args


if __name__ == "__main__":
    App(parse_args()).run()
