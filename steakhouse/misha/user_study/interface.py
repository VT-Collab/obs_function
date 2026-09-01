"""Playable SUBTASK-ONLY interface for the steakhouse user study.

    python -m user_study.interface --layout back_bar --partner fov
    python -m user_study.interface --list                the layouts you can name
    python -m user_study.interface --list-robots          every teammate policy

SAME ENVIRONMENT, SAME SETUP, SAME FILTER AS play.py -- see that file's header
for what --layout, --partner, --fov, --forget, --subtask-beta/rho etc. mean,
for the recipe, and for the registry of teammate policies (robot/methods.py).
This script only ever seats you as the HUMAN; the teammate is whichever
policy --partner names, exactly as in play.py's human-seat branch, cone and
all.

THE ONE DIFFERENCE THIS FILE EXISTS FOR: you never press an arrow key or
SPACE. You click a SUBTASK -- "get meat", "chop the onion", "deliver the
dish" -- from a menu, and your character walks there and interacts on its
own, one subtask at a time, exactly the way human/limited_vision_human.py
already walks and interacts every tick. What is different is entirely
upstream of that walk: WHICH subtask happens next is your click instead of
the ladder's own argmax.

WHY A MENU ROW IS GRAYED OUT, EXACTLY. A row is clickable only when it is,
right now, one of common/tasks.py's legal_subtasks() run against the ACTUAL
HUMAN MODEL's OWN belief -- human/limited_vision_human.py's BeliefView,
`_core.view`, with `optimistic` forced off so a station never observed, or
observed too long ago, is never offered as a guess (BeliefView._maybe()
treats both as the identical answer, UNKNOWN). Nothing here is simulated
twice or reimplemented: `_core.observe()` is the one and only thing that
ever writes this belief, called every tick straight off the real env state,
so the menu is exactly what LimitedVisionHuman itself has genuinely SEEN and
DONE over the episode so far -- not a separate approximation of it, and not
ground truth either: a dispenser you have not walked past is not
selectable, same as it is not walkable, same as it would not be nameable if
you tried to describe the kitchen out loud right now. See
user_study/human/human_behavior.py's legal_menu() for the one place this is
decided.

"LOOK FOR X" rows work the same way, off the SAME belief's sighting book:
`core.sightings` is written only when `state.players[robot].position` is
genuinely, geometrically inside `geo.visible_cells()` built from your own
real position, orientation and fov -- an exact fact about the geometry, not
an approximation of it. What it still cannot capture is human ATTENTION: a
robot crossing the very edge of a wide cone for a single tick counts as
"seen" by this geometric test even if a real participant's eyes were on the
menu at that instant, and closing that gap needs an actual model of
attention and noise. That model is user_study/human/approximate_human_model
.py -- but it is not the human YOU play; it is the one the robot's FOV
filter runs as a shadow, to infer your cone from your actions without ever
seeing your real clicks. See that file's docstring, not this one, for it.

THE FULL LABEL LIST is always shown regardless of what is currently legal,
because a study participant is briefed on the RECIPE in advance same as any
play.py player -- what they do not know in advance is the MAP, and that is
exactly what this belief gates. "LOOK FOR" rows are the one part of the menu
that come and go, because which items are even worth looking for changes
over the episode.

THE HUMAN MODEL is human/limited_vision_human.py's LimitedVisionHuman
itself, reused exactly as written, not reimplemented, for both the menu
(above) and the walk: once a row is picked, user_study/human/human_behavior
.py's HumanBehavior hands LimitedVisionHuman.action() that ONE (tier, verb,
cell) in place of the ladder's own top pick (see that file for how), so the
walking, turning, INTERACTing and belief bookkeeping are the unmodified
method doing exactly what it already does for the autopilot in play.py.

Controls
    click a subtask row   walk there and do it, one subtask at a time
    SPACE                  stop where you are (cancels the current subtask)
    TAB                    show/hide what your teammate is thinking
    p                       pause
    BACKSPACE              restart the episode
    ESC / q                 quit
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
HERE = os.path.dirname(os.path.abspath(__file__))
MISHA = os.path.dirname(HERE)
sys.path.insert(0, MISHA)

import pygame                                                     # noqa: E402
# `import play` (not `from play import ...` piecemeal for everything) sets
# overcooked_ai_py.LAYOUTS_DIR as a side effect of being imported, exactly the
# way play.py itself needs it set -- see play.py's own module-level lines.
# Reusing its View/colours/RECIPE/layout_names keeps this screen's rendering
# and vocabulary identical to play.py's rather than a fork of it.
import play                                                       # noqa: E402
from play import (View, layout_names, wrap, RECIPE, FOV_CHOICES, S,   # noqa: E402
                  BG, FG, DIM, HOT, ACCENT, BLUE,
                  _action_name, _subtask_str)

from overcooked_ai_py.mdp.actions import Action                   # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv     # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402

from common.tasks import TIER_NAME                                # noqa: E402
from human.limited_vision_human import (FORGET_HORIZON,           # noqa: E402
                                        _PREFIX as LOOK_PREFIX)
from robot.methods import (DEFAULTS, Drivers, GROUPS, HOTKEYS,     # noqa: E402
                           METHOD_KEYS, METHODS, in_group, listing,
                           make_robot, resolve)
from user_study.human.human_behavior import HumanBehavior         # noqa: E402
from user_study.robot.fov_posterior import SubtaskFOVPosterior    # noqa: E402
from user_study.robot.filter import FOVFilter                     # noqa: E402

# --partner fov/fov-greedy/fov-solo/fov-bayes: built from user_study's OWN
# FOVFilter (user_study/robot/filter.py) and SubtaskFOVPosterior, not
# robot/methods.py's registry entry (robot/filter/core/my_fov_filter.py --
# read-only, outside user_study/, and its currently-uncommitted local edit
# flips weight_seen/seen_bonus negative, which SCORES being seen as bad
# instead of good -- confirmed live: the robot dropped a carried item and
# left instead of lingering near the human). See Episode.reset() for where
# this is used. Deliberately only these four -- the "clean" baseline
# pairings every other fov-* row in the registry is a research SWEEP of
# (cap, fov_decay, frozen_fov, ...) that user_study/robot/filter.py has no
# equivalent knobs for; rerouting those would either silently collapse them
# all to the same behaviour as plain "fov" or require inventing matching
# knobs nobody asked for, so they still play through the original registry
# entry, unchanged.
_FOV_BASELINE = {"fov": "handoff", "fov-greedy": "greedy",
                 "fov-solo": "solo", "fov-bayes": "bayes"}

HUMAN_STUDY_ENV_HORIZON = 400

# (verb, participant-facing label). A FIXED display order, not a priority
# order -- the live tier (read off legal_subtasks() every tick, via
# common.tasks.TIER_NAME) is what the ladder would actually have ranked by;
# this order just keeps every row in the same place every tick so a
# participant's eyes do not have to re-hunt for "chop" every time the menu
# reflows. "explore" is not a real ladder verb -- legal_subtasks() never
# returns it, T_EXPLORE is the ladder's OWN empty-handed fallback -- and is
# appended as the one row that is always clickable, because looking around is
# always something a participant is allowed to choose to do.
VERB_ROWS = [
    ("deliver",           "carry the dish to the serving hatch"),
    ("take_dish",         "pick up a finished dish"),
    ("add_garnish",       "add a garnish to the plated steak"),
    ("plate_steak",       "plate the steak"),
    ("combine",           "combine what you're holding with a counter item"),
    ("take_steak_dish",   "pick up a plated steak"),
    ("take_garnish_dish", "pick up a garnished plate"),
    ("collect_plate",     "collect a washed plate from the sink"),
    ("collect_garnish",   "collect chopped garnish from the board"),
    ("take_washed_plate", "pick up a washed plate from a counter"),
    ("take_garnish",      "pick up garnish from a counter"),
    ("chop",              "chop the onion on the board"),
    ("wash",              "wash the plate in the sink"),
    ("load_pot",          "put the meat on the grill"),
    ("load_board",        "put the onion on the board"),
    ("load_sink",         "put the plate in the sink"),
    ("take_meat",         "pick up meat from a counter"),
    ("take_onion",        "pick up an onion from a counter"),
    ("take_plate",        "pick up a plate from a counter"),
    ("get_meat",          "get meat from the dispenser"),
    ("get_onion",         "get an onion from the dispenser"),
    ("get_plate",         "get a plate from the dispenser"),
    ("stash",             "put down what you're holding on a counter"),
    ("explore",           "look around / walk somewhere new"),
]


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
class Episode:
    """The env, the human-seat View, the teammate, and the HUD's inputs.

    Deliberately not play.py's Episode: there is only ever one seat here
    (you are always the human), so there is no you_idx/mate_idx split, no
    robot-seat branch, and no autopilot -- the participant's clicks are the
    only source of action for this seat, same as a real study is meant to
    have no scripted stand-in driving the human.
    """

    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.human_idx = args.human_index
        self.robot_idx = 1 - self.human_idx

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

        self.human = HumanBehavior(self.mdp, cfg["fov"], self.human_idx,
                                   args.forget, args.seed,
                                   enable_look_for=not args.no_look_for)

        human_kw = {"forget_horizon": args.forget,
                   "enable_look_for": not args.no_look_for}
        fov_baseline = _FOV_BASELINE.get(cfg["partner"])
        if fov_baseline is not None:
            # See _FOV_BASELINE's own module-level comment for why this
            # branch exists at all. The wrapped BASELINE still comes from
            # the real registry -- built the identical way _fov() itself
            # builds it, METHODS[key].build(), so it can never drift from
            # what every other row in the table uses -- only the FILTER
            # LAYER on top is user_study/robot/filter.py's instead of
            # robot/filter/core/my_fov_filter.py's.
            method_cfg = dict(DEFAULTS, seed=args.seed, human_kw=human_kw)
            if args.subtask_beta is not None:
                method_cfg["beta"] = args.subtask_beta
            if args.subtask_rho is not None:
                method_cfg["rho"] = args.subtask_rho
            baseline, base_drv = METHODS[fov_baseline].build(
                self.mdp, self.robot_idx, self.human_idx, method_cfg)
            post = SubtaskFOVPosterior(self.mdp, human_index=self.human_idx,
                                       seed=args.seed, human_kw=human_kw)
            self.mate = FOVFilter(self.mdp, baseline, post,
                                  agent_index=self.robot_idx)
            self.post = Drivers(base_drv.members + [post], cone=post)
            # ONE posterior drives the robot AND the debug overlay below, so
            # what gets displayed can never disagree with what the robot is
            # actually doing -- unlike the make_robot() branch, where
            # debug_post is a second, independent instrument watching in
            # parallel (see its own comment there for why that's fine for
            # every OTHER teammate, which never touches user_study/robot/
            # fov_posterior.py at all).
            self.debug_post = None if args.no_debug else post
        else:
            self.mate, self.post = make_robot(cfg["partner"], self.mdp,
                                              self.robot_idx, self.human_idx,
                                              args.seed, args.top_k, args.depth,
                                              beta=args.subtask_beta,
                                              rho=args.subtask_rho,
                                              human_kw=human_kw)
            # DEBUG INSTRUMENT, independent of --partner: a
            # SubtaskFOVPosterior that watches your real actions every tick
            # purely to maintain its own belief about your cone and current
            # subtask, same as a live user_study/robot/filter.py teammate
            # would -- but it never touches your_action/mate_action, so it
            # changes nothing about how the episode actually plays. See
            # _draw_debug() for what gets shown and where.
            self.debug_post = (None if args.no_debug else
                               SubtaskFOVPosterior(self.mdp,
                                                   human_index=self.human_idx,
                                                   seed=args.seed,
                                                   human_kw=human_kw))
        self.mate_name = cfg["partner"]
        self.human.prime(self.env.state)

        self.events = ["click a subtask to begin"]
        self.reward, self.done, self.mate_info, self.trace = 0, False, {}, []
        self.view.observe(self.env.state, self.env.t)

    # -- the tick -------------------------------------------------------
    def tick(self):
        """One env step, sourced entirely from the participant's last click.
        Returns True when the subtask that drove it just finished."""
        state = self.env.state
        held_before = self.held(self.human_idx)

        your_action, human_info, finished = self.human.step(state)
        mate_action, self.mate_info = self.mate.action(state)
        if self.post is not None:
            self.post.update(state, your_action)
        if self.debug_post is not None:
            self.debug_post.update(state, your_action)

        joint = [None, None]
        joint[self.human_idx] = your_action
        joint[self.robot_idx] = mate_action
        nxt, rew, done, _ = self.env.step(tuple(joint))

        self.reward += rew
        self.done = done or self.mdp.is_terminal(nxt)
        if rew > 0:
            self._event("DELIVERED  +%d" % rew)
        elif self.held(self.human_idx) != held_before:
            self._event("you now hold %s" % (self.held(self.human_idx) or "nothing"))

        if self.args.log:
            self.trace.append({
                "t": self.env.t - 1,
                "you": _action_name(your_action),
                "you_subtask": _subtask_str(human_info.get("subtask")),
                "mate": _action_name(mate_action),
                "mate_subtask": _subtask_str(self.mate_info.get("subtask")),
                "fov_post": self.posterior_str(),
                "reward": rew,
            })
        self.view.observe(nxt, self.env.t)
        return finished

    def tick_grill(self):
        """Real-time grill cooking while nobody's ticket is running -- see
        play.py's THE CLOCK. Only called while the human has nothing chosen,
        so this and tick() never fire in the same moment and the grill is
        never advanced twice for one span of real time."""
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

    def debug_lines(self):
        """Two lines for the bottom-left DEBUG overlay: what subtask you are
        ACTUALLY doing (straight off HumanBehavior -- the ground truth, no
        inference involved), and what fov/subtask self.debug_post's
        highest-weight particle currently believes, straight off the same
        `.map_fov()`/`.best_shadow(fov).committed` this session's own
        faithfulness testing used. Returns [] if --no-debug."""
        if self.debug_post is None:
            return []
        you = ("you: %s" % _subtask_str(self.human.last_subtask)
              if self.human.chosen else "you: (nothing chosen)")
        fov = self.debug_post.map_fov()
        p = self.debug_post.p.get(fov, 0.0)
        best = self.debug_post.best_shadow(fov)
        guess = _subtask_str(best.last_subtask) if best.committed else "(exploring / nothing)"
        robot = "robot thinks: fov=%d (p=%.2f)  %s" % (fov, p, guess)
        return [you, robot]

    def summary(self):
        return {"layout": self.cfg["layout"], "player": self.args.player,
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


# ---------------------------------------------------------------------------
# screens
# ---------------------------------------------------------------------------
class App:
    HEAD, FOOT, PAD = 62, 96, 12
    MIN_W = 1080
    ROW_H = 20                        # one menu row
    MENU_W = 400                      # fixed right-hand panel width
    MENU_HEAD = 34                    # "SUBTASKS" title + hint line
    MAX_LOOK_ROWS = 4                 # window-sizing headroom for LOOK FOR rows
    SETUP_SIZE = (1000, 860)

    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.display.set_caption("steakhouse user study -- subtask control")
        self.screen = pygame.display.set_mode(self.SETUP_SIZE)
        self.font = pygame.font.SysFont("couriernew", 16)
        self.bold = pygame.font.SysFont("couriernew", 16, bold=True)
        self.small = pygame.font.SysFont("couriernew", 13)
        self.smallbold = pygame.font.SysFont("couriernew", 13, bold=True)
        self.big = pygame.font.SysFont("couriernew", 26, bold=True)
        self.clock = pygame.time.Clock()
        self.layouts = layout_names()
        self.cfg = {"fov": args.fov, "partner": args.partner,
                    "layout": args.layout}
        self._menu_rects = []             # [(pygame.Rect, verb)], set each draw

    def run(self):
        while True:
            if not self.args.skip_setup:
                self.setup_screen()
            ep = self.play()
            if ep is None:
                break
            ep.write_log()
            print("%s  player=%s fov=%d partner=%s  ->  %d/%d orders in %d ticks"
                  % (ep.cfg["layout"], self.args.player, ep.cfg["fov"],
                     ep.mate_name, ep.delivered, ep.orders_total, ep.env.t))
            if self.args.skip_setup or not self.end_screen(ep):
                break
        pygame.quit()

    # -- instructions ---------------------------------------------------
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
                if pygame.K_1 <= e.key <= pygame.K_5:
                    cfg["fov"] = FOV_CHOICES[e.key - pygame.K_1]
                elif e.key in (pygame.K_UP, pygame.K_DOWN):
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

        line("STEAKHOUSE USER STUDY -- subtask control", ACCENT, self.big, 42)
        for t in wrap(self.font,
                      "You always play the human seat, with the same limited "
                      "vision cone play.py's human seat has. The only "
                      "difference: you never move a cursor. You click a "
                      "SUBTASK from the menu and your character walks there "
                      "and does it. Rows you have not discovered, or cannot "
                      "do right now, are grayed out.", w - 60):
            line(t, FG)
        y += 6
        line("player id: %s" % self.args.player, DIM, self.small, 18, 50)
        y += 8

        line("VISION CONE", ACCENT, self.bold, 24)
        x = 50
        for i, f in enumerate(FOV_CHOICES):
            sel = cfg["fov"] == f
            img = (self.bold if sel else self.font).render(
                "[%d] %d deg" % (i + 1, f), True, HOT if sel else DIM)
            self.screen.blit(img, (x, y))
            x += img.get_width() + 18
        y += 32

        line("TEAMMATE POLICY  (UP/DOWN moves, or press a letter)",
             ACCENT, self.bold, 22)
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
        assert y <= self.screen.get_height(), (
            "setup screen overflows SETUP_SIZE by %dpx -- raise it"
            % (y - self.screen.get_height()))
        pygame.display.flip()

    # -- the game ---------------------------------------------------------
    def play(self):
        args = self.args
        ep = Episode(self.cfg, args)
        gw, gh = ep.mdp.width, ep.mdp.height
        info = pygame.display.Info()
        fit = min((info.current_w * 0.92 - 2 * self.PAD - self.MENU_W) / gw,
                  (info.current_h * 0.82 - self.HEAD - self.FOOT) / gh)
        scale = args.scale or max(14, min(S, int(fit)))
        pw, ph = gw * scale, gh * scale
        # +1 header row and headroom for a few simultaneous LOOK FOR rows --
        # how many is genuinely dynamic (the sighting book), so this is a
        # roomy fixed allowance rather than an exact count.
        menu_h = (self.MENU_HEAD + len(VERB_ROWS) * self.ROW_H
                 + (1 + self.MAX_LOOK_ROWS) * self.ROW_H)
        win_w = max(self.PAD * 3 + pw + self.MENU_W, self.MIN_W)
        win_h = self.HEAD + max(ph, menu_h) + self.FOOT
        self.screen = pygame.display.set_mode((win_w, win_h))

        show_mate, paused, dirty = False, False, True
        step_dt = 1.0 / args.step_rate if args.step_rate > 0 else 0.0
        grill_dt = (1.0 / args.grill_fps if args.grill_fps > 0 else 0.0)
        next_step = time.time() + step_dt
        next_grill = time.time() + grill_dt
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self._handle_click(ep, self._window_to_surface(e.pos))
                    dirty = True
                if e.type != pygame.KEYDOWN:
                    continue
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None
                if e.key == pygame.K_BACKSPACE:
                    ep.reset()
                    dirty = True
                elif e.key == pygame.K_SPACE:
                    ep.human.cancel()
                    dirty = True
                elif e.key == pygame.K_TAB:
                    show_mate, dirty = not show_mate, True
                elif e.key == pygame.K_p:
                    paused, dirty = not paused, True

            if paused or ep.done:
                next_step = time.time() + step_dt
                next_grill = time.time() + grill_dt
            elif ep.human.chosen is not None:
                if time.time() >= next_step:
                    ep.tick()
                    next_step = time.time() + step_dt
                    next_grill = time.time() + grill_dt   # see tick_grill's docstring
                    dirty = True
            else:
                next_step = time.time() + step_dt          # don't bank ticks while idle
                if grill_dt and time.time() >= next_grill:
                    ep.tick_grill()
                    next_grill = time.time() + grill_dt
                    dirty = True

            if dirty:
                self._draw_game(ep, scale, (pw, ph), show_mate, paused)
                dirty = False
            if ep.done:
                return ep
            if args.max_ticks and ep.env.t >= args.max_ticks:
                return ep
            self.clock.tick(60)

    def _window_to_surface(self, pos):
        """MOUSEBUTTONDOWN's `pos` is in the OS WINDOW's own coordinate
        space, which on a HiDPI/Retina display can be a different size from
        the pygame Surface we actually drew the menu into and built
        `_menu_rects` against -- a well-known SDL2 quirk, and this script's
        first use of mouse input, so nothing in play.py's keyboard-only
        interface would have hit it. Rescale by the ratio between the two so
        a click that visually lands on a row also collides with its Rect;
        on a non-scaled display the two sizes already match and this is the
        identity transform."""
        try:
            ww, wh = pygame.display.get_window_size()
        except Exception:
            return pos
        sw, sh = self.screen.get_size()
        if not ww or not wh or (ww, wh) == (sw, sh):
            return pos
        return (pos[0] * sw / ww, pos[1] * sh / wh)

    def _handle_click(self, ep, pos):
        for rect, verb in self._menu_rects:
            if rect.collidepoint(pos) and verb in ep.human.legal:
                _tier, _dist, cell = ep.human.legal[verb]
                ep.human.choose(verb, cell)
                return

    def _panel(self, ep):
        """The game-view surface _draw_game blits, before scaling. Split out
        so a debug subclass (see interface2.py) can swap in an omniscient
        view -- View.robot_seat(), full truth + a translucent tint outside
        the human's cone, same as watch.py's default panel -- without
        touching anything else _draw_game does, and without this class ever
        needing to know that override exists. Purely a render source: it has
        no bearing on ep.human.legal or any belief/inference machinery,
        which all read ep.human/ep.debug_post directly, never ep.view.
        """
        return ep.view.human_seat(ep.env.t)

    def _view_caption(self, ep):
        """The one line of HUD text describing what the game-view panel
        above is showing. Split out for the same reason as _panel -- a
        debug subclass overriding one should almost always override the
        other, since they describe the same surface."""
        return "fov %d -- black is what you cannot see" % ep.cfg["fov"]

    def _draw_game(self, ep, scale, size, show_mate, paused):
        self.screen.fill(BG)
        panel = self._panel(ep)
        self.screen.blit(pygame.transform.smoothscale(panel, size),
                         (self.PAD, self.HEAD))

        head = ("t %d/%d   orders %d/%d   holding: %s   teammate: %s"
                % (ep.env.t, self.args.horizon, ep.delivered, ep.orders_total,
                   ep.held(ep.human_idx) or "nothing", ep.mate_name))
        self.screen.blit(self.bold.render(head, True, BLUE), (self.PAD, 8))
        if ep.human.chosen:
            verb, cell = ep.human.chosen
            cur = "current: %s%s" % (verb, " @ %s" % (cell,) if cell else "")
        else:
            cur = "current: (nothing chosen -- click a subtask)"
        self.screen.blit(self.small.render(
            "%s   |   %s"
            % (self._view_caption(ep), "PAUSED" if paused else cur), True,
            HOT if paused else DIM), (self.PAD, 29))
        self.screen.blit(self.small.render(
            "click a subtask   SPACE stop   TAB teammate   p pause   "
            "BACKSPACE restart   ESC quit", True, DIM), (self.PAD, 44))

        self._draw_menu(ep, size)

        y = self.HEAD + size[1] + 6
        for text, colour in self._footer(ep, show_mate):
            self.screen.blit(self.small.render(text, True, colour), (self.PAD, y))
            y += 16

        self._draw_debug(ep, size)

        if ep.done:
            self._banner("EPISODE OVER -- %d/%d orders in %d ticks"
                         % (ep.delivered, ep.orders_total, ep.env.t))
        pygame.display.flip()

    def _draw_debug(self, ep, size):
        """Bottom-left corner of the GAME VIEW PANEL, not the window --
        that distinction is the fix for a real bug: the window's own height
        is `HEAD + max(ph, menu_h) + FOOT`, and `menu_h` (the subtask list,
        ~594px fixed) is NOT checked against the screen's actual height the
        way the game view's own `fit` calculation in play() is. On a screen
        short enough that HEAD+menu_h+FOOT exceeds it, the window gets
        created taller than the display, and anything pinned to the
        window's bottom edge -- this used to be -- falls off-screen along
        with it. The game view panel itself is always on-screen by
        construction, so anchoring here instead is what actually guarantees
        visibility rather than assuming the window fits. ep.debug_lines()
        is [] under --no-debug."""
        lines = ep.debug_lines()
        if not lines:
            return
        box_h = 16 * len(lines) + 6
        y0 = self.HEAD + size[1] - box_h
        box = pygame.Surface((min(420, size[0]), box_h))
        box.fill(BG)
        box.set_alpha(215)
        self.screen.blit(box, (self.PAD, y0))
        y = y0 + 3
        for text in lines:
            self.screen.blit(self.small.render("DEBUG  " + text, True, ACCENT),
                             (self.PAD + 4, y))
            y += 16

    def _draw_menu(self, ep, size):
        """The subtask checklist. One rect per row goes into self._menu_rects
        for _handle_click to hit-test against -- the mechanism steak_study.py
        used for its own checkbox panel, redone against this ladder's own
        verb vocabulary (common/tasks.py) instead of that script's."""
        x0 = self.PAD * 2 + size[0]
        y = self.HEAD
        self.screen.blit(self.bold.render("SUBTASKS", True, ACCENT), (x0, y))
        y += 20
        self.screen.blit(self.small.render(
            "grayed = not possible right now", True, DIM), (x0, y))
        y += self.MENU_HEAD - 20

        self._menu_rects = []
        for verb, label in VERB_ROWS:
            legal = verb in ep.human.legal
            selected = ep.human.chosen is not None and ep.human.chosen[0] == verb
            rect = pygame.Rect(x0, y, self.MENU_W - self.PAD, self.ROW_H)
            if selected:
                pygame.draw.rect(self.screen, (60, 90, 60), rect)
            box = "[x]" if selected else "[ ]"
            suffix = ""
            if legal and verb != "explore":
                tier, dist, _cell = ep.human.legal[verb]
                suffix = "   %s, %d away" % (TIER_NAME.get(tier, "?"), dist)
            elif legal:
                suffix = "   ALWAYS AVAILABLE"
            colour = (HOT if selected else FG) if legal else DIM
            font = self.smallbold if (legal and not selected) or selected else self.small
            self.screen.blit(font.render("%s %s%s" % (box, label, suffix), True, colour),
                             (x0 + 4, y + 2))
            if legal:
                self._menu_rects.append((rect, verb))
            y += self.ROW_H

        # LOOK-FOR ROWS. Not part of the fixed VERB_ROWS vocabulary -- these
        # come and go over the episode as human_behavior.legal_menu() reads
        # the sighting book, one row per item the robot was last seen
        # carrying that is still worth going to check on. See that method
        # and the class docstring for why this is a click, not an autopick.
        look_verbs = sorted(v for v in ep.human.legal if v.startswith(LOOK_PREFIX))
        if look_verbs:
            y += 4
            self.screen.blit(self.small.render(
                "LOOK FOR  (robot was last seen carrying it)", True, ACCENT),
                (x0, y))
            y += self.ROW_H
            for verb in look_verbs:
                item = verb[len(LOOK_PREFIX):]
                selected = ep.human.chosen is not None and ep.human.chosen[0] == verb
                tier, dist, _cell = ep.human.legal[verb]
                rect = pygame.Rect(x0, y, self.MENU_W - self.PAD, self.ROW_H)
                if selected:
                    pygame.draw.rect(self.screen, (60, 90, 60), rect)
                box = "[x]" if selected else "[ ]"
                font = self.smallbold if selected else self.small
                self.screen.blit(font.render(
                    "%s look for %s   %s, %d away"
                    % (box, item, TIER_NAME.get(tier, "?"), dist), True,
                    HOT if selected else FG), (x0 + 4, y + 2))
                self._menu_rects.append((rect, verb))
                y += self.ROW_H

    def _footer(self, ep, show_mate):
        out = [(" | ".join(ep.events[-2:]), FG)]
        if not show_mate:
            out.append(("TAB shows what your teammate is doing", DIM))
            return out
        out.append(("teammate subtask: %s" % _subtask_str(ep.mate_info.get("subtask")),
                    HOT))
        if ep.mate_info.get("top3"):
            kind = ep.mate_info.get("top3_kind", "p")
            fmt = (lambda s: "%.2f" % s) if kind == "p" else (lambda s: "%.0ft" % s)
            out.append(("weighs %-5s  %s"
                        % ("p" if kind == "p" else "ticks",
                           "   ".join("%s%s %s" % (">" if take else " ", lab, fmt(sc))
                                      for lab, sc, take in ep.mate_info["top3"])),
                        ACCENT))
        post = ep.posterior()
        if post:
            top = max(post, key=post.get)
            bars = "  ".join("%d:%s%.2f" % (f, "*" if f == top else " ", p)
                             for f, p in sorted(post.items()))
            out.append(("filter's belief about YOUR cone   %s" % bars, HOT))
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

    # -- end ------------------------------------------------------------
    def end_screen(self, ep):
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
                ("all orders served!" if ep.delivered == ep.orders_total
                 else "time up.", ACCENT, self.big),
                ("%d/%d orders in %d ticks   (reward %d)"
                 % (ep.delivered, ep.orders_total, ep.env.t, ep.reward), FG, self.font),
                ("fov %d   teammate %s   layout %s   player %s"
                 % (ep.cfg["fov"], ep.mate_name, ep.cfg["layout"], self.args.player),
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
    p.add_argument("--layout", default="divide", help="name in layout/layouts/")
    p.add_argument("--list", action="store_true", help="print the layouts and exit")
    p.add_argument("--list-robots", action="store_true",
                   help="print the robot methods and exit")
    p.add_argument("--fov", type=int, default=90, choices=FOV_CHOICES,
                   help="your vision cone, in degrees")
    p.add_argument("--partner", "--robot", default="handoff",
                   help="teammate policy: %s" % ", ".join(METHOD_KEYS))
    p.add_argument("--horizon", type=int, default=HUMAN_STUDY_ENV_HORIZON)
    p.add_argument("--forget", type=int, default=FORGET_HORIZON,
                   help="ticks before a station's contents fade to '?'")
    p.add_argument("--no-look-for", action="store_true",
                   help="turn off 'look for X' rows for you (no sighting "
                        "book), and the TEAMMATE's shadow humans stop "
                        "forecasting that you would use them either, so the "
                        "two stay matched -- see the module docstring")
    p.add_argument("--no-memory", action="store_true",
                   help="no memory at all: everything outside the cone is black")
    p.add_argument("--no-debug", action="store_true",
                   help="turn off the bottom-left DEBUG overlay -- your "
                        "actual subtask vs. a SubtaskFOVPosterior's live "
                        "belief about your fov/subtask, run purely as an "
                        "instrument regardless of --partner. Off saves the "
                        "per-tick particle-filter cost if it is not needed")
    p.add_argument("--scale", type=int, default=0, help="px per tile (0 = fit screen)")
    p.add_argument("--grill-fps", type=float, default=2.0,
                   help="how fast the grill cooks in real time while you have "
                        "nothing chosen (default 2). Board and sink only move "
                        "under an INTERACT, same as play.py")
    p.add_argument("--step-rate", type=float, default=3.0,
                   help="how fast your character executes a chosen subtask, "
                        "in steps/second (default 3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human-index", type=int, default=1, choices=[0, 1])
    p.add_argument("--top-k", type=int, default=3, help="legacy -- see play.py")
    p.add_argument("--depth", type=int, default=40, help="legacy -- see play.py")
    p.add_argument("--subtask-beta", type=float, default=None,
                   help="stochastic teammates: sharpness of pi ~ value**beta")
    p.add_argument("--subtask-rho", type=float, default=None,
                   help="stochastic teammates: probability of holding the "
                        "current sub-task each tick")
    p.add_argument("--player", default="anon", help="participant id, for the log")
    p.add_argument("--skip-setup", action="store_true",
                   help="straight into the game with the flags given here")
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
    key = resolve(args.partner)
    if key is None:
        raise SystemExit("no partner %r. Available:\n%s"
                         % (args.partner, listing()))
    args.partner = key
    return args


if __name__ == "__main__":
    App(parse_args()).run()
