"""WATCH the two agents play each other, with the human's blind area greyed out.

    python watch.py --layout divide --fov 30 --robot handoff --fps 4
    python watch.py --list                       the layouts you can name

Nobody is at the keyboard here: LimitedVisionHuman takes the human seat and one
of the robot policies takes the other. You see the FULL kitchen -- the grey is
not hiding anything from YOU, it marks what the HUMAN cannot currently see, so
the thing to watch is the gap: the robot puts a washed plate on a counter, the
counter is grey, and the human walks past it for thirty ticks because as far as
it is concerned nothing is there.

The same policies play.py offers, out of the one registry in robot/methods.py --
`python watch.py --list-robots` prints the table. Everything above the qmdp block
is a theta-blind control, so whatever an FOV-aware policy beats them by is
attributable to the cone:

    --robot greedy       nearest job first. No partner model at all.
    --robot solo         the same, plus it gives up a station in the dividing
                         wall when the human would reach it first
    --robot handoff      solo plus staging, choosing the counter nearest ITSELF
    --robot bayes        infers the human's SUBTASK from their actions and takes
                         another one. Its guess is printed in the HUD.
    --robot bayes-noip   bayes with inverse planning off: the belief is the prior
                         every tick. The control for the control.
    --robot qmdp         infers the human's CONE from their actions; the
                         posterior is printed in the HUD and moves while you
                         watch. Rolls out handoff's shortlist.
    --robot qmdp-greedy  the same filter over greedy's shortlist,
    --robot qmdp-solo    over solo's,
    --robot qmdp-bayes   over bayes's -- slow, bayes runs inside every rollout.
                         The filter only RE-RANKS what its baseline proposes, so
                         which baseline it wraps sets its ceiling.
    --robot qmdp-sparse  qmdp scored on deliveries alone, shaping off
    --robot qmdp-map     qmdp against the likeliest cone, not the whole posterior

Everything above picks the top of its own ranking. The last group DRAWS instead,
from the distribution that policy really induces -- see robot/nominal_policy/
subtask_dist.py. Given the sub-task the low-level action is a pure function for
all of them, so the sub-task choice is the only place variation can live:

    --robot greedy-stoch   greedy, sub-task drawn from its own value prior
    --robot solo-stoch     solo, likewise
    --robot handoff-stoch  handoff, likewise
    --robot bayes-prior    bayes with inverse planning OFF, drawn from its prior
    --robot bayes-post     bayes drawn from its TRUE posterior. The pair
                           bayes-prior / bayes-post differs in evidence alone,
                           so the gap between them is what inference buys.

Add --both to put the human's own view (black outside the cone, memory fading to
'?') beside the grey one -- the same tick drawn from both points of view.

Everything here runs on one clock: --fps ticks a second, both agents acting every
tick. That is the difference from play.py, where only the grill self-ticks
because a human at the keyboard should not be rate-limited.

Every flag
    --layout NAME     kitchen to run, from layout/layouts/ (default divide)
    --list            print the layout names and exit
    --list-robots     print the robot methods above, grouped, and exit
    --robot NAME      which policy takes the robot seat (default handoff). The
                      list is robot/methods.py and nothing else -- both scripts
                      read that one registry, so a name that works here works in
                      play.py --partner too
    --fov DEG         the human's cone: 30, 60, 90, 180 or 360 (default 30). This
                      is the TRUE theta; the qmdp-* robots do not get told it
    --both            also draw the human's own blacked-out view, side by side
    --fps N           ticks per second (default 4). Both agents act every tick
    --paused          start paused, so you can step in with '.'
    --hold            keep the window up after the episode ends
    --horizon N       episode length in ticks (default 400)
    --forget N        ticks before a station's contents fade to '?' in the
                      human's memory. It is the human's model, not the robot's
    --scale PX        pixels per tile; 0 (default) fits the screen
    --seed N          seeds the human and the robot (default 0)
    --human-index I   which player index wears the cone: 0 or 1 (default 1, the
                      blue hat, as everywhere else in this package)
    --top-k N         qmdp-*: how many of the baseline's subtasks get rolled out
                      (default 3). The filter can only re-rank what it sees, so
                      this is the width of its shortlist
    --depth N         qmdp-*: rollout horizon in ticks (default 40). With
                      --top-k it is the whole cost of the filter, and they
                      multiply
    --subtask-beta B  stochastic methods: sharpness of the sub-task
                      distribution, pi ~ value**B (default 8). B->inf is the
                      deterministic argmax, B=1 is bayes's own team-value prior,
                      B=0 is uniform over legal sub-tasks
    --subtask-rho R   stochastic methods: probability of HOLDING the current
                      sub-task each tick (default 0.95, ~20 ticks). R=0 re-draws
                      every tick, R=1 commits until the job is done. It does not
                      change the distribution -- the sticky kernel leaves pi
                      exactly stationary -- only how long a draw lasts
    --max-ticks N     stop after N ticks; 0 (default) means run to --horizon
    --save DIR        write one png per TICK to DIR, for ffmpeg

Controls
    SPACE     pause / resume
    .         advance exactly one tick (pauses first)
    + / -     faster / slower
    r         restart
    q / ESC   quit

play.py is the same world with you in one of the seats.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# play.py owns the sys.path shim, the layouts-dir override and the renderer;
# importing it keeps the two scripts drawing the same kitchen the same way.
from play import (View, _subtask_str, layout_names,                # noqa: E402
                  FOV_CHOICES, BG, FG, DIM, HOT, ACCENT, GREEN, BLUE, S)

import pygame                                                      # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv       # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,         # noqa: E402
                                        FORGET_HORIZON)
from robot.methods import (METHOD_KEYS, listing,                    # noqa: E402
                           make_robot, resolve)


class Watcher:
    # FOOT fits the tallest footer at 16px a line: human, robot, the drawn-from
    # distribution, the partner-intent guess and the cone posterior. Adding a
    # sixth line means raising this or it renders off the bottom edge.
    HEAD, FOOT, PAD, GAP = 62, 94, 12, 14

    def __init__(self, args):
        self.args = args
        self.human_idx = args.human_index
        self.robot_idx = 1 - self.human_idx
        self.mdp = SteakHouseGridworld.from_layout_name(args.layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=args.horizon,
                                          info_level=0)
        self.orders_total = len(self.env.state.order_list or [])

        pygame.init()
        pygame.display.set_caption("no_larping -- fov%d human vs %s robot on %s"
                                   % (args.fov, args.robot, args.layout))
        self.panels = ["robot", "human"] if args.both else ["robot"]
        gw, gh = self.mdp.width, self.mdp.height
        info = pygame.display.Info()
        n = len(self.panels)
        fit = min((info.current_w * 0.92 - self.GAP * (n - 1) - 2 * self.PAD) / (n * gw),
                  (info.current_h * 0.82 - self.HEAD - self.FOOT) / gh)
        self.scale = args.scale or max(14, min(S, int(fit)))
        self.pw, self.ph = gw * self.scale, gh * self.scale
        self.screen = pygame.display.set_mode(
            (max(self.pw * n + self.GAP * (n - 1) + 2 * self.PAD, 820),
             self.ph + self.HEAD + self.FOOT))
        self.font = pygame.font.SysFont("couriernew", 15, bold=True)
        self.small = pygame.font.SysFont("couriernew", 13)
        self.clock = pygame.time.Clock()

        self.view = View(self.mdp, self.human_idx, args.fov, args.forget)
        self.reset()

    def reset(self):
        a = self.args
        self.env.reset()
        self.view.reset()
        self.human = LimitedVisionHuman(self.mdp, a.fov, agent_index=self.human_idx,
                                        forget_horizon=a.forget, seed=a.seed)
        self.robot, self.post = make_robot(a.robot, self.mdp, self.robot_idx,
                                           self.human_idx, a.seed, a.top_k, a.depth,
                                           beta=a.subtask_beta, rho=a.subtask_rho)
        self.reward, self.done = 0, False
        self.h_info, self.r_info = {}, {}
        self.frame = 0
        self.view.observe(self.env.state, self.env.t)

    @property
    def delivered(self):
        return self.orders_total - len(self.env.state.order_list or [])

    # -- one tick -----------------------------------------------------------
    def tick(self):
        state = self.env.state
        r_act, self.r_info = self.robot.action(state)
        h_act, self.h_info = self.human.action(state)
        if self.post is not None:
            self.post.update(state, h_act)
        joint = [None, None]
        joint[self.robot_idx], joint[self.human_idx] = r_act, h_act
        nxt, rew, done, _ = self.env.step(tuple(joint))
        self.reward += rew
        self.done = done or self.mdp.is_terminal(nxt)
        self.view.observe(nxt, self.env.t)

    # -- drawing ------------------------------------------------------------
    def draw(self, paused, fps):
        self.screen.fill(BG)
        title = {"robot": "FULL STATE -- grey is what the HUMAN cannot see",
                 "human": "THE HUMAN'S OWN VIEW -- black is unseen, '?' is forgotten"}
        colour = {"robot": GREEN, "human": BLUE}
        surf = {"robot": lambda: self.view.robot_seat(),
                "human": lambda: self.view.human_seat(self.env.t)}
        for i, key in enumerate(self.panels):
            x = self.PAD + i * (self.pw + self.GAP)
            self.screen.blit(self.font.render(title[key], True, colour[key]),
                             (x, self.HEAD - 19))
            self.screen.blit(
                pygame.transform.smoothscale(surf[key](), (self.pw, self.ph)),
                (x, self.HEAD))

        head = ("t %d/%d   orders %d/%d   human fov %d (%s hat)   robot %s (%s hat)"
                % (self.env.t, self.args.horizon, self.delivered, self.orders_total,
                   self.args.fov, "blue" if self.human_idx == 1 else "green",
                   self.args.robot, "green" if self.human_idx == 1 else "blue"))
        self.screen.blit(self.font.render(head, True, FG), (self.PAD, 8))
        self.screen.blit(self.small.render(
            "%s at %.1f ticks/s    SPACE pause   . step   +/- speed   r restart"
            "   q quit" % ("PAUSED" if paused else "running", fps),
            True, HOT if paused else DIM), (self.PAD, 30))

        y = self.HEAD + self.ph + 6
        for text, col in self._footer():
            self.screen.blit(self.small.render(text, True, col), (self.PAD, y))
            y += 16
        pygame.display.flip()

    def _footer(self):
        def held(i):
            o = self.env.state.players[i].held_object
            return o.name if o else "-"
        out = [("human  %-12s %s" % (held(self.human_idx),
                                     _subtask_str(self.h_info.get("subtask"))), BLUE),
               ("robot  %-12s %s" % (held(self.robot_idx),
                                     _subtask_str(self.r_info.get("subtask"))), GREEN)]
        if self.r_info.get("subtask_dist"):
            # The distribution the sub-task was DRAWN from, and whether this tick
            # held the previous draw or took a new one. Until the stochastic
            # methods existed there was no such object to print.
            d = self.r_info["subtask_dist"]
            top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
            out.append(("robot draws from   %s   [%s]"
                        % ("  ".join("%s %s:%.2f" % (v, c, p)
                                     for (_, v, c), p in top),
                           self.r_info.get("subtask_redraw", "?")), ACCENT))
        if "partner_subtask" in self.r_info:
            # a robot that infers the human's INTENT rather than their cone
            out.append(("robot thinks the human's JOB is  %s   p=%.2f  H=%.2f bits"
                        % (_subtask_str(self.r_info["partner_subtask"]),
                           self.r_info.get("partner_p") or 0.0,
                           self.r_info.get("partner_entropy") or 0.0), HOT))
        post = dict(self.post.p) if self.post is not None else self.r_info.get("fov_post")
        if post:
            top = max(post, key=post.get)
            out.append(("robot's belief about the cone (true %d)   %s"
                        % (self.args.fov,
                           "  ".join("%d:%s%.2f" % (f, "*" if f == top else " ", p)
                                     for f, p in sorted(post.items()))), HOT))
        else:
            out.append(("%s is blind to the cone by construction (the --robot "
                        "qmdp-* methods infer it)" % self.args.robot, DIM))
        return out

    # -- loop ---------------------------------------------------------------
    def run(self):
        a = self.args
        fps, paused, step_once, running = a.fps, a.paused, False, True
        dirty = True
        if a.save:
            os.makedirs(a.save, exist_ok=True)
        next_tick = time.time()
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    dirty = True
                    if e.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_PERIOD:
                        paused, step_once = True, True
                    elif e.key == pygame.K_r:
                        self.reset()
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        fps = min(fps * 1.5, 60)
                    elif e.key == pygame.K_MINUS:
                        fps = max(fps / 1.5, 0.25)

            stepped = False
            if not self.done and (step_once or (not paused and time.time() >= next_tick)):
                self.tick()
                next_tick = time.time() + 1.0 / max(fps, 0.01)
                step_once, stepped, dirty = False, True, True

            if dirty:
                self.draw(paused, fps)
                dirty = False
                if a.save and (stepped or self.frame == 0):   # one png per TICK
                    pygame.image.save(self.screen,
                                      os.path.join(a.save, "f%04d.png" % self.frame))
                    self.frame += 1
            if self.done and not a.hold:
                break
            if a.max_ticks and self.env.t >= a.max_ticks:
                break
            self.clock.tick(60)

        print("%s  fov %d  robot %s  ->  %d/%d orders, reward %d, %d ticks"
              % (a.layout, a.fov, a.robot, self.delivered, self.orders_total,
                 self.reward, self.env.t))
        if a.save:
            print("wrote %d frames to %s\n  ffmpeg -framerate %d -i %s/f%%04d.png "
                  "-pix_fmt yuv420p out.mp4" % (self.frame, a.save, int(fps), a.save))
        pygame.quit()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", default="divide")
    p.add_argument("--list", action="store_true", help="print the layouts and exit")
    p.add_argument("--list-robots", action="store_true",
                   help="print the robot methods and exit")
    p.add_argument("--fov", type=int, default=30, choices=FOV_CHOICES)
    p.add_argument("--robot", default="handoff", help=", ".join(METHOD_KEYS))
    p.add_argument("--both", action="store_true",
                   help="also draw the human's own blacked-out view")
    p.add_argument("--fps", type=float, default=4.0, help="ticks per second")
    p.add_argument("--paused", action="store_true", help="start paused")
    p.add_argument("--hold", action="store_true",
                   help="keep the window up after the episode ends")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--forget", type=int, default=FORGET_HORIZON)
    p.add_argument("--scale", type=int, default=0, help="px per tile (0 = fit screen)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human-index", type=int, default=1, choices=[0, 1])
    p.add_argument("--top-k", type=int, default=3,
                   help="qmdp-*: subtasks rolled out. It and --depth are the two "
                        "cost knobs and they multiply")
    p.add_argument("--depth", type=int, default=40, help="qmdp-*: rollout horizon")
    p.add_argument("--subtask-beta", type=float, default=None,
                   help="stochastic methods: sharpness of pi ~ value**beta "
                        "(default 8; inf = argmax, 1 = bayes's prior, 0 = uniform)")
    p.add_argument("--subtask-rho", type=float, default=None,
                   help="stochastic methods: probability of holding the current "
                        "sub-task each tick (default 0.95; 0 re-draws every tick, "
                        "1 commits until done). Does not change pi")
    p.add_argument("--max-ticks", type=int, default=0)
    p.add_argument("--save", default=None, help="write every frame to this dir as png")
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
    # validated HERE rather than left to make_robot, so a typo fails before a
    # window opens and the message lists what you could have typed instead.
    key = resolve(args.robot)
    if key is None:
        raise SystemExit("no robot %r. Available:\n%s" % (args.robot, listing()))
    args.robot = key
    return args


if __name__ == "__main__":
    Watcher(parse_args()).run()
