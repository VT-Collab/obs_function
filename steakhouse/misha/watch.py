"""WATCH the two agents play each other, with the human's blind area greyed out.

    python watch.py --layout back_bar --fov 30 --robot fov --fps 4
    python watch.py --list                       the layouts you can name
    python watch.py --list-robots                every policy name

ONE FILTER LIVES HERE: `--robot fov` (and its `fov-*` baseline-paired rows).
It carries no task knowledge at all and scores a plan by how many of its ticks
the human would see the robot on. It shares the certainty gate, the cone
posterior and the baseline's own sub-task distribution with qmdp.py, its
predecessor -- deleted; robot/methods.py's module docstring and RESULTS.md
have what it did and why it's gone. Every `fov*` row below is this filter.

Nobody is at the keyboard here: the scripted human takes the human seat and one
of the robot policies takes the other. You see the FULL kitchen -- the grey is
not hiding anything from YOU, it marks what the HUMAN cannot currently see, so
the thing to watch is the gap: the robot puts a washed plate on a counter, the
counter is grey, and the human walks past it for thirty ticks because as far as
it is concerned nothing is there.

THE HUMAN LOOKS FOR THINGS BY DEFAULT. On top of the ordinary ladder it will go
and LOOK for whatever it last saw the robot carrying, if it has not seen that
item put down anywhere and the item outranks what it is doing -- dropping what it
holds to do so, unless it is on its way to deliver. `--no-look-for` turns that
off and reproduces the plain ladder human action for action, so it is a control
on one agent rather than a different agent -- literally, since it is a flag on
LimitedVisionHuman and not a subclass of it. Watch the sub-task line for
`look_for_*`.

THE ROBOT POLICIES ARE A BASELINE COLUMN AND ONE FILTER LAID OVER IT. That is
the whole list -- `--list-robots` prints it from the single registry in
robot/methods.py, which play.py reads too, so a name that works here works there.

  BASELINES. Theta-blind: none of them reads the cone, so whatever an FOV-aware
  policy beats them by is attributable to theta and nothing else. Each adds a
  little to the one above, so a difference between two of them has few causes.
  All of them DRAW their sub-task rather than taking the argmax -- see below for
  why that is not a detail.

    --robot greedy   nearest job first. No partner model at all.
    --robot solo     greedy + a contention penalty: any candidate CELL the
                           human's shortest path reaches sooner is demoted
                           WITHIN its tier, so it is ceded only when something
                           else in that tier exists. Not only stations -- it hits
                           stash targets too, and mostly does: it pushes drops
                           away from the counters the human is near, which on
                           these layouts are the only channel between the rooms.
                           Measured over 54 matched episodes it delivers FEWER
                           dishes than greedy (22 vs 35).
    --robot handoff  solo + staging, and it is two changes rather than
                           one. When holding a shareable item with nothing
                           better than a stash available it prepends every
                           reachable free counter as a target, ordered by
                           distance from ITSELF -- it cannot see what the human
                           can. It also biases the `stash` verb within its tier,
                           which does not reorder its own ranking but does shift
                           the distribution the fov layer reads.
    --robot bayes     + intent: infers WHICH JOB the human is doing from
                           their actions and takes a different one. Its guess is
                           in the HUD. Models their MIND, never their EYES.
    --robot bayes-noip    CONTROL: bayes with inverse planning OFF, so observed
                           actions are never folded in. Not frozen at the prior
                           -- the belief is still carried and re-projected onto
                           each tick's allocation set -- but EVIDENCE is the only
                           knob that differs, so bayes-noip against bayes
                           is what the inference buys.

  THE SAME FILTER OVER EACH. `fov-solo` is not a cousin of `fov`; it is the
  SAME class over a different floor. There is one row per baseline because the
  layer degrades to the baseline it wraps, which makes that baseline both its
  floor and the thing it must beat.

  All of them infer the human's CONE from their actions -- the posterior is in
  the HUD and moves while you watch -- then search over HOW to do their
  baseline's job: which legal target cell, which first step, whether to wait a
  tick. Watch the "exec" line in the footer. It reads "follows baseline" on the
  large majority of ticks; the few where it flips to DEVIATES are the entire
  mechanism.

  THE "robot weighs" LINE is the one to read alongside it: the three sub-tasks
  the robot was choosing between this tick, with `>` on the one it took. Every
  method prints it, so it is the same line for a baseline and for the filter --
  but the number is not the same quantity, and the tag says which. A baseline
  shows the PROBABILITY it drew from, so the marked row is frequently NOT the
  top one; that gap is the draw. The filter shows TICKS, lower better, so its
  marked row always IS the top one -- but they are different ticks and the cands
  table below says which: C = regret + detour - cap*s, ticks of BASELINE COST,
  and it can go negative because a plan that is seen a lot buys itself credit.
  Run the filter and its own baseline on the same seed and the interesting
  ticks are where the two orderings disagree -- that is the layer re-ranking
  its baseline's jobs.

  THE FoV-ONLY FILTER. It has no recipe, no orders-left, no handoff
  model and no tail; the cost is

      C = R(cell) + detour(action, cell) - bonus

      bonus = cap * SUM over sightings j  decay^(j-1) * gamma^(tick_j - 1)

  where R is the baseline's own sub-task regret in ticks, detour is the extra
  walking this first action costs, and the bonus is the only thing the filter
  contributes: HOW OFTEN AND HOW SOON, OVER THE TICKS THIS JOB TAKES, WOULD THE
  HUMAN SEE ME. A sum over the trajectory -- the tick a stash lands counts once,
  like every other tick, and is never asked about on its own, because "does this
  drop get witnessed" is the handoff question and the handoff is task knowledge.
  Read it in the cands table: `seen 3/7` is three of the seven ticks.

  TWO DIALS, ONE BUDGET. `cap` is what the FIRST sighting is worth, in TICKS of
  the baseline's own cost; `decay` is what each one after it is worth relative to
  the one before. So the most any plan can be worth is cap/(1-decay) -- 8 ticks at
  the defaults -- and that, not cap, is the number to compare against the 21.4
  ticks that make one rung of the sub-task ladder. At or below a rung of BUDGET
  the filter provably cannot move the robot to a different TIER of job; it only
  changes WHICH cell, and how it gets there. Run fov-base and fov-free either side
  of the sweep to see the extremes.

    --robot fov            over handoff, cap = 4, budget 8 ticks -- the headline
    --robot fov-greedy     over greedy
    --robot fov-solo       over solo
    --robot fov-bayes      over bayes
    --robot fov-base       PARITY CONTROL: cap = 0. It MUST play the baseline
                           tick for tick, and that is exact rather than
                           empirical -- it falls out of R + detour
                           being zero for the baseline's own action. If this row
                           ever deviates, the theorem is wrong, not the run.
    --robot fov-fixed      THETA-BLIND CONTROL: cone frozen at 90. fov against
                           fov-fixed is what inferring the cone bought.
    --robot fov-free       UNBOUNDED CONTROL: cap = 1e6, the FoV term overrides
                           the ladder entirely. This is what the filter looks like
                           with no budget, and it is the row that shows the cap is
                           doing something: it crosses tier rungs and the capped
                           rows do not.
    --robot fov-c1         the sweep, in ticks per SIGHTING: 1, 2, 8 and 10.7,
    --robot fov-c2         which are budgets of 2, 4, 16 and 21.4. 21.4 is exactly
    --robot fov-c8         one rung of the ladder, so fov-c21 is the largest
    --robot fov-c21        budget that still cannot change which TIER is done.
    --robot fov-d02        the decay sweep at a FIXED budget of 8 ticks: 0.2 puts
    --robot fov-d08        nearly the whole budget on being seen ONCE, 0.8 spreads
                           it over many sightings. Against `fov` at decay 0.5 this
                           varies the SHAPE of the spend with its size held still.

WHY EVERY BASELINE DRAWS. The filter consumes a DISTRIBUTION over
sub-tasks. A deterministic policy does not have one, so it could only be handed a
Boltzmann reconstruction of what a policy that never draws anything might have
drawn -- and then scored against a baseline that never held those preferences.
There is now one kind of baseline and both sides of every pair use it. Given the
sub-task the low-level action is a pure function for every policy here, so the
sub-task choice is the only place the variation lives. There is no deterministic
variant to pick instead -- there used to be, and removing it is what made the
comparison honest.

robot/filter/RESULTS.md has the measured numbers.

Add --both to put the human's own view (black outside the cone, memory fading to
'?') beside the grey one -- the same tick drawn from both points of view.

Everything here runs on one clock: --fps ticks a second, both agents acting every
tick. That is the difference from play.py, where only the grill self-ticks
because a human at the keyboard should not be rate-limited.

Every flag
    --layout NAME     kitchen to run, from layout/layouts/ (default divide)
    --list            print the layout names and exit
    --list-robots     print the robot methods above, grouped, and exit
    --robot NAME      which policy takes the robot seat (default handoff).
                      The list is robot/methods.py and nothing else -- both
                      scripts read that one registry, so a name that works here
                      works in play.py --partner too
    --fov DEG         the human's cone: 30, 60, 90, 180 or 360 (default 30). This
                      is the TRUE theta; the fov-* robots do not get told it
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
    --top-k N         legacy, and it NO LONGER REACHES THE fov-* METHODS. It is
                      threaded into the build config, but FOVFilter does not
                      read it -- the shortlist width is `m` and the rollout
                      budget is `depth`, both set in robot/methods.py where the
                      row is registered. Left in place because the sweep tooling
                      still passes it
    --depth N         same: legacy, unread by the filter. Its head horizon is
                      `T`, and the head normally stops earlier anyway, at the
                      first INTERACT. See robot/filter/DESIGN.md section 5b for
                      the knobs that are real
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
# The look-for is ON by default, and --no-look-for is a control on the SAME
# agent rather than a second agent: enable_look_for=False reproduces the base
# ladder human action for action. One class is what makes the shadows the filter
# builds necessarily the same agent as the one in the seat.
from human.limited_vision_human import (LimitedVisionHuman,         # noqa: E402
                                        FORGET_HORIZON)
from robot.methods import (METHOD_KEYS, listing,                    # noqa: E402
                           make_robot, resolve)


class Watcher:
    # FOOT fits the tallest footer at 16px a line: human, robot, the top-3
    # sub-tasks it was weighing, the execution-layer line, the partner-intent
    # guess and the cone posterior. Adding a seventh line means raising this or
    # it renders off the bottom edge. WIDTH is the other constraint and is easier
    # to miss, because an over-long line is simply cut off at the window edge
    # with nothing to say so: baselines.short_subtask exists to keep the
    # top-3 line inside it, and has the measured numbers.
    HEAD, FOOT, PAD, GAP = 62, 110, 12, 14
    # FOOT is the DEFAULT only. The candidate table adds a line per counter, so
    # the real footer height is computed in __init__ before the window is sized;
    # leaving it constant is what cut the bottom lines off.

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
        # Read the HUD knobs BEFORE sizing the window: the footer's height depends
        # on how many candidate rows it will draw.
        self.debug = getattr(args, "debug", False)
        self.debug_n = getattr(args, "debug_n", 12)
        self.hud_cands = getattr(args, "hud_cands", 6)
        # base lines + the table's header, its rows, and its "... n more" line
        self.FOOT = self.FOOT + 16 * (self.hud_cands + 2 if self.hud_cands else 0)
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
        # For the numbers drawn INSIDE tiles. Scaled to the tile so it still
        # fits when --scale is small.
        self.tiny = pygame.font.SysFont("couriernew",
                                       max(9, min(13, self.scale // 2)), bold=True)
        self.clock = pygame.time.Clock()

        self.view = View(self.mdp, self.human_idx, args.fov, args.forget)
        # (self.debug / debug_n / hud_cands are set above, before window sizing)
        self.reset()

    def reset(self):
        a = self.args
        self.env.reset()
        self.view.reset()
        self.human = LimitedVisionHuman(self.mdp, a.fov,
                                        agent_index=self.human_idx,
                                        forget_horizon=a.forget, seed=a.seed,
                                        enable_look_for=not a.no_look_for)
        # THE FILTER MUST SIMULATE THE HUMAN THAT IS ACTUALLY PLAYING. Its cone
        # posterior builds shadow humans and its forecast rolls one of them
        # forward; a shadow built differently from the seat predicts a cone
        # belonging to nobody. There is only one human class, so the class can no
        # longer be got wrong -- but the FLAGS can, so pass the same ones used
        # for self.human above.
        self.robot, self.post = make_robot(a.robot, self.mdp, self.robot_idx,
                                           self.human_idx, a.seed, a.top_k, a.depth,
                                           beta=a.subtask_beta, rho=a.subtask_rho,
                                           human_kw={"forget_horizon": a.forget,
                                                     "enable_look_for":
                                                         not a.no_look_for})
        self.reward, self.done = 0, False
        self.h_info, self.r_info = {}, {}
        self.frame = 0
        self.view.observe(self.env.state, self.env.t)

    @property
    def delivered(self):
        return self.orders_total - len(self.env.state.order_list or [])

    # -- debug ---------------------------------------------------------------
    def debug_dump(self, r_act):
        """Print what the filter was choosing between, this tick, to stdout.

        Three blocks, because three different things go wrong and they look the
        same on screen:

          JOBS   the top-m the BASELINE handed over, and its own realised pick. If
                 the counter you expected is not owned by any of these jobs it was
                 never in the mask and no scoring detail will explain its absence.
          CANDS  every (counter, press tick) that was scored, with C and the tick
                 the human's cone first covers it. Sorted by C, so the top line is
                 what the filter believes and the rest is what it turned down.
          PICK   what it committed to and emitted.

        `sight` is the number that matters for a stash: a counter the partner never
        looks at is charged `blind` instead, and two counters with the same C but
        different sight are NOT the same choice.
        """
        i = self.r_info
        if not i or i.get("gated"):
            print("t%-4d GATED  p(MAP)=%.2f  -- below certainty, baseline acted"
                  % (self.env.t, i.get("p_map", 0.0)))
            return
        me = self.env.state.players[self.robot_idx]
        held = me.held_object.name if me.held_object else "-"
        print("\nt%-4d pos%s held=%-12s act=%-8s  base=%s"
              % (self.env.t, tuple(me.position), held, str(r_act),
                 _subtask_str(i.get("base_subtask"))))
        jobs = i.get("jobs") or []
        print("     JOBS (top %d from the baseline): %s"
              % (len(jobs), "  ".join("%s/%s" % j for j in jobs) or "-"))
        cands = i.get("cands") or []
        if not cands:
            # THREE different silences and they are worth telling apart: no_reach
            # is geometry (nothing the baseline offered can be walked to);
            # pi_no_ref and stale_pick are the FoV filter standing down because pi
            # gave the baseline's own realised draw no mass, so there was no
            # reference to price a deviation against. Printing them all as a bare
            # "none" is how a layer that is doing nothing for a whole stretch looks
            # identical to one that is boxed in.
            #
            # The KEYS ARE READ FROM THE FILTER, not guessed. This branch spent its
            # first outing testing `pi_blind`, which no filter has ever set -- so
            # all 50 stand-down ticks in a bayes episode printed a bare "CANDS
            # none", which is the exact failure the paragraph above says it
            # prevents. If you rename a key in fov_filter.py, rename it here.
            why = ("  (no_reach)" if i.get("no_reach") else
                   "  (pi_no_ref -- pi does not name the baseline's own draw)"
                   if i.get("pi_no_ref") else
                   "  (stale_pick -- the baseline's draw is not in this tick's pi)"
                   if i.get("stale_pick") else "")
            print("     CANDS none%s" % why)
        else:
            # ONE ROW PER COUNTER, not one per (counter, tick, first action). The
            # search scores a plan per first action, so the raw table repeats the
            # same counter five times with the same C and that buries the thing you
            # are looking for -- which counter, and why. Rows collapse to the best
            # plan for each counter, with `plans` saying how many were scored and
            # `press@` the tick of the best one.
            per = {}
            for verb, cell, t_end, c, k, pk, av, tl in cands:
                cur = per.get(cell)
                if cur is None or c < cur[2]:
                    per[cell] = [verb, t_end, c, k, 0, pk, tl]
                per[cell][4] += 1
            rows = sorted(per.items(), key=lambda kv: kv[1][2])
            pick_cell = i["plan"][1] if i.get("plan") else None
            print("     CANDS %d counters (%d plans)   mask=%d   states=%d"
                  % (len(rows), len(cands), i.get("n_mask", 0),
                     i.get("n_states", 0)))
            print("       %-10s %-9s %-7s %-7s %-6s %-9s %s"
                  % ("verb", "counter", "press@", "C", "plans", "sees at",
                     "PICKS IT UP AT"))
            for cell, (verb, t_end, c, k, n, pk, tl) in rows[:self.debug_n]:
                print("       %-10s %-9s t+%-5d %-7.1f %-6d %-9s %s%s"
                      % (verb, str(cell), t_end, c, n,
                         ("t+%d" % k) if k is not None else "never",
                         ("t+%d" % pk) if pk is not None
                         else ("no: never looked (+%.0f)"
                               % getattr(self.robot, "no_handoff", 0.0) if k is None
                               else "no: cannot reach (+%.0f)"
                               % getattr(self.robot, "no_handoff", 0.0)),
                         "  <-- PICKED" if cell == pick_cell else ""))
            if len(rows) > self.debug_n:
                worst = rows[-1][1][2]
                print("       ... %d more counters, C up to %.1f"
                      % (len(rows) - self.debug_n, worst))
        print("     PICK  %s @ t+%s   C=%.1f  gain %+.1f%s"
              % (_subtask_str(i.get("subtask")), i.get("plan_t", "?"),
                 i.get("Q", float("nan")), i.get("gain", 0.0),
                 "  [held plan]" if i.get("held_plan") else ""))

    # -- one tick -----------------------------------------------------------
    def tick(self):
        state = self.env.state
        r_act, self.r_info = self.robot.action(state)
        if self.debug:
            self.debug_dump(r_act)
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

        self._draw_cands()

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

    def _draw_cands(self):
        """Ring every scored counter on the grid and write its C inside it.

        Coordinates in the footer are unreadable -- you cannot tell where (4, 6) is
        without counting tiles -- so the ranking is drawn where the counters
        actually are. The PICKED counter gets a thick ring, the runners-up thin
        ones, and the number in each is C in ticks, lower better. `sees` is not
        drawn per tile because there is no room; the footer table carries it.

        Only on the "robot" panel: it is the full-state seat, so a counter the human
        cannot see is still visible to YOU there, which is the whole point of
        comparing the two panels.
        """
        cands = self.r_info.get("cands")
        if not cands or "robot" not in self.panels:
            return
        best = {}
        for _verb, cell, t_end, c, k, pk, av, tl in cands:
            if cell not in best or c < best[cell][0]:
                best[cell] = (c, t_end, k, pk)
        rows = sorted(best.items(), key=lambda kv: kv[1][0])[:max(self.hud_cands, 1)]
        pick = self.r_info["plan"][1] if self.r_info.get("plan") else None
        x0 = self.PAD + self.panels.index("robot") * (self.pw + self.GAP)
        for rank, (cell, (c, _t, _k, _pk)) in enumerate(rows):
            cx, cy = cell
            r = pygame.Rect(x0 + cx * self.scale, self.HEAD + cy * self.scale,
                            self.scale, self.scale)
            taken = cell == pick
            # Rank shades from the pick's colour down to DIM, so "second best" reads
            # as second best without needing the number.
            f = rank / float(max(len(rows) - 1, 1))
            col = ACCENT if taken else tuple(
                int(a + (b - a) * f) for a, b in zip(GREEN, DIM))
            pygame.draw.rect(self.screen, col, r, 3 if taken else 1)
            lab = self.tiny.render("%d" % round(c), True, col)
            self.screen.blit(lab, (r.x + 2, r.y + max(0, self.scale - 12)))

    def _footer(self):
        def held(i):
            o = self.env.state.players[i].held_object
            return o.name if o else "-"
        out = [("human  %-12s %s" % (held(self.human_idx),
                                     _subtask_str(self.h_info.get("subtask"))), BLUE),
               ("robot  %-12s %s" % (held(self.robot_idx),
                                     _subtask_str(self.r_info.get("subtask"))), GREEN)]
        if self.r_info.get("top3"):
            # THE THREE SUB-TASKS THE ROBOT WAS CHOOSING BETWEEN, with the one it
            # took marked `>`. Every method emits this, baseline and filter alike,
            # so the line reads the same way across the whole method table -- but
            # the NUMBER is not the same quantity in the two cases, and the label
            # says which: a baseline shows the PROBABILITY it drew from, the
            # filter shows a cost in ticks, where lower wins.
            #
            # On a baseline the marked row is often not the top one -- that is
            # the draw, and it is the quickest way to see a sampled policy behave
            # unlike a greedy one. On the filter the marked row IS the cheapest,
            # and the thing to watch for is the two orderings disagreeing: that
            # is the tick where the layer re-ranked its baseline's jobs.
            kind = self.r_info.get("top3_kind", "p")
            fmt = (lambda s: "%.2f" % s) if kind == "p" else (lambda s: "%.0ft" % s)
            out.append(("robot weighs %-5s  %s%s"
                        % ("p" if kind == "p" else "ticks",
                           "   ".join("%s%s %s" % (">" if take else " ", lab, fmt(sc))
                                      for lab, sc, take in self.r_info["top3"]),
                           ("   [%s]" % self.r_info["subtask_redraw"]
                            if self.r_info.get("subtask_redraw") else "")), ACCENT))
        # THE CANDIDATE TABLE, ON SCREEN. `robot weighs` above collapses the whole
        # search to one number per JOB; this is the level below it -- one line per
        # COUNTER.
        #
        # THE FoV FILTER'S OWN QUESTION IN EIGHT SLOTS, and printing under the
        # wrong headings would be worse than printing nothing -- `cands_kind`
        # is what the non-fov branch below used to guard against, back when
        # qmdp.py's search asked a question about the COUNTER instead (when
        # does the partner's cone reach it, when do they have the item,
        # because a stash's value to qmdp WAS the handoff). The FoV filter has
        # no such notion: it carries no task knowledge, so it cannot and must
        # not claim to know when a drop gets collected. Its slots 4/5 are the
        # first and last tick the partner's cone is on THE ROBOT, and slot 7
        # is the FoV half of the cost, -cap*s, not a tail.
        cands = self.r_info.get("cands")
        if cands:
            fov_kind = self.r_info.get("cands_kind") == "fov"
            per = {}
            for verb, cell, t_end, c, k, pk, av, tl in cands:
                cur = per.get(cell)
                if cur is None or c < cur[1]:
                    per[cell] = (verb, c, t_end, k, pk, tl)
            rows = sorted(per.items(), key=lambda kv: kv[1][1])
            pick = self.r_info["plan"][1] if self.r_info.get("plan") else None
            out.append(("cands  %d counters, best first   press@ you press,  " % len(rows)
                        + ("seen  ticks their cone is ON YOU / ticks the job takes,"
                           "  fov  what that bought"
                           if fov_kind else
                           "sees your cone reaches it,  GOT you hold it"), DIM))
            for cell, (verb, c, t_end, k, pk, tl) in rows[:self.hud_cands]:
                taken = cell == pick
                if fov_kind:
                    # k is the COUNT of ticks seen and pk the length of the job, so
                    # this reads "seen 3/7". It is deliberately not a t+ range: the
                    # score is a count over a trajectory and a range would suggest
                    # the filter cares WHEN, which is the moment-shaped question it
                    # was built not to ask.
                    line = ("  %s %-9s %-10s press@t+%-3d C %-7.1f =fov %-7.1f seen %s"
                            % (">" if taken else " ", str(cell), verb, t_end, c, tl,
                               "%d/%d" % (k, pk) if pk else "-"))
                else:
                    line = ("  %s %-9s %-10s press@t+%-3d C %-7.1f =tail %-6.1f sees %-7s GOT %s"
                            % (">" if taken else " ", str(cell), verb, t_end, c, tl,
                               ("t+%d" % k) if k is not None else "never",
                               ("t+%d" % pk) if pk is not None
                               else ("no -- never looked" if k is None
                                     else "no -- cannot reach")))
                out.append((line, ACCENT if taken else DIM))
            if len(rows) > self.hud_cands:
                out.append(("    ... %d more, up to C %.1f"
                            % (len(rows) - self.hud_cands, rows[-1][1][1]), DIM))
        if "theta_entropy" in self.r_info:
            # THE EXECUTION LAYER, and the thing worth watching is the DEVIATED
            # flag: on the vast majority of ticks this line reads "follows
            # baseline" and the robot is literally its nominal policy. When it
            # flips, the rollouts have found a placement whose payoff beats the
            # baseline's own step by more than eps -- that tick is the whole
            # mechanism, and without this line it is invisible on screen.
            dev = self.r_info.get("deviated")
            plan = self.r_info.get("plan")
            head = "DEVIATES" if dev else "follows baseline"
            bits = ["exec   %-16s" % head]
            if plan:
                bits.append("plan %s@%s" % (plan[0], plan[1]))
            if dev and self.r_info.get("gain") is not None:
                bits.append("gain %+.2f" % self.r_info["gain"])
            bits.append("H(theta) %.2f over %d cone%s"
                        % (self.r_info["theta_entropy"],
                           self.r_info.get("n_cones", 1),
                           "" if self.r_info.get("n_cones", 1) == 1 else "s"))
            if self.r_info.get("channel") is not None:
                bits.append("channel %s" % ("open" if self.r_info["channel"] else "shut"))
            out.append(("   ".join(bits), HOT if dev else DIM))
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
                        "fov-* methods infer it)" % self.args.robot, DIM))
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
    p.add_argument("--no-look-for", action="store_true",
                   help="scripted human: turn OFF 'go and find what I last saw "
                        "the robot carrying'. The CONTROL -- reproduces the plain "
                        "ladder human action for action")
    p.add_argument("--scale", type=int, default=0, help="px per tile (0 = fit screen)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human-index", type=int, default=1, choices=[0, 1])
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
    p.add_argument("--debug", action="store_true",
                   help="print the candidate table every tick: the top-m jobs the "
                        "baseline handed over, every (counter, press tick) scored "
                        "with its C and when the human's cone first covers it, and "
                        "what was picked. Use with --fps 1 or --paused")
    p.add_argument("--debug-n", type=int, default=12,
                   help="--debug: candidates to print per tick, best C first")
    p.add_argument("--hud-cands", type=int, default=6,
                   help="rows of the on-screen candidate table (0 hides it)")
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
