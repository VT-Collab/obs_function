"""The environment: one episode's worth of kitchen, plus the human factory.

Mirrors filter/baseline.py's Kitchen deliberately -- same thin shape, same
collisions-off rule, same DNF convention -- so numbers out of this package are
directly comparable to numbers out of that one. Two things are added:

    occlude=True on the human      the whole point of the new layouts. Without
                                   it the interior walls block movement but not
                                   sight, and we are back to a cone-only model.
    robot station restriction      the division-of-labour lever.

===========================================================================
COLLISIONS ARE OFF, ALWAYS
===========================================================================
Same rule as the filter package, same reason: with collisions on, a robot can
help or hinder by physically blocking a corridor, and any measured win is
confounded by traffic. Off, the only way to affect the human is through what
they KNOW and which subtasks are left for them -- which is what we are trying to
measure. It lives in Kitchen.__init__ so no caller can forget it.

This matters more here than it did on gc00. These layouts have narrow corridors
by construction, so the blocking channel would be large if it were switched on.

===========================================================================
THE STATION RESTRICTION, AND WHY IT IS ENFORCED ON THE ACTION
===========================================================================
`restrict` names station KINDS the robot may not use. The rule is applied by
turning the robot's INTERACT into a STAY when it is facing a forbidden station,
which is the least invasive place to put it:

    not in the mdp      the physics stay stock, so the human, the shadows and
                        the rollout all run the same dynamics they always did.
                        A rule baked into resolve_interacts would silently apply
                        inside every counterfactual too, and the shadows would
                        start modelling a human who cannot use the grill.
    not in the planner  a planner that merely declines to CHOOSE forbidden
                        subtasks is a soft rule the module could route around.
                        Converting the action makes it a property of the world.

The human is never restricted. That asymmetry is the point: it puts the human on
the critical path for whatever the robot cannot do, so their field of view stops
being cosmetic and starts determining whether the order gets made.
"""

import random

import _paths  # noqa: F401  MUST be first
import layouts

from overcooked_ai_py.mdp.overcooked_mdp import (                    # noqa: E402
    SteakHouseGridworld, Action, BASE_REW_SHAPING_PARAMS)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv        # noqa: E402

from fov.human.agent.limited_vision_human import (                   # noqa: E402
    LimitedVisionSteakHuman, TERRAIN_KIND, Belief, UNKNOWN)
from fov.human.planning.steak_planner import SteakMotionPlanner      # noqa: E402


ROBOT_INDEX = 0
HUMAN_INDEX = 1
N_ACTIONS = 6
HORIZON = 400
N_ORDERS = 4
#see filter/baseline.py for the full argument; kept identical so completion_time
#means the same number in both packages.
DNF_PENALTY = 100


#=========================================================================
#BURN: a steak left ready on the grill past BURN_GRACE ticks is ruined.
#
#WHY IT HAD TO EXIST, measured rather than assumed. With stock physics the whole
#funnel checked out except the last step: actionable situations on 83-99% of
#ticks, the module able to act on 100% of them, the filter recovering the true
#cone 8/8 -- and the cost function still anti-correlated with team time, worse
#the more it was listened to (+17.8 / +22.5 / +22.0 at beta=1).
#
#The reason was that IGNORANCE WAS CHEAP. A human who walks to a station and
#finds it taken simply re-decides from where they stand; measured, that costs a
#few ticks (n_abandoned 14-30, true_wasted 0-4). Meanwhile every detour the robot
#takes to inform or to cover costs schedule time directly. So informing was worth
#less than walking, and a cost function that ranks informing highly ranks
#something that does not matter.
#
#Burning changes the price, not the mechanism. "Nobody is watching the grill"
#stops costing a few steps of latency and starts costing a whole cook cycle plus
#a fresh meat fetch, while the robot's detour still costs two. That is what puts
#the two sides of the trade on comparable scales.
#
#THE MODULE LEARNS NOTHING ABOUT IT. Burning is a property of the WORLD. The cost
#function still reads only the partner's belief gap; it does not know a steak
#from a plate and has no term for burning. What changes is that a stale belief
#about the grill now precedes a real loss, so the terms that were always pointing
#at it finally point at something expensive. The module stays task-agnostic.
#
#Applied in the wrapper, never in the mdp, so the physics the shadows and the
#rollout simulate stay stock -- exactly the same reason the robot restriction
#would have lived here. BURN_GRACE=None disables it and recovers stock Overcooked.
#=========================================================================
BURN_GRACE = 12


def disable_collisions(mdp):
    """Make agents pass through each other, for this mdp instance only.

    Patches the bound method on the INSTANCE, not the class, so importing this
    module cannot change physics for anything else in the process.
    """
    def _no_collisions(old_positions, new_positions):
        return new_positions
    mdp._handle_collisions = _no_collisions
    return mdp


class Kitchen:
    """One episode's worth of environment. Owns the mdp and the clock, nothing
    else -- the tick loop lives in the callers, because the robot has to
    interleave perceive / decide / observe-human / step in a specific order."""

    def __init__(self, layout, n_orders=N_ORDERS, horizon=HORIZON,
                 restrict=(), burn_grace=BURN_GRACE):
        self.layout = layout
        self.n_orders = n_orders
        self.horizon = horizon
        #station kinds the ROBOT may not use. A tuple, not a list, because it is
        #configuration and nothing should be appending to it at runtime.
        self.restrict = frozenset(restrict)
        self.burn_grace = burn_grace
        #{grill_loc: tick it first became ready}. Reset per episode.
        self._ready_since = {}
        self.n_burned = 0

        self.mdp = SteakHouseGridworld.from_layout_name(
            layout, start_order_list=["steak"] * n_orders,
            rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
        #TRAP 1 from baseline.py: a string start_order_list makes len() count
        #characters and no delivery ever fires. Cheap assert, expensive bug.
        assert not isinstance(self.mdp.start_order_list, str)

        disable_collisions(self.mdp)
        self.env = None

    # ---- the restriction ------------------------------------------------
    def robot_may_interact(self, state):
        """False iff the robot is facing a station kind it is barred from."""
        if not self.restrict:
            return True
        p = state.players[ROBOT_INDEX]
        fx = p.position[0] + p.orientation[0]
        fy = p.position[1] + p.orientation[1]
        #guard negatives explicitly: python list indices wrap, so mtx[-1][-1]
        #quietly returns the terrain from the opposite corner of the kitchen
        #instead of raising. Same trap as cost_function.robot_line.
        if fx < 0 or fy < 0:
            return True
        try:
            kind = TERRAIN_KIND.get(self.mdp.terrain_mtx[fy][fx], "")
        except Exception:
            return True
        return kind not in self.restrict

    def apply_restriction(self, state, robot_action):
        """-> the action the robot actually gets to play."""
        if robot_action == Action.INTERACT and not self.robot_may_interact(state):
            return Action.STAY
        return robot_action

    # ---- the episode ----------------------------------------------------
    def reset(self):
        #+10 on the env horizon ON PURPOSE: our own horizon check ends the
        #episode, so the mdp's terminal stays a clean "they finished early"
        #signal instead of colliding with the timeout. Matches Kitchen in the
        #filter package -- change it and the two stop being comparable.
        self.env = OvercookedEnv.from_mdp(self.mdp, info_level=0,
                                          horizon=self.horizon + 10)
        self._ready_since = {}
        self.n_burned = 0
        return self.state

    @property
    def state(self):
        return self.env.state

    @property
    def t(self):
        return self.env.t

    def _grills(self):
        out = []
        for y, row in enumerate(self.mdp.terrain_mtx):
            for x, c in enumerate(row):
                if c == 'P':
                    out.append((x, y))
        return out

    def _apply_burn(self):
        """Ruin any steak that has been sitting ready too long.

        Removing the object leaves the grill EMPTY, which is a state both agents
        perceive through their normal channels -- no special signal, no privileged
        notification. A human who was not looking simply finds an empty grill
        later and has to start again, which is the whole point.
        """
        if not self.burn_grace:
            return
        st = self.state
        for loc in self._grills():
            obj = st.objects.get(loc)
            ready = False
            if obj is not None and getattr(obj, "name", "") == "steak":
                try:
                    ready = self.mdp.steak_ready_at_location(st, loc)
                except Exception:
                    ready = False
            if not ready:
                self._ready_since.pop(loc, None)
                continue
            t0 = self._ready_since.setdefault(loc, self.t)
            if self.t - t0 >= self.burn_grace:
                del st.objects[loc]
                self._ready_since.pop(loc, None)
                self.n_burned += 1

    def step(self, robot_action, human_action):
        """-> (sparse_reward, done). Applies the robot restriction first."""
        robot_action = self.apply_restriction(self.state, robot_action)
        joint = (robot_action, human_action)
        _, reward, done, info = self.env.step(joint)
        sparse = info["sparse_r_by_agent"] if "sparse_r_by_agent" in info else None
        sparse = sum(sparse) if sparse is not None else reward
        self._apply_burn()
        return sparse, done


def seed_familiar(agent, mdp):
    """Give `agent` the MAP of the kitchen but none of its STATE.

    Populates exactly the four things observe() would have written by looking at
    every tile -- seen_cells, known_terrain, the station location lists -- and
    then seeds every timed station with an UNKNOWN belief carrying an ancient
    `seen_at`, which is the same thing observe() does the first time a pot falls
    in the cone. So: the agent knows WHERE the grill is and does not know
    WHETHER THE STEAK IS DONE.

    ===================================================================
    WHY THIS IS NOT CHEATING, AND WHY IT MAKES FOV MATTER MORE
    ===================================================================
    The human model was deliberately changed to discover station LOCATIONS by
    looking ("MISHA NEW CHANGE", limited_vision_human.py:427), on the argument
    that a narrow cone should have to map the kitchen first. On a 3x3 room that
    costs about four ticks. On a 100-tile ring corridor it is an unsolved search
    problem, and the probe shows the result: performance stops being monotonic
    in FOV and starts being a lottery on whether wandering happened to find the
    grill -- island DNFs at 120 and 180 while succeeding at 90 and 360.

    That is a floor effect, and it is exactly as useless as gc00's ceiling: at
    both ends the human's competence is pinned by something other than what they
    can see, so assistance has nothing to move.

    Seeding the map separates the two kinds of ignorance that were confounded:

        WHERE things are   a search problem. Solved once, never changes, and a
                           real cook knows their own kitchen. Not what the paper
                           is about.
        WHAT STATE they    a perception problem. Changes on a hidden clock,
        are in             decays out of memory after FORGET_HORIZON, and is
                           recoverable only by looking. THIS is the gap the
                           robot reasons about.

    Nothing about state is revealed: beliefs start UNKNOWN, they are still
    written only by `observe` through `visible()`, and they still decay. The
    no-cheat boundary is untouched -- what moves is which question the episode
    is asking.

    APPLIED FROM OUTSIDE, never by editing limited_vision_human.py. That file is
    frozen and carries the 31-layout validation; this package must not be able
    to change results attributed to it. Turning `familiar` off recovers the
    stock agent bit for bit.

    THE SHADOWS NEED THE SAME TREATMENT. The robot's hypothesis-humans inside
    the filter must make the same assumption as the real human, or the robot is
    modelling a partner who is lost when theirs is not. Whatever this function
    does here, inference must do to every shadow it builds.
    """
    for y, row in enumerate(mdp.terrain_mtx):
        for x, terrain in enumerate(row):
            cell = (x, y)
            agent.seen_cells.add(cell)
            agent.known_terrain[cell] = terrain
            kind = TERRAIN_KIND.get(terrain)
            if kind and cell not in agent.stations[kind]:
                agent.stations[kind].append(cell)
                if kind in ("pot", "board", "sink"):
                    #location known, state NOT. The ancient timestamp is what
                    #makes it read as "never observed" rather than "observed at
                    #tick 0", which would be a free look at the true state.
                    agent.beliefs[cell] = Belief(UNKNOWN, -10 ** 9)
    return agent


def make_human(mdp, fov, seed, temperature=0.5, occlude=True, familiar=True):
    """A fresh LimitedVisionSteakHuman with a reproducible sampler.

    Seeding the GLOBAL stream first is load-bearing and the order is the whole
    point: LimitedVisionSteakHuman.reset() seeds its own private RNG by drawing
    from `random`, so this is what makes every arm start from the SAME draw
    sequence and diverge only because the world diverged. Get it backwards and a
    "paired" comparison is not paired.

    occlude DEFAULTS TO TRUE here, unlike the human's own signature. On these
    layouts it is the entire experiment -- it is what makes a wall block sight
    and not just movement, and it is the only source of ignorance left at
    fov=360 (where _visible_cone returns True unconditionally). Passing False
    reduces these layouts back to cone-only, which is a useful ablation and a
    disastrous default.

    planner=SteakMotionPlanner(mdp, None): the None is CORRECT. The second
    argument would be a MediumLevelPlanner, and handing the human one silently
    gives it a map of the whole kitchen, which destroys the experiment. The
    planner object is kept only to look up which station a subtask targets.
    """
    random.seed(seed)
    planner = SteakMotionPlanner(mdp, None)
    h = LimitedVisionSteakHuman(mdp, fov, planner, agent_index=HUMAN_INDEX,
                                temperature=temperature, occlude=occlude)
    if familiar:
        seed_familiar(h, mdp)
    return h


def stage():
    """Install our layouts where from_layout_name can see them."""
    return layouts.stage()
