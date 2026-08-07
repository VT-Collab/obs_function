"""THE COST FUNCTION.  No reward, no recipe, no delivery -- only the partner.

===========================================================================
THE TWO IDEAS
===========================================================================
    1. DIVISION OF LABOUR.  Do not do the job your partner is already doing.
    2. OBSERVABILITY.       When you do change your partner's world, prefer
                            changes they can SEE over changes they cannot.

Both are read off the same hidden state: the partner's FIELD OF VIEW and their
current SUBTASK, estimated online by the Bayesian filter. Neither reads reward.
There is no notion of a steak, a plate, an order or a score anywhere in this
file; it never touches sparse reward, shaped reward, `num_orders_remaining`, or
the step budget. Every term is a statement about how much a candidate action
would change what the partner KNOWS and INTENDS.

NOTE ON SCOPE: physical collision avoidance is deliberately NOT part of this
cost function. An earlier version had `interference` / `self_block` terms that
charged the robot for blocking the partner's footstep, and they dominated
everything else -- but they are a different mechanism (motion prediction), they
are not FOV-graded, and they are out of scope for this module. They were
removed. What is left is entirely task- and belief-level.

===========================================================================
WHY THE VISIBILITY SPLIT IS THE ENTIRE SIGNAL
===========================================================================
A cooperating agent constantly invalidates its partner's plans -- taking the
last plate, filling the pot they were walking to. That is not a defect; it is
division of labour. What makes it costly is not the invalidation, it is the
SILENCE. LimitedVisionSteakHuman documents its own mechanism precisely:

  * `_commit_still_useful` abandons an errand ONLY on a CONCRETE belief, and
    concrete beliefs are written only inside the vision cone. So a human who
    sees the robot fill the pot re-plans mid-trip and loses nothing; a human
    who does not walks the whole wasted errand and pays for it
    (`n_wasted_commits`).
  * `_weights` / `_available_advancing` down-weight or yield tasks the human
    SEES the teammate carrying or standing at -- again, only in-cone.

So the SAME robot action is nearly free against a wide cone and expensive
against a narrow one, and the difference is exactly the quantity the Bayesian
filter estimates. That is why this cost is FOV-driven rather than FOV-flavoured:
delete the posterior and every term collapses to a constant.

===========================================================================
THE TERMS
===========================================================================
All six are DIFFERENCES between two futures that share the same human action:

    s_a   = T(s, (a, h))       the robot takes candidate action a
    s_ref = T(s, (STAY, h))    the robot does nothing

Differencing against a fixed reference is what makes each number attributable
to the ROBOT. Anything the human did to itself, anything the pots did on their
own, and anything the human was going to discover this tick appears identically
in both and cancels.

  blindside        [cost]     robot changes a station and the human ends up
                              CONFIDENTLY WRONG about it. Signed: an action
                              that repairs a false belief scores negative.
                              A belief that is UNKNOWN is NOT blindsided -- a
                              human who knows they do not know goes and looks.
  silent_invalid   [cost]     robot breaks the precondition of the subtask the
                              posterior says the human is on, unseen.
  visible_invalid  [benefit]  same break, but in view: this is LEGIBLE
                              take-over. The human re-plans immediately, which
                              is the coordination the human model was built to
                              reward.
  restored         [benefit]  robot RE-ENABLES a subtask the human wanted
                              (pulling the ready steak out re-empties the pot).
  strand           [cost]     the human's own decision code, run on its
                              hypothetical post-action beliefs, would put more
                              mass on look/wait than it would have otherwise.
  shift            [either]   raw total-variation movement of the human's
                              subtask distribution -- "how much would the human
                              model change", with the sign left to a weight.
  legible_yield    [benefit]  the SIGNALLING term, and the only one a MOVEMENT
                              action can score on. The human model has two
                              channels that react to the teammate's visible
                              pose alone: `ROBOT_HELD_FETCH` (seeing you carry
                              meat down-weights their own pickup_meat) and
                              `STATION_TASKS` (seeing you stand at the board
                              makes them yield the board). Both are FOV-gated.
                              This term is the probability mass the human moves
                              OFF exactly those tasks -- division of labour
                              bought by standing where you can be seen -- and it
                              is only counted when it does NOT strand them.
                              Against a blind partner it is identically zero,
                              which is the correct answer: there is no point
                              signalling to someone who is not looking.
  task_overlap     [cost]     DIVISION OF LABOUR, and the term a MOVEMENT
                              action scores on most often. The posterior says
                              which LINE (pot / board / sink) the partner is
                              working; this charges the robot for engaging that
                              line MORE THAN AVERAGE, and pays it for engaging
                              a line the partner has left alone.
                              CENTRED on purpose: the raw "how much does this
                              action engage line k" summed over k is just "walk
                              toward stations", a blunt attractor with no idea
                              WHICH station, and using it uncentred wrecks the
                              policy (measured: 3 deliveries -> 1). Subtracting
                              the mean claim removes that component and leaves
                              only the part that says whose job this is.
                              Ungated by visibility, unlike `approach`:
                              duplicated work is wasted whether or not the
                              partner notices it happening.
  approach         [cost]     the DEPTH TERM. blindside and silent_invalid can
                              only fire on the single tick the robot INTERACTS,
                              because that is the only action a one-step
                              transition lets change a station -- measured, they
                              are live on 1-4% of ticks. But the damage is
                              decided ten steps earlier, when the robot starts
                              WALKING to a station its partner is already
                              committed to and cannot see. This term is that
                              approach, made continuous: the (contested,
                              unobserved) posterior mass on each station kind,
                              times how much closer this action gets the robot
                              to it. Distances are exact BFS over the true
                              walkable map, precomputed once per layout, so it
                              costs a dict lookup per action.
                              The contested factor is pure human model; the
                              distance is geometry. Neither is reward.

===========================================================================
ADAPTIVE DEFERENCE -- WHY THIS DOES NOT MAKE A TIMID ROBOT
===========================================================================
Protecting the plan of a partner who has no plan is pointless. `blindside` and
`silent_invalid` are scaled by

    defer = 1 - P(the human's current subtask is explore / check_* / wait)

read straight off the predictive posterior. A blind human spends its opening
phase mapping the kitchen, so defer ~ 0, every deference term vanishes, and the
robot is left with the unmodified baseline -- which is the right answer, and is
also what fov/robot/policy/old/RESULTS.md found empirically ("blind human ->
the robot should just cook the whole pipeline itself"). A sighted, productive
partner drives defer -> 1 and the robot starts coordinating. Nothing hand-codes
that schedule; it falls out of the human model.
"""
import _paths  # noqa: F401

from overcooked_ai_py.mdp.overcooked_mdp import Action, EVENT_TYPES

from fov.human.agent.limited_vision_human import (
    ROBOT_HELD_FETCH, STATION_TASKS)

from human_model import believed_from, NON_ADVANCING, UNKNOWN

#Precondition of every subtask that the ROBOT can break, as (station kind,
#station values under which the subtask still makes sense). Mirrors
#LimitedVisionSteakHuman's own `_sample_still_helpful` / `_commit_still_useful`
#/ `_forced_held` line for line, with one difference: those are evaluated on
#the human's BELIEFS (where UNKNOWN means "keep going"), and these are
#evaluated on the TRUE world, where UNKNOWN cannot occur.
#
#This table is human-model structure, not task knowledge: it says which
#station a subtask needs and in what condition, which is exactly what the human
#agent's own code says. It contains no ordering, no value, and no goal.
PRECONDITION = {
    "pickup_meat":      ("pot",   ("empty",)),
    "drop_meat":        ("pot",   ("empty",)),
    "pickup_onion":     ("board", ("empty",)),
    "drop_onion":       ("board", ("empty",)),
    "chop_onion":       ("board", ("chopping",)),
    "pickup_plate":     ("sink",  ("empty",)),
    "drop_plate":       ("sink",  ("empty",)),
    "heat_washed_plate":   ("sink",  ("washing",)),
    "pickup_washed_plate": ("sink",  ("ready",)),
    "pickup_steak":     ("pot",   ("ready",)),
    "pickup_garnish":   ("board", ("ready",)),
}

TERM_NAMES = ("blindside", "silent_invalid", "visible_invalid", "restored",
              "strand", "shift", "legible_yield", "approach", "task_overlap")

#Default weights. Signs are FIXED by the argument above (costs positive,
#benefits negative); only the magnitudes are swept, and the sweep is run on
#seeds disjoint from the reported ones -- see RESULTS.md.
DEFAULT_WEIGHTS = {
    "blindside":       1.0,
    "silent_invalid":  2.0,
    "visible_invalid": -0.5,
    "restored":        -0.5,
    "strand":          1.0,
    "shift":           0.0,
    "legible_yield":   -2.0,
    "approach":        1.0,
    "task_overlap":    1.0,
}


#Named weight vectors. `full` is what the module ships with and what every
#tuning grid deviates from. `collision` and `kb` are the two halves the
#decomposition in RESULTS.md splits the cost function into -- kept here so the
#result is reproducible with one flag rather than a hand-typed JSON blob.
#
#`division` = the two duplication terms (which station kind is whose job).
#`belief`   = the eight terms about what the partner KNOWS.
#Both are read off the same (FOV, subtask) posterior; neither reads reward.
def _half(keep):
    w = {k: 0.0 for k in DEFAULT_WEIGHTS}
    for k in keep:
        w[k] = DEFAULT_WEIGHTS[k]
    return w


PRESETS = {
    "full": dict(DEFAULT_WEIGHTS),
    "division": _half(("task_overlap",)),
    "belief": _half(("blindside", "silent_invalid", "visible_invalid",
                     "restored", "strand", "shift", "legible_yield",
                     "approach")),
}


def counterfactual_transition(mdp, state, joint_action):
    """`mdp.get_state_transition` with rollout=True, and no assertions.

    The stock method is called with rollout=False, which writes every object it
    creates into `mdp.object_id_dict` -- a registry shared with the REAL
    episode. Speculating over 6 robot actions every tick through that path
    would grow the registry without bound and shift the ids the real trajectory
    hands out. `rollout=True` is the mdp's own flag for exactly this: it takes
    the object counter off the state instead, and skips the registry write.
    Nothing else differs -- interacts, movement and environment effects are the
    same calls in the same order (overcooked_mdp.py:863).
    """
    events = {e: [False] * mdp.num_players for e in EVENT_TYPES}
    nxt = state.deepcopy()
    mdp.resolve_interacts(nxt, joint_action, events, rollout=True)
    mdp.resolve_movement(nxt, joint_action)
    mdp.step_environment_effects(nxt)
    return nxt


class ObservableDivergenceCost:
    """Evaluates the seven terms for one (robot action, human action) pair."""

    def __init__(self, mdp, human_index=1, robot_index=0, weights=None):
        self.mdp = mdp
        self.hi = human_index
        self.ri = robot_index
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        #TRUE station cells. The robot is a fully-observable agent -- this is
        #the same map build_full_state already hands its network. It is never
        #used to answer a question about what the HUMAN knows; every such
        #question goes through a shadow's own `stations` / `beliefs`.
        self.station_cells = {
            "pot": list(mdp.get_pot_locations()),
            "board": list(mdp.get_chopping_board_locations()),
            "sink": list(mdp.get_sink_locations()),
        }
        self.dist = _distance_fields(mdp, self.station_cells)
        self.walkable = set(mdp.get_valid_player_positions())

    # -- ground truth --------------------------------------------------------

    def robot_redundant_tasks(self, state):
        """The human tasks THIS robot pose makes redundant, per the human's own
        two teammate-awareness tables. Reads only the robot's own body and the
        true station map -- both things the robot obviously knows about itself.
        Whether the human ACTS on any of it is decided entirely by whether the
        human can see the robot, which is what the cost term measures."""
        rp = state.players[self.ri]
        out = set()
        held = rp.held_object.name if rp.held_object else None
        fetch = ROBOT_HELD_FETCH.get(held)
        if fetch:
            out.add(fetch)
        faced = (rp.position[0] + rp.orientation[0],
                 rp.position[1] + rp.orientation[1])
        for kind, cells in self.station_cells.items():
            if faced in cells:
                out |= set(STATION_TASKS.get(kind, ()))
        return out

    def true_kind(self, shadow, state, kind, cells):
        """Aggregate the TRUE state of `cells` the way the human aggregates its
        beliefs about them (LimitedVisionSteakHuman.believed's preference
        order). `shadow._true_state_of` is a pure function of (mdp, state,
        loc) -- the shadow only supplies the mdp and the station timings."""
        if not cells:
            return None
        vals = [shadow._true_state_of(state, c) for c in cells]
        for pref in ("ready", "cooking", "chopping", "washing", "occupied",
                     "empty"):
            if pref in vals:
                return pref
        return None

    # -- one candidate -------------------------------------------------------

    def evaluate(self, hm, state, s_a, s_ref, hypotheses, human_action,
                 robot_action=None):
        """Seven raw term values for one candidate action.

        hm          PredictiveHumanModel (for beliefs_after / scratch)
        state       s, the state being planned in
        s_a         T(s, (a, h))
        s_ref       T(s, (STAY, h))
        hypotheses  the predictive posterior as a list of
                    {fov, shadow, subtask, prob, next_sampled}
        """
        terms = dict.fromkeys(TERM_NAMES, 0.0)

        if not hypotheses:
            return terms
        r_after = tuple(s_a.players[self.ri].position)
        r_ref = tuple(s_ref.players[self.ri].position)
        probe = hypotheses[0]["shadow"]     # any shadow: _true_state_of is pure

        # ---- DIVISION OF LABOUR -------------------------------------------
        # WHICH LINE is the partner on? Read straight off the posterior: every
        # subtask in PRECONDITION names the station kind it needs, and that
        # kind IS the line ("pickup_meat" and "drop_meat" are both the pot
        # line, which is exactly how the human's own _sample_still_helpful
        # tests them). Subtasks with no precondition (deliver, explore,
        # check_*, dump_item) claim no line, so their mass simply does not
        # appear here -- a partner that is lost claims nothing and the robot is
        # free.
        line = {"pot": 0.0, "board": 0.0, "sink": 0.0}
        for h in hypotheses:
            spec = PRECONDITION.get(h["subtask"])
            if spec is not None:
                line[spec[0]] += h["prob"]

        # HOW MUCH does this candidate engage each line? One step of approach
        # counts 1 (exact BFS over the true walkable map, precomputed); actually
        # CHANGING a station of that kind counts _ACT_WEIGHT, because it is the
        # commitment the approach was only heading toward.
        engage = {}
        for k, cells in self.station_cells.items():
            d_a = self.dist[k].get(r_after, _FAR)
            d_r = self.dist[k].get(r_ref, _FAR)
            e = float(d_r - d_a) if (d_a < _FAR and d_r < _FAR) else 0.0
            for c in cells:
                if probe._true_state_of(s_a, c) != probe._true_state_of(s_ref, c):
                    e += _ACT_WEIGHT
                    break
            engage[k] = e
        #CENTRE the claim. sum_k engage[k] is "am I walking toward stations at
        #all", which is not a coordination signal and swamps the one we want;
        #subtracting the mean claim leaves only "am I engaging the line my
        #partner is on, relative to the lines they are not".
        mean_line = sum(line.values()) / float(len(line))
        for k, e in engage.items():
            terms["task_overlap"] += (line[k] - mean_line) * e

        #group the hypotheses by fov: the perception-level terms (blindside)
        #depend on the cone only, the plan-level terms on (cone, subtask)
        by_fov = {}
        for h in hypotheses:
            by_fov.setdefault(h["fov"], []).append(h)

        for fov, hyps in by_fov.items():
            shadow = hyps[0]["shadow"]
            p_fov = sum(h["prob"] for h in hyps)
            if p_fov <= 0.0:
                continue

            bel_a, seen_a = hm.beliefs_after(shadow, s_a)
            bel_ref, seen_ref = hm.beliefs_after(shadow, s_ref)

            # ---- blindside: a station the robot changed, that the human now
            # holds a CONFIDENT and WRONG belief about.
            blind = 0.0
            for loc, bel in bel_a.items():
                if loc == shadow.ROBOT:
                    continue
                v_a = shadow._true_state_of(s_a, loc)
                v_ref = shadow._true_state_of(s_ref, loc)
                if v_a == v_ref:
                    continue                      # the robot changed nothing here
                b_a = bel.value
                b_ref = bel_ref.get(loc, bel).value
                #UNKNOWN is NOT a blindside: a human who knows they do not
                #know emits a check_* and goes to look. Only a belief that is
                #both CONFIDENT and WRONG buys a wasted trip.
                wrong_a = (b_a != UNKNOWN and b_a != v_a)
                wrong_ref = (b_ref != UNKNOWN and b_ref != v_ref)
                blind += float(wrong_a) - float(wrong_ref)
            terms["blindside"] += p_fov * blind

            # ---- plan-level terms, per subtask hypothesis
            #cache the human's own decision code per distinct `committed` value
            dist_cache_a, dist_cache_ref = {}, {}
            sc_a = hm.scratch(shadow, bel_a, seen_a)
            sc_ref = hm.scratch(shadow, bel_ref, seen_ref)
            redundant = self.robot_redundant_tasks(s_a)

            for h in hyps:
                p = h["prob"]
                if p <= 0.0:
                    continue

                # -- precondition break / restore, on the TRUE world
                spec = PRECONDITION.get(h["subtask"])
                if spec is not None:
                    kind, ok = spec
                    #the stations THIS human has actually found; if it has
                    #found none of that kind it has no errand there
                    cells = shadow.stations.get(kind, [])
                    t_a = self.true_kind(shadow, s_a, kind, cells)
                    t_ref = self.true_kind(shadow, s_ref, kind, cells)
                    if t_ref is not None and t_a is not None and t_a != t_ref:
                        if (t_ref in ok) and (t_a not in ok):
                            #does the human find out? their post-action belief
                            #about that kind matches the truth iff they saw it
                            if believed_from(bel_a, cells) == t_a:
                                terms["visible_invalid"] += p
                            else:
                                terms["silent_invalid"] += p
                        elif (t_ref not in ok) and (t_a in ok):
                            terms["restored"] += p

                # -- what the human's OWN policy would do next, under each future
                committed = h.get("next_sampled")
                key = committed
                if key not in dist_cache_a:
                    dist_cache_a[key] = _safe_distribution(sc_a, s_a, committed)
                    dist_cache_ref[key] = _safe_distribution(sc_ref, s_ref,
                                                             committed)
                d_a, d_ref = dist_cache_a[key], dist_cache_ref[key]

                na_a = sum(v for k, v in d_a.items() if k in NON_ADVANCING)
                na_ref = sum(v for k, v in d_ref.items() if k in NON_ADVANCING)
                terms["strand"] += p * (na_a - na_ref)

                #signalling: mass moved OFF the tasks this robot pose makes
                #redundant, but only when the human is not left with nothing.
                #Zero unless the human can actually see the robot.
                if redundant and na_a <= na_ref + 1e-9:
                    moved = sum(max(0.0, d_ref.get(z, 0.0) - d_a.get(z, 0.0))
                                for z in redundant)
                    terms["legible_yield"] += p * moved

                keys = set(d_a) | set(d_ref)
                tv = 0.5 * sum(abs(d_a.get(k, 0.0) - d_ref.get(k, 0.0))
                               for k in keys)
                terms["shift"] += p * tv

                # -- APPROACH: the same deference, made continuous.
                # If this hypothesis has the human working toward station kind
                # `kind`, and this human cannot currently SEE that kind, then
                # every step the robot takes toward it is a step toward a
                # silent invalidation that has not happened yet. Weight the
                # step by how contested the station is, not by any notion of
                # what is there.
                if spec is not None:
                    kind = spec[0]
                    cells = shadow.stations.get(kind, [])
                    if cells and believed_from(bel_a, cells) != self.true_kind(
                            shadow, s_a, kind, cells):
                        #the human's post-action belief about this kind does
                        #NOT match reality -> they are not watching it
                        d_a_k = self.dist[kind].get(
                            tuple(s_a.players[self.ri].position), _FAR)
                        d_r_k = self.dist[kind].get(
                            tuple(s_ref.players[self.ri].position), _FAR)
                        if d_a_k < _FAR and d_r_k < _FAR:
                            terms["approach"] += p * float(d_r_k - d_a_k)

        return terms

    def combine(self, terms, defer):
        """Weighted sum. `defer` scales only the two DEFERENCE terms."""
        w = self.weights
        return (defer * w["blindside"] * terms["blindside"]
                + defer * w["silent_invalid"] * terms["silent_invalid"]
                + w["visible_invalid"] * terms["visible_invalid"]
                + w["restored"] * terms["restored"]
                + w["strand"] * terms["strand"]
                + w["shift"] * terms["shift"]
                + w["legible_yield"] * terms["legible_yield"]
                + defer * w["approach"] * terms["approach"]
                + defer * w["task_overlap"] * terms["task_overlap"])


_FAR = 999
#one INTERACT that changes a station is worth this many steps of walking
#toward it, when scoring how much a candidate engages a line of work
_ACT_WEIGHT = 3.0


def _distance_fields(mdp, station_cells):
    """{kind: {cell: steps to the nearest tile you can interact with a station
    of that kind from}}. One multi-source BFS per kind over the true walkable
    map, done once per layout.

    The robot legitimately knows the map -- build_full_state already hands its
    network every terrain plane. This is never used to answer a question about
    what the HUMAN knows; the human's own routing stays its own BFS over cells
    it has actually seen.
    """
    walk = set(mdp.get_valid_player_positions())
    dirs = ((0, -1), (0, 1), (1, 0), (-1, 0))
    fields = {}
    for kind, cells in station_cells.items():
        dist = {}
        frontier = []
        for c in cells:
            for d in dirs:
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in walk and nb not in dist:
                    dist[nb] = 0
                    frontier.append(nb)
        while frontier:
            nxt = []
            for cell in frontier:
                for d in dirs:
                    nb = (cell[0] + d[0], cell[1] + d[1])
                    if nb in walk and nb not in dist:
                        dist[nb] = dist[cell] + 1
                        nxt.append(nb)
            frontier = nxt
        fields[kind] = dist
    return fields


def _safe_distribution(scratch_shadow, state, committed):
    """`subtask_distribution` on a hypothetical knowledge base.

    This calls the HUMAN'S OWN code -- no predicate is re-implemented here. It
    draws no random numbers, so it is safe to run speculatively (that is the
    same guarantee `execute()` carries and the reason the filter can use it)."""
    try:
        return scratch_shadow.subtask_distribution(state, committed)
    except Exception:
        return {}


def defer_weight(hypotheses):
    """1 - P(the human is on a look / wait), off the predictive posterior.

    A partner with no plan has no plan to protect. This is the single knob that
    turns the module from "coordinate carefully" (sighted partner) into "get
    out of the way of the baseline" (blind partner), and it is read from the
    filter rather than from the true FOV.
    """
    if not hypotheses:
        return 0.0
    non_adv = sum(h["prob"] for h in hypotheses
                  if h["subtask"] in NON_ADVANCING)
    return max(0.0, min(1.0, 1.0 - non_adv))
