"""Actual human behavior from user study selection.

The human model interface.py actually seats: every action it ever takes is
whatever subtask the study participant clicked, executed by an unmodified
human/limited_vision_human.py LimitedVisionHuman, against a belief that
LimitedVisionHuman built by genuinely watching the real env state through
its own cone -- never told the truth, never given a click it did not
receive. See HumanBehavior's docstring for the mechanism.

legal_menu() -- the "what does this belief make legal right now" computation
-- is factored out to a module function because
user_study/human/approximate_human_model.py needs the exact same belief-
gated legality (ordinary ladder AND look-for) to build the DISTRIBUTION a
shadow samples from; the two files must never compute this two different
ways, or a filter shadow could offer a subtask a real human study run of
this class would not.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # .../user_study/human
USER_STUDY = os.path.dirname(HERE)                            # .../user_study
MISHA = os.path.dirname(USER_STUDY)                            # .../misha
sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, MISHA)

from overcooked_ai_py.mdp.actions import Action                # noqa: E402

from common import geometry as geo                             # noqa: E402
from common import tasks as _tasks                             # noqa: E402
from common.tasks import (T_EXPLORE, T_FETCH, MEAT, ONION, PLATE,  # noqa: E402
                          legal_subtasks)
from human.limited_vision_human import (LimitedVisionHuman,    # noqa: E402
                                        _PREFIX as LOOK_PREFIX)

# NO SURPLUS CAP IN THE USER STUDY. common/tasks.py's SURPLUS_AT=2 stops
# LimitedVisionHuman's own autonomous ladder from endlessly overproducing --
# right for an agent with no other judgment, wrong for a study participant
# with real agency, who should be free to fetch another onion even with two
# garnish already waiting, and who has no way to tell that a menu row
# disappeared for that reason rather than, say, not having found the
# dispenser yet. common/tasks.py is outside user_study/ and never edited --
# this scopes the change to exactly the one call below (and, by extension,
# to ApproximateHumanModel/SubtaskFOVPosterior, which both go through this
# same function) by patching the module CONSTANT for the duration of that
# call only, same pattern as `view.optimistic` just below.
_NO_SURPLUS = 10 ** 9


def reset_explore_patience(core, verb):
    """The PATIENCE-clock reset a genuine (non-explore) commitment earns,
    factored out so every caller that forces a subtask through `core`
    applies it identically.

    `core.explore_ticks` is LimitedVisionHuman's own count of UNBROKEN
    exploring ticks (human/limited_vision_human.py's `_explore()`: after
    `patience` ticks it blacklists the current tile and restarts). Normally
    `_ladder_decide()` resets it to 0 itself the instant real work is found
    -- see that method's own docstring: "a spell of useful work in between
    means the next spell of exploring deserves a full budget rather than
    the remainder of the last one." Every caller in this package that
    forces a subtask through `core` does so by SHADOWING
    `core._ladder_decide` with a lambda that returns the forced tuple
    directly (see HumanBehavior's own docstring on why this works) --
    which means `_ladder_decide()`'s BODY, reset included, never runs. This
    call is what puts that side effect back for whichever `verb` this tick
    actually forced through.

    Confirmed missing in practice for TWO of this package's three
    users before this fix: SubtaskFOVPosterior's observers call
    `core._explore()` directly every tick, for every fov, to price the
    "explore" candidate's prediction -- entirely outside `_ladder_decide()`
    -- so an observer that has been quietly tracking a NON-explore
    hypothesis for many ticks still has `explore_ticks` climbing the whole
    time, and its very first "explore" prediction once the human genuinely
    switches is built on a wildly stale PATIENCE clock (confirmed live: by
    tick 80 it had already wrapped past `patience` multiple times,
    blacklisting frontier tiles a freshly-exploring human never would have
    given up on, making that fov's own predicted direction diverge from the
    real one). ApproximateHumanModel.action() had a comment invoking "the
    same reasoning as HumanBehavior/_ladder_decide's own PATIENCE reset"
    but never actually called it -- this fixes that gap too, which matters
    because `SubtaskFOVPosterior.best_shadow()` clones an OBSERVER (not a
    fresh instance) into the ApproximateHumanModel that
    user_study/robot/filter.py rolls forward: any correction made here
    reaches that rollout automatically, through the same shared object and
    the same shared method, with nothing extra to wire up.
    """
    if verb != "explore":
        core.explore_ticks = 0


def legal_menu(core, state, look_commit):
    """(ranked, legal) for `core`'s belief, having just genuinely observed
    `state` through its own cone.

    `ranked` is common/tasks.py's raw legal_subtasks() output -- [(tier,
    verb, cell), ...] -- kept around because callers need to re-check an
    exact (verb, cell) match later without recomputing it.

    `legal` is verb -> (tier, dist, cell), the WHOLE menu: the ordinary
    ladder run STRICT (`view.optimistic = False`: a station whose contents
    are unknown or have decayed is never guessed at), EXCEPT for one single
    carve-out -- the held-branch LOAD verb (load_board/load_pot/load_sink)
    for whatever raw item is currently in hand is additionally offered if a
    known station's EMPTY status specifically is unconfirmed. See the
    two-pass block below for why: optimism belongs on "empty" alone, not on
    "ready"/"in_progress" too, even though BeliefView._maybe() drives all
    three through the identical `optimistic` flag. Two things were tried and
    rejected before this: forcing `optimistic=False` everywhere left
    load_board missing the instant the board's contents record was unknown
    or had decayed, even holding the very onion that made it the obvious
    next move; forcing `optimistic=True` everywhere overcorrected and made
    "collect_garnish"/"chop" legal for boards the participant had only ever
    glimpsed once from a distance, never seen anyone use, and had no reason
    to believe held anything at all. "explore" is always available, plus one
    row per item core.sightings
    still remembers seeing the robot carry that is worth going to check on,
    priced and targeted with core's own `_look_tier`/`_candidates` -- the
    SAME calls the autonomous look-for itself uses, not reimplemented; the
    only thing not reused from LimitedVisionHuman's own `_scan()` is its
    "must strictly beat the ladder's own top pick" cutoff, which belongs to
    an agent picking for itself, not to a menu a chooser reads from.

    `look_commit` is the caller's own {item: committed counter cell} dict,
    read and mutated in place across calls. Without it, a look-for's target
    would re-argmin every tick and could flip between equally-good counters
    as WALK distance shifts under an unmoving L1-from-remembered-position
    term, with nothing about the sighting itself having changed -- the same
    pacing hazard rank()'s STICKY WITHIN A TIER and _scan()'s own COMMIT
    comment exist to prevent elsewhere in this package.
    """
    core.observe(state)
    view = core.view
    me = state.players[core.agent_index]
    pos = tuple(me.position)
    held = me.held_object.name if me.held_object else None
    walk = view.walkable | {pos}
    field = geo.dist_field(walk, pos)

    def dist(cell):
        return geo.path_len_in(field, walk, cell)
    ok = lambda c: dist(c) is not None                        # noqa: E731

    prev_surplus = _tasks.SURPLUS_AT
    _tasks.SURPLUS_AT = _NO_SURPLUS
    try:
        view.optimistic = False
        ranked = legal_subtasks(view, held, ok, allow_stash=True)

        # OPTIMISM SCOPED TO "EMPTY" ONLY, NOT TO "READY"/"IN_PROGRESS" TOO.
        # A station believed EMPTY is worth walking over and trying to LOAD:
        # arrival reveals the truth either way, so a wrong guess costs at
        # most a wasted trip -- views.py's own "OPTIMISM UNDER DECAY"
        # justification, and the one place it was written for. A station
        # guessed READY or IN_PROGRESS is a much bigger, ungrounded claim --
        # "there is specifically a half-finished thing sitting there right
        # now" -- with no cheap way to find out you were wrong (you'd just
        # be offering "chop"/"collect_garnish" for a board no one has ever
        # actually used). So: only the held-branch LOAD verb for whatever
        # raw is in hand gets a second, optimistic pass, and only its own
        # rows are pulled in -- everything else stays exactly what the
        # strict pass produced, stash fallback included if that's what it
        # was.
        load_verb = {"meat": "load_pot", "onion": "load_board",
                    "plate": "load_sink"}.get(held)
        if load_verb and load_verb not in {v for _t, v, _c in ranked}:
            view.optimistic = True
            optimistic = legal_subtasks(view, held, ok, allow_stash=True)
            ranked = ranked + [s for s in optimistic if s[1] == load_verb]
    finally:
        view.optimistic = True     # the agent's own resting default
        _tasks.SURPLUS_AT = prev_surplus

    # A RAW ITEM NEEDS ONLY ITSELF, FOR THE USER STUDY, AND STAYS LEGAL UNTIL
    # THE HUMAN HAS ACTUALLY SEEN IT UNAVAILABLE. legal_subtasks() gates both
    # get_<raw> (from the dispenser) and take_<raw> (already sitting on a
    # counter) behind actionable()/handoffable() -- for plate, "I already
    # believe SOME sink is empty" -- because it exists to stop
    # LimitedVisionHuman's own autonomous ladder fetching/picking up a raw
    # item with nowhere yet believed to put it (see actionable()'s
    # TERMINATION docstring: a ladder that re-decides every tick with no
    # other judgment can loop forever on that). None of that applies to a
    # study participant, who decides once, per click, and is not going to
    # pick a plate up and put it down forever just because no sink is known
    # yet -- they can plainly see the plate sitting there, or the dispenser
    # standing there, whether or not they have ever found where it goes.
    # Confirmed live twice: get_onion missing the instant the dispenser came
    # into view with the board still unseen, and (this fix) an already-
    # visible loose plate on a counter not pickable because no sink had been
    # seen either. So here, empty-handed only, both verbs are added purely
    # on "is the dispenser/counter known and reachable" -- nothing else.
    # counters_holding() itself is unaffected (unconditionally strict in
    # views.py: still only counts a counter this belief has actually,
    # freshly seen holding the item -- this does not let the menu guess an
    # item onto a counter, only removes the downstream-destination
    # requirement once one is genuinely there). The only thing that can ever
    # take either off the menu is `ok` itself going False -- the human's own
    # belief saying the path there is blocked -- direct observation, never
    # an inference about a destination station, surplus, or the other verb.
    if held is None:
        ranked_verbs = {v for _t, v, _c in ranked}
        for raw, disp in (("meat", MEAT), ("onion", ONION), ("plate", PLATE)):
            take_verb = "take_" + raw
            if take_verb not in ranked_verbs:
                cells = [c for c in view.counters_holding(raw) if ok(c)]
                ranked = ranked + [(T_FETCH, take_verb, c) for c in cells]

            verb = "get_" + raw
            if verb in ranked_verbs:
                continue
            cells = [c for c in view.stations(disp) if ok(c)]
            ranked = ranked + [(T_FETCH, verb, c) for c in cells]

    best = {}
    for tier, verb, cell in ranked:
        d = dist(cell)
        if verb not in best or d < best[verb][1]:
            best[verb] = (tier, d, cell)

    for item in sorted(core.sightings):
        tier = core._look_tier(item, held, ok)
        if tier is None:
            continue
        found = view.counters_holding(item)
        if found and (core.suppress_unreachable or any(ok(c) for c in found)):
            continue                    # already know exactly where it is
        cands = core._candidates(item, dist)
        if not cands:
            continue
        committed = look_commit.get(item)
        if committed in cands:
            goal = committed
        else:
            rpos = core.sightings[item][0]
            goal = min(cands, key=lambda c: (abs(c[0] - rpos[0])
                                             + abs(c[1] - rpos[1]), dist(c), c))
            look_commit[item] = goal
        best[LOOK_PREFIX + item] = (tier, dist(goal), goal)

    # drop commitments for items the sighting book no longer tracks, or
    # they would just accumulate for the rest of the episode
    for item in list(look_commit):
        if item not in core.sightings:
            del look_commit[item]

    best["explore"] = (T_EXPLORE, None, None)
    return ranked, best


class HumanBehavior:
    """LimitedVisionHuman's full subtask vocabulary -- the ordinary ladder
    AND the look-for -- judged against exactly what `_core` (the actual
    human model this study runs) has itself genuinely seen and done; the
    CHOICE among whatever is genuinely legal is the study participant's
    instead of the ladder's own argmax.

    Composition, not a subclass: `_core` is a plain LimitedVisionHuman, used
    unmodified for perception, movement AND its sighting book. `_refresh_menu`
    reads `_core.view` -- its own real BeliefView -- through `legal_menu()`
    (see that function); there is no second view, no seeding of belief with
    information `_core` has not actually earned by looking. `_core.observe()`
    is the one and only thing that ever writes that belief, called every tick
    inside `legal_menu()`, so pathing (`_core.action()`'s own A* over
    `_core.view.walkable`) and the menu are reading the identical, single
    belief and can never disagree about what is reachable.

    Each tick this class temporarily shadows `_core._ladder_decide` with a
    function that returns the participant's own pick, then calls
    `_core.action()` completely unmodified -- so every bit of walking,
    turning, INTERACTing and the `t`/PATIENCE bookkeeping is the SAME tested
    code path play.py's --autopilot already runs, and the only thing that
    changed is which (tier, verb, cell) got handed to it.

    LOOK-FOR STAYS ON: `take_meat` only becomes legal once meat is ALREADY
    sitting on a counter `_core` has seen, so there is no ordinary-ladder
    subtask at all for the window where the robot is still CARRYING it --
    "look for meat" is the genuine subtask for that window. What is turned
    off is only the AUTO-PICK: `_choose()` would normally call
    `_look_candidate()` to silently swap the ladder's own top pick for a
    look-for that strictly beats it, and that is exactly the kind of
    override a study interface must never do behind a participant's click.
    So `_look_candidate` is neutralised once, in __init__, to always return
    None, and `legal_menu()` reads the sighting book directly to offer
    "look for X" as its own row instead.

    Shadowing works because both methods are looked up on the INSTANCE
    first: `self._core._ladder_decide = <a plain function>` shadows the
    class method for this one object, and a function found in an instance's
    own __dict__ is called exactly as passed -- no `self` is bound the way a
    class method's would be -- so `lambda s, _f=forced: _f` receives `state`
    as `s` and ignores it. `_ladder_decide` is reassigned every tick, before
    every `action()` call, so it is always this tick's pick and never a
    stale one; `_look_candidate` needs setting only once, since it is always
    the same constant None.
    """

    def __init__(self, mdp, fov, agent_index, forget_horizon, seed,
                enable_look_for=True):
        self._core = LimitedVisionHuman(mdp, fov, agent_index=agent_index,
                                        forget_horizon=forget_horizon,
                                        seed=seed,
                                        enable_look_for=enable_look_for)
        self._core._look_candidate = lambda state, base: None
        self.chosen = None     # (verb, cell) picked from the menu, or None
        self.legal = {}        # verb -> (tier, dist, cell): this tick's menu
        self._ranked = []
        self._look_commit = {}     # item -> counter cell; see legal_menu()

    @property
    def view(self):
        return self._core.view

    @property
    def last_subtask(self):
        """(TIER_NAME, verb, cell) for whatever `chosen` most recently drove
        through `_core.action()` -- set there, not tracked separately, so
        this can go stale (keep showing a past subtask) once `chosen` goes
        back to None. Callers that want "is anything actually happening
        right now" should check `chosen` first, same as `step()` does."""
        return self._core.last_subtask

    def prime(self, state):
        """Build the first menu, without spending a tick. Needed only right
        after reset() -- step() keeps the menu current on every later tick."""
        self._refresh_menu(state)

    def _refresh_menu(self, state):
        self._ranked, self.legal = legal_menu(self._core, state, self._look_commit)

    def choose(self, verb, cell=None):
        """Called from the UI when the participant clicks a menu row."""
        self.chosen = (verb, cell)

    def cancel(self):
        self.chosen = None

    def _forced(self):
        """This tick's override for `_core._ladder_decide`, or None if the
        participant's pick is not (or no longer) strictly legal.

        Returns a 5-tuple to match what `_core.action()` unpacks as
        `tier, _, _, cell, verb` -- the two middle slots are `rank()`'s
        demotion/distance, which `action()`'s own dispatch never reads (see
        its unpacking), so 0 is a correct filler and not a loss of
        information: this class does not use rank()'s robot-aware demotion
        at all, since the participant already committed to one exact
        (verb, cell) instead of letting a ladder rank several.
        """
        if self.chosen is None:
            return None
        verb, cell = self.chosen
        if verb == "explore":
            return (T_EXPLORE, 0, 0, None, "explore")
        if verb.startswith(LOOK_PREFIX):
            # look-for rows are not in `_ranked` -- common/tasks.py's ladder
            # does not know this verb at all, see human/limited_vision_human
            # .py's THE LOOK-FOR -- so re-check the SAME way legal_menu()
            # built it: still tracked, and still pointed at the cell the
            # participant actually clicked (the sweep can retarget a
            # different counter tick to tick as candidates get cleared; if
            # it has, this is correctly "no longer the thing you picked"
            # rather than silently walking somewhere else).
            entry = self.legal.get(verb)
            if entry is None or entry[2] != cell:
                return None
            reset_explore_patience(self._core, verb)
            tier, dist, c = entry
            return (tier, 0, dist, c, verb)
        for tier, v, c in self._ranked:
            if v == verb and c == cell:
                reset_explore_patience(self._core, verb)
                return (tier, 0, 0, c, v)
        return None

    def step(self, state):
        """One env tick. Returns (action, info, finished).

        `finished` means the subtask that was chosen coming into this call
        just ran out (completed) or was made illegal by something that
        happened elsewhere (the robot took it, the grill finished, ...), so
        the menu needs a fresh pick before anything moves again. `state` is
        the TRUE, CURRENT env state -- there is no belief to fall behind any
        more, so this is detected on the very tick it becomes true, not one
        tick later the way a belief-gated version of this would have to.

        `_refresh_menu` is called first, unconditionally, so `_forced()`
        always judges the participant's pick against THIS tick's ground
        truth -- not last tick's, and not action()'s own (unused) internal
        `_ladder_decide` call, which reads no state this class has not
        already read here. It is also the only place `core.observe()` gets
        called on a tick where nothing is chosen, which is what keeps the
        LOOK-FOR sighting book current even while you are standing still
        deciding what to click next.
        """
        self._refresh_menu(state)
        forced = self._forced()
        if forced is None:
            self._core.t += 1
            finished = self.chosen is not None
            self.chosen = None
            return Action.STAY, {"subtask": None}, finished
        self._core._ladder_decide = lambda s, _f=forced: _f
        act, info = self._core.action(state)
        return act, info, False
