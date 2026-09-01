"""Approximate human model that ROLLS FORWARD one committed hypothesis --
from the outside, never seeing the real click.

Separate from human_behavior.py on purpose. HumanBehavior IS the study
participant: every action it takes is the actual click they made. This file
is what user_study/robot/fov_posterior.py's best_shadow() hands to
user_study/robot/filter.py to simulate several ticks forward, when there is
no real click to read -- that per-tick inference is done elsewhere, by
SubtaskFOVPosterior.update() reweighting every legal candidate directly
against the real action. By the time best_shadow() is asked for one of
these, the posterior has already DECIDED which subtask it believes most:
this class's only job is to faithfully carry that single decision forward,
tick by tick, deterministically -- not to re-decide anything.

THE CANDIDATE SET is human_behavior.legal_menu()'s ordinary ladder PLUS its
look-for rows PLUS "explore" -- the same three things HumanBehavior offers a
real participant, reused rather than recomputed a second way, so a shadow
can never predict a subtask a real participant's own menu would not have
offered. See ApproximateHumanModel's docstring for exactly how a tick's held
subtask is picked and carried through LimitedVisionHuman's own dispatch.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # .../user_study/human
USER_STUDY = os.path.dirname(HERE)                            # .../user_study
MISHA = os.path.dirname(USER_STUDY)                            # .../misha
sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, MISHA)

from common.tasks import T_EXPLORE, TIER_NAME                  # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,    # noqa: E402
                                        FORGET_HORIZON,
                                        _PREFIX as LOOK_PREFIX)
from robot.nominal_policy.baselines import subtask_pi           # noqa: E402
from user_study.human.human_behavior import (legal_menu,         # noqa: E402
                                             reset_explore_patience)


class ApproximateHumanModel:
    """Deterministically carries forward ONE committed subtask hypothesis --
    built from `_core`'s own genuinely-earned belief, same as HumanBehavior,
    but choosing by ARGMAX over `subtask_pi` instead of reading a click.

    Composition, not a subclass, over the SAME `_core` = plain
    LimitedVisionHuman that HumanBehavior wraps -- perception, movement and
    the sighting book, all unmodified. `_look_candidate` is neutralised the
    same way and for the same reason: `action()` below is the ONLY thing
    that gets to choose this shadow's subtask, and `_choose()`'s own
    auto-pick would otherwise silently overrule that choice with the
    ladder's ordinary look-for preemption, which is not what this class is
    modelling.

    THE CANDIDATE SET is human_behavior.legal_menu()'s ordinary ladder PLUS
    its look-for rows PLUS "explore" -- the same three things HumanBehavior
    offers a real participant, reused rather than recomputed a second way,
    so a shadow can never predict a subtask a real participant's OWN menu
    would not have offered. "explore" is priced as (T_EXPLORE, "explore",
    pos) -- see action()'s own comment for why using the agent's own
    position as a placeholder target correctly prices it far below any real
    candidate without needing subtask_pi to know anything special about it.
    An earlier version of this class only offered "explore" as the fallback
    when the candidate set was empty (robot/nominal_policy/baselines.py's
    `_BaseRobot.action()`'s own convention for STAY, which is right for a
    baseline robot but was wrong here): once ANY ordinary subtask became
    legal, the shadow could never predict "explore" again for the rest of
    the episode no matter what the real participant actually clicked,
    confirmed live -- a human who explored for 150 straight ticks still had
    the shadow confidently commit to "get_onion" for 48 of them, purely
    because that verb had become legal in the shared belief.

    THE PICK IS max(subtask_pi(candidates, pos, orient, walk), key=...) --
    the single highest-value candidate, never sampled. An earlier version
    drew stochastically from that same distribution (a genuine sticky-hold
    kernel, reused unchanged from _BaseRobot.action()) on the reasoning that
    a real participant's own uncertainty should show up as spread in the
    shadow's behavior. That reasoning belongs to the per-tick INFERENCE
    problem (SubtaskFOVPosterior.update(), which does track every candidate
    at once, weighted, exactly for this reason), not to this class's job:
    once best_shadow() has already picked the posterior's single best
    hypothesis, sampling a DIFFERENT one for THIS class to roll forward
    injects noise with no evidentiary basis -- the same forecast run twice
    from the same state could score a plan differently for no real reason.
    "Forced" is still real, though: a redraw still happens the moment the
    held pick falls out of the candidate set or a strictly more urgent tier
    appears (the world changed under it, not a change of mind), and in that
    case the new pick is ALSO the argmax, not a fresh sample.

    DISPATCH IS UNMODIFIED LimitedVisionHuman.action(), exactly the way
    HumanBehavior hands it a click: once a subtask is picked, `action()`
    shadows `_core._ladder_decide` with that ONE (tier, verb, cell) and
    calls `_core.action(state)`, so walking, turning, INTERACTing, PATIENCE
    and `_core.t` bookkeeping are the same tested code path -- nothing here
    reimplements movement.
    """

    def __init__(self, mdp, fov, agent_index=1, forget_horizon=FORGET_HORIZON,
                seed=0, enable_look_for=True):
        self._core = LimitedVisionHuman(mdp, fov, agent_index=agent_index,
                                        forget_horizon=forget_horizon,
                                        seed=seed,
                                        enable_look_for=enable_look_for)
        self._core._look_candidate = lambda state, base: None
        self.committed = None      # (verb, cell), NOT the tier -- see _find
        self.last_subtask = None
        self._look_commit = {}     # item -> counter cell; see legal_menu()

    @property
    def t(self):
        return self._core.t

    def observe(self, state):
        return self._core.observe(state)

    def _find(self, cand, committed):
        """The current candidate list's entry for a (verb, cell), or None.

        Matched on (verb, cell), NOT the tier -- a subtask whose tier moved
        because the world moved is the same job, and treating it as gone
        would force a redraw every time a pot finished somewhere. Mirrors
        _BaseRobot._find() in baselines.py exactly.
        """
        if committed is None:
            return None
        for sub in cand:
            if (sub[1], sub[2]) == committed:
                return sub
        return None

    def action(self, state):
        core = self._core
        ranked, best = legal_menu(core, state, self._look_commit)
        me = state.players[core.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = core.view.walkable | {pos}

        # "explore" is ALWAYS a candidate, matching HumanBehavior's own
        # always-clickable row -- not a last-resort fallback used only when
        # `cand` would otherwise be empty (see the class docstring's OLD
        # explanation, now wrong, and the module docstring for why: a
        # participant clicking explore while other rows are legal is a
        # real, common choice, and a shadow that can never predict it once
        # anything else becomes legal cannot represent that participant at
        # all -- confirmed live: a human who explores for 150 straight
        # ticks still had the shadow confidently commit to "get_onion" for
        # 48 of them, purely because that verb became legal in the shared
        # belief. Priced as (T_EXPLORE, "explore", pos) -- `pos`, the
        # agent's OWN current cell, is a placeholder target that only
        # matters to subtask_pi's own arithmetic: T_EXPLORE is already the
        # single WORST (highest-numbered, least urgent) tier there is, so
        # the log(TIER_GAIN)*(T_EXPLORE - tier) term that gives every real
        # candidate its edge is exactly 0 for this one -- structurally
        # cheap, never zero-probability. `_core.action()`'s own dispatch
        # never reads `cell` for the "explore" verb (see human/limited_
        # vision_human.py's action(): `if verb == "explore": act =
        # self._explore(pos, orient)`), so the placeholder is inert
        # everywhere except pricing.
        cand = (list(ranked)
               + [(tier, verb, cell) for verb, (tier, _dist, cell) in best.items()
                  if verb.startswith(LOOK_PREFIX)]
               + [(T_EXPLORE, "explore", pos)])

        # subtask_pi's own default beta (robot/nominal_policy/baselines.py)
        # is irrelevant to what follows -- beta is documented there as "a
        # sharpness knob on a fixed distribution, not a second ranking
        # rule," so max(pi, key=pi.get) picks the same candidate at any
        # beta > 0. Passed at its default rather than threaded through as a
        # constructor argument nobody needs any more.
        pi = subtask_pi(cand, pos, orient, walk)

        if not pi:
            held = (T_EXPLORE, "explore", pos)
        else:
            prev = self._find(cand, self.committed)
            top_tier = min(c[0] for c in cand)
            # FORCED, not RECONSIDERED: a redraw happens only when the held
            # pick fell out of the candidate set or a strictly more urgent
            # tier appeared -- the world changed under it, never a spontaneous
            # change of mind. And the redraw is always the argmax: this class
            # commits to and rolls forward the posterior's single best
            # hypothesis, never a sampled alternative -- see the class
            # docstring's THE PICK section for why.
            held = max(pi, key=pi.get) if prev is None or top_tier < prev[0] else prev

        tier, verb, cell = held
        self.committed = None if verb == "explore" else (verb, cell)
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        # See reset_explore_patience()'s own docstring: shadowing
        # `_ladder_decide` below (needed to force THIS tick's picked subtask
        # through) skips its body, which is also where the PATIENCE clock
        # would normally get reset the instant real work is chosen -- this
        # was previously just a comment ("same reasoning as ... PATIENCE
        # reset") with no call behind it.
        reset_explore_patience(core, verb)

        # _ladder_decide's contract is the 5-tuple _core.action() unpacks as
        # `tier, _, _, cell, verb` (see human/limited_vision_human.py's
        # rank()) -- the two middle slots are demotion/distance, which
        # action()'s own dispatch never reads, so 0 is a correct filler and
        # not a loss of information; `held` above is the plain 3-tuple
        # subtask_pi()/legal_menu() both deal in.
        forced = (tier, 0, 0, cell, verb)
        core._ladder_decide = lambda s, _f=forced: _f
        return core.action(state)

    def clone(self):
        """A faithful copy sharing the mdp, mirroring
        human/limited_vision_human.py's own LimitedVisionHuman.clone() --
        needed because fov_posterior.py's best_shadow() hands filter.py's
        _forecast() one of these to roll forward several ticks, and
        _forecast() clones it again per candidate branch so no branch can
        corrupt the shadow shared with the others -- each clone must carry
        the CURRENT state (committed subtask, belief, sighting book), not
        restart fresh at t=0.

        `_core.clone()` copies every field it names by hand and nothing
        else, so the INSTANCE-level `_look_candidate` override made in
        __init__ does not survive it automatically -- reapplied here, or a
        cloned shadow's look-for could silently start auto-picking again.
        """
        c = type(self).__new__(type(self))
        c._core = self._core.clone()
        c._core._look_candidate = lambda state, base: None
        c.committed = self.committed
        c.last_subtask = self.last_subtask
        c._look_commit = dict(self._look_commit)
        return c

    def set_agent_index(self, i):
        self._core.set_agent_index(i)
        self.committed = None

    def set_mdp(self, mdp):
        self._core.set_mdp(mdp)
