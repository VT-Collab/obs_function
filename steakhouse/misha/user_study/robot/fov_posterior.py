"""P(theta, subtask | the human's actions), for a human who does not run a
ladder.

Companion to robot/filter/core/fov_posterior.py (untouched, read-only
reference), not a replacement for it: that file's FOVPosterior scores ONE
deterministic LimitedVisionHuman shadow per candidate cone, which is right
for a human whose actions come from a ladder argmax and wrong for this
study's human, whose actions come from a click among several reasonable
menu rows -- see user_study/human/approximate_human_model.py's module
docstring for why a single deterministic guess per cone cannot represent
that.

SECOND VERSION OF THIS FILE, AND WHY. The first version kept a small
population of independent ApproximateHumanModel PARTICLES per cone, each
sampling its own subtask via a sticky kernel (hold with probability rho,
else redraw) -- "more clones", same per-particle soft-gate likelihood as
robot/filter/core/fov_posterior.py. It worked, but it was SLOW to notice a
genuine subtask switch: a particle's own decision to reconsider was a blind
~5%-per-tick coin flip completely decoupled from whether its predictions had
been RIGHT lately, and even a particle that DID reconsider would usually
just re-draw its old, objectively-higher-value pick again, since
subtask_pi's value function has nothing to do with what the human is
actually doing right now. Measured live: a human who fetched meat for 6
ticks and then hard-switched to exploring and stayed there had EVERY ONE of
8 particles still saying "fetch meat" 41 ticks later.

THIS VERSION reweights EVERY currently-legal subtask, for every cone, EVERY
TICK -- not just whichever one a sampled particle happens to be holding.
`self.weights[f]` is a dict {(tier, verb, cell): weight}, and each tick:
every candidate's predicted action is computed fresh (the SAME "what would
pursuing this subtask do right now" question, just asked of ALL of them
instead of one sampled draw), scored against the real action with the exact
same alpha/miss soft-gate as before, and multiplied into that candidate's
OWN running weight. A subtask that falls out of the current legal set (the
human moved on, it became saturated, whatever) is dropped outright -- no
weight for it survives into the next tick's dict at all. The moment "get
onion" stops matching and "explore" starts, "explore"'s weight starts
climbing on THAT tick, not whenever a random reconsideration happens to
land on it.

No particle population, no resampling, no effective-sample-size tracking --
none of that machinery is needed once every hypothesis is reweighted
directly instead of sampled, so it is gone rather than kept as a second
code path. ONE observer per fov (a plain LimitedVisionHuman) still tracks
that cone's belief, same as before; what changed is that a whole candidate
set is priced against it each tick instead of one particle's sampled draw.

THIRD REVISION: A BRAND NEW CANDIDATE'S STARTING WEIGHT IS UNIFORM, NOT
subtask_pi-PRICED. The second version above priced a just-legal candidate's
first weight with subtask_pi's value(tau)**beta softmax -- the same
machinery play.py's baseline robots use to CHOOSE -- reused here as a
Bayesian prior on the reasoning that a rational human is more likely to go
after a high-value, urgent, nearby subtask. Measured live, that reasoning
was too strong to serve as a PRIOR: at BETA=8.0 the softmax is sharp enough
that the instant a human picked up an onion, "load_board" (the one clearly
best remaining option) jumped to weight 0.9999 that SAME tick, before that
tick's own action evidence could meaningfully move anything -- and it did
this even while the human was, in that specific run, still just exploring,
nowhere near the board. The posterior was reading what the value function
made newly POSSIBLE as if it were evidence about what the human was
actually DOING. So a brand-new candidate now starts from an equal share of
its cone's mass among however many are currently legal, and only genuinely
accumulated per-tick likelihood (see LEAK below) is allowed to separate one
from another -- the same standard every already-tracked candidate's weight
has always had to meet.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))              # .../user_study/robot
USER_STUDY = os.path.dirname(HERE)                              # .../user_study
MISHA = os.path.dirname(USER_STUDY)                              # .../misha
sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, MISHA)

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.tasks import T_EXPLORE, TIER_NAME                      # noqa: E402
from human.limited_vision_human import LimitedVisionHuman          # noqa: E402
# Read-only reuse of the original posterior's own constants, so a cone
# hypothesis set or action-space assumption can never quietly drift between
# the two filters.
from robot.filter.core.fov_posterior import FOVS, N_ACTIONS         # noqa: E402
from user_study.human.approximate_human_model import ApproximateHumanModel  # noqa: E402
from user_study.human.human_behavior import (legal_menu, LOOK_PREFIX,  # noqa: E402
                                             reset_explore_patience)


class SubtaskFOVPosterior:
    """P(theta, subtask | a_H_{0:t}) by soft-gate likelihood, exact -- a
    discrete Bayes filter over the JOINT (cone, subtask) space, not a
    sampled approximation of it.

    Public surface matches robot/filter/core/fov_posterior.py's FOVPosterior
    -- `.p` (dict: fov -> probability, summing to 1), `.update(state,
    human_action)`, `.map_fov()` -- so anything that reads a posterior does
    not need to know which filter built it. `.best_shadow(fov)` is not part
    of that original surface; see its own docstring.

    `self.observers[f]` is ONE plain LimitedVisionHuman per cone, used only
    for perception (belief, discovery, the look-for sighting book) -- never
    asked to decide anything itself. `self.weights[f]` is {(tier, verb,
    cell): weight}, all `sum(len(w) for w in weights.values())` entries
    sharing ONE normalisation (they sum to 1 together, not per-fov), and
    `self.p[f]` is simply the sum of fov f's own entries -- the fov-level
    posterior is the MARGINAL over the (fov, subtask) joint this class
    actually tracks.
    """

    def __init__(self, mdp, fovs=FOVS, alpha=0.9, human_index=1, seed=0,
                human_kw=None):
        self.mdp, self.fovs, self.alpha = mdp, tuple(fovs), alpha
        self.human_index = human_index
        kw = dict(human_kw or {})
        self.human_kw = kw
        self.observers = {
            f: LimitedVisionHuman(mdp, f, agent_index=human_index,
                                  seed=seed * 10007 + f * 101, **kw)
            for f in self.fovs
        }
        self._look_commit = {f: {} for f in self.fovs}
        self.weights = {f: {} for f in self.fovs}
        self.p = {f: 1.0 / len(self.fovs) for f in self.fovs}
        self.predicted = {}    # fov -> {subtask: predicted action}, this tick

    def update(self, state, human_action):
        """Reweight every currently-legal subtask, for every cone, against
        `human_action`, drop whatever fell out of legality, renormalise.
        Call this ONCE per tick, after the robot has decided -- same
        contract as the original FOVPosterior, for the same reason:
        `core.observe()` is what advances an observer's own perception, and
        a second call in the same tick would observe it twice.
        """
        miss = (1.0 - self.alpha) / (N_ACTIONS - 1)
        raw_by_fov = {}
        self.predicted = {}

        for f in self.fovs:
            core = self.observers[f]
            core.t += 1
            ranked, best_menu = legal_menu(core, state, self._look_commit[f])
            me = state.players[core.agent_index]
            pos, orient = tuple(me.position), tuple(me.orientation)
            walk = core.view.walkable | {pos}

            # "explore" is always a candidate, matching HumanBehavior's own
            # always-clickable row.
            cand = (list(ranked)
                   + [(tier, verb, cell) for verb, (tier, _d, cell)
                      in best_menu.items() if verb.startswith(LOOK_PREFIX)]
                   + [(T_EXPLORE, "explore", pos)])

            # TRACKING key vs PRICING entry: every ordinary/look-for verb's
            # cell is a fixed station, so its identity in `self.weights[f]`
            # is stable tick to tick and its weight can accumulate evidence
            # via repeated likelihood multiplies. "explore" has no station --
            # its only candidate cell is the agent's OWN current position,
            # which changes on every step. Keying its weight on that `pos`
            # would make "explore" look like a brand-new candidate
            # instead of ever accumulating -- confirmed live: this bug alone
            # was enough to visibly destabilise fov-level accuracy under a
            # human who explores a lot (divide/seed=1 dropped to 31%
            # fov-correct before this fix). `stable()` collapses explore's
            # cell to a fixed None for tracking/lookup purposes only;
            # action-prediction below still uses the real `cand` entries
            # with the real `pos`.
            def stable(sub):
                tier, verb, cell = sub
                return (tier, verb, None) if verb == "explore" else sub

            cand_stable = [stable(s) for s in cand]

            # PRIOR IS UNIFORM, NOT VALUE-WEIGHTED -- THIS IS THE WHOLE
            # POINT OF A FILTER THAT WATCHES EVIDENCE. An earlier version
            # seeded a brand-new candidate's starting weight from
            # subtask_pi(cand, ...) -- the same value(tau)**beta softmax
            # baseline robots use to CHOOSE, reused here as a Bayesian
            # PRIOR on the reasoning that a rational human is more likely to
            # pursue a high-value, urgent, nearby subtask. In practice, at
            # BETA=8.0 that softmax is sharp enough to be near-certainty on
            # its own: confirmed live, the instant a human picked up an
            # onion, "load_board" (now the only decent option) jumped to
            # weight 0.9999 within that SAME tick, before any of the tick's
            # own action evidence could meaningfully move it, and even while
            # the human was actually still just exploring (not moving toward
            # the board at all). That is the posterior CHEATING -- reading
            # what became newly POSSIBLE off the value function instead of
            # actually watching what the human DOES next. So every
            # currently-legal candidate starts from an equal share of the
            # fov's mass; only genuinely accumulated per-tick likelihood
            # (below) is allowed to separate them, the same way every
            # already-tracked candidate's weight has always had to earn its
            # standing.
            #
            # LEAK: every tick, blend a small slice of that same uniform
            # share back into each candidate's accumulated weight, before
            # this tick's likelihood. Pure multiplicative reweighting (raw =
            # old*lik forever) is numerically irreversible: a candidate that
            # is legal but wrong for many consecutive ticks (which "explore"
            # always is, right up until the tick it becomes correct, since
            # it is the one candidate that never falls out of legality and
            # so never gets the fresh-prior restart an ordinary subtask gets
            # by being dropped and re-added) gets ground down by repeated
            # `miss` multiplies to something like 1e-58 -- and recovering
            # from that would take over a thousand consecutive correct
            # ticks even at alpha=0.9, i.e. never, within any real episode.
            # LEAK is the standard fix for this in any recursive Bayes/HMM
            # filter (a "forgetting factor"/process-noise term) -- it does
            # not reintroduce particles or resampling, and does not change
            # the drop-when-illegal rule below: a candidate missing from
            # `cand_stable` this tick still gets nothing, dropped exactly as
            # before. It only keeps a legal-but-losing candidate's floor
            # high enough that a few genuinely matching ticks can pull it
            # back into contention instead of it being mathematically dead.
            LEAK = 0.03
            fov_mass = self.p.get(f, 1.0 / len(self.fovs))
            prior = fov_mass / len(cand_stable) if cand_stable else 0.0
            old = self.weights.get(f, {})
            raw = {}
            for s in cand_stable:
                base = old.get(s, prior)
                raw[s] = (1.0 - LEAK) * base + LEAK * prior

            # PATIENCE RESET, before pricing explore. `core._explore()` is
            # called directly here, outside `core.action()`/`_ladder_decide()`
            # entirely, so it never gets the reset `_ladder_decide()` itself
            # would normally apply the instant real work is found -- see
            # reset_explore_patience()'s own docstring for why that left this
            # fov's own "explore" prediction built on a PATIENCE clock that
            # had climbed the whole time it was tracking some OTHER
            # hypothesis, diverging from what a freshly-exploring real human
            # would do. The observer never chooses anything itself, so the
            # signal used here is this fov's own CURRENT best belief (as of
            # the end of last tick, before this tick's reweighting below):
            # as long as we currently think the human is doing genuine work,
            # keep the clock at a fresh 0, exactly mirroring how a real
            # human's own clock stays pinned at 0 for as long as they keep
            # being handed a genuine pick.
            reset_explore_patience(core, self.best_subtask(f)[1])

            # explore's predicted action is computed ONCE per fov per tick,
            # mutating that fov's observer's own explore state (frontier,
            # PATIENCE) exactly like a continuously-exploring hypothesis
            # would evolve -- geo.step_towards below is stateless and reads
            # none of that, so computing this first or last makes no
            # difference to anything else in this loop.
            explore_act = core._explore(pos, orient)
            preds = {}
            for s, s_key in zip(cand, cand_stable):
                _tier, verb, cell = s
                if verb == "explore":
                    predicted = explore_act
                else:
                    move, arrived = geo.step_towards(walk, pos, orient, cell)
                    predicted = Action.INTERACT if arrived else (move or Action.STAY)
                preds[s_key] = predicted
                lik = self.alpha if predicted == human_action else miss
                raw[s_key] *= lik

            raw_by_fov[f] = raw          # drops anything not in `cand_stable`
            self.predicted[f] = preds

        total = sum(sum(w.values()) for w in raw_by_fov.values())
        if total <= 0:
            n = sum(len(w) for w in raw_by_fov.values()) or 1
            raw_by_fov = {f: {s: 1.0 / n for s in w} for f, w in raw_by_fov.items()}
            total = 1.0
        self.weights = {f: {s: x / total for s, x in w.items()}
                        for f, w in raw_by_fov.items()}
        self.p = {f: sum(w.values()) for f, w in self.weights.items()}
        return self.p

    def map_fov(self):
        return max(self.p, key=self.p.get)

    def best_subtask(self, fov):
        """The single highest-weight (tier, verb, cell) for `fov`, or the
        synthetic explore triple if nothing is tracked yet."""
        w = self.weights.get(fov)
        if not w:
            return (T_EXPLORE, "explore", None)
        return max(w, key=w.get)

    def best_shadow(self, fov):
        """A ROLLOUT-CAPABLE ApproximateHumanModel, seeded from this fov's
        current best subtask hypothesis and its observer's real, current
        belief -- for callers (user_study/robot/filter.py) that need to
        simulate several ticks forward, not just read a single tick's
        reweighted belief the way this class's own update() does. The
        discrete filter above never needs one of these itself; only
        external forecasting does, which is why it is built lazily here
        rather than kept running as a second, redundant model.

        The shadow rolls forward DETERMINISTICALLY from exactly this pick --
        see ApproximateHumanModel's own docstring for why it no longer
        samples.
        """
        core = self.observers[fov]
        tier, verb, cell = self.best_subtask(fov)
        shadow = ApproximateHumanModel.__new__(ApproximateHumanModel)
        shadow._core = core.clone()
        shadow._core._look_candidate = lambda state, base: None
        shadow.committed = None if verb == "explore" else (verb, cell)
        shadow.last_subtask = (TIER_NAME[tier], verb, cell)
        shadow._look_commit = dict(self._look_commit.get(fov, {}))
        return shadow
