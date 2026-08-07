"""Joint Bayesian inference of the human's FOV and current subtask.

The paper's b_t: a joint posterior over the hidden states (theta, tau_t),
maintained from the observed STATES and ACTIONS only.

    predict   bbar(theta,tau) = SUM_tau' T_theta(tau | tau', s) b(theta,tau')
    update    b(theta,tau)   ~=  L_theta(a | s, tau) * bbar(theta,tau)
    FOV       P(theta)        = SUM_tau b(theta,tau)

theta is static; tau is a hidden Markov state. One shadow LimitedVisionSteakHuman
per candidate theta carries that hypothesis's FOV-gated decaying beliefs, so both
kernels are the ones a human with FOV=theta would actually be running.

NO ORACLE. The filter never sees the true FOV or the true subtask. Its only
inputs are the world state and the human's emitted action. (update() previously
took the human's internal info["subtask"] label - that was a cheat, and the
subtask is now inferred instead.)

tau = (current, sampled), mirroring the human's two commitment layers:
  current  self._current, the action()-level errand lock; None after arrival
  sampled  self._sampled, the sampler's sticky draw; survives held-object
           excursions and is NOT cleared on errand completion

Three things this file must get right, all easy to break by "tidying":

1. T_theta composes BOTH layers in action()'s order. subtask_distribution() is
   an exact closed form of decide(), but decide() is only action()'s INNER half;
   the COMMITTED errand lock is checked first. Using subtask_distribution alone
   models a human that does not exist - COMMITTED and SAMPLING_SUBTASKS share
   only 4 of 13 subtasks, and _commit_still_useful / _sample_still_helpful
   disagree wherever pickup_plate or pickup_washed_plate is live. (pickup_meat and
   pickup_onion are the only two where they agree, and they are the most
   frequent - so this bug hides from aggregate accuracy. Regression-test on
   pickup_plate / pickup_washed_plate / pickup_steak / dump_item.)

2. The decay clock must be driven by hand. self.t is incremented ONLY on the
   last line of action(); observe() never touches it. A filter that observes but
   never acts leaves every shadow at t=0, making the decay test
   `self.t - seen_at > forget_horizon` permanently false - nothing decays, and
   the shadows model perfect memory instead of limited vision.

3. Normalisation is JOINT over (theta, tau). Per-theta normalisation divides out
   the predictive-likelihood ratio that carries all the FOV evidence.
"""
import math
from collections import defaultdict

from overcooked_ai_py.mdp.overcooked_mdp import Action

from fov.human.agent.limited_vision_human import (
    LimitedVisionSteakHuman, SAMPLING_SUBTASKS, COMMITTED)
from fov.human.planning.steak_planner import SteakMotionPlanner

# EXACT by default: no smoothing. Execution is deterministic given
# (theta, tau, beliefs), so L is a hard delta and a mismatch is real evidence
# against (theta, tau), not noise. Safe because the kernel is an exact
# description of the human: the TRUE theta always retains at least one tau that
# explains the observed action, so it can never be annihilated. Verified - with
# eps=0 the true FOV survived 72/72 episodes with 0 discarded ticks, and accuracy
# was identical to eps=0.02, i.e. the smoothing was inert. Left as a parameter
# only for experiments with a human that is NOT this model.
EPSILON = 0.0
PRUNE = 1e-12       # numerical hygiene: drop states that cannot come back


def fov_prior(candidate_fovs):
    return {f: 1.0 / len(candidate_fovs) for f in candidate_fovs}


class SamplingBayesFOVInference:
    """Joint (FOV, subtask) filter driven by observed actions."""

    def __init__(self, mdp, mlp, candidate_fovs, human_agent_index=1,
                 prior=None, epsilon=EPSILON):
        self.candidate_fovs = list(candidate_fovs)
        self.epsilon = epsilon
        self.agent_index = human_agent_index
        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior

        planner = SteakMotionPlanner(mdp, mlp)
        self.shadows = {f: LimitedVisionSteakHuman(mdp, f, planner,
                                                   agent_index=human_agent_index)
                        for f in self.candidate_fovs}
        self.n_actions = len(Action.ALL_ACTIONS)

        # b = the joint belief over (fov, tau): "which cone does this human
        # have, and what are they doing right now?"  FOV is fixed for the whole
        # episode; tau changes every tick.
        #
        #     (f, (current, sampled))
        #      |    |        `-- self._sampled : the LINE of work I chose
        #      |    `-- self._current : the STEP I am executing right now
        #      `-- the FOV hypothesis
        #
        # Both are just a subtask NAME (a string) or None. Neither is a
        # distribution. "Line" is not a separate vocabulary - there is no
        # "steak" token; sampled=pickup_meat simply MEANS "I am on the meat
        # line", and it is tested that way (_sample_still_helpful asks "is the
        # pot still empty?", not "am I still walking to the meat").
        #
        # VALUES ------------------------------------------------------------
        #
        # sampled   7: None, or one of the 6 you can freely CHOOSE
        #              pickup_meat   pickup_onion     chop_onion
        #              pickup_plate  heat_washed_plate   pickup_washed_plate
        #
        # current  18: None, or ANY subtask - those 6, plus 11 you never choose
        #              and only get HANDED:
        #              hands full -> drop_meat drop_onion drop_plate dump_item
        #                            deliver pickup_steak pickup_garnish
        #              dont know  -> check_pot check_board check_sink explore
        #                            (and "wait")
        #
        # sampled's vocabulary is a SUBSET of current's, and the 11 extras are
        # exactly the FORCED ones - which is why they never overwrite sampled.
        #
        # COMBINATIONS (29 of the 126 ever occur) ----------------------------
        #
        #     (None,        None)          episode start: nothing chosen, nothing
        #                                  in progress
        #     (pickup_meat, pickup_meat)   doing exactly what I chose
        #     (drop_meat,   pickup_meat)   hands full - still on the meat line
        #     (check_pot,   pickup_onion)  off looking - still on the onion line
        #     (None,        pickup_meat)   step finished, line still stands
        #     (anything,    None)          no standing choice: never made one, or
        #                                  it got invalidated (pot filled, robot
        #                                  took it)
        #     NEVER: sampled holding a forced-only name like drop_meat.
        #
        # WHY TWO -----------------------------------------------------------
        #
        # current is wiped the instant a step finishes, so it cannot answer
        # "what was I working on?". Only sampled survives that, and the next
        # decision needs it: still worth doing -> continue; not -> redraw the
        # whole softmax. Two completely different distributions, so one variable
        # is not enough.
        #
        # current is a lock, but one that BREAKS ON SIGHT: seeing the teammate
        # carry the thing you are fetching, or seeing your destination already
        # taken, drops the errand - while a belief that merely DECAYED to unknown
        # does not (forgetting is not observing). Both triggers are FOV-gated: a
        # wide cone re-plans mid-trip, a narrow cone walks the wasted errand to
        # the end. That asymmetry IS the FOV effect, and it is what the filter
        # reads the cone off of.
        #
        # t=0: nothing chosen, nothing in progress - exactly what reset() leaves
        # behind - so every hypothesis puts ALL its prior mass on (None, None),
        # and marginalising over tau hands back the FOV prior untouched.
        self.b = {(f, (None, None)): self.prior[f] for f in self.candidate_fovs}

        # The decay clock we drive into every shadow by hand. The human advances
        # self.t on the last line of action(), which the filter never calls, so
        # without this counter every shadow would sit at t=0 forever and nothing
        # would ever be forgotten. Counts COMPLETED ticks.
        self.tick = 0

        # Diagnostic: ticks where the hypotheses actually disagreed about how
        # likely the observed action was. When they all agree, every theta gets
        # multiplied by the same number and the FOV marginal does not move - so
        # this counts how many ticks carried real evidence. A low value means the
        # layout is weakly identifiable, NOT that the filter is broken.
        self.n_informative = 0

        # Diagnostic: ticks thrown away because a shadow raised, or because no
        # (theta, tau) could explain the action at all. Should stay at 0; if it
        # climbs, the kernel has drifted out of sync with the human model.
        self.n_skipped = 0

        # P(tau_t), marginalised over theta, for the tick update() JUST finished.
        # Read it AFTER update(state, action) and it describes THAT tick.
        #
        # "last_" means "from the most recent update() call", NOT "the previous
        # tick" - easy to misread. Reading it BEFORE calling update() hands you
        # the previous tick's answer instead. Measured on steak_mid_1: read after
        # update it matches the true current subtask 1.000 of the time; read
        # before update, 0.784 (which is just how often the human happened to
        # stay on the same subtask two ticks running).
        #
        # This is the quantity the old oracle used to be HANDED for free.
        # Holding it here is how we show it is now inferred instead.
        self.last_subtask_post = {}

    # -- kernels ------------------------------------------------------------

    def _transition(self, sh, state, tau, forced):
        """T_theta(. | tau, s) as {subtask: prob}. Reproduces action()'s two
        layers in order. Second return value: whether self._sampled would have
        been left untouched (errand lock held, or the forced-by-held branch)."""
        current, sampled = tau
        if current in COMMITTED and sh._commit_still_useful(current):
            return {current: 1.0}, True
        return sh.subtask_distribution(state, sampled), forced

    def _execute(self, sh, state, subtask, explore_act, memo):
        """L_theta's deterministic core: the action shadow `sh` would emit while
        executing `subtask`, and whether that completes the errand.

        Delegates to the human's OWN execute() - literally the same code path
        action() runs - so the likelihood cannot drift from the policy it is
        meant to model. (This used to be a hand-copy of action()'s execution
        branch. It matched, but any future change to the human's routing would
        have silently desynchronised the two.) execute() writes nothing and
        consumes no RNG, so calling it speculatively for many hypothetical tau is
        safe; its n_explore bump is returned to the caller and dropped here,
        because a speculative query is not a real exploration step.

        Memoised per (shadow, tick): the map depends only on (state, subtask,
        beliefs), so re-deriving it for every tau' that leads to the same subtask
        is waste. explore_act is hoisted for the same reason - it is
        tau-invariant and by far the most expensive call in the tick.
        """
        if subtask in memo:
            return memo[subtask]
        act, arrived, _ = sh.execute(state, subtask, explore_act)
        memo[subtask] = (act, arrived)
        return memo[subtask]

    # -- filter -------------------------------------------------------------

    def update(self, state, observed_action):
        """Take in one (world state, action the human just took) and update the
        belief. Call this BEFORE stepping the environment, so the shadows judge
        the same state the human actually acted on.

        THE WHOLE IDEA, IN PLAIN WORDS
        --------------------------------------------------------------------
        We keep a SCORE for every guess of the form:

            "the human has cone F, and right now they are doing task T"

        Each tick we ask every guess: IF that were true, which button would the
        human press? Then we look at the button they REALLY pressed.

            guessed right -> multiply that guess's score UP
            guessed wrong -> multiply that guess's score DOWN

        Finally we divide every score by the total so they add back up to 1.
        Do that every tick and the wrong cones slowly starve to zero.

        That is all this function does. The rest is bookkeeping.
        """

        # =================================================================
        # STEP 1 - every shadow LOOKS at the world.
        # Each shadow is a pretend-human with its own cone, so each one sees a
        # different slice and ends up believing different things. This is what
        # makes them disagree later, which is the entire source of our signal.
        # We do this for ALL shadows, even ones whose score is currently zero,
        # because a shadow that stops looking would have frozen, stale beliefs.
        # =================================================================
        try:
            for shadow in self.shadows.values():
                shadow.t = self.tick        # tell it what time it is (for decay)
                shadow.observe(state)       # let it look
        except Exception:
            return self._skip()             # something broke; skip this tick

        # =================================================================
        # STEP 2 - sort my current beliefs into one bucket per cone.
        #
        # self.b is one flat dict that looks like:
        #     { (30, tauA): 0.1,  (30, tauB): 0.2,  (90, tauA): 0.7, ... }
        # The key is a PAIR: (which cone, which task-state).
        #
        # We want it regrouped so we can walk one cone at a time:
        #     { 30: {tauA: 0.1, tauB: 0.2},  90: {tauA: 0.7}, ... }
        # =================================================================
        beliefs_by_fov = {}
        for fov in self.candidate_fovs:
            beliefs_by_fov[fov] = {}

        for key, probability in self.b.items():
            fov = key[0]                    # first half of the pair
            tau = key[1]                    # second half of the pair
            if probability > 0.0:
                beliefs_by_fov[fov][tau] = probability

        # =================================================================
        # STEP 3 - score every guess against what the human actually did.
        #
        # We fill in four running totals. IMPORTANT: these hold raw SCORES, not
        # probabilities. Nothing adds up to 1 until STEP 4.
        # =================================================================

        # the new belief we are building: {(cone, next task-state) -> score}
        # "next" because it describes where the human will be AFTER this action
        belief_next_unnormalised = defaultdict(float)

        # what task the human is doing RIGHT NOW: {task -> score}
        subtask_weight = defaultdict(float)

        # per cone: score BEFORE seeing the action, and AFTER
        weight_before_action = {}
        weight_after_action = {}

        for fov, shadow in self.shadows.items():
            my_beliefs = beliefs_by_fov[fov]        # {task-state -> probability}

            # how much do I believe in this cone right now, in total?
            total_before = 0.0
            for probability in my_beliefs.values():
                total_before += probability
            weight_before_action[fov] = total_before

            if len(my_beliefs) == 0:
                continue                    # this cone is already ruled out

            try:
                # The "wander around and look" action. It is the same no matter
                # which task we are testing, and it is the slowest thing here, so
                # work it out once and reuse it.
                explore_action = shadow._explore_action(state)

                # Is this shadow's next task decided for it by whatever is in its
                # hands? (_transition needs to know, so it can tell whether the
                # free-choice memory would have been overwritten.)
                held_object = state.players[self.agent_index].held_object
                held_name = held_object.name if held_object else None
                forced_by_hands = shadow._forced_held(held_name) is not None

                execute_cache = {}          # {task -> (action, finished?)}
                total_after = 0.0           #A plain running sum. It starts at zero and accumulates every weight this cone produces:

                # ---- for every task-state I think this cone might be in ----
                for tau, belief_in_tau in my_beliefs.items():

                    # Which task could that lead to this tick, and how likely is
                    # each one? Usually this is a single certain answer; only at
                    # a free choice does it spread over several.
                    next_subtasks, sampler_untouched = self._transition(
                        shadow, state, tau, forced_by_hands)

                    # ---- for every task it might now be doing ----
                    for subtask, prob_of_subtask in next_subtasks.items():
                        if prob_of_subtask <= 0.0:
                            continue        # impossible, do not bother

                        # If it were doing that task, which button would it press?
                        predicted_action, errand_done = self._execute(
                            shadow, state, subtask, explore_action, execute_cache)

                        # THE ONE LINE WHERE REAL EVIDENCE ENTERS.
                        # Did that match the button the human really pressed?
                        if predicted_action == observed_action:
                            prob_of_action = 1.0 - self.epsilon      # good guess
                        else:
                            prob_of_action = self.epsilon / max(1, self.n_actions - 1)

                        # Where does this leave the human's two memories next tick?
                        #   current: nothing, if the errand just finished;
                        #            otherwise the task still being worked on
                        if errand_done:
                            next_current = None
                        else:
                            next_current = subtask

                        #   sampled: unchanged if the chooser was never asked;
                        #            otherwise the new free choice - or nothing,
                        #            if what happened was a look/wait, not a choice
                        if sampler_untouched:
                            next_sampled = tau[1]           # keep the old memory
                        elif subtask in SAMPLING_SUBTASKS:
                            next_sampled = subtask          # a real new choice
                        else:
                            next_sampled = None             # a look or a wait

                        next_tau = (next_current, next_sampled)

                        # The score for this whole explanation:
                        #   how much I believed it  x  how likely this task
                        #   x  how well it predicted the action
                        weight = belief_in_tau * prob_of_subtask * prob_of_action

                        belief_next_unnormalised[(fov, next_tau)] += weight
                        subtask_weight[subtask] += weight
                        total_after += weight

                weight_after_action[fov] = total_after

            except Exception:
                # If ONE cone blows up we must throw away the WHOLE tick. Quietly
                # dropping a single cone is the same as handing it a perfect
                # score once everything is divided by the total - i.e. rewarding
                # it for crashing.
                return self._skip()

        # =================================================================
        # STEP 4 - turn the scores back into probabilities.
        #
        # Divide everything by ONE grand total, across cones AND task-states
        # together. Do NOT normalise each cone separately: that would cancel out
        # exactly the difference between cones, which is the thing we are trying
        # to measure.
        # =================================================================
        total_weight = 0.0
        for weight in belief_next_unnormalised.values():
            total_weight += weight

        if total_weight <= 0.0:
            return self._skip()         # nothing explains what we saw

        # first pass: divide by the total, and drop anything so tiny it can
        # never come back (keeps the dictionary from growing forever)
        surviving = {}
        for key, weight in belief_next_unnormalised.items():
            probability = weight / total_weight
            if probability > PRUNE:
                surviving[key] = probability

        # second pass: those tiny drops cost us a little mass, so divide again
        # to make the survivors add up to exactly 1
        surviving_total = 0.0
        for probability in surviving.values():
            surviving_total += probability

        self.b = {}
        for key, probability in surviving.items():
            self.b[key] = probability / surviving_total

        # =================================================================
        # STEP 5 - our answer to "what task is the human on right now?"
        # Same scores, just added up per task instead of per cone.
        # =================================================================
        total_subtask_weight = 0.0
        for weight in subtask_weight.values():
            total_subtask_weight += weight

        self.last_subtask_post = {}
        if total_subtask_weight > 0.0:
            for subtask, weight in subtask_weight.items():
                self.last_subtask_post[subtask] = weight / total_subtask_weight

        # =================================================================
        # STEP 6 - did this tick teach us anything about the cone?
        #
        # For each cone: (score after) / (score before) is exactly how likely the
        # observed action was under that cone. If every cone gives the same
        # number, they all get multiplied by the same amount, the cone belief
        # does not move at all, and the tick was wasted. We just count those.
        # =================================================================
        action_likelihood_per_fov = []
        for fov in weight_after_action:
            before = weight_before_action.get(fov, 0.0)
            if before > 0.0:
                action_likelihood_per_fov.append(weight_after_action[fov] / before)

        if len(action_likelihood_per_fov) > 1:
            spread = max(action_likelihood_per_fov) - min(action_likelihood_per_fov)
            if spread > 1e-9:
                self.n_informative += 1

        # =================================================================
        # STEP 7 - move the clock forward and hand back the cone belief.
        # =================================================================
        self._advance_clock()
        return self.posterior()

    def _skip(self):
        self.n_skipped += 1
        self._advance_clock()
        return self.posterior()

    def _advance_clock(self):
        self.tick += 1
        for sh in self.shadows.values():
            sh.t = self.tick

    # -- readouts -----------------------------------------------------------

    #posterior per fov
    def posterior(self):
        out = {f: 0.0 for f in self.candidate_fovs}
        for (f, _), p in self.b.items():
            out[f] += p
        return out

    #posterior per subtask
    def subtask_posterior(self):
        """P(tau_t) marginalised over theta - inferred, never supplied."""
        return dict(self.last_subtask_post)

    #maximum a posteriori/most likely fov
    def map_fov(self):
        p = self.posterior()
        return max(p, key=p.get)

    #maximum a posteriori/most likely subtask
    def map_subtask(self):
        return max(self.last_subtask_post, key=self.last_subtask_post.get) \
            if self.last_subtask_post else None


    def p_true(self, true_fov):
        """Scoring helper for the eval harness only - never an input."""
        return self.posterior().get(true_fov, 0.0)

    #entropy of the posterior distrbituion over both fov and subtask
    def entropy(self):
        return -sum(p * math.log(p) for p in self.posterior().values() if p > 0)
