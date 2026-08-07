"""
FOV inference file
Watches the human's behaviour and infers which FOV they may have, together with
the entropy of that belief and the posterior over what subtask they are on.

One shadow LimitedVisionSteakHuman is held per candidate FOV; each carries the
FOV-gated, decaying beliefs a human with that cone would actually have, so both
kernels below are the ones the real human is running.

NO ORACLE. The only inputs are the world state and the human's emitted ACTION.
The true FOV and the human's internal subtask label are never read.
"""

import math
from collections import defaultdict

from overcooked_ai_py.mdp.overcooked_mdp import Action


#import the actual human model
from fov.human.agent.limited_vision_human import (
    LimitedVisionSteakHuman, SAMPLING_SUBTASKS, COMMITTED
)
from fov.human.planning.steak_planner import SteakMotionPlanner


#Likelihood smoothing. Execution is deterministic given (fov, tau, beliefs), so
#a mismatch is real evidence, not noise - the validated filter in
#old/inference/bayes_fov_sampling.py uses 0.0 (exact) and measured the smoothing
#to be inert. A small floor is the safe version: it stops one surprising tick
#from annihilating a hypothesis outright.
FLOOR = 1e-6
PRUNE = 1e-12       # numerical hygiene: drop states that cannot come back


#set the prior as uniform across all candidates
def fov_prior(candidate_fovs):
    distribution = {f: 1.0/len(candidate_fovs) for f in candidate_fovs}
    return distribution


class SamplingBayesFOVInference:
    """
    Joint (FOV, subtask) filter driven by the human's observed actions.
    """
    def __init__(self, mdp, mlp, candidiate_fovs,
                 human_agent_index = 1, prior=None,
                 floor=FLOOR):

        self.candidate_fovs = list(candidiate_fovs)
        self.floor = floor
        self.agent_index = human_agent_index

        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior

        #planner is shared no matter the fov
        planner = SteakMotionPlanner(mdp, mlp)
        #initialize the shadows, one per hypothesis
        self.shadows = {
            fov: LimitedVisionSteakHuman(mdp, fov, planner, agent_index=human_agent_index)
            for fov in self.candidate_fovs
        }

        self.n_actions = len(Action.ALL_ACTIONS)

        # b = the joint belief over (fov, tau).
        # Tau is further broken into sampled and current
        # Within an episode the FOV is CONSTANT and tau changes every tick.
        #
        #     (f, (current, sampled))
        #      |    |        `-- self._sampled
        #      |    `-- self._current
        #      `-- the FOV hypothesis (fixed for the whole episode)
        #
        # BOTH halves of tau are a single subtask NAME (a string) or None.
        # Neither is a distribution. Two are needed because the human keeps two
        # separate memories, and its next decision reads BOTH.
        #
        # VALUES ------------------------------------------------------------
        #
        # sampled   7: None, or one of the 6 you can freely CHOOSE
        #              pickup_meat   pickup_onion     chop_onion
        #              pickup_plate  heat_washed_plate   pickup_washed_plate
        #
        # current  18: None, or ANY subtask - above 6, plus 11 you never choose
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
        #     current   LOWER LEVEL; EG: pickup_steak. when current stops being valid, control passes
        #               UP to the decision layer — which re-decides, but usually does not
        #               resample: hands full skips the sampler entirely, and when
        #               consulted it more often finds the line still worth pursuing and continues.
        #               Rewritten every tick, and
        #             reset to None the instant the errand completes (arrival).
        #             It holds the decision steady only while it is in COMMITTED
        #             *and* _commit_still_useful() says so - it is a lock that
        #             BREAKS ON SIGHT, not a lock that ignores the world:
        #               - see the teammate carrying the thing you are fetching
        #                 (_robot_redundant)          -> break, re-decide
        #               - see the destination already occupied -> break
        #               - belief merely decayed to UNKNOWN     -> HOLD
        #                 (forgetting is not observing; dropping an errand
        #                 because a belief went stale is dithering)
        #             Note this has nothing to do with holding an item: walking
        #             to fetch meat empty-handed sets current=pickup_meat and it
        #             holds under exactly the same rules. What the hands change
        #             is WHICH subtask you get (_forced_held), not whether the
        #             lock applies.
        #
        #             Both break conditions are FOV-GATED, and that asymmetry IS
        #             the FOV effect the whole paper rests on: a wide cone sees
        #             the teammate grab the meat and re-plans mid-trip, a narrow
        #             cone never sees it and walks the wasted errand to the end.
        #
        #    sampled   HIGHER LEVEL; EG: STEAK; the last thing I freely CHOSE. Only the sampler writes it,
        #             so it can change ONLY when the hands are empty AND the
        #             errand lock has released AND the old choice is no longer
        #             helpful. A subtask forced by a held object never overwrites
        #             it, so it survives a whole pickup -> carry -> drop
        #             excursion, outliving several values of `current`.
        #
        # WHY TWO -----------------------------------------------------------
        # current is wiped the instant a step finishes, so it cannot answer
        # "what was I working on?". Only sampled survives that, and the next
        # decision needs it: still worth doing -> continue; not -> redraw the
        # whole softmax. Two completely different distributions, so one variable
        # is not enough.
        #
        # t=0: nothing chosen, nothing in progress - exactly what reset() leaves
        # behind - so every hypothesis puts ALL its prior mass on (None, None),
        # and marginalising over tau hands back the FOV prior untouched.

        self.b = {
            #fov, current, sampled
            (f, (None, None)): self.prior[f] for f in self.candidate_fovs
        }

        #simulates the same forgetfulness the human has
        self.tick = 0

        # Cache of P(tau_t) from the most recent update, marginalised over theta.
        # Kept only so subtask_posterior() / map_subtask() can be read AFTER
        # update(state, action); it describes THAT tick, not the next one.
        self.last_subtask_post = {}

        #-------------Diagnostic only------------------
        #ticks where the hypotheses actually disagreed about how likely
        #the observed action was
        self.n_informative = 0

        # Diagnostic: ticks thrown away because a shadow raised, or because no
        # (theta, tau) could explain the action at all. Should stay at 0; if it
        # climbs, the kernel has drifted out of sync with the human model.
        self.n_skipped = 0


    #-------------------------Functions calling human helpers--------------
    def _transition(self, shadow, state, tau, forced):
        """
        T_theta(. | tau, s). Read it as "check the lock on the current subtask
        layer; if it broke, re-decide."

        Second return value: whether the sampler's memory (self._sampled) would
        have been left untouched - true when the errand lock held, or when the
        held object forced the choice.
        """
        current, sampled = tau
        #check if current low level subtask is still helpful
        #current in COMMITTED — fixed lookup. "Is this the kind of thing you can lock onto in principle?" Same answer forever.
        #then check if actually still useful
        if current in COMMITTED and shadow._commit_still_useful(current):
            return {current: 1.0}, True
        #otherwise, sample the higher level
        return shadow.subtask_distribution(state, sampled), forced

    def _execute(self, shadow, state, subtask, explore_act, memory):
        """If this shadow were doing subtask right now, what action would it take — and
        whether or not that action would be the action that finish the errand?

        it's the prediction half of the filter. You ask each shadow "what would you do?",
        then compare against what the real human actually did.
        """

        if subtask in memory:
            return memory[subtask]

        #shadow.execute is the human turning a subtask into an actual action
        action, arrived, _ = shadow.execute(state, subtask, explore_act)
        memory[subtask] = (action, arrived)

        return memory[subtask]

    #------------------ENV or SHADOW ticks---------------------
    def _skip(self):
        self.n_skipped += 1
        self._advance_clock()
        return self.posterior()

    def _advance_clock(self):
        self.tick += 1
        for sh in self.shadows.values():
            sh.t = self.tick

    #------------------ACTUAL UPDATING STEP -------------
    def update(self, state, observed_action):
        """
        Observe one (state, ACTION) pair and update the posterior.
        Call BEFORE stepping the env, so shadows score the state the human acted on.

        The second argument is the human's emitted PRIMITIVE ACTION - a physical
        event anyone in the kitchen can watch. It is deliberately NOT the human's
        internal subtask label; inferring that is this filter's job.

        Quick note on python notation:
            self.shadows = {30: <shadow>, 60: <shadow>, 90: <shadow>, 120: <shadow>, 180: <shadow>, 360: <shadow>}
            method	        gives you	        example
            .keys()	        just the keys	    30, 60, 90, 120, 180, 360
            .values()	    just the values	    <shadow>, <shadow>, …
            .items()	    both, as pairs	    (30, <shadow>), (60, <shadow>), …

        """
        # =================================================================
        #STEP 1 - every shadow LOOKS at the world.
        #update each shadow agent's observed history and tick first; let them see current state
        # =================================================================
        try:
            for sh in self.shadows.values():
                sh.t = self.tick
                sh.observe(state) #update shadow agent
        except Exception:
            #hopefully this doesnt happen unless something is wrong
            return self._skip()

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
        for fov, shadow in self.shadows.items():
            beliefs_by_fov[fov] = {}

        for key, probability in self.b.items():
            #get fov and tau (tau in and of itself is a pair)
            #split key into fov and tau
            fov = key[0]
            tau = key[1]

            if probability > 0.0:
                #use both fov and tau
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

        #what task you think the human is doing RIGHT NOW: {task -> score}
        subtask_weight = defaultdict(float)

        # per cone: score BEFORE seeing the action, and AFTER
        weight_before_action = {}
        weight_after_action = {}

        #loop over each shadow with fov
        for fov, shadow in self.shadows.items():
            my_beliefs = beliefs_by_fov[fov]        # {task-state -> probability}, comes from self.b

            # how much do I believe in this cone right now, in total?
            total_before = 0.0
            for probability in my_beliefs.values():
                total_before += probability
            weight_before_action[fov] = total_before

            # this cone is already ruled out
            if len(my_beliefs) == 0:
                continue

            try:
                # the explore action itself aka what would agent do if they are exploring
                # The "wander around and look" action. It is the same no matter
                # which task we are testing, and it is the slowest thing here, so
                # work it out once and reuse it.
                explore_action = shadow._explore_action(state)

                # Is this shadow's next task decided for it by whatever is in its
                # hands? (_transition needs to know, so it can tell whether the
                # free-choice memory would have been overwritten.)
                held_object = state.players[self.agent_index].held_object
                held_name = held_object.name if held_object else None
                #T/F for whether the task is forced by what is held or not
                forced_by_hands = shadow._forced_held(held_name) is not None

                execute_cache = {}          # {task -> (action, finished?)}
                total_after = 0.0           #A plain running sum. It starts at zero and accumulates every weight this cone produces:

                # ---- for every task-state I think this cone might be in ----
                for tau, belief_in_tau in my_beliefs.items():

                    # Given it was in this task-state, which task could it be on
                    # THIS tick, and how likely is each? Usually a single certain
                    # answer; only at a free choice does it spread over several.
                    next_subtasks, sampler_untouched = self._transition(
                        shadow, state, tau, forced_by_hands)

                    # ---- for every task it might now be doing ----
                    for subtask, prob_of_subtask in next_subtasks.items():

                        if prob_of_subtask <= 0.0:
                            continue #impossible, do not bother

                        #if it were doing that task, which button would it press?
                        predicted_action, errand_done = self._execute(
                            shadow, state, subtask, explore_action, execute_cache)

                        # THE ONE LINE WHERE REAL EVIDENCE ENTERS.
                        # Did that match the button the human really pressed?
                        if predicted_action == observed_action:
                            prob_of_action = 1.0 - self.floor      # good guess
                        else:
                            prob_of_action = self.floor / max(1, self.n_actions - 1)

                        # Is this errand done or not
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
                            next_sampled = subtask          # a real new choice, already in loop above
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
            if probability > PRUNE:  #only keep the ones that are not too small
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
        # STEP 4 bucketed the same scores by cone; this buckets them by task.
        # Same total mass, different grouping, so normalise it on its own.
        # The result is P(tau_t) MARGINALISED over the cone - the joint is in
        # self.b, and is the thing to read if you need P(tau | fov).
        # =================================================================
        total_subtask_weight = 0.0
        for weight in subtask_weight.values():
            total_subtask_weight += weight

        self.last_subtask_post = {}
        if total_subtask_weight > 0.0:
            for subtask, weight in subtask_weight.items():
                self.last_subtask_post[subtask] = weight / total_subtask_weight


        # =================================================================
        # STEP 6 - DIAGNOSIS ONLY: did this tick teach us anything about the cone?
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


    #-------FOR BUILDING COST FUNCTION, INTEGRATE LATER---------------------
    # def subtask_distri_diff():
    #     """
    #     return measure of how much the subtask distribution would change
    #     """

    # def kb_diff():
    #     """
    #     return measure of how much the knowledge base difference would change
    #     """

    # -- readouts -----------------------------------------------------------

    #posterior per fov, marginalised over tau
    def posterior(self):
        out = {f: 0.0 for f in self.candidate_fovs}
        for (f, _), p in self.b.items():
            out[f] += p
        return out

    #posterior per subtask, marginalised over fov
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

    #entropy of the posterior over FOV only (not the joint with subtask)
    def entropy(self):
        return -sum(p * math.log(p) for p in self.posterior().values() if p > 0)
