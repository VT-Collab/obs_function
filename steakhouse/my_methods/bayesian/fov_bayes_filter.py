"""
Online Bayesian inference over the human's field of view (FOV) parameter theta,
given a trajectory of observed actions a^H_{1:t} and world states s_{1:t}.

Formula:
    P(theta | a^H_{1:t}) ~ P(theta) * PROD_{i=1..t} pi^H(a^H_i | s_{1:i}, theta, tau_i(theta))

tau_i(theta) is DETERMINISTIC given (theta, s_{1:i}): each hypothesis agent calls
ml_action(state) - which internally runs update(state), the FOV-limited knowledge
base update - to get its own tau without ever touching the real human agent.

pi^H is computed from that hypothesis's own KB: the medium-level planner produces
motion goals, the lowest-cost goal is selected, and the observed action is scored
either epsilon-greedily against the planner's best low-level action or
Boltzmann-rationally against one-step-ahead plan costs.

Numerics: log-space posterior, log-sum-exp normalisation.

MISHA NEW CHANGE - this module is a direct port of the validated minigrid design
(minigrid-ai/robot/estimation/bayesian_posterior/bayes_fov.py) into steakhouse,
replacing the ad-hoc linear-space filter that previously lived inline in
fov_full_episode_test.py (class StickyFOVBayesFilter). Three things changed and
all three matter:

  1. LOG SPACE. The old filter multiplied beliefs in linear space and then
     clamped them with a `belief_floor` to stop underflow. That clamp is not
     neutral - it resurrects every losing hypothesis to floor/n EVERY step, so
     the posterior can never get more confident than the floor allows and a
     single lucky step can flip the argmax. Log-space accumulation plus
     log-sum-exp (see _log_normalise) removes the underflow problem entirely, so
     the floor - and the distortion it caused - is gone.

  2. LIKELIHOOD IS A SWITCH, NOT A GUESS. minigrid scores the observed action
     epsilon-greedily against the hypothesis planner's best action. The previous
     steakhouse edit had switched unilaterally to a Boltzmann softmax over
     one-step-ahead motion costs, on the theory that near-one-hot likelihoods
     were too binary. That theory was never tested. Both are implemented here
     (likelihood="greedy" | "boltzmann") so the batch can measure which one
     actually converges instead of assuming.

  3. CRASHED SHADOWS ARE GENUINELY NEUTRAL. Crashes here are overwhelmingly the
     known pre-existing `assert len(motion_goals) != 0` library bug (see
     CARC_NOTES.md "Known gotchas"), i.e. a failure of OUR model, not evidence
     about the human. An earlier version skipped only the crashed hypothesis's
     update while still updating the others - but since _log_normalise applies a
     common shift, that is arithmetically identical to awarding the crashed
     hypothesis likelihood 1.0, a systematic REWARD for crashing (measured at
     -5.55 nats against the true FOV on one layout). The whole STEP is now
     dropped when any hypothesis crashes, so no hypothesis gains or loses
     relative to another. crash_penalty=True charges the crashed one MIN_P.
"""
import math

from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel

CANDIDATE_FOVS = [30, 90, 180]

# Smallest likelihood we will take a log of. Mirrors minigrid's 1e-9 guard: a
# hypothesis that finds the observed action impossible gets a large but FINITE
# penalty, so it can still be revived by later evidence.
MIN_P = 1e-9


def fov_prior(candidate_fovs):
    """Uniform prior over the candidate FOVs."""
    return {fov: 1.0 / len(candidate_fovs) for fov in candidate_fovs}


def apply_initial_kb(agent, start_state, mode="fov"):
    """MISHA NEW CHANGE - control what a vision-limited agent KNOWS at t=0.

    The stock init_knowledge_base (agent.py:1075-1087) copies EVERY object in
    start_state into the knowledge base with no FOV filtering, so every
    hypothesis starts omniscient and identical - hypotheses can then only come
    to differ once the world changes somewhere only some of them can see.

    modes:
      "omniscient" - stock behaviour, everything known at t=0
      "fov"        - drop objects outside this agent's vision cone at t=0, so
                     narrow-FOV hypotheses start genuinely more ignorant
      "empty"      - forget all loose objects at t=0 (maximally ignorant)

    NOTE the station summaries (pot_states / chop_states / sink_states) are
    deliberately left intact: kb_to_state_info (agent.py:1088) indexes into them
    unconditionally and a malformed one raises rather than degrading. At t=0 the
    kitchen is empty anyway, so those carry no information to hide - which is
    also why this knob alone cannot manufacture divergence. It only helps when
    the world actually contains something to be ignorant OF.
    """
    agent.init_knowledge_base(start_state)
    if mode == "omniscient":
        return
    special = ("pot_states", "sink_states", "chop_states", "other_player")
    for key in [k for k in agent.knowledge_base if k not in special]:
        obj = agent.knowledge_base[key]
        pos = getattr(obj, "position", None)
        if mode == "empty" or pos is None:
            del agent.knowledge_base[key]
            continue
        try:
            visible = agent.in_bound(start_state, pos, vision_bound=agent.vision_bound / 2)
        except Exception:
            visible = True
        if not visible:
            del agent.knowledge_base[key]


class SteakBayesFOVInference:
    """
    Posterior distribution P(theta) over candidate FOV values.

    One hypothesis agent runs per candidate theta. Each agent maintains its own
    FOV-specific knowledge base, so each hypothesis sees only what a human with
    that FOV would have seen - and therefore chooses subtasks only that human
    would have chosen. All hypotheses are anchored to the REAL human's pose (they
    are spectators of the true trajectory, they do not simulate their own).
    """

    def __init__(self, mlam, start_state, candidate_fovs=None, human_agent_index=1,
                 prior=None, likelihood="greedy", epsilon=0.05, ll_temp=1.0,
                 crash_penalty=False, agent_cls=StickySubtaskHumanModel,
                 initial_kb="fov", kb_update_delay=2):
        """
        mlam: medium-level planner shared with the real game (stateless w.r.t. FOV)
        start_state: the game's starting OvercookedState
        candidate_fovs: FOV hypotheses in degrees, e.g. (30, 90, 180)
        human_agent_index: which player index is the human being watched
        prior: {fov: probability}; defaults to uniform
        likelihood: "greedy" (epsilon-greedy around the planner's best action,
            matching minigrid) or "boltzmann" (softmax over one-step-ahead plan
            costs toward the chosen goal)
        epsilon: exploration mass for likelihood="greedy"
        ll_temp: temperature for likelihood="boltzmann" (lower = sharper)
        crash_penalty: if True, a hypothesis whose planner raises is penalised
            with MIN_P; if False (default) that hypothesis simply gets no update
            on that step
        """
        self.candidate_fovs = list(candidate_fovs or CANDIDATE_FOVS)
        self.human_agent_index = human_agent_index
        self.likelihood = likelihood
        self.epsilon = epsilon
        self.ll_temp = ll_temp
        self.crash_penalty = crash_penalty
        self.initial_kb = initial_kb
        self.n_actions = len(Action.ALL_ACTIONS)

        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior
        self.log_posterior = {fov: math.log(self.prior[fov]) for fov in self.candidate_fovs}

        # MISHA NEW CHANGE - kb_update_delay must MATCH the watched human's.
        # It is the knob that makes FOV observable at all: a fact only enters the
        # knowledge base after being held in view that many consecutive steps
        # (agent.py:1425), and at the project default of 0 every FOV learns the
        # same things, so the posterior has nothing to separate. Measured, only
        # delay 2-3 produces divergence at all. If the shadows ran at a different
        # delay from ground truth, the likelihood would be misspecified for every
        # hypothesis - including the true one - which is worse than no evidence.
        self.kb_update_delay = kb_update_delay
        self.hypothesis_agents = {}
        for fov in self.candidate_fovs:
            agent = agent_cls(mlam, start_state, vision_limit=True, vision_bound=fov,
                              debug=False, ll_boltzmann_rational=(likelihood == "boltzmann"),
                              ll_temp=ll_temp, kb_update_delay=kb_update_delay)
            agent.set_agent_index(human_agent_index)
            apply_initial_kb(agent, start_state, initial_kb)
            self.hypothesis_agents[fov] = agent

        # Diagnostics: distinguishes "no divergent signal ever reached the
        # filter" from "signal was present but the update didn't act on it".
        self.n_steps_seen = 0
        self.n_goal_divergent_steps = 0
        self.n_crashes = 0
        self.n_crash_skipped_steps = 0
        self.n_low_level_discrimination_steps = 0

    # -- Episode management ---------------------------------------------------

    def reset(self, state):
        """Call at the START of each episode."""
        for agent in self.hypothesis_agents.values():
            apply_initial_kb(agent, state, self.initial_kb)
            agent.prev_chosen_subtask = None
            agent.prev_state = None
        prior = fov_prior(self.candidate_fovs)
        self.log_posterior = {fov: math.log(prior[fov]) for fov in self.candidate_fovs}
        self.n_steps_seen = 0
        self.n_goal_divergent_steps = 0
        self.n_crashes = 0
        self.n_crash_skipped_steps = 0
        self.n_low_level_discrimination_steps = 0

    # -- Core update ----------------------------------------------------------

    def update(self, state, observed_action):
        """
        Observe one (state_t, a^H_t) pair and update the posterior.

        For each hypothesis theta:
          1. ml_action(state)  - let the hypothesis "see" through FOV=theta
                                 (this is where its KB is updated) and pick tau
          2. p = pi^H(a^H_t | s_t, theta, tau)  - likelihood of the observed action
          3. log P(theta) += log p              - Bayes update in log-space

        Then normalise so the distribution sums to 1 in probability space.

        IMPORTANT ORDERING: call this BEFORE stepping the environment, so every
        hypothesis processes the same state the real human actually acted on.
        """
        goals_seen = []
        # Per-hypothesis detail for the most recent step, so diagnostics can
        # inspect what each shadow did without re-calling ml_action (which would
        # advance that shadow's knowledge base a second time on the same state).
        self.last_step_detail = {}
        probs = {}
        for fov, agent in self.hypothesis_agents.items():
            p, goal = self._action_prob(agent, state, observed_action)
            goals_seen.append(goal)
            probs[fov] = p
            self.last_step_detail[fov] = dict(
                p=p, goal=goal, subtask=getattr(agent, "prev_chosen_subtask", None))

        crashed = [f for f, p in probs.items() if p is None]
        self.n_crashes += len(crashed)

        # MISHA NEW CHANGE - crash abstention is now ACTUALLY neutral.
        #
        # The previous version simply skipped a crashed hypothesis's update
        # while still updating the others, then renormalised. Because
        # _log_normalise applies a common shift, skipping is arithmetically
        # identical to giving the crashed hypothesis likelihood 1.0 - i.e. a
        # systematic REWARD for crashing, larger than any real likelihood the
        # surviving hypotheses can earn. An audit measured this at 73% of
        # likelihood-differing steps on one layout, worth -5.55 nats. The old
        # docstring called it "uninformative"; that was mathematically false.
        #
        # These crashes are overwhelmingly the known library bug
        # (assert len(motion_goals) != 0, agent.py:1676 - see CARC_NOTES.md),
        # i.e. a failure of OUR model rather than evidence about the human. So
        # the honest handling is to drop the STEP ENTIRELY when any hypothesis
        # crashed: nobody gains or loses relative to anybody else. Set
        # crash_penalty=True to instead charge the crashed hypothesis MIN_P.
        if crashed and not self.crash_penalty:
            self.n_crash_skipped_steps += 1
        else:
            for fov, p in probs.items():
                if p is None:
                    self.log_posterior[fov] += math.log(MIN_P)
                else:
                    self.log_posterior[fov] += math.log(max(p, MIN_P))

        self.n_steps_seen += 1
        if len(set(goals_seen)) > 1:
            self.n_goal_divergent_steps += 1
        # Diagnostic: steps where hypotheses sharing a subtask still got
        # different likelihoods. That is low-level (motion-goal) discrimination
        # leaking in, which the design intends to exclude - it should stay 0.
        live = {f: d for f, d in self.last_step_detail.items() if d["p"] is not None}
        by_subtask = {}
        for f, d in live.items():
            by_subtask.setdefault(d["subtask"], []).append(probs[f])
        if any(len(v) > 1 and len(set(v)) > 1 for v in by_subtask.values()):
            self.n_low_level_discrimination_steps += 1

        self._log_normalise()
        return self.posterior()

    def _action_prob(self, agent, state, observed_action):
        """pi^H(observed_action | state, this hypothesis's KB).

        Returns (probability, chosen_goal). probability is None if this
        hypothesis's planner could not produce a plan for this state.
        """
        start_pos_and_or = state.players_pos_and_or[self.human_agent_index]
        try:
            # ml_action() calls self.update(state) internally (agent.py:1484) -
            # that is the per-hypothesis, FOV-limited knowledge base update.
            motion_goals = agent.ml_action(state)
            chosen_goal, best_action = agent.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals)
            if chosen_goal is None or best_action is None:
                return None, "NOPLAN"

            if self.likelihood == "boltzmann":
                _, action_probs = agent.boltzmann_rational_ll_action(
                    start_pos_and_or, chosen_goal)
                p = action_probs[Action.ACTION_TO_INDEX[observed_action]]
            else:
                # Epsilon-greedy around the planner's best action - the minigrid
                # form. (1-eps) mass on the predicted action, eps spread evenly.
                hit = (observed_action == best_action)
                p = (1.0 - self.epsilon) * float(hit) + self.epsilon / self.n_actions
            return p, chosen_goal
        except Exception:
            return None, "EXC"

    # -- Posterior queries ----------------------------------------------------

    def posterior(self):
        """Current P(theta) as a probability dict (NOT log-space)."""
        return {fov: math.exp(lp) for fov, lp in self.log_posterior.items()}

    def map_fov(self):
        """MAP estimate: argmax_theta P(theta | a^H_{1:t})."""
        return max(self.log_posterior, key=self.log_posterior.get)

    def entropy(self):
        """Shannon entropy H = -sum P(theta) log P(theta), in nats.
        High = uncertain, low = confident."""
        H = 0.0
        for lp in self.log_posterior.values():
            H -= math.exp(lp) * lp
        return H

    # -- Helpers --------------------------------------------------------------

    def _log_normalise(self):
        """Rescale the un-normalised log-posteriors so they sum to 1 in
        probability space, without overflow or underflow (log-sum-exp)."""
        max_lp = max(self.log_posterior.values())
        log_Z = max_lp + math.log(
            sum(math.exp(lp - max_lp) for lp in self.log_posterior.values())
        )
        self.log_posterior = {fov: lp - log_Z for fov, lp in self.log_posterior.items()}

    # -- Backwards-compatible aliases ----------------------------------------
    # fov_full_episode_test.py / fov_inference_batch.py were written against the
    # old StickyFOVBayesFilter API (step/estimate/belief). Keep them working.

    def step(self, state, human_action):
        self.update(state, human_action)
        return self.belief

    def estimate(self):
        return self.map_fov()

    @property
    def belief(self):
        post = self.posterior()
        return [post[fov] for fov in self.candidate_fovs]
