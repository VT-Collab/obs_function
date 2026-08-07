"""
AssistModule - the FOV-aware ROBOT MODULE. ALL decision + FOV logic lives HERE,
never in the env. The env is a dumb host: it hands the module the baseline's
suggested primitive action and asks for the final one.

The module owns:
  * the Bayesian FOV inference (SamplingBayesFOVInference) + shadow humans
  * its OWN greedy cook (the robot has full vision, so this is legitimate and
    kept separate from the env's greedy)
  * the decision policy: given the baseline's suggested primitive, DEFER or
    OVERRIDE (take over the specific track the human cannot finish)

Nothing about the world's dynamics is decided here; nothing about the robot's
policy is decided in the env.

Decision strategies (mode):
  "subtask"  - PREDICT the human's subtask from the shadow; take over ONLY if
               they are committed to a track whose station they have NOT
               discovered (believed == UNKNOWN). Principled but, as measured,
               weaker than "assist" on some layouts.
  "assist"   - take over while the human is inferred MODERATELY blind
               (blind_fov_min < map_fov <= blind_fov_max). Cruder, but the
               stronger performer in the first sweep.
  "smart"    - USE THE WHOLE POSTERIOR, not just map_fov. For the track the human
               is currently committed to (holding meat/onion/plate), compute the
               posterior-weighted probability the human has DISCOVERED the drop
               station. If that probability is low, they cannot finish -> take
               over THAT track. Marginalises over every candidate FOV's shadow
               belief; entropy-gated. This is the "be smart with the inference"
               rule.
  anything else -> always defer (pure baseline; the control).
"""
import math

from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from overcooked_ai_py.mdp.overcooked_mdp import Action, ObjectState
from steakhouse.fov.robot.policy.old.inference.bayes_fov_sampling import SamplingBayesFOVInference
from steakhouse.fov.robot.policy.old.baseline.features import station_locs, _station_state
from fov.human.agent.limited_vision_human import UNKNOWN
from fov.human.planning.steak_planner import SUBTASK_TARGETS


def confidence(entropy, n):
    return max(0.0, 1.0 - entropy / math.log(max(2, n)))


class AssistModule:
    def __init__(self, mdp, mlp, candidate_fovs, human_index=1, robot_index=0,
                 mode="subtask", conf_min=0.4, blind_fov_min=45, blind_fov_max=90,
                 trust_thr=0.5):
        self.mdp, self.mlp = mdp, mlp
        self.hi, self.ri = human_index, robot_index
        self.mode = mode
        self.conf_min = conf_min
        self.blind_fov_min = blind_fov_min
        self.blind_fov_max = blind_fov_max
        self.trust_thr = trust_thr
        self.inf = SamplingBayesFOVInference(mdp, mlp, candidate_fovs,
                                             human_agent_index=human_index)
        self._worker = None
        # diagnostics only
        self.n_step = 0
        self.n_override = 0
        self.n_takeover = 0

    # ---- inference ----------------------------------------------------------
    def observe(self, state, human_subtask):
        """Feed the human's observed subtask + the pre-step state into the filter."""
        if human_subtask is not None and state is not None:
            try:
                self.inf.update(state, human_subtask)
            except Exception:
                pass

    def confident(self):
        return confidence(self.inf.entropy(),
                          len(self.inf.candidate_fovs)) >= self.conf_min

    def estimate(self):
        return self.inf.map_fov(), self.inf.entropy()

    # ---- the robot's own greedy cook (full vision) --------------------------
    def _greedy(self, state):
        if self._worker is None:
            self._worker = GreedySteakHumanModel(self.mlp)
            self._worker.set_agent_index(self.ri)
        try:
            a, _ = self._worker.action(state)
            return a
        except Exception:
            return Action.STAY

    def _takeover(self, state):
        """Robot greedy with the (stuck) human's held object HIDDEN, so the robot
        stops deferring (other_has_X) to a partner who cannot finish."""
        hp = state.players[self.hi]
        saved = hp.held_object
        try:
            hp.held_object = None
            return self._greedy(state)
        finally:
            hp.held_object = saved

    # ---- reading the human --------------------------------------------------
    def predict_human(self, state):
        """(predicted subtask, shadow) from the shadow at the inferred FOV."""
        sh = self.inf.shadows.get(self.inf.map_fov())
        if sh is None:
            return None, None
        try:
            dist = sh.subtask_distribution(state, self.inf.committed)
            return (max(dist, key=dist.get) if dist else None), sh
        except Exception:
            return None, sh

    def human_stuck(self, state):
        """Is the human committed to a subtask whose target station they have NOT
        discovered (believed == UNKNOWN)? Decided by their PREDICTED SUBTASK +
        their inferred beliefs - not a raw FOV band."""
        tau, sh = self.predict_human(state)
        if tau is None or sh is None:
            return False
        kind = SUBTASK_TARGETS.get(tau, "")
        return kind in ("pot", "board", "sink") and sh.believed(kind) == UNKNOWN

    # ---- the SMART signal: marginalise discovery over the whole posterior ----
    _DROP = {"meat": "pot", "onion": "board", "plate": "sink"}

    def p_discovered(self, station_kind):
        """Posterior-weighted P(the human has DISCOVERED station_kind). Sums the
        posterior mass of every candidate FOV whose shadow already knows where
        that station is (believed != UNKNOWN). Uses the FULL distribution, not
        just the MAP - so a bimodal posterior is handled honestly."""
        post = self.inf.posterior()
        z = sum(post.values()) or 1.0
        p = 0.0
        for fov, w in post.items():
            sh = self.inf.shadows.get(fov)
            if sh is not None:
                try:
                    if sh.believed(station_kind) != UNKNOWN:
                        p += w
                except Exception:
                    pass
        return p / z

    def smart_takeover_track(self, state):
        """If the human holds an ingredient whose drop-station the posterior says
        they probably HAVEN'T found (p_discovered < trust_thr), return that track;
        else None. This is the track the human cannot finish."""
        held = state.players[self.hi].held_object
        if held is None:
            return None
        st = self._DROP.get(getattr(held, "name", ""))
        if st is None:
            return None
        return st if self.p_discovered(st) < self.trust_thr else None

    # ---- COORDINATION: divide labor, don't just take over -------------------
    _ING = {"pot": "meat", "board": "onion", "sink": "plate"}

    def _needs_load(self, state, kind):
        cells = station_locs(self.mdp).get(kind, [])
        return any(_station_state(self.mdp, state, c) == "empty" for c in cells)

    def coordinate_action(self, state):
        """Division of labor. If the human is free-handed, PRE-ASSIGN them the
        empty track they are MOST able to cover (highest posterior discovery) by
        briefly faking them holding that ingredient - so the robot's own greedy
        DEFERS that track (other_has_X) and instead covers a DIFFERENT one. The
        robot thus complements the human's likely contribution rather than racing
        them for the same station."""
        hp = state.players[self.hi]
        saved = hp.held_object
        try:
            if saved is None:
                best, best_p = None, -1.0
                for kind in ("pot", "board", "sink"):
                    if self._needs_load(state, kind):
                        p = self.p_discovered(kind)
                        if p > best_p:
                            best_p, best = p, kind
                if best is not None and best_p >= self.trust_thr:
                    hp.held_object = ObjectState("coord_fake", self._ING[best],
                                                 hp.position)
            return self._greedy(state)
        finally:
            hp.held_object = saved

    # ---- the decision -------------------------------------------------------
    def robot_action(self, state, baseline_primitive):
        """Return (primitive_action, tag). Defer => baseline_primitive unchanged.
        This is the ONLY thing the env calls to influence the robot."""
        self.n_step += 1
        # unsure -> obey the baseline (never worse than baseline)
        if not self.confident():
            return baseline_primitive, "defer"

        # COORDINATE: take over what they can't reach; complement what they can
        if self.mode == "coordinate":
            track = self.smart_takeover_track(state)
            if track is not None:
                self.n_override += 1; self.n_takeover += 1
                return self._takeover(state), "takeover:" + track
            self.n_override += 1
            return self.coordinate_action(state), "coordinate"

        # take-over-only rules
        fire = False
        tag = "takeover"
        if self.mode == "subtask":
            fire = self.human_stuck(state)
        elif self.mode == "assist":
            mf = self.inf.map_fov()
            fire = mf is not None and self.blind_fov_min < mf <= self.blind_fov_max
        elif self.mode == "smart":
            track = self.smart_takeover_track(state)
            fire = track is not None
            if fire:
                tag = "takeover:" + track
        elif self.mode == "both":
            track = self.smart_takeover_track(state)
            mf = self.inf.map_fov()
            band = mf is not None and self.blind_fov_min < mf <= self.blind_fov_max
            fire = (track is not None) or band
            if track is not None:
                tag = "takeover:" + track
        if fire:
            self.n_override += 1
            self.n_takeover += 1
            return self._takeover(state), tag
        return baseline_primitive, "defer"
