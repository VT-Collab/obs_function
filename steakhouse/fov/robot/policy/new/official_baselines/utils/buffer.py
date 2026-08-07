"""
Rollout buffer: the notebook PPO writes into while playing, and reads back
from while learning.

===========================================================================
WHAT IT HOLDS
===========================================================================
N = E * num_agents rows. Everything is stored (T, N, ...) or (T+1, N, ...).
T = episode_length, the number of ticks in one rollout.

Why T+1 for some of them: to compute the return at tick T-1 you need the
value of the state you LANDED on, which is one step past the end. So obs,
rnn_states, masks and values all carry one extra slot.

    obs          (T+1, N, 23, W, H)
    rnn_actor    (T+1, N, L, H)      GRU memory going INTO each tick
    rnn_critic   (T+1, N, L, H)
    masks        (T+1, N, 1)         0 = episode restarted here, wipe memory
    values       (T+1, N, 1)         critic's guess at each tick
    actions      (T,   N, 1)
    log_probs    (T,   N, 1)         logp_OLD -- frozen at collection time
    rewards      (T,   N, 1)
    returns      (T,   N, 1)         filled in by compute_returns()
    advantages   (T,   N, 1)         filled in by compute_returns()

Every per-tick quantity is a (N, 1) COLUMN, not a flat (N,). That is what
lets them all stack and subtract without broadcasting into (N, N) by
accident -- the same trap as `hxs * masks` in rnn.py.

===========================================================================
GAE -- WHAT compute_returns IS DOING
===========================================================================
The naive target for the critic is the discounted sum of every future
reward. That is unbiased but extremely noisy: one lucky delivery 200 ticks
later swamps the signal.

The other extreme is the one-step TD error:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
              "was this tick better than the critic expected?"

Low noise, but it inherits all the critic's bias.

GAE interpolates between them with lambda:

    A_t = delta_t + gamma * lam * A_{t+1}          (walked BACKWARDS)

    lam = 0  -> pure one-step TD    (low variance, high bias)
    lam = 1  -> full monte-carlo    (no bias, high variance)
    lam ~ 0.95 is the usual compromise.

Then returns = advantages + values, and that is what the critic regresses
against.

The masks matter here: masks[t+1] == 0 means the episode ended at t, so the
next state belongs to a DIFFERENT episode. Multiplying by it zeroes both the
bootstrap and the accumulated advantage, which stops reward from leaking
backwards across an episode boundary.

===========================================================================
recurrent_generator -- WHY MINIBATCHES ARE OVER ENVS, NOT OVER TICKS
===========================================================================
A normal PPO buffer shuffles all T*N rows and cuts them into random
minibatches. You CANNOT do that with a GRU: the memory at tick t only makes
sense if ticks 0..t-1 were fed in first, in order.

So minibatching happens over the N rows instead. Each minibatch takes a
subset of rows and hands over their FULL T-tick sequence, plus the single
rnn_state each row started the rollout with. Time order is preserved inside
every minibatch, which is exactly what RNNLayer's training branch expects.

The flatten is (T, mb) -> (T*mb) with reshape, i.e. TICK-MAJOR: the first
mb rows are tick 0 for every row in the minibatch. RNNLayer un-flattens with
x.view(T, N, -1), which assumes exactly that order. Do not swap it.
"""

import numpy as np
import torch


class SelfPlayBuffer:

    def __init__(self, episode_length, n_rows, obs_shape, recurrent_N, hidden_size,
                 gamma=0.99, gae_lambda=0.95):
        T, N = episode_length, n_rows
        self.T, self.N = T, N
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((T + 1, N, *obs_shape), dtype=np.float32)
        self.rnn_actor = np.zeros((T + 1, N, recurrent_N, hidden_size), dtype=np.float32)
        self.rnn_critic = np.zeros((T + 1, N, recurrent_N, hidden_size), dtype=np.float32)
        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        #bad_masks: 0 only when the episode was CUT OFF by the horizon rather
        #than genuinely ending. "proper time limits", as ZSC-Eval calls it.
        self.bad_masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.values = np.zeros((T + 1, N, 1), dtype=np.float32)

        self.actions = np.zeros((T, N, 1), dtype=np.int64)
        self.log_probs = np.zeros((T, N, 1), dtype=np.float32)
        self.rewards = np.zeros((T, N, 1), dtype=np.float32)
        self.returns = np.zeros((T, N, 1), dtype=np.float32)
        self.advantages = np.zeros((T, N, 1), dtype=np.float32)

        self.step = 0

    def reset(self, obs):
        """call once after envs.reset(). obs is (N, 23, W, H)."""
        self.obs[0] = obs
        self.rnn_actor[0] = 0.0
        self.rnn_critic[0] = 0.0
        self.masks[0] = 1.0
        self.step = 0

    def insert(self, obs, rnn_actor, rnn_critic, actions, log_probs, values,
               rewards, masks, bad_masks=None):
        """everything for ONE tick. obs/rnn/masks land in slot t+1, the rest in t."""
        t = self.step
        if bad_masks is not None:
            self.bad_masks[t + 1] = bad_masks
        self.obs[t + 1] = obs
        self.rnn_actor[t + 1] = rnn_actor
        self.rnn_critic[t + 1] = rnn_critic
        self.masks[t + 1] = masks
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.step = t + 1

    def compute_returns(self, next_values):
        """
        next_values: (N, 1) -- the critic's guess for the state we STOPPED on.
        Without it the rollout would pretend the world ends at T and
        undervalue everything near the cut.
        """
        self.values[self.T] = next_values

        gae = np.zeros((self.N, 1), dtype=np.float32)
        for t in reversed(range(self.T)):
            #masks[t+1] == 0 -> the episode ended at t, so V(s_{t+1}) belongs
            #to a different episode. zero it out, and zero the running gae too.
            nonterminal = self.masks[t + 1]

            delta = (self.rewards[t]
                     + self.gamma * self.values[t + 1] * nonterminal
                     - self.values[t])

            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae

            #PROPER TIME LIMITS. bad_masks[t+1] == 0 means the episode did not
            #really end at t, it was cut off by the horizon. The line above has
            #already zeroed the bootstrap as if the world ended -- which is
            #wrong, the game was still going. Zeroing gae here throws that bad
            #advantage away and falls back to the critic's own estimate
            #(returns[t] = 0 + values[t]), instead of teaching it that every
            #state near the horizon is worthless.
            #This fires on ESSENTIALLY EVERY episode here: the only true
            #terminal is running the order list out, which an untrained policy
            #never does.
            gae = gae * self.bad_masks[t + 1]

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def after_update(self):
        """carry the last tick forward to become the first tick of the next rollout"""
        self.obs[0] = self.obs[-1]
        self.rnn_actor[0] = self.rnn_actor[-1]
        self.rnn_critic[0] = self.rnn_critic[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]
        self.step = 0

    def recurrent_generator(self, num_mini_batch, device, data_chunk_length=10):
        """
        Yields minibatches of CHUNKS -- short contiguous runs of ticks, each
        carrying the GRU memory that was live at its own start tick.

        WHY CHUNKS AND NOT WHOLE T-TICK SEQUENCES
        The obvious version hands each minibatch the full T ticks for a subset
        of rows. It is correct, but with T=200 it means:
          * the GRU loop in rnn.py runs 200 sequential steps per minibatch
          * backprop-through-time holds all 200 activations in memory
          * you can never have more minibatches than you have rows
        Cutting T into chunks of data_chunk_length (10 by default) shortens the
        BPTT chain 20x, drops the memory, and decouples the minibatch count
        from N -- N*(T/10) chunks to spread around instead of just N.

        WHAT MAKES IT LEGAL
        A chunk starting at tick s is only meaningful if the GRU memory going
        INTO tick s is the real one. That is why the buffer stores rnn_actor /
        rnn_critic at EVERY tick, not just tick 0: chunk (row n, start s) picks
        up rnn_actor[s, n] and carries on from exactly where the rollout was.
        Episode boundaries inside a chunk still get handled -- masks ride along
        and rnn.py zeroes the memory wherever one lands.

        Each minibatch yields:
            obs             (Lc*mb, 23, W, H)     Lc = data_chunk_length
            rnn_*           (mb, L, H)            memory at each chunk's start
            masks           (Lc*mb, 1)
            ...
        RNNLayer sees x.size(0) = Lc*mb != hxs.size(0) = mb, takes its training
        branch, and un-flattens with view(Lc, mb, -1) -- so the flatten below
        must be TICK-MAJOR. Do not swap it.
        """
        T, N = self.T, self.N
        Lc = min(data_chunk_length, T)
        n_chunks_per_row = T // Lc
        assert n_chunks_per_row >= 1, f"episode_length {T} < data_chunk_length {Lc}"

        #every (row, start_tick) pair is one independent training chunk
        chunks = [(n, c * Lc) for n in range(N) for c in range(n_chunks_per_row)]
        total = len(chunks)
        assert total >= num_mini_batch, (
            f"{total} chunks cannot split into {num_mini_batch} minibatches -- "
            f"raise n_rollout_threads or lower data_chunk_length")

        #normalize advantages across the whole batch -- standard PPO. keeps the
        #gradient scale stable no matter how big the rewards happen to be.
        #
        #GUARD: if every reward in the rollout was 0 the advantages are pure
        #float noise (~1e-9) off a critic that has learned V=0. Dividing that
        #by its own std rescales the noise to UNIT SIZE and hands the policy a
        #full-strength gradient built from nothing -- entropy collapses, clip
        #fraction pins near 0.5, and the run looks busy while learning garbage.
        #Below the threshold there is no signal to normalize, so emit zeros and
        #let the update be a no-op instead of amplifying rounding error.
        adv = self.advantages
        adv_std = adv.std()
        if adv_std < 1e-6:
            adv = np.zeros_like(adv)
        else:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)

        perm = np.random.permutation(total)
        mb_size = total // num_mini_batch

        def _t(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(device)

        def _gather(arr, picks):
            #(Lc, mb, ...) tick-major, then flattened to (Lc*mb, ...)
            out = np.stack([arr[s:s + Lc, n] for n, s in picks], axis=1)
            return out.reshape(-1, *arr.shape[2:])

        for i in range(num_mini_batch):
            idx = perm[i * mb_size:(i + 1) * mb_size]
            if len(idx) == 0:
                continue
            picks = [chunks[j] for j in idx]

            yield dict(
                obs=_t(_gather(self.obs, picks)),
                #the memory live at each chunk's OWN start tick -- not tick 0
                rnn_actor=_t(np.stack([self.rnn_actor[s, n] for n, s in picks])),
                rnn_critic=_t(np.stack([self.rnn_critic[s, n] for n, s in picks])),
                actions=_t(_gather(self.actions, picks)),
                old_log_probs=_t(_gather(self.log_probs, picks)),
                returns=_t(_gather(self.returns, picks)),
                masks=_t(_gather(self.masks, picks)),
                advantages=_t(_gather(adv, picks)),
                #V(s) as it was AT COLLECTION TIME. needed for the clipped
                #value loss: the critic gets the same trust-region treatment as
                #the actor, so one update cannot yank the value function far
                #from the predictions the advantages were computed against.
                value_preds=_t(_gather(self.values, picks)),
            )
