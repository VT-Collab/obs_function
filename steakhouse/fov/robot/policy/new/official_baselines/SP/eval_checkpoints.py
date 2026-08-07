"""
Load trained specialist checkpoints and actually PLAY them.

This is the only honest verification: a checkpoint that loads without error
can still be a policy that never delivers. This rolls episodes and reports
real delivery reward.

RUN IT ON A COMPUTE NODE, NOT THE LOGIN NODE. numpy segfaults on the CARC
login node ("Importing the numpy C-extensions failed" -> Segmentation fault).

    sbatch SP/run_eval_checkpoints.sbatch
    # or
    srun -A biyik_1173 -p main -c 4 --mem 16G -t 00:20:00 \
        python SP/eval_checkpoints.py steak_gc00,steak_api 5

usage:  python eval_checkpoints.py <comma,separated,layouts> [episodes] [stochastic]
    layouts     names as in fov/layouts_final/layouts/
    episodes    rollouts per layout (default 5)
    stochastic  pass "stochastic" to sample instead of argmax

MAX SCORE: delivery_reward=20, is_terminal at len(order_list)<=1, so with
n_orders=4 you deliver 3 -> 60.00 is a perfect episode.
"""
import os
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # official_baselines/
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "..")))

from algorithm.rMAPPOPolicy import R_MAPPOPolicy          # noqa: E402
from utils.env_wrapper import SteakSelfPlayEnv            # noqa: E402

CKDIR = "/scratch1/%s/steakhouse_sp/specialist" % os.environ.get("USER", "mishafu")


def evaluate(layout, episodes=5, deterministic=True, seed=123):
    path = os.path.join(CKDIR, f"{layout}_seed1", f"sp_{layout}.pt")
    if not os.path.exists(path):
        return None, "MISSING", None

    #obs_shape MUST come from the layout itself. CNNBase ends in
    #Linear(32*W*H, 64), so a checkpoint only fits the grid it trained on.
    env = SteakSelfPlayEnv(layout, n_orders=4, horizon=400, seed=seed)
    args = types.SimpleNamespace(hidden_size=64, recurrent_N=1,
                                 lr=5e-4, critic_lr=5e-4, opti_eps=1e-5)
    policy = R_MAPPOPolicy(args, env.obs_shape, env.obs_shape, act_dim=6)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    try:
        policy.actor.load_state_dict(ck["actor"])
        policy.critic.load_state_dict(ck["critic"])
    except Exception as e:
        return None, f"LOAD FAIL {type(e).__name__}", env.obs_shape

    returns = []
    for _ in range(episodes):
        obs = env.reset()
        #(num_agents, recurrent_N, hidden). both chefs are rows.
        rnn = torch.zeros(env.num_agents, 1, 64)
        masks = torch.ones(env.num_agents, 1)
        total = 0.0
        for _t in range(400):
            with torch.no_grad():
                actions, _, rnn = policy.actor(
                    torch.from_numpy(obs).float(), rnn, masks, deterministic)
            obs, sparse, _shaped, done, _trunc = env.step(
                actions.squeeze(-1).numpy())
            total += sparse
            if done:
                break
        returns.append(total)

    return returns, ck.get("episode", "?"), env.obs_shape


def main(argv):
    layouts = argv[1].split(",")
    episodes = int(argv[2]) if len(argv) > 2 else 5
    deterministic = not (len(argv) > 3 and argv[3].startswith("stoch"))

    mode = "argmax" if deterministic else "sampled"
    print(f"mode={mode}  episodes={episodes}  max_possible=60.00\n")
    print(f"{'layout':<18}{'ckpt_ep':>8}{'obs_shape':>15}{'mean':>8}{'best':>6}{'worst':>7}")
    for lay in layouts:
        rets, ep, shape = evaluate(lay, episodes, deterministic)
        if rets is None:
            print(f"{lay:<18}{str(ep):>8}{str(shape or ''):>15}")
            continue
        print(f"{lay:<18}{str(ep):>8}{str(shape):>15}"
              f"{np.mean(rets):8.2f}{max(rets):6.0f}{min(rets):7.0f}")


if __name__ == "__main__":
    main(sys.argv)
