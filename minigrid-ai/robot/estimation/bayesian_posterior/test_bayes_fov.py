"""Quick accuracy test: run BayesFOVInference over 10 seeds × 3 true FOVs."""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import gymnasium as gym
from human.agents.bayes_agent import BayesHumanAgent
from human.planning.bayes_planner import BayesianPlanner
from robot.estimation.bayesian_posterior.bayes_fov import BayesFOVInference, CANDIDATE_FOVS

SEEDS     = list(range(100))
TRUE_FOVS = [60, 120, 180]
def run_one(seed: int, true_fov: int) -> dict:
    env = gym.make("MiniGrid-LockedRoom-v0")
    env.reset(seed=seed)
    state = env.unwrapped

    true_agent   = BayesHumanAgent(fov=true_fov)
    true_planner = BayesianPlanner()
    true_agent.init_knowledge_base(state)

    inf = BayesFOVInference(candidate_fovs=CANDIDATE_FOVS)
    inf.reset(state)

    done = False
    t = 0
    while not done:
        state = env.unwrapped
        subtask = true_agent.select_subtask(state)
        action  = true_planner.next_action(subtask, state, true_agent.knowledge_base)
        if action is None:
            action = 2

        inf.update(state, action)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1

    env.close()
    final_post = inf.posterior()
    map_fov    = inf.map_fov()
    return {"seed": seed, "true": true_fov, "map": map_fov,
            "correct": map_fov == true_fov, "steps": t,
            "P60":  final_post[60],
            "P120": final_post[120],
            "P180": final_post[180]}


if __name__ == "__main__":
    results = []
    print(f"{'seed':>4}  {'true':>4}  {'map':>4}  {'ok':>3}  "
          f"{'P(60)':>7}  {'P(120)':>7}  {'P(180)':>7}  {'steps':>5}")
    print("-" * 60)

    for true_fov in TRUE_FOVS:
        for seed in SEEDS:
            r = run_one(seed, true_fov)
            results.append(r)
            ok = "✓" if r["correct"] else "✗"
            print(f"{r['seed']:>4}  {r['true']:>4}  {r['map']:>4}  {ok:>3}  "
                  f"{r['P60']:>7.3f}  {r['P120']:>7.3f}  {r['P180']:>7.3f}  "
                  f"{r['steps']:>5}")
        print()

    total   = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}")

    for fov in TRUE_FOVS:
        sub = [r for r in results if r["true"] == fov]
        c   = sum(r["correct"] for r in sub)
        print(f"  true={fov:>3}: {c}/{len(sub)} = {c/len(sub):.0%}")
