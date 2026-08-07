"""
Checkpoint history + progress log.  THE FILE THAT EXISTS BECAUSE FCP NEEDS IT.

===========================================================================
WHY
===========================================================================
SP/self_play.py saves every --save_interval episodes to THE SAME FILENAME:

    torch.save(..., os.path.join(save_dir, f"sp_{run_name}.pt"))

so each save destroys the one before it and a finished run leaves exactly one
set of weights: the last. That is fine for "give me a trained policy" and fatal
for anything that needs a policy's HISTORY.

FCP stage 1 is precisely that. Its population is not N trained agents, it is
N agents caught at three different skill levels:

    init    random-ish, before it can do anything
    mid     halfway competent
    final   fully trained

The whole point of FCP is that the ego agent has to cooperate with partners of
varying skill, so it cannot overfit to one specific well-trained dance. With
only final weights you can build a pool of clones, which is not FCP.

MEP, TrajeDi and COLE stage 1 have the same requirement.

===========================================================================
WHAT THIS WRITES
===========================================================================
    <save_dir>/
        progress.jsonl              one line per logged episode
        actor_periodic_<ep>.pt      actor only, every save_interval
        final.pt                    {actor, critic, value_norm, episode, args}
        sp_<run_name>.pt            byte-identical copy of final.pt

`actor_periodic_<ep>.pt` mirrors ZSC-Eval's naming (they write
actor_periodic_<version>.pt) so the pool-selection logic below is the same
logic, reading the same shape of thing.

Periodic saves are ACTOR ONLY on purpose. A pool partner is frozen -- it is
never trained, never bootstrapped, never asked for a value -- so its critic is
dead weight, and at 20 saves x 77 runs the critic would double the footprint on
/scratch1 for nothing. `final.pt` keeps both, because that one you may well
want to resume or fine-tune.

`sp_<run_name>.pt` is written as well so that every existing loader --
eval_checkpoints.py, filter/baseline.py::find_checkpoint -- finds what it
expects without a single line changing anywhere else.

===========================================================================
THE MID CHECKPOINT IS CHOSEN BY SCORE, NOT BY CLOCK
===========================================================================
The tempting shortcut is "mid = the checkpoint halfway through training".
ZSC-Eval does NOT do that (extract_sp_models.py), and the difference matters:

    steak_gc00     first delivery ep  80   ->  halfway is already near-final
    steak_cram     first delivery ep 300   ->  halfway is still a random walk
    steak_none_3   first delivery ep 460   ->  halfway is nothing at all

Learning curves here are sigmoid and the inflection point moves by 400
episodes across layouts, so the same clock position means completely different
skill in different kitchens. ZSC instead picks the checkpoint whose SCORE is
closest to half the final score, which is what "half as good" actually means.

That is only possible if the score at each save is recorded -- hence
progress.jsonl. Without it this rule cannot be applied after the fact, which is
the second reason self_play.py could not have produced an FCP pool.
"""

import json
import os
import shutil

import torch


class RunWriter:
    """Owns everything a run writes to disk. One per training process."""

    def __init__(self, save_dir, run_name, args=None):
        self.save_dir = save_dir
        self.run_name = run_name
        self.args = dict(vars(args)) if args is not None and not isinstance(args, dict) else (args or {})
        os.makedirs(save_dir, exist_ok=True)

        self.progress_path = os.path.join(save_dir, "progress.jsonl")
        #append, not truncate: a requeued or resumed job should extend the
        #history rather than silently delete the part that already ran
        self._progress = open(self.progress_path, "a", buffering=1)

    # ------------------------------------------------------------------ log
    def log(self, record):
        """Append one JSON line. Must include at least episode / env_steps /
        sparse_ret -- the pool selector below reads exactly those."""
        self._progress.write(json.dumps(record) + "\n")

    # ---------------------------------------------------------------- saves
    def save_periodic(self, actor, episode, extra=None):
        """actor_periodic_<episode>.pt -- a NEW file every time, never a
        rewrite. This is the entire reason this module exists."""
        path = os.path.join(self.save_dir, f"actor_periodic_{episode}.pt")
        payload = {"actor": actor.state_dict(), "episode": int(episode),
                   "args": self.args}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def save_final(self, actor, critic, episode, value_norm=None, extra=None):
        """final.pt plus the sp_<run>.pt alias every existing loader expects."""
        payload = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                   "episode": int(episode), "args": self.args}
        if value_norm is not None:
            vn = value_norm.state_dict()
            if vn:
                payload["value_norm"] = vn
        if extra:
            payload.update(extra)

        final = os.path.join(self.save_dir, "final.pt")
        torch.save(payload, final)
        #the compatibility alias. copy rather than symlink -- /scratch1 gets
        #tarred and rsynced around and a dangling symlink is a silent 3am bug
        shutil.copyfile(final, os.path.join(self.save_dir, f"sp_{self.run_name}.pt"))
        return final

    def close(self):
        try:
            self._progress.close()
        except Exception:
            pass


# =========================================================================
# POOL SELECTION -- reading a finished run back for FCP / MEP / TrajeDi
# =========================================================================
def read_progress(save_dir):
    """progress.jsonl -> list of dicts, in order, bad lines skipped."""
    path = os.path.join(save_dir, "progress.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def available_versions(save_dir):
    """Episode numbers that actually have an actor_periodic_<ep>.pt on disk."""
    out = []
    for name in os.listdir(save_dir):
        if name.startswith("actor_periodic_") and name.endswith(".pt"):
            try:
                out.append(int(name[len("actor_periodic_"):-len(".pt")]))
            except ValueError:
                continue
    return sorted(out)


def select_pool_checkpoints(save_dir, score_key="sparse_ret"):
    """One run -> {"init": path, "mid": path, "final": path}.

    ZSC-Eval's rule (extract_sp_models.py), applied to our own log:

        init   the earliest saved checkpoint
        mid    the saved checkpoint whose score is closest to final_score / 2
        final  the last saved checkpoint

    final_score is the mean of the last 5 logged points, not the single last
    one, because sparse_ret is a running average over sampled rollouts and the
    last point alone is noisy -- CARC_RUNS section 5 measured swings of up to
    8.0 between an episode's peak and its final logged value.

    Returns paths for whatever it can find; a run whose progress log is missing
    still yields init/final by position, so a lost log degrades to the
    clock-based rule instead of failing outright.
    """
    versions = available_versions(save_dir)
    if not versions:
        return {}

    picked = {"init": versions[0], "final": versions[-1]}

    #`sparse_ret` is logged as null until the first episode completes in some
    #env, so a row can carry the key with a None value. float(None) raises, and
    #this runs weeks after training when nobody is watching -- drop them.
    rows = [r for r in read_progress(save_dir)
            if r.get(score_key) is not None and "episode" in r]
    if rows:
        scores = [float(r[score_key]) for r in rows]
        final_score = sum(scores[-5:]) / len(scores[-5:])
        target = final_score / 2.0

        #match logged episodes to the versions that were actually written --
        #logging and saving run on different intervals, so they do not line up
        best_ep, best_delta = None, float("inf")
        for r, s in zip(rows, scores):
            ep = int(r["episode"])
            nearest = min(versions, key=lambda v: abs(v - ep))
            delta = abs(target - s)
            if delta < best_delta:
                best_delta, best_ep = delta, nearest
        if best_ep is not None:
            picked["mid"] = best_ep

    #no usable log -> fall back to the clock. documented as second best above.
    picked.setdefault("mid", versions[len(versions) // 2])

    return {tag: os.path.join(save_dir, f"actor_periodic_{ep}.pt")
            for tag, ep in picked.items()}
