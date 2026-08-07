"""Named experiment grids, so the sbatch array index means something stable.

    python configs.py <grid>            # print the config list, one per line
    python configs.py <grid> --n        # how many tasks the array needs
    python configs.py <grid> --args I   # the argv for array task I

Every grid is a list of dicts of evaluate.py flags. Keeping them here rather
than in the sbatch means the sweep is reproducible from the repo alone and the
array index -> config mapping cannot drift between submissions.

SEED HYGIENE. Tuning grids run on TRAIN seeds (0-15); the confirmation grid
runs on TEST seeds (100-129) that no weight was ever selected on. Anything
reported in RESULTS.md comes from the TEST seeds.
"""
import json
import os
import sys

from _paths import USABLE_LAYOUTS
from cost import DEFAULT_WEIGHTS

#Filled in after the tuning stage. Overridable from the environment so the
#final grid can be re-pointed without editing code mid-campaign.
BEST_BETA = os.environ.get("QMDP_BETA", "2.0")
BEST_WEIGHTS = os.environ.get("QMDP_WEIGHTS", "")
#The baseline's own best play temperature, from grid_basetemp. Applied to BOTH
#arms in the headline comparison so the module is scored against the strongest
#version of the policy it modifies.
BASE_T = os.environ.get("QMDP_BASE_T", "1.0")

TRAIN_SEEDS = "0-15"
TEST_SEEDS = "100-129"
ALL_FOVS = "30,60,90,120,180,360"
CAND = "30,60,90,120,180,360"
#four kitchens spanning the size range, used for tuning so the search does not
#see the layouts it will be scored on more than it has to
TUNE_LAYOUTS = "steak_gc00,steak_gc01,steak_gc03,steak_cram"


def _base(**kw):
    d = dict(fovs=ALL_FOVS, candidate_fovs=CAND, horizon="400",
             n_orders="4", blend="bias", sample=True, progress="0")
    d.update(kw)
    return d


# ---------------------------------------------------------------- grids

def grid_beta():
    """How hard to lean on the module. Baseline arm included in every task."""
    out = []
    for layout in USABLE_LAYOUTS:
        out.append(_base(layouts=layout, seeds=TRAIN_SEEDS,
                         lams="0.0,0.5,1.0,2.0,4.0",
                         tag="beta:%s" % layout))
    return out


def grid_weights():
    """Coordinate search over the cost weights, at a fixed beta.

    Signs are FIXED by the argument in cost.py (costs positive, benefits
    negative); only magnitudes move, and 0 is always in the set so the search
    can switch a term OFF. That keeps the search from finding a win by
    inverting the theory.
    """
    axes = {
        "blindside":       [0.0, 3.0],
        "silent_invalid":  [0.0, 6.0],
        "visible_invalid": [0.0, -2.0],
        "restored":        [0.0, -2.0],
        "strand":          [0.0, 3.0],
        "shift":           [0.5, 2.0],
        "legible_yield":   [0.0, -6.0],
        "approach":        [0.0, 0.5, 3.0],
        "interference":    [0.0, 3.0],
        "self_block":      [0.0, 3.0],
    }
    #centre = the shipped defaults, so every task is a ONE-AXIS deviation from
    #a single reference point and the table reads as an ablation
    centre = dict(DEFAULT_WEIGHTS)
    out = []
    seen = set()
    for name, values in axes.items():
        for v in values:
            w = dict(centre)
            w[name] = v
            key = json.dumps(w, sort_keys=True, separators=(',', ':'))
            if key in seen:
                continue
            seen.add(key)
            out.append(_base(layouts=TUNE_LAYOUTS, seeds=TRAIN_SEEDS,
                             lams="0.0," + BEST_BETA, weights=key,
                             tag="w:%s=%s" % (name, v)))
    #and the centre itself, so the deviations have something to beat
    out.append(_base(layouts=TUNE_LAYOUTS, seeds=TRAIN_SEEDS,
                     lams="0.0," + BEST_BETA,
                     weights=json.dumps(centre, sort_keys=True,
                                        separators=(',', ':')),
                     tag="w:centre"))
    return out


def grid_decompose():
    """Split the cost function in half and run each half alone.

    The nine terms fall into two families, and they answer different questions:

      KNOWLEDGE-BASE terms  blindside, silent_invalid, visible_invalid,
                            restored, strand, shift, legible_yield, approach
                            -- all FOV-graded, all about what the partner knows
      COLLISION terms       interference, self_block
                            -- not FOV-graded, but still model-based: you
                            cannot know you are about to block your partner
                            without predicting where they are going

    If only the collision half works, the honest headline is "predicting the
    partner's motion helps", not "reasoning about their knowledge helps". This
    grid is what decides which sentence RESULTS.md gets to use.
    """
    kb_only = dict(DEFAULT_WEIGHTS)
    kb_only["interference"] = 0.0
    kb_only["self_block"] = 0.0
    coll_only = {k: 0.0 for k in DEFAULT_WEIGHTS}
    coll_only["interference"] = DEFAULT_WEIGHTS["interference"]
    coll_only["self_block"] = DEFAULT_WEIGHTS["self_block"]
    dump = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))
    out = []
    for layout in USABLE_LAYOUTS:
        for name, w in (("kb_only", kb_only), ("collision_only", coll_only),
                        ("full", dict(DEFAULT_WEIGHTS))):
            out.append(_base(layouts=layout, seeds=TRAIN_SEEDS,
                             lams="0.0," + BEST_BETA, weights=dump(w),
                             tag="dec:%s:%s" % (layout, name)))
    return out


def grid_basetemp2():
    """basetemp went to T=3.0 and the baseline was STILL improving, so the
    optimum is further out. Extend. (It cannot be unbounded: T -> inf is a
    uniform random walk, which delivers nothing.)"""
    out = []
    for layout in USABLE_LAYOUTS:
        for t in ("3.0", "4.0", "6.0", "10.0"):
            out.append(_base(layouts=layout, seeds=TRAIN_SEEDS, lams="0.0",
                             base_temperature=t,
                             tag="bt:%s:%s" % (layout, t)))
    return out


def grid_kb_sign():
    """Is the knowledge-base half USELESS, or BACKWARDS?

    `kb_only` came out at exactly chance. Two very different explanations:
    the belief-level signal carries nothing, or it carries something and the
    theory has the sign wrong -- i.e. against this partner the robot should
    SEEK to invalidate the human's plans (take over hard) rather than protect
    them. Negating every KB weight distinguishes those. If the negated version
    wins, the deference story is wrong and RESULTS.md has to say so; if it also
    sits at chance, the signal really is absent.

    Run normalised so this is at the same strength as the positive version.
    """
    kb = dict(DEFAULT_WEIGHTS)
    kb["interference"] = 0.0
    kb["self_block"] = 0.0
    neg = {k: -v for k, v in kb.items()}
    dump = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))
    out = []
    for layout in USABLE_LAYOUTS:
        cfg = _base(layouts=layout, seeds=TRAIN_SEEDS,
                    lams="0.0,1.0,2.0", weights=dump(neg),
                    tag="kbneg:%s" % layout)
        cfg["normalize_q"] = True
        out.append(cfg)
    return out


def grid_basetemp():
    """FAIRNESS. Find the BASELINE's own best operating point before comparing.

    The self-play policy was trained with sampled actions, but the temperature
    it happens to have converged to is not automatically the best one to PLAY
    at -- and measured, it is not: flattening its softmax buys tens of steps.
    Comparing a module against an arbitrarily-tempered baseline would credit
    the module with a gain anyone could have had for free.

    So: sweep the baseline alone over temperature, pick the best T, and run the
    headline A/B with BOTH arms at that T. beta=0 only; nothing here involves
    the module.
    """
    out = []
    for layout in USABLE_LAYOUTS:
        for t in ("1.0", "1.5", "2.0", "3.0"):
            cfg = _base(layouts=layout, seeds=TRAIN_SEEDS, lams="0.0",
                        base_temperature=t,
                        tag="bt:%s:%s" % (layout, t))
            out.append(cfg)
    return out


def grid_beta_at_t():
    """The beta sweep again, with both arms at the baseline's best temperature.

    BASE_T is set from grid_basetemp's answer. Re-tuning beta here matters:
    a flatter baseline is a different thing to bias, so the strength that was
    right at T=1 need not be right at T*.
    """
    out = []
    for layout in USABLE_LAYOUTS:
        cfg = _base(layouts=layout, seeds=TRAIN_SEEDS,
                    lams="0.0,1.0,2.0,4.0,8.0,16.0",
                    base_temperature=BASE_T,
                    tag="betaT:%s" % layout)
        out.append(cfg)
    return out


def grid_norm():
    """The decomposition again, with `beta` made comparable across halves.

    The first pass ran every half at the same beta on the RAW score scale. But
    the knowledge-base terms have about a quarter of the collision terms'
    spread across the six actions, so at a shared beta they were applied at a
    quarter of the strength -- which reads as "the KB half does nothing"
    whether or not it does. `--normalize_q` rescales the scores to unit spread
    first, so beta is a pure strength knob and the two halves are compared at
    equal force. This grid is what decides whether the KB result is real or an
    artefact of scale.
    """
    kb = dict(DEFAULT_WEIGHTS)
    kb["interference"] = 0.0
    kb["self_block"] = 0.0
    coll = {k: 0.0 for k in DEFAULT_WEIGHTS}
    coll["interference"] = DEFAULT_WEIGHTS["interference"]
    coll["self_block"] = DEFAULT_WEIGHTS["self_block"]
    dump = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))
    out = []
    for layout in USABLE_LAYOUTS:
        for name, w in (("kb_only", kb), ("collision_only", coll),
                        ("full", dict(DEFAULT_WEIGHTS))):
            cfg = _base(layouts=layout, seeds=TRAIN_SEEDS,
                        lams="0.0,0.5,1.0,2.0", weights=dump(w),
                        tag="norm:%s:%s" % (layout, name))
            cfg["normalize_q"] = True
            out.append(cfg)
    return out


def grid_collision_fovtest():
    """Does predicting the partner's MOTION need the cone posterior?

    The collision terms turned out to carry the win, and they are not
    FOV-graded in the way the belief terms are. But they are still model-based:
    you cannot know you are about to block your partner without predicting
    where the partner is going, and that prediction comes out of the joint
    (cone, subtask) posterior. This pins how much of it needs the POSTERIOR as
    opposed to any fixed cone.
    """
    coll = {k: 0.0 for k in DEFAULT_WEIGHTS}
    coll["interference"] = DEFAULT_WEIGHTS["interference"]
    coll["self_block"] = DEFAULT_WEIGHTS["self_block"]
    w = json.dumps(coll, sort_keys=True, separators=(",", ":"))
    out = []
    for layout in USABLE_LAYOUTS:
        for m in ("real", "fixed:30", "fixed:360", "noise"):
            out.append(_base(layouts=layout, seeds=TRAIN_SEEDS,
                             lams=BEST_BETA, module_mode=m, weights=w,
                             tag="cft:%s:%s" % (layout, m)))
        out.append(_base(layouts=layout, seeds=TRAIN_SEEDS, lams="0.0",
                         weights=w, tag="cft:%s:baseline" % layout))
    return out


def grid_fovtest():
    """Does INFERRING the cone matter, once the collision terms are removed?

    The first ablation pass showed `fixed:30` and `fixed:360` matching `real`
    at the shipped weights. That is not surprising in hindsight: `interference`
    and `self_block` dominate by firing on ~10-30% of ticks, and neither is
    FOV-graded -- they need the partner's predicted MOTION, not its cone. Any
    cone hypothesis predicts motion about equally well.

    So this grid re-runs the ablation with the collision terms OFF, leaving
    only the eight knowledge-base terms, which are the FOV-graded ones. If the
    inferred posterior beats a pinned cone HERE, that is the FOV result; if it
    does not, the honest headline is partner-motion prediction and RESULTS.md
    says so.
    """
    kb = dict(DEFAULT_WEIGHTS)
    kb["interference"] = 0.0
    kb["self_block"] = 0.0
    w = json.dumps(kb, sort_keys=True, separators=(",", ":"))
    out = []
    for layout in USABLE_LAYOUTS:
        for m in ("real", "fixed:30", "fixed:360", "noise"):
            out.append(_base(layouts=layout, seeds=TRAIN_SEEDS,
                             lams=BEST_BETA, module_mode=m, weights=w,
                             tag="fovtest:%s:%s" % (layout, m)))
        out.append(_base(layouts=layout, seeds=TRAIN_SEEDS, lams="0.0",
                         weights=w, tag="fovtest:%s:baseline" % layout))
    return out


def grid_ablate():
    """The controls. Same beta, same everything, only the module replaced."""
    modes = ["real", "uniform", "noise", "shuffle", "fixed:30", "fixed:360"]
    out = []
    for layout in USABLE_LAYOUTS:
        for m in modes:
            out.append(_base(layouts=layout, seeds=TRAIN_SEEDS,
                             lams=BEST_BETA, module_mode=m,
                             tag="abl:%s:%s" % (layout, m)))
        out.append(_base(layouts=layout, seeds=TRAIN_SEEDS, lams="0.0",
                         tag="abl:%s:baseline" % layout))
    return out


def grid_final():
    """HELD-OUT TABLE. TEST seeds 100-129, which no beta was selected on.

    Two tasks per layout:
      T=1.0     baseline (beta=0) vs baseline+module (beta=BEST_BETA)
      T=1000    a UNIFORM-RANDOM robot, beta=0 vs beta=BEST_BETA. It is here
                because the trained self-play policy paired with this human
                does NOT beat a random robot (RESULTS.md section 0), so "beats
                the baseline" alone is too low a bar -- and because "does the
                module help a policy that knows nothing" is the cleanest
                statement of what the module itself contributes.
    """
    out = []
    for layout in USABLE_LAYOUTS:
        for t in ("1.0", "1000.0"):
            cfg = _base(layouts=layout, seeds=TEST_SEEDS,
                        lams="0.0," + BEST_BETA, base_temperature=t,
                        tag="final:%s:T%s" % (layout, t))
            if BEST_WEIGHTS:
                cfg["weights"] = BEST_WEIGHTS
            out.append(cfg)
    return out


def grid_final_argmax():
    """The same A/B with the robot taking the argmax instead of sampling.

    Reported separately and NOT as the headline: argmax cripples the baseline
    (a deterministic recurrent policy loops), so the gap there is inflated.
    """
    out = []
    for layout in USABLE_LAYOUTS:
        cfg = _base(layouts=layout, seeds=TEST_SEEDS,
                    lams="0.0," + BEST_BETA, base_temperature=BASE_T,
                    tag="argmax:%s" % layout)
        if BEST_WEIGHTS:
            cfg["weights"] = BEST_WEIGHTS
        cfg["sample"] = False
        out.append(cfg)
    return out


GRIDS = {
    "beta": grid_beta,
    "weights": grid_weights,
    "ablate": grid_ablate,
    "decompose": grid_decompose,
    "fovtest": grid_fovtest,
    "norm": grid_norm,
    "basetemp": grid_basetemp,
    "kb_sign": grid_kb_sign,
    "basetemp2": grid_basetemp2,
    "beta_at_t": grid_beta_at_t,
    "collision_fovtest": grid_collision_fovtest,
    "final": grid_final,
    "final_argmax": grid_final_argmax,
}


def to_argv(cfg, out_path):
    """Flags for evaluate.py. NOTHING may contain a space: run_qmdp.sbatch
    expands this with bash word-splitting."""
    argv = []
    for k, v in cfg.items():
        if k in ("sample", "normalize_q"):
            if v:
                argv.append("--%s" % k)
            continue
        argv.extend(["--%s" % k, str(v)])
    argv.extend(["--out", out_path])
    return argv


def main(argv):
    name = argv[0]
    cfgs = GRIDS[name]()
    if "--n" in argv:
        print(len(cfgs))
        return 0
    if "--args" in argv:
        i = int(argv[argv.index("--args") + 1])
        out = argv[argv.index("--out") + 1] if "--out" in argv else "r.jsonl"
        print(" ".join(to_argv(cfgs[i], out)))
        return 0
    for i, c in enumerate(cfgs):
        print(i, json.dumps(c))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
