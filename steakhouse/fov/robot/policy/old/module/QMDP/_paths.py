"""Import-path shim so this package can be run from anywhere.

Everything in QMDP/ is NEW code. It reads three existing trees and MODIFIES
NONE of them:

    steakhouse/overcooked_ai_py/                 the mdp
    steakhouse/fov/human/agent/                  the limited-vision human
    steakhouse/fov/robot/policy/old/inference/   SamplingBayesFOVInference
    steakhouse/fov/robot/policy/new/official_baselines/
                                                 the SP baseline (net + features)

`official_baselines` is written to be run with itself on sys.path
(`SP/self_play.py` does the same insert at its top), so `utils.features` and
`algorithm.rMAPPOPolicy` resolve as top-level modules. We reproduce that here
rather than rewriting their imports, because rewriting them would mean editing
files outside this folder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#  .../fov/robot/policy/old/module/QMDP  ->  6 levels up is the repo root
STEAKHOUSE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
BASELINES = os.path.join(STEAKHOUSE, "fov", "robot", "policy", "new",
                         "official_baselines")
REPO_PARENT = os.path.dirname(STEAKHOUSE)

for p in (STEAKHOUSE, BASELINES, REPO_PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

#Checkpoints. Overridable so the same code runs on a laptop and on CARC.
#CARC:   /scratch1/$USER/steakhouse_sp/specialist/<layout>_seed<N>/sp_<layout>.pt
#laptop: QMDP/ckpt/sp_<layout>.pt   (a few pulled down for development)
CKPT_ROOT = os.environ.get(
    "STEAK_SP_CKPT_ROOT",
    "/scratch1/%s/steakhouse_sp/specialist" % os.environ.get("USER", "mishafu"))
LOCAL_CKPT = os.path.join(HERE, "ckpt")


def checkpoint_path(layout, seed=1):
    """Where the self-play specialist for `layout` lives, CARC first."""
    carc = os.path.join(CKPT_ROOT, "%s_seed%d" % (layout, seed),
                        "sp_%s.pt" % layout)
    if os.path.exists(carc):
        return carc
    local = os.path.join(LOCAL_CKPT, "sp_%s.pt" % layout)
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        "no self-play checkpoint for %s (looked in %s and %s)"
        % (layout, carc, local))


def stage_layouts():
    """`SteakHouseGridworld.from_layout_name` only reads
    overcooked_ai_py/data/layouts/. The validated library lives in
    fov/layouts_final/layouts/. Copy any that are missing -- never overwrite,
    so this cannot clobber an existing file (same `cp -n` the sbatch scripts
    in official_baselines/SP already run before training)."""
    import shutil
    import glob
    dst = os.path.join(STEAKHOUSE, "overcooked_ai_py", "data", "layouts")
    src = os.path.join(STEAKHOUSE, "fov", "layouts_final", "layouts")
    if not os.path.isdir(src) or not os.path.isdir(dst):
        return
    for f in glob.glob(os.path.join(src, "*.layout")):
        target = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(target):
            shutil.copy(f, target)


#The 11 layouts whose self-play specialist actually learned to deliver.
#Source: official_baselines/SP/CARC_RUNS.md, re-read off the slurm logs
#2026-08-03 (final sparse of the last logged episode, delivery_reward=20,
#60.00 = perfect). Everything else scored 0.00 and has no usable baseline to
#compare against -- a module cannot "outperform" a policy that never delivers.
USABLE_LAYOUTS = [
    "steak_gc00",   # 60.00
    "steak_gc07",   # 59.60
    "steak_gc06",   # 59.60
    "steak_gs00",   # 59.20
    "steak_gc01",   # 59.20
    "steak_api",    # 59.20
    "steak_gc03",   # 58.80
    "steak_gc04",   # 57.20
    "steak_cram2",  # 57.20
    "steak_gc05",   # 56.40
    "steak_cram",   # 50.40
]
