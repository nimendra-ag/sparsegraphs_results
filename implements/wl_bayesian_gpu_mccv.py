"""Monte Carlo CV benchmark for WL + Bayesian dictionary learning (BPFA, GPU).

This is the headline number for the paper: mean +/- std over `MASTER_SEEDS`
independent resamples of the train/val/test partition, run one OS process per
seed so no seed inherits another's RAM/VRAM.

The protocol itself — the split proportions, threshold tuning on val, the
metric set, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown — lives in utils/mccv.py. It is shared with
implements/wl_fddl_gpu_mccv.py so every arm is produced by identical machinery
and only the dictionary learner differs.

One OS process per seed matters more here than for the CPU arm: it guarantees
each seed starts with a clean CUDA context, so a seed cannot inherit the
previous one's VRAM fragmentation.

BPFA is unsupervised, so — like AKSVD and unlike FDDL — its atoms carry no
class identity and the SRC arms are recorded as NaN.

Usage
-----
    python implements/wl_bayesian_gpu_mccv.py                 # full run
    python implements/wl_bayesian_gpu_mccv.py --seed 7 --out-dir results/<run>
    python implements/wl_bayesian_gpu_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.wl import WL
from dict_learners.bayesian_gpu import BayesianDLGPU

# --- Monte Carlo CV configuration -------------------------------------------
# Each master seed = one fully reproducible run (its own train/val/test
# partition + its own model initialisation). 5 is the practical minimum;
# raise to 10 for a more stable std if compute allows.
#
# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "wl_bayesian_gpu"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: WL(seed=seed),
        dict_learner_factory=lambda seed: BayesianDLGPU(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
