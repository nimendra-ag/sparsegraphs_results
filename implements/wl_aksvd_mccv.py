"""Monte Carlo CV benchmark for WL + AKSVD.

This is the headline number for the paper: mean +/- std over `MASTER_SEEDS`
independent resamples of the train/val/test partition, run one OS process per
seed so no seed inherits another's RAM/VRAM.

The protocol itself — the split proportions, threshold tuning on val, the
metric set, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown — lives in utils/mccv.py, unchanged from the version this
script used to carry inline. It is shared with implements/wl_aksvd_mccv.py
so every arm is produced by identical machinery and only the encoder differs.

Usage
-----
    python implements/wl_aksvd_mccv.py                 # full run
    python implements/wl_aksvd_mccv.py --seed 7 --out-dir results/<run>
    python implements/wl_aksvd_mccv.py --aggregate --out-dir results/<run>
    python implements/wl_aksvd_mccv.py --dataset-id 41   # a different NCI screen
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.wl import WL
from dict_learners.aksvd import AKSVD

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
# NCI screen to benchmark (1, 33, 41, 47, 81, 83, 109, 123, 145); override on
# the command line with --dataset-id. It lands in the run folder name
# ("mc_cv_wl_aksvd_nci_full_id33_...") and in the run manifest.
DATASET_ID = 33
IMPLEMENTATION = "wl_aksvd"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: WL(seed=seed),
        dict_learner_factory=lambda seed: AKSVD(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        dataset_id=DATASET_ID,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
