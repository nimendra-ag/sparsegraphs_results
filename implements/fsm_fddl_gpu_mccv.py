"""Monte Carlo CV benchmark for FSM + FDDL (GPU).

This is the headline number for the FSM arm: mean +/- std over `MASTER_SEEDS`
independent resamples of the train/val/test partition, run one OS process per
seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which implements/wl_fddl_gpu_mccv.py
and implements/wl_edge_fddl_gpu_mccv.py also run on — so the FSM, node-WL and
edge-WL numbers are produced by identical machinery and only the encoder
differs.

FSM replaces WL subtree hashes with ego-network *shape signatures*: for every
node, the radius-`r` neighbourhood summarised by its node/edge counts and degree
sequence. Its vocabulary is trimmed on a fixed hybrid budget (frequency +
per-class support variance) followed by a |Pearson| > 0.95 collinearity drop,
not the adaptive energy/elbow cut WL and EdgeWL use — so the embedding width
here is set by `n_vocab` and the correlation filter rather than by the data's
score curve.

Signatures are plain strings (no builtin hash()), so unlike EdgeWL this arm
carries no PYTHONHASHSEED caveat.

Usage
-----
    python implements/fsm_fddl_gpu_mccv.py                 # full run
    python implements/fsm_fddl_gpu_mccv.py --seed 7 --out-dir results/<run>
    python implements/fsm_fddl_gpu_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.fsm import FSM
from dict_learners.fddl_gpu import FDDLGPU


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "fsm_fddl_gpu"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: FSM(seed=seed),
        dict_learner_factory=lambda seed: FDDLGPU(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
