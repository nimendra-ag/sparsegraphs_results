"""Monte Carlo CV benchmark for FSM + CS-FDDL (GPU).

This is the headline number for the FSM cost-sensitive arm: mean +/- std over
`MASTER_SEEDS` independent resamples of the train/val/test partition, run one OS
process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which implements/fsm_fddl_gpu_mccv.py
also runs on — so this and the vanilla FDDL baseline on the same encoder are
produced by identical machinery and only the dict learner differs.

FSM replaces WL subtree hashes with ego-network *shape signatures*: for every
node, the radius-`r` neighbourhood summarised by its node/edge counts and degree
sequence. Its vocabulary is trimmed on a fixed frequency budget (drop signatures
seen fewer than `min_count` times, keep the top `n_vocab` by global frequency),
not the adaptive energy/elbow cut WL and EdgeWL use — so the embedding width
here is set by `min_count`/`n_vocab` rather than by the data's score curve.

Signatures are plain strings (no builtin hash()), so unlike EdgeWL this arm
carries no PYTHONHASHSEED caveat.

Usage
-----
    python implements/fsm_csfddl_gpu_mccv.py                 # full run
    python implements/fsm_csfddl_gpu_mccv.py --seed 7 --out-dir results/<run>
    python implements/fsm_csfddl_gpu_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.fsm import FSM
from dict_learners.csfddl_gpu import CSFDDLGPU


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "fsm_csfddl_gpu"

# Cost-sensitive weighting scheme. Held fixed across seeds so the only thing
# varying between runs is the partition + initialisation.
WEIGHTING = "inverse_freq"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: FSM(seed=seed),
        dict_learner_factory=lambda seed: CSFDDLGPU(weighting=WEIGHTING, seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
