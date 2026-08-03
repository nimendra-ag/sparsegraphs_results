"""Monte Carlo CV benchmark for FSM + AKSVD.

This is the headline number for the FSM + AKSVD arm: mean +/- std over
`MASTER_SEEDS` independent resamples of the train/val/test partition, run one OS
process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which implements/wl_aksvd_mccv.py and
implements/fsm_fddl_gpu_mccv.py also run on — so this arm is produced by
identical machinery and differs from wl_aksvd only in the encoder, and from
fsm_fddl_gpu only in the dictionary learner.

FSM replaces WL subtree hashes with ego-network *shape signatures*: for every
node, the radius-`r` neighbourhood summarised by its node/edge counts,
node-label multiset and degree sequence. Its vocabulary is trimmed on a fixed
frequency budget (drop signatures seen fewer than `min_count` times, keep the
top `n_vocab` by global frequency), not the adaptive energy/elbow cut WL and
EdgeWL use — so the embedding width here is set by `min_count`/`n_vocab` rather
than by the data's score curve, and since nothing is scored per feature,
`n_scored_features` in the run manifest is None for this arm.

AKSVD is unsupervised and its atoms carry no class identity, so — exactly as in
implements/wl_aksvd_mccv.py — the SRC arms in the results CSV stay NaN. FSM's
trim is unsupervised too, so this arm never reads the vocab_train labels before
the classifier stage.

Signatures are plain ASCII strings (no builtin hash()), so unlike EdgeWL this
arm carries no PYTHONHASHSEED caveat.

Usage
-----
    python implements/fsm_aksvd_mccv.py                 # full run
    python implements/fsm_aksvd_mccv.py --seed 7 --out-dir results/<run>
    python implements/fsm_aksvd_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.fsm import FSM
from dict_learners.aksvd import AKSVD


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "fsm_aksvd"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: FSM(seed=seed),
        dict_learner_factory=lambda seed: AKSVD(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
