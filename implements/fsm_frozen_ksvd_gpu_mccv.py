"""Monte Carlo CV benchmark for FSM + Frozen K-SVD (GPU).

This is the headline number for the FSM + Frozen K-SVD arm: mean +/- std over
`MASTER_SEEDS` independent resamples of the train/val/test partition, run one OS
process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which
implements/wl_frozen_ksvd_gpu_mccv.py and implements/fsm_fddl_gpu_mccv.py also
run on — so this arm is produced by identical machinery and differs from
wl_frozen_ksvd_gpu only in the encoder, and from fsm_fddl_gpu only in the
dictionary learner.

One OS process per seed matters more here than for the CPU arms: it guarantees
each seed starts with a clean CUDA context, so a seed cannot inherit the
previous one's VRAM fragmentation.

FSM replaces WL subtree hashes with ego-network *shape signatures*: for every
node, the radius-`r` neighbourhood summarised by its node/edge counts,
node-label multiset and degree sequence. Its vocabulary is trimmed on a fixed
frequency budget (drop signatures seen fewer than `min_count` times, keep the
top `n_vocab` by global frequency), not the adaptive energy/elbow cut WL and
EdgeWL use — so the embedding width here is set by `min_count`/`n_vocab` rather
than by the data's score curve, and since nothing is scored per feature,
`n_scored_features` in the run manifest is None for this arm.

WIDTH — unlike gspan_cork_frozen_ksvd_gpu, this arm needs no width check before
a long run: FSM's budget caps the embedding at `n_vocab` columns (1000 by
default), so the learner's 96-atom base plus 32 residual atoms per added class
stays comfortably undercomplete. Re-check `n_components_base` against the
`n_selected_features` column only if `n_vocab` is lowered.

Frozen K-SVD is the only supervised stage: FSM's trim never reads the labels
(unlike gspan_cork_frozen_ksvd_gpu, where CORK uses them too). As in
implements/wl_frozen_ksvd_gpu_mccv.py, its dictionary blocks are not equal-width
(base vs per-class residual), so the SRC arms — which slice a dictionary at a
fixed stride — are recorded as NaN.

Signatures are plain ASCII strings (no builtin hash()), so unlike EdgeWL this
arm carries no PYTHONHASHSEED caveat.

COST — read `per_run_timings.csv` before scaling this up. `fit_encoder_dict`
covers the ego-signature extraction plus the staged K-SVD fit, and the
`sparse_codes_*` phases cover OMP coding. The encoder is cheap here (one ego
graph per node, no isomorphism testing), so if either phase dominates the levers
are on the learner: `n_components_base`, `n_components_residual`, `max_iter` and
`n_non_zero_coefs`.

Usage
-----
    python implements/fsm_frozen_ksvd_gpu_mccv.py                 # full run
    python implements/fsm_frozen_ksvd_gpu_mccv.py --seed 7 --out-dir results/<run>
    python implements/fsm_frozen_ksvd_gpu_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.fsm import FSM
from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "fsm_frozen_ksvd_gpu"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: FSM(seed=seed),
        dict_learner_factory=lambda seed: FrozenKSVDLearnerGPU(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
