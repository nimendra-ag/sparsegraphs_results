"""Monte Carlo CV benchmark for gSpan-CORK + Frozen K-SVD (GPU).

This is the headline number for the gSpan-CORK + Frozen K-SVD arm: mean +/- std
over `MASTER_SEEDS` independent resamples of the train/val/test partition, run
one OS process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which
implements/wl_frozen_ksvd_gpu_mccv.py and implements/gspan_cork_fddl_gpu_mccv.py
also run on — so this arm is produced by identical machinery and differs from
wl_frozen_ksvd_gpu only in the encoder, and from gspan_cork_fddl_gpu only in the
dictionary learner.

One OS process per seed matters more here than for the CPU arms: it guarantees
each seed starts with a clean CUDA context, so a seed cannot inherit the
previous one's VRAM fragmentation.

gSpan-CORK replaces WL subtree hashes with *frequent subgraph patterns*: gSpan
mines every subgraph occurring in at least `min_support_ratio` of the vocabulary
split, then CORK greedily keeps the patterns that eliminate the most cross-class
correspondences. The embedding is therefore a binary "contains pattern"
indicator rather than L2-normalised counts, and the width is set by CORK's
stopping rule (`cork_tolerance` / `cork_max_features`) rather than by the
adaptive energy/elbow cut. Since there is no per-feature score curve,
`n_scored_features` in the run manifest is None for this arm — only
`n_selected_features` is meaningful.

WIDTH — worth checking before a long run. CORK selects far fewer features than
WL's energy cut does (the existing gspan_cork_fddl_gpu run kept 93 patterns),
while the learner's defaults are a 96-atom base plus 32 residual atoms per added
class. That combination runs, but it fits an overcomplete dictionary to a very
short signal; size `n_components_base` / `n_components_residual` against the
`n_selected_features` column in the results CSV rather than inheriting the
WL-sized defaults.

Both stages are supervised here: CORK uses the vocab_train labels to select
patterns and Frozen K-SVD uses them again to stage base vs residual atoms. As in
implements/wl_frozen_ksvd_gpu_mccv.py, its dictionary blocks are not equal-width
(base vs per-class residual), so the SRC arms — which slice a dictionary at a
fixed stride — are recorded as NaN.

Patterns are matched by VF2 subgraph isomorphism on their labels (no builtin
hash()), so unlike EdgeWL this arm carries no PYTHONHASHSEED caveat.

COST — read `per_run_timings.csv` before scaling this up. `fit_encoder_dict`
covers gSpan mining plus CORK selection plus the staged K-SVD fit; the
`sparse_codes_*` phases cover the VF2 encoding and OMP coding. CORK runs batched
on the GPU and the encoding is parallelised across cores, but gSpan mining is
recursive pure Python and is not parallelised — if `fit_encoder_dict` dominates,
the levers are `min_support_ratio`, `max_num_vertices` and `cork_max_features`,
all set on the encoder below.

Usage
-----
    python implements/gspan_cork_frozen_ksvd_gpu_mccv.py                 # full run
    python implements/gspan_cork_frozen_ksvd_gpu_mccv.py --seed 7 --out-dir results/<run>
    python implements/gspan_cork_frozen_ksvd_gpu_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.gspan_cork import GSpanCORK
from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "gspan_cork_frozen_ksvd_gpu"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: GSpanCORK(seed=seed),
        dict_learner_factory=lambda seed: FrozenKSVDLearnerGPU(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
