"""Monte Carlo CV benchmark for gSpan-CORK + Online DL (scikit-learn).

This is the headline number for the gSpan-CORK + Online DL arm: mean +/- std
over `MASTER_SEEDS` independent resamples of the train/val/test partition, run
one OS process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric set,
the SRC arms, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown) lives in utils/mccv.py, which
implements/wl_online_dl_mccv.py and implements/gspan_cork_fddl_gpu_mccv.py also
run on — so this arm is produced by identical machinery and differs from
wl_online_dl only in the encoder, and from gspan_cork_fddl_gpu only in the
dictionary learner.

gSpan-CORK replaces WL subtree hashes with *frequent subgraph patterns*: gSpan
mines every subgraph occurring in at least `min_support_ratio` of the vocabulary
split, then CORK greedily keeps the patterns that eliminate the most cross-class
correspondences. The embedding is therefore a binary "contains pattern"
indicator rather than L2-normalised counts, and the width is set by CORK's
stopping rule (`cork_tolerance` / `cork_max_features`) rather than by the
adaptive energy/elbow cut. Since there is no per-feature score curve,
`n_scored_features` in the run manifest is None for this arm — only
`n_selected_features` is meaningful.

Online dictionary learning is unsupervised and its atoms carry no class
identity, so — exactly as in implements/wl_online_dl_mccv.py — the SRC arms in
the results CSV stay NaN. The labels are used only by CORK's selection.

Patterns are matched by VF2 subgraph isomorphism on their labels (no builtin
hash()), so unlike EdgeWL this arm carries no PYTHONHASHSEED caveat.

COST — read `per_run_timings.csv` before scaling this up. `fit_encoder_dict`
covers gSpan mining plus CORK selection plus the sklearn DictionaryLearning fit;
the `sparse_codes_*` phases cover the VF2 encoding and OMP coding. This arm has
two independent expensive stages: gSpan mining is recursive pure Python and is
not parallelised (levers: `min_support_ratio`, `max_num_vertices`,
`cork_max_features`, all set on the encoder below), and the sklearn fit scales
with `dimensions` (see dict_learners/online_dl.py). The timing CSV is what tells
you which one you are actually paying for.

Usage
-----
    python implements/gspan_cork_online_dl_mccv.py                 # full run
    python implements/gspan_cork_online_dl_mccv.py --seed 7 --out-dir results/<run>
    python implements/gspan_cork_online_dl_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.gspan_cork import GSpanCORK
from dict_learners.online_dl import OnlineDL


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "gspan_cork_online_dl"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: GSpanCORK(seed=seed),
        dict_learner_factory=lambda seed: OnlineDL(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
