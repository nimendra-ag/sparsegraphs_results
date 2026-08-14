"""Export the deployable WL(min_count) + AKSVD model bundle.

The selection-ablation counterpart to implements/wl_aksvd.py. Both scripts run
the same dataset, the same split seed, the same AKSVD learner and the same WL
subtree hashing; the ONLY difference is the encoder's vocabulary trim — `WL`'s
supervised discriminative score + energy cut here becomes `WLMinCount`'s
unsupervised raw-frequency budget (see graph_encoders/wl_with_min_count.py).
Any difference in the numbers is therefore attributable to the trim alone.

Keep MIN_COUNT and N_VOCAB in step with wl_min_count_aksvd_mccv.py and with any
other wl_min_count_* implementation, so the encoder configuration is constant
across dictionary learners and a learner-vs-learner comparison is not also a
comparison of two different vocabularies.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_min_count_aksvd_mccv.py
(Monte Carlo CV, mean +/- std over the master seeds), NOT from this script's
single-split test metrics.

Note: no elbow/analytics plot is written into the bundle for this arm. Nothing
is scored per feature, so there is no score curve to draw; utils/export.py
skips it (the `getattr(encoder, "selection_scores_", None)` guard at
export.py:131), exactly as it does for the FSM arms.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl_with_min_count import WLMinCount
from dict_learners.aksvd import AKSVD


DATASET = "nci_full"
IMPLEMENTATION = "wl_min_count_aksvd"
EXPORT_SEED = 42

# --- the knob this ablation is about -----------------------------------------
# MIN_COUNT: drop any subtree hash occurring fewer than this many times across
#   the whole vocab_train corpus (RAW occurrences, repeats within one graph
#   included). 1 disables the filter entirely, which is what
#   implements/graph2vec_.py uses; FSM's default is 2. This is the ONLY thing to
#   sweep here.
# N_VOCAB: OFF (None). It is a hard top-N-by-frequency cap, and leaving it on
#   defeats the experiment: min_count only ever deletes from the frequency tail,
#   which the cap has already deleted, so every min_count below the N-th
#   feature's count yields a byte-identical vocabulary. Set it to an int only if
#   you deliberately want a fixed-width arm — and then do not sweep min_count.
#   The encoder prints which of the two constraints actually bound.
#
# NOT a knob to touch here: wl_iterations. It must stay at the WL baseline's 2
# (graph_encoders/wl.py), or this stops being a selection ablation and becomes a
# comparison of two different feature spaces.
#
# To sweep, edit MIN_COUNT AND change IMPLEMENTATION (e.g. append "_mc5"), or
# the new run will land in the same artifacts/ folder as the previous one.
MIN_COUNT = 10
N_VOCAB = None


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = WLMinCount(
        n_vocab=N_VOCAB,
        min_count=MIN_COUNT,
        seed=EXPORT_SEED,
    )
    dict_learner = AKSVD(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
