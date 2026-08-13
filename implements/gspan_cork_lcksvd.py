"""Export the deployable gSpan-CORK + LC-KSVD model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from gspan_cork_lcksvd_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

This is implements/wl_lcksvd.py with the encoder swapped: gSpan mines frequent
subgraph patterns and CORK selects the discriminative subset, instead of WL
subtree hashes scored and cut by the energy criterion. The dictionary learner
and everything downstream (zero-row filtering, scaling, evaluation, bundle
layout) is shared code, so this arm is directly comparable both with wl_lcksvd
(same learner, different encoder) and with gspan_cork_fddl_gpu (same encoder,
different learner).

LC-KSVD is supervised, so it consumes the vocab_train labels that
utils/pipeline.fit_encoder_and_dictionary already forwards to every learner —
the export path itself is unchanged from the unsupervised arms. Note this makes
the arm doubly supervised: CORK also uses those labels to select patterns.

CAVEATS
  * The embedding is a BINARY indicator matrix (graph contains pattern / does
    not), not WL's L2-normalised counts — that is what CORK's correspondence
    criterion is defined over. LC-KSVD casts it to float on the way in
    (dict_learners/lcksvd.py), so no conversion is needed here.
  * CORK exposes no per-feature score curve, so this arm has no
    `selection_scores_` and the export skips the elbow analytics plot.
  * Cost is dominated by gSpan MINING, which is recursive pure Python and is
    not parallelised. CORK selection runs batched on the GPU and the VF2
    encoding is parallelised across cores, so if this run is slow the lever is
    `min_support_ratio` (raise it) and `max_num_vertices` (lower it), not the
    hardware. `cork_max_features` bounds the selection phase directly.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.gspan_cork import GSpanCORK
from dict_learners.lcksvd import LCKSVDLearner


DATASET = "nci_full"
IMPLEMENTATION = "gspan_cork_lcksvd"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = GSpanCORK(seed=EXPORT_SEED)
    dict_learner = LCKSVDLearner(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
