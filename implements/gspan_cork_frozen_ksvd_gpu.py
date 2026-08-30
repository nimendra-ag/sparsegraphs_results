"""Export the deployable gSpan-CORK + Frozen K-SVD (GPU) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from
gspan_cork_frozen_ksvd_gpu_mccv.py (Monte Carlo CV, mean +/- std over 5 seeds),
NOT from this script's single-split test metrics.

This is implements/wl_frozen_ksvd_gpu.py with the encoder swapped: gSpan mines
frequent subgraph patterns and CORK selects the discriminative subset, instead
of WL subtree hashes scored and cut by the energy criterion. The learner and
everything downstream (zero-row filtering, scaling, evaluation, bundle layout)
is shared code, so this arm is directly comparable both with wl_frozen_ksvd_gpu
(same learner, different encoder) and with gspan_cork_fddl_gpu (same encoder,
different learner).

Frozen K-SVD is supervised: it learns a base dictionary from the majority class
and then freezes it while each remaining class adds residual atoms. It therefore
consumes the vocab_train labels that utils/pipeline.fit_encoder_and_dictionary
already forwards to every learner — the export path itself is unchanged from the
unsupervised arms. Note this makes the arm doubly supervised: CORK also uses
those labels to select patterns.

FrozenKSVDLearnerGPU falls back to CPU torch when no GPU is present, so this
script runs anywhere; it is only *worth* running where CUDA is available.

CAVEATS
  * The embedding is a BINARY indicator matrix (graph contains pattern / does
    not), not WL's L2-normalised counts — that is what CORK's correspondence
    criterion is defined over. The int8 matrix is promoted to the learner's
    `dtype` (float32) on the way to the device.
  * WIDTH. CORK selects far fewer features than WL's energy cut does — the
    existing gspan_cork_fddl_gpu run kept only 93 patterns. The learner's
    default base dictionary is 96 atoms, i.e. already wider than the input, and
    each added class appends 32 more. That runs (verified), but a base wider
    than the feature dimension is an overcomplete fit to a very short signal, so
    set `n_components_base` / `n_components_residual` against the
    `n_selected_features` this encoder actually produces rather than inheriting
    the WL-sized defaults.
  * `base_label` defaults to -1, i.e. the learner assumes -1 is the majority
    class — true for nci_full, as in the WL arm.
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
from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU


DATASET = "nci_full"
IMPLEMENTATION = "gspan_cork_frozen_ksvd_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = GSpanCORK(seed=EXPORT_SEED)
    dict_learner = FrozenKSVDLearnerGPU(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
