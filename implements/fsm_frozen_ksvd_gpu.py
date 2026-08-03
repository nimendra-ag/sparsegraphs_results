"""Export the deployable FSM + Frozen K-SVD (GPU) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from fsm_frozen_ksvd_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

Identical to implements/wl_frozen_ksvd_gpu.py except for the encoder: FSM
vectorises each graph as a bag of ego-network *shape signatures* (radius-`r`
neighbourhood around every node, summarised by node/edge counts, node-label
multiset and degree sequence) instead of WL subtree hashes. The learner and
everything downstream (zero-row filtering, scaling, evaluation, bundle layout)
is shared code, so this arm is directly comparable both with wl_frozen_ksvd_gpu
(same learner, different encoder) and with fsm_fddl_gpu (same encoder, different
learner).

Frozen K-SVD is supervised: it learns a base dictionary from the majority class
and then freezes it while each remaining class adds residual atoms. It therefore
consumes the vocab_train labels that utils/pipeline.fit_encoder_and_dictionary
already forwards to every learner — the export path itself is unchanged from the
unsupervised arms. Note this makes it the ONLY supervised stage in this arm:
unlike gspan_cork_frozen_ksvd_gpu, the labels play no part in choosing the
features.

FrozenKSVDLearnerGPU falls back to CPU torch when no GPU is present, so this
script runs anywhere; it is only *worth* running where CUDA is available.

CAVEATS
  * FSM trims its vocabulary on a fixed frequency budget (drop signatures seen
    fewer than `min_count` times, keep the top `n_vocab` by global frequency)
    rather than the adaptive energy/elbow cut WL and EdgeWL use. It exposes no
    `selection_scores_`, so the export skips the elbow analytics plot here.
  * WIDTH is predictable in this arm, which is the one thing it has over
    gspan_cork_frozen_ksvd_gpu: the embedding is at most `n_vocab` columns wide
    (1000 by default) rather than however many patterns CORK happened to keep,
    so the learner's 96-atom base plus 32 residual atoms per added class is
    comfortably undercomplete. If `n_vocab` is lowered, re-check it against
    `n_components_base` — a base wider than the feature dimension is an
    overcomplete fit to a short signal.
  * `base_label` defaults to -1, i.e. the learner assumes -1 is the majority
    class — true for nci_full, as in the WL arm.
  * The embedding is a RAW COUNT matrix, not WL's L2-normalised counts, so row
    magnitude scales with molecule size and the staged K-SVD objective weights
    large molecules proportionally more.

Signatures are plain ASCII strings (no builtin hash()), so unlike EdgeWL this
arm carries no PYTHONHASHSEED caveat.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.fsm import FSM
from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU


DATASET = "nci_full"
IMPLEMENTATION = "fsm_frozen_ksvd_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = FSM(seed=EXPORT_SEED)
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
