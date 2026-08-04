"""Export the deployable FSM + Bayesian (BPFA, GPU) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from fsm_bayesian_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

Identical to implements/wl_bayesian_gpu.py except for the encoder: FSM vectorises
each graph as a bag of ego-network *shape signatures* (radius-`r` neighbourhood
around every node, summarised by node/edge counts, node-label multiset and
degree sequence) instead of WL subtree hashes. The learner and everything
downstream (zero-row filtering, scaling, evaluation, bundle layout) is shared
code, so this arm is directly comparable both with wl_bayesian_gpu (same
learner, different encoder) and with fsm_fddl_gpu (same encoder, different
learner).

BayesianDLGPU falls back to CPU torch when no GPU is present, so this script
runs anywhere; it is only *worth* running where CUDA is available.

CAVEATS
  * FSM trims its vocabulary on a fixed frequency budget (drop signatures seen
    fewer than `min_count` times, keep the top `n_vocab` by global frequency)
    rather than the adaptive energy/elbow cut WL and EdgeWL use. It exposes no
    `selection_scores_`, so the export skips the elbow analytics plot here.
  * The embedding is a RAW COUNT matrix, not WL's L2-normalised counts. BPFA's
    Gaussian-noise likelihood assumes a roughly homogeneous scale across rows;
    FSM row magnitudes grow with molecule size, so read the reported `noise std`
    with that in mind — it is fitting one global noise level over rows of very
    different norms.
  * BPFA is unsupervised and FSM's trim is unsupervised too, so the vocab_train
    labels that utils/pipeline.fit_encoder_and_dictionary forwards are unused by
    both stages of this arm.

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
from dict_learners.bayesian_gpu import BayesianDLGPU


DATASET = "nci_full"
IMPLEMENTATION = "fsm_bayesian_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = FSM(seed=EXPORT_SEED)
    dict_learner = BayesianDLGPU(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
