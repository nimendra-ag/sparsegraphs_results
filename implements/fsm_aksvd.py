"""Export the deployable FSM + AKSVD model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from fsm_aksvd_mccv.py (Monte
Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split test
metrics.

Identical to implements/wl_aksvd.py except for the encoder: FSM vectorises each
graph as a bag of ego-network *shape signatures* (radius-`r` neighbourhood
around every node, summarised by node/edge counts, node-label multiset and
degree sequence) instead of WL subtree hashes. Everything downstream (zero-row
filtering, AKSVD, scaling, evaluation, bundle layout) is shared code, so this
arm is directly comparable both with wl_aksvd (same learner, different encoder)
and with fsm_fddl_gpu (same encoder, different learner).

CAVEATS
  * FSM trims its vocabulary on a fixed frequency budget (drop signatures seen
    fewer than `min_count` times, keep the top `n_vocab` by global frequency)
    rather than the adaptive energy/elbow cut WL and EdgeWL use. It exposes no
    `selection_scores_`, so the export skips the elbow analytics plot here.
  * The embedding is a RAW COUNT matrix, not WL's L2-normalised counts, so row
    magnitude scales with molecule size. AKSVD consumes it unchanged (it
    normalises its atoms, not its inputs); the MaxAbsScaler in the export
    pipeline acts on the sparse codes, downstream of the dictionary.
  * AKSVD is unsupervised and FSM's trim is unsupervised too, so the
    vocab_train labels that utils/pipeline.fit_encoder_and_dictionary forwards
    are unused by both stages of this arm.

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
from dict_learners.aksvd import AKSVD


DATASET = "nci_full"
IMPLEMENTATION = "fsm_aksvd"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = FSM(seed=EXPORT_SEED)
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
