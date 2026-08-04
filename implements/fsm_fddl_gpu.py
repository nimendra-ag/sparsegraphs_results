"""Export the deployable FSM + FDDL model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

Identical to implements/wl_fddl_gpu.py except for the encoder: FSM vectorises
each graph as a bag of ego-network *shape signatures* (radius-`r` neighbourhood
around every node, summarised by node/edge counts + degree sequence) instead of
WL subtree hashes. Everything downstream (zero-row filtering, FDDL, scaling,
evaluation, bundle layout) is shared code, so the runs are directly comparable.

The eval/ metrics written here are a single-split provenance figure, NOT a
headline result. The headline for this arm is the MC-CV mean +/- std from
implements/fsm_fddl_gpu_mccv.py (as it is for WL and EdgeWL from their own
_mccv scripts).

CAVEAT — FSM trims its vocabulary on a fixed frequency budget (drop signatures
seen fewer than `min_count` times, keep the top `n_vocab` by global frequency),
rather than the adaptive energy/elbow cut WL and EdgeWL use. It therefore
exposes no `selection_scores_`, and the export skips the elbow analytics plot
for this arm.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.fsm import FSM
from dict_learners.fddl_gpu import FDDLGPU


DATASET = "nci_full"
IMPLEMENTATION = "fsm_fddl_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = FSM(seed=EXPORT_SEED)
    dict_learner = FDDLGPU(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
