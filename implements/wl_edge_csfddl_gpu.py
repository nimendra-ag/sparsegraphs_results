"""Export the deployable EdgeWL + CS-FDDL model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_edge_csfddl_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

This is the cost-sensitive arm of wl_edge_fddl_gpu.py: identical encoder, split
and export machinery, only the dict learner differs (CSFDDLGPU re-weights the
reconstruction and Fisher gradients per class, so the minority class is not
drowned out by the majority).

CAVEAT — EdgeWL hashes its refined edge labels with builtin hash(), which
CPython salts per process. The exported bundle therefore only reloads correctly
in a process started with the same PYTHONHASHSEED; run this script and any
serving process as e.g. `PYTHONHASHSEED=0 python implements/wl_edge_csfddl_gpu.py`
if the bundle is meant to be deployed. The metrics under eval/ are computed
in-process and are unaffected either way.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl_edge import EdgeWL
from dict_learners.csfddl_gpu import CSFDDLGPU


DATASET = "nci_full"
IMPLEMENTATION = "wl_edge_csfddl_gpu"
EXPORT_SEED = 42

# Cost-sensitive weighting scheme. 'inverse_freq' (w_i = N / (C * n_i)) is the
# standard balanced choice; 'inverse_sqrt' rebalances more gently and
# 'effective_number' follows Cui et al., CVPR 2019.
WEIGHTING = "inverse_freq"


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = EdgeWL(seed=EXPORT_SEED)
    dict_learner = CSFDDLGPU(weighting=WEIGHTING, seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
