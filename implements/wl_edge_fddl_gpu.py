"""Export the deployable EdgeWL + FDDL model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_edge_fddl_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

Identical to implements/wl_fddl_gpu.py except for the encoder: EdgeWL refines
edge labels instead of node subtrees. Everything downstream (selection, scaling,
FDDL, evaluation, bundle layout) is shared code, so the two runs are directly
comparable.

CAVEAT — EdgeWL hashes its refined edge labels with builtin hash(), which
CPython salts per process. The exported bundle therefore only reloads correctly
in a process started with the same PYTHONHASHSEED; run this script and any
serving process as e.g. `PYTHONHASHSEED=0 python implements/wl_edge_fddl_gpu.py`
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
from dict_learners.fddl_gpu import FDDLGPU


DATASET = "nci_full"
IMPLEMENTATION = "wl_edge_fddl_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = EdgeWL(seed=EXPORT_SEED)
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
