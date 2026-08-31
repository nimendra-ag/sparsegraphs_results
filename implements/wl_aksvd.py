"""Export the deployable WL + FDDL model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_fddl_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

Thanks to utils/export.export_pipeline, adding a future implementation (e.g.
wl_aksvd, fsm_fddl) is just: build its encoder + dict_learner and call
export_pipeline with a new IMPLEMENTATION name.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl import WL
from dict_learners.aksvd import AKSVD


DATASET = "nci_full"
# Which NCI screen to train on (1, 33, 41, 47, 81, 83, 109, 123, 145). It is
# recorded in the bundle's manifest and in its folder name
# ("wl_aksvd_nci_full_id33_atoms<N>_<start>_<end>"), so bundles built on
# different screens never overwrite or get confused with each other.
DATASET_ID = 33
IMPLEMENTATION = "wl_aksvd"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET, id=DATASET_ID)

    encoder = WL(seed=EXPORT_SEED)
    dict_learner = AKSVD(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        dataset_id=DATASET_ID,
        split_seed=EXPORT_SEED,
    )
