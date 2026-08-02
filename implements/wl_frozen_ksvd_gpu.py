"""Export the deployable WL + Frozen K-SVD (GPU) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_frozen_ksvd_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

Frozen K-SVD is supervised: it learns a base dictionary from the majority class
and then freezes it while each remaining class adds residual atoms. It therefore
consumes the vocab_train labels that utils/pipeline.fit_encoder_and_dictionary
already forwards to every learner — the export path itself is unchanged from the
unsupervised arms.

This is the GPU twin of implements/wl_frozen_ksvd.py: same model, same staging,
same protocol, with the sparse coding and the atom updates on the device. The
two arms are reported separately because the ported cores use a dense
deterministic SVD for atom init where the CPU cores use ARPACK, and float32
where the CPU cores use float64 — different (equally valid) dictionaries. See
the parity note in dict_learners/frozen_ksvd_gpu.py.

FrozenKSVDLearnerGPU falls back to CPU torch when no GPU is present, so this
script runs anywhere; it is only *worth* running where CUDA is available.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl import WL
from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU


DATASET = "nci_full"
IMPLEMENTATION = "wl_frozen_ksvd_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = WL(seed=EXPORT_SEED)
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
