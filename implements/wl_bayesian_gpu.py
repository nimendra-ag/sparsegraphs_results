"""Export the deployable WL + Bayesian (BPFA, GPU) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_bayesian_gpu_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

This is the GPU twin of implements/wl_bayesian.py: same model, same sampler,
same protocol, with the linear algebra on the device. The two arms are reported
separately because torch's RNG is not NumPy's, so they draw different (equally
valid) posterior sample paths — see the parity note in
dict_learners/bayesian_dl_gpu.py.

BayesianDLGPU falls back to CPU torch when no GPU is present, so this script
runs anywhere; it is only *worth* running where CUDA is available.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl import WL
from dict_learners.bayesian_gpu import BayesianDLGPU


DATASET = "nci_full"
IMPLEMENTATION = "wl_bayesian_gpu"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = WL(seed=EXPORT_SEED)
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
