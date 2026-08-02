"""Export the deployable WL + Bayesian (BPFA) model bundle.

A single training pass on ONE fixed split (the same 3-tier train/val/test
partition MC-CV uses: vocab_train 50% / ML_train 20% / val 15% / test 15%). It
saves the artifact bundle the application loads for inference, the raw
sparse-code cache, and — for provenance only — this instance's held-out test
metrics under eval/.

The headline performance for the paper comes from wl_bayesian_mccv.py
(Monte Carlo CV, mean +/- std over 5 seeds), NOT from this script's single-split
test metrics.

BPFA is unsupervised, so it ignores the vocab_train labels that
utils/pipeline.fit_encoder_and_dictionary forwards to every learner — the export
path itself is unchanged from the other arms. `dimensions` is the *maximum*
dictionary size; the beta process prior infers the effective size, which is
recorded in the saved dict_learner config as `effective_dictionary_size`.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from graph_encoders.wl import WL
from dict_learners.bayesian import BayesianDL


DATASET = "nci_full"
IMPLEMENTATION = "wl_bayesian"
EXPORT_SEED = 42


if __name__ == "__main__":
    data_loader = GraphDataLoader()
    graphs, y = data_loader.load(DATASET)

    encoder = WL(seed=EXPORT_SEED)
    dict_learner = BayesianDL(seed=EXPORT_SEED)

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        split_seed=EXPORT_SEED,
    )
