"""Monte Carlo CV benchmark for EdgeWL + LC-KSVD.

This is the headline number for the edge-WL + LC-KSVD arm: mean +/- std over
`MASTER_SEEDS` independent resamples of the train/val/test partition, run one
OS process per seed so no seed inherits another's RAM/VRAM.

The protocol itself (split proportions, threshold tuning on val, the metric
set, the CSV layout, the resumable per-seed appends, the per-phase timing
breakdown) lives in utils/mccv.py, which implements/wl_lcksvd_mccv.py also runs
on — so the edge-WL and node-WL numbers are produced by identical machinery and
only the encoder differs.

Usage
-----
    python implements/wl_edge_lcksvd_mccv.py            # full run
    python implements/wl_edge_lcksvd_mccv.py --seed 7 --out-dir results/<run>
    python implements/wl_edge_lcksvd_mccv.py --aggregate --out-dir results/<run>

EdgeWL hashes its refined edge labels with builtin hash(), which CPython salts
per process. Each worker trains AND infers inside one process, so every seed's
metrics are internally consistent and this benchmark is unaffected; only saved
bundles (implements/wl_edge_lcksvd.py) care about the salt.
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.wl_edge import EdgeWL
from dict_learners.lcksvd import LCKSVDLearner


# 5 distinct seeds drawn at random from 0-100 (no repeats: a repeated seed
# would re-run an identical partition and overwrite its own row in the CSV).
# Only the orchestrating parent reads this; workers are told their seed on the
# command line, so re-drawing it in each subprocess is harmless.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "wl_edge_lcksvd"


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: EdgeWL(seed=seed),
        dict_learner_factory=lambda seed: LCKSVDLearner(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
