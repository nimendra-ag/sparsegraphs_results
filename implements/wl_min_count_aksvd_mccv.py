"""Monte Carlo CV benchmark for WL(min_count) + AKSVD.

The selection-ablation counterpart to implements/wl_aksvd_mccv.py: identical
protocol, identical dataset, identical master seeds, identical dictionary
learner, identical WL subtree hashing — the ONLY difference is that the encoder
trims its vocabulary on FSM/graph2vec's raw-frequency budget instead of WL's
supervised discriminative score + energy cut. Run both and the delta isolates
the selection mechanism.

Keep MIN_COUNT and N_VOCAB in step with wl_min_count_aksvd.py and with any other
wl_min_count_* implementation, so the encoder configuration is constant across
dictionary learners and a learner-vs-learner comparison is not also a comparison
of two different vocabularies.

What to expect in the manifest for this arm (all handled defensively upstream,
no code changes needed): `n_scored_features` is None because nothing is scored
per feature (mccv.py:387), `selection` reads "min_count", and `energy` /
`min_features` are None. `n_selected_features` is the real embedding width and,
with N_VOCAB off, it resamples with the partition the way the energy cut's does
— it is the number of hashes clearing MIN_COUNT on that seed's vocab_train.
(With a cap on it would instead pin to N_VOCAB on every seed, which is how you
can tell at a glance that a run was capped.)

The protocol itself — the split proportions, threshold tuning on val, the
metric set, the CSV layout, the resumable per-seed appends, the per-phase
timing breakdown — lives in utils/mccv.py and is shared with every other arm,
so only the encoder differs.

Usage
-----
    python implements/wl_min_count_aksvd_mccv.py                 # full run
    python implements/wl_min_count_aksvd_mccv.py --seed 7 --out-dir results/<run>
    python implements/wl_min_count_aksvd_mccv.py --aggregate --out-dir results/<run>
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mccv
from graph_encoders.wl_with_min_count import WLMinCount
from dict_learners.aksvd import AKSVD


# --- Monte Carlo CV configuration -------------------------------------------
# Each master seed = one fully reproducible run (its own train/val/test
# partition + its own model initialisation). Shared with the WL baseline so the
# two arms are compared on the SAME partitions, not merely the same protocol.
MASTER_SEEDS = mccv.default_master_seeds()
DATASET = "nci_full"
IMPLEMENTATION = "wl_min_count_aksvd"

# --- the knob this ablation is about -----------------------------------------
# See implements/wl_min_count_aksvd.py for what each one does, and why N_VOCAB is
# off. They are module constants rather than CLI flags on purpose: utils/mccv.main
# owns the argument parser and relaunches THIS script per seed with only
# --seed/--out-dir, so an extra flag would not survive into the worker processes.
#
# To sweep min_count (1 / 2 / 5 / 10 / 20 is a sensible ladder), edit MIN_COUNT
# AND change IMPLEMENTATION (e.g. "wl_min_count_aksvd_mc5") — otherwise the new
# run appends to the previous run's results folder and the rows become
# unattributable. Do NOT change wl_iterations: it must stay at the WL baseline's
# 2, or this stops being a selection ablation.
MIN_COUNT = 2
N_VOCAB = None


if __name__ == "__main__":
    mccv.main(
        encoder_factory=lambda seed: WLMinCount(
            n_vocab=N_VOCAB, min_count=MIN_COUNT, seed=seed
        ),
        dict_learner_factory=lambda seed: AKSVD(seed=seed),
        implementation=IMPLEMENTATION,
        dataset=DATASET,
        worker_script=__file__,
        master_seeds=MASTER_SEEDS,
    )
