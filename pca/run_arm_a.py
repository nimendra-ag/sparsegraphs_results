"""Arm A, single split: WL (full vocabulary) -> PCA -> FDDL -> classifiers.

The comparison this arm sets up:

    baseline (implements/wl_fddl_gpu.py)
        WL -> discriminative score -> energy cut (10130 -> 1751 hashes) -> FDDL
    Arm A (this file)
        WL -> discriminative score -> NO cut (10130 hashes) -> PCA (-> d) -> FDDL

So the only thing that differs is how ~10k scored subtree hashes become the
dictionary learner's input: a supervised subset, or an unsupervised rotation.

This is a single-split script, and that is a deliberate limit on what it can
claim. It fits one partition, so its metrics carry split luck as well as any real
effect -- treat the output as a wiring check and a first signal, not as a
reportable comparison against the baseline's MC-CV summary.

Examples
--------
    # 2-minute wiring check on a small dataset
    python pca/run_arm_a.py --dataset mutag --dim 128

    # the real single-split run
    python pca/run_arm_a.py --dataset nci_full --dim 256

    # let the variance spectrum choose the width (the PCA analogue of the
    # energy cut) instead of fixing it
    python pca/run_arm_a.py --dataset nci_full --pca-energy 0.99
"""

import sys
import os
# Add the project root to sys.path (matches implements/*.py).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from utils.graph_data import GraphDataLoader
from utils.export import export_pipeline
from dict_learners.fddl_gpu import FDDLGPU
from pca.wl_full import WLFull
from pca.reducer import PCAReducer
from pca.pca_dict_learner import PCAThenDictLearner


def build_implementation_name(dim, pca_energy, center):
    """Encode the arm's distinguishing settings in the artifact folder name.

    Bundles are named `<implementation>_<dataset>_atoms<N>_<start>_<end>`, and
    `atoms` reports FDDL's atom count, which is identical across arms. Without
    the PCA width in the implementation string, two arms at different widths
    would produce visually indistinguishable folders.
    """
    kind = "pca" if center else "svd"
    width = f"{dim}" if dim is not None else f"E{int(round(pca_energy * 100))}"
    return f"wl_{kind}{width}_fddl_gpu"


def main():
    parser = argparse.ArgumentParser(
        description="Arm A: WL (no feature cut) -> PCA -> FDDL, single split."
    )
    parser.add_argument("--dataset", default="nci_full",
                        choices=["nci_full", "mutag", "ptc_mr", "ogbg_molhiv"])
    parser.add_argument("--dim", type=int, default=None,
                        help="Fixed PCA width. Mutually exclusive with --pca-energy.")
    parser.add_argument("--pca-energy", type=float, default=None,
                        help="Variance fraction target, e.g. 0.99. The PCA "
                             "analogue of the encoder's energy cut.")
    parser.add_argument("--max-components", type=int, default=2048,
                        help="Component budget when using --pca-energy.")
    parser.add_argument("--no-center", action="store_true",
                        help="Use TruncatedSVD (no mean-centering) instead of "
                             "PCA. Report it as SVD if you use this.")
    parser.add_argument("--atoms", type=int, default=128,
                        help="FDDL atoms per class (k). Keep this identical to "
                             "the baseline run you are comparing against.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.dim is None) == (args.pca_energy is None):
        parser.error("pass exactly one of --dim or --pca-energy")

    implementation = build_implementation_name(
        args.dim, args.pca_energy, center=not args.no_center
    )
    print(f"=== Arm A | {implementation} | dataset={args.dataset} | "
          f"seed={args.seed} ===", flush=True)

    graphs, y = GraphDataLoader().load(args.dataset)

    encoder = WLFull(seed=args.seed)
    reducer = PCAReducer(
        n_components=args.dim,
        energy=args.pca_energy,
        max_components=args.max_components,
        center=not args.no_center,
        random_state=args.seed,
    )
    # Same FDDL hyperparameters as the baseline; only its input space differs.
    dict_learner = PCAThenDictLearner(
        reducer=reducer,
        inner=FDDLGPU(k=args.atoms, seed=args.seed),
    )

    export_pipeline(
        encoder,
        dict_learner,
        graphs,
        y,
        implementation=implementation,
        dataset=args.dataset,
        split_seed=args.seed,
    )


if __name__ == "__main__":
    main()
