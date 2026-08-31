"""Evaluate graph2vec on NCI1 as a non-dictionary baseline.

Single training pass on ONE fixed 70/30 stratified split. graph2vec has no
dictionary stage — it learns a whole-corpus Doc2Vec embedding over WL subtree
identifiers — so unlike wl_fddl_gpu / fsm_aksvd this arm does not go through
utils.export.export_pipeline and exports no deployable bundle. What it shares
with every other arm is the *reporting*: the same utils.evaluator.Evaluator, the
same four classifiers in the same order, and the same
results/<impl>_<dataset>_atoms<D>_<start>_<end>/ run folder holding the metric
tables and the three confusion-matrix images per model.

`n_atoms` in the report header is the embedding width (δ), which is what the
classifiers actually see — the same convention implements/sf_test.py uses for
the SF baseline's spectrum width.

CAVEATS
  * The 70/30 split has no validation slice, so each model's decision threshold
    is tuned on the test labels. The dictionary arms and sf_test.py instead tune
    on a held-out val split and reuse that threshold on test. These numbers are
    therefore optimistic relative to the MC-CV / k-fold arms and are not
    directly comparable to them.
  * graph2vec is transductive here: the embedding is fitted on ALL graphs
    before the split, so test graphs contribute to the Doc2Vec corpus. This is
    how the original graph2vec evaluation is set up, and it is why the encoder
    is fitted once outside the split rather than per-fold.
  * The embedding is dense and roughly zero-centred, so it is standardised with
    StandardScaler rather than the MaxAbsScaler the sparse-code arms use.

Usage
-----
    python implements/graph2vec_.py                        # defaults below
    python implements/graph2vec_.py --dim 128 --epochs 50
"""

import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.evaluator import Evaluator
from utils.graph_data import GraphDataLoader, dataset_load_kwargs, dataset_tag
from utils.seeding import seed_everything
from graph2vec.graph2vec import Graph2Vec


DATASET = "nci_full"
# NCI screen to run on (1, 33, 41, 47, 81, 83, 109, 123, 145). Carried into the
# report folder name and header through DATASET_TAG.
DATASET_ID = 33
DATASET_TAG = dataset_tag(DATASET, DATASET_ID)
IMPLEMENTATION = "graph2vec"

# Seed for the graph2vec embedding, the train/test split and the classifiers.
MODEL_SEED = 42
# Fraction of the corpus held out for the final evaluation.
TEST_SIZE = 0.5

# graph2vec hyper-parameters. Exposed as CLI flags so a sweep does not need a
# code edit; the constants are the defaults every reported run used.
DIMENSIONS = 1024      # embedding width (delta)
WL_ITERATIONS = 2      # WL iteration depth (D)
EPOCHS = 100           # Doc2Vec training epochs
LEARNING_RATE = 0.025  # HogWild! learning rate (alpha)
MIN_COUNT = 1          # minimum subgraph frequency to keep a WL feature
WORKERS = 4            # gensim worker threads

# The NCI loader writes the atom symbol to each node's "feature" attribute, so
# WL hashing starts from atom identity rather than falling back to degree.
ATTRIBUTED = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="graph2vec baseline on NCI1")
    p.add_argument("--dim", type=int, default=DIMENSIONS,
                   help="Embedding dimension (delta)")
    p.add_argument("--wl-depth", type=int, default=WL_ITERATIONS,
                   help="WL iteration depth (D)")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="Doc2Vec training epochs")
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                   help="HogWild! learning rate (alpha)")
    p.add_argument("--min-count", type=int, default=MIN_COUNT,
                   help="Minimum WL subgraph frequency")
    p.add_argument("--workers", type=int, default=WORKERS,
                   help="Gensim worker threads")
    p.add_argument("--seed", type=int, default=MODEL_SEED, help="Random seed")
    return p.parse_args()


def build_embeddings(graphs, args):
    """Embed every graph with graph2vec. Returns (X, dimensions).

    The whole corpus is embedded in one pass (see the transductive caveat in the
    module docstring), so this runs once before the split rather than per-split.
    """
    print(f"Fitting graph2vec | dim={args.dim} wl_depth={args.wl_depth} "
          f"epochs={args.epochs} min_count={args.min_count}")
    model = Graph2Vec(
        wl_iterations=args.wl_depth,
        attributed=ATTRIBUTED,
        dimensions=args.dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
    )

    t0 = time.perf_counter()
    model.fit(graphs)
    print(f"Pre-training completed in {time.perf_counter() - t0:.1f}s")

    X = model.get_embedding()

    # The width reported in folder names / report headers is the width the
    # classifiers actually see, so assert the two cannot drift apart.
    print(f"graph2vec embedding matrix shape (n_graphs, n_features): {X.shape}")
    print(f"Requested delta = {args.dim} | actual feature count = {X.shape[1]}")
    assert X.shape[1] == args.dim, (
        f"graph2vec embedding width {X.shape[1]} != requested dimension {args.dim}"
    )

    return X, args.dim


def main():
    args = parse_args()
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_t0 = time.perf_counter()

    seed_everything(args.seed)
    graphs, y = GraphDataLoader().load(
        DATASET, **dataset_load_kwargs(DATASET, DATASET_ID))
    y = np.array(y)

    X, dimensions = build_embeddings(graphs, args)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed, stratify=y,
    )

    # Dense, roughly zero-centred embedding -> standardise rather than MaxAbs.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Split sizes | train={len(X_train_s)} test={len(X_test_s)}")

    # Same evaluator, same four models, same order as utils/export.py and
    # implements/sf_test.py, so the saved report is drop-in comparable.
    evaluator = Evaluator(
        X_train_s, y_train, X_test_s, y_test,
        implementation=IMPLEMENTATION, dataset=DATASET_TAG,
        n_atoms=dimensions, random_state=args.seed, started_at=started_at,
    )
    evaluator.predict_logistic_regression()
    evaluator.predict_gradient_boosting()
    evaluator.predict_svm()
    evaluator.predict_random_forest()

    run_folder = evaluator.save_report()
    print(f"\nRun complete in {time.perf_counter() - total_t0:.1f}s -> {run_folder}")


if __name__ == "__main__":
    main()
