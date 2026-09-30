"""Monte Carlo CV benchmark for graph2vec (non-dictionary baseline).

graph2vec is transductive: the Doc2Vec embedding is fitted on ALL graphs
before the partition (same convention as the original paper and as
implements/graph2vec_.py). Each MC-CV seed then resamples a stratified
train/val/test split of the pre-computed embedding matrix, tunes decision
thresholds on val, and reports on the held-out test split — the same
protocol every other _mccv arm uses.

Because the embedding is dense and roughly zero-centred, it is standardised
with StandardScaler (not the MaxAbsScaler the sparse-code arms use).

SRC classifiers are not applicable (graph2vec has no class-partitioned
dictionary), so the SRC columns are recorded as NaN to keep the CSV schema
identical across all implementations.

Usage
-----
    python implements/graph2vec_mccv.py                      # full run
    python implements/graph2vec_mccv.py --seed 7 --out-dir results/<run>
    python implements/graph2vec_mccv.py --aggregate --out-dir results/<run>
    python implements/graph2vec_mccv.py --dataset-id 41
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import mccv
from utils.evaluator import Evaluator
from utils.graph_data import (NCI_IDS, GraphDataLoader, available_datasets,
                              dataset_load_kwargs, dataset_tag,
                              resolve_dataset_id)
from utils.seeding import seed_everything, derive_seeds
from graph2vec.graph2vec import Graph2Vec


DATASET = "nci_full"
DATASET_ID = 1
IMPLEMENTATION = "graph2vec"

MASTER_SEEDS = mccv.default_master_seeds()

DIMENSIONS = 1024
WL_ITERATIONS = 2
EPOCHS = 100
LEARNING_RATE = 0.025
MIN_COUNT = 1
WORKERS = 4
ATTRIBUTED = True

SRC_ARMS = ("SRC_pure", "SRC_fddl")


def build_embeddings(graphs, args):
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
        seed=args.embed_seed,
    )

    t0 = time.perf_counter()
    model.fit(graphs)
    print(f"Pre-training completed in {time.perf_counter() - t0:.1f}s")

    X = model.get_embedding()
    assert X.shape[1] == args.dim, (
        f"graph2vec embedding width {X.shape[1]} != requested dimension {args.dim}"
    )
    return X


def run_once(master_seed, X, y, args, dataset, dataset_id=None):
    timings = {}
    seed_t0 = time.perf_counter()
    tag = dataset_tag(dataset, dataset_id)

    seed_everything(master_seed)
    s_split, _, _, s_clf = derive_seeds(master_seed, 4)

    with mccv._phase(timings, "partition"):
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=mccv.TEST_SIZE,
            random_state=s_split,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=mccv.VAL_SIZE_OF_REMAINDER,
            random_state=s_split,
            stratify=y_train_full,
        )

    with mccv._phase(timings, "scaling"):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

    seed_manifest = {
        "master_seed": int(master_seed),
        "n_selected_features": int(args.dim),
        "n_scored_features": None,
        "selection": None,
        "energy": None,
        "min_features": None,
        "total_atoms": int(args.dim),
        "encoder_class": "Graph2Vec",
        "dict_learner_class": None,
        "derived_seeds": {"split": int(s_split), "classifier": int(s_clf)},
        "split_sizes": {"train": len(X_train), "val": len(X_val),
                        "test": len(X_test)},
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }

    print("Tuning thresholds on the validation split...")
    evaluator_val = Evaluator(
        X_train_s, y_train, X_val_s, y_val,
        implementation=IMPLEMENTATION, dataset=tag,
        n_atoms=args.dim, random_state=s_clf,
    )
    with mccv._phase(timings, "val_logreg"):
        evaluator_val.predict_logistic_regression()
    with mccv._phase(timings, "val_gboost"):
        evaluator_val.predict_gradient_boosting()
    with mccv._phase(timings, "val_svm"):
        evaluator_val.predict_svm()
    with mccv._phase(timings, "val_rf"):
        evaluator_val.predict_random_forest()
    val_thresholds = evaluator_val.get_thresholds()

    print("Evaluating on the held-out test split...")
    evaluator_test = Evaluator(
        X_train_s, y_train, X_test_s, y_test,
        implementation=IMPLEMENTATION, dataset=tag,
        n_atoms=args.dim, random_state=s_clf,
        fixed_thresholds=val_thresholds,
    )

    row = {}
    with mccv._phase(timings, "test_logreg"):
        row.update(mccv._flatten(
            "LogisticRegression", evaluator_test.predict_logistic_regression()))
    with mccv._phase(timings, "test_gboost"):
        row.update(mccv._flatten(
            "GradientBoosting", evaluator_test.predict_gradient_boosting()))
    with mccv._phase(timings, "test_svm"):
        row.update(mccv._flatten("LinearSVM", evaluator_test.predict_svm()))
    with mccv._phase(timings, "test_rf"):
        row.update(mccv._flatten(
            "RandomForest", evaluator_test.predict_random_forest()))

    nan_metrics = {k: float("nan") for k in mccv.METRIC_KEYS}
    for arm in SRC_ARMS:
        row.update(mccv._flatten(arm, nan_metrics))

    timings["seed_total"] = time.perf_counter() - seed_t0
    seed_manifest["seed_total_sec"] = round(timings["seed_total"], 3)

    print(f"\n----- timing breakdown | seed={master_seed} -----")
    for label, secs in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        share = 100.0 * secs / timings["seed_total"] if timings["seed_total"] else 0.0
        print(f"  {label:22s} {secs:8.1f}s  ({share:4.1f}%)")

    return row, args.dim, timings, seed_manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Monte Carlo CV for graph2vec baseline."
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Worker mode: run this single master seed.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Run folder for per_run_metrics.csv.")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate an existing --out-dir and exit.")
    p.add_argument("--dataset", type=str, default=None,
                   choices=available_datasets(),
                   help="Dataset override (e.g. nci_balanced).")
    p.add_argument("--dataset-id", type=int, default=None,
                   help=f"NCI screen id. Known: {list(NCI_IDS)}.")
    p.add_argument("--fail-fast", action="store_true",
                   help="Abort on first seed failure.")
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
    p.add_argument("--embed-seed", type=int, default=42,
                   help="Seed for the graph2vec embedding (fixed across MC-CV seeds)")
    return p.parse_args(argv)


def main():
    args = parse_args()

    dataset = args.dataset if args.dataset is not None else DATASET
    dataset_id = args.dataset_id if args.dataset_id is not None else DATASET_ID

    if args.aggregate:
        if not args.out_dir:
            raise SystemExit("--aggregate requires --out-dir")
        mccv.aggregate_and_report(args.out_dir)
        return

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = dataset_tag(dataset, dataset_id)

    if args.seed is not None:
        if not args.out_dir:
            raise SystemExit("--seed requires --out-dir")
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            "results", f"mc_cv_{IMPLEMENTATION}_{tag}_{started_at}"
        )

    os.makedirs(out_dir, exist_ok=True)

    seed_everything(args.embed_seed)
    graphs, y = GraphDataLoader().load(
        dataset, **dataset_load_kwargs(dataset, dataset_id))
    y = np.array(y)

    X = build_embeddings(graphs, args)

    if args.seed is not None:
        print(f"\n########## Monte Carlo CV run | master_seed={args.seed} ##########")
        row, total_atoms, timings, seed_manifest = run_once(
            args.seed, X, y, args, dataset, dataset_id=dataset_id)
        mccv.append_run_row(out_dir, args.seed, total_atoms, row)
        mccv.append_timings_row(out_dir, args.seed, timings)
        mccv.append_manifest_entry(out_dir, seed_manifest,
                                   implementation=IMPLEMENTATION,
                                   dataset=dataset, dataset_id=dataset_id)
        return

    seeds = MASTER_SEEDS
    print(f"Monte Carlo CV | seeds={seeds} | out_dir={out_dir}")
    mccv.init_manifest(out_dir, IMPLEMENTATION, dataset, seeds, started_at,
                       dataset_id=dataset_id)

    failed = []
    for seed in seeds:
        print(f"\n########## Monte Carlo CV run | master_seed={seed} ##########")
        try:
            row, total_atoms, timings, seed_manifest = run_once(
                seed, X, y, args, dataset, dataset_id=dataset_id)
            mccv.append_run_row(out_dir, seed, total_atoms, row)
            mccv.append_timings_row(out_dir, seed, timings)
            mccv.append_manifest_entry(out_dir, seed_manifest,
                                       implementation=IMPLEMENTATION,
                                       dataset=dataset, dataset_id=dataset_id)
        except Exception as e:
            msg = f"seed={seed} FAILED: {e}"
            if args.fail_fast:
                raise SystemExit(f"Aborting (--fail-fast): {msg}")
            print(f"\n!!! {msg} - skipping.")
            failed.append(seed)

    if len(failed) == len(seeds):
        raise SystemExit("All seeds failed; nothing to aggregate.")

    total_atoms, n_runs = mccv.aggregate_and_report(out_dir)
    ended_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    mccv.finalize_manifest(out_dir, total_atoms, failed, ended_at)

    if failed:
        print(f"\n*** WARNING: {len(failed)} of {len(seeds)} seeds failed: {failed}")

    final_dir = os.path.join(
        "results",
        f"mc_cv_{IMPLEMENTATION}_{tag}_dim{args.dim}_{started_at}_{ended_at}",
    )
    try:
        os.rename(out_dir, final_dir)
        print(f"\nRun complete ({n_runs} seeds) -> {final_dir}")
    except OSError as e:
        print(f"\nRun complete ({n_runs} seeds) -> {out_dir} "
              f"(folder rename skipped: {e})")


if __name__ == "__main__":
    main()
