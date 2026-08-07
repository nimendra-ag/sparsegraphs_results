import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MaxAbsScaler

from utils import mccv
from utils.evaluator import Evaluator
from utils.graph_data import GraphDataLoader
from utils.seeding import seed_everything
from sf.sf import SF


DATASET = "nci_full"
IMPLEMENTATION = "sf"

# 10-fold stratified CV, seed 1 — unchanged from the original notebook.
N_SPLITS = 10
CV_SEED = 1
# Seed for the SF embedding, the inner validation split and the classifiers.
MODEL_SEED = 42
# Slice of each training fold held back to tune decision thresholds. Taken out
# of the TRAINING half only, so the test fold stays untouched by tuning.
VAL_SIZE = 0.15

# Prefixes for the per-fold metric columns; the first four are evaluated, the
# SRC arms exist only to keep the CSV schema identical across implementations.
SRC_ARMS = ("SRC_pure", "SRC_fddl")


def build_embeddings(graphs):
    """Embed every graph with SF. Returns (X, dimensions).

    Following the notebook, the embedding width is the dataset's average node
    count, so the spectrum is truncated at a length the typical graph can
    actually supply (shorter graphs are zero-padded inside SF).
    """
    dimensions = int(np.round(np.mean([G.number_of_nodes() for G in graphs])))
    print(f"Average number of nodes (embedding dimension) set to: {dimensions}")

    print("Fitting SF model and extracting embeddings...")
    sf_model = SF(dimensions=128, seed=MODEL_SEED)
    sf_model.fit(graphs)
    return sf_model.get_embedding(), dimensions


def run_fold(fold, X, y, train_index, test_index, dimensions, save_report_to=None):
    """Evaluate one CV fold. Returns (row, timings, split_sizes).

    The training half is split once more into an inner training set and a
    validation set: classifiers are fit on the former, decision thresholds tuned
    on the latter, and the tuned thresholds are then reused unchanged on the
    held-out test fold — the same protocol utils/mccv.py runs.
    """
    timings = {}
    fold_t0 = time.perf_counter()

    X_train_full, y_train_full = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    with mccv._phase(timings, "partition"):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=VAL_SIZE, random_state=MODEL_SEED, stratify=y_train_full,
        )

    # Scaled the same way as the dictionary arms, so classifier hyper-parameters
    # (which are shared) see inputs on a comparable scale.
    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # --- tune thresholds on the VALIDATION slice -----------------------------
    print("Tuning thresholds on the validation slice...")
    evaluator_val = Evaluator(
        X_train_s, y_train, X_val_s, y_val,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=dimensions, random_state=MODEL_SEED,
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

    # --- final evaluation on the held-out TEST fold --------------------------
    print("Evaluating on the held-out test fold...")
    evaluator_test = Evaluator(
        X_train_s, y_train, X_test_s, y_test,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=dimensions, random_state=MODEL_SEED,
        fixed_thresholds=val_thresholds,  # reuse validation-tuned thresholds
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

    # SF has no class-partitioned dictionary, so the SRC arms are undefined.
    # NaN rather than dropped columns keeps one CSV schema across the project.
    nan_metrics = {k: float("nan") for k in mccv.METRIC_KEYS}
    for arm in SRC_ARMS:
        row.update(mccv._flatten(arm, nan_metrics))

    if save_report_to is not None:
        evaluator_test.save_report(run_folder=save_report_to)

    timings["seed_total"] = time.perf_counter() - fold_t0

    print(f"\n----- timing breakdown | fold={fold} -----")
    for label, secs in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        share = 100.0 * secs / timings["seed_total"] if timings["seed_total"] else 0.0
        print(f"  {label:22s} {secs:8.1f}s  ({share:4.1f}%)")

    split_sizes = {
        "train": len(X_train), "val": len(X_val), "test": len(X_test),
    }
    return row, timings, split_sizes


def fold_manifest_entry(fold, dimensions, split_sizes, seconds):
    """Per-fold provenance, in the shape utils/mccv.py's manifest expects.

    `n_selected_features` is the SF embedding width: for a dictionary arm it is
    the vocabulary the encoder's adaptive cut kept, and for SF it is likewise
    the number of features that actually reach the classifiers.
    """
    return {
        "master_seed": int(fold),          # fold number, see module docstring
        "n_selected_features": int(dimensions),
        "n_scored_features": None,         # SF applies no feature selection
        "selection": None,
        "energy": None,
        "min_features": None,
        "total_atoms": int(dimensions),    # SF has no dictionary
        "encoder_class": "SF",
        "dict_learner_class": None,
        "derived_seeds": {"cv": CV_SEED, "model": MODEL_SEED},
        "split_sizes": {k: int(v) for k, v in split_sizes.items()},
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "seed_total_sec": round(seconds, 3),
    }


def main():
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        "results", f"kfold_{IMPLEMENTATION}_{DATASET}_{started_at}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"{N_SPLITS}-fold CV | out_dir={out_dir}")

    seed_everything(MODEL_SEED)
    graphs, y = GraphDataLoader().load(DATASET)
    y = np.array(y)

    X, dimensions = build_embeddings(graphs)

    manifest_path = mccv.init_manifest(
        out_dir, IMPLEMENTATION, DATASET, seeds=range(1, N_SPLITS + 1),
        started_at=started_at,
    )
    print(f"(manifest at {manifest_path}; folds are recorded under 'master_seed')")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"\n########## {N_SPLITS}-fold CV | fold={fold}/{N_SPLITS} ##########")
        # Keep one fold's full report (metric tables + confusion-matrix images)
        # as a worked example; ten copies would be 120 PNGs for no extra insight.
        report_dir = os.path.join(out_dir, "fold_01_report") if fold == 1 else None
        row, timings, split_sizes = run_fold(
            fold, X, y, train_index, test_index, dimensions,
            save_report_to=report_dir,
        )
        mccv.append_run_row(out_dir, fold, dimensions, row)
        mccv.append_timings_row(out_dir, fold, timings)
        mccv.append_manifest_entry(
            out_dir,
            fold_manifest_entry(fold, dimensions, split_sizes,
                                timings["seed_total"]),
            implementation=IMPLEMENTATION, dataset=DATASET,
        )

    # The aggregator is shared with the MC-CV arms so the summary CSV is
    # byte-for-byte the same shape; it labels rows "seeds", which here are folds.
    print(f"\nAggregating the {N_SPLITS} CV folds "
          f'(the shared aggregator below prints folds as "seeds"):')
    total_atoms, n_runs = mccv.aggregate_and_report(out_dir)
    ended_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    mccv.finalize_manifest(out_dir, total_atoms, failed_seeds=[], ended_at=ended_at)

    # Match the naming convention of the other run folders.
    final_dir = os.path.join(
        "results",
        f"kfold_{IMPLEMENTATION}_{DATASET}_dim{dimensions}_{started_at}_{ended_at}",
    )
    try:
        os.rename(out_dir, final_dir)
        print(f"\nRun complete ({n_runs} folds) -> {final_dir}")
    except OSError as e:
        print(f"\nRun complete ({n_runs} folds) -> {out_dir} "
              f"(folder rename skipped: {e})")


if __name__ == "__main__":
    main()
