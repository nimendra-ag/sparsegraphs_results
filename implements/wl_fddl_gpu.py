import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from utils.seeding import seed_everything, derive_seeds
from utils.src_classifier import SRCClassifier
from dict_learners.fddl_gpu import FDDLGPU
from graph_encoders.wl import WL


# --- Monte Carlo CV configuration -------------------------------------------
# Each master seed = one fully reproducible run (its own train/val/test
# partition + its own model initialisation). 5 is the practical minimum;
# raise to 10 for a more stable std if compute allows.
MASTER_SEEDS = [41, 42, 43, 44, 45]
DATASET = "nci_full"
IMPLEMENTATION = "wl_fddl_gpu"

# Load data once (deterministic); the *partition* is resampled per run.
data_loader = GraphDataLoader()
graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels


def _flatten(prefix, metrics, keys):
    """Pull a subset of scalar metrics out of a result dict under a prefix."""
    return {f"{prefix}/{k}": float(metrics[k]) for k in keys if k in metrics}


def run_once(master_seed):
    """One full pipeline execution under a single master seed.

    Resamples the stratified train/val/test partition, fits WL + FDDL, tunes
    decision thresholds on the validation split, then reports the final metrics
    on the held-out test split (thresholds reused, so no test-set leakage).

    Returns a flat {metric_name: value} dict of scalar test metrics.
    """
    # Global RNGs (for libraries that read global state, e.g. gensim/sklearn),
    # plus independent sub-seeds for the components we control.
    seed_everything(master_seed)
    s_split, s_wl, s_fddl, s_clf = derive_seeds(master_seed, 4)

    # --- 1. Resample the partition (this is the MC-CV resampling step) -------
    G_train_full, G_test, y_train_full, y_test = train_test_split(
        graphs, y,
        test_size=0.15,
        random_state=s_split,
        stratify=y,
    )
    G_train, G_val, y_train, y_val = train_test_split(
        G_train_full, y_train_full,
        test_size=0.15 / 0.85,  # 15% of the full dataset
        random_state=s_split,
        stratify=y_train_full,
    )
    G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
        G_train, y_train,
        test_size=2 / 7,  # vocab_train : ML_train = 5 : 2 -> 50% / 20% of full
        random_state=s_split,
        stratify=y_train,
    )

    # --- 2. Fit WL + FDDL (seeds injected) -----------------------------------
    wl = WL(seed=s_wl)
    graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)

    fddl_gpu = FDDLGPU(seed=s_fddl)
    scaler = MaxAbsScaler()

    fddl_gpu.fit(training_graph_embeddings=graph_embeddings, y_train=y_vocab_train)
    total_atoms = fddl_gpu.D.shape[1]

    graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
    X_ML_train = fddl_gpu.infer(graph_embeddings_ml_train)
    X_ML_train_scaled = scaler.fit_transform(X_ML_train)

    # --- 3. Tune thresholds on the VALIDATION split --------------------------
    graph_embeddings_ml_val = wl.generate_inferencing_embeddings(G_val)
    X_ML_val = fddl_gpu.infer(graph_embeddings_ml_val)
    X_ML_val_scaled = scaler.transform(X_ML_val)

    evaluator_val = Evaluator(
        X_ML_train_scaled, y_ML_train, X_ML_val_scaled, y_val,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=total_atoms, random_state=s_clf,
    )
    evaluator_val.predict_logistic_regression()
    evaluator_val.predict_gradient_boosting()
    evaluator_val.predict_svm()
    evaluator_val.predict_random_forest()
    val_thresholds = evaluator_val.get_thresholds()

    # --- 4. Final evaluation on the held-out TEST split ----------------------
    graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
    X_ML_test = fddl_gpu.infer(graph_embeddings_ml_test)
    X_ML_test_scaled = scaler.transform(X_ML_test)

    evaluator_test = Evaluator(
        X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=total_atoms, random_state=s_clf,
        fixed_thresholds=val_thresholds,  # reuse validation-tuned thresholds
    )

    sk_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    src_keys = ["balanced_acc", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]

    row = {}
    row.update(_flatten("LogisticRegression", evaluator_test.predict_logistic_regression(), sk_keys))
    row.update(_flatten("GradientBoosting", evaluator_test.predict_gradient_boosting(), sk_keys))
    row.update(_flatten("LinearSVM", evaluator_test.predict_svm(), sk_keys))
    row.update(_flatten("RandomForest", evaluator_test.predict_random_forest(), sk_keys))

    # SRC-native classifiers (deterministic given the trained dictionary)
    src_pure = SRCClassifier(fddl_gpu, gamma=0.0)
    row.update(_flatten("SRC_pure", src_pure.evaluate(graph_embeddings_ml_test, y_test), src_keys))
    src_fddl = SRCClassifier(fddl_gpu, gamma=0.5)
    row.update(_flatten("SRC_fddl", src_fddl.evaluate(graph_embeddings_ml_test, y_test), src_keys))

    return row, total_atoms


def _t_ci_halfwidth(vals, confidence=0.95):
    """95% CI half-width using the t-distribution (correct for small n).

    Falls back to a normal approximation if SciPy is unavailable.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    sem = np.std(vals, ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        crit = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    except ImportError:
        crit = 1.96
    return float(crit * sem)


def aggregate_and_report(rows, total_atoms):
    """Print mean +/- sample-std (+ 95% t-CI) and persist per-run + summary CSV."""
    metric_names = list(rows[0].keys())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"mc_cv_{IMPLEMENTATION}_{DATASET}_atoms{total_atoms}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Per-run raw metrics (one row per seed) — every run reproducible on demand.
    per_run_path = os.path.join(out_dir, "per_run_metrics.csv")
    with open(per_run_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["master_seed"] + metric_names)
        for seed, row in zip(MASTER_SEEDS, rows):
            writer.writerow([seed] + [row[m] for m in metric_names])

    # Summary: mean, sample std (ddof=1), 95% t-CI half-width.
    summary_path = os.path.join(out_dir, "summary_mean_std.csv")
    print(f"\n{'='*72}\nMonte Carlo CV over {len(rows)} runs (seeds={MASTER_SEEDS})\n{'='*72}")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std_ddof1", "ci95_halfwidth", "n"])
        for m in metric_names:
            vals = np.array([row[m] for row in rows], dtype=float)
            mean, std = vals.mean(), vals.std(ddof=1)
            ci = _t_ci_halfwidth(vals)
            writer.writerow([m, f"{mean:.4f}", f"{std:.4f}", f"{ci:.4f}", len(vals)])
            print(f"{m:34s}: {mean:.4f} +/- {std:.4f}   (95% CI +/-{ci:.4f})")

    print(f"\nSaved per-run metrics -> {per_run_path}")
    print(f"Saved summary        -> {summary_path}")
    return out_dir


if __name__ == "__main__":
    rows, atoms = [], None
    for seed in MASTER_SEEDS:
        print(f"\n########## Monte Carlo CV run | master_seed={seed} ##########")
        row, atoms = run_once(seed)
        rows.append(row)
    aggregate_and_report(rows, atoms)
