import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import argparse
import subprocess
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from utils.seeding import seed_everything, derive_seeds
from utils.src_classifier import SRCClassifier
from utils import pipeline
from dict_learners.fddl_gpu import FDDLGPU
from graph_encoders.wl import WL


# --- Monte Carlo CV configuration -------------------------------------------
# Each master seed = one fully reproducible run (its own train/val/test
# partition + its own model initialisation). 5 is the practical minimum;
# raise to 10 for a more stable std if compute allows.
#
# Execution model: to keep every seed's runtime identical, each seed runs in
# its OWN OS process (see the orchestrator in __main__). When that process
# exits, the OS reclaims 100% of its RAM and destroys the CUDA context, so all
# VRAM — including fragmentation and any driver spill into shared memory — is
# returned. This is why seed 5 runs as fast as seed 1 (no cross-seed leak).
MASTER_SEEDS = [41, 42, 43, 44, 45]
DATASET = "nci_full"
IMPLEMENTATION = "wl_fddl_gpu"

PER_RUN_FILE = "per_run_metrics.csv"
SUMMARY_FILE = "summary_mean_std.csv"
TIMINGS_FILE = "per_run_timings.csv"
TIMINGS_SUMMARY_FILE = "summary_timings.csv"


@contextmanager
def _phase(timings, label):
    """Time a block of work, print its duration, and accumulate it under `label`
    in the ordered `timings` dict. Wall-clock (perf_counter) is what matters here
    since GPU/paging stalls are the thing we're hunting."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        timings[label] = timings.get(label, 0.0) + dt
        print(f"    [timer] {label:22s} {dt:8.1f}s", flush=True)


# Load data lazily and once per process. The orchestrator parent never touches
# it (it only spawns workers), so we avoid loading the whole dataset there.
_DATA = None


def _get_data():
    global _DATA
    if _DATA is None:
        data_loader = GraphDataLoader()
        _DATA = (data_loader.nci_full_graphs, data_loader.nci_full_labels)
    return _DATA


def _flatten(prefix, metrics, keys):
    """Pull a subset of scalar metrics out of a result dict under a prefix."""
    return {f"{prefix}/{k}": float(metrics[k]) for k in keys if k in metrics}


def run_once(master_seed):
    """One full pipeline execution under a single master seed.

    Resamples the stratified train/val/test partition, fits WL + FDDL, tunes
    decision thresholds on the validation split, then reports the final metrics
    on the held-out test split (thresholds reused, so no test-set leakage).

    Returns (row, total_atoms, timings) where `timings` is an ordered
    {phase_label: seconds} dict of the wall-clock spent in each phase.
    """
    timings = {}  # ordered: insertion order == pipeline order
    seed_t0 = time.perf_counter()

    with _phase(timings, "data_load"):
        graphs, y = _get_data()

    # Global RNGs (for libraries that read global state, e.g. gensim/sklearn),
    # plus independent sub-seeds for the components we control.
    seed_everything(master_seed)
    s_split, s_wl, s_fddl, s_clf = derive_seeds(master_seed, 4)

    # --- 1. Resample the partition (this is the MC-CV resampling step) -------
    with _phase(timings, "partition"):
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

    # --- 2. Fit WL + FDDL via the shared pipeline (seeds injected) -----------
    wl = WL(seed=s_wl)
    fddl_gpu = FDDLGPU(seed=s_fddl)
    scaler = MaxAbsScaler()

    with _phase(timings, "fit_wl_fddl"):
        pipeline.fit_encoder_and_dictionary(wl, fddl_gpu, G_vocab_train, y_vocab_train)
    total_atoms = fddl_gpu.n_atoms()

    # Codes for every downstream split, computed once and reused.
    with _phase(timings, "sparse_codes_train"):
        X_ML_train_scaled = scaler.fit_transform(pipeline.sparse_codes(wl, fddl_gpu, G_ML_train))
    with _phase(timings, "sparse_codes_val"):
        X_ML_val_scaled = scaler.transform(pipeline.sparse_codes(wl, fddl_gpu, G_val))
    with _phase(timings, "sparse_codes_test"):
        X_ML_test_scaled = scaler.transform(pipeline.sparse_codes(wl, fddl_gpu, G_test))

    # --- 3. Tune thresholds on the VALIDATION split --------------------------
    # Each classifier is timed separately so a data-dependent blow-up (e.g.
    # LinearSVC failing to converge on an unlucky partition) is visible per seed.
    print("Tuning thresholds on the validation split...")
    evaluator_val = Evaluator(
        X_ML_train_scaled, y_ML_train, X_ML_val_scaled, y_val,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=total_atoms, random_state=s_clf,
    )
    with _phase(timings, "val_logreg"):
        evaluator_val.predict_logistic_regression()
    with _phase(timings, "val_gboost"):
        evaluator_val.predict_gradient_boosting()
    with _phase(timings, "val_svm"):
        evaluator_val.predict_svm()
    with _phase(timings, "val_rf"):
        evaluator_val.predict_random_forest()
    val_thresholds = evaluator_val.get_thresholds()

    # --- 4. Final evaluation on the held-out TEST split ----------------------
    print("Evaluating on the held-out test split...")
    evaluator_test = Evaluator(
        X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test,
        implementation=IMPLEMENTATION, dataset=DATASET,
        n_atoms=total_atoms, random_state=s_clf,
        fixed_thresholds=val_thresholds,  # reuse validation-tuned thresholds
    )

    sk_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    src_keys = ["balanced_acc", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]

    row = {}
    with _phase(timings, "test_logreg"):
        lr_res = evaluator_test.predict_logistic_regression()
    row.update(_flatten("LogisticRegression", lr_res, sk_keys))
    with _phase(timings, "test_gboost"):
        gb_res = evaluator_test.predict_gradient_boosting()
    row.update(_flatten("GradientBoosting", gb_res, sk_keys))
    with _phase(timings, "test_svm"):
        svm_res = evaluator_test.predict_svm()
    row.update(_flatten("LinearSVM", svm_res, sk_keys))
    with _phase(timings, "test_rf"):
        rf_res = evaluator_test.predict_random_forest()
    row.update(_flatten("RandomForest", rf_res, sk_keys))

    # SRC-native classifiers (deterministic given the trained dictionary).
    # SRC scores against the encoder-space embeddings (not the sparse codes).
    with _phase(timings, "src_embeddings"):
        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
    with _phase(timings, "src_pure"):
        src_pure = SRCClassifier(fddl_gpu, gamma=0.0)
        row.update(_flatten("SRC_pure", src_pure.evaluate(graph_embeddings_ml_test, y_test), src_keys))
    with _phase(timings, "src_fddl"):
        src_fddl = SRCClassifier(fddl_gpu, gamma=0.5)
        row.update(_flatten("SRC_fddl", src_fddl.evaluate(graph_embeddings_ml_test, y_test), src_keys))

    timings["seed_total"] = time.perf_counter() - seed_t0

    # Per-seed breakdown, biggest phase first, so the bottleneck is obvious.
    print(f"\n----- timing breakdown | seed={master_seed} -----")
    for label, secs in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        share = 100.0 * secs / timings["seed_total"] if timings["seed_total"] else 0.0
        print(f"  {label:22s} {secs:8.1f}s  ({share:4.1f}%)")

    return row, total_atoms, timings


# --- Per-seed persistence ---------------------------------------------------
# Each worker process appends exactly one row here and then exits. The file is
# the single source of truth the aggregator reads; keeping it append-only makes
# the whole run resumable — if the machine dies at seed 4, seeds 1-3 survive and
# only the missing seeds need re-running.

def append_run_row(out_dir, master_seed, total_atoms, row):
    """Append one seed's metrics to per_run_metrics.csv (writing the header if
    the file does not exist yet). Column order follows the row dict, which is
    constructed deterministically in run_once, so every seed lines up."""
    os.makedirs(out_dir, exist_ok=True)
    per_run_path = os.path.join(out_dir, PER_RUN_FILE)
    metric_names = list(row.keys())
    write_header = not os.path.exists(per_run_path)
    with open(per_run_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["master_seed", "total_atoms"] + metric_names)
        writer.writerow([master_seed, total_atoms] + [row[m] for m in metric_names])
    print(f"Appended seed={master_seed} metrics -> {per_run_path}")


def append_timings_row(out_dir, master_seed, timings):
    """Append one seed's per-phase wall-clock (seconds) to per_run_timings.csv.

    Column order follows the `timings` dict, which is built in a fixed pipeline
    order in run_once, so every seed's row lines up under the same header."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, TIMINGS_FILE)
    phase_names = list(timings.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["master_seed"] + phase_names)
        writer.writerow([master_seed] + [f"{timings[p]:.3f}" for p in phase_names])
    print(f"Appended seed={master_seed} timings -> {path}")


def _read_per_run(out_dir):
    """Read per_run_metrics.csv -> (rows_by_seed, metric_names, total_atoms).

    De-duplicates by seed keeping the LAST row for each, so a re-run of a single
    seed cleanly supersedes its earlier result instead of double-counting.
    """
    per_run_path = os.path.join(out_dir, PER_RUN_FILE)
    if not os.path.exists(per_run_path):
        raise FileNotFoundError(f"No per-run metrics found at {per_run_path}")

    with open(per_run_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        metric_names = header[2:]  # drop "master_seed", "total_atoms"
        rows_by_seed = {}
        total_atoms = None
        for raw in reader:
            if not raw:
                continue
            seed = int(raw[0])
            total_atoms = int(raw[1])
            rows_by_seed[seed] = {m: float(v) for m, v in zip(metric_names, raw[2:])}
    return rows_by_seed, metric_names, total_atoms


def aggregate_timings(out_dir):
    """Best-effort cross-seed timing report: per-phase mean +/- std (seconds)
    over every seed, printed and written to summary_timings.csv. Silently does
    nothing if no timings file exists (e.g. an older run). This is the view that
    answers 'which phase varies across seeds', i.e. is a seed slow because of
    SVM convergence, sparse coding, FDDL, etc."""
    path = os.path.join(out_dir, TIMINGS_FILE)
    if not os.path.exists(path):
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        phase_names = header[1:]  # drop "master_seed"
        by_seed = {}
        for raw in reader:
            if not raw:
                continue
            by_seed[int(raw[0])] = [float(v) for v in raw[1:]]

    seeds = sorted(by_seed)
    cols = np.array([by_seed[s] for s in seeds], dtype=float)  # rows=seeds, cols=phases

    print(f"\n{'-'*72}\nPer-phase wall-clock over {len(seeds)} seeds (seconds)\n{'-'*72}")
    print(f"{'phase':22s} {'mean':>9s} {'std':>8s} {'min':>8s} {'max':>8s}")
    summary_path = os.path.join(out_dir, TIMINGS_SUMMARY_FILE)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "mean_sec", "std_sec", "min_sec", "max_sec", "n"])
        for j, phase in enumerate(phase_names):
            v = cols[:, j]
            mean = v.mean()
            std = v.std(ddof=1) if len(v) > 1 else 0.0
            writer.writerow([phase, f"{mean:.2f}", f"{std:.2f}",
                             f"{v.min():.2f}", f"{v.max():.2f}", len(v)])
            print(f"{phase:22s} {mean:9.1f} {std:8.1f} {v.min():8.1f} {v.max():8.1f}")
    print(f"Saved timing summary -> {summary_path}")


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


def aggregate_and_report(out_dir):
    """Read per_run_metrics.csv, print mean +/- sample-std (+ 95% t-CI), and
    persist the summary CSV alongside the per-run file.

    Returns (total_atoms, n_runs) so the orchestrator can finalise the folder
    name with the true atom count.
    """
    rows_by_seed, metric_names, total_atoms = _read_per_run(out_dir)
    seeds = sorted(rows_by_seed)
    rows = [rows_by_seed[s] for s in seeds]

    summary_path = os.path.join(out_dir, SUMMARY_FILE)
    print(f"\n{'='*72}\nMonte Carlo CV over {len(rows)} runs (seeds={seeds})\n{'='*72}")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std_ddof1", "ci95_halfwidth", "n"])
        for m in metric_names:
            vals = np.array([row[m] for row in rows], dtype=float)
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ci = _t_ci_halfwidth(vals)
            writer.writerow([m, f"{mean:.4f}", f"{std:.4f}", f"{ci:.4f}", len(vals)])
            print(f"{m:34s}: {mean:.4f} +/- {std:.4f}   (95% CI +/-{ci:.4f})")

    print(f"\nSaved per-run metrics -> {os.path.join(out_dir, PER_RUN_FILE)}")
    print(f"Saved summary        -> {summary_path}")

    # Cross-seed timing breakdown (best-effort; no-op if timings weren't logged).
    aggregate_timings(out_dir)

    return total_atoms, len(rows)


# --- Worker: run exactly one seed -------------------------------------------

def run_seed_worker(master_seed, out_dir):
    """Run one seed and append its metrics, then return. Designed to be the
    whole lifetime of a subprocess so that process exit frees all RAM/VRAM."""
    print(f"\n########## Monte Carlo CV run | master_seed={master_seed} ##########")
    row, total_atoms, timings = run_once(master_seed)
    append_run_row(out_dir, master_seed, total_atoms, row)
    append_timings_row(out_dir, master_seed, timings)


# --- Orchestrator: one process per seed -------------------------------------

def orchestrate(seeds, fail_fast=False):
    """Spawn one fresh subprocess per seed (sequentially), then aggregate.

    Each subprocess fully exits before the next starts, so the OS reclaims all
    of its memory and CUDA context — every seed runs on a clean machine.

    By default a crashed seed is logged and skipped so the surviving seeds still
    produce a summary (its partial CSV row is simply absent, and the summary's
    per-metric `n` reflects how many seeds actually contributed). Pass
    fail_fast=True to abort the whole run on the first failure instead. If EVERY
    seed fails (the deterministic-bug case) the run exits non-zero rather than
    reporting a hollow success.
    """
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        "results", f"mc_cv_{IMPLEMENTATION}_{DATASET}_{started_at}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Monte Carlo CV | one process per seed | out_dir={out_dir}")

    failed = []
    for seed in seeds:
        print(f"\n>>> launching worker for master_seed={seed} ...")
        # Inherit stdout/stderr so the ~4h worker logs stream live.
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--seed", str(seed), "--out-dir", out_dir],
        )
        if result.returncode != 0:
            msg = f"seed={seed} FAILED (exit code {result.returncode})"
            if fail_fast:
                raise SystemExit(f"Aborting (--fail-fast): {msg}")
            print(f"\n!!! {msg} - skipping and continuing with remaining seeds.")
            failed.append(seed)

    if len(failed) == len(seeds):
        raise SystemExit(
            f"All {len(seeds)} seeds failed; nothing to aggregate. "
            f"This usually means a deterministic bug, not a transient fault."
        )

    total_atoms, _ = aggregate_and_report(out_dir)

    if failed:
        succeeded = [s for s in seeds if s not in failed]
        print(
            f"\n*** WARNING: {len(failed)} of {len(seeds)} seeds failed and were "
            f"skipped: {failed}. Summary is over the {len(succeeded)} surviving "
            f"seeds: {succeeded}. Re-run a failed seed with "
            f"`--seed <N> --out-dir {out_dir}` then `--aggregate` to fill it in."
        )

    # Finalise the folder name with the atom count + start/end timestamps, to
    # match the single-process convention. Best-effort: never lose results to a
    # rename failure (e.g. an open handle / antivirus lock on Windows).
    ended_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(
        "results",
        f"mc_cv_{IMPLEMENTATION}_{DATASET}_atoms{total_atoms}_{started_at}_{ended_at}",
    )
    try:
        os.rename(out_dir, final_dir)
        print(f"\nRun complete -> {final_dir}")
    except OSError as e:
        print(f"\nRun complete -> {out_dir} (folder rename skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo CV for WL+FDDL, one OS process per seed."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Worker mode: run this single master seed and append its metrics.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Run folder for per_run_metrics.csv (required with --seed/--aggregate).",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Aggregate an existing --out-dir into summary_mean_std.csv and exit.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Abort the whole run on the first seed failure "
             "(default: skip the failed seed and continue).",
    )
    args = parser.parse_args()

    if args.aggregate:
        if not args.out_dir:
            parser.error("--aggregate requires --out-dir")
        aggregate_and_report(args.out_dir)
    elif args.seed is not None:
        if not args.out_dir:
            parser.error("--seed requires --out-dir (the shared run folder)")
        run_seed_worker(args.seed, args.out_dir)
    else:
        # Default: orchestrate one process per seed over MASTER_SEEDS.
        orchestrate(MASTER_SEEDS, fail_fast=args.fail_fast)
