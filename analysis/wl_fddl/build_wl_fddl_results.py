"""Collect every WL + FDDL MC-CV run into one table for the wl_fddl analysis.

Sources (all read as-is, nothing imputed):
  results/mc_cv_wl_fddl_gpu_nci_full_id<ID>_atoms<K>_*/
      per_run_metrics.csv    -> <metric>_mean / _std columns
      per_run_timings.csv    -> *_sec_mean columns
      manifest.json          -> split sizes, feature-selection stats
      summary_mean_std.csv   -> cross-check of every mean (build_report)

Grain of wl_fddl_all_results.csv: one row per (dataset, atoms, classifier),
averaged over seeds. A cell run more than once is averaged over the seeds of
all its runs; `n_runs` / `n_seeds` say how many went in.

Run:  python analysis/wl_fddl/build_wl_fddl_results.py
Out:  analysis/wl_fddl/wl_fddl_all_results.csv
      analysis/wl_fddl/build_report.txt
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "wl_fddl_all_results.csv"
OUT_REPORT = OUT_DIR / "build_report.txt"

RUN_GLOB = "mc_cv_wl_fddl_gpu_nci_full_id*_atoms*"
RUN_RE = re.compile(
    r"^mc_cv_wl_fddl_gpu_nci_full_id(?P<id>\d+)_atoms(?P<atoms>\d+)_"
    r"(?P<start>\d{8}_\d{6})_(?P<end>\d{8}_\d{6})$"
)

CLASSIFIERS = ["LogisticRegression", "LinearSVM", "RandomForest",
               "GradientBoosting", "SRC_pure", "SRC_fddl"]
CLASSIFIER_FAMILY = {
    "LogisticRegression": "linear", "LinearSVM": "linear",
    "RandomForest": "tree_ensemble", "GradientBoosting": "tree_ensemble",
    "SRC_pure": "sparse_repr", "SRC_fddl": "sparse_repr",
}
METRICS = ["Macro-F1", "MCC", "ROC-AUC", "Macro-PR-AUC", "Accuracy",
           "Macro-Precision", "Macro-Recall",
           "Minority-F1", "Minority-Precision", "Minority-Recall",
           "Minority-PR-AUC"]

# Classifier -> (validation-phase, test-phase) timing names in per_run_timings.
# SRC classifiers have no validation phase; they share src_embeddings instead.
CLASSIFIER_TIMING = {
    "LogisticRegression": ("val_logreg", "test_logreg"),
    "LinearSVM": ("val_svm", "test_svm"),
    "RandomForest": ("val_rf", "test_rf"),
    "GradientBoosting": ("val_gboost", "test_gboost"),
    "SRC_pure": (None, "src_pure"),
    "SRC_fddl": (None, "src_fddl"),
}
SHARED_TIMINGS = ["data_load", "fit_encoder_dict", "sparse_codes_train",
                  "sparse_codes_val", "sparse_codes_test", "src_embeddings",
                  "seed_total"]

# summary_mean_std.csv is written to 4 dp; allow that rounding.
CHECK_TOL = 6e-5


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(vals: list[float], dp: int = 4) -> float:
    return round(statistics.fmean(vals), dp)


def std(vals: list[float], dp: int = 4) -> float | str:
    return round(statistics.stdev(vals), dp) if len(vals) > 1 else ""


def load_run(run_dir: Path, m: re.Match) -> dict:
    return dict(
        dir=run_dir, dataset_id=int(m["id"]), atoms=int(m["atoms"]),
        started_at=m["start"],
        manifest=json.loads((run_dir / "manifest.json").read_text()),
        summary={r["metric"]: r
                 for r in read_csv(run_dir / "summary_mean_std.csv")},
        per_seed=read_csv(run_dir / "per_run_metrics.csv"),
        seed_timings=read_csv(run_dir / "per_run_timings.csv"),
    )


def check_run(run: dict, problems: list[str]) -> None:
    man = run["manifest"]
    name = run["dir"].name
    if man.get("total_atoms") != run["atoms"]:
        problems.append(f"{name}: manifest total_atoms={man.get('total_atoms')}"
                        f" != folder atoms {run['atoms']}")
    if man.get("dataset_id") != run["dataset_id"]:
        problems.append(f"{name}: manifest dataset_id mismatch")
    if man.get("failed_seeds") or man.get("missing_seeds"):
        problems.append(f"{name}: failed={man.get('failed_seeds')} "
                        f"missing={man.get('missing_seeds')}")
    if len(run["seed_timings"]) != len(run["per_seed"]):
        problems.append(f"{name}: {len(run['per_seed'])} metric rows but "
                        f"{len(run['seed_timings'])} timing rows")
    for clf in CLASSIFIERS:
        for met in METRICS:
            key = f"{clf}/{met}"
            m = statistics.fmean(float(r[key]) for r in run["per_seed"])
            reported = float(run["summary"][key]["mean"])
            if abs(m - reported) > CHECK_TOL:
                problems.append(f"{name}: {key} per-seed mean {m:.5f} "
                                f"!= summary {reported:.4f}")


def cell_rows(dataset_id: int, atoms: int, runs: list[dict]) -> list[dict]:
    """Average one (dataset, atoms) cell over every seed of every run in it."""
    man = runs[0]["manifest"]
    split = man["runs"][0]["split_sizes"]
    seeds = [s for r in runs for s in r["manifest"]["runs"]]
    metric_rows = [row for r in runs for row in r["per_seed"]]
    timing_rows = [row for r in runs for row in r["seed_timings"]]

    def phase(name: str | None) -> float | str:
        if name is None:
            return ""
        return mean([float(t[name]) for t in timing_rows], 2)

    base = {
        "encoder": "wl",
        "dict_learner": "fddl",
        "dataset": man["dataset"],
        "dataset_id": dataset_id,
        "dataset_label": f"NCI{dataset_id}",
        "total_atoms": atoms,
        "n_runs": len(runs),
        "n_seeds": len(metric_rows),
        "split_vocab_train": split["vocab_train"],
        "split_ml_train": split["ml_train"],
        "split_val": split["val"],
        "split_test": split["test"],
        "feature_selection": man["runs"][0]["selection"],
        "energy": man["runs"][0]["energy"],
        "n_scored_features_mean": mean([s["n_scored_features"] for s in seeds], 1),
        "n_selected_features_mean": mean([s["n_selected_features"] for s in seeds], 1),
        "n_selected_features_std": std([s["n_selected_features"] for s in seeds], 2),
        "kept_fraction_mean": mean([s["kept_fraction"] for s in seeds]),
    }
    for name in SHARED_TIMINGS:
        base[f"{name}_sec_mean"] = phase(name)

    rows = []
    for clf in CLASSIFIERS:
        row = dict(base, classifier=clf, classifier_family=CLASSIFIER_FAMILY[clf])
        for met in METRICS:
            vals = [float(r[f"{clf}/{met}"]) for r in metric_rows]
            row[f"{met}_mean"] = mean(vals)
            row[f"{met}_std"] = std(vals)
        val_phase, test_phase = CLASSIFIER_TIMING[clf]
        row["clf_val_sec_mean"] = phase(val_phase)
        row["clf_test_sec_mean"] = phase(test_phase)
        rows.append(row)
    return rows


def main() -> None:
    problems: list[str] = []
    skipped: list[str] = []
    cells = defaultdict(list)
    for d in sorted((ROOT / "results").glob(RUN_GLOB)):
        m = RUN_RE.match(d.name)
        if not m or not (d / "summary_mean_std.csv").exists():
            skipped.append(d.name)
            continue
        run = load_run(d, m)
        check_run(run, problems)
        cells[(run["dataset_id"], run["atoms"])].append(run)

    out = []
    for key in sorted(cells):
        out.extend(cell_rows(*key, sorted(cells[key], key=lambda r: r["started_at"])))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    ids = sorted({k[0] for k in cells})
    atoms = sorted({k[1] for k in cells})
    n_runs = sum(len(v) for v in cells.values())
    lines = [f"wl_fddl runs found: {n_runs}  ->  {len(cells)} combinations, "
             f"{len(out)} rows", "",
             "Coverage (runs per combination, '-' = not recorded):",
             "atoms    " + "".join(f"NCI{i:<6}" for i in ids)]
    for a in atoms:
        lines.append(f"{a:<9}" + "".join(
            f"{len(cells.get((i, a), [])) or '-':<9}" for i in ids))
    lines += ["", "Combinations with >1 run (averaged over all their seeds):"]
    lines += [f"  NCI{k[0]} atoms{k[1]}: " + ", ".join(r["dir"].name for r in v)
              for k, v in sorted(cells.items()) if len(v) > 1] or ["  none"]
    lines += ["", "Skipped folders:"] + ([f"  {s}" for s in skipped] or ["  none"])
    lines += ["", "Consistency checks (manifest + per-seed vs summary means):"]
    lines += [f"  {p}" for p in problems] or ["  all passed"]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
