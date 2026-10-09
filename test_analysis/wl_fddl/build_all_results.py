"""Build all_results.csv for the WL + FDDL MC-CV runs.

One row per (execution folder, classifier). Every metric is the mean over the
seeds of that execution (std over seeds kept only for the headline metrics so
figures can show error bars). Reads the raw per_run_metrics.csv / timings /
manifest.json of every results/mc_cv_wl_fddl_gpu_* folder.

    python test_analysis/wl_fddl/build_all_results.py
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "all_results.csv"

FOLDER_RE = re.compile(r"^mc_cv_wl_fddl_gpu_(?P<ds>nci_full)_id(?P<id>\d+)_atoms(?P<atoms>\d+)_")
CLASSIFIERS = {
    "LogisticRegression": "linear",
    "LinearSVM": "linear",
    "RandomForest": "tree",
    "GradientBoosting": "tree",
    "SRC_pure": "src",
    "SRC_fddl": "src",
}
METRICS = ["Macro-F1", "MCC", "ROC-AUC", "Macro-PR-AUC", "Accuracy",
           "Macro-Precision", "Macro-Recall", "Minority-F1",
           "Minority-Precision", "Minority-Recall", "Minority-PR-AUC"]
STD_METRICS = ["Macro-F1", "MCC", "Minority-F1", "Minority-PR-AUC"]
CLF_TIMING = {"LogisticRegression": "logreg", "LinearSVM": "svm",
              "RandomForest": "rf", "GradientBoosting": "gboost",
              "SRC_pure": "src_pure", "SRC_fddl": "src_fddl"}


def main():
    rows = []
    for folder in sorted(RESULTS.iterdir()):
        m = FOLDER_RE.match(folder.name)
        if not m:
            continue
        per_run = pd.read_csv(folder / "per_run_metrics.csv")
        timings = pd.read_csv(folder / "per_run_timings.csv")
        manifest = json.loads((folder / "manifest.json").read_text())
        runs = manifest["runs"]
        base = {
            "run_folder": folder.name,
            "dataset": f"NCI{m['id']}",
            "total_atoms": int(m["atoms"]),
            "n_seeds": len(per_run),
            "n_test_graphs": runs[0]["split_sizes"]["test"],
            "n_wl_features_selected": round(sum(r["n_selected_features"] for r in runs) / len(runs), 1),
            "fit_encoder_dict_sec": round(timings["fit_encoder_dict"].mean(), 2),
            # encoder + dictionary fit + sparse coding of every split: the cost of
            # producing the features, shared by all classifiers of the execution
            "features_sec": round(timings[["fit_encoder_dict", "sparse_codes_train",
                                           "sparse_codes_val", "sparse_codes_test"]].sum(axis=1).mean(), 2),
            "seed_total_sec": round(timings["seed_total"].mean(), 2),
        }
        for clf, family in CLASSIFIERS.items():
            row = dict(base, classifier=clf, family=family)
            for met in METRICS:
                col = per_run[f"{clf}/{met}"]
                row[met] = round(col.mean(), 4)
                if met in STD_METRICS:
                    row[f"{met}_std"] = round(col.std(ddof=1), 4)
            tkey = CLF_TIMING[clf]
            if clf.startswith("SRC"):  # SRC also needs the src_embeddings pass
                row["clf_sec"] = round(timings[tkey].mean() + timings["src_embeddings"].mean(), 2)
            else:  # fit+tune on val, then refit+score on test
                row["clf_sec"] = round(timings[f"val_{tkey}"].mean() + timings[f"test_{tkey}"].mean(), 2)
            row["pipeline_sec"] = round(row["features_sec"] + row["clf_sec"], 2)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["dataset", "total_atoms", "run_folder", "classifier"],
                                        key=lambda s: s.str[3:].astype(int) if s.name == "dataset" else s)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows from {df.run_folder.nunique()} executions")
    print(df.groupby(["total_atoms", "dataset"]).run_folder.nunique().unstack().fillna(0).astype(int))


if __name__ == "__main__":
    main()
