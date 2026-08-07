"""Aggregate every encoder x dict-learner result into one tidy CSV.

Preference order per (encoder, dict_learner, atoms, classifier) cell:
  1. MC-CV mean over seeds  (results/mc_cv_*)
  2. single-split artifact   (artifacts/*)
  3. an empty placeholder row

Scope: csfddl and wl_edge are excluded (see EXCLUDED_*), leaving a 3 x 6 grid.

Run:  python analysis/build_all_results.py
Out:  analysis/all_results.csv, analysis/build_report.txt
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "analysis" / "all_results.csv"
OUT_REPORT = ROOT / "analysis" / "build_report.txt"

DATASET = "nci_full"

# Longest-first: gspan_cork must be tried before wl, wl_edge before wl.
ENCODER_TOKENS = ["gspan_cork", "wl_edge", "fsm", "wl"]

# Raw implementation token -> canonical learner name. Longest-first.
DICT_LEARNER_TOKENS = [
    ("frozen_ksvd_gpu", "frozen_ksvd"),
    ("frozen_ksvd", "frozen_ksvd"),
    ("bayesian_gpu", "bayesian"),
    ("bayesian", "bayesian"),
    ("csfddl_gpu", "csfddl"),
    ("fddl_gpu", "fddl"),
    ("online_dl", "online_dl"),
    ("lcksvd", "lcksvd"),
    ("aksvd", "aksvd"),
]

EXCLUDED_ENCODERS = {"wl_edge"}
EXCLUDED_LEARNERS = {"csfddl"}

ENCODERS = ["wl", "fsm", "gspan_cork"]
LEARNERS = ["fddl", "aksvd", "lcksvd", "online_dl", "frozen_ksvd", "bayesian"]

CLASSIFIERS = ["LogisticRegression", "GradientBoosting", "LinearSVM", "RandomForest"]

# Pretty name in artifact .txt -> canonical classifier key.
ARTIFACT_CLF_NAMES = {
    "Logistic Regression": "LogisticRegression",
    "Gradient Boosting": "GradientBoosting",
    "Linear SVM": "LinearSVM",
    "Random Forest": "RandomForest",
}

# Canonical output column -> (MC-CV column suffix, artifact (row_label, column)).
# Artifact column is "minority" or "macro"; ROC-AUC/MCC come from the symmetric block.
METRICS = [
    ("macro_precision", "Macro-Precision", ("Precision", "macro")),
    ("macro_recall", "Macro-Recall", ("Recall", "macro")),
    ("macro_f1", "Macro-F1", ("F1", "macro")),
    ("macro_pr_auc", "Macro-PR-AUC", ("PR-AUC (AP)", "macro")),
    ("minority_precision", "Minority-Precision", ("Precision", "minority")),
    ("minority_recall", "Minority-Recall", ("Recall", "minority")),
    ("minority_f1", "Minority-F1", ("F1", "minority")),
    ("minority_pr_auc", "Minority-PR-AUC", ("PR-AUC (AP)", "minority")),
    ("accuracy", "Accuracy", ("Accuracy", "macro")),
    ("roc_auc", "ROC-AUC", ("__symmetric__", "ROC-AUC")),
    ("mcc", "MCC", ("__symmetric__", "MCC")),
]

# Spread is kept for the metrics actually plotted; empty on artifact rows.
STD_METRICS = ["macro_f1", "minority_f1", "minority_pr_auc", "mcc", "roc_auc"]

DIR_RE = re.compile(
    r"^(?:mc_cv_)?(?P<impl>.+?)_(?P<dataset>nci_full|mutag|ptc_mr|ogbg_molhiv)"
    r"(?:_atoms(?P<atoms>\d+))?"
    r"_(?P<start>\d{8}_\d{6})(?:_(?P<end>\d{8}_\d{6}))?$"
)


def parse_dirname(name: str):
    """Split a run directory name into its factors, or return None."""
    m = DIR_RE.match(name)
    if not m:
        return None
    impl = m.group("impl")

    encoder = None
    for tok in ENCODER_TOKENS:
        if impl == tok or impl.startswith(tok + "_"):
            encoder, rest = tok, impl[len(tok) :].lstrip("_")
            break
    if encoder is None:
        return None

    learner = learner_impl = None
    for tok, canon in DICT_LEARNER_TOKENS:
        if rest == tok:
            learner, learner_impl = canon, tok
            break
    if learner is None:
        return None

    return {
        "encoder": encoder,
        "dict_learner": learner,
        "dict_learner_impl": learner_impl,
        "dataset": m.group("dataset"),
        "atoms": int(m.group("atoms")) if m.group("atoms") else None,
        "run_started": m.group("start"),
        "implementation": impl,
    }


def in_scope(info) -> bool:
    return (
        info["dataset"] == DATASET
        and info["atoms"] is not None
        and info["encoder"] not in EXCLUDED_ENCODERS
        and info["dict_learner"] not in EXCLUDED_LEARNERS
    )


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# --------------------------------------------------------------------------
# MC-CV
# --------------------------------------------------------------------------


def read_mccv(run_dir: Path):
    """Return {classifier: {metric: (mean, std, n)}} or None if unusable."""
    per_run = run_dir / "per_run_metrics.csv"
    if not per_run.exists():
        return None, "no per_run_metrics.csv"

    with per_run.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, "per_run_metrics.csv empty"

    header = set(rows[0])
    # Reject the pre-drift schema (Acc / F1 / pr_auc ...) rather than guess a mapping.
    expected = {f"{c}/{m}" for c in CLASSIFIERS for _, m, _ in METRICS}
    if not expected & header:
        return None, "legacy metric schema"

    out = {}
    for clf in CLASSIFIERS:
        per_metric = {}
        for col, mccv_name, _ in METRICS:
            key = f"{clf}/{mccv_name}"
            if key not in header:
                continue
            vals = [v for v in (_num(r.get(key)) for r in rows) if v is not None]
            if not vals:
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else None
            per_metric[col] = (mean, std, len(vals))
        if per_metric:
            out[clf] = per_metric
    return (out or None), (None if out else "no usable classifier columns")


def mccv_seed_count(run_dir: Path) -> int:
    per_run = run_dir / "per_run_metrics.csv"
    if not per_run.exists():
        return 0
    with per_run.open(newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def mccv_fit_seconds(run_dir: Path):
    path = run_dir / "summary_timings.csv"
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") == "fit_encoder_dict":
                return _num(row.get("mean_sec"))
    return None


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

CLF_HEADER_RE = re.compile(r"^=====\s*(?P<name>.+?)\s*\(threshold=(?P<thr>[\d.]+)\)\s*=====")
SYMMETRIC_RE = re.compile(r"^(ROC-AUC|MCC)\s+(-?[\d.]+)")


def read_artifact(run_dir: Path):
    """Parse eval/results_*.txt -> {classifier: {metric: value, 'threshold': t}}."""
    matches = sorted((run_dir / "eval").glob("results_*.txt"))
    if not matches:
        return None, "no eval/results_*.txt"

    lines = matches[0].read_text(encoding="utf-8", errors="replace").splitlines()

    out, clf, thr = {}, None, None
    in_main_table = False  # the Minority/Majority/Macro/Baseline block
    in_symmetric = False

    for line in lines:
        header = CLF_HEADER_RE.match(line)
        if header:
            clf = ARTIFACT_CLF_NAMES.get(header.group("name"))
            thr = _num(header.group("thr"))
            if clf:
                out[clf] = {"threshold": thr}
            in_main_table = in_symmetric = False
            continue

        if clf is None or clf not in out:
            continue

        if line.startswith("Minority(+1)") or "Minority(+1)" in line and "Macro" in line:
            in_main_table, in_symmetric = True, False
            continue
        if line.startswith("Symmetric"):
            in_main_table, in_symmetric = False, True
            continue
        # Everything after this heading repeats metric labels with a different
        # meaning; stop consuming rows for this classifier.
        if line.startswith("Chance baseline") or line.startswith("Confusion Matrix"):
            in_main_table = in_symmetric = False
            continue

        if in_symmetric:
            sym = SYMMETRIC_RE.match(line.strip())
            if sym:
                col = "roc_auc" if sym.group(1) == "ROC-AUC" else "mcc"
                out[clf][col] = _num(sym.group(2))
            continue

        if in_main_table:
            parts = line.rsplit(None, 4)
            if len(parts) != 5:
                continue
            label, minority, majority, macro, _baseline = parts
            label = label.strip()
            for col, _mccv, (art_label, which) in METRICS:
                if art_label == label:
                    out[clf][col] = _num(minority if which == "minority" else macro)

    out = {k: v for k, v in out.items() if len(v) > 1}
    return (out or None), (None if out else "no classifier blocks parsed")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def pick_best(candidates):
    """Most seeds wins; latest run_started breaks ties."""
    return max(candidates, key=lambda c: (c["n_seeds"], c["run_started"]))


def main() -> None:
    report = []
    mccv_cands, art_cands = {}, {}

    for run_dir in sorted((ROOT / "results").iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("mc_cv_"):
            continue
        info = parse_dirname(run_dir.name)
        if info is None:
            report.append(f"SKIP  unparsed dirname     results/{run_dir.name}")
            continue
        if not in_scope(info):
            continue
        metrics, why = read_mccv(run_dir)
        if metrics is None:
            report.append(f"SKIP  {why:<22} results/{run_dir.name}")
            continue
        key = (info["encoder"], info["dict_learner"], info["atoms"])
        mccv_cands.setdefault(key, []).append(
            {
                **info,
                "metrics": metrics,
                "n_seeds": mccv_seed_count(run_dir),
                "fit_seconds": mccv_fit_seconds(run_dir),
                "run_id": run_dir.name,
            }
        )

    for run_dir in sorted((ROOT / "artifacts").iterdir()):
        if not run_dir.is_dir():
            continue
        info = parse_dirname(run_dir.name)
        if info is None or not in_scope(info):
            continue
        metrics, why = read_artifact(run_dir)
        if metrics is None:
            report.append(f"SKIP  {why:<22} artifacts/{run_dir.name}")
            continue
        key = (info["encoder"], info["dict_learner"], info["atoms"])
        art_cands.setdefault(key, []).append(
            {**info, "metrics": metrics, "n_seeds": 1, "run_id": run_dir.name}
        )

    # An artifact is only a fallback when the *combination* has no MC-CV at all,
    # not merely when this atom size is missing -- otherwise one row of a sweep
    # would silently switch estimand.
    mccv_combos = {(e, d) for (e, d, _a) in mccv_cands}

    fieldnames = (
        [
            "encoder",
            "dict_learner",
            "dict_learner_impl",
            "implementation",
            "dataset",
            "atoms",
            "classifier",
            "source",
            "n_seeds",
        ]
        + [c for c, _, _ in METRICS]
        + [f"{c}_std" for c in STD_METRICS]
        + ["threshold", "fit_seconds", "run_started", "run_id"]
    )

    out_rows = []
    filled, missing = [], []

    for encoder in ENCODERS:
        for learner in LEARNERS:
            keys = sorted(
                k for k in (set(mccv_cands) | set(art_cands)) if k[0] == encoder and k[1] == learner
            )
            use_mccv = (encoder, learner) in mccv_combos
            keys = [k for k in keys if (k in mccv_cands) if use_mccv] or keys

            emitted = False
            for key in keys:
                if use_mccv:
                    if key not in mccv_cands:
                        continue
                    cands, source = mccv_cands[key], "mccv"
                else:
                    if key not in art_cands:
                        continue
                    cands, source = art_cands[key], "artifact"

                chosen = pick_best(cands)
                for other in cands:
                    if other is not chosen:
                        report.append(
                            f"DEDUP dropped {other['run_id']} "
                            f"(n={other['n_seeds']}) in favour of {chosen['run_id']} "
                            f"(n={chosen['n_seeds']})"
                        )

                for clf in CLASSIFIERS:
                    vals = chosen["metrics"].get(clf)
                    if not vals:
                        continue
                    row = {
                        "encoder": encoder,
                        "dict_learner": learner,
                        "dict_learner_impl": chosen["dict_learner_impl"],
                        "implementation": chosen["implementation"],
                        "dataset": chosen["dataset"],
                        "atoms": chosen["atoms"],
                        "classifier": clf,
                        "source": source,
                        "n_seeds": chosen["n_seeds"],
                        "threshold": "",
                        "fit_seconds": chosen.get("fit_seconds") or "",
                        "run_started": chosen["run_started"],
                        "run_id": chosen["run_id"],
                    }
                    for col, _, _ in METRICS:
                        if source == "mccv":
                            entry = vals.get(col)
                            row[col] = round(entry[0], 6) if entry else ""
                        else:
                            v = vals.get(col)
                            row[col] = round(v, 6) if v is not None else ""
                    for col in STD_METRICS:
                        entry = vals.get(col) if source == "mccv" else None
                        row[f"{col}_std"] = (
                            round(entry[1], 6) if entry and entry[1] is not None else ""
                        )
                    if source == "artifact" and vals.get("threshold") is not None:
                        row["threshold"] = round(vals["threshold"], 6)
                    out_rows.append(row)
                    emitted = True

            if emitted:
                filled.append((encoder, learner))
            else:
                missing.append((encoder, learner))
                out_rows.append(
                    {
                        **{f: "" for f in fieldnames},
                        "encoder": encoder,
                        "dict_learner": learner,
                        "implementation": f"{encoder}_{learner}",
                        "dataset": DATASET,
                        "source": "missing",
                    }
                )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_mccv = sum(1 for r in out_rows if r["source"] == "mccv")
    n_art = sum(1 for r in out_rows if r["source"] == "artifact")
    n_miss = sum(1 for r in out_rows if r["source"] == "missing")

    summary = [
        f"rows written        : {len(out_rows)}  -> {OUT_CSV.relative_to(ROOT)}",
        f"  source=mccv       : {n_mccv}",
        f"  source=artifact   : {n_art}",
        f"  source=missing    : {n_miss}",
        f"combinations filled : {len(filled)} / {len(ENCODERS) * len(LEARNERS)}",
        f"combinations empty  : {sorted(missing)}",
        "",
        "notes / skips / dedup decisions:",
    ] + (report or ["  (none)"])

    OUT_REPORT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
