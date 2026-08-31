"""Aggregate every encoder x dict-learner result, plus the baselines, into one CSV.

Source preference per method (an encoder+learner pipeline, or a baseline):
  1. MC-CV over seeds          results/mc_cv_*      source=mccv
  2. repeated k-fold           results/kfold_*      source=kfold
  3. single-split run          artifacts/*  or  results/graph2vec_*   source=artifact
  4. nothing                   -> one empty placeholder row, source=missing

The preference is applied **per method, not per cell**: if a method has any
repeated-evaluation run at all, its single-split runs are dropped entirely.
Mixing the two inside one method would put a 1-draw estimate and a 5-draw mean
in the same column with nothing to tell them apart.

Baselines (SF, graph2vec, GCN) land in the same schema with family=baseline and
an empty encoder/dict_learner; `method` is populated for every row.

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
# Which NCI screen to report on; None keeps every screen (each one then gets its
# own rows, since runs on different screens are not comparable). Runs predating
# the id-in-the-folder-name convention parse as dataset_id=None.
DATASET_ID = None

# Longest-first: gspan_cork before wl, wl_edge before wl.
ENCODER_TOKENS = ["gspan_cork", "wl_edge", "fsm", "wl"]

# Raw implementation token -> canonical learner. Longest-first. The _gpu/_cpu
# split is an implementation detail of the same learner, so it collapses here
# and survives in dict_learner_impl.
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

BASELINE_TOKENS = ["graph2vec", "gcn", "sf"]

ENCODERS = ["wl", "wl_edge", "fsm", "gspan_cork"]
LEARNERS = ["fddl", "csfddl", "aksvd", "lcksvd", "online_dl", "frozen_ksvd", "bayesian"]
BASELINES = ["sf", "graph2vec", "gcn"]

# Training-set columns and the mostly-empty SRC arms never enter the table.
EXCLUDED_CLASSIFIERS = {"SRC_pure", "SRC_fddl"}

CLASSIFIER_ORDER = ["LogisticRegression", "GradientBoosting", "LinearSVM", "RandomForest", "GCN"]

ARTIFACT_CLF_NAMES = {
    "Logistic Regression": "LogisticRegression",
    "Gradient Boosting": "GradientBoosting",
    "Linear SVM": "LinearSVM",
    "Random Forest": "RandomForest",
}

# canonical column -> (repeated-run column suffix, artifact (row label, which column))
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

STD_METRICS = ["macro_f1", "minority_f1", "minority_pr_auc", "mcc", "roc_auc"]

SOURCE_RANK = {"mccv": 3, "kfold": 2, "artifact": 1}
REPEATED = {"mccv", "kfold"}

DIR_RE = re.compile(
    r"^(?P<prefix>mc_cv_|kfold_)?(?P<impl>.+?)_(?P<dataset>nci_full|mutag|ptc_mr|ogbg_molhiv)"
    # NCI screens are numbered; runs before that id was recorded have no suffix.
    r"(?:_id(?P<dataset_id>\d+))?"
    r"(?:_(?:atoms|dim)(?P<atoms>\d+))?"
    r"_(?P<start>\d{8}_\d{6})(?:_(?P<end>\d{8}_\d{6}))?$"
)


def parse_dirname(name: str):
    m = DIR_RE.match(name)
    if not m:
        return None
    impl = m.group("impl")
    prefix = m.group("prefix") or ""

    base = {
        "dataset": m.group("dataset"),
        "dataset_id": int(m.group("dataset_id")) if m.group("dataset_id") else None,
        "atoms": int(m.group("atoms")) if m.group("atoms") else None,
        "run_started": m.group("start"),
        "implementation": impl,
        "source": {"mc_cv_": "mccv", "kfold_": "kfold"}.get(prefix, "artifact"),
    }

    if impl in BASELINE_TOKENS:
        return {**base, "family": "baseline", "method": impl,
                "encoder": "", "dict_learner": "", "dict_learner_impl": ""}

    for enc in ENCODER_TOKENS:
        if impl.startswith(enc + "_"):
            rest = impl[len(enc) + 1:]
            for tok, canon in DICT_LEARNER_TOKENS:
                if rest == tok:
                    return {**base, "family": "pipeline", "method": f"{enc}+{canon}",
                            "encoder": enc, "dict_learner": canon, "dict_learner_impl": tok}
            return None
    return None


def in_scope(info) -> bool:
    if DATASET_ID is not None and info["dataset_id"] != DATASET_ID:
        return False
    return info["dataset"] == DATASET and info["atoms"] is not None


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def classify_status(vals: dict) -> str:
    """Flag runs that never predict the positive class or rank at chance."""
    rec, auc = vals.get("minority_recall"), vals.get("roc_auc")
    if (rec is not None and rec == 0.0) or (auc is not None and auc <= 0.5):
        return "degenerate"
    return "ok"


# --------------------------------------------------------------------------
# repeated runs (MC-CV and k-fold share the harness output format)
# --------------------------------------------------------------------------


def read_repeated(run_dir: Path):
    per_run = run_dir / "per_run_metrics.csv"
    if not per_run.exists():
        return None, "no per_run_metrics.csv"

    with per_run.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, "per_run_metrics.csv empty"

    header = set(rows[0])
    # Discover the arms present rather than assuming the four downstream
    # classifiers -- GCN reports a single end-to-end arm instead.
    arms = sorted({c.split("/")[0] for c in header if "/" in c})
    arms = [a for a in arms if a not in EXCLUDED_CLASSIFIERS and not a.endswith("_train")]
    if not arms:
        return None, "no usable classifier columns"

    expected = {f"{a}/{m}" for a in arms for _, m, _ in METRICS}
    if not expected & header:
        return None, "legacy metric schema"

    out = {}
    for arm in arms:
        per_metric = {}
        for col, name, _ in METRICS:
            key = f"{arm}/{name}"
            if key not in header:
                continue
            vals = [v for v in (_num(r.get(key)) for r in rows) if v is not None]
            if not vals:
                continue
            per_metric[col] = (
                statistics.fmean(vals),
                statistics.stdev(vals) if len(vals) > 1 else None,
            )
        if per_metric:
            out[arm] = per_metric
    return (out or None), (None if out else "no usable classifier columns")


def repeated_n(run_dir: Path) -> int:
    per_run = run_dir / "per_run_metrics.csv"
    if not per_run.exists():
        return 0
    with per_run.open(newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def repeated_fit_seconds(run_dir: Path):
    path = run_dir / "summary_timings.csv"
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") == "fit_encoder_dict":
                return _num(row.get("mean_sec"))
    return None


# --------------------------------------------------------------------------
# single-split runs (a results_*.txt report)
# --------------------------------------------------------------------------

CLF_HEADER_RE = re.compile(r"^=====\s*(?P<name>.+?)\s*\(threshold=(?P<thr>[\d.]+)\)\s*=====")
SYMMETRIC_RE = re.compile(r"^(ROC-AUC|MCC)\s+(-?[\d.]+)")


def find_report(run_dir: Path):
    for pattern in ("eval/results_*.txt", "results_*.txt"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def read_single_split(run_dir: Path):
    report = find_report(run_dir)
    if report is None:
        return None, "no results_*.txt"

    lines = report.read_text(encoding="utf-8", errors="replace").splitlines()
    out, clf = {}, None
    in_main = in_sym = False

    for line in lines:
        header = CLF_HEADER_RE.match(line)
        if header:
            clf = ARTIFACT_CLF_NAMES.get(header.group("name"))
            if clf:
                out[clf] = {"threshold": _num(header.group("thr"))}
            in_main = in_sym = False
            continue
        if clf is None or clf not in out:
            continue

        if "Minority(+1)" in line and "Macro" in line:
            in_main, in_sym = True, False
            continue
        if line.startswith("Symmetric"):
            in_main, in_sym = False, True
            continue
        # Past these headings the same row labels reappear with a different
        # meaning (chance floors), so stop consuming for this classifier.
        if line.startswith("Chance baseline") or line.startswith("Confusion Matrix"):
            in_main = in_sym = False
            continue

        if in_sym:
            sym = SYMMETRIC_RE.match(line.strip())
            if sym:
                out[clf]["roc_auc" if sym.group(1) == "ROC-AUC" else "mcc"] = _num(sym.group(2))
            continue

        if in_main:
            parts = line.rsplit(None, 4)
            if len(parts) != 5:
                continue
            label, minority, _majority, macro, _baseline = parts
            for col, _n, (art_label, which) in METRICS:
                if art_label == label.strip():
                    out[clf][col] = _num(minority if which == "minority" else macro)

    out = {k: v for k, v in out.items() if len(v) > 1}
    return (out or None), (None if out else "no classifier blocks parsed")


# --------------------------------------------------------------------------


def pick_best(candidates):
    """Repeated beats single-split; then more seeds; then the later run."""
    return max(candidates,
               key=lambda c: (SOURCE_RANK[c["source"]], c["n_seeds"], c["run_started"]))


def main() -> None:
    report, cands = [], {}

    def collect(run_dir: Path, reader, label):
        info = parse_dirname(run_dir.name)
        if info is None:
            report.append(f"SKIP  unparsed dirname     {label}/{run_dir.name}")
            return
        if not in_scope(info):
            return
        metrics, why = reader(run_dir)
        if metrics is None:
            report.append(f"SKIP  {why:<24} {label}/{run_dir.name}")
            return
        n = repeated_n(run_dir) if info["source"] in REPEATED else 1
        cands.setdefault(
            (info["method"], info["atoms"], info["dataset_id"]), []
        ).append({
            **info, "metrics": metrics, "n_seeds": n,
            "fit_seconds": repeated_fit_seconds(run_dir) if info["source"] in REPEATED else None,
            "run_id": run_dir.name,
        })

    for run_dir in sorted((ROOT / "results").iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if name.startswith(("mc_cv_", "kfold_")):
            collect(run_dir, read_repeated, "results")
        else:
            # graph2vec writes a single-split report straight into results/
            collect(run_dir, read_single_split, "results")

    for run_dir in sorted((ROOT / "artifacts").iterdir()):
        if run_dir.is_dir():
            collect(run_dir, read_single_split, "artifacts")

    # Per-method source preference, applied before any per-cell choice.
    best_source = {}
    for (method, _atoms, _dataset_id), lst in cands.items():
        for c in lst:
            best_source[method] = max(best_source.get(method, 0), SOURCE_RANK[c["source"]])

    fieldnames = (
        ["family", "method", "encoder", "dict_learner", "dict_learner_impl",
         "implementation", "dataset", "dataset_id", "atoms", "classifier",
         "source", "n_seeds", "status"]
        + [c for c, _, _ in METRICS]
        + [f"{c}_std" for c in STD_METRICS]
        + ["threshold", "fit_seconds", "run_started", "run_id"]
    )

    methods = ([("pipeline", f"{e}+{d}") for e in ENCODERS for d in LEARNERS]
               + [("baseline", b) for b in BASELINES])

    out_rows, filled, missing = [], [], []

    for family, method in methods:
        keys = sorted((k for k in cands if k[0] == method),
                      key=lambda k: (k[1], k[2] if k[2] is not None else -1))
        rank = best_source.get(method, 0)
        emitted = False

        for key in keys:
            pool = [c for c in cands[key] if SOURCE_RANK[c["source"]] == rank]
            if not pool:
                for c in cands[key]:
                    report.append(
                        f"DROP  {c['run_id']} (source={c['source']}) -- "
                        f"{method} has a higher-ranked source available")
                continue
            chosen = pick_best(pool)
            for other in cands[key]:
                if other is not chosen:
                    report.append(
                        f"DEDUP dropped {other['run_id']} (source={other['source']}, "
                        f"n={other['n_seeds']}) for {chosen['run_id']} "
                        f"(source={chosen['source']}, n={chosen['n_seeds']})")

            arms = [a for a in CLASSIFIER_ORDER if a in chosen["metrics"]]
            arms += [a for a in sorted(chosen["metrics"]) if a not in CLASSIFIER_ORDER]
            for arm in arms:
                vals = chosen["metrics"][arm]
                flat = {c: (vals[c][0] if chosen["source"] in REPEATED else vals.get(c))
                        for c, _, _ in METRICS if c in vals}
                row = {
                    "family": family, "method": method,
                    "encoder": chosen["encoder"], "dict_learner": chosen["dict_learner"],
                    "dict_learner_impl": chosen["dict_learner_impl"],
                    "implementation": chosen["implementation"], "dataset": chosen["dataset"],
                    "dataset_id": chosen.get("dataset_id") if chosen.get("dataset_id") is not None else "",
                    "atoms": chosen["atoms"], "classifier": arm,
                    "source": chosen["source"], "n_seeds": chosen["n_seeds"],
                    "status": classify_status(flat),
                    "threshold": "", "fit_seconds": chosen.get("fit_seconds") or "",
                    "run_started": chosen["run_started"], "run_id": chosen["run_id"],
                }
                for col, _, _ in METRICS:
                    v = flat.get(col)
                    row[col] = round(v, 6) if v is not None else ""
                for col in STD_METRICS:
                    sd = vals.get(col)[1] if (chosen["source"] in REPEATED and col in vals) else None
                    row[f"{col}_std"] = round(sd, 6) if sd is not None else ""
                if chosen["source"] == "artifact" and vals.get("threshold") is not None:
                    row["threshold"] = round(vals["threshold"], 6)
                out_rows.append(row)
                emitted = True

        (filled if emitted else missing).append(method)
        if not emitted:
            out_rows.append({**{f: "" for f in fieldnames}, "family": family,
                             "method": method, "dataset": DATASET, "source": "missing",
                             "encoder": method.split("+")[0] if "+" in method else "",
                             "dict_learner": method.split("+")[1] if "+" in method else "",
                             "implementation": method.replace("+", "_")})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {s: sum(1 for r in out_rows if r["source"] == s)
              for s in ("mccv", "kfold", "artifact", "missing")}
    degen = [r for r in out_rows if r["status"] == "degenerate"]

    summary = [
        f"rows written        : {len(out_rows)}  -> {OUT_CSV.relative_to(ROOT)}",
        *[f"  source={s:<10}: {n}" for s, n in counts.items()],
        f"methods filled      : {len(filled)} / {len(methods)}",
        f"methods empty       : {missing}",
        f"degenerate rows     : {len(degen)}"
        + (f"  ({sorted({r['method'] for r in degen})})" if degen else ""),
        "",
        "notes / skips / dedup decisions:",
    ] + (report or ["  (none)"])

    OUT_REPORT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[:8]))


if __name__ == "__main__":
    main()
