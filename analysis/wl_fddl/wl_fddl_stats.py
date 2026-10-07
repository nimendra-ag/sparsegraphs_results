"""Derived tables for the wl_fddl analysis (every number quoted in the report).

All inputs are recorded per-combination means/stds from wl_fddl_all_results.csv.
Significance tests are Welch t-tests computed from those summary statistics
(mean, std, n_seeds), since each combination was run on its own random seeds
and nothing can be paired across atom sizes.

Run:  python analysis/wl_fddl/wl_fddl_stats.py
Out:  analysis/wl_fddl/tables/*.csv
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from _common import (CLASSIFIERS, COMMON_ATOMS, DATASETS, LINE_CLASSIFIERS,
                     METRICS, ROOT, TABLE_DIR, load)

HEADLINE = ["Macro-F1", "MCC", "ROC-AUC", "Minority-PR-AUC"]
# Nemenyi critical value q_0.05 for k=6 classifiers (Demsar 2006, Table 5a).
NEMENYI_Q05_K6 = 2.850


def welch(a: pd.Series, b: pd.Series, metric: str) -> float:
    return stats.ttest_ind_from_stats(
        a[f"{metric}_mean"], a[f"{metric}_std"], a["n_seeds"],
        b[f"{metric}_mean"], b[f"{metric}_std"], b["n_seeds"],
        equal_var=False).pvalue


def cell(df, ds, atoms, clf) -> pd.Series:
    return df[(df.dataset_label == ds) & (df.total_atoms == atoms)
              & (df.classifier == clf)].iloc[0]


def best_atoms(df) -> pd.DataFrame:
    """Best atom size per (dataset, classifier, metric), tested against 128."""
    out = []
    for metric in HEADLINE:
        for ds in DATASETS:
            for clf in CLASSIFIERS:
                sub = df[(df.dataset_label == ds) & (df.classifier == clf)]
                best = sub.loc[sub[f"{metric}_mean"].idxmax()]
                base = cell(df, ds, 128, clf)
                worst = sub.loc[sub[f"{metric}_mean"].idxmin()]
                out.append({
                    "metric": metric, "dataset": ds, "classifier": clf,
                    "best_atoms": best.total_atoms,
                    "best_mean": best[f"{metric}_mean"],
                    "best_std": best[f"{metric}_std"],
                    "at_128_mean": base[f"{metric}_mean"],
                    "delta_best_vs_128": round(best[f"{metric}_mean"] - base[f"{metric}_mean"], 4),
                    "p_best_vs_128": (round(welch(best, base, metric), 4)
                                      if best.total_atoms != 128 else np.nan),
                    "worst_atoms": worst.total_atoms,
                    "range_across_atoms": round(best[f"{metric}_mean"] - worst[f"{metric}_mean"], 4),
                })
    return pd.DataFrame(out)


def delta_128_to_2048(df) -> pd.DataFrame:
    """128 -> 2048 change: the widest span recorded on every dataset."""
    out = []
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            a, b = cell(df, ds, 128, clf), cell(df, ds, 2048, clf)
            row = {"dataset": ds, "classifier": clf}
            for metric in HEADLINE + ["Accuracy", "Minority-Recall", "Minority-Precision"]:
                row[f"{metric}_128"] = a[f"{metric}_mean"]
                row[f"{metric}_2048"] = b[f"{metric}_mean"]
                row[f"{metric}_delta"] = round(b[f"{metric}_mean"] - a[f"{metric}_mean"], 4)
                row[f"{metric}_p"] = round(welch(b, a, metric), 4)
            out.append(row)
    return pd.DataFrame(out)


def adjacent_steps(df) -> pd.DataFrame:
    """Change between consecutive atom sizes (Welch p, unadjusted)."""
    out = []
    for ds in DATASETS:
        atoms = sorted(df[df.dataset_label == ds].total_atoms.unique())
        for clf in CLASSIFIERS:
            for lo, hi in zip(atoms, atoms[1:]):
                a, b = cell(df, ds, lo, clf), cell(df, ds, hi, clf)
                row = {"dataset": ds, "classifier": clf, "from_atoms": lo, "to_atoms": hi}
                for metric in ["Macro-F1", "MCC"]:
                    row[f"{metric}_delta"] = round(b[f"{metric}_mean"] - a[f"{metric}_mean"], 4)
                    row[f"{metric}_p"] = round(welch(b, a, metric), 4)
                out.append(row)
    return pd.DataFrame(out)


def classifier_ranks(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean rank of each classifier over the 22 (dataset, atoms) combinations."""
    ranks, tests = [], []
    for metric in HEADLINE + ["Macro-PR-AUC", "Accuracy"]:
        piv = df.pivot_table(index=["dataset_label", "total_atoms"], columns="classifier",
                             values=f"{metric}_mean", observed=True)[CLASSIFIERS]
        r = piv.rank(axis=1, ascending=False)
        wins = (piv.eq(piv.max(axis=1), axis=0)).sum()
        n, k = piv.shape
        chi2, p = stats.friedmanchisquare(*[piv[c] for c in CLASSIFIERS])
        cd = NEMENYI_Q05_K6 * math.sqrt(k * (k + 1) / (6 * n))
        tests.append({"metric": metric, "n_blocks": n, "k": k,
                      "friedman_chi2": round(chi2, 3), "friedman_p": p,
                      "nemenyi_cd_0.05": round(cd, 3)})
        for clf in CLASSIFIERS:
            ranks.append({"metric": metric, "classifier": clf,
                          "mean_rank": round(r[clf].mean(), 3),
                          "n_first": int(wins[clf]), "n_blocks": n,
                          "mean_value": round(piv[clf].mean(), 4),
                          "min_value": piv[clf].min(), "max_value": piv[clf].max()})
    return pd.DataFrame(ranks), pd.DataFrame(tests)


def rf_pairwise(df) -> pd.DataFrame:
    """RF vs each other classifier, paired over the 22 combinations.

    Nemenyi is conservative; a paired test shows whether RF's per-combination
    lead is consistent. The 22 blocks share datasets, so they are not fully
    independent - read the p-values as descriptive."""
    out = []
    for metric in HEADLINE:
        piv = df.pivot_table(index=["dataset_label", "total_atoms"], columns="classifier",
                             values=f"{metric}_mean", observed=True)
        for clf in CLASSIFIERS[1:]:
            d = piv["RandomForest"] - piv[clf]
            wins = int((d > 0).sum())
            out.append({"metric": metric, "versus": clf, "n_blocks": len(d),
                        "rf_wins": wins, "mean_rf_lead": round(d.mean(), 4),
                        "min_rf_lead": round(d.min(), 4), "max_rf_lead": round(d.max(), 4),
                        "wilcoxon_p": stats.wilcoxon(d).pvalue,
                        "sign_test_p": stats.binomtest(wins, len(d)).pvalue})
    return pd.DataFrame(out)


def rf_vs_linear(df) -> pd.DataFrame:
    out = []
    for (ds, atoms), g in df.groupby(["dataset_label", "total_atoms"], observed=True):
        g = g.set_index("classifier")
        row = {"dataset": ds, "atoms": atoms}
        for metric in ["Macro-F1", "MCC"]:
            lin = g.loc[["LogisticRegression", "LinearSVM"], f"{metric}_mean"]
            row[f"{metric}_rf"] = g.loc["RandomForest", f"{metric}_mean"]
            row[f"{metric}_best_linear"] = lin.max()
            row[f"{metric}_best_linear_clf"] = lin.idxmax()
            row[f"{metric}_gap"] = round(row[f"{metric}_rf"] - lin.max(), 4)
        row["gb_minus_rf_macro_f1"] = round(g.loc["GradientBoosting", "Macro-F1_mean"]
                                            - g.loc["RandomForest", "Macro-F1_mean"], 4)
        row["gb_fit_sec"] = g.loc["GradientBoosting", "clf_val_sec_mean"]
        row["rf_fit_sec"] = g.loc["RandomForest", "clf_val_sec_mean"]
        out.append(row)
    return pd.DataFrame(out)


def src_pure_vs_fddl(df) -> pd.DataFrame:
    p = df[df.classifier == "SRC_pure"].set_index(["dataset_label", "total_atoms"])
    f = df[df.classifier == "SRC_fddl"].set_index(["dataset_label", "total_atoms"])
    out = []
    for metric in METRICS:
        d = f[f"{metric}_mean"] - p[f"{metric}_mean"]
        out.append({"metric": metric, "n_combinations": len(d),
                    "max_abs_diff": round(d.abs().max(), 4),
                    "mean_abs_diff": round(d.abs().mean(), 5),
                    "n_fddl_higher": int((d > 0).sum()), "n_equal": int((d == 0).sum()),
                    "n_pure_higher": int((d < 0).sum()),
                    "mean_seed_std_pure": round(p[f"{metric}_std"].mean(), 4)})
    return pd.DataFrame(out)


def runtime(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (ds, atoms), g in df.groupby(["dataset_label", "total_atoms"], observed=True):
        g = g.set_index("classifier")
        b = g.loc["RandomForest"]
        classical = sum(g.loc[c, "clf_val_sec_mean"] + g.loc[c, "clf_test_sec_mean"]
                        for c in ["LogisticRegression", "LinearSVM", "RandomForest", "GradientBoosting"])
        src = (b.src_embeddings_sec_mean + g.loc["SRC_pure", "clf_test_sec_mean"]
               + g.loc["SRC_fddl", "clf_test_sec_mean"])
        sparse = b.sparse_codes_train_sec_mean + b.sparse_codes_val_sec_mean + b.sparse_codes_test_sec_mean
        rows.append({"dataset": ds, "atoms": atoms, "data_load": b.data_load_sec_mean,
                     "fit_encoder_dict": b.fit_encoder_dict_sec_mean,
                     "sparse_coding": round(sparse, 2), "classical_classifiers": round(classical, 2),
                     "gboost_only": round(g.loc["GradientBoosting", "clf_val_sec_mean"]
                                          + g.loc["GradientBoosting", "clf_test_sec_mean"], 2),
                     "src_classifiers": round(src, 2), "seed_total": b.seed_total_sec_mean})
    rt = pd.DataFrame(rows)
    rt["seed_total_x_vs_128"] = rt.groupby("dataset").seed_total.transform(lambda s: round(s / s.iloc[0], 2))
    rt["fit_x_vs_128"] = rt.groupby("dataset").fit_encoder_dict.transform(lambda s: round(s / s.iloc[0], 2))
    rt["fit_share"] = round(rt.fit_encoder_dict / rt.seed_total, 3)
    rt["sparse_share"] = round(rt.sparse_coding / rt.seed_total, 3)
    rt["classical_share"] = round(rt.classical_classifiers / rt.seed_total, 3)

    slopes = []
    for ds, g in rt.groupby("dataset"):
        for lo in (128, 512):
            s = g[g.atoms >= lo]
            for col in ["seed_total", "fit_encoder_dict"]:
                k = np.polyfit(np.log2(s.atoms), np.log2(s[col]), 1)[0]
                slopes.append({"dataset": ds, "quantity": col, "atoms_from": lo,
                               "atoms_to": s.atoms.max(), "loglog_slope": round(k, 3)})
    return rt, pd.DataFrame(slopes)


def cost_performance(df) -> pd.DataFrame:
    out = []
    for ds in DATASETS:
        for clf in ["RandomForest", "LogisticRegression", "LinearSVM"]:
            g = df[(df.dataset_label == ds) & (df.classifier == clf)].sort_values("total_atoms")
            for (_, a), (_, b) in zip(g.iterrows(), g.iloc[1:].iterrows()):
                dt = b.seed_total_sec_mean - a.seed_total_sec_mean
                dm = b["Macro-F1_mean"] - a["Macro-F1_mean"]
                out.append({"dataset": ds, "classifier": clf,
                            "from_atoms": a.total_atoms, "to_atoms": b.total_atoms,
                            "extra_sec_per_seed": round(dt, 2),
                            "macro_f1_delta": round(dm, 4),
                            "macro_f1_per_100s": round(100 * dm / dt, 4)})
    return pd.DataFrame(out)


def metric_ranges(df) -> pd.DataFrame:
    """How far each metric moves across atom sizes (max - min), per dataset."""
    out = []
    for clf in CLASSIFIERS:
        for metric in ["Accuracy", "ROC-AUC", "Macro-PR-AUC", "Macro-F1", "MCC",
                       "Minority-F1", "Minority-PR-AUC", "Minority-Recall", "Minority-Precision"]:
            r = df[df.classifier == clf].groupby("dataset_label", observed=True)[f"{metric}_mean"]
            span = r.max() - r.min()
            out.append({"classifier": clf, "metric": metric,
                        "mean_range_over_datasets": round(span.mean(), 4),
                        "min_value": df.loc[df.classifier == clf, f"{metric}_mean"].min(),
                        "max_value": df.loc[df.classifier == clf, f"{metric}_mean"].max(),
                        **{f"range_{ds}": round(span[ds], 4) for ds in DATASETS}})
    return pd.DataFrame(out)


def datasets(df) -> pd.DataFrame:
    common = df[df.total_atoms.isin(COMMON_ATOMS)]
    out = []
    for ds in DATASETS:
        g = common[common.dataset_label == ds]
        row = {"dataset": ds, "split_vocab_train": g.split_vocab_train.iloc[0],
               "split_ml_train": g.split_ml_train.iloc[0], "split_test": g.split_test.iloc[0]}
        for clf in LINE_CLASSIFIERS:
            h = g[g.classifier == clf]
            row[f"{clf}_MCC_mean_128_2048"] = round(h["MCC_mean"].mean(), 4)
            row[f"{clf}_MacroF1_mean_128_2048"] = round(h["Macro-F1_mean"].mean(), 4)
        out.append(row)
    d = pd.DataFrame(out)
    for clf in LINE_CLASSIFIERS:
        d[f"{clf}_MCC_rank"] = d[f"{clf}_MCC_mean_128_2048"].rank(ascending=False).astype(int)
    return d


def variability(df) -> pd.DataFrame:
    out = []
    for clf in CLASSIFIERS:
        for ds in DATASETS:
            g = df[(df.classifier == clf) & (df.dataset_label == ds)]
            rho, p = stats.spearmanr(g.total_atoms, g.MCC_std)
            out.append({"classifier": clf, "dataset": ds,
                        "mcc_std_min": g.MCC_std.min(), "mcc_std_max": g.MCC_std.max(),
                        "mcc_std_median": round(g.MCC_std.median(), 4),
                        "macro_f1_std_median": round(g["Macro-F1_std"].median(), 4),
                        "spearman_atoms_vs_mcc_std": round(rho, 3), "spearman_p": round(p, 4)})
    return pd.DataFrame(out)


def features(df) -> pd.DataFrame:
    g = df[df.classifier == "RandomForest"]
    return g[["dataset_label", "total_atoms", "n_scored_features_mean",
              "n_selected_features_mean", "n_selected_features_std",
              "kept_fraction_mean"]].rename(columns={"dataset_label": "dataset"})


def replicate() -> pd.DataFrame:
    """The two independent NCI1 / 2048-atom runs, read from their own summaries."""
    runs = sorted((ROOT / "results").glob("mc_cv_wl_fddl_gpu_nci_full_id1_atoms2048_*"))
    s = [pd.read_csv(r / "summary_mean_std.csv").set_index("metric") for r in runs]
    seeds = [sorted(json.loads((r / "manifest.json").read_text())["completed_seeds"]) for r in runs]
    out = []
    for clf in CLASSIFIERS:
        for metric in HEADLINE:
            k = f"{clf}/{metric}"
            out.append({"classifier": clf, "metric": metric,
                        "run1": s[0].loc[k, "mean"], "run1_std": s[0].loc[k, "std_ddof1"],
                        "run2": s[1].loc[k, "mean"], "run2_std": s[1].loc[k, "std_ddof1"],
                        "abs_diff": round(abs(s[1].loc[k, "mean"] - s[0].loc[k, "mean"]), 4),
                        "p_welch": round(stats.ttest_ind_from_stats(
                            s[0].loc[k, "mean"], s[0].loc[k, "std_ddof1"], 5,
                            s[1].loc[k, "mean"], s[1].loc[k, "std_ddof1"], 5,
                            equal_var=False).pvalue, 4),
                        "run1_dir": runs[0].name, "run2_dir": runs[1].name,
                        "run1_seeds": " ".join(map(str, seeds[0])),
                        "run2_seeds": " ".join(map(str, seeds[1]))})
    return pd.DataFrame(out)


def main() -> None:
    df = load()
    TABLE_DIR.mkdir(exist_ok=True)
    ranks, friedman = classifier_ranks(df)
    rt, slopes = runtime(df)
    tables = {
        "best_atoms": best_atoms(df),
        "delta_128_to_2048": delta_128_to_2048(df),
        "adjacent_steps": adjacent_steps(df),
        "classifier_ranks": ranks,
        "friedman_tests": friedman,
        "rf_pairwise": rf_pairwise(df),
        "rf_vs_linear": rf_vs_linear(df),
        "src_pure_vs_fddl": src_pure_vs_fddl(df),
        "runtime": rt,
        "runtime_scaling": slopes,
        "cost_performance": cost_performance(df),
        "metric_ranges": metric_ranges(df),
        "datasets": datasets(df),
        "variability": variability(df),
        "features": features(df),
        "replicate_nci1_2048": replicate(),
    }
    for name, t in tables.items():
        t.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        print(f"wrote tables/{name}.csv  ({len(t)} rows)")


if __name__ == "__main__":
    main()
