"""Figures for the wl_fddl analysis (see WL_FDDL_ANALYSIS.md for the reading).

Reads wl_fddl_all_results.csv and the tables written by wl_fddl_stats.py, so
run that first.

Run:  python analysis/wl_fddl/wl_fddl_stats.py
      python analysis/wl_fddl/make_wl_fddl_figures.py
Out:  analysis/wl_fddl/figures/*.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from _common import (CLASSIFIERS, CLF_COLOR, CLF_MARKER, DATASETS, DS_COLOR,
                     DS_MARKER, FIG_DIR, GRID, INK, INK_2, LINE_CLASSIFIERS,
                     MUTED, SEQ_BLUE, SHORT, SLOTS, SURFACE, TABLE_DIR,
                     apply_style, atoms_axis, load)

N_NOTE = "Mean of 5 seeds per point (NCI1 @ 2048: 10 seeds over 2 runs)"
JITTER = np.linspace(-0.06, 0.06, len(LINE_CLASSIFIERS))  # in log2 units
METRIC_MARKERS = ["o", "s", "^", "D", "v"]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"{name}.csv")


def recorded_atoms(df, ds) -> list[int]:
    return sorted(df[df.dataset_label == ds].total_atoms.unique())


def clf_handles(classifiers) -> list:
    return [Line2D([], [], color=CLF_COLOR[c], marker=CLF_MARKER[c], lw=2,
                   markeredgecolor=SURFACE, markeredgewidth=1.2, label=SHORT[c])
            for c in classifiers]


def ds_handles() -> list:
    return [Line2D([], [], color=DS_COLOR[d], marker=DS_MARKER[d], lw=2,
                   markeredgecolor=SURFACE, markeredgewidth=1.2, label=d)
            for d in DATASETS]


def finish(fig, name: str, title: str, subtitle: str, handles=None) -> None:
    """Reserve a fixed band (in points) above the axes for title, subtitle and
    legend, so they never collide whatever the figure height."""
    band = 50 + (26 if handles else 0)
    fig.tight_layout(rect=(0, 0, 1, 1 - band / (fig.get_figheight() * 72)))

    def at(dy):
        return offset_copy(fig.transFigure, fig=fig, x=0, y=-dy, units="points")

    fig.text(0.01, 1, title, transform=at(4), va="top", ha="left",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1, subtitle, transform=at(26), va="top", ha="left",
             fontsize=9.5, color=INK_2)
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1),
                   bbox_transform=at(42), ncol=len(handles), fontsize=9.5,
                   handlelength=2.2, columnspacing=1.6, labelcolor=INK_2,
                   borderaxespad=0, borderpad=0)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


# ---------------------------------------------------------------------------
# 01-04  metric vs dictionary size, one panel per dataset
# ---------------------------------------------------------------------------
def metric_vs_atoms(df, metric: str, name: str, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for j, clf in zip(JITTER, LINE_CLASSIFIERS):
            g = df[(df.dataset_label == ds) & (df.classifier == clf)]
            x = g.total_atoms * 2 ** j
            ax.errorbar(x, g[f"{metric}_mean"], yerr=g[f"{metric}_std"], color=CLF_COLOR[clf],
                        lw=1, alpha=0.45, capsize=0, fmt="none", zorder=2)
            ax.plot(x, g[f"{metric}_mean"], color=CLF_COLOR[clf], marker=CLF_MARKER[clf],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        atoms_axis(ax, recorded_atoms(df, ds))
        ax.set_title(ds, loc="left")
    axes[0].set_ylabel(metric)
    finish(fig, name, title,
           f"{N_NOTE}; bars = ±1 std across seeds. SRC (fddl) is not drawn: "
           "it matches SRC (pure) to within 0.004 on every metric (fig 10).",
           clf_handles(LINE_CLASSIFIERS))


# ---------------------------------------------------------------------------
# 05  change from 128 -> 2048 atoms
# ---------------------------------------------------------------------------
def gain_128_2048(_df) -> None:
    t = table("delta_128_to_2048")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    w = 0.16
    y0 = np.arange(len(DATASETS))
    for ax, metric in zip(axes, ["Macro-F1", "MCC"]):
        for i, clf in enumerate(LINE_CLASSIFIERS):
            g = t[t.classifier == clf].set_index("dataset").loc[DATASETS]
            ys = y0 + (i - 2) * w
            ax.barh(ys, g[f"{metric}_delta"], height=w - 0.03, color=CLF_COLOR[clf])
            for yy, v, p in zip(ys, g[f"{metric}_delta"], g[f"{metric}_p"]):
                if p < 0.05:
                    ax.text(v + (0.003 if v >= 0 else -0.003), yy, "*", va="center",
                            ha="left" if v >= 0 else "right", color=INK, fontsize=10)
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.set_yticks(y0)
        ax.set_yticklabels(DATASETS)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Change in {metric}", loc="left")
        ax.set_xlabel(f"{metric} at 2048 atoms − {metric} at 128 atoms")
    axes[0].invert_yaxis()  # shared y: inverting once flips both panels
    finish(fig, "fig05_change_128_to_2048",
           "Growing the dictionary from 128 to 2048 atoms helps the linear models, "
           "hurts SRC, and leaves RF flat",
           "2048 is the largest size recorded on all four datasets. * = Welch t-test "
           "p < 0.05 (5 vs 5 seeds; 10 for NCI1 @ 2048), unadjusted.",
           [Patch(color=CLF_COLOR[c], label=SHORT[c]) for c in LINE_CLASSIFIERS])


# ---------------------------------------------------------------------------
# 06  RF lead over the best linear classifier
# ---------------------------------------------------------------------------
def rf_gap(_df) -> None:
    t = table("rf_vs_linear")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, ["Macro-F1", "MCC"]):
        for ds in DATASETS:
            g = t[t.dataset == ds]
            ax.plot(g.atoms, g[f"{metric}_gap"], color=DS_COLOR[ds], marker=DS_MARKER[ds],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.axhline(0, color=INK_2, lw=0.9)
        atoms_axis(ax)
        ax.set_title(f"RF − best linear classifier, {metric}", loc="left")
        ax.set_ylabel(f"{metric} gap")
    finish(fig, "fig06_rf_lead_over_linear",
           "Random Forest's lead over the linear classifiers shrinks as the dictionary grows",
           "Best linear = the higher of LogReg and LinSVM in each combination. "
           "Above 0 = RF ahead (it is ahead in all 22).", ds_handles())


# ---------------------------------------------------------------------------
# 07  overview heatmap (the table view of every headline number)
# ---------------------------------------------------------------------------
def heatmap(df) -> None:
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9.4))
    rows = df[["dataset_label", "total_atoms"]].drop_duplicates()
    labels = [f"{d}  ·  {a}" for d, a in rows.itertuples(index=False)]
    for ax, metric in zip(axes, ["Macro-F1", "MCC"]):
        piv = df.pivot_table(index=["dataset_label", "total_atoms"], columns="classifier",
                             values=f"{metric}_mean", observed=True)[CLASSIFIERS]
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
        for (i, j), v in np.ndenumerate(piv.values):
            dark = (v - lo) / (hi - lo) > 0.55
            best = v == piv.values[i].max()
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK, fontweight="bold" if best else "normal")
        ax.set_xticks(range(len(CLASSIFIERS)))
        ax.set_xticklabels([SHORT[c] for c in CLASSIFIERS])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.grid(False)
        for b in np.cumsum(rows.groupby("dataset_label", observed=True).size().values)[:-1]:
            ax.axhline(b - 0.5, color=SURFACE, lw=3)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(metric, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelcolor=INK_2)
    finish(fig, "fig07_overview_heatmap",
           "Every recorded wl_fddl result: dataset × atoms × classifier",
           f"{N_NOTE}. Bold = best classifier in the row.")


# ---------------------------------------------------------------------------
# 08  classifier mean rank (Friedman / Nemenyi)
# ---------------------------------------------------------------------------
def ranks(_df) -> None:
    r, f = table("classifier_ranks"), table("friedman_tests").set_index("metric")
    metrics = ["Macro-F1", "MCC", "Minority-PR-AUC", "ROC-AUC"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    y = np.arange(len(CLASSIFIERS))
    for ax, metric in zip(axes, metrics):
        g = r[r.metric == metric].set_index("classifier").loc[CLASSIFIERS]
        cd = f.loc[metric, "nemenyi_cd_0.05"]
        best = g.mean_rank.min()
        ax.axvspan(best, best + cd, color=GRID, alpha=0.7, lw=0)
        ax.hlines(y, 1, g.mean_rank, color=GRID, lw=1)
        ax.scatter(g.mean_rank, y, color=SLOTS[0], s=60, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        for yy, v, n in zip(y, g.mean_rank, g.n_first):
            ax.text(v + 0.12, yy, f"{v:.2f}" + (f"  ({n}× 1st)" if n else ""),
                    va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[c] for c in CLASSIFIERS])
        ax.set_xlim(1, 6.9)
        ax.set_xticks(range(1, 7))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean rank (1 = best)")
        ax.set_title(f"{metric}   (Friedman p = {f.loc[metric, 'friedman_p']:.1e})", loc="left",
                     fontsize=10)
    axes[0].invert_yaxis()
    finish(fig, "fig08_classifier_ranks",
           "Random Forest ranks first in every one of the 22 combinations on Macro-F1, MCC "
           "and Minority-PR-AUC",
           f"Grey band = Nemenyi critical difference (α = 0.05, CD = {f.loc['MCC', 'nemenyi_cd_0.05']:.2f}) "
           "from the best mean rank. ROC-AUC 1st-place counts include ties "
           "(SRC pure/fddl tie once).")


# ---------------------------------------------------------------------------
# 09  minority-class precision/recall trajectory as atoms grow
# ---------------------------------------------------------------------------
def pr_trajectory(df) -> None:
    clfs = ["RandomForest", "LogisticRegression", "SRC_pure"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for clf in clfs:
            g = df[(df.dataset_label == ds) & (df.classifier == clf)].sort_values("total_atoms")
            x, y = g["Minority-Recall_mean"], g["Minority-Precision_mean"]
            ax.plot(x, y, color=CLF_COLOR[clf], lw=1.4, alpha=0.6, zorder=2)
            sizes = np.interp(np.log2(g.total_atoms), [7, 12], [25, 110])
            ax.scatter(x, y, s=sizes, color=CLF_COLOR[clf], marker=CLF_MARKER[clf],
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)
            if clf == "SRC_pure":
                for i in (0, len(g) - 1):
                    ax.annotate(str(g.total_atoms.iloc[i]), (x.iloc[i], y.iloc[i]),
                                xytext=(6, 5), textcoords="offset points", fontsize=8,
                                color=INK_2)
        ax.set_title(ds, loc="left")
        ax.set_xlabel("Minority recall")
    axes[0].set_ylabel("Minority precision")
    finish(fig, "fig09_minority_precision_recall_paths",
           "Larger dictionaries push SRC towards minority recall at the cost of precision",
           "Paths connect atom sizes in order; marker size grows with atoms "
           "(SRC end points labelled). RF and LogReg move far less than SRC.",
           clf_handles(clfs))


# ---------------------------------------------------------------------------
# 10  SRC_pure vs SRC_fddl
# ---------------------------------------------------------------------------
def src_identity(df) -> None:
    t = table("src_pure_vs_fddl")
    p = df[df.classifier == "SRC_pure"].reset_index(drop=True)
    f = df[df.classifier == "SRC_fddl"].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    xs = np.concatenate([p[f"{m}_mean"] for m in t.metric])
    ys = np.concatenate([f[f"{m}_mean"] for m in t.metric])
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1)
    a1.scatter(xs, ys, s=22, color=SLOTS[0], edgecolor=SURFACE, linewidth=0.8, zorder=3)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_aspect("equal")
    a1.set_xlabel("SRC (pure)")
    a1.set_ylabel("SRC (fddl)")
    a1.set_title(f"All {len(xs)} metric means (11 metrics × 22 combinations)", loc="left",
                 fontsize=10)

    t = t.sort_values("max_abs_diff")
    a2.barh(t.metric, t.max_abs_diff, color=SLOTS[0], height=0.6, label="largest |Δ| over 22 combinations")
    a2.scatter(t.mean_seed_std_pure, t.metric, color=INK, marker="|", s=120, zorder=3,
               label="mean seed std of SRC (pure)")
    for v, m in zip(t.max_abs_diff, t.metric):
        a2.text(v + 0.0005, m, f"{v:.4f}", va="center", fontsize=8.5, color=INK_2)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Absolute difference")
    a2.set_title("Largest gap per metric vs seed-to-seed noise", loc="left", fontsize=10)
    a2.legend(loc="lower right", fontsize=8.5, labelcolor=INK_2)
    finish(fig, "fig10_src_pure_vs_fddl",
           "SRC (pure) and SRC (fddl) give practically the same answers",
           "Every gap between the two SRC classifiers is several times smaller than the "
           "spread across seeds of either one.")


# ---------------------------------------------------------------------------
# 11  which metrics move with dictionary size
# ---------------------------------------------------------------------------
def metric_sensitivity(df) -> None:
    metrics = ["Accuracy", "ROC-AUC", "Macro-PR-AUC", "Macro-F1", "MCC"]
    colors = dict(zip(metrics, SLOTS))
    clfs = ["RandomForest", "LogisticRegression", "SRC_pure"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.8), sharey=True)
    for r, clf in enumerate(clfs):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            g = df[(df.dataset_label == ds) & (df.classifier == clf)].sort_values("total_atoms")
            for k, m in enumerate(metrics):
                ax.plot(g.total_atoms, g[f"{m}_mean"] - g[f"{m}_mean"].iloc[0],
                        color=colors[m], marker=METRIC_MARKERS[k], markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            ax.axhline(0, color=INK_2, lw=0.9)
            atoms_axis(ax, recorded_atoms(df, ds))
            if r < 2:
                ax.set_xlabel("")
            if r == 0:
                ax.set_title(ds, loc="left")
            if c == 0:
                ax.set_ylabel(f"{SHORT[clf]}\nchange from 128 atoms")
    finish(fig, "fig11_metric_sensitivity",
           "ROC-AUC and Macro-PR-AUC barely react to dictionary size; MCC and Macro-F1 "
           "carry the effect (and Accuracy, for SRC)",
           "Each line is a metric's mean minus its own value at 128 atoms. Rows: classifier; "
           "columns: dataset.",
           [Line2D([], [], color=colors[m], marker=METRIC_MARKERS[k], lw=2,
                   markeredgecolor=SURFACE, label=m) for k, m in enumerate(metrics)])


# ---------------------------------------------------------------------------
# 12  runtime scaling and breakdown
# ---------------------------------------------------------------------------
def runtime(_df) -> None:
    t = table("runtime")
    t["other_classical"] = t.classical_classifiers - t.gboost_only
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1, 1.2]})
    for ds in DATASETS:
        g = t[t.dataset == ds]
        a1.plot(g.atoms, g.seed_total, color=DS_COLOR[ds], marker=DS_MARKER[ds],
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=ds)
    a1.set_yscale("log", base=2)
    yt = [32, 64, 128, 256, 512, 1024]
    a1.set_yticks(yt)
    a1.set_yticklabels([str(v) for v in yt])
    a1.minorticks_off()
    atoms_axis(a1)
    a1.set_ylabel("Seconds per seed (log scale)")
    a1.set_title("Total time per seed (NCI1, NCI33, NCI47 overlap)", loc="left", fontsize=10)
    a1.legend(loc="upper left", labelcolor=INK_2)

    g = t[t.dataset == "NCI1"]
    parts = [("data_load", "Data load"), ("fit_encoder_dict", "WL + dictionary fit"),
             ("sparse_coding", "Sparse coding (train/val/test)"),
             ("gboost_only", "Gradient Boosting (val + test)"),
             ("other_classical", "LogReg + LinSVM + RF (val + test)"),
             ("src_classifiers", "SRC (embeddings + pure + fddl)")]
    colors = ["#c3c2b7"] + SLOTS[:5]
    x = np.arange(len(g))
    bottom = np.zeros(len(g))
    for (col, lab), colr in zip(parts, colors):
        a2.bar(x, g[col], bottom=bottom, color=colr, width=0.62, label=lab,
               edgecolor=SURFACE, linewidth=1.5)
        bottom += g[col].values
    for xi, tot in zip(x, g.seed_total):
        a2.text(xi, tot + 8, f"{tot:.0f}s", ha="center", fontsize=8.5, color=INK_2)
    a2.set_xticks(x)
    a2.set_xticklabels(g.atoms.astype(str))
    a2.set_xlabel("Dictionary atoms")
    a2.set_ylabel("Seconds per seed")
    a2.grid(axis="x", visible=False)
    a2.set_title("Where the time goes (NCI1)", loc="left", fontsize=10)
    handles, labels = a2.get_legend_handles_labels()
    a2.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=8.5, labelcolor=INK_2)
    finish(fig, "fig12_runtime",
           "Runtime grows almost linearly with atoms, and Gradient Boosting, not the "
           "dictionary, is the largest cost",
           "Mean wall-clock time per seed. The phases sum to the recorded seed total.")


# ---------------------------------------------------------------------------
# 13  cost vs performance
# ---------------------------------------------------------------------------
def cost_vs_perf(df) -> None:
    clfs = ["RandomForest", "LogisticRegression"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for clf in clfs:
            g = df[(df.dataset_label == ds) & (df.classifier == clf)].sort_values("total_atoms")
            ax.plot(g.seed_total_sec_mean, g["Macro-F1_mean"], color=CLF_COLOR[clf],
                    marker=CLF_MARKER[clf], markeredgecolor=SURFACE, markeredgewidth=1.2)
            for _, row in g.iterrows():
                ax.annotate(str(row.total_atoms), (row.seed_total_sec_mean, row["Macro-F1_mean"]),
                            xytext=(0, -13 if clf == "LogisticRegression" else 7),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512])
        ax.set_xticklabels(["32", "64", "128", "256", "512"])
        ax.minorticks_off()
        ax.set_xlabel("Seconds per seed (log scale)")
        ax.set_title(ds, loc="left")
    axes[0].set_ylabel("Macro-F1")
    finish(fig, "fig13_cost_vs_macro_f1",
           "Past 1024 atoms, Random Forest gains at most +0.005 Macro-F1 for 2–4× the run time",
           "Points are labelled with their atom count. LogReg keeps improving on NCI1, NCI33 "
           "and NCI47 but never catches RF.", clf_handles(clfs))


# ---------------------------------------------------------------------------
# 14  seed-to-seed variability
# ---------------------------------------------------------------------------
def variability(df) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for clf in LINE_CLASSIFIERS:
            g = df[(df.dataset_label == ds) & (df.classifier == clf)]
            ax.plot(g.total_atoms, g.MCC_std, color=CLF_COLOR[clf], marker=CLF_MARKER[clf],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        atoms_axis(ax, recorded_atoms(df, ds))
        ax.set_title(ds, loc="left")
    axes[0].set_ylabel("Std of MCC across seeds")
    finish(fig, "fig14_seed_variability",
           "Seed-to-seed spread of MCC shows no consistent trend with dictionary size",
           "Sample std (ddof = 1) over 5 seeds (10 for NCI1 @ 2048).",
           clf_handles(LINE_CLASSIFIERS))


# ---------------------------------------------------------------------------
# 15  replicate runs at NCI1 / 2048
# ---------------------------------------------------------------------------
def replicate(_df) -> None:
    t = table("replicate_nci1_2048")
    metrics = ["Macro-F1", "MCC", "ROC-AUC", "Minority-PR-AUC"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    y = np.arange(len(CLASSIFIERS))
    for ax, m in zip(axes, metrics):
        g = t[t.metric == m].set_index("classifier").loc[CLASSIFIERS]
        ax.hlines(y, g.run1, g.run2, color=MUTED, lw=2)
        ax.scatter(g.run1, y, color=SLOTS[0], s=55, zorder=3, edgecolor=SURFACE)
        ax.scatter(g.run2, y, color=SLOTS[1], s=55, marker="s", zorder=3, edgecolor=SURFACE)
        for yy, a, b, d in zip(y, g.run1, g.run2, g.abs_diff):
            ax.text(max(a, b) + 0.006, yy, f"Δ {d:.4f}", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[c] for c in CLASSIFIERS])
        ax.grid(axis="y", visible=False)
        ax.set_title(m, loc="left")
        lo, hi = min(g.run1.min(), g.run2.min()), max(g.run1.max(), g.run2.max())
        ax.set_xlim(lo - 0.01, hi + 0.045)
    axes[0].invert_yaxis()
    r = t.iloc[0]
    finish(fig, "fig15_replicate_runs",
           "Two independent runs of the same setting (NCI1, 2048 atoms): the noise floor",
           f"Each point is a 5-seed mean. Run 1 seeds: {r.run1_seeds}.  Run 2 seeds: "
           f"{r.run2_seeds}.",
           [Line2D([], [], color=SLOTS[0], marker="o", lw=0, label="Run 1"),
            Line2D([], [], color=SLOTS[1], marker="s", lw=0, label="Run 2")])


# ---------------------------------------------------------------------------
# 16  feature selection is independent of atoms
# ---------------------------------------------------------------------------
def features(_df) -> None:
    t = table("features")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for ds in DATASETS:
        g = t[t.dataset == ds]
        kw = dict(color=DS_COLOR[ds], marker=DS_MARKER[ds], markeredgecolor=SURFACE,
                  markeredgewidth=1.2)
        a1.errorbar(g.total_atoms, g.n_selected_features_mean, yerr=g.n_selected_features_std,
                    capsize=0, elinewidth=1, **kw)
        a2.plot(g.total_atoms, 100 * g.kept_fraction_mean, **kw)
    for ax in (a1, a2):
        atoms_axis(ax)
    a1.set_ylim(0, None)
    a1.set_ylabel("WL features kept (mean ± std)")
    a1.set_title("Features passed to the dictionary", loc="left")
    a2.set_ylim(0, None)
    a2.set_ylabel("% of scored WL features kept")
    a2.set_title("Kept fraction (energy = 0.99)", loc="left")
    finish(fig, "fig16_selected_features",
           "The input feature set does not change with dictionary size",
           "Feature selection runs before dictionary learning, so every atom-size effect "
           "comes from the dictionary itself.", ds_handles())


# ---------------------------------------------------------------------------
# 17  dataset difficulty
# ---------------------------------------------------------------------------
def dataset_difficulty(_df) -> None:
    t = table("datasets").set_index("dataset").loc[DATASETS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    y = np.arange(len(LINE_CLASSIFIERS))
    for ax, (key, lab) in zip(axes, [("MCC", "MCC"), ("MacroF1", "Macro-F1")]):
        for ds in DATASETS:
            vals = [t.loc[ds, f"{c}_{key}_mean_128_2048"] for c in LINE_CLASSIFIERS]
            ax.scatter(vals, y, color=DS_COLOR[ds], marker=DS_MARKER[ds], s=60, zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
        for yy, c in zip(y, LINE_CLASSIFIERS):
            v = [t.loc[d, f"{c}_{key}_mean_128_2048"] for d in DATASETS]
            ax.hlines(yy, min(v), max(v), color=GRID, lw=2, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[c] for c in LINE_CLASSIFIERS])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"{lab}, mean over 128–2048 atoms")
        ax.set_title(lab, loc="left")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    sizes = ";  ".join(f"{d} {int(t.loc[d, 'split_ml_train'])}/{int(t.loc[d, 'split_test'])}"
                       for d in DATASETS)
    finish(fig, "fig17_dataset_difficulty",
           "NCI33 is the hardest screen for every classifier",
           f"Averaged over the 5 atom sizes recorded on every dataset. "
           f"Classifier train/test graphs: {sizes}.", ds_handles())


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(exist_ok=True)
    df = load()
    metric_vs_atoms(df, "Macro-F1", "fig01_macro_f1_vs_atoms",
                    "Macro-F1 vs dictionary size: RF leads everywhere, linear models climb, SRC falls")
    metric_vs_atoms(df, "MCC", "fig02_mcc_vs_atoms", "MCC vs dictionary size")
    metric_vs_atoms(df, "Minority-PR-AUC", "fig03_minority_pr_auc_vs_atoms",
                    "Minority-class PR-AUC vs dictionary size")
    metric_vs_atoms(df, "ROC-AUC", "fig04_roc_auc_vs_atoms",
                    "ROC-AUC vs dictionary size: every result sits between 0.795 and 0.864")
    gain_128_2048(df)
    rf_gap(df)
    heatmap(df)
    ranks(df)
    pr_trajectory(df)
    src_identity(df)
    metric_sensitivity(df)
    runtime(df)
    cost_vs_perf(df)
    variability(df)
    replicate(df)
    features(df)
    dataset_difficulty(df)


if __name__ == "__main__":
    main()
