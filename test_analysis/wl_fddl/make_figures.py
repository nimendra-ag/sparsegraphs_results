"""Figures + key numbers for the WL + FDDL analysis (see REPORT.md).

Reads all_results.csv (build_all_results.py). Where one (dataset, atoms) cell
has two executions (NCI1 @ 2048) they are averaged first, so every cell counts
once.

    python test_analysis/wl_fddl/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# fixed classifier -> colour/marker (validated categorical slots 1-5, light mode)
STYLE = {
    "RandomForest":       ("#2a78d6", "o", "Random Forest"),
    "LogisticRegression": ("#eb6834", "s", "Logistic Reg."),
    "LinearSVM":          ("#1baf7a", "^", "Linear SVM"),
    "GradientBoosting":   ("#eda100", "D", "Gradient Boost."),
    "SRC_pure":           ("#e87ba4", "v", "SRC (dictionary)"),
}
LEARNED = ["RandomForest", "LogisticRegression", "LinearSVM", "GradientBoosting"]
DATASETS = ["NCI1", "NCI33", "NCI41", "NCI47"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def load():
    df = pd.read_csv(HERE / "all_results.csv")
    num = df.select_dtypes("number").columns
    g = df.groupby(["dataset", "total_atoms", "classifier"], as_index=False)[num].mean()
    g["total_atoms"] = g.total_atoms.astype(int)
    return g


def atoms_axis(ax, atoms):
    ax.set_xscale("log", base=2)
    ax.set_xticks(atoms)
    ax.set_xticklabels([str(a) for a in atoms])
    ax.minorticks_off()


def fig1_overview(g):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = g[g.dataset == ds]
        for clf, (c, m, lab) in STYLE.items():
            s = sub[sub.classifier == clf].sort_values("total_atoms")
            ax.plot(s.total_atoms, s["Macro-F1"], color=c, marker=m, label=lab)
            if clf == "RandomForest":
                ax.fill_between(s.total_atoms, s["Macro-F1"] - s["Macro-F1_std"],
                                s["Macro-F1"] + s["Macro-F1_std"], color=c, alpha=0.12, lw=0)
        atoms_axis(ax, sorted(sub.total_atoms.unique()))
        ax.set_title(ds, loc="left")
        ax.set_xlabel("Dictionary size (total atoms)")
    axes[0].set_ylabel("Macro-F1 (test, mean of 5 seeds)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Fig. 1  Macro-F1 vs dictionary size, per dataset and classifier "
                 "(shaded = ±1 seed-std for Random Forest)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_overview_macro_f1.png", dpi=170)
    plt.close(fig)


def fig2_atoms_effect(g):
    base = g[g.total_atoms <= 2048]  # 128-2048 exists for all four datasets
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.05, 1, 1.15]})

    # (a) gain 128 -> 2048 per classifier, one dot per dataset
    ax = axes[0]
    q = base.pivot_table(index=["classifier", "dataset"], columns="total_atoms", values="Macro-F1")
    gain = (q[2048] - q[128]).unstack(1)
    order = list(STYLE)
    for i, clf in enumerate(order):
        c, m, lab = STYLE[clf]
        vals = gain.loc[clf, DATASETS].values
        ax.barh(i, vals.mean(), color=c, alpha=0.35, height=0.6)
        ax.scatter(vals, np.full(4, i), color=c, marker=m, s=34, zorder=3, edgecolor=SURF, lw=0.8)
        ax.text(0.088, i, f"{vals.mean():+.3f}", va="center", ha="right", color=INK, fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([STYLE[c][2] for c in order])
    ax.invert_yaxis()
    ax.set_xlim(-0.14, 0.09)
    ax.set_xticks([-0.12, -0.08, -0.04, 0, 0.04])
    ax.set_xlabel("Δ Macro-F1, 128 → 2048 atoms (bar = mean)")
    ax.set_title("(a) Who benefits from a larger dictionary?", loc="left")
    ax.grid(axis="y", visible=False)

    # (b) RF lead over the best linear model vs atoms
    ax = axes[1]
    p = g[g.classifier.isin(LEARNED)].pivot_table(index=["dataset", "total_atoms"],
                                                    columns="classifier", values="Macro-F1")
    lead = (p.RandomForest - p[["LogisticRegression", "LinearSVM"]].max(axis=1)).unstack(0)
    noise = g[g.classifier == "RandomForest"]["Macro-F1_std"].mean()
    ax.axhspan(0, noise, color=GRID, alpha=0.9, lw=0)
    ax.text(150, noise * 0.5, "within 1 seed-std (noise)", va="center", fontsize=8, color=INK2)
    for ds in DATASETS:
        s = lead[ds].dropna()
        ax.plot(s.index, s.values, color="#86b6ef", lw=1.2, marker="o", ms=3.5)
    mean_lead = lead.loc[[a for a in lead.index if a <= 2048]].mean(axis=1)
    ax.plot(mean_lead.index, mean_lead.values, color=STYLE["RandomForest"][0], lw=2.5, marker="o",
            label="mean of 4 datasets")
    ax.plot([], [], color="#86b6ef", lw=1.2, marker="o", ms=3.5, label="individual dataset")
    atoms_axis(ax, sorted(g.total_atoms.unique()))
    ax.set_ylim(0, 0.05)
    ax.set_xlabel("Dictionary size (total atoms)")
    ax.set_ylabel("RF − best linear (Macro-F1)")
    ax.set_title("(b) RF lead shrinks as atoms grow", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)

    # (c) cost vs quality, averaged over the 4 datasets
    ax = axes[2]
    m = base.groupby(["classifier", "total_atoms"], as_index=False)[["Macro-F1", "pipeline_sec"]].mean()
    for clf in LEARNED:
        c, mk, lab = STYLE[clf]
        s = m[m.classifier == clf].sort_values("total_atoms")
        ax.plot(s.pipeline_sec, s["Macro-F1"], color=c, marker=mk, label=lab)
        if clf in ("RandomForest", "LogisticRegression"):
            pts = s if clf == "RandomForest" else s[s.total_atoms.isin([128, 2048])]
            for _, r in pts.iterrows():
                xy, ha = ((7, -3), "left") if (clf, r.total_atoms) == ("LogisticRegression", 128)                     else (((0, 7) if clf == "RandomForest" else (0, -13)), "center")
                ax.annotate(f"{int(r.total_atoms)}", (r.pipeline_sec, r["Macro-F1"]),
                            textcoords="offset points", xytext=xy, ha=ha,
                            fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200])
    ax.set_xticklabels(["25", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("Time per seed: features + classifier (s, log)")
    ax.set_ylabel("Macro-F1 (mean of 4 datasets)")
    ax.set_title("(c) Quality vs cost (labels = atoms)", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Fig. 2  Effect of dictionary size: linear models need many atoms, Random Forest does not",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_dictionary_size_effect.png", dpi=170)
    plt.close(fig)
    return gain, lead, m


def fig3_src(g):
    base = g[g.total_atoms <= 2048]
    m = base.groupby(["classifier", "total_atoms"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    atoms = sorted(base.total_atoms.unique())

    ax = axes[0]
    for clf in ("SRC_pure", "RandomForest"):
        c, mk, lab = STYLE[clf]
        s = m[m.classifier == clf].sort_values("total_atoms")
        ax.plot(s.total_atoms, s["Minority-Recall"], color=c, marker=mk, label=f"{lab}: recall")
        ax.plot(s.total_atoms, s["Minority-Precision"], color=c, marker=mk, ls=(0, (1, 1.5)),
                mfc=SURF, label=f"{lab}: precision")
    atoms_axis(ax, atoms)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Dictionary size (total atoms)")
    ax.set_ylabel("Minority (active) class")
    ax.set_title("(a) SRC trades precision for recall as atoms grow", loc="left")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    ax = axes[1]
    for clf, (c, mk, lab) in STYLE.items():
        s = m[m.classifier == clf].sort_values("total_atoms")
        ax.plot(s.total_atoms, s["MCC"], color=c, marker=mk, label=lab)
    atoms_axis(ax, atoms)
    ax.set_xlabel("Dictionary size (total atoms)")
    ax.set_ylabel("MCC (mean of 4 datasets)")
    ax.set_title("(b) MCC: SRC falls while learned classifiers rise", loc="left")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.suptitle("Fig. 3  The FDDL dictionary used directly as a classifier (SRC) degrades with size "
                 "(SRC with/without Fisher term are identical; one line shown)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_src_degradation.png", dpi=170)
    plt.close(fig)


def fig4_metrics(g):
    metrics = ["Accuracy", "ROC-AUC", "Macro-PR-AUC", "Macro-F1", "Minority-F1", "MCC", "Minority-PR-AUC"]
    p = g.groupby("classifier")[metrics].mean()
    rel = 100 * p / p.loc["RandomForest"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axvline(100, color=STYLE["RandomForest"][0], lw=2)
    ax.text(99.6, len(metrics) - 0.45, "Random Forest = 100", color=INK2, fontsize=8.5, ha="right", va="center")
    offs = {"LogisticRegression": -0.24, "LinearSVM": -0.08, "GradientBoosting": 0.08, "SRC_pure": 0.24}
    for clf, dy in offs.items():
        c, mk, lab = STYLE[clf]
        ax.scatter(rel.loc[clf, metrics], np.arange(len(metrics)) + dy, color=c, marker=mk, s=40,
                   label=lab, zorder=3, edgecolor=SURF, lw=0.8)
    for i, met in enumerate(metrics):
        lo = rel.loc[["LogisticRegression", "LinearSVM", "GradientBoosting", "SRC_pure"], met].min()
        ax.plot([lo, 100], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.text(48, i, f"RF = {p.loc['RandomForest', met]:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(47, 104)
    ax.set_xlabel("Score as % of Random Forest (mean over all 22 dataset × atom cells)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(metrics) - 0.2, -0.6)
    ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Fig. 4  Accuracy and ROC-AUC compress the differences that MCC and minority metrics expose",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_metric_sensitivity.png", dpi=170)
    plt.close(fig)
    return rel


def key_numbers(g, gain, lead, cost, rel):
    lines = []
    p = g[g.classifier.isin(LEARNED)]
    for met in ["Macro-F1", "MCC", "Minority-PR-AUC", "ROC-AUC"]:
        t = p.pivot_table(index=["dataset", "total_atoms"], columns="classifier", values=met)
        best_lin = t[["LogisticRegression", "LinearSVM"]].max(axis=1)
        d = t.RandomForest - best_lin
        wins = (t.idxmax(axis=1) == "RandomForest").sum()
        fr = stats.friedmanchisquare(*[t[c] for c in LEARNED]).pvalue
        wx = stats.wilcoxon(t.RandomForest, best_lin).pvalue
        ranks = t.rank(axis=1, ascending=False).mean().round(2).to_dict()
        lines.append(f"{met}: RF best in {wins}/{len(t)} cells; RF-best_linear mean {d.mean():+.3f} "
                     f"[{d.min():+.3f},{d.max():+.3f}]; Friedman p={fr:.1e}; Wilcoxon p={wx:.1e}; ranks {ranks}")
    lines.append("\nMacro-F1 gain 128->2048 (rows=classifier):\n" + gain.round(3).to_string())
    lines.append("\nRF lead over best linear (Macro-F1):\n" + lead.round(3).to_string())
    for clf in g.classifier.unique():
        s = g[g.classifier == clf]
        rho, pv = stats.spearmanr(np.log2(s.total_atoms), s["Macro-F1"])
        lines.append(f"Spearman log2(atoms)~Macro-F1 {clf}: rho={rho:+.2f} p={pv:.1e}")
    s = g.pivot_table(index=["dataset", "total_atoms"], columns="classifier", values="Macro-F1")
    lines.append(f"max |SRC_fddl - SRC_pure| Macro-F1 = {(s.SRC_fddl - s.SRC_pure).abs().max():.4f}")
    lines.append("\nMean seed-std: " + g.groupby("classifier")[["Macro-F1_std", "MCC_std"]].mean().round(4).to_string())
    lines.append("\nCost vs quality (mean of 4 datasets, atoms<=2048):\n" + cost.round(3).to_string())
    lines.append("\nScore as % of RF:\n" + rel.round(1).to_string())
    raw = pd.read_csv(HERE / "all_results.csv")
    r = raw[(raw.dataset == "NCI1") & (raw.total_atoms == 2048)]
    rep = r.pivot_table(index="classifier", columns="run_folder", values=["Macro-F1", "MCC"])
    gap = pd.DataFrame({m: (rep[m].iloc[:, 0] - rep[m].iloc[:, 1]).abs() for m in ["Macro-F1", "MCC"]})
    lines.append("\nNCI1@2048 replicate executions, |difference|:\n" + gap.round(4).to_string())
    (HERE / "key_numbers.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    g = load()
    fig1_overview(g)
    gain, lead, cost = fig2_atoms_effect(g)
    fig3_src(g)
    rel = fig4_metrics(g)
    key_numbers(g, gain, lead, cost, rel)


if __name__ == "__main__":
    main()
