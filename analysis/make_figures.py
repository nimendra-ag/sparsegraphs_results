"""Standard evaluation scenarios for the encoder x dict-learner study.

Reads analysis/all_results.csv (built by build_all_results.py) and writes one
figure per scenario to analysis/figures/.

Every scenario fixes some factors and varies exactly one, so each figure answers
a single question. The fixed factors are named on the figure's second title line.

Styling follows utils/elbow.py, the project's existing figure house style:
default matplotlib chrome (white surface, full spine box, DejaVu Sans), the
tab10 palette, framed legends at fontsize 8, ``grid(alpha=0.25)``, monospace
summary boxes in a rounded #cccccc frame, and 150 dpi output.

Run with the project env:
  "C:/Users/Puldith CE/miniconda3/envs/genv12/python.exe" analysis/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis" / "all_results.csv"
FIGDIR = ROOT / "analysis" / "figures"

DPI = 150

# --- house palette (tab10, as used across utils/elbow.py) --------------------
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
         "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

GREY = "#7f7f7f"
NOTE_GREY = "#666666"
BOX_EDGE = "#cccccc"
SHADE = "#2ca02c"          # the green "selected/kept" shading in elbow.py

# Marker shape doubles as an identity channel, so a series is never told apart
# by colour alone -- the same reason elbow.py rings its cut markers.
LEARNERS = ["fddl", "aksvd", "lcksvd", "online_dl", "frozen_ksvd", "bayesian"]
L_COLOR = dict(zip(LEARNERS, TAB10[:6]))
L_MARK = dict(zip(LEARNERS, ["o", "s", "^", "D", "v", "P"]))

ENCODERS = ["wl", "fsm", "gspan_cork"]
E_COLOR = dict(zip(ENCODERS, TAB10[:3]))
E_MARK = dict(zip(ENCODERS, ["o", "s", "^"]))

CLASSIFIERS = ["LogisticRegression", "GradientBoosting", "LinearSVM", "RandomForest"]
CLF_SHORT = {
    "LogisticRegression": "LogReg",
    "GradientBoosting": "GBoost",
    "LinearSVM": "LinSVM",
    "RandomForest": "RF",
}

METRIC_LABEL = {
    "roc_auc": "ROC-AUC",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
    "minority_f1": "Minority-F1",
    "minority_pr_auc": "Minority-PR-AUC",
    "minority_recall": "Minority-Recall",
    "minority_precision": "Minority-Precision",
    "accuracy": "Accuracy",
    "macro_pr_auc": "Macro-PR-AUC",
}

# Only the knobs elbow.py actually sets; everything else stays at the
# matplotlib default so these figures sit beside the elbow suite unchanged.
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI})


def style(ax, xlabel="", ylabel="", which="major"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which=which, alpha=0.25)
    ax.set_axisbelow(True)


def title2(n_lines_owner, main, detail):
    """The house two-line title: headline, then the fixed factors beneath it."""
    text = f"{main}\n{detail}"
    if isinstance(n_lines_owner, mpl.figure.Figure):
        n_lines_owner.suptitle(text, fontsize=12)
    else:
        n_lines_owner.set_title(text)


def summary_box(ax, lines, x=0.985, y=0.98, ha="right"):
    """Monospace panel in a rounded frame -- elbow.py's ``_summary_box``."""
    ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes,
        fontsize=8, family="monospace", ha=ha, va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=BOX_EDGE, alpha=0.9),
        zorder=8,
    )


def footnote(ax, text):
    ax.text(0.5, -0.13, text, transform=ax.transAxes, fontsize=7.5,
            color=NOTE_GREY, ha="center", va="top")


def log2_axis(ax, atoms, fontsize=None):
    vals = sorted({int(a) for a in atoms})
    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.set_xticklabels([str(v) for v in vals], **({"fontsize": fontsize} if fontsize else {}))
    ax.tick_params(axis="x", which="minor", bottom=False)


def repel(ax, items, fontsize=8, max_iter=140, pad=1.4):
    """Place point labels, then push overlapping ones apart in display space.

    items: (x, y, text, colour).
    """
    fig = ax.figure
    texts, offs = [], []
    for x, y, s, c in items:
        t = ax.annotate(s, (x, y), xytext=(0, 11), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize, color=c, zorder=6)
        texts.append(t)
        offs.append([0.0, 11.0])

    for _ in range(max_iter):
        fig.canvas.draw()
        boxes = [t.get_window_extent() for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                moved = True
                dx = 0.0 if abs(boxes[i].x0 - boxes[j].x0) < 1 else (
                    pad if boxes[i].x0 > boxes[j].x0 else -pad)
                dy = pad if boxes[i].y0 >= boxes[j].y0 else -pad
                offs[i][0] += dx * 0.5
                offs[i][1] += dy
                offs[j][0] -= dx * 0.5
                offs[j][1] -= dy
                texts[i].set_position(tuple(offs[i]))
                texts[j].set_position(tuple(offs[j]))
                boxes[i] = texts[i].get_window_extent()
                boxes[j] = texts[j].get_window_extent()
        if not moved:
            break
    return texts


def zoom_to_data(ax, xs, ys, pad=0.12):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xr = max(xs.max() - xs.min(), 1e-6)
    yr = max(ys.max() - ys.min(), 1e-6)
    ax.set_xlim(xs.min() - xr * pad * 1.6, xs.max() + xr * pad * 1.6)
    ax.set_ylim(ys.min() - yr * pad * 2.0, ys.max() + yr * pad * 2.2)


def save(fig, name, tight=True):
    """Write the figure. ``tight=False`` for figures that already ran
    ``tight_layout`` with a reserved margin -- a second unconstrained call
    would expand the axes back over whatever that margin was holding."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def load():
    df = pd.read_csv(CSV)
    df = df[df["source"] != "missing"].copy()
    df["atoms"] = df["atoms"].astype(int)
    return df


def best_per_combo(df, clf=None, by="macro_f1"):
    """One row per (encoder, dict_learner): the atoms that maximise `by`."""
    sub = df if clf is None else df[df["classifier"] == clf]
    sub = sub.dropna(subset=[by])
    return sub.loc[sub.groupby(["encoder", "dict_learner"])[by].idxmax()]


# ---------------------------------------------------------------------------
# S1 -- Dictionary-size sweep: which learner wins?
# ---------------------------------------------------------------------------
def s1(df, encoder="wl", clf="RandomForest", metric="roc_auc"):
    sub = df[(df["encoder"] == encoder) & (df["classifier"] == clf)]
    fig, ax = plt.subplots(figsize=(10, 6))

    ends, rows = [], []
    for learner in LEARNERS:
        d = sub[sub["dict_learner"] == learner].sort_values("atoms")
        if d.empty:
            continue
        c = L_COLOR[learner]
        if d[f"{metric}_std"].notna().any():
            ax.fill_between(
                d["atoms"], d[metric] - d[f"{metric}_std"], d[metric] + d[f"{metric}_std"],
                color=c, alpha=0.12, linewidth=0, zorder=1,
            )
        ax.plot(d["atoms"], d[metric], color=c, lw=1.6, marker=L_MARK[learner],
                markersize=5, markeredgecolor="black", markeredgewidth=0.6,
                label=learner, zorder=3)
        peak = d.loc[d[metric].idxmax()]
        ends.append((d.iloc[-1][metric], learner, c))
        rows.append(f"{learner:<12}{peak[metric]:.4f} @ {int(peak['atoms']):>5}")

    style(ax, "dictionary atoms (log\u2082, sorted ascending)", METRIC_LABEL[metric])
    log2_axis(ax, sub["atoms"])
    xmax = int(sub["atoms"].max())
    ax.set_xlim(right=xmax * 2.6)

    # Open headroom above the curves so the summary panel has somewhere to sit
    # that is not on top of a line.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.30)
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = sorted(ends, key=lambda e: e[0])
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < gap:
            placed[i] = (placed[i - 1][0] + gap, placed[i][1], placed[i][2])
    for y, text, c in placed:
        ax.annotate(text, (xmax, y), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=8, color=c)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    summary_box(ax, [f"best {METRIC_LABEL[metric]} per learner", "-" * 30] + rows,
                x=0.015, y=0.985, ha="left")
    title2(ax, f"Dictionary size sweep \u2014 {METRIC_LABEL[metric]}",
           f"encoder: {encoder}   classifier: {CLF_SHORT[clf]}   "
           f"band: \u00b11 SD across MC-CV seeds")
    save(fig, f"s1_learner_sweep_{encoder}_{CLF_SHORT[clf]}_{metric}.png")


# ---------------------------------------------------------------------------
# S2 -- Encoder comparison, one panel per learner
# ---------------------------------------------------------------------------
def s2(df, clf="RandomForest", metric="macro_f1"):
    sub = df[df["classifier"] == clf]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

    for ax, learner in zip(axes.ravel(), LEARNERS):
        d0 = sub[sub["dict_learner"] == learner]
        for enc in ENCODERS:
            d = d0[d0["encoder"] == enc].sort_values("atoms")
            if d.empty:
                continue
            c = E_COLOR[enc]
            single = (d["source"] == "artifact").all()
            ax.plot(d["atoms"], d[metric], color=c, lw=0 if single else 1.6,
                    linestyle="none" if single else "-", marker=E_MARK[enc],
                    markersize=6, markerfacecolor="white" if single else c,
                    markeredgecolor=c if single else "black",
                    markeredgewidth=1.4 if single else 0.6, zorder=3)
        ax.set_title(learner, fontsize=10)
        style(ax, "", "")
        log2_axis(ax, sub["atoms"], fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("dictionary atoms (log\u2082)")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL[metric])

    handles = [
        Line2D([], [], color=E_COLOR[e], marker=E_MARK[e], markersize=6, lw=1.6,
               markeredgecolor="black", markeredgewidth=0.6, label=e)
        for e in ("wl", "fsm")
    ] + [
        Line2D([], [], color=E_COLOR["gspan_cork"], marker=E_MARK["gspan_cork"],
               markersize=6, linestyle="none", markerfacecolor="white",
               markeredgecolor=E_COLOR["gspan_cork"], markeredgewidth=1.4,
               label="gspan_cork (single split)")
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               fontsize=8, framealpha=0.9)
    title2(fig, f"Encoder comparison \u2014 {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   one panel per dictionary learner   "
           "hollow markers are single-split artifacts, not MC-CV means")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    save(fig, f"s2_encoder_comparison_{CLF_SHORT[clf]}_{metric}.png", tight=False)


# ---------------------------------------------------------------------------
# S3 -- Does the ranking survive a change of metric?
# ---------------------------------------------------------------------------
def s3(df, encoder="wl", clf="RandomForest"):
    metrics = ["roc_auc", "macro_f1", "mcc", "minority_f1", "minority_pr_auc", "accuracy"]
    sub = df[(df["encoder"] == encoder) & (df["classifier"] == clf)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, metric in zip(axes.ravel(), metrics):
        for learner in LEARNERS:
            d = sub[sub["dict_learner"] == learner].sort_values("atoms")
            if d.empty:
                continue
            ax.plot(d["atoms"], d[metric], color=L_COLOR[learner], lw=1.6,
                    marker=L_MARK[learner], markersize=4.5, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=3)
        ax.set_title(METRIC_LABEL[metric] + ("  (flat by construction)"
                                             if metric == "accuracy" else ""),
                     fontsize=10)
        style(ax, "", "")
        log2_axis(ax, sub["atoms"], fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("dictionary atoms (log\u2082)")

    handles = [
        Line2D([], [], color=L_COLOR[l], marker=L_MARK[l], markersize=6, lw=1.6,
               markeredgecolor="black", markeredgewidth=0.6, label=l)
        for l in LEARNERS
    ]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               fontsize=8, framealpha=0.9)
    title2(fig, "Metric robustness \u2014 is the winner metric-dependent?",
           f"encoder: {encoder}   classifier: {CLF_SHORT[clf]}   "
           "each panel carries its own y-scale")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    save(fig, f"s3_metric_robustness_{encoder}_{CLF_SHORT[clf]}.png", tight=False)


# ---------------------------------------------------------------------------
# S4 -- The combination matrix
# ---------------------------------------------------------------------------
def s4(df, clf="RandomForest", metric="macro_f1"):
    best = best_per_combo(df, clf=clf, by=metric)
    grid = np.full((len(LEARNERS), len(ENCODERS)), np.nan)
    notes = {}
    for _, r in best.iterrows():
        i, j = LEARNERS.index(r["dict_learner"]), ENCODERS.index(r["encoder"])
        grid[i, j] = r[metric]
        notes[(i, j)] = (int(r["atoms"]), r["source"])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    im = ax.imshow(grid, cmap="Blues", vmin=lo - 0.012, vmax=hi + 0.005, aspect="auto")

    for i in range(len(LEARNERS)):
        for j in range(len(ENCODERS)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "no run", ha="center", va="center",
                        color=GREY, fontsize=9, style="italic")
                continue
            atoms, source = notes[(i, j)]
            fg = "white" if (grid[i, j] - lo) / max(hi - lo, 1e-9) > 0.55 else "black"
            ax.text(j, i - 0.10, f"{grid[i, j]:.4f}", ha="center", va="center",
                    color=fg, fontsize=13)
            ax.text(j, i + 0.20, f"{atoms} atoms" + ("" if source == "mccv" else "  (1 split)"),
                    ha="center", va="center", color=fg, fontsize=8, family="monospace")

    ax.set_xticks(range(len(ENCODERS)), ENCODERS)
    ax.set_yticks(range(len(LEARNERS)), LEARNERS)
    ax.set_xticks(np.arange(-0.5, len(ENCODERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LEARNERS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(METRIC_LABEL[metric])

    title2(ax, f"Combination matrix \u2014 best {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   "
           "each cell is that combination's best dictionary size, named in the cell")
    save(fig, f"s4_combination_matrix_{CLF_SHORT[clf]}_{metric}.png")


# ---------------------------------------------------------------------------
# S5 -- Is the representation good, or is Random Forest carrying it?
# ---------------------------------------------------------------------------
def s5(df, metric="macro_f1"):
    mean_over_clf = df.groupby(["encoder", "dict_learner", "atoms"])[metric].mean().reset_index()
    picks = mean_over_clf.loc[
        mean_over_clf.groupby(["encoder", "dict_learner"])[metric].idxmax()]

    rows, labels = [], []
    for enc in ENCODERS:
        for learner in LEARNERS:
            p = picks[(picks["encoder"] == enc) & (picks["dict_learner"] == learner)]
            if p.empty:
                continue
            atoms = p.iloc[0]["atoms"]
            d = df[(df["encoder"] == enc) & (df["dict_learner"] == learner)
                   & (df["atoms"] == atoms)]
            rows.append([
                d[d["classifier"] == c][metric].mean() if not d[d["classifier"] == c].empty
                else np.nan for c in CLASSIFIERS
            ])
            src = d["source"].iloc[0]
            labels.append(f"{enc} / {learner} / {int(atoms)}"
                          + ("" if src == "mccv" else "  (1 split)"))

    grid = np.array(rows)
    fig, ax = plt.subplots(figsize=(10, 9))
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    im = ax.imshow(grid, cmap="Blues", vmin=lo - 0.012, vmax=hi + 0.005, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            fg = "white" if (grid[i, j] - lo) / max(hi - lo, 1e-9) > 0.55 else "black"
            ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center",
                    color=fg, fontsize=9, family="monospace")

    ax.set_xticks(range(len(CLASSIFIERS)), [CLF_SHORT[c] for c in CLASSIFIERS])
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(CLASSIFIERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(METRIC_LABEL[metric])

    title2(ax, f"Classifier \u00d7 representation \u2014 {METRIC_LABEL[metric]}",
           "each row is that combination's best dictionary size, averaged over classifiers\n"
           "a wide LogReg\u2192RF gap means the codes are not linearly separable")
    save(fig, f"s5_classifier_interaction_{metric}.png")


# ---------------------------------------------------------------------------
# S6 -- What does the accuracy cost?
# ---------------------------------------------------------------------------
def s6(df, clf="RandomForest", metric="macro_f1"):
    allrf = df[df["classifier"] == clf]
    sub = allrf[allrf["fit_seconds"].notna()]
    best = best_per_combo(sub, clf=clf, by=metric)
    if best.empty:
        print("  skip s6: no timing data")
        return

    # Older runs predate summary_timings.csv, so the timed subset can hide a
    # combination's true best dictionary size. Name the affected ones.
    partial = []
    for (enc, learner), grp in allrf.groupby(["encoder", "dict_learner"]):
        timed = grp["fit_seconds"].notna().sum()
        if 0 < timed < len(grp):
            partial.append(f"{enc}/{learner} ({timed}/{len(grp)} sizes timed)")

    fig, ax = plt.subplots(figsize=(10, 6.4))
    pts = best.sort_values("fit_seconds")

    front, best_y = [], -np.inf
    for _, r in pts.iterrows():
        if r[metric] > best_y:
            best_y = r[metric]
            front.append((r["fit_seconds"], r[metric]))

    for _, r in pts.iterrows():
        ax.scatter(r["fit_seconds"], r[metric], s=70, color=E_COLOR[r["encoder"]],
                   marker=E_MARK[r["encoder"]], edgecolor="black", linewidth=0.6, zorder=5)

    ax.set_xscale("log")
    style(ax, "encoder + dictionary fit time (seconds, log scale)", METRIC_LABEL[metric])

    handles = [
        Line2D([], [], color=E_COLOR[e], marker=E_MARK[e], markersize=7, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.6, label=e)
        for e in ENCODERS if e in set(pts["encoder"])
    ]
    if len(front) >= 2:
        ax.step([f[0] for f in front], [f[1] for f in front], where="post",
                color=GREY, lw=1.2, ls=":", zorder=2)
        handles.append(Line2D([], [], color=GREY, ls=":", lw=1.2, label="Pareto frontier"))
    else:
        # One point is cheapest *and* best, so a staircase would render as
        # nothing. Ring it -- but the claim holds only over the timed subset.
        fx, fy = front[0]
        ax.scatter(fx, fy, s=340, facecolor="none", edgecolor=SHADE,
                   linewidth=1.3, linestyle="--", zorder=4)
        handles.append(Line2D([], [], marker="o", markersize=11, linestyle="none",
                              markerfacecolor="none", markeredgecolor=SHADE,
                              label="cheapest and best of the timed runs"))

    # Headroom above for the topmost label (which would otherwise hit the
    # title), and below for the legend (which would otherwise sit on a point).
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.16, hi + (hi - lo) * 0.13)

    repel(ax, [(r["fit_seconds"], r[metric], f"{r['dict_learner']}\n{int(r['atoms'])}", "black")
               for _, r in pts.iterrows()])
    ax.legend(loc="lower right", handles=handles, fontsize=8, framealpha=0.9)
    title2(ax, f"Cost vs performance \u2014 {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   one point per combination at its best "
           "dictionary size   up and to the left is better")
    if partial:
        footnote(ax, f"only runs with a recorded fit time are shown ({len(sub)} of "
                     f"{len(allrf)} {CLF_SHORT[clf]} rows); partly-timed combinations, "
                     f"whose best size may be missing here: {', '.join(partial)}.")
    save(fig, f"s6_cost_vs_performance_{CLF_SHORT[clf]}_{metric}.png")


# ---------------------------------------------------------------------------
# S7 -- Leaderboard, with the noise shown
# ---------------------------------------------------------------------------
def s7(df, metric="macro_f1", top=20):
    d = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ys = np.arange(len(d))
    labels = []
    for y, (_, r) in zip(ys, d.iterrows()):
        mccv = r["source"] == "mccv"
        c = TAB10[0] if mccv else TAB10[1]
        std = r.get(f"{metric}_std", np.nan)
        if mccv and pd.notna(std):
            ax.plot([r[metric] - std, r[metric] + std], [y, y], color=c, alpha=0.45,
                    lw=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(r[metric], y, s=70, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        labels.append(f"{r['encoder']} / {r['dict_learner']} / {int(r['atoms'])} / "
                      f"{CLF_SHORT[r['classifier']]}")

    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_ylim(-0.8, len(d) - 0.2)
    style(ax, METRIC_LABEL[metric], "")
    ax.grid(axis="y", visible=False)

    lead, runner = d.iloc[-1], d.iloc[-2]
    handles = [
        Line2D([], [], color=TAB10[0], marker="o", markersize=7, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.6,
               label="MC-CV mean   (bar = \u00b11 SD)"),
    ]
    # Only advertise the single-split colour when one actually made the cut.
    if (d["source"] == "artifact").any():
        handles.append(
            Line2D([], [], color=TAB10[1], marker="o", markersize=7, linestyle="none",
                   markeredgecolor="black", markeredgewidth=0.6,
                   label="single split   (no spread available)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    def pm(row):
        # A single-split row carries no spread; say so rather than printing nan.
        s = row.get(f"{metric}_std", np.nan)
        return f"{row[metric]:.4f} \u00b1 {s:.4f}" if pd.notna(s) else f"{row[metric]:.4f} (1 split)"

    summary_box(ax, [
        f"leader   : {pm(lead)}",
        f"runner-up: {pm(runner)}",
        f"gap      : {lead[metric] - runner[metric]:.4f}",
        f"rank-{top:<4}: {d.iloc[0][metric]:.4f}",
        f"spread   : {lead[metric] - d.iloc[0][metric]:.4f} over {top} rows",
    ], x=0.015, y=0.985, ha="left")
    title2(ax, f"Leaderboard \u2014 top {top} configurations by {METRIC_LABEL[metric]}",
           "rows are encoder / dictionary learner / atoms / classifier\n"
           "overlapping bars mean the ranking is not resolved by the data")
    save(fig, f"s7_leaderboard_{metric}.png")


# ---------------------------------------------------------------------------
# S8 -- Where does each combination sit on the minority trade-off?
# ---------------------------------------------------------------------------
def s8(df, clf="RandomForest"):
    best = best_per_combo(df, clf=clf, by="minority_f1")
    fig, ax = plt.subplots(figsize=(10, 7))

    xs = best["minority_recall"].to_numpy(float)
    ys = best["minority_precision"].to_numpy(float)
    style(ax, "Minority-Recall", "Minority-Precision")
    zoom_to_data(ax, xs, ys)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    r_grid = np.linspace(max(x0, 1e-3), x1, 400)
    for f1 in np.arange(0.20, 0.61, 0.05):
        p = f1 * r_grid / np.maximum(2 * r_grid - f1, 1e-9)
        ok = (p > y0) & (p < y1) & (p > 0)
        if ok.sum() < 2:
            continue
        ax.plot(r_grid[ok], p[ok], color=GREY, ls=":", lw=1.0, alpha=0.6, zorder=1)
        ax.annotate(f"F1={f1:.2f}", (r_grid[ok][-1], p[ok][-1]), xytext=(-2, 4),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=GREY, zorder=1)

    for _, r in best.iterrows():
        ax.scatter(r["minority_recall"], r["minority_precision"], s=70,
                   color=E_COLOR[r["encoder"]], marker=E_MARK[r["encoder"]],
                   edgecolor="black", linewidth=0.6, zorder=5)
    repel(ax, [(r["minority_recall"], r["minority_precision"],
                f"{r['dict_learner']}\n{int(r['atoms'])}", "black")
               for _, r in best.iterrows()])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.legend(handles=[
        Line2D([], [], color=E_COLOR[e], marker=E_MARK[e], markersize=7, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.6, label=e)
        for e in ENCODERS
    ], loc="upper left", fontsize=8, framealpha=0.9)
    title2(ax, "Minority-class operating point",
           f"classifier: {CLF_SHORT[clf]}   one point per combination at its best "
           "Minority-F1   dotted curves are iso-F1")
    save(fig, "s8_minority_tradeoff.png")


# Headline metrics rendered as a matched pair. Both are reported side by side
# because they answer different questions on an imbalanced corpus: ROC-AUC
# scores the ranking irrespective of the decision threshold, Macro-F1 scores the
# thresholded decision and so is sensitive to the calibration in thresholds.json.
HEADLINE_METRICS = ["macro_f1", "roc_auc"]


def main():
    df = load()
    print(f"loaded {len(df)} rows from {CSV.relative_to(ROOT)}")

    # S3 already carries both metrics as panels; S8 lives in the
    # precision/recall plane, where a ranking metric has no axis to occupy.
    s3(df)
    s8(df)

    for metric in HEADLINE_METRICS:
        print(f"  -- {METRIC_LABEL[metric]} --")
        s1(df, metric=metric)
        s2(df, metric=metric)
        s4(df, metric=metric)
        s5(df, metric=metric)
        s6(df, metric=metric)
        s7(df, metric=metric)


if __name__ == "__main__":
    main()
