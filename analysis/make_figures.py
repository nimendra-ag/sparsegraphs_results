"""Standard evaluation scenarios for the encoder x dict-learner study.

Reads analysis/all_results.csv (built by build_all_results.py) and writes every
figure each scenario admits into its own subdirectory of analysis/figures/.

Every scenario fixes some factors and varies exactly one, so each figure answers
a single question. The fixed factors are named on the figure's second title line.
A scenario is enumerated over the full cross-product of the factors it holds
fixed, so S1 (fixed encoder x classifier x metric) yields one figure per
encoder x classifier x metric rather than a single example.

  scenario  question it answers                          fixed              varies
  S1        which learner wins, does more atoms help?    enc, clf, metric   learner x atoms
  S2        which encoder wins?                          clf, metric        encoder x atoms, faceted by learner
  S3        is the winner metric-dependent?              enc, clf           learner x atoms x metric
  S4        the combination matrix                       clf, metric        encoder x learner
  S5        is the representation good, or the clf?      metric             combination x classifier
  S6        what does the accuracy cost?                 clf, metric        combination, against fit time
  S7        what is actually best, is the lead real?     metric             everything, ranked
  S8        precision/recall operating point             clf                combination

Only three metrics are reported: ROC-AUC scores the ranking irrespective of the
decision threshold, Macro-F1 scores the thresholded decision on both classes,
and Minority-F1 isolates the class that actually matters on this corpus.

Styling follows utils/elbow.py, the project's existing figure house style:
default matplotlib chrome (white surface, full spine box, DejaVu Sans), the
tab10 palette, framed legends at fontsize 8, ``grid(alpha=0.25)``, monospace
summary boxes in a rounded #cccccc frame, and 150 dpi output.

Run with the project env:
  "C:/Users/Puldith CE/miniconda3/envs/genv12/python.exe" analysis/make_figures.py

Optional filters, to re-render a slice without rebuilding all of it:
  ... analysis/make_figures.py --only s1 s4 --metrics roc_auc
"""

from __future__ import annotations

import argparse
import textwrap
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
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# Preferred display order. Anything the CSV grows that is not named here is
# appended in sorted order rather than dropped, so a new learner or encoder
# shows up in the figures without editing this file.
LEARNER_ORDER = ["fddl", "csfddl", "aksvd", "lcksvd", "online_dl",
                 "frozen_ksvd", "bayesian"]
ENCODER_ORDER = ["wl", "wl_edge", "fsm", "gspan_cork"]
CLASSIFIER_ORDER = ["LogisticRegression", "LinearSVM", "GradientBoosting",
                    "RandomForest"]

CLF_SHORT = {
    "LogisticRegression": "LogReg",
    "GradientBoosting": "GBoost",
    "LinearSVM": "LinSVM",
    "RandomForest": "RF",
    "GCN": "GCN",
}

# --- reference methods ------------------------------------------------------
# The baselines carry no encoder and no dictionary, so they cannot occupy a
# point on the encoder x learner axes. They are drawn instead as reference
# levels: one dark dashed rule per method, held to a single neutral colour so
# they read as a group of thresholds rather than as three more competitors.
BASELINES = ["sf", "graph2vec", "gcn"]
B_COLOR = "#3f3f3f"
B_DASH = {
    "sf": (0, (7, 3)),
    "graph2vec": (0, (4, 1.6, 1, 1.6)),
    "gcn": (0, (1.6, 2.2)),
}
B_MARK = {"sf": "*", "graph2vec": "h", "gcn": "d"}

METRIC_LABEL = {
    "roc_auc": "ROC-AUC",
    "macro_f1": "Macro-F1",
    "minority_f1": "Minority-F1",
    "mcc": "MCC",
    "minority_pr_auc": "Minority-PR-AUC",
    "minority_recall": "Minority-Recall",
    "minority_precision": "Minority-Precision",
    "accuracy": "Accuracy",
    "macro_pr_auc": "Macro-PR-AUC",
}

# The three metrics this study reports, in the order they appear as panels.
METRICS = ["roc_auc", "macro_f1", "minority_f1"]

# Filled in by ``configure`` once the CSV has been read.
LEARNERS: list[str] = []
ENCODERS: list[str] = []
CLASSIFIERS: list[str] = []
L_COLOR: dict[str, str] = {}
L_MARK: dict[str, str] = {}
E_COLOR: dict[str, str] = {}
E_MARK: dict[str, str] = {}

# Only the knobs elbow.py actually sets; everything else stays at the
# matplotlib default so these figures sit beside the elbow suite unchanged.
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI})


def ordered(values, preferred):
    """Preferred order first, then any newcomer, sorted. Never drops a level."""
    present = set(values)
    head = [v for v in preferred if v in present]
    return head + sorted(present - set(head))


def configure(df):
    """Derive the factor levels and their colour/marker assignment from the data."""
    global LEARNERS, ENCODERS, CLASSIFIERS
    pipe = df[df["family"] == "pipeline"]
    LEARNERS = ordered(pipe["dict_learner"].dropna(), LEARNER_ORDER)
    ENCODERS = ordered(pipe["encoder"].dropna(), ENCODER_ORDER)
    CLASSIFIERS = ordered(pipe["classifier"].dropna(), CLASSIFIER_ORDER)
    L_COLOR.update(zip(LEARNERS, TAB10))
    L_MARK.update(zip(LEARNERS, MARKERS))
    E_COLOR.update(zip(ENCODERS, TAB10))
    E_MARK.update(zip(ENCODERS, MARKERS))


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


def footnote(ax, text, width=118):
    """Wrapped note under the axes. Returns its line count so the caller can
    reserve the matching bottom margin -- tight_layout does not see text placed
    in axes coordinates below the axes, and an unwrapped note runs off the
    canvas once enough combinations have to be named."""
    wrapped = textwrap.fill(text, width)
    ax.text(0.5, -0.125, wrapped, transform=ax.transAxes, fontsize=7.5,
            color=NOTE_GREY, ha="center", va="top")
    return wrapped.count("\n") + 1


def log2_axis(ax, atoms, fontsize=None, min_sep=0.55):
    vals = sorted({int(a) for a in atoms})
    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    # The sizes are not all powers of two -- wl_edge ends at 20000, which sits
    # only 0.29 of an octave from 16384 and would print on top of it. Keep the
    # tick, drop the label, whenever a neighbour is closer than ``min_sep``
    # octaves.
    labels, prev = [], None
    for v in vals:
        lg = np.log2(v)
        labels.append("" if prev is not None and lg - prev < min_sep else str(v))
        if labels[-1]:
            prev = lg
    # Five-digit labels need the tilt; the plain power-of-two runs do not.
    wide = max(vals) >= 8192
    ax.set_xticklabels(labels, rotation=45 if wide else 0,
                       ha="right" if wide else "center",
                       **({"fontsize": fontsize} if fontsize else {}))
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


def save(fig, subdir, name, tight=True):
    """Write the figure into its scenario's subdirectory.

    ``tight=False`` for figures that already ran ``tight_layout`` with a
    reserved margin -- a second unconstrained call would expand the axes back
    over whatever that margin was holding.
    """
    out = FIGDIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
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


def spread_col(df, metric):
    """The ``<metric>_std`` column if the CSV carries one, else None.

    Not every metric has a recorded spread -- ROC-AUC, Macro-F1 and Minority-F1
    do, the precision/recall pair does not.
    """
    col = f"{metric}_std"
    return col if col in df.columns else None


def best_per_combo(df, clf=None, by="macro_f1"):
    """One row per (encoder, dict_learner): the atoms that maximise `by`."""
    sub = df if clf is None else df[df["classifier"] == clf]
    sub = sub.dropna(subset=[by])
    if sub.empty:
        return sub
    return sub.loc[sub.groupby(["encoder", "dict_learner"])[by].idxmax()]


def source_note(source):
    """Short qualifier for a row whose number is not an MC-CV mean."""
    return {"mccv": "", "kfold": "  (k-fold)", "artifact": "  (1 split)"}.get(source, "")


def baseline_series(df, metric, clf=None):
    """Each reference method's full sweep, as (method, frame, own).

    ``sf`` and ``graph2vec`` feed the same four classifiers as the pipelines, so
    they are read at whichever classifier the figure fixed -- comparing a
    pipeline's RF score against a baseline's LogReg score would confound the two
    factors the study separates. ``gcn`` is an end-to-end network with no
    interchangeable classifier stage (``own=True``); its sweep stands in every
    panel.

    A baseline's ``atoms`` is the width of the representation it hands the
    classifier -- graph2vec's embedding dimension, gcn's hidden size -- which is
    the same quantity the pipelines sweep as dictionary atoms. That shared
    meaning is what lets the sweeps share an x-axis.
    """
    base = df[df["family"] == "baseline"].dropna(subset=[metric])
    out = []
    for m in BASELINES:
        d = base[base["method"] == m]
        if d.empty:
            continue
        own = set(d["classifier"]) == {"GCN"}
        if clf is not None and not own:
            d = d[d["classifier"] == clf]
            if d.empty:
                continue
        out.append((m, d.sort_values("atoms"), own))
    return out


def baseline_best_rows(df, metric, clf=None):
    """Each sweep reduced to its best row, as (method, row, own, n_widths)."""
    return [(m, d.loc[d[metric].idxmax()], own, d["atoms"].nunique())
            for m, d, own in baseline_series(df, metric, clf=clf)]


def baseline_levels(df, metric, clf=None):
    """The same picks as (method, value, atoms, own, n_widths) for ruling."""
    return [(m, float(r[metric]), int(r["atoms"]), own, n)
            for m, r, own, n in baseline_best_rows(df, metric, clf=clf)]


def baseline_label(m, v, atoms, own, inside=True, n=1):
    """Name a single reference number, and say how it was picked.

    A baseline swept over several widths is being reported at its argmax, which
    is only a fair comparison because the pipeline it sits beside is reported
    the same way. Saying "best of 7 widths" rather than a bare "@2048" keeps
    that selection visible instead of passing the number off as the method's
    one true score.
    """
    if own and n <= 1:
        tag = "own net"
    elif n > 1:
        tag = f"best of {n} widths, @{atoms}"
    else:
        tag = f"fixed {atoms}-dim"
    return f"{m} {v:.4f} ({tag})" + ("" if inside else " — off scale")


def draw_baseline_sweeps(ax, series, metric, keep_ylim=True, ylim=None):
    """Draw each reference across the width axis the pipelines already occupy.

    A method with one width (sf, at a fixed 26 features) has nothing to sweep
    and stays a rule. A method with several is drawn as its own curve, so the
    comparison is like for like: the reader sees both sweeps, not one sweep
    against the other's ceiling. That distinction matters here -- graph2vec's
    Minority-F1 moves 0.24 -> 0.37 across its dimensions under LogReg, further
    than the gap between most of the dictionary learners being compared.
    """
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    handles = []
    for m, d, own in series:
        vals = d[metric].to_numpy(float)
        swept = len(d) > 1
        if swept:
            ax.plot(d["atoms"], vals, color=B_COLOR, linestyle=B_DASH[m], lw=1.3,
                    alpha=0.85, marker=B_MARK[m], markersize=4.5,
                    markerfacecolor="white", markeredgecolor=B_COLOR,
                    markeredgewidth=0.9, zorder=2)
            inside = vals.max() >= lo and vals.min() <= hi
            label = (f"{m} {vals.min():.4f}–{vals.max():.4f} over {len(d)} widths"
                     + ("" if inside else " — off scale"))
        else:
            v = float(vals[0])
            inside = lo <= v <= hi
            if inside:
                ax.axhline(v, color=B_COLOR, linestyle=B_DASH[m], lw=1.3,
                           alpha=0.75, zorder=2)
            label = baseline_label(m, v, int(d["atoms"].iloc[0]), own, inside, n=1)
        handles.append(Line2D(
            [], [], color=B_COLOR, linestyle=B_DASH[m], lw=1.3,
            marker=B_MARK[m] if swept else None, markersize=4.5,
            markerfacecolor="white", markeredgecolor=B_COLOR, markeredgewidth=0.9,
            label=label))
    if keep_ylim:
        ax.set_ylim(lo, hi)
    return handles


def draw_baselines(ax, levels, keep_ylim=True, ylim=None):
    """Rule each reference level across the panel; return its legend handles.

    The y-range is restored afterwards so a distant baseline cannot flatten the
    curves the figure is about -- Minority-F1 for gcn is 0.0000, which would
    otherwise squash every pipeline into the top third of the panel. A level
    that falls outside the restored range is not drawn, but still earns a
    legend entry carrying its value and an "off scale" note, so it is reported
    rather than silently dropped.
    """
    # ``ylim`` is passed explicitly by the faceted scenarios: their panels share
    # a y-axis, so once one panel's rule has widened the range, asking a later
    # panel for its own limits would report the widened one.
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    handles = []
    for m, v, atoms, own, n in levels:
        inside = lo <= v <= hi
        if inside:
            ax.axhline(v, color=B_COLOR, linestyle=B_DASH[m], lw=1.3,
                       alpha=0.75, zorder=2)
        handles.append(Line2D([], [], color=B_COLOR, linestyle=B_DASH[m], lw=1.3,
                              label=baseline_label(m, v, atoms, own, inside, n)))
    if keep_ylim:
        ax.set_ylim(lo, hi)
    return handles


# ---------------------------------------------------------------------------
# S1 -- Dictionary-size sweep: which learner wins?
#       fixed: encoder, classifier, metric     varies: learner x atoms
# ---------------------------------------------------------------------------
def s1(df, encoder, clf, metric, base=None):
    sub = df[(df["encoder"] == encoder) & (df["classifier"] == clf)].dropna(subset=[metric])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    ends, rows = [], []
    n_single = 0
    for learner in LEARNERS:
        d = sub[sub["dict_learner"] == learner].sort_values("atoms")
        if d.empty:
            continue
        c = L_COLOR[learner]
        # The trend across sizes is the question here, so the line stays whole
        # even where the protocol changes mid-series (wl_edge/fddl). The fill of
        # each marker says how that point was measured: solid = MC-CV mean,
        # hollow = one split.
        one = d["source"] == "artifact"
        ax.plot(d["atoms"], d[metric], color=c, lw=1.6, label=learner, zorder=3)
        if (~one).any():
            ax.plot(d["atoms"][~one], d[metric][~one], color=c, linestyle="none",
                    marker=L_MARK[learner], markersize=5, markerfacecolor=c,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        if one.any():
            ax.plot(d["atoms"][one], d[metric][one], color=c, linestyle="none",
                    marker=L_MARK[learner], markersize=5, markerfacecolor="white",
                    markeredgecolor=c, markeredgewidth=1.3, zorder=4)
        n_single += int(one.sum())
        peak = d.loc[d[metric].idxmax()]
        # Anchor the end label to that curve's own last point: learners do not
        # all reach the same dictionary size, so a shared x would strand labels
        # far from the line they name.
        ends.append((float(d.iloc[-1]["atoms"]), float(d.iloc[-1][metric]), learner, c))
        rows.append(f"{learner:<12}{peak[metric]:.4f} @ {int(peak['atoms']):>5}")

    if not rows:
        plt.close(fig)
        return

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
    placed = sorted(ends, key=lambda e: e[1])
    for i in range(1, len(placed)):
        if placed[i][1] - placed[i - 1][1] < gap:
            placed[i] = (placed[i][0], placed[i - 1][1] + gap, placed[i][2], placed[i][3])
    for x, y, text, c in placed:
        ax.annotate(text, (x, y), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=8, color=c)

    # Two legends: the learners the figure varies, and the reference levels it
    # is measured against. Keeping them apart stops a baseline from reading as
    # an eighth dictionary learner.
    ax.add_artist(ax.legend(loc="lower left", fontsize=8, framealpha=0.9))
    series = baseline_series(base if base is not None else df, metric, clf=clf)
    if series:
        handles = draw_baseline_sweeps(ax, series, metric)
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
                  title="baselines (swept ones share this x-axis)", title_fontsize=8)
    summary_box(ax, [f"best {METRIC_LABEL[metric]} per learner", "-" * 30] + rows,
                x=0.015, y=0.985, ha="left")
    # Say what the figure actually shows. The seed spread is no longer drawn --
    # seven overlapping bands buried the curves they were meant to qualify --
    # so the line is a mean with the uncertainty left off. S7 carries the SD
    # for the configurations where the ranking actually turns on it.
    if n_single == len(sub):
        provenance = "every point is a single split \u2014 no seed averaging"
    elif n_single:
        provenance = (f"points are MC-CV means, spread not shown;   hollow markers are "
                      f"the {n_single} of {len(sub)} points measured on one split")
    else:
        provenance = "points are MC-CV means across seeds; spread not shown"
    # Provenance gets its own line -- inlined, it runs off the canvas on the
    # encoders that mix protocols.
    title2(ax, f"Dictionary size sweep \u2014 {METRIC_LABEL[metric]}",
           f"encoder: {encoder}   classifier: {CLF_SHORT[clf]}\n{provenance}")
    save(fig, "s1_learner_sweep", f"s1_{encoder}_{CLF_SHORT[clf]}_{metric}.png")


# ---------------------------------------------------------------------------
# S2 -- Encoder comparison, one panel per learner
#       fixed: classifier, metric              varies: encoder x atoms
# ---------------------------------------------------------------------------
def s2(df, clf, metric, base=None):
    sub = df[df["classifier"] == clf].dropna(subset=[metric])
    if sub.empty:
        return
    learners = [l for l in LEARNERS if not sub[sub["dict_learner"] == l].empty]
    ncol = 4 if len(learners) > 6 else 3
    nrow = int(np.ceil(len(learners) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), sharey=True,
                             squeeze=False)
    flat = axes.ravel()

    any_single = False
    for ax, learner in zip(flat, learners):
        d0 = sub[sub["dict_learner"] == learner]
        for enc in ENCODERS:
            d = d0[d0["encoder"] == enc].sort_values("atoms")
            if d.empty:
                continue
            c = E_COLOR[enc]
            # wl_edge/fddl mixes four MC-CV sizes with sixteen single-split
            # ones, so fill is decided per point, not per series: a line may
            # only join points measured the same way. Single-split points are
            # hollow and never connected -- they carry no spread and are not
            # comparable to a seed-averaged neighbour.
            mc = d[d["source"] != "artifact"]
            one = d[d["source"] == "artifact"]
            any_single |= not one.empty
            if not mc.empty:
                ax.plot(mc["atoms"], mc[metric], color=c, lw=1.6,
                        linestyle="-" if len(mc) > 1 else "none",
                        marker=E_MARK[enc], markersize=6, markerfacecolor=c,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=3)
            if not one.empty:
                ax.plot(one["atoms"], one[metric], color=c, lw=0, linestyle="none",
                        marker=E_MARK[enc], markersize=6, markerfacecolor="white",
                        markeredgecolor=c, markeredgewidth=1.4, zorder=3)
        ax.set_title(learner, fontsize=10)
        style(ax, "", "")
        log2_axis(ax, sub["atoms"], fontsize=8)

    # The panels share a y-axis, so the references are measured once against the
    # common range and then drawn into every panel.
    series = baseline_series(base if base is not None else df, metric, clf=clf)
    base_handles = []
    if series:
        ylim = flat[0].get_ylim()
        for ax in flat[:len(learners)]:
            base_handles = draw_baseline_sweeps(ax, series, metric,
                                                keep_ylim=False, ylim=ylim)
        flat[0].set_ylim(*ylim)

    for ax in flat[len(learners):]:
        ax.set_visible(False)
    for ax in flat[max(0, len(learners) - ncol):len(learners)]:
        ax.set_xlabel("dictionary atoms (log\u2082)")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL[metric])

    # Colour and shape name the encoder; fill names the evaluation protocol.
    # Keeping the two channels in separate legend entries is what lets a mixed
    # series such as wl_edge/fddl be read correctly.
    handles = [
        Line2D([], [], color=E_COLOR[e], marker=E_MARK[e], markersize=6, lw=1.6,
               markerfacecolor=E_COLOR[e], markeredgecolor="black",
               markeredgewidth=0.6, label=e)
        for e in ENCODERS if not sub[sub["encoder"] == e].empty
    ]
    if any_single:
        handles.append(
            Line2D([], [], color=GREY, marker="o", markersize=6, linestyle="none",
                   markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.4,
                   label="hollow = single split (not connected)"))
    if base_handles:
        # Encoders left, reference levels right, so the dashed rules are never
        # mistaken for a fifth encoder.
        fig.legend(handles=handles, ncol=len(handles), loc="lower left",
                   bbox_to_anchor=(0.012, 0.002), fontsize=8, framealpha=0.9)
        fig.legend(handles=base_handles, ncol=len(base_handles), loc="lower right",
                   bbox_to_anchor=(0.988, 0.002), fontsize=8, framealpha=0.9,
                   title="baselines", title_fontsize=8)
    else:
        fig.legend(handles=handles, ncol=len(handles), loc="lower center",
                   bbox_to_anchor=(0.5, 0.002), fontsize=8, framealpha=0.9)
    title2(fig, f"Encoder comparison \u2014 {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   one panel per dictionary learner   "
           "hollow markers are single-split artifacts, not MC-CV means")
    fig.tight_layout(rect=[0, 0.075, 1, 0.93])
    save(fig, "s2_encoder_comparison", f"s2_{CLF_SHORT[clf]}_{metric}.png", tight=False)


# ---------------------------------------------------------------------------
# S3 -- Does the ranking survive a change of metric?
#       fixed: encoder, classifier             varies: learner x atoms x metric
# ---------------------------------------------------------------------------
def s3(df, encoder, clf, base=None):
    sub = df[(df["encoder"] == encoder) & (df["classifier"] == clf)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.0 * len(METRICS), 5.2),
                             squeeze=False)
    flat = axes.ravel()

    # Rank each learner under each metric, so the caption can state outright
    # whether the winner changed rather than leaving it to the reader's eye.
    winners, base_handles = {}, {}
    for ax, metric in zip(flat, METRICS):
        for learner in LEARNERS:
            d = sub[sub["dict_learner"] == learner].dropna(subset=[metric]).sort_values("atoms")
            if d.empty:
                continue
            ax.plot(d["atoms"], d[metric], color=L_COLOR[learner], lw=1.6,
                    marker=L_MARK[learner], markersize=4.5, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=3)
            winners.setdefault(metric, []).append((d[metric].max(), learner))
        # Each panel is a different metric on its own y-scale, so the references
        # are re-read per panel rather than shared.
        series = baseline_series(base if base is not None else df, metric, clf=clf)
        if series:
            base_handles[metric] = draw_baseline_sweeps(ax, series, metric)
            # The values differ panel to panel, so the reference legend has to
            # live in the panel rather than at figure level. These panels hold
            # nothing but curves -- no summary box for "best" to miss -- so
            # automatic placement beats any fixed corner across 16 figures whose
            # curves rise in some and fall in others.
            ax.legend(handles=base_handles[metric], loc="best", fontsize=7,
                      framealpha=0.9, title="baselines", title_fontsize=7)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        style(ax, "dictionary atoms (log\u2082)", METRIC_LABEL[metric])
        log2_axis(ax, sub["atoms"], fontsize=8)

    tops = {m: max(v)[1] for m, v in winners.items() if v}
    if tops:
        verdict = (f"same winner under all {len(tops)} metrics: {next(iter(tops.values()))}"
                   if len(set(tops.values())) == 1
                   else "winner changes with the metric: "
                        + ", ".join(f"{METRIC_LABEL[m]}\u2192{w}" for m, w in tops.items()))
    else:
        verdict = ""

    handles = [
        Line2D([], [], color=L_COLOR[l], marker=L_MARK[l], markersize=6, lw=1.6,
               markeredgecolor="black", markeredgewidth=0.6, label=l)
        for l in LEARNERS if not sub[sub["dict_learner"] == l].empty
    ]
    fig.legend(handles=handles, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 0.002), fontsize=8, framealpha=0.9)
    title2(fig, "Metric robustness \u2014 is the winner metric-dependent?",
           f"encoder: {encoder}   classifier: {CLF_SHORT[clf]}   "
           f"each panel carries its own y-scale\n{verdict}")
    # tight_layout fits the whole axes box -- panel titles included -- inside
    # the rect, so the top must sit high or a band of dead space opens up under
    # the three-line suptitle.
    fig.tight_layout(rect=[0, 0.085, 1, 0.94])
    save(fig, "s3_metric_robustness", f"s3_{encoder}_{CLF_SHORT[clf]}.png", tight=False)


# ---------------------------------------------------------------------------
# S4 -- The combination matrix
#       fixed: classifier, metric              varies: encoder x learner
# ---------------------------------------------------------------------------
def s4(df, clf, metric, base=None):
    best = best_per_combo(df, clf=clf, by=metric)
    if best.empty:
        return
    grid = np.full((len(LEARNERS), len(ENCODERS)), np.nan)
    notes = {}
    for _, r in best.iterrows():
        i, j = LEARNERS.index(r["dict_learner"]), ENCODERS.index(r["encoder"])
        grid[i, j] = r[metric]
        notes[(i, j)] = (int(r["atoms"]), r["source"])

    levels = baseline_levels(base if base is not None else df, metric, clf=clf)

    # A cell has no room for a rule, so the baselines get their own strip under
    # the matrix on the same colour scale: a reference darker than the cell
    # above it is a reference the pipeline did not beat.
    fig = plt.figure(figsize=(2.6 * len(ENCODERS) + 2,
                              1.1 * len(LEARNERS) + 2.5 + (1.5 if levels else 0)))
    if levels:
        gs = fig.add_gridspec(2, 1, height_ratios=[len(LEARNERS), 1.0], hspace=0.20)
        ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    else:
        ax, axb = fig.add_subplot(1, 1, 1), None

    # One scale across both panels, but set from the matrix alone. Letting a
    # distant reference into the norm is the colour-space version of letting it
    # into the y-range: gcn at 0.5527 would drag vmin down far enough to wash
    # every pipeline cell into the same blue. A baseline outside the range is
    # clamped and says so in its cell.
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    im = ax.imshow(grid, cmap="Blues", vmin=lo - 0.012, vmax=hi + 0.005, aspect="auto")

    for i in range(len(LEARNERS)):
        for j in range(len(ENCODERS)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "no run", ha="center", va="center",
                        color=GREY, fontsize=9, style="italic")
                continue
            atoms, source = notes[(i, j)]
            fg = "white" if (grid[i, j] - lo) / max(hi - lo, 1e-9) > 0.55 else "black"
            ax.text(j, i - 0.14, f"{grid[i, j]:.4f}", ha="center", va="center",
                    color=fg, fontsize=13)
            ax.text(j, i + 0.22, f"{atoms} atoms{source_note(source)}",
                    ha="center", va="center", color=fg, fontsize=8, family="monospace")

    ax.set_xticks(range(len(ENCODERS)), ENCODERS)
    ax.set_yticks(range(len(LEARNERS)), LEARNERS)
    ax.set_xticks(np.arange(-0.5, len(ENCODERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LEARNERS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    if levels:
        strip = np.array([[v for _, v, _, _, _ in levels]])
        axb.imshow(strip, cmap="Blues", vmin=lo - 0.012, vmax=hi + 0.005, aspect="auto")
        for j, (m, v, atoms, own, n) in enumerate(levels):
            fg = "white" if (v - lo) / max(hi - lo, 1e-9) > 0.55 else "black"
            # Matches how the matrix cells above are chosen -- best over the
            # widths that method swept -- so the strip is a like-for-like row.
            note = f"@{atoms} of {n} widths" if n > 1 else (
                "own net" if own else f"{atoms} atoms, fixed")
            if v < lo:
                note += "  (below scale)"
            elif v > hi:
                note += "  (above scale)"
            axb.text(j, -0.14, f"{v:.4f}", ha="center", va="center", color=fg, fontsize=13)
            axb.text(j, 0.22, note, ha="center", va="center", color=fg,
                     fontsize=8, family="monospace")
        axb.set_xticks(range(len(levels)), [m for m, _, _, _, _ in levels])
        axb.set_yticks([0], ["baselines"])
        axb.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
        axb.set_yticks([-0.5, 0.5], minor=True)
        axb.grid(which="minor", color="white", linewidth=2.5)
        axb.grid(which="major", visible=False)
        axb.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=[ax, axb] if levels else ax, fraction=0.045, pad=0.03)
    cb.set_label(METRIC_LABEL[metric])

    title2(ax, f"Combination matrix \u2014 best {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   "
           "each cell is that combination's best dictionary size, named in the cell"
           + ("\nthe strip beneath carries the baselines on the matrix's colour scale"
              if levels else ""))
    save(fig, "s4_combination_matrix", f"s4_{CLF_SHORT[clf]}_{metric}.png",
         tight=not levels)


# ---------------------------------------------------------------------------
# S5 -- Is the representation good, or is the classifier carrying it?
#       fixed: metric                          varies: combination x classifier
# ---------------------------------------------------------------------------
def s5(df, metric, base=None):
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return
    # Pick each combination's dictionary size on the classifier-averaged score,
    # so the row is not chosen by the same classifier it is meant to compare.
    mean_over_clf = sub.groupby(["encoder", "dict_learner", "atoms"])[metric].mean().reset_index()
    picks = mean_over_clf.loc[
        mean_over_clf.groupby(["encoder", "dict_learner"])[metric].idxmax()]

    rows, labels = [], []
    for enc in ENCODERS:
        for learner in LEARNERS:
            p = picks[(picks["encoder"] == enc) & (picks["dict_learner"] == learner)]
            if p.empty:
                continue
            atoms = p.iloc[0]["atoms"]
            d = sub[(sub["encoder"] == enc) & (sub["dict_learner"] == learner)
                    & (sub["atoms"] == atoms)]
            rows.append([
                d[d["classifier"] == c][metric].mean() if not d[d["classifier"] == c].empty
                else np.nan for c in CLASSIFIERS
            ])
            labels.append(f"{enc} / {learner} / {int(atoms)}"
                          + source_note(d["source"].iloc[0]))

    if not rows:
        return
    n_pipe = len(rows)

    # This is the one scenario the baselines slot straight into: sf and
    # graph2vec drive the same four classifiers, so they are just more rows.
    # gcn is its own network, which earns a fifth column that only it fills.
    cols = list(CLASSIFIERS)
    bsub = (base if base is not None else df)
    bsub = bsub[bsub["family"] == "baseline"].dropna(subset=[metric])
    if not bsub[bsub["classifier"] == "GCN"].empty:
        cols.append("GCN")
    for m in BASELINES:
        d = bsub[bsub["method"] == m]
        if d.empty:
            continue
        # Same rule as the pipeline rows: the size is chosen on the
        # classifier-averaged score, not by the best single classifier.
        per_atoms = d.groupby("atoms")[metric].mean()
        atoms = per_atoms.idxmax()
        d = d[d["atoms"] == atoms]
        rows.append([d[d["classifier"] == c][metric].mean()
                     if not d[d["classifier"] == c].empty else np.nan for c in cols])
        labels.append(f"{m} / {int(atoms)}" + source_note(d["source"].iloc[0]))
    # The pipeline rows predate the GCN column and must be padded to match.
    for i in range(n_pipe):
        rows[i] += [np.nan] * (len(cols) - len(rows[i]))

    grid = np.array(rows)
    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 4.5, 0.42 * len(labels) + 3.0))
    # Norm from the pipeline rows only, for the reason S4 does it: gcn scores
    # 0.0000 on Minority-F1, and letting that set vmin would spend most of the
    # ramp on a range no pipeline occupies. Baseline cells clamp; every cell
    # prints its exact value regardless.
    lo, hi = np.nanmin(grid[:n_pipe]), np.nanmax(grid[:n_pipe])
    im = ax.imshow(grid, cmap="Blues", vmin=lo - 0.012, vmax=hi + 0.005, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            fg = "white" if (grid[i, j] - lo) / max(hi - lo, 1e-9) > 0.55 else "black"
            ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center",
                    color=fg, fontsize=9, family="monospace")

    ax.set_xticks(range(len(cols)), [CLF_SHORT[c] for c in cols])
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    # A hard rule where the pipelines end and the references begin -- without
    # it the baselines read as three more combinations.
    if len(labels) > n_pipe:
        ax.axhline(n_pipe - 0.5, color="black", lw=2.0, zorder=6)
        for lab in ax.get_yticklabels()[n_pipe:]:
            lab.set_color(B_COLOR)
            lab.set_fontweight("bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(METRIC_LABEL[metric])

    title2(ax, f"Classifier \u00d7 representation \u2014 {METRIC_LABEL[metric]}",
           "each row is that combination at the dictionary size that maximises its "
           "classifier-averaged score\na wide LogReg\u2192RF gap means the codes are "
           "not linearly separable; baselines are ruled off at the foot")
    save(fig, "s5_classifier_interaction", f"s5_{metric}.png")


# ---------------------------------------------------------------------------
# S6 -- What does the accuracy cost?
#       fixed: classifier, metric              varies: combination vs fit time
# ---------------------------------------------------------------------------
def s6(df, clf, metric, base=None):
    allclf = df[df["classifier"] == clf].dropna(subset=[metric])
    sub = allclf[allclf["fit_seconds"].notna()]
    best = best_per_combo(sub, clf=clf, by=metric)
    if best.empty:
        print(f"  skip s6 {CLF_SHORT[clf]}/{metric}: no timing data")
        return

    # Older runs predate summary_timings.csv, so the timed subset can hide a
    # combination's true best dictionary size. Name the affected ones.
    partial = []
    for (enc, learner), grp in allclf.groupby(["encoder", "dict_learner"]):
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

    # No baseline records a fit time, so they cannot be points on this plane --
    # they enter as levels, answering "is the pipeline's cost buying anything"
    # without claiming to be on the cost axis.
    levels = baseline_levels(base if base is not None else df, metric, clf=clf)

    # Headroom above for the topmost label and the baseline legend -- the good
    # points sit up and to the left, exactly where that legend goes, so the band
    # has to be opened for it rather than dropped on top of them. Below, room
    # for the encoder legend.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.16, hi + (hi - lo) * (0.32 if levels else 0.13))

    base_handles = draw_baselines(ax, levels) if levels else []

    repel(ax, [(r["fit_seconds"], r[metric], f"{r['dict_learner']}\n{int(r['atoms'])}", "black")
               for _, r in pts.iterrows()])
    ax.add_artist(ax.legend(loc="lower right", handles=handles, fontsize=8, framealpha=0.9))
    if base_handles:
        ax.legend(handles=base_handles, loc="upper left", fontsize=8, framealpha=0.9,
                  title="baselines (no fit time recorded)", title_fontsize=8)
    title2(ax, f"Cost vs performance \u2014 {METRIC_LABEL[metric]}",
           f"classifier: {CLF_SHORT[clf]}   one point per combination at its best "
           "dictionary size   up and to the left is better")
    nlines = 0
    if partial:
        nlines = footnote(
            ax, f"only runs with a recorded fit time are shown ({len(sub)} of "
                f"{len(allclf)} {CLF_SHORT[clf]} rows); partly-timed combinations, "
                f"whose best size may be missing here: {', '.join(partial)}.")
    if nlines:
        # Reserve the wrapped note's height; tight_layout cannot measure text
        # anchored below the axes in axes coordinates.
        fig.tight_layout(rect=[0, 0.055 + 0.022 * nlines, 1, 1])
    save(fig, "s6_cost_vs_performance", f"s6_{CLF_SHORT[clf]}_{metric}.png",
         tight=not nlines)


# ---------------------------------------------------------------------------
# S7 -- Leaderboard, with the noise shown
#       fixed: metric                          varies: everything, ranked
# ---------------------------------------------------------------------------
def s7(df, metric, top=20, baselines=True):
    """Rank every configuration. ``baselines`` keeps the non-pipeline reference
    methods (sf, graph2vec, gcn) in the race, which is the point of S7 -- a
    dictionary pipeline that cannot beat graph2vec has not earned its cost."""
    pool = df if baselines else df[df["family"] == "pipeline"]
    d = pool.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top).iloc[::-1]
    if len(d) < 2:
        return
    std_col = spread_col(d, metric)

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 2.6))
    ys = np.arange(len(d))
    labels, kinds = [], set()
    for y, (_, r) in zip(ys, d.iterrows()):
        base = r["family"] == "baseline"
        c = TAB10[2] if base else (TAB10[0] if r["source"] == "mccv" else TAB10[1])
        kinds.add("baseline" if base else r["source"])
        std = r.get(std_col, np.nan) if std_col else np.nan
        if pd.notna(std):
            ax.plot([r[metric] - std, r[metric] + std], [y, y], color=c, alpha=0.45,
                    lw=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(r[metric], y, s=70, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        labels.append(
            f"{r['method']} / {int(r['atoms'])} / {CLF_SHORT.get(r['classifier'], r['classifier'])}"
            if base else
            f"{r['encoder']} / {r['dict_learner']} / {int(r['atoms'])} / "
            f"{CLF_SHORT[r['classifier']]}")

    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_ylim(-0.8, len(d) - 0.2)
    style(ax, METRIC_LABEL[metric], "")
    ax.grid(axis="y", visible=False)

    lead, runner = d.iloc[-1], d.iloc[-2]
    # Only advertise a colour when a row of that kind actually made the cut.
    legend_spec = [
        ("mccv", TAB10[0], "pipeline, MC-CV mean   (bar = \u00b11 SD)"),
        ("artifact", TAB10[1], "pipeline, single split   (no spread available)"),
        ("baseline", TAB10[2], "baseline method (sf / graph2vec / gcn)"),
    ]
    handles = [
        Line2D([], [], color=col, marker="o", markersize=7, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.6, label=lab)
        for key, col, lab in legend_spec if key in kinds
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    def pm(row):
        # A single-split row carries no spread; say so rather than printing nan.
        s = row.get(std_col, np.nan) if std_col else np.nan
        return f"{row[metric]:.4f} \u00b1 {s:.4f}" if pd.notna(s) else f"{row[metric]:.4f} (no SD)"

    overlap = ""
    if std_col and pd.notna(lead.get(std_col)) and pd.notna(runner.get(std_col)):
        overlap = ("yes" if lead[metric] - lead[std_col] < runner[metric] + runner[std_col]
                   else "no")
        overlap = f"1 SD overlap: {overlap}"

    summary_box(ax, [
        f"leader   : {pm(lead)}",
        f"runner-up: {pm(runner)}",
        f"gap      : {lead[metric] - runner[metric]:.4f}",
        f"rank-{top:<4}: {d.iloc[0][metric]:.4f}",
        f"spread   : {lead[metric] - d.iloc[0][metric]:.4f} over {top} rows",
    ] + ([overlap] if overlap else []), x=0.015, y=0.985, ha="left")
    scope = "all methods" if baselines else "pipelines only"
    title2(ax, f"Leaderboard \u2014 top {top} configurations by {METRIC_LABEL[metric]}",
           f"{scope}   rows are encoder / dictionary learner / atoms / classifier\n"
           "overlapping bars mean the ranking is not resolved by the data")
    save(fig, "s7_leaderboard",
         f"s7_{metric}_top{top}{'' if baselines else '_pipelines_only'}.png")


# ---------------------------------------------------------------------------
# S8 -- Where does each combination sit on the minority trade-off?
#       fixed: classifier                      varies: combination
# ---------------------------------------------------------------------------
def s8(df, clf, base=None):
    best = best_per_combo(df, clf=clf, by="minority_f1")
    best = best.dropna(subset=["minority_precision", "minority_recall"])
    if len(best) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 7))

    # This plane has axes the baselines can actually occupy, so they are drawn
    # as points rather than rules. A method that never predicts the minority
    # class sits at the origin -- a real result, but one that would blow the
    # window open, so it is kept out of the extent and reported in the legend.
    bpts = [(m, r, own, n) for m, r, own, n in
            baseline_best_rows(base if base is not None else df, "minority_f1", clf=clf)
            if pd.notna(r["minority_precision"]) and pd.notna(r["minority_recall"])]
    scaled = [(m, r, own, n) for m, r, own, n in bpts
              if r["minority_recall"] > 0 and r["minority_precision"] > 0]

    xs = np.concatenate([best["minority_recall"].to_numpy(float),
                         [r["minority_recall"] for _, r, _, _ in scaled]])
    ys = np.concatenate([best["minority_precision"].to_numpy(float),
                         [r["minority_precision"] for _, r, _, _ in scaled]])
    style(ax, "Minority-Recall", "Minority-Precision")
    zoom_to_data(ax, xs, ys)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Iso-F1 contours, drawn only where they cross the visible window so the
    # panel is not covered in curves that say nothing about these points.
    r_grid = np.linspace(max(x0, 1e-3), x1, 400)
    f1s = list(best["minority_f1"]) + [r["minority_f1"] for _, r, _, _ in scaled]
    f1_lo = max(0.05, np.floor(min(f1s) * 20) / 20 - 0.05)
    f1_hi = min(0.95, np.ceil(max(f1s) * 20) / 20 + 0.05)
    for f1 in np.arange(f1_lo, f1_hi + 1e-9, 0.05):
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
    labels = [(r["minority_recall"], r["minority_precision"],
               f"{r['dict_learner']}\n{int(r['atoms'])}", "black")
              for _, r in best.iterrows()]

    base_handles = []
    for m, r, own, n in bpts:
        inside = (x0 <= r["minority_recall"] <= x1
                  and y0 <= r["minority_precision"] <= y1)
        if inside:
            ax.scatter(r["minority_recall"], r["minority_precision"], s=130,
                       color=B_COLOR, marker=B_MARK[m], edgecolor="white",
                       linewidth=0.8, zorder=7)
            labels.append((r["minority_recall"], r["minority_precision"], m, B_COLOR))
        base_handles.append(Line2D(
            [], [], color=B_COLOR, marker=B_MARK[m], markersize=9, linestyle="none",
            markeredgecolor="white", markeredgewidth=0.8,
            label=baseline_label(m, r["minority_f1"], int(r["atoms"]), own, inside, n)))

    repel(ax, labels)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.add_artist(ax.legend(handles=[
        Line2D([], [], color=E_COLOR[e], marker=E_MARK[e], markersize=7, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.6, label=e)
        for e in ENCODERS if e in set(best["encoder"])
    ], loc="upper left", fontsize=8, framealpha=0.9))
    if base_handles:
        ax.legend(handles=base_handles, loc="lower left", fontsize=8, framealpha=0.9,
                  title="baselines (label is Minority-F1)", title_fontsize=8)
    title2(ax, "Minority-class operating point",
           f"classifier: {CLF_SHORT[clf]}   one point per combination at its best "
           "Minority-F1   dotted curves are iso-F1")
    save(fig, "s8_minority_tradeoff", f"s8_{CLF_SHORT[clf]}.png")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def run_all(df, only=None, metrics=None):
    metrics = metrics or METRICS
    want = (lambda s: True) if not only else (lambda s: s in set(only))
    # The baselines carry no encoder and no dictionary, so they cannot occupy a
    # cell or a curve on the encoder x learner axes. Every scenario therefore
    # takes the pipelines as its data and the full frame as ``base``, from which
    # it draws the reference levels in whatever form its axes allow: a ruled
    # line (S1-S3, S6), a strip on the same colour scale (S4), extra rows (S5),
    # or plain points (S8, which has axes they genuinely share).
    pipe = df[df["family"] == "pipeline"]

    if want("s1"):
        print("-- S1 dictionary size sweep --")
        for enc in ENCODERS:
            for clf in CLASSIFIERS:
                for m in metrics:
                    s1(pipe, enc, clf, m, base=df)

    if want("s2"):
        print("-- S2 encoder comparison --")
        for clf in CLASSIFIERS:
            for m in metrics:
                s2(pipe, clf, m, base=df)

    if want("s3"):
        print("-- S3 metric robustness --")
        for enc in ENCODERS:
            for clf in CLASSIFIERS:
                s3(pipe, enc, clf, base=df)

    if want("s4"):
        print("-- S4 combination matrix --")
        for clf in CLASSIFIERS:
            for m in metrics:
                s4(pipe, clf, m, base=df)

    if want("s5"):
        print("-- S5 classifier x representation --")
        for m in metrics:
            s5(pipe, m, base=df)

    if want("s6"):
        print("-- S6 cost vs performance --")
        for clf in CLASSIFIERS:
            for m in metrics:
                s6(pipe, clf, m, base=df)

    if want("s7"):
        print("-- S7 leaderboard --")
        for m in metrics:
            # Two cuts of the same ranking: with the reference methods in the
            # race, and restricted to the pipelines under study.
            s7(df, m, baselines=True)
            s7(df, m, baselines=False)

    if want("s8"):
        print("-- S8 minority trade-off --")
        for clf in CLASSIFIERS:
            s8(pipe, clf, base=df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", metavar="Sn",
                    help="render only these scenarios, e.g. --only s1 s4")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, metavar="METRIC",
                    help=f"restrict to a subset of {', '.join(METRICS)}")
    args = ap.parse_args()

    df = load()
    configure(df)
    print(f"loaded {len(df)} rows from {CSV.relative_to(ROOT)}")
    print(f"  encoders   : {', '.join(ENCODERS)}")
    print(f"  learners   : {', '.join(LEARNERS)}")
    print(f"  classifiers: {', '.join(CLASSIFIERS)}")
    print(f"  metrics    : {', '.join(METRIC_LABEL[m] for m in (args.metrics or METRICS))}")

    only = [s.lower() for s in args.only] if args.only else None
    run_all(df, only=only, metrics=args.metrics)


if __name__ == "__main__":
    main()
