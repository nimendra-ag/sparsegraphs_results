"""S8 (LinSVM) minority operating point, with a zoomed panel on the winners.

Left panel : the full S8 figure for one classifier -- one point per
             encoder x learner combination at its best Minority-F1, baselines
             as dark points, dotted iso-F1 curves.
Right panel: the dashed window from the left panel, enlarged. The region whose
             Minority-F1 exceeds graph2vec's best (above graph2vec's iso-F1
             curve) is shaded; combinations inside it are drawn at full
             strength and labelled with their Minority-F1 and the margin over
             graph2vec, the rest are faded.

"Outperforms graph2vec" means a higher Minority-F1 than graph2vec's best width
under the same classifier -- the same number S8's baseline legend reports.

Reuses the helpers and house style of make_figures.py, but reads the CSV that
sits beside this file (old_analysis/all_results.csv).

Run:
  python old_analysis/make_s8_zoom.py                 # LinearSVM
  python old_analysis/make_s8_zoom.py --clf LogisticRegression
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

import make_figures as mf

HERE = Path(__file__).resolve().parent
mf.CSV = HERE / "all_results.csv"
OUTDIR = HERE / "figures" / "s8_minority_tradeoff"

REF = "graph2vec"           # the baseline the zoom is measured against
ZOOM_EDGE = "#3f3f3f"       # dashed window, arrow and zoom-panel frame
WIN_SHADE = "#2ca02c"       # elbow.py's "selected/kept" green


def iso_f1(r, f1):
    """Precision that gives ``f1`` at recall ``r`` (nan where unreachable)."""
    den = 2 * r - f1
    return np.where(den > 0, f1 * r / np.where(den > 0, den, 1), np.nan)


def draw_iso(ax, f1s, x0, x1, y0, y1, fmt="{:.2f}"):
    r = np.linspace(max(x0, 1e-3), x1, 400)
    for f1 in f1s:
        p = iso_f1(r, f1)
        ok = (p > y0) & (p < y1)
        if ok.sum() < 2:
            continue
        ax.plot(r[ok], p[ok], color=mf.GREY, ls=":", lw=1.0, alpha=0.6, zorder=1)
        if fmt:
            ax.annotate("F1=" + fmt.format(f1), (r[ok][-1], p[ok][-1]), xytext=(-2, 4),
                        textcoords="offset points", ha="right", fontsize=8,
                        color=mf.GREY, zorder=1)


def point(ax, r, faded=False, s=70):
    ax.scatter(r["minority_recall"], r["minority_precision"], s=s,
               color=mf.E_COLOR[r["encoder"]], marker=mf.E_MARK[r["encoder"]],
               edgecolor="black", linewidth=0.6, zorder=5,
               alpha=0.3 if faded else 1.0)


def baseline_point(ax, m, r, s=130):
    ax.scatter(r["minority_recall"], r["minority_precision"], s=s,
               color=mf.B_COLOR, marker=mf.B_MARK[m], edgecolor="white",
               linewidth=0.8, zorder=7)


def encoder_legend(ax, encoders):
    return ax.legend(handles=[
        Line2D([], [], color=mf.E_COLOR[e], marker=mf.E_MARK[e], markersize=7,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6, label=e)
        for e in mf.ENCODERS if e in encoders
    ], loc="upper left", fontsize=8, framealpha=0.9)


def full_panel(ax, best, bpts):
    """The original S8 plot (make_figures.s8), drawn into ``ax``."""
    scaled = [(m, r, own, n) for m, r, own, n in bpts
              if r["minority_recall"] > 0 and r["minority_precision"] > 0]
    xs = np.concatenate([best["minority_recall"].to_numpy(float),
                         [r["minority_recall"] for _, r, _, _ in scaled]])
    ys = np.concatenate([best["minority_precision"].to_numpy(float),
                         [r["minority_precision"] for _, r, _, _ in scaled]])
    mf.style(ax, "Minority-Recall", "Minority-Precision")
    mf.zoom_to_data(ax, xs, ys)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    f1s = list(best["minority_f1"]) + [r["minority_f1"] for _, r, _, _ in scaled]
    f1_lo = max(0.05, np.floor(min(f1s) * 20) / 20 - 0.05)
    f1_hi = min(0.95, np.ceil(max(f1s) * 20) / 20 + 0.05)
    draw_iso(ax, np.arange(f1_lo, f1_hi + 1e-9, 0.05), x0, x1, y0, y1)

    for _, r in best.iterrows():
        point(ax, r)
    labels = [(r["minority_recall"], r["minority_precision"],
               f"{r['dict_learner']}\n{int(r['atoms'])}", "black")
              for _, r in best.iterrows()]

    base_handles = []
    for m, r, own, n in bpts:
        inside = (x0 <= r["minority_recall"] <= x1
                  and y0 <= r["minority_precision"] <= y1)
        if inside:
            baseline_point(ax, m, r)
            labels.append((r["minority_recall"], r["minority_precision"], m, mf.B_COLOR))
        base_handles.append(Line2D(
            [], [], color=mf.B_COLOR, marker=mf.B_MARK[m], markersize=9,
            linestyle="none", markeredgecolor="white", markeredgewidth=0.8,
            label=mf.baseline_label(m, r["minority_f1"], int(r["atoms"]), own, inside, n)))

    mf.repel(ax, labels)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.add_artist(encoder_legend(ax, set(best["encoder"])))
    ax.legend(handles=base_handles, loc="lower left", fontsize=8, framealpha=0.9,
              title="baselines (label is Minority-F1)", title_fontsize=8)
    ax.set_title("All combinations")


def zoom_window(win, ref):
    """Window around the winners and the reference point, with a little room."""
    xs = np.append(win["minority_recall"].to_numpy(float), ref["minority_recall"])
    ys = np.append(win["minority_precision"].to_numpy(float), ref["minority_precision"])
    return xs.min() - 0.022, xs.max() + 0.025, ys.min() - 0.015, ys.max() + 0.025


def zoom_panel(ax, best, ref, ref_f1, ref_atoms, ref_n, window):
    x0, x1, y0, y1 = window
    mf.style(ax, "Minority-Recall", "Minority-Precision")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Shade everything that beats graph2vec: above its iso-F1 curve.
    r = np.linspace(x0, x1, 400)
    p = np.clip(np.nan_to_num(iso_f1(r, ref_f1), nan=y1), y0, y1)
    ax.fill_between(r, p, y1, color=WIN_SHADE, alpha=0.09, lw=0, zorder=0)
    ax.plot(r, np.where(p < y1, p, np.nan), color=mf.B_COLOR,
            ls=mf.B_DASH[REF], lw=1.4, zorder=2)

    step = 0.025
    draw_iso(ax, np.arange(np.floor(y0 / step) * step, 0.6, step), x0, x1, y0, y1,
             fmt="{:.3f}")

    inside = best[best["minority_recall"].between(x0, x1)
                  & best["minority_precision"].between(y0, y1)]
    labels = []
    for _, row in inside.iterrows():
        beats = row["minority_f1"] > ref_f1
        point(ax, row, faded=not beats, s=110 if beats else 70)
        if beats:
            sd = row.get("minority_f1_std")
            sd = f" ± {sd:.3f}" if pd.notna(sd) else ""
            text = (f"{row['encoder']} + {row['dict_learner']} @{int(row['atoms'])}\n"
                    f"F1 {row['minority_f1']:.4f}{sd}\n"
                    f"+{row['minority_f1'] - ref_f1:.4f} vs {REF}")
            labels.append((row["minority_recall"], row["minority_precision"], text, "black"))
        else:
            labels.append((row["minority_recall"], row["minority_precision"],
                           f"{row['dict_learner']} {int(row['atoms'])}\n"
                           f"F1 {row['minority_f1']:.4f}", mf.NOTE_GREY))

    baseline_point(ax, REF, ref, s=170)
    labels.append((ref["minority_recall"], ref["minority_precision"],
                   f"{REF} @{ref_atoms}\nF1 {ref_f1:.4f}", mf.B_COLOR))
    mf.repel(ax, labels, fontsize=8.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    for s in ax.spines.values():
        s.set_edgecolor(ZOOM_EDGE)
        s.set_linestyle((0, (4, 2)))
        s.set_linewidth(1.2)

    ax.legend(handles=[
        Patch(facecolor=WIN_SHADE, alpha=0.25, edgecolor="none",
              label=f"Minority-F1 > {REF} ({ref_f1:.4f})"),
        Line2D([], [], color=mf.B_COLOR, ls=mf.B_DASH[REF], lw=1.4,
               label=f"{REF} iso-F1 (best of {ref_n} widths)"),
        Line2D([], [], color="#999999", marker="o", markersize=7, alpha=0.4,
               linestyle="none", markeredgecolor="black", label="does not beat " + REF),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"Zoom: combinations that outperform {REF}")


def make(clf):
    df = mf.load()
    mf.configure(df)
    pipe = df[df["family"] == "pipeline"]
    best = mf.best_per_combo(pipe, clf=clf, by="minority_f1")
    best = best.dropna(subset=["minority_precision", "minority_recall"])
    bpts = [(m, r, own, n) for m, r, own, n in
            mf.baseline_best_rows(df, "minority_f1", clf=clf)
            if pd.notna(r["minority_precision"]) and pd.notna(r["minority_recall"])]
    ref_rows = [(r, n) for m, r, _, n in bpts if m == REF]
    if not ref_rows:
        raise SystemExit(f"no {REF} row for {clf}")
    ref, ref_n = ref_rows[0]
    ref_f1, ref_atoms = float(ref["minority_f1"]), int(ref["atoms"])

    win = best[best["minority_f1"] > ref_f1].sort_values("minority_f1", ascending=False)
    if win.empty:
        raise SystemExit(f"nothing beats {REF} under {clf}")
    print(f"{REF} best Minority-F1 under {clf}: {ref_f1:.4f} @{ref_atoms}")
    for _, r in win.iterrows():
        print(f"  {r['encoder']:>10} + {r['dict_learner']:<12} @{int(r['atoms']):<5} "
              f"F1 {r['minority_f1']:.4f}  (+{r['minority_f1'] - ref_f1:.4f})")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(19, 8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # Set by hand: the arrow crosses the gap between the panels, which
    # tight_layout cannot account for.
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.075, top=0.875, wspace=0.3)
    full_panel(axl, best, bpts)
    window = zoom_window(win, ref)
    zoom_panel(axr, best, ref, ref_f1, ref_atoms, ref_n, window)

    # Dashed window on the full plot, and an arrow from it to the zoom panel.
    x0, x1, y0, y1 = window
    axl.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor=ZOOM_EDGE, ls=(0, (4, 2)), lw=1.4, zorder=9))
    fig.add_artist(ConnectionPatch(
        xyA=(x1, (y0 + y1) / 2), coordsA=axl.transData,
        xyB=(-0.13, 0.5), coordsB=axr.transAxes,      # stop short of the y-label
        arrowstyle="simple,head_width=0.9,head_length=0.9,tail_width=0.35",
        facecolor="white", edgecolor=ZOOM_EDGE, lw=1.2, mutation_scale=18,
        connectionstyle="arc3,rad=-0.15", zorder=10))

    mf.title2(fig, "Minority-class operating point",
              f"classifier: {mf.CLF_SHORT[clf]}   one point per combination at its best "
              f"Minority-F1   dotted curves are iso-F1   "
              f"{len(win)} combination(s) beat {REF}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    # PNG for viewing, PDF (vector, same layout) for \includegraphics in LaTeX.
    for ext in ("png", "pdf"):
        path = OUTDIR / f"s8_{mf.CLF_SHORT[clf]}_zoom.{ext}"
        fig.savefig(path, dpi=mf.DPI)
        print(f"wrote {path.relative_to(HERE.parent)}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clf", default="LinearSVM", choices=mf.CLASSIFIER_ORDER)
    make(ap.parse_args().clf)


if __name__ == "__main__":
    main()
