"""Data-driven cut-point detection for a ranked score curve.

Given per-feature scores sorted in *descending* order, the sorted curve looks
like a few highly-scored features followed by a long, flat tail of near-noise.
The "elbow" is the index where the marginal value of keeping the
next feature collapses -- the natural place to cut.

This module finds that index from the geometry of the curve itself, so the
number of features kept adapts to each dataset instead of being a fixed
fraction (the arbitrary 50th-percentile cut).

Nothing here depends on WL, the vocab, or labels: it consumes a score array and
returns where to cut.
"""

import numpy as np


def find_elbow_cut(scores, sorted_desc=True, min_keep=1):
    """Find the elbow of a decreasing score curve.

    Method: the classic "max distance to the chord". Normalise the sorted
    curve to the unit square, draw the straight line from the first point to
    the last, and pick the point farthest from that line -- for a convex
    decreasing curve that is exactly the elbow.

    Parameters
    ----------
    scores : array-like of float
        Feature scores, assumed sorted descending (highest first). Pass
        ``sorted_desc=False`` to have them sorted here.
    min_keep : int, default 1
        Floor on how many features to keep, for near-degenerate curves.

    Returns
    -------
    (n_keep, threshold) : (int, float)
        Keep ``n_keep`` features; ``threshold`` is the score of the last kept
        one (equivalently: keep every feature with ``score >= threshold``).
    """
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    if n < 3 or (y[0] - y[-1]) == 0:          # too short or flat -> keep all
        return n, float(y[-1])

    # normalise both axes to [0, 1] so distance is scale-invariant
    x_n = np.arange(n, dtype=float) / (n - 1)
    y_n = (y - y[-1]) / (y[0] - y[-1])

    # perpendicular distance of every point to the chord (0,1)->(1,0)
    x1, y1, x2, y2 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    dist = np.abs((y2 - y1) * x_n - (x2 - x1) * y_n + (x2 * y1 - y2 * x1)) / denom

    elbow = int(np.argmax(dist))
    n_keep = min(max(elbow + 1, int(min_keep)), n)
    return n_keep, float(y[n_keep - 1])


def find_energy_cut(scores, energy=0.99, sorted_desc=True, min_keep=1):
    """Keep the top features that together hold a fraction of the total score.

    An alternative to the geometric elbow. Where ``find_elbow_cut`` asks *where
    does the marginal value of the next feature collapse*, this asks *how many
    of the top features do I need to retain X% of the signal*. On the brutally
    front-loaded curves this pipeline produces, the elbow is statistically
    conservative -- it cuts as soon as individual scores go flat, discarding a
    long tail of weak-but-not-worthless features that still lift downstream AUC.
    An energy threshold reaches deliberately into that tail: it walks the sorted
    curve accumulating score and stops once the cumulative sum first reaches
    ``energy`` of the grand total.

    The score used here is exactly the ranking score -- ``total_presence *
    discriminative_score`` in the WL encoder -- so "energy" means "fraction of
    the summed discriminative mass", not an information-theoretic quantity.

    Parameters
    ----------
    scores : array-like of float
        Feature scores, assumed sorted descending (highest first). Pass
        ``sorted_desc=False`` to have them sorted here. Assumed non-negative,
        as the WL scores are; a cumulative fraction is only meaningful then.
    energy : float, default 0.99
        Target fraction of the total score to retain, in (0, 1]. 0.99 keeps
        enough features to cover 99% of the summed score. Larger keeps more.
    min_keep : int, default 1
        Floor on how many features to keep, for near-degenerate curves.

    Returns
    -------
    (n_keep, threshold) : (int, float)
        Keep ``n_keep`` features; ``threshold`` is the score of the last kept
        one (equivalently: keep every feature with ``score >= threshold``).
    """
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    total = float(y.sum())
    if n == 0:
        return 0, 0.0
    if total <= 0:                            # flat / all-zero -> keep all
        return n, float(y[-1])

    # smallest prefix whose cumulative score first reaches energy * total.
    # searchsorted finds the first index where the running sum crosses the
    # target; +1 turns that 0-based index into a count.
    target = float(energy) * total
    idx = int(np.searchsorted(np.cumsum(y), target))
    n_keep = min(max(idx + 1, int(min_keep)), n)
    return n_keep, float(y[n_keep - 1])


# --- plotting -------------------------------------------------------------
# The curve is brutally front-loaded: a couple of hundred features carry the
# signal and the remaining ~97% is a flat tail. One linear plot cannot show
# both ends, so every run renders four views of the same curve:
#
#   elbow.png         linear, full range. The reference figure -- the chord and
#                     the max-distance geometry are only truthful here.
#   elbow_zoom.png    the same linear axes cropped to the steep head.
#   elbow_logx.png    log rank: spreads the head across the page.
#   elbow_loglog.png  log rank + log score: additionally exposes the tail's
#                     decay, which every linear view pins to zero.
#
# (log-x + log-y *is* log-log -- one figure, not two.)
#
# The chord is drawn only on the linear figures. It is a straight line in
# linear space by construction, so on a log axis it would render as a curve
# that invites reading the elbow off geometry the method never used.

PCT_COLORS = {25: "#9467bd", 50: "#d62728", 75: "#8c564b"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")            # headless: no display needed
    import matplotlib.pyplot as plt
    return plt


def _prepare(scores, sorted_desc, min_keep):
    """Sorted curve + its elbow: the geometry every figure below shares."""
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size
    n_keep, threshold = find_elbow_cut(y, sorted_desc=True, min_keep=min_keep)
    return y, n, n_keep, n_keep - 1, threshold


def _summary_box(ax, n, n_keep, threshold, x=0.985, y=0.60):
    """Elbow vs the fixed-fraction cuts, as a monospace panel."""
    lines = [
        f"total features : {n}",
        f"elbow keep     : {n_keep} ({100.0 * n_keep / n:.1f}%)",
        f"elbow threshold: {threshold:.6g}",
        f"25th pct keep  : {min(int(round(n * 0.25)), n)}",
        f"50th pct keep  : {min(int(round(n * 0.50)), n)}",
        f"75th pct keep  : {min(int(round(n * 0.75)), n)}",
    ]
    ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes,
        fontsize=8, family="monospace", ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )


def _save(fig, save_path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _plt().close(fig)
    return save_path


def _draw_linear(ax, y, n, n_keep, elbow_idx, threshold, title, hi=None):
    """The shared linear figure: curve, chord, percentile margins, elbow.

    ``hi`` crops the rank axis to [0, hi] for the zoomed view. Percentile marks
    outside that window are skipped rather than drawn off-screen.
    """
    ranks = np.arange(n)
    span = y[0] - y[-1] + 1e-12
    x_span = hi if hi else n         # scales the annotation offsets to the view

    ax.plot(ranks, y, color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(0, elbow_idx, color="#2ca02c", alpha=0.08)

    # chord used by the max-distance elbow method (provenance detail)
    ax.plot([0, n - 1], [y[0], y[-1]], color="#7f7f7f", ls=":", lw=1.2,
            label="chord (max-distance method)")

    # 25th / 50th / 75th percentile margins (rank-based fixed-fraction cuts)
    for p, c in PCT_COLORS.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        if hi and idx > hi:          # off the cropped view entirely
            continue
        ax.axvline(idx, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"{p}th pct\n(keep {idx})",
            xy=(idx, y[idx]), xytext=(idx, y[idx] + 0.06 * span),
            color=c, fontsize=8, ha="center", va="bottom",
        )

    ax.scatter([elbow_idx], [y[elbow_idx]], color="#ff7f0e", zorder=5, s=70,
               edgecolor="black", linewidth=0.6,
               label=f"elbow: keep {n_keep} ({100.0 * n_keep / n:.1f}%)")
    ax.annotate(
        f"elbow @ {elbow_idx}\nthreshold={threshold:.4g}",
        xy=(elbow_idx, y[elbow_idx]),
        xytext=(elbow_idx + 0.04 * x_span, y[elbow_idx] + 0.15 * span),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        fontsize=9, color="#ff7f0e",
    )

    if hi:
        ax.axhline(threshold, color="#ff7f0e", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlim(0, hi)

    ax.set_xlabel("feature rank (sorted by score, descending)")
    ax.set_ylabel("discriminative score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    _summary_box(ax, n, n_keep, threshold)


def _title(kind, title, n):
    return (f"WL feature-selection elbow{kind}"
            + (f" — {title}" if title else "")
            + f"\ntotal features: {n}")


def plot_elbow_curve(scores, save_path, sorted_desc=True, min_keep=1, title=None):
    """Render the sorted-score curve with the elbow cut and save it to disk.

    The figure shows, on the decreasing score curve:
      * the elbow point and the number of features it keeps,
      * the 25th / 50th / 75th percentile margins (rank-based cuts) so the
        data-driven elbow can be compared against fixed-fraction cuts such as
        the old 50th-percentile rule,
      * the chord used by the max-distance method, for provenance,
      * a summary box of counts / thresholds.

    Parameters
    ----------
    scores : array-like of float
        Feature scores (assumed sorted descending; pass ``sorted_desc=False``
        otherwise). The full curve, *not* just the kept portion.
    save_path : str
        Where to write the PNG. Parent directories are created if missing.
    title : str, optional
        Extra context (e.g. "wl_fddl_gpu / nci_full") shown in the title.

    Returns
    -------
    str
        The path the figure was written to.
    """
    plt = _plt()
    y, n, n_keep, elbow_idx, threshold = _prepare(scores, sorted_desc, min_keep)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_linear(ax, y, n, n_keep, elbow_idx, threshold, _title("", title, n))
    return _save(fig, save_path)


def plot_elbow_curve_zoom(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                          zoom_to=None):
    """The linear figure cropped to the steep head.

    Identical geometry to ``plot_elbow_curve``, just a narrower rank axis. The
    point of zooming rather than reaching for a log axis is that the chord
    stays straight and the elbow stays where the max-distance method actually
    put it, so nothing in the figure has to be mentally un-distorted.

    Parameters
    ----------
    zoom_to : int, optional
        Highest rank shown. Defaults to 4x the elbow, which leaves the cut
        about a quarter of the way into the window.
    """
    plt = _plt()
    y, n, n_keep, elbow_idx, threshold = _prepare(scores, sorted_desc, min_keep)
    hi = min(int(zoom_to) if zoom_to else max(4 * n_keep, 50), n - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_linear(ax, y, n, n_keep, elbow_idx, threshold,
                 _title(f" — zoom on ranks 0–{hi}", title, n), hi=hi)
    return _save(fig, save_path)


def plot_elbow_curve_logx(scores, save_path, sorted_desc=True, min_keep=1, title=None):
    """Same curve on a logarithmic rank axis.

    Adjustments the log axis forces:
      * ranks are shifted to 1-based, because log(0) is undefined;
      * the chord is not drawn -- it is straight only in linear space, so here
        it would render as a curve and invite reading the elbow off geometry
        the method never used. It stays on the linear figures.
      * the percentile labels are staggered vertically -- log rank crushes the
        25/50/75% marks into the last third of the page, where they would
        otherwise overprint each other.

    The elbow is still the linear-space one: ``find_elbow_cut`` measures
    distance to the chord in normalised *linear* coordinates and this figure
    only re-draws that result. Log rank is a lens on the curve, not a different
    cut rule -- hence the footnote.
    """
    plt = _plt()
    y, n, n_keep, elbow_idx, threshold = _prepare(scores, sorted_desc, min_keep)
    x = np.arange(1, n + 1)          # 1-based: log(0) is undefined
    span = y[0] - y[-1] + 1e-12

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    ax.plot(x, y, color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(1, elbow_idx + 1, color="#2ca02c", alpha=0.08)

    for (p, c), lift in zip(PCT_COLORS.items(), (0.22, 0.14, 0.06)):
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx + 1, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"{p}th pct (keep {idx})",
            xy=(idx + 1, y[idx]), xytext=(idx + 1, y[idx] + lift * span),
            color=c, fontsize=8, ha="right", va="bottom",
        )

    ax.scatter([elbow_idx + 1], [y[elbow_idx]], color="#ff7f0e", zorder=5, s=70,
               edgecolor="black", linewidth=0.6,
               label=f"elbow: keep {n_keep} ({100.0 * n_keep / n:.1f}%)")
    ax.annotate(
        f"elbow @ {elbow_idx}\nthreshold={threshold:.4g}",
        xy=(elbow_idx + 1, y[elbow_idx]),
        # sits above the staggered percentile labels, not across them
        xytext=(max((elbow_idx + 1) * 1.3, 2.0), y[elbow_idx] + 0.45 * span),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        fontsize=9, color="#ff7f0e", ha="left",
    )

    ax.set_xlabel("feature rank (1-based, log scale)")
    ax.set_ylabel("discriminative score")
    ax.set_title(_title(" — log rank", title, n))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    _summary_box(ax, n, n_keep, threshold)
    ax.text(
        0.5, -0.13,
        "log rank only rescales the view: the elbow is computed in linear rank space "
        "(see the linear figures for the chord it is measured against).",
        transform=ax.transAxes, fontsize=7.5, color="#666666", ha="center", va="top",
    )

    return _save(fig, save_path)


def plot_elbow_curve_loglog(scores, save_path, sorted_desc=True, min_keep=1, title=None):
    """Same curve on log rank *and* log score (i.e. log-log).

    Adjustments the log-log axes force, on top of the log-rank ones:
      * scores of exactly 0 -- a feature equally present in both classes scores
        0 -- cannot be drawn on a log axis at all. They are dropped and counted
        in the footnote rather than silently clipped to some invented floor.
      * the chord is not drawn, for the same reason as on the log-rank figure.

    This is the only view in which the tail is legible; every linear view pins
    it to zero. It is therefore the view that shows whether the tail decays
    like a power law (a straight line here) or dies off faster.
    """
    plt = _plt()
    y, n, n_keep, elbow_idx, threshold = _prepare(scores, sorted_desc, min_keep)
    x = np.arange(1, n + 1)

    pos = y > 0                      # log(<=0) is undefined -> cannot be shown
    n_dropped = int((~pos).sum())
    if not pos.any():                # degenerate: nothing to draw on log-y
        pos = np.ones_like(y, dtype=bool)
        n_dropped = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x[pos], y[pos], color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(1, elbow_idx + 1, color="#2ca02c", alpha=0.08)

    # labels omitted here: log-log leaves no clear band for them
    for p, c in PCT_COLORS.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx + 1, color=c, ls="--", lw=1.0, alpha=0.8)

    if y[elbow_idx] > 0:
        ax.scatter([elbow_idx + 1], [y[elbow_idx]], color="#ff7f0e", zorder=5, s=70,
                   edgecolor="black", linewidth=0.6,
                   label=f"elbow: keep {n_keep} ({100.0 * n_keep / n:.1f}%)")
        ax.axhline(threshold, color="#ff7f0e", ls=":", lw=0.9, alpha=0.7)
        ax.annotate(
            f"elbow @ {elbow_idx}\nthreshold={threshold:.4g}",
            xy=(elbow_idx + 1, y[elbow_idx]),
            xytext=(max((elbow_idx + 1) * 3.0, 4.0), y[elbow_idx] * 4.0),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
            fontsize=9, color="#ff7f0e",
        )

    ax.set_xlabel("feature rank (1-based, log scale)")
    ax.set_ylabel("discriminative score (log scale)")
    ax.set_title(_title(" — log rank / log score", title, n))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    _summary_box(ax, n, n_keep, threshold, y=0.98)

    note = ("log axes only rescale the view: the elbow is computed in linear rank "
            "space (see the linear figures for the chord it is measured against).")
    if n_dropped:
        note += (f"\n{n_dropped} of {n} features score exactly 0 and cannot be drawn "
                 f"on a log axis; they are omitted from this view.")
    ax.text(
        0.5, -0.13, note, transform=ax.transAxes,
        fontsize=7.5, color="#666666", ha="center", va="top",
    )

    return _save(fig, save_path)


def plot_elbow_suite(scores, analytics_dir, stem="wl_feature_selection_elbow",
                     sorted_desc=True, min_keep=1, title=None):
    """Write all four views of the curve into ``analytics_dir``.

    Returns the written paths, in the order described at the top of this
    section (linear, zoomed linear, log rank, log-log).
    """
    import os
    variants = (
        (plot_elbow_curve, f"{stem}.png"),
        (plot_elbow_curve_zoom, f"{stem}_zoom.png"),
        (plot_elbow_curve_logx, f"{stem}_logx.png"),
        (plot_elbow_curve_loglog, f"{stem}_loglog.png"),
    )
    return [
        fn(scores, os.path.join(analytics_dir, filename),
           sorted_desc=sorted_desc, min_keep=min_keep, title=title)
        for fn, filename in variants
    ]
