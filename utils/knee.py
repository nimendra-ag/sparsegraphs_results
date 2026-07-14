"""Data-driven cut-point detection for a ranked score curve.

Given per-feature scores sorted in *descending* order, the sorted curve looks
like a few highly-scored features followed by a long, flat tail of near-noise.
The "knee" (a.k.a. elbow) is the index where the marginal value of keeping the
next feature collapses -- the natural place to cut.

This module finds that index from the geometry of the curve itself, so the
number of features kept adapts to each dataset instead of being a fixed
fraction (the arbitrary 50th-percentile cut).

Nothing here depends on WL, the vocab, or labels: it consumes a score array and
returns where to cut.
"""

import numpy as np


def find_knee_cut(scores, sorted_desc=True, min_keep=1):
    """Find the knee (elbow) of a decreasing score curve.

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

    knee = int(np.argmax(dist))
    n_keep = min(max(knee + 1, int(min_keep)), n)
    return n_keep, float(y[n_keep - 1])


def plot_knee_curve(scores, save_path, sorted_desc=True, min_keep=1, title=None):
    """Render the sorted-score curve with the knee cut and save it to disk.

    The figure shows, on the decreasing score curve:
      * the knee point and the number of features it keeps,
      * the 25th / 50th / 75th percentile margins (rank-based cuts) so the
        data-driven knee can be compared against fixed-fraction cuts such as
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
    import os
    import matplotlib
    matplotlib.use("Agg")            # headless: no display needed
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    n_keep, threshold = find_knee_cut(y, sorted_desc=True, min_keep=min_keep)
    knee_idx = n_keep - 1
    ranks = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    # full score curve
    ax.plot(ranks, y, color="#1f77b4", lw=1.6, label="feature score (sorted)")

    # kept region shading
    ax.axvspan(0, knee_idx, color="#2ca02c", alpha=0.08)

    # chord used by the max-distance knee method (provenance detail)
    ax.plot([0, n - 1], [y[0], y[-1]], color="#7f7f7f", ls=":", lw=1.2,
            label="chord (max-distance method)")

    # 25th / 50th / 75th percentile margins (rank-based fixed-fraction cuts)
    pct_colors = {25: "#9467bd", 50: "#d62728", 75: "#8c564b"}
    for p, c in pct_colors.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"{p}th pct\n(keep {idx})",
            xy=(idx, y[idx]), xytext=(idx, y[idx] + 0.06 * (y[0] - y[-1] + 1e-12)),
            color=c, fontsize=8, ha="center", va="bottom",
        )

    # knee point
    ax.scatter([knee_idx], [y[knee_idx]], color="#ff7f0e", zorder=5, s=70,
               edgecolor="black", linewidth=0.6,
               label=f"knee: keep {n_keep} ({100.0 * n_keep / n:.1f}%)")
    ax.annotate(
        f"knee @ {knee_idx}\nthreshold={threshold:.4g}",
        xy=(knee_idx, y[knee_idx]),
        xytext=(knee_idx + 0.04 * n, y[knee_idx] + 0.15 * (y[0] - y[-1] + 1e-12)),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        fontsize=9, color="#ff7f0e",
    )

    ax.set_xlabel("feature rank (sorted by score, descending)")
    ax.set_ylabel("discriminative score")
    ax.set_title(
        f"WL feature-selection elbow"
        + (f" — {title}" if title else "")
        + f"\ntotal features: {n}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # summary box: knee vs the fixed-fraction cuts
    lines = [
        f"total features : {n}",
        f"knee keep      : {n_keep} ({100.0 * n_keep / n:.1f}%)",
        f"knee threshold : {threshold:.6g}",
        f"25th pct keep  : {min(int(round(n * 0.25)), n)}",
        f"50th pct keep  : {min(int(round(n * 0.50)), n)}  (old cut)",
        f"75th pct keep  : {min(int(round(n * 0.75)), n)}",
    ]
    ax.text(
        0.985, 0.60, "\n".join(lines), transform=ax.transAxes,
        fontsize=8, family="monospace", ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
