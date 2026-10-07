"""Shared loading, constants and plot style for the wl_fddl analysis scripts.

Everything downstream reads wl_fddl_all_results.csv (built by
build_wl_fddl_results.py); the only exception is the replicate comparison,
which reads the two NCI1/2048 run summaries straight from results/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV = HERE / "wl_fddl_all_results.csv"
FIG_DIR = HERE / "figures"
TABLE_DIR = HERE / "tables"

DATASETS = ["NCI1", "NCI33", "NCI41", "NCI47"]
# Atom sizes recorded for every dataset; 4096 exists for NCI1/NCI33 only.
COMMON_ATOMS = [128, 256, 512, 1024, 2048]

# Display order == categorical slot order. RF / LR / SRC_pure take slots 1-3
# because they share the scatter plots, where only the first three slots
# validate all-pairs.
CLASSIFIERS = ["RandomForest", "LogisticRegression", "SRC_pure",
               "LinearSVM", "GradientBoosting", "SRC_fddl"]
# SRC_fddl tracks SRC_pure to within 0.004 on every metric (fig10), so the
# line charts draw SRC_pure only.
LINE_CLASSIFIERS = CLASSIFIERS[:5]
SHORT = {"RandomForest": "RF", "LogisticRegression": "LogReg",
         "SRC_pure": "SRC (pure)", "LinearSVM": "LinSVM",
         "GradientBoosting": "GBoost", "SRC_fddl": "SRC (fddl)"}

METRICS = ["Macro-F1", "MCC", "ROC-AUC", "Macro-PR-AUC", "Accuracy",
           "Macro-Precision", "Macro-Recall", "Minority-F1",
           "Minority-Precision", "Minority-Recall", "Minority-PR-AUC"]

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "X"]
CLF_COLOR = dict(zip(CLASSIFIERS, SLOTS))
CLF_MARKER = dict(zip(CLASSIFIERS, MARKERS))
DS_COLOR = dict(zip(DATASETS, SLOTS))
DS_MARKER = dict(zip(DATASETS, MARKERS))

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["classifier"] = pd.Categorical(df["classifier"], CLASSIFIERS, ordered=True)
    df["dataset_label"] = pd.Categorical(df["dataset_label"], DATASETS, ordered=True)
    return df.sort_values(["dataset_label", "total_atoms", "classifier"]).reset_index(drop=True)


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def atoms_axis(ax, atoms=(128, 256, 512, 1024, 2048, 4096)) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(atoms))
    ax.set_xticklabels([str(a) for a in atoms])
    ax.minorticks_off()
    ax.set_xlabel("Dictionary atoms")
