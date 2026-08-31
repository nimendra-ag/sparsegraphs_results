"""
Graph classification on NCI (binary) with a GCN, faithful to the original
Kipf & Welling backbone, benchmarked with Monte Carlo cross-validation.

Pipeline:
  1. Load molecules from an SDF with RDKit -> NetworkX graphs + integer labels.
  2. Convert each NetworkX graph to a PyG Data object:
       - node features  : one-hot of the atom symbol over a FIXED 45-element
                           vocabulary (shared across all graphs) -> in_dim = 45
       - edge_index     : bonds, made bidirectional for the undirected graph
       - y              : binary class index (raw labels mapped to {0, 1})
  3. Train a GCN: two GCNConv layers (original backbone) + global mean-pool
     readout + linear classifier.
  4. Report the same metric set as every other arm in implements/ (see
     utils/mccv.METRIC_KEYS) on the held-out test split, for each master seed.

Results format — identical to the `*_mccv.py` scripts (wl_fddl_gpu_mccv.py,
wl_lcksvd_mccv.py, ...), because it is produced by the same utils/mccv
persistence + aggregation machinery:

    results/mc_cv_gcn_nci_full_atoms<hidden>_<start>_<end>/
        per_run_metrics.csv    one row per master seed, one column per metric
        summary_mean_std.csv   mean +/- sample-std (+ 95% t-CI) over the seeds
        per_run_timings.csv    per-phase wall-clock, one row per seed
        summary_timings.csv    per-phase mean/std/min/max over the seeds
        manifest.json          run + per-seed provenance (hyperparameters here)

The GCN has no dictionary, so the CSV's `total_atoms` column carries the hidden
width instead — the closest capacity knob this model has, kept under the shared
column name so the tables still line up with the dictionary-learning arms.

Metric notes:
  * Macro-*        unweighted mean over both classes (both count equally).
  * Accuracy / ROC-AUC / MCC   symmetric over the whole split.
  * Minority-*     the minority class alone — the headline figures under
                   imbalance. Minority-PR-AUC is average_precision_score with
                   the minority class as pos_label.
  * ROC-AUC and the PR-AUCs use P(class = 1) from the softmax, not the hard
    prediction.

Architecture / hyperparameters mirror the original GCN paper:
  hidden = 16, dropout = 0.5, Adam lr = 0.01, weight decay 5e-4 (first layer).

Usage
-----
    python implements/train_gcn_graph_nci.py                 # full MC-CV run
    python implements/train_gcn_graph_nci.py --seed 7 --out-dir results/<run>
    python implements/train_gcn_graph_nci.py --aggregate --out-dir results/<run>

Extra dependencies: rdkit, networkx, scikit-learn.
    pip install rdkit networkx scikit-learn
"""

from __future__ import annotations

import os
import re
import sys
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from utils import mccv
from utils.graph_data import dataset_tag
from utils.seeding import seed_everything, derive_seeds


# --------------------------------------------------------------------------- #
# Run identity (used for the results folder name + manifest, like the siblings)
# --------------------------------------------------------------------------- #
IMPLEMENTATION = "gcn"
DATASET = "nci_full"


def sdf_dataset_id(sdf_path):
    """NCI screen id from the SDF file name ("33total-connect.sdf" -> 33).

    This arm is pointed straight at an SDF rather than going through
    GraphDataLoader, so the id has to be recovered from the path to reach the
    folder name and manifest the way it does for every other arm. An
    unrecognised name yields None, which simply leaves the id out.
    """
    m = re.match(r"(\d+)total-connect", os.path.basename(str(sdf_path)))
    return int(m.group(1)) if m else None

# Each master seed = one fully reproducible run (its own train/test partition +
# its own model initialisation). Only the orchestrating parent reads this;
# workers are told their seed on the command line.
MASTER_SEEDS = mccv.default_master_seeds()


# --------------------------------------------------------------------------- #
# Fixed node-label vocabulary (order fixed -> stable one-hot columns)
# --------------------------------------------------------------------------- #
NODE_LABELS = [
    'Ac', 'Ag', 'As', 'Au', 'B', 'Bi', 'Br', 'C', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga',
    'Gd', 'Ge', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
    'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn', 'Te', 'Th', 'Ti', 'Tl', 'U', 'V', 'Y', 'Zn', 'Zr'
]
LABEL_TO_IDX = {sym: i for i, sym in enumerate(NODE_LABELS)}
NUM_NODE_FEATURES = len(NODE_LABELS)


def one_hot_nodes(symbols: list[str]) -> Tensor:
    """Convert a graph's atom-symbol strings to a [num_nodes, 45] one-hot tensor."""
    try:
        idx = torch.tensor([LABEL_TO_IDX[s] for s in symbols], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(
            f"Atom symbol {exc} is not in the 45-element vocabulary. "
            f"If you keep explicit hydrogens (removeHs=False), add 'H' to "
            f"NODE_LABELS or load with removeHs=True."
        ) from exc
    return F.one_hot(idx, num_classes=NUM_NODE_FEATURES).float()


# --------------------------------------------------------------------------- #
# Data loading: SDF -> NetworkX graphs + labels  (your loader)
# --------------------------------------------------------------------------- #
def load_nci_sdf(filepath: str):
    """Read an NCI SDF file into (graphs, labels)."""
    from rdkit import Chem  # lazy import so the file imports without rdkit
    import networkx as nx

    print(f"Loading NCI dataset from {filepath}")
    graphs, labels = [], []

    supplier = Chem.SDMolSupplier(filepath, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue

        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        labels.append(int(float(mol.GetProp("value"))))
        graphs.append(G)

    print(f"Loaded {len(graphs)} graphs")
    return graphs, labels


# --------------------------------------------------------------------------- #
# NetworkX -> PyG Data
# --------------------------------------------------------------------------- #
def networkx_to_data(G, label_idx: int) -> Data:
    """Convert one NetworkX molecule graph to a PyG Data object."""
    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}

    symbols = [G.nodes[n]["label"] for n in nodes]
    x = one_hot_nodes(symbols)

    if G.number_of_edges() > 0:
        edges = []
        for u, v in G.edges():
            a, b = node_map[u], node_map[v]
            edges.append((a, b))
            edges.append((b, a))  # undirected -> add both directions
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # isolated atom(s)

    y = torch.tensor([label_idx], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_dataset(graphs, labels) -> list[Data]:
    """Convert all graphs, mapping the two raw label values to {0, 1}."""
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError(f"Expected binary labels; found {unique}")
    label_map = {unique[0]: 0, unique[1]: 1}  # e.g. {-1: 0, +1: 1}
    print(f"Label mapping (raw -> class index): {label_map}")
    return [networkx_to_data(G, label_map[lab]) for G, lab in zip(graphs, labels)]


# --------------------------------------------------------------------------- #
# Model: original 2x GCNConv backbone + pooling readout + linear
# --------------------------------------------------------------------------- #
class GraphGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# --------------------------------------------------------------------------- #
# Train / predict / metrics
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total += float(loss) * data.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_pred, y_prob) numpy arrays. y_prob = P(class 1)."""
    model.eval()
    ys, preds, probs = [], [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        prob1 = F.softmax(logits, dim=1)[:, 1]      # probability of positive class
        ys.append(data.y.cpu())
        preds.append(logits.argmax(dim=1).cpu())
        probs.append(prob1.cpu())
    return (
        torch.cat(ys).numpy(),
        torch.cat(preds).numpy(),
        torch.cat(probs).numpy(),
    )


def _scores_for(y_prob, label):
    """Score vector treating `label` as the positive class.

    `y_prob` is P(class = 1), so the complementary class is scored by 1 - y_prob.
    Needed because average precision is asymmetric — it ignores true negatives,
    so each class needs its own ranking to be scored fairly. Same rule as
    utils/evaluator.Evaluator._scores_for.
    """
    return y_prob if label == 1 else 1.0 - y_prob


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """The metric set in utils/mccv.METRIC_KEYS, computed the same way the
    sklearn Evaluator computes it, so a GCN row is directly comparable to a
    dictionary-learning row.

    Class roles (minority / majority) are read off the split itself rather than
    assumed, so the Minority-* columns mean the same thing here as everywhere
    else even if a resample shifts the ratio.
    """
    labels, counts = np.unique(y_true, return_counts=True)
    if len(labels) < 2:
        raise ValueError(f"Split contains a single class ({labels.tolist()}); "
                         f"metrics are undefined.")
    majority_label = labels[np.argmax(counts)]
    minority_label = labels[np.argmin(counts)]

    # Per-class precision / recall / F1, ordered [minority, majority].
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=[minority_label, majority_label],
        average=None, zero_division=0,
    )

    # Average precision ignores true negatives, so each class is scored against
    # its own ranking rather than sharing one score vector.
    ap_minority = average_precision_score(
        y_true, _scores_for(y_prob, minority_label), pos_label=minority_label
    )
    ap_majority = average_precision_score(
        y_true, _scores_for(y_prob, majority_label), pos_label=majority_label
    )

    return {
        # Macro-averaged: both classes count equally regardless of size.
        "Macro-Precision": float(prec_pc.mean()),
        "Macro-Recall": float(rec_pc.mean()),
        "Macro-F1": float(f1_pc.mean()),
        "Macro-PR-AUC": float((ap_minority + ap_majority) / 2),
        # Symmetric over the whole split — no per-class variant in a binary
        # problem, so reported once.
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        # Minority class alone — the headline figures under imbalance.
        "Minority-Precision": float(prec_pc[0]),
        "Minority-Recall": float(rec_pc[0]),
        "Minority-F1": float(f1_pc[0]),
        "Minority-PR-AUC": float(ap_minority),
    }


def report(name: str, metrics: dict[str, float], n_graphs: int) -> None:
    print(f"\n{name} set ({n_graphs} graphs):")
    for key in mccv.METRIC_KEYS:
        print(f"  {key:20s}: {metrics[key]:.4f}")


# --------------------------------------------------------------------------- #
# One Monte Carlo CV run (a single master seed)
# --------------------------------------------------------------------------- #
# Loaded lazily and once per process. The orchestrator parent never touches it
# (it only spawns workers), so the SDF is never parsed there.
_DATA = {}


def _get_dataset(sdf_path):
    if sdf_path not in _DATA:
        graphs, labels = load_nci_sdf(sdf_path)
        _DATA[sdf_path] = build_dataset(graphs, labels)
    return _DATA[sdf_path]


def _seed_manifest_entry(master_seed, args, seeds, split_sizes):
    """Provenance for one seed, in the shape utils/mccv writes for every arm.

    `n_selected_features` is the field the shared manifest summariser folds into
    a run-level mean/std. The GCN has no adaptive feature selection — its input
    vocabulary is the fixed NODE_LABELS atom table — so it is constant here, but
    it is recorded under the same name so the manifests stay readable side by
    side.
    """
    return {
        "master_seed": int(master_seed),
        # --- features ---
        "n_selected_features": int(NUM_NODE_FEATURES),
        "n_scored_features": int(NUM_NODE_FEATURES),
        "selection": "fixed_atom_vocabulary",
        # --- model (the GCN's analogue of the dictionary block) ---
        "total_atoms": int(args.hidden),
        "encoder_class": "GCNConv x2 + global_mean_pool",
        "dict_learner_class": None,
        "hyperparameters": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sdf": args.sdf,
        },
        # --- reproducibility ---
        "derived_seeds": {k: int(v) for k, v in seeds.items()},
        "split_sizes": {k: int(v) for k, v in split_sizes.items()},
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_once(master_seed: int, args):
    """One full training + evaluation under a single master seed.

    Resamples the train/test partition and re-initialises the model, so each
    seed is an independent draw. Returns (row, timings, seed_manifest) in the
    shape utils/mccv persists.
    """
    timings = {}  # ordered: insertion order == pipeline order
    seed_t0 = time.perf_counter()

    with mccv._phase(timings, "data_load"):
        dataset = _get_dataset(args.sdf)

    # Global RNGs plus independent sub-seeds for the components we control.
    seed_everything(master_seed)
    s_split, s_model, s_train, _ = derive_seeds(master_seed, 4)

    # --- 1. Resample the partition (this is the MC-CV resampling step) -------
    with mccv._phase(timings, "partition"):
        gen = torch.Generator().manual_seed(s_split)
        perm = torch.randperm(len(dataset), generator=gen).tolist()
        shuffled = [dataset[i] for i in perm]
        n_train = int(0.8 * len(shuffled))
        train_ds, test_ds = shuffled[:n_train], shuffled[n_train:]

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Build the model (weights seeded independently of the split) ------
    with mccv._phase(timings, "build_model"):
        torch.manual_seed(s_model)
        model = GraphGCN(
            in_channels=NUM_NODE_FEATURES,
            hidden_channels=args.hidden,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            [
                {"params": model.conv1.parameters(), "weight_decay": args.weight_decay},
                {"params": model.conv2.parameters(), "weight_decay": 0.0},
                {"params": model.lin.parameters(), "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )

    seed_manifest = _seed_manifest_entry(
        master_seed, args,
        seeds={"split": s_split, "model": s_model, "train": s_train},
        split_sizes={"train": len(train_ds), "test": len(test_ds)},
    )

    # --- 3. Train ------------------------------------------------------------
    with mccv._phase(timings, "train"):
        torch.manual_seed(s_train)
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device)
            if epoch == 1 or epoch % 10 == 0:
                y_true, y_pred, _ = predict(model, test_loader, device)
                test_acc = accuracy_score(y_true, y_pred)
                print(f"Epoch {epoch:03d} | train_loss={loss:.4f} | test_acc={test_acc:.4f}",
                      flush=True)

    # --- 4. Final metric report on both splits -------------------------------
    row = {}
    with mccv._phase(timings, "eval_train"):
        y_true, y_pred, y_prob = predict(model, train_eval_loader, device)
        train_metrics = compute_metrics(y_true, y_pred, y_prob)
    report("Train", train_metrics, len(train_ds))
    row.update(mccv._flatten("GCN_train", train_metrics))

    with mccv._phase(timings, "eval_test"):
        y_true, y_pred, y_prob = predict(model, test_loader, device)
        test_metrics = compute_metrics(y_true, y_pred, y_prob)
    report("Test", test_metrics, len(test_ds))
    # Unsuffixed prefix == held-out test, matching the sibling MC-CV scripts
    # where "LogisticRegression/..." are the test-split numbers.
    row.update(mccv._flatten("GCN", test_metrics))

    timings["seed_total"] = time.perf_counter() - seed_t0
    seed_manifest["seed_total_sec"] = round(timings["seed_total"], 3)

    # Per-seed breakdown, biggest phase first, so the bottleneck is obvious.
    print(f"\n----- timing breakdown | seed={master_seed} -----")
    for label, secs in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        share = 100.0 * secs / timings["seed_total"] if timings["seed_total"] else 0.0
        print(f"  {label:22s} {secs:8.1f}s  ({share:4.1f}%)")

    return row, timings, seed_manifest


# --------------------------------------------------------------------------- #
# Worker / orchestrator (same execution model as utils/mccv: 1 process per seed)
# --------------------------------------------------------------------------- #
def run_seed_worker(master_seed, out_dir, args):
    """Run one seed and append its metrics, then return. Designed to be the
    whole lifetime of a subprocess so that process exit frees all RAM/VRAM."""
    print(f"\n########## Monte Carlo CV run | master_seed={master_seed} ##########")
    row, timings, seed_manifest = run_once(master_seed, args)
    # The GCN has no dictionary; the shared `total_atoms` column carries the
    # hidden width, which is this model's capacity knob.
    mccv.append_run_row(out_dir, master_seed, args.hidden, row)
    mccv.append_timings_row(out_dir, master_seed, timings)
    mccv.append_manifest_entry(out_dir, seed_manifest,
                               implementation=IMPLEMENTATION, dataset=DATASET,
                               dataset_id=sdf_dataset_id(args.sdf))


def _worker_argv(args):
    """The hyperparameter flags to forward to each worker subprocess, so every
    seed trains the model the parent was asked for rather than the defaults."""
    return [
        "--sdf", str(args.sdf),
        "--hidden", str(args.hidden),
        "--dropout", str(args.dropout),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]


def orchestrate(seeds, args):
    """Spawn one fresh subprocess per seed (sequentially), then aggregate.

    Mirrors utils/mccv.orchestrate — each subprocess fully exits before the next
    starts, so the OS reclaims all of its memory and CUDA context and every seed
    runs on a clean machine. A crashed seed is logged and skipped by default so
    the surviving seeds still produce a summary.
    """
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Which NCI screen this run trained on is part of its identity, so it goes
    # in the folder name and the manifest.
    dataset_id = sdf_dataset_id(args.sdf)
    tag = dataset_tag(DATASET, dataset_id)
    out_dir = os.path.join("results", f"mc_cv_{IMPLEMENTATION}_{tag}_{started_at}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Monte Carlo CV | one process per seed | out_dir={out_dir}")
    mccv.init_manifest(out_dir, IMPLEMENTATION, DATASET, seeds, started_at,
                       dataset_id=dataset_id)

    failed = []
    for seed in seeds:
        print(f"\n>>> launching worker for master_seed={seed} ...")
        # Inherit stdout/stderr so the worker logs stream live.
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--seed", str(seed), "--out-dir", out_dir] + _worker_argv(args),
        )
        if result.returncode != 0:
            msg = f"seed={seed} FAILED (exit code {result.returncode})"
            if args.fail_fast:
                raise SystemExit(f"Aborting (--fail-fast): {msg}")
            print(f"\n!!! {msg} - skipping and continuing with remaining seeds.")
            failed.append(seed)

    if len(failed) == len(seeds):
        raise SystemExit(
            f"All {len(seeds)} seeds failed; nothing to aggregate. "
            f"This usually means a deterministic bug, not a transient fault."
        )

    total_atoms, _ = mccv.aggregate_and_report(out_dir)
    ended_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    mccv.finalize_manifest(out_dir, total_atoms, failed, ended_at)

    if failed:
        succeeded = [s for s in seeds if s not in failed]
        print(
            f"\n*** WARNING: {len(failed)} of {len(seeds)} seeds failed and were "
            f"skipped: {failed}. Summary is over the {len(succeeded)} surviving "
            f"seeds: {succeeded}. Re-run a failed seed with "
            f"`--seed <N> --out-dir {out_dir}` then `--aggregate` to fill it in."
        )

    # Finalise the folder name with the capacity + start/end timestamps, to match
    # the sibling convention. Best-effort: never lose results to a rename failure
    # (e.g. an open handle / antivirus lock on Windows).
    final_dir = os.path.join(
        "results",
        f"mc_cv_{IMPLEMENTATION}_{tag}_atoms{total_atoms}_{started_at}_{ended_at}",
    )
    try:
        os.rename(out_dir, final_dir)
        print(f"\nRun complete -> {final_dir}")
    except OSError as e:
        print(f"\nRun complete -> {out_dir} (folder rename skipped: {e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo CV for a GCN on NCI, one OS process per seed."
    )
    p.add_argument("--sdf", default="datasets/NCI_full/1total-connect.sdf")
    p.add_argument("--hidden", type=int, default=32) # 16, 32 64
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    # --- MC-CV control flags (same three modes as the *_mccv.py scripts) ---
    p.add_argument(
        "--seed", type=int, default=None,
        help="Worker mode: run this single master seed and append its metrics.",
    )
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="Run folder for per_run_metrics.csv (required with --seed/--aggregate).",
    )
    p.add_argument(
        "--aggregate", action="store_true",
        help="Aggregate an existing --out-dir into summary_mean_std.csv and exit.",
    )
    p.add_argument(
        "--master-seeds", type=int, nargs="+", default=None,
        help="Master seeds to run (default: a random draw, see utils/mccv).",
    )
    p.add_argument(
        "--fail-fast", action="store_true",
        help="Abort the whole run on the first seed failure "
             "(default: skip the failed seed and continue).",
    )
    return p.parse_args(argv)


def main() -> None:
    parser_args = parse_args()

    if parser_args.aggregate:
        if not parser_args.out_dir:
            raise SystemExit("--aggregate requires --out-dir")
        mccv.aggregate_and_report(parser_args.out_dir)
    elif parser_args.seed is not None:
        if not parser_args.out_dir:
            raise SystemExit("--seed requires --out-dir (the shared run folder)")
        run_seed_worker(parser_args.seed, parser_args.out_dir, parser_args)
    else:
        # Default: orchestrate one process per seed.
        seeds = parser_args.master_seeds or MASTER_SEEDS
        orchestrate(seeds, parser_args)


if __name__ == "__main__":
    main()
