"""
Graph classification on NCI (binary) with a GCN, faithful to the original
Kipf & Welling backbone, with a full metric report.

Pipeline:
  1. Load molecules from an SDF with RDKit -> NetworkX graphs + integer labels.
  2. Convert each NetworkX graph to a PyG Data object:
       - node features  : one-hot of the atom symbol over a FIXED 45-element
                           vocabulary (shared across all graphs) -> in_dim = 45
       - edge_index     : bonds, made bidirectional for the undirected graph
       - y              : binary class index (raw labels mapped to {0, 1})
  3. Train a GCN: two GCNConv layers (original backbone) + global mean-pool
     readout + linear classifier.
  4. Report accuracy, precision, recall, F1, ROC-AUC and PR-AUC on BOTH the
     train and test splits.

Metric notes:
  * precision / recall / F1 / PR-AUC are reported for the POSITIVE class
    (class index 1, i.e. the larger raw label such as +1).
  * ROC-AUC and PR-AUC use the predicted probability P(class = 1), not the hard
    prediction; PR-AUC here is sklearn's average_precision_score.

Architecture / hyperparameters mirror the original GCN paper:
  hidden = 16, dropout = 0.5, Adam lr = 0.01, weight decay 5e-4 (first layer).

Extra dependencies: rdkit, networkx, scikit-learn.
    pip install rdkit networkx scikit-learn
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


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


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC metrics require both classes to be present in y_true.
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def report(name: str, model, loader, device) -> None:
    y_true, y_pred, y_prob = predict(model, loader, device)
    m = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n{name} set ({len(y_true)} graphs):")
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall   : {m['recall']:.4f}")
    print(f"  f1       : {m['f1']:.4f}")
    print(f"  roc_auc  : {m['roc_auc']:.4f}")
    print(f"  pr_auc   : {m['pr_auc']:.4f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCN graph classification on NCI.")
    p.add_argument("--sdf", default="datasets/NCI_full/1total-connect.sdf")
    p.add_argument("--hidden", type=int, default=32) # 16, 32 64
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, labels = load_nci_sdf(args.sdf)
    dataset = build_dataset(graphs, labels)

    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(dataset), generator=gen).tolist()
    dataset = [dataset[i] for i in perm]
    n_train = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:n_train], dataset[n_train:]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

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

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        if epoch == 1 or epoch % 10 == 0:
            y_true, y_pred, _ = predict(model, test_loader, device)
            test_acc = accuracy_score(y_true, y_pred)
            print(f"Epoch {epoch:03d} | train_loss={loss:.4f} | test_acc={test_acc:.4f}")

    # Final metric report on both splits.
    report("Train", model, train_loader, device)
    report("Test", model, test_loader, device)


if __name__ == "__main__":
    main()