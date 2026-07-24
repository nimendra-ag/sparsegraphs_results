"""
Evaluate karateclub's Graph2Vec on the NCI1 dataset.

- 70/30 stratified train/test split
- Evaluation via the project's Evaluator class (LinearSVC, LR, RF, GB)
- Embedding dimension = 1024
"""

import argparse
import inspect
import logging
import os
import sys
import time

import networkx as nx
import numpy as np
from graph2vec import Graph2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_graph2vec")


def _supports_node_attribute() -> bool:
    """Check if installed karateclub version supports use_node_attribute."""
    sig = inspect.signature(Graph2Vec.__init__)
    return "use_node_attribute" in sig.parameters


def _encode_node_features(graphs: list[nx.Graph], attr: str = "feature") -> list[nx.Graph]:
    """
    Fallback for older karateclub versions that lack `use_node_attribute`.
    Encodes string node features into integer 'label' attributes.
    """
    label_map: dict[str, int] = {}
    processed = []

    for g in graphs:
        g_copy = g.copy()
        for node in g_copy.nodes():
            feat = str(g_copy.nodes[node].get(attr, g_copy.degree(node)))
            if feat not in label_map:
                label_map[feat] = len(label_map)
            g_copy.nodes[node]["label"] = label_map[feat]
        processed.append(g_copy)

    logger.info("Encoded %d distinct node features into integer labels", len(label_map))
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="graph2vec on NCI1")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension (δ)")
    parser.add_argument("--wl-depth", type=int, default=2, help="WL iteration depth (D)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (ε)")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate (α)")
    parser.add_argument("--min-count", type=int, default=1, help="Min subgraph frequency")
    parser.add_argument("--workers", type=int, default=4, help="Gensim worker threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ---- Load NCI1 -------------------------------------------------------
    logger.info("Loading NCI1 dataset…")
    loader = GraphDataLoader()
    graphs, labels = loader.nci_full_graphs, loader.nci_full_labels
    labels = np.array(labels)
    logger.info("Loaded %d graphs, %d classes", len(graphs), len(set(labels)))

    # ---- Handle node features across karateclub versions -----------------
    use_attr = _supports_node_attribute()

    if use_attr:
        logger.info("karateclub supports use_node_attribute — using 'feature' directly")
        model = Graph2Vec(
            wl_iterations=args.wl_depth,
            use_node_attribute="feature",
            dimensions=args.dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )
    else:
        logger.info(
            "karateclub lacks use_node_attribute — encoding features into 'label' attr"
        )
        graphs = _encode_node_features(graphs, attr="feature")
        model = Graph2Vec(
            wl_iterations=args.wl_depth,
            attributed=True,
            dimensions=args.dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )

    # ---- Train graph2vec -------------------------------------------------
    logger.info(
        "Training graph2vec — dim=%d, wl_depth=%d, epochs=%d",
        args.dim, args.wl_depth, args.epochs,
    )
    t0 = time.perf_counter()
    model.fit(graphs)
    pre_training_time = time.perf_counter() - t0
    logger.info("Pre-training completed in %.1f s", pre_training_time)

    embeddings = model.get_embedding()
    logger.info("Embedding shape: %s", embeddings.shape)

    # ---- 70/30 stratified split ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=args.seed, stratify=labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ---- Evaluate using project Evaluator --------------------------------
    os.makedirs("./cm", exist_ok=True)

    evaluator = Evaluator(X_train, y_train, X_test, y_test, random_state=args.seed)

    results = {}
    results["Linear SVM"] = evaluator.predict_svm()
    results["Logistic Regression"] = evaluator.predict_logistic_regression()
    results["Random Forest"] = evaluator.predict_random_forest()
    results["Gradient Boosting"] = evaluator.predict_gradient_boosting()

    # ---- Summary ---------------------------------------------------------
    logger.info("=" * 60)
    logger.info("NCI1 — graph2vec results (dim=%d, wl=%d, epochs=%d)",
                args.dim, args.wl_depth, args.epochs)
    logger.info("-" * 60)
    logger.info("%-22s %8s %8s %8s %8s %8s",
                "Model", "Prec", "Rec", "F1", "ROC-AUC", "PR-AUC")
    logger.info("-" * 60)
    for model_name, metrics in results.items():
        logger.info(
            "%-22s %8.4f %8.4f %8.4f %8.4f %8.4f",
            model_name,
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1-Score"],
            metrics["ROC-AUC"],
            metrics["PR-AUC"],
        )
    logger.info("-" * 60)
    logger.info("Pre-training time: %.1f s", pre_training_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()