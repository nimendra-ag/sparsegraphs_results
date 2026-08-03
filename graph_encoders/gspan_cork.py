"""
GSpan-CORK Graph Encoder
=========================

Combines gSpan frequent subgraph mining with CORK feature selection
to produce binary indicator vectors for graph classification.

Designed as a drop-in replacement for the WL encoder in the
dictionary-learning pipeline (WL_AKSVD).

Pipeline:
    1. gSpan mines frequent subgraphs from training graphs
    2. Binary indicator matrix: X[i,j] = 1 iff graph i contains subgraph j
    3. CORK selects the discriminative subset of subgraph features
    4. Reduced binary matrix is returned for downstream use (AKSVD, SVM, etc.)

At inference time, subgraph isomorphism is checked against the
CORK-selected subgraph patterns only.

References:
    - gSpan: Yan & Han, ICDM 2002
    - CORK:  Thoma et al., SDM 2009
"""

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from graph_encoders.gspan import GSpan
from graph_encoders.cork import CORK


class GSpanCORK:
    """
    Graph encoder using gSpan + CORK.

    Parameters
    ----------
    min_support_ratio : float
        Minimum support as a fraction of the training set size.
        E.g. 0.10 means a subgraph must appear in >= 10 % of graphs.
    max_num_vertices : int
        Maximum number of atoms (vertices) in mined subgraph patterns.
    min_num_vertices : int
        Minimum pattern size to keep (default 2; single-edge patterns
        are rarely discriminative enough on their own).
    cork_tolerance : int
        CORK's early-stopping tolerance on remaining correspondences.
    cork_max_features : int or None
        Hard cap on CORK-selected features.
    node_label_attr : str
        Node attribute name in your NetworkX graphs.
    edge_label_attr : str or None
        Edge attribute name.  None → unlabelled edges.
    verbose : bool
        Print progress from gSpan and CORK.

    Usage
    -----
        encoder = GSpanCORK(min_support_ratio=0.10, max_num_vertices=10)
        X_train = encoder.generate_training_embeddings(G_train, y_train)
        X_test  = encoder.generate_inferencing_embeddings(G_test)
    """

    def __init__(
        self,
        min_support_ratio=0.10,
        max_num_vertices=10,
        min_num_vertices=2,
        cork_tolerance=0,
        cork_max_features=None,
        node_label_attr="feature",
        edge_label_attr=None,
        verbose=True,
    ):
        self.min_support_ratio = min_support_ratio
        self.max_num_vertices = max_num_vertices
        self.min_num_vertices = min_num_vertices
        self.cork_tolerance = cork_tolerance
        self.cork_max_features = cork_max_features
        self.node_label_attr = node_label_attr
        self.edge_label_attr = edge_label_attr
        self.verbose = verbose

        # Populated during training
        self._gspan = None
        self._cork = None
        self._selected_subgraphs = []  # list of nx.Graph (the CORK-selected patterns)
        self._all_subgraphs_meta = []  # full gSpan output before CORK selection

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def generate_training_embeddings(self, graphs, labels):
        """
        Mine subgraphs, select via CORK, return binary indicator matrix.

        Parameters
        ----------
        graphs : list of nx.Graph
            Training graphs with node attribute ``node_label_attr``.
        labels : array-like, shape (n_graphs,)
            Binary class labels (e.g. +1 / -1).

        Returns
        -------
        np.ndarray, shape (n_graphs, n_selected_features)
            Binary indicator matrix restricted to CORK-selected subgraphs.
        """
        labels = np.asarray(labels)
        n_graphs = len(graphs)

        # --- Step 1: gSpan ---
        min_support = max(1, int(self.min_support_ratio * n_graphs))
        if self.verbose:
            print(f"[GSpanCORK] Running gSpan: {n_graphs} graphs, "
                  f"min_support={min_support} ({self.min_support_ratio:.0%}), "
                  f"max_vertices={self.max_num_vertices}")

        self._gspan = GSpan(
            min_support=min_support,
            min_num_vertices=self.min_num_vertices,
            max_num_vertices=self.max_num_vertices,
            verbose=self.verbose,
        )
        self._gspan.run(
            graphs,
            node_label_attr=self.node_label_attr,
            edge_label_attr=self.edge_label_attr,
        )

        self._all_subgraphs_meta = self._gspan.get_frequent_subgraphs_as_nx()
        n_patterns = len(self._all_subgraphs_meta)

        if self.verbose:
            print(f"[GSpanCORK] gSpan found {n_patterns} frequent subgraphs")

        if n_patterns == 0:
            raise RuntimeError(
                "gSpan found 0 frequent subgraphs. Try lowering "
                "min_support_ratio or increasing max_num_vertices."
            )

        # --- Step 2: Build full binary indicator matrix ---
        # gSpan already tells us which graph IDs contain each pattern,
        # so we don't need subgraph isomorphism here.
        X_full = np.zeros((n_graphs, n_patterns), dtype=np.int8)

        for j, meta in enumerate(self._all_subgraphs_meta):
            for gid in meta["graph_ids"]:
                X_full[gid, j] = 1

        if self.verbose:
            density = X_full.sum() / X_full.size
            print(f"[GSpanCORK] Indicator matrix: {X_full.shape}, "
                  f"density={density:.4f}")

        # --- Step 3: CORK feature selection ---
        self._cork = CORK(
            tolerance=self.cork_tolerance,
            max_features=self.cork_max_features,
            verbose=self.verbose,
        )
        X_selected = self._cork.fit_transform(X_full, labels)

        # Store the selected subgraph patterns for inference
        self._selected_subgraphs = [
            self._all_subgraphs_meta[i] for i in self._cork.selected_indices_
        ]

        if self.verbose:
            print(f"[GSpanCORK] CORK selected {len(self._selected_subgraphs)} "
                  f"discriminative subgraphs")

        return X_selected

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_inferencing_embeddings(self, graphs):
        """
        Encode unseen graphs using the CORK-selected subgraph vocabulary.

        For each graph, checks subgraph isomorphism against every selected
        pattern.  This is the expensive step at inference time, but the
        number of patterns is small (typically 15–66 per the paper).

        Parameters
        ----------
        graphs : list of nx.Graph

        Returns
        -------
        np.ndarray, shape (n_graphs, n_selected_features)
        """
        if not self._selected_subgraphs:
            raise RuntimeError(
                "No selected subgraphs. Call generate_training_embeddings first."
            )

        n_graphs = len(graphs)
        n_features = len(self._selected_subgraphs)
        X = np.zeros((n_graphs, n_features), dtype=np.int8)

        if self.verbose:
            print(f"[GSpanCORK] Encoding {n_graphs} graphs against "
                  f"{n_features} selected subgraph patterns...")

        for i, g in enumerate(graphs):
            for j, meta in enumerate(self._selected_subgraphs):
                pattern = meta["graph"]
                if self._is_subgraph(pattern, g):
                    X[i, j] = 1

            if self.verbose and (i + 1) % 500 == 0:
                print(f"  encoded {i + 1}/{n_graphs} graphs")

        if self.verbose:
            density = X.sum() / X.size
            print(f"[GSpanCORK] Inference matrix: {X.shape}, "
                  f"density={density:.4f}")

        return X

    # ------------------------------------------------------------------
    # Subgraph isomorphism check
    # ------------------------------------------------------------------

    def _is_subgraph(self, pattern, graph):
        """
        Check whether *pattern* is a subgraph of *graph* using VF2
        via NetworkX.

        Both graphs are expected to carry vertex labels under
        ``self.node_label_attr`` and (optionally) edge labels under
        ``self.edge_label_attr``.
        """
        if pattern.number_of_nodes() > graph.number_of_nodes():
            return False
        if pattern.number_of_edges() > graph.number_of_edges():
            return False

        # Label-matching functions
        node_attr = self.node_label_attr

        def node_match(n1, n2):
            return n1.get(node_attr) == n2.get(node_attr)

        if self.edge_label_attr:
            edge_attr = self.edge_label_attr

            def edge_match(e1, e2):
                return e1.get(edge_attr) == e2.get(edge_attr)
        else:
            edge_match = None

        matcher = isomorphism.GraphMatcher(
            graph, pattern,
            node_match=node_match,
            edge_match=edge_match,
        )
        return matcher.subgraph_is_isomorphic()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_subgraphs(self):
        """Return metadata dicts for the CORK-selected subgraphs."""
        return list(self._selected_subgraphs)

    def get_all_subgraphs(self):
        """Return metadata dicts for all gSpan-mined subgraphs (pre-CORK)."""
        return list(self._all_subgraphs_meta)

    def get_correspondence_trace(self):
        """Return the list of remaining correspondences per CORK iteration."""
        if self._cork is None:
            return []
        return list(self._cork.correspondence_trace_)
