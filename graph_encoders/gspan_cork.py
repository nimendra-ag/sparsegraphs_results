"""gSpan + CORK frequent-subgraph encoder.

Same contract as `graph_encoders.wl.WL` and `graph_encoders.wl_edge.EdgeWL` — a
`GraphEncoder` subclass with `generate_training_embeddings(graphs, labels)` /
`generate_inferencing_embeddings(graphs)` and a self-contained save()/load()
bundle format — but the features are *frequent subgraph patterns* rather than
refined node/edge labels, and the feature selection is CORK's correspondence
criterion rather than the adaptive energy/elbow cut on a discriminative-score
curve.

Pipeline (unchanged from the original implementation):
    1. gSpan mines frequent subgraphs from the training graphs
    2. Binary indicator matrix: X[i, j] = 1 iff graph i contains subgraph j
    3. CORK greedily selects the discriminative subset of subgraph features
    4. The reduced binary matrix is returned for downstream use (FDDL, AKSVD,
       SVM, ...)

At inference time subgraph isomorphism is checked against the CORK-selected
patterns only.

Differences from WL/EdgeWL that callers should know about:

  * The embedding is a BINARY indicator matrix (int8), not an L2-normalised
    count vector. That is what CORK's quality criterion is defined over, so it
    is left exactly as the algorithm produces it.
  * There is no `selection_scores_` score curve — CORK selects greedily by
    correspondence reduction, not by ranking a per-feature score — so the
    elbow analytics plot in utils/export.py is skipped for this arm, and
    `n_scored_features` in an MC-CV manifest is None.
  * `_check_graph` (the Karate-Club integrity pass that adds a self-loop to
    every node) is deliberately NOT applied: gSpan's DFS-code canonical form
    has no notion of self-loops, and injecting them would change what is mined.

Cost
----
This arm is far more expensive than WL, and in two distinct places:

  * MINING — gSpan is a recursive, pointer-chasing search in pure Python. It is
    not parallelised here and dominates whenever `min_support_ratio` is low or
    `max_num_vertices` high.
  * ENCODING — every unseen graph is tested against every selected pattern with
    VF2 subgraph isomorphism. That part IS parallelised (`n_jobs`) and guarded
    by a cheap label-multiset prefilter that rejects impossible (graph, pattern)
    pairs before VF2 is entered.

Selection (CORK) used to be a third hotspot; it now runs batched on the GPU by
default — see `graph_encoders.cork`. None of these changes alter which patterns
are mined or selected.

References:
    - gSpan: Yan & Han, ICDM 2002
    - CORK:  Thoma et al., SDM 2009
"""

import os
import json
from collections import Counter

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from joblib import Parallel, delayed, effective_n_jobs

from graph_encoders.graph_encoder import GraphEncoder
from graph_encoders.gspan import GSpan
from graph_encoders.cork import CORK


# Spawning a worker pool costs a fixed ~0.5s (loky reuses it across later calls
# in the same process, but the first encode pays it). Below these thresholds the
# VF2 work saved does not repay that, so encoding stays in-process regardless of
# `n_jobs`. Work is estimated as n_graphs * n_patterns, i.e. the number of
# (graph, pattern) tests, since the pattern count varies hugely by run.
_PARALLEL_MIN_GRAPHS = 256
_PARALLEL_MIN_TESTS = 20_000


def _vf2_is_subgraph(pattern, graph, node_attr, edge_attr):
    """
    Check whether *pattern* is a subgraph of *graph* using VF2
    via NetworkX.

    Both graphs are expected to carry vertex labels under ``node_attr``
    and (optionally) edge labels under ``edge_attr``.

    Module-level (rather than a method) so the parallel encoder can hand it to
    worker processes without pickling the whole encoder.
    """
    if pattern.number_of_nodes() > graph.number_of_nodes():
        return False
    if pattern.number_of_edges() > graph.number_of_edges():
        return False

    # Label-matching functions
    def node_match(n1, n2):
        return n1.get(node_attr) == n2.get(node_attr)

    if edge_attr:
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


def _label_counts(graph, node_attr):
    """Multiset of node labels, read exactly as `node_match` reads them."""
    return Counter(data.get(node_attr) for _, data in graph.nodes(data=True))


def _encode_graphs(graphs, patterns, pattern_label_counts, node_attr, edge_attr,
                   progress_every=0, offset=0, total=None):
    """Binary indicator block for `graphs` against `patterns`.

    The label-multiset test in front of VF2 is a *necessary* condition for
    subgraph isomorphism under `node_match`: the mapping is injective and
    label-preserving, so the pattern cannot need more nodes of any label than
    the graph has. It therefore only skips calls that would have returned
    False — the result is identical to calling VF2 unconditionally, which is
    what makes it safe to add.
    """
    X = np.zeros((len(graphs), len(patterns)), dtype=np.int8)

    for i, g in enumerate(graphs):
        g_counts = _label_counts(g, node_attr)

        for j, pattern in enumerate(patterns):
            p_counts = pattern_label_counts[j]
            if any(g_counts[label] < count for label, count in p_counts.items()):
                continue
            if _vf2_is_subgraph(pattern, g, node_attr, edge_attr):
                X[i, j] = 1

        if progress_every and (offset + i + 1) % progress_every == 0:
            print(f"  encoded {offset + i + 1}/{total} graphs")

    return X


class GSpanCORK(GraphEncoder):
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
    cork_backend : {"auto", "torch", "python"}
        How CORK evaluates its greedy scan. "auto" (default) batches it on the
        GPU when PyTorch is available; "python" forces the reference set-based
        implementation. See `graph_encoders.cork`.
    node_label_attr : str
        Node attribute name in your NetworkX graphs.
    edge_label_attr : str or None
        Edge attribute name.  None → unlabelled edges.
    n_jobs : int
        Worker processes this encoder may use. -1 (default) uses every core;
        1 keeps everything in-process. Applies to both parallelisable phases —
        gSpan's frequent 1-edge roots and the VF2 encoding of unseen graphs —
        neither of which changes its output as a result. CORK selection is
        unaffected (it batches on the GPU instead). Lower this if mining runs
        out of memory: each worker holds its own copy of the graph database and
        of the root it is mining.
    cache_inference : bool
        Memoise the most recent `generate_inferencing_embeddings` result and
        return it when called again with the *same* list object. The MC-CV
        harness encodes the test split twice per seed — once for the sparse
        codes, once for the SRC arms (utils/mccv.py) — and for this encoder
        that second pass is a full VF2 sweep. Set False to disable.
    verbose : bool
        Print progress from gSpan and CORK.
    seed : int
        Kept for interface parity with WL/EdgeWL (the MC-CV harness constructs
        every encoder as `factory(seed)`). gSpan and CORK are deterministic, so
        this only reseeds the global RNGs.

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
        cork_backend="auto",
        node_label_attr="feature",
        edge_label_attr=None,
        n_jobs=-1,
        cache_inference=True,
        verbose=True,
        seed=42,
    ):
        super().__init__(name="CorrespondenceAwareGSpanCORK")

        self.seed = seed
        self.min_support_ratio = min_support_ratio
        self.max_num_vertices = max_num_vertices
        self.min_num_vertices = min_num_vertices
        self.cork_tolerance = cork_tolerance
        self.cork_max_features = cork_max_features
        self.cork_backend = cork_backend
        self.node_label_attr = node_label_attr
        self.edge_label_attr = edge_label_attr
        self.n_jobs = n_jobs
        self.cache_inference = cache_inference
        self.verbose = verbose

        # Embedding width. Set by create_vocab() once CORK has chosen; declared
        # here so the artifact manifest / MC-CV manifest can read it defensively
        # on an unfitted encoder (mirrors WL.n_vocab).
        self.n_vocab = 0

        # Populated during training
        self._gspan = None
        self._cork = None
        self._selected_subgraphs = []  # the CORK-selected patterns (metadata dicts)
        self._all_subgraphs_meta = []  # full gSpan output before CORK selection

        # Derived caches, both invalidated whenever the vocabulary changes.
        self._label_counts_cache = None  # per-pattern node-label multisets
        self._infer_cache = None         # (graphs list, embedding) of size 1

    # ------------------------------------------------------------------
    # Step 1: frequent subgraph mining (gSpan)
    # ------------------------------------------------------------------

    def mine_subgraphs(self, graphs):
        """Run gSpan over `graphs` and return its frequent patterns as metadata.

        The support threshold is absolute, derived from `min_support_ratio` and
        the size of *this* graph set — so an MC-CV resample and the single-split
        export both mine at the same relative support.

        Returns
        -------
        list of dict
            One entry per frequent subgraph, as produced by
            ``GSpan.get_frequent_subgraphs_as_nx()``: ``graph`` (nx.Graph),
            ``support``, ``graph_ids``, ``dfs_code``, ``num_vertices``,
            ``num_edges``.
        """
        n_graphs = len(graphs)

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
            n_jobs=self.n_jobs,
        )
        self._gspan.run(
            graphs,
            node_label_attr=self.node_label_attr,
            edge_label_attr=self.edge_label_attr,
        )

        subgraphs_meta = self._gspan.get_frequent_subgraphs_as_nx()

        if self.verbose:
            print(f"[GSpanCORK] gSpan found {len(subgraphs_meta)} frequent subgraphs")

        if len(subgraphs_meta) == 0:
            raise RuntimeError(
                "gSpan found 0 frequent subgraphs. Try lowering "
                "min_support_ratio or increasing max_num_vertices."
            )

        return subgraphs_meta

    # ------------------------------------------------------------------
    # Step 2: full binary indicator matrix
    # ------------------------------------------------------------------

    def build_indicator_matrix(self, n_graphs, subgraphs_meta):
        """Binary occurrence matrix over ALL mined patterns (pre-CORK).

        gSpan already records which graph IDs contain each pattern, so no
        subgraph isomorphism is needed on the training path — that cost is only
        paid at inference (`calc_coefficients`).
        """
        n_patterns = len(subgraphs_meta)
        X_full = np.zeros((n_graphs, n_patterns), dtype=np.int8)

        for j, meta in enumerate(subgraphs_meta):
            for gid in meta["graph_ids"]:
                X_full[gid, j] = 1

        if self.verbose:
            density = X_full.sum() / X_full.size
            print(f"[GSpanCORK] Indicator matrix: {X_full.shape}, "
                  f"density={density:.4f}")

        return X_full

    # ------------------------------------------------------------------
    # Step 3: CORK feature selection — the "vocabulary"
    # ------------------------------------------------------------------

    def create_vocab(self, X_full, labels):
        """Select the discriminative pattern subset and return the reduced matrix.

        The analogue of `WL.create_vocab`: it fixes both *which* patterns become
        features and, through `self._cork.selected_indices_`, their column order.
        That order IS the embedding column order and must be preserved on disk.

        Unlike WL/EdgeWL there is no score curve to cut — CORK greedily picks
        whichever remaining pattern eliminates the most cross-class
        correspondences, stopping at `cork_tolerance` / `cork_max_features`.
        """
        self._cork = CORK(
            tolerance=self.cork_tolerance,
            max_features=self.cork_max_features,
            verbose=self.verbose,
            backend=self.cork_backend,
        )
        X_selected = self._cork.fit_transform(X_full, labels)

        # Store the selected subgraph patterns for inference
        self._set_selected_subgraphs([
            self._all_subgraphs_meta[i] for i in self._cork.selected_indices_
        ])

        if self.verbose:
            print(f"[GSpanCORK] CORK selected {len(self._selected_subgraphs)} "
                  f"discriminative subgraphs")

        return X_selected

    def _set_selected_subgraphs(self, metas):
        """Install a new vocabulary and drop everything derived from the old one."""
        self._selected_subgraphs = metas
        self.n_vocab = len(metas)
        self._label_counts_cache = None
        self._infer_cache = None

    # ------------------------------------------------------------------
    # Vectorisation of unseen graphs
    # ------------------------------------------------------------------

    def _pattern_label_counts(self):
        """Node-label multiset per selected pattern (computed once, reused)."""
        if self._label_counts_cache is None:
            self._label_counts_cache = [
                _label_counts(meta["graph"], self.node_label_attr)
                for meta in self._selected_subgraphs
            ]
        return self._label_counts_cache

    def _resolve_n_jobs(self, n_graphs, n_patterns):
        """Workers to use for this encode, after the small-input cutoff."""
        if self.n_jobs == 1:
            return 1
        if n_graphs < _PARALLEL_MIN_GRAPHS:
            return 1
        if n_graphs * n_patterns < _PARALLEL_MIN_TESTS:
            return 1
        return max(1, effective_n_jobs(self.n_jobs))

    def calc_coefficients(self, graphs):
        """Binary indicator matrix of `graphs` against the selected vocabulary.

        The counterpart of `WL.calc_coefficients`, but there is no count/vocab
        lookup to do: membership must be decided by subgraph isomorphism. This
        is the expensive step at inference time, so it is split across `n_jobs`
        worker processes — the per-graph tests are fully independent, and the
        blocks are re-stacked in input order, so the result does not depend on
        how the work was divided.
        """
        if not self._selected_subgraphs:
            raise RuntimeError(
                "No selected subgraphs. Call generate_training_embeddings first."
            )

        n_graphs = len(graphs)
        n_features = len(self._selected_subgraphs)
        patterns = [meta["graph"] for meta in self._selected_subgraphs]
        label_counts = self._pattern_label_counts()
        n_jobs = self._resolve_n_jobs(n_graphs, n_features)

        if self.verbose:
            print(f"[GSpanCORK] Encoding {n_graphs} graphs against "
                  f"{n_features} selected subgraph patterns"
                  f" (n_jobs={n_jobs})...")

        if n_jobs == 1:
            X = _encode_graphs(
                graphs, patterns, label_counts,
                self.node_label_attr, self.edge_label_attr,
                progress_every=500 if self.verbose else 0,
                offset=0, total=n_graphs,
            )
        else:
            # Two chunks per worker: enough to even out graphs of wildly
            # different sizes, without re-pickling the pattern set into more
            # tasks than that buys. Measured best of {1, 2, 4, 8} on a
            # 2000-graph x 11-pattern encode (7.7x vs 4.6x at one chunk each).
            bounds = np.array_split(np.arange(n_graphs), min(n_graphs, n_jobs * 2))
            blocks = Parallel(n_jobs=n_jobs)(
                delayed(_encode_graphs)(
                    [graphs[i] for i in idx], patterns, label_counts,
                    self.node_label_attr, self.edge_label_attr,
                )
                for idx in bounds if len(idx)
            )
            X = np.vstack(blocks)

        if self.verbose:
            density = X.sum() / X.size
            print(f"[GSpanCORK] Inference matrix: {X.shape}, "
                  f"density={density:.4f}")

        return X

    # ------------------------------------------------------------------
    # GraphEncoder contract
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
        self._set_seed()
        labels = np.asarray(labels)

        self._all_subgraphs_meta = self.mine_subgraphs(graphs)
        X_full = self.build_indicator_matrix(len(graphs), self._all_subgraphs_meta)
        X_selected = self.create_vocab(X_full, labels)

        return X_selected

    def generate_inferencing_embeddings(self, graphs):
        """
        Encode unseen graphs using the CORK-selected subgraph vocabulary.

        Parameters
        ----------
        graphs : list of nx.Graph

        Returns
        -------
        np.ndarray, shape (n_graphs, n_selected_features)
        """
        self._set_seed()

        # Identity, not equality: cheap, and exact because the cache holds a
        # reference to the list, so its id() cannot be recycled while cached.
        if self._infer_cache is not None and self._infer_cache[0] is graphs:
            if self.verbose:
                print(f"[GSpanCORK] Reusing cached encoding for these "
                      f"{len(graphs)} graphs (skipping the VF2 sweep)")
            return self._infer_cache[1].copy()

        X = self.calc_coefficients(graphs)
        if self.cache_inference:
            self._infer_cache = (graphs, X)
        return X

    # ------------------------------------------------------------------
    # Subgraph isomorphism check
    # ------------------------------------------------------------------

    def _is_subgraph(self, pattern, graph):
        """Instance-level view of `_vf2_is_subgraph`, bound to this encoder's
        label attributes. Kept so the check remains reachable as a method."""
        return _vf2_is_subgraph(
            pattern, graph, self.node_label_attr, self.edge_label_attr
        )

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

    # --- Persistence ---------------------------------------------------------
    # Mirrors WL/EdgeWL: the only *learned* state is the ordered vocabulary —
    # here the CORK-selected subgraph patterns rather than a list of hashes. The
    # pattern order IS the embedding column order, so it must be preserved
    # exactly on disk.
    #
    # Patterns are written as plain node/edge lists rather than via
    # networkx.node_link_data so the bundle format does not move with NetworkX
    # releases. `graph_ids` (which TRAINING graphs contained the pattern) is not
    # persisted: those indices refer to a split that no longer exists at serving
    # time. `support` is kept because it is the useful summary of the same fact.
    #
    # Unlike EdgeWL there is no PYTHONHASHSEED caveat — a pattern is matched by
    # VF2 subgraph isomorphism on its labels, not by a salted hash.
    _CONFIG_FILE = "gspan_cork_config.json"
    _VOCAB_FILE = "gspan_cork_vocab.json"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "min_support_ratio": self.min_support_ratio,
            "max_num_vertices": self.max_num_vertices,
            "min_num_vertices": self.min_num_vertices,
            "cork_tolerance": self.cork_tolerance,
            "cork_max_features": self.cork_max_features,
            "cork_backend": self.cork_backend,
            "node_label_attr": self.node_label_attr,
            "edge_label_attr": self.edge_label_attr,
            "n_jobs": self.n_jobs,
            "cache_inference": self.cache_inference,
            "verbose": self.verbose,
            "n_vocab": self.n_vocab,
            "seed": self.seed,
            # Provenance: which CORK path actually produced this vocabulary.
            "cork_backend_used": getattr(self._cork, "backend_used_", None),
        }

    def _pattern_to_dict(self, meta):
        """One selected pattern -> a JSON-safe dict (node/edge lists + metadata)."""
        g = meta["graph"]
        return {
            "nodes": [[n, data.get(self.node_label_attr)]
                      for n, data in g.nodes(data=True)],
            # Edge labels live under 'label' (see GSpan.get_frequent_subgraphs_as_nx),
            # independently of node_label_attr/edge_label_attr.
            "edges": [[u, v, data.get("label")] for u, v, data in g.edges(data=True)],
            "support": int(meta["support"]),
            "dfs_code": meta["dfs_code"],
            "num_vertices": int(meta["num_vertices"]),
            "num_edges": int(meta["num_edges"]),
        }

    def _pattern_from_dict(self, entry):
        """Inverse of `_pattern_to_dict`: rebuild the metadata dict + nx.Graph."""
        g = nx.Graph()
        for node, label in entry["nodes"]:
            g.add_node(node, **{self.node_label_attr: label})
        for u, v, label in entry["edges"]:
            g.add_edge(u, v, label=label)
        return {
            "graph": g,
            "support": entry["support"],
            # Not persisted (training-split row indices); kept as an empty set so
            # the metadata dict keeps one shape whether fitted or loaded.
            "graph_ids": set(),
            "dfs_code": entry["dfs_code"],
            "num_vertices": entry["num_vertices"],
            "num_edges": entry["num_edges"],
        }

    def save(self, dirpath: str) -> None:
        if not self._selected_subgraphs:
            raise ValueError(
                "GSpanCORK has no selected subgraphs to save; fit the encoder first."
            )
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # Preserve order — it is the embedding column order.
        vocab_serialisable = [
            self._pattern_to_dict(meta) for meta in self._selected_subgraphs
        ]
        with open(os.path.join(dirpath, self._VOCAB_FILE), "w", encoding="utf-8") as f:
            json.dump(vocab_serialisable, f)

    @classmethod
    def load(cls, dirpath: str) -> "GSpanCORK":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)

        encoder = cls(
            min_support_ratio=config["min_support_ratio"],
            max_num_vertices=config["max_num_vertices"],
            min_num_vertices=config["min_num_vertices"],
            cork_tolerance=config["cork_tolerance"],
            cork_max_features=config["cork_max_features"],
            # .get keeps bundles written before these knobs existed loadable at
            # their current defaults; none of them affect which patterns the
            # saved vocabulary contains, only how encoding is executed.
            cork_backend=config.get("cork_backend", "auto"),
            node_label_attr=config["node_label_attr"],
            edge_label_attr=config["edge_label_attr"],
            n_jobs=config.get("n_jobs", -1),
            cache_inference=config.get("cache_inference", True),
            verbose=config.get("verbose", True),
            seed=config["seed"],
        )

        with open(os.path.join(dirpath, cls._VOCAB_FILE), encoding="utf-8") as f:
            entries = json.load(f)
        # The saved vocabulary length is authoritative for the embedding width.
        encoder._set_selected_subgraphs([
            encoder._pattern_from_dict(entry) for entry in entries
        ])
        return encoder
