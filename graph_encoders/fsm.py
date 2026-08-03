"""Frequent-Shape-Motif (FSM) ego-network encoder.

Same contract as `graph_encoders.wl.WL` and `graph_encoders.wl_edge.EdgeWL` — a
`GraphEncoder` subclass constructed as `FSM(seed=...)`, with
`generate_training_embeddings(graphs, labels)` /
`generate_inferencing_embeddings(graphs)` and a self-contained save()/load()
bundle format — but the features are ego-network *shape signatures* rather than
refined node/edge labels: for every node, the radius-`r` neighbourhood
summarised by its node count, edge count, sorted node-label multiset and sorted
degree sequence.

Only the feature *extraction* and the vocabulary trim differ from WL. Everything
downstream (zero-row filtering, dictionary learning, scaling, evaluation, bundle
layout) is shared code, so an FSM run and a WL run differ in exactly one place
and their numbers are comparable.

Differences from WL/EdgeWL that callers should know about:

  * SELECTION — the vocabulary is trimmed on a fixed budget: signatures seen
    fewer than `min_count` times globally are dropped, the rest are ranked by
    global frequency and the top `n_vocab` are kept. It is unsupervised, so
    `labels` is accepted (for interface parity with WL/EdgeWL, which the MC-CV
    and export harnesses call positionally) but not used.
  * There is no `selection_scores_` score curve — nothing is scored on a
    discriminative criterion — so the elbow analytics plot in utils/export.py
    is skipped for this arm, and `n_scored_features` in an MC-CV manifest is
    None.
  * The embedding is a RAW COUNT matrix, not the L2-normalised one WL/EdgeWL
    produce. That is what the FSM formulation is defined over, so it is left
    exactly as the algorithm produces it.
  * `_check_graph` (the Karate-Club integrity pass that adds a self-loop to
    every node) is deliberately NOT applied: a self-loop would raise every
    node's degree and inflate the ego-graph edge count, changing every
    signature this encoder produces.

Signatures are plain ASCII strings (no builtin hash()), so unlike EdgeWL this
arm carries no PYTHONHASHSEED caveat.
"""

import os
import json
from collections import Counter

import numpy as np
import networkx as nx

from graph_encoders.graph_encoder import GraphEncoder


class FSM(GraphEncoder):
    def __init__(
            self,
            radius: int = 1,
            n_vocab: int = 1000,
            min_count: int = 2,
            node_label_attr: str = "feature",
            seed: int = 42
    ):
        super().__init__(name="FSM")

        # FSM's encoding is deterministic; the seed exists only so this encoder
        # matches the WL/EdgeWL constructor surface (utils/mccv.py and
        # utils/export.py hand every encoder a seed) and so it can be recorded
        # in the saved config.
        self.seed = seed
        self.vocab = None
        self.embeddings = None
        self.radius = radius
        self.n_vocab = n_vocab
        self.min_count = min_count
        # Node attribute holding the discrete node label. Every loader in
        # utils/graph_data.py writes the atom symbol / TU node label under
        # "feature", which is also what WL and GSpanCORK read.
        self.node_label_attr = node_label_attr

    def _get_subgraph_signature(self, sub_g):
        num_nodes = sub_g.number_of_nodes()
        num_edges = sub_g.number_of_edges()

        # Get sorted degree sequence
        degrees = sorted([d for n, d in sub_g.degree()])
        deg_str = ",".join(map(str, degrees))

        # Check if the graph has node labels (like NCI chemical symbols).
        # We grab the first node to check for the label attribute.
        first_node = list(sub_g.nodes())[0]
        if self.node_label_attr in sub_g.nodes[first_node]:
            # Extract all labels in the subgraph and sort them alphabetically
            labels = sorted(
                str(sub_g.nodes[n][self.node_label_attr]) for n in sub_g.nodes()
            )
            label_str = ",".join(labels)

            # The signature now includes the chemical makeup!
            # e.g., N3_E2_L[C,C,O]_D[1,1,2]
            return f"N{num_nodes}_E{num_edges}_L[{label_str}]_D[{deg_str}]"

        # Fallback for unlabelled graphs (like reddit10k)
        return f"N{num_nodes}_E{num_edges}_D[{deg_str}]"

    def extract_subgraphs(self, target_graphs):
        documents = []
        for G in target_graphs:
            doc_words = []
            for node in G.nodes():
                ego_graph = nx.ego_graph(G, node, radius=self.radius, center=True, undirected=True)
                signature = self._get_subgraph_signature(ego_graph)
                doc_words.append(signature)
            documents.append(doc_words)
        return documents

    def create_vocab(self, documents, labels=None):
        """Fixed-budget frequency trim; `labels` is accepted but unused.

        The analogue of `WL.create_vocab`: it fixes both *which* signatures
        become features and, through the sort order, their column order. That
        order IS the embedding column order and must be preserved on disk.

        Unlike WL/EdgeWL there is no discriminative score curve to cut — the
        trim is the fixed `min_count` / `n_vocab` budget below, which is why
        this encoder exposes no `selection_scores_`.
        """
        global_counts = Counter()
        for doc in documents:
            global_counts.update(doc)

        frequent_subgraphs = {word: count for word, count in global_counts.items() if count >= self.min_count}
        sorted_vocab = sorted(frequent_subgraphs.items(), key=lambda item: item[1], reverse=True)
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        print(f"Total Features {len(global_counts)}")
        print(f"selected {len(trimmed_vocab)} from the fixed frequency budget "
              f"(min_count={self.min_count}, n_vocab={self.n_vocab})")

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, documents):
        # Index map for O(1) lookup, exactly as EdgeWL does: iterating the whole
        # vocabulary per document is the same result, only slower.
        vocab_index = {word: idx for idx, (word, _) in enumerate(self.vocab)}

        sparse_vector = np.zeros((len(documents), self.n_vocab))

        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for word, count in words_count.items():
                if word in vocab_index:
                    sparse_vector[i, vocab_index[word]] = count

        # NOTE: raw counts, deliberately NOT L2-normalised (WL/EdgeWL are). Rows
        # with no vocabulary signature stay all-zero; the training path drops
        # them in utils/pipeline._drop_zero_embeddings, the inference path keeps
        # them so every graph still gets a prediction.
        return sparse_vector

    def generate_training_embeddings(self, graphs, labels=None):
        """Extract signatures, build the locked vocabulary, return the Y matrix.

        `labels` is part of the shared encoder contract (utils/pipeline.py calls
        every encoder as `generate_training_embeddings(graphs, y)`); FSM's trim
        is unsupervised, so it is passed through to `create_vocab` and ignored.
        """
        self._set_seed()
        documents = self.extract_subgraphs(graphs)
        self.vocab = self.create_vocab(documents, labels)
        self.embeddings = self.calc_coefficients(documents)
        return self.embeddings

    def generate_inferencing_embeddings(self, graphs):
        """Extract signatures from new data and map them to the EXISTING vocabulary."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call generate_training_embeddings first.")

        self._set_seed()
        documents = self.extract_subgraphs(graphs)
        return self.calc_coefficients(documents)

    # --- Persistence ---------------------------------------------------------
    # Mirrors WL/EdgeWL: the only *learned* state is `vocab` (the ordered,
    # frequency-trimmed shape signatures). Everything else is hyperparameters.
    # The vocab order IS the embedding column order, so it must be preserved
    # exactly on disk. Signatures are plain ASCII strings (no process-salted
    # hashing), so a saved bundle reloads correctly in any interpreter.
    _CONFIG_FILE = "fsm_config.json"
    _VOCAB_FILE = "fsm_vocab.json"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "radius": self.radius,
            "n_vocab": self.n_vocab,
            "min_count": self.min_count,
            "node_label_attr": self.node_label_attr,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self.vocab is None:
            raise ValueError("FSM has no vocab to save; fit the encoder first.")
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # Preserve order; signatures are strings and the stored value is a
        # global frequency count -> JSON safe.
        vocab_serialisable = [[str(sig), int(count)] for sig, count in self.vocab]
        with open(os.path.join(dirpath, self._VOCAB_FILE), "w", encoding="utf-8") as f:
            json.dump(vocab_serialisable, f)

    @classmethod
    def load(cls, dirpath: str) -> "FSM":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        with open(os.path.join(dirpath, cls._VOCAB_FILE), encoding="utf-8") as f:
            vocab = [(sig, count) for sig, count in json.load(f)]

        encoder = cls(
            radius=config["radius"],
            n_vocab=config["n_vocab"],
            min_count=config["min_count"],
            # .get keeps bundles written before this knob existed loadable; the
            # default is the attribute every loader in utils/graph_data.py uses.
            node_label_attr=config.get("node_label_attr", "feature"),
            seed=config["seed"],
        )
        encoder.vocab = vocab
        # The saved vocab length is authoritative for the embedding width.
        encoder.n_vocab = len(vocab)
        return encoder
