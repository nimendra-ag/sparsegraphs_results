"""Edge-centric Weisfeiler-Lehman encoder.

Same contract as `graph_encoders.wl.WL` — same constructor surface, the same
data-driven feature-selection cut, the same L2-normalised count embedding, and
the same save()/load() bundle format — but the WL refinement runs over EDGES
rather than nodes: each edge starts as `<node>-<bond>-<node>` and is refined by
the multiset of labels on the edges sharing an endpoint with it.

Only the feature *extraction* differs from WL. Everything downstream of the raw
subtree/subedge multiset (scoring, selection, vectorisation, persistence) is
deliberately identical, so an EdgeWL run and a WL run differ in exactly one
place and their numbers are comparable.
"""

import os
import json

from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from utils.elbow import find_elbow_cut, find_energy_cut

from collections import Counter


class EdgeWL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50,
            max_vocab: int = 10000,
            selection: str = "energy",
            energy: float = 0.99,
            seed: int = 42
    ):

        super().__init__(name="ImbalanceAwareEdgeWL")

        self.seed = seed
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_features = min_features
        # Hard ceiling on the kept vocabulary, applied AFTER the adaptive cut.
        # Edge refinement produces a far larger raw feature space than node WL
        # (one label per edge per iteration, and the merged labels are almost
        # all unique by iteration 2), so this is the guard that keeps the
        # embedding width — and therefore the dictionary — tractable.
        self.max_vocab = max_vocab
        # Feature-selection cut on the sorted discriminative-score curve:
        #   "energy" -> keep the top features holding `energy` of the total
        #               score (reaches into the tail, recovers AUC);
        #   "elbow"  -> max-distance-to-chord elbow (conservative, fastest).
        self.selection = selection
        self.energy = energy

    def create_wl_hash(self, graph_list):
        documents = []

        for graph in graph_list:
            g = self._check_graph(graph)

            # Initialize edge labels
            edge_labels = {}
            for u, v, data in g.edges(data=True):
                node_u = str(g.nodes[u].get("feature", "UNK"))
                node_v = str(g.nodes[v].get("feature", "UNK"))
                edge_type = str(data.get("bond_type", "SINGLE"))

                # Canonical ordering of node labels
                pair = sorted([node_u, node_v])
                label = f"{pair[0]}-{edge_type}-{pair[1]}"
                edge_labels[tuple(sorted((u, v)))] = label

            graph_features = list(edge_labels.values())

            # WL iterations on edges
            for _ in range(self.wl_iterations):
                new_edge_labels = {}

                for (u, v), current_label in edge_labels.items():
                    neighbor_labels = []

                    # Edges touching u (excluding the current edge)
                    for nbr in g.neighbors(u):
                        if nbr == v:
                            continue
                        edge = tuple(sorted((u, nbr)))
                        if edge in edge_labels:
                            neighbor_labels.append(edge_labels[edge])

                    # Edges touching v (excluding the current edge)
                    for nbr in g.neighbors(v):
                        if nbr == u:
                            continue
                        edge = tuple(sorted((v, nbr)))
                        if edge in edge_labels:
                            neighbor_labels.append(edge_labels[edge])

                    neighbor_labels.sort()

                    merged_label = (
                        current_label + "_" + "_".join(neighbor_labels)
                    )
                    # NOTE: builtin hash() of a str is salted per interpreter
                    # process (PYTHONHASHSEED). Within one process training and
                    # inference agree, so single-run results are sound; a vocab
                    # written by save() is only meaningful to a process started
                    # with the same PYTHONHASHSEED. See save() for the warning.
                    hashed_label = str(hash(merged_label))
                    new_edge_labels[tuple(sorted((u, v)))] = hashed_label

                edge_labels = new_edge_labels
                graph_features.extend(edge_labels.values())

            documents.append(
                TaggedDocument(
                    words=graph_features,
                    tags=[str(len(documents))]
                )
            )

        return documents

    def create_vocab(self, corpus, labels):
        unique_classes = sorted(set(labels))
        n_classes = len(unique_classes)

        # Per-class document frequency and class sizes
        class_df = {c: Counter() for c in unique_classes}
        class_counts = Counter(labels)

        for doc, label in zip(corpus, labels):
            unique_features = set(doc.words)
            for feature in unique_features:
                class_df[label][feature] += 1

        all_features = set()
        for df in class_df.values():
            all_features.update(df.keys())

        scored_vocab = []

        for feature in all_features:
            # Normalized document frequency per class
            p = {
                c: class_df[c][feature] / class_counts[c]
                for c in unique_classes
            }

            # Mean pairwise Hellinger distance
            hellinger_sum = 0.0
            n_pairs = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    ci, cj = unique_classes[i], unique_classes[j]
                    hellinger_sum += abs(np.sqrt(p[ci]) - np.sqrt(p[cj]))
                    n_pairs += 1

            discriminative_score = hellinger_sum / n_pairs if n_pairs > 0 else 0.0

            total_presence = sum(p.values()) / n_classes

            score = total_presence * discriminative_score
            scored_vocab.append((feature, score))

        # Sort features by discriminative importance
        scored_vocab.sort(key=lambda x: x[1], reverse=True)

        # selection

        scores = np.array([x[1] for x in scored_vocab])

        # Normalize the discriminative scores to [0, 1] by dividing by the max.
        # Scores are non-negative, so 0 stays the natural "no signal" floor and
        # the top-ranked feature becomes 1.0. scored_vocab is scaled to match.
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
            scored_vocab = [(word, score / max_score) for word, score in scored_vocab]

        #-------------------------------------
        #------------ Mean - Std -------------
        #-------------------------------------
        # threshold = scores.mean() - scores.std()
        # trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

        #-------------------------------------
        #---Arbitary Percentile Cut (25th)----
        #-------------------------------------
        # print(f"Total Features {len(scores)}")
        # l_scores = len(scores)
        # decile_ = 1
        # # Clamp to the last valid index: int(N * 1.0) == N would run one past the
        # # end of `scores`. decile_ == 1 therefore keeps every feature (threshold
        # # = the smallest score); decile_ < 1 keeps the top fraction.
        # temp = min(int(l_scores * decile_), l_scores - 1)
        # threshold = scores[temp]
        # trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

        #-------------------------------------
        #-------- Adaptive Feature Cut -------
        #-------------------------------------
        # scored_vocab is sorted descending, so `scores` is a decreasing curve.
        # Both cuts are data-driven (no fixed percentile). The energy cut keeps
        # the top features covering `self.energy` of the summed score, which
        # reaches into the weak tail the elbow discards -- trading a little
        # runtime for the AUC that tail carries. See utils/elbow.py.
        print(f"Total Features {len(scores)}")
        if self.selection == "elbow":
            n_keep, threshold = find_elbow_cut(scores, sorted_desc=True)
            print(f"elbow cut at index {n_keep} (threshold {threshold:.6g})")
        elif self.selection == "energy":
            n_keep, threshold = find_energy_cut(
                scores, energy=self.energy, sorted_desc=True,
                min_keep=self.min_features,
            )
            print(f"energy cut ({self.energy:.4g}) at index {n_keep} "
                  f"(threshold {threshold:.6g})")
        elif self.selection == "none":
            # No cut: score and rank as usual, then keep everything. Exists for
            # the comparison arms in pca/, which need the full scored vocabulary
            # so that selection is the ONLY thing differing between arms.
            #
            # Not expressible as energy=1.0: features scoring exactly 0 (equal
            # presence in both classes) make the cumulative-score curve plateau
            # before the last rank, so an energy target of 1.0 still cuts them.
            n_keep, threshold = len(scores), float(scores[-1])
            print(f"no cut: keeping all {n_keep} features "
                  f"(lowest score {threshold:.6g})")
        else:
            raise ValueError(
                f"unknown selection method {self.selection!r}; "
                "expected 'energy', 'elbow' or 'none'"
            )

        # Keep the full (pre-trim) score curve so the elbow can be plotted later,
        # once the artifact bundle directory exists (see utils/export.py).
        self.selection_scores_ = scores
        trimmed_vocab = scored_vocab[:n_keep]

        # fallback if too few selected
        print(f"selected {len(trimmed_vocab)} from the adaptive selection method")
        if len(trimmed_vocab) < self.min_features:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        # Cap vocabulary size. The list is score-sorted, so truncating keeps the
        # strongest features; this only ever bites when the energy cut reaches
        # deep into the tail of an edge vocabulary that is orders of magnitude
        # larger than the node-WL one.
        if len(trimmed_vocab) > self.max_vocab:
            trimmed_vocab = trimmed_vocab[:self.max_vocab]
            print(f"Capped vocabulary to {self.max_vocab}")

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus):
        # Build index map for O(1) lookup
        vocab_index = {word: idx for idx, (word, _) in enumerate(self.vocab)}

        sparse_vector = np.zeros((len(corpus), self.n_vocab))

        for i, doc in enumerate(corpus):
            word_counts = Counter(doc.words)
            for word, count in word_counts.items():
                if word in vocab_index:
                    sparse_vector[i, vocab_index[word]] = count

        # Row-wise L2 normalisation, exactly as WL does: the dictionary learner
        # compares embeddings by inner product, so raw counts would make large
        # molecules dominate purely by size. Zero rows (no vocabulary feature
        # present) are left as zeros rather than divided by 0 -- the training
        # path drops them in utils/pipeline._drop_zero_embeddings.
        norms = np.linalg.norm(sparse_vector, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sparse_vector = sparse_vector / norms

        return sparse_vector

    def generate_training_embeddings(self, graphs, labels):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        self.vocab = self.create_vocab(documents, labels)
        train_graph_embeddings = self.calc_coefficients(documents)
        return train_graph_embeddings

    def generate_inferencing_embeddings(self, graphs):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        infer_graph_embeddings = self.calc_coefficients(documents)
        return infer_graph_embeddings

    # --- Persistence ---------------------------------------------------------
    # Mirrors WL: the only *learned* state is `vocab` (the ordered,
    # feature-selected edge labels). Everything else is hyperparameters. The
    # vocab order IS the embedding column order, so it must be preserved
    # exactly on disk.
    _CONFIG_FILE = "wl_edge_config.json"
    _VOCAB_FILE = "wl_edge_vocab.json"

    # Recorded in the config so a loaded bundle is auditable: the refined edge
    # labels come from builtin hash(), which is salted per process.
    _HASH_SCHEME = "builtin-salted"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "wl_iterations": self.wl_iterations,
            "attributed": self.attributed,
            "erase_base_features": self.erase_base_features,
            "n_vocab": self.n_vocab,
            "min_features": self.min_features,
            "max_vocab": self.max_vocab,
            "selection": self.selection,
            "energy": self.energy,
            "seed": self.seed,
            "hash_scheme": self._HASH_SCHEME,
        }

    def save(self, dirpath: str) -> None:
        if self.vocab is None:
            raise ValueError("EdgeWL has no vocab to save; fit the encoder first.")
        os.makedirs(dirpath, exist_ok=True)

        # Loud, because a silently-wrong bundle is worse than a failed one: the
        # refined edge labels are builtin hash() values, and CPython salts str
        # hashing per process. A vocab saved here will NOT match the labels
        # generated by a differently-salted process, so a reloaded encoder would
        # produce all-zero embeddings without raising. Launch both the exporting
        # and the serving process with the same PYTHONHASHSEED (e.g.
        # `PYTHONHASHSEED=0`) before trusting this bundle for inference.
        # Note: seeding.seed_everything sets PYTHONHASHSEED at *runtime*, which
        # CPython ignores -- it is only read at interpreter start-up.
        print(
            "WARNING: EdgeWL vocabulary uses process-salted builtin hash(); this "
            "bundle only reloads correctly under the same PYTHONHASHSEED. "
            "Single-process training/inference is unaffected."
        )

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # Preserve order; words are stringified hashes, scores are floats -> JSON safe.
        vocab_serialisable = [[str(word), float(score)] for word, score in self.vocab]
        with open(os.path.join(dirpath, self._VOCAB_FILE), "w", encoding="utf-8") as f:
            json.dump(vocab_serialisable, f)

    @classmethod
    def load(cls, dirpath: str) -> "EdgeWL":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        with open(os.path.join(dirpath, cls._VOCAB_FILE), encoding="utf-8") as f:
            vocab = [(word, score) for word, score in json.load(f)]

        encoder = cls(
            wl_iterations=config["wl_iterations"],
            attributed=config["attributed"],
            erase_base_features=config["erase_base_features"],
            n_vocab=config["n_vocab"],
            min_features=config["min_features"],
            # .get keeps older bundles (written before these keys existed)
            # loadable at their pre-existing defaults.
            max_vocab=config.get("max_vocab", 10000),
            selection=config.get("selection", "energy"),
            energy=config.get("energy", 0.99),
            seed=config["seed"],
        )
        encoder.vocab = vocab
        # The saved vocab length is authoritative for the embedding width.
        encoder.n_vocab = len(vocab)
        return encoder
