import os
import json

from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from graph_encoders.wlkernalsubtree import WeisfeilerLehmanHashing
from utils.elbow import find_elbow_cut

from collections import Counter


class WL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50,
            seed: int = 42
    ):

        super().__init__(name="ImbalanceAwareWL")

        self.seed = seed
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_features = min_features

    def create_wl_hash(self, graph_list):

        documents = []

        for graph in graph_list:
            g = self._check_graph(graph)

            document = WeisfeilerLehmanHashing(
                g, self.wl_iterations, self.attributed, self.erase_base_features)

            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    def create_vocab(self, corpus, labels):
        majority_df = Counter()
        minority_df = Counter()

        majority_graphs = 0
        minority_graphs = 0

        for doc, label in zip(corpus, labels):

            # unique subtree hashes in this graph
            # document frequency instead of raw counts
            unique_words = Counter(doc.words)
            if label == -1:
                majority_graphs += 1
                for word in unique_words:
                    majority_df[word] += 1
            else:
                minority_graphs += 1
                for word in unique_words:
                    minority_df[word] += 1

        all_words = set(list(majority_df.keys()) + list(minority_df.keys()))

        scored_vocab = []

        for word in all_words:
            p_majority = majority_df[word] / majority_graphs

            p_minority = (minority_df[word] / minority_graphs)

            discriminative_score = abs(np.sqrt(p_majority) - np.sqrt(p_minority))

            total_presence = p_majority + p_minority

            # Final score

            score = total_presence * discriminative_score
            scored_vocab.append((word, score))

        # Sort features by discriminative importance
        scored_vocab = sorted(
            scored_vocab,
            key=lambda x: x[1],
            reverse=True
        )

        # selection
        
        scores = np.array([x[1] for x in scored_vocab])

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
        # decile_ = 0.25
        # temp = int(l_scores * decile_)
        # threshold = scores[temp]
        # trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

        #-------------------------------------
        #------------ Elbow Cut --------------
        #-------------------------------------
        # scored_vocab is sorted descending, so `scores` is a decreasing curve.
        # Cut at the elbow (data-driven) instead of a fixed percentile.
        print(f"Total Features {len(scores)}")
        n_keep, threshold = find_elbow_cut(scores, sorted_desc=True)
        print(f"elbow cut at index {n_keep} (threshold {threshold:.6g})")

        # # Keep the full (pre-trim) score curve so the elbow can be plotted later,
        # # once the artifact bundle directory exists (see utils/export.py).
        self.selection_scores_ = scores
        trimmed_vocab = scored_vocab[:n_keep]


        # fallback if too few selected
        print(f"selected {len(trimmed_vocab)} from the adaptive selection method")
        if len(trimmed_vocab) < 50:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        i = 0
        for corpus in corpus:
            words = corpus.words

            words_count = Counter(corpus.words)
            j = 0
            for atom, _ in self.vocab:
                sparse_vector[i][j] = words_count[atom]
                j = j + 1

            i = i + 1

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
        infer_graph_embeddings = self.calc_coefficients(
            documents
        )
        return infer_graph_embeddings

    # --- Persistence ---------------------------------------------------------
    # The only *learned* state is `vocab` (the ordered, feature-selected subtree
    # hashes). Everything else is hyperparameters. The vocab order IS the
    # embedding column order, so it must be preserved exactly on disk.
    _CONFIG_FILE = "wl_config.json"
    _VOCAB_FILE = "wl_vocab.json"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "wl_iterations": self.wl_iterations,
            "attributed": self.attributed,
            "erase_base_features": self.erase_base_features,
            "n_vocab": self.n_vocab,
            "min_features": self.min_features,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self.vocab is None:
            raise ValueError("WL has no vocab to save; fit the encoder first.")
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # Preserve order; words are md5 hex strings, scores are floats -> JSON safe.
        vocab_serialisable = [[str(word), float(score)] for word, score in self.vocab]
        with open(os.path.join(dirpath, self._VOCAB_FILE), "w", encoding="utf-8") as f:
            json.dump(vocab_serialisable, f)

    @classmethod
    def load(cls, dirpath: str) -> "WL":
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
            seed=config["seed"],
        )
        encoder.vocab = vocab
        # The saved vocab length is authoritative for the embedding width.
        encoder.n_vocab = len(vocab)
        return encoder