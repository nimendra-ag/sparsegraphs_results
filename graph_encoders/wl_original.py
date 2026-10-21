import os
import json

from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from graph_encoders.wlkernalsubtree import WeisfeilerLehmanHashing

from collections import Counter


class WL_Original(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1800,
            min_count: int = 1,
            epochs: int = 10,
            seed: int = 42
    ):

        super().__init__(name="WL_Original")

        self.seed = seed
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.epochs = epochs

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

    def create_vocab(self, corpus):
        d2v_model = Doc2Vec(vector_size=self.n_vocab, min_count=self.min_count, epochs=self.epochs)

        total_words, corpus_count = d2v_model.scan_vocab(
            corpus_iterable=corpus, corpus_file=None,
            progress_per=10000, trim_rule=None
        )
        d2v_model.corpus_count = corpus_count
        d2v_model.corpus_total_words = total_words
        d2v_model.prepare_vocab(update=False, keep_raw_vocab=True, trim_rule=None)

        sorted_vocab = sorted(d2v_model.raw_vocab.items(), key=lambda item: item[1], reverse=True)
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        print(f"Total unique features: {len(sorted_vocab)}, kept top {len(trimmed_vocab)}")
        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab


    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        i = 0
        for doc in corpus:
            words_count = Counter(doc.words)
            j = 0
            for atom, _ in self.vocab:
                sparse_vector[i][j] = words_count[atom]
                j = j + 1
            i = i + 1

        return sparse_vector

    def generate_training_embeddings(self, graphs, labels):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        self.vocab = self.create_vocab(documents)
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
            "min_count": self.min_count,
            "epochs": self.epochs,
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
    def load(cls, dirpath: str) -> "WL_Original":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        with open(os.path.join(dirpath, cls._VOCAB_FILE), encoding="utf-8") as f:
            vocab = [(word, score) for word, score in json.load(f)]

        encoder = cls(
            wl_iterations=config["wl_iterations"],
            attributed=config["attributed"],
            erase_base_features=config["erase_base_features"],
            n_vocab=config["n_vocab"],
            min_count=config.get("min_count", 1),
            epochs=config.get("epochs", 10),
            seed=config["seed"],
        )
        encoder.vocab = vocab
        encoder.n_vocab = len(vocab)
        return encoder