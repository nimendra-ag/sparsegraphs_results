from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from graph_encoders.wlkernalsubtree import WeisfeilerLehmanHashing
from collections import Counter


class WL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50
    ):
        super().__init__(name="ImbalanceAwareWL")

        self.seed = 42
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_features = min_features

        # Populated by create_vocab(); consumed by WLAKSVDInterpreter
        self.class_df: dict = {}
        self.class_counts: Counter = Counter()

    def create_wl_hash(self, graph_list):
        documents = []

        for graph in graph_list:
            g = self._check_graph(graph)
            document = WeisfeilerLehmanHashing(
                g, self.wl_iterations, self.attributed, self.erase_base_features
            )
            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    def create_vocab(self, corpus, labels):
        unique_classes = sorted(set(labels))
        n_classes = len(unique_classes)

        # Per-class document frequency and class sizes
        class_df = {c: Counter() for c in unique_classes}
        class_counts = Counter(labels)

        for doc, label in zip(corpus, labels):
            unique_words = set(doc.words)
            for word in unique_words:
                class_df[label][word] += 1

    
        self.class_df = class_df
        self.class_counts = class_counts

        all_words = set()
        for df in class_df.values():
            all_words.update(df.keys())

        scored_vocab = []

        for word in all_words:
            # Normalized document frequency per class
            p = {
                c: class_df[c][word] / class_counts[c]
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
            scored_vocab.append((word, score))

        scored_vocab.sort(key=lambda x: x[1], reverse=True)

        scores = np.array([x[1] for x in scored_vocab])
        threshold = scores.mean() - scores.std()
        trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

        print(f"Selected {len(trimmed_vocab)} features via adaptive selection")
        if len(trimmed_vocab) < self.min_features:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus):
        # Build index map for O(1) lookup instead of linear scan
        vocab_index = {word: idx for idx, (word, _) in enumerate(self.vocab)}

        sparse_vector = np.zeros((len(corpus), self.n_vocab))

        for i, doc in enumerate(corpus):
            word_counts = Counter(doc.words)
            for word, count in word_counts.items():
                if word in vocab_index:
                    sparse_vector[i, vocab_index[word]] = count

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