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

        threshold = scores.mean() - scores.std()
        trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

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