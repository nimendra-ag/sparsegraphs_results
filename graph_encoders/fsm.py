from collections import Counter
import numpy as np
import networkx as nx
import pandas as pd
from graph_encoders.graph_encoder import GraphEncoder


class FSM(GraphEncoder):
    def __init__(
            self,
            radius: int = 1,
            n_vocab: int = 1000,
            min_count: int = 5
    ):
        super().__init__(name="FSM")
        self.radius = radius
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.vocab = None
        self.embeddings = None

    def _get_subgraph_signature(self, sub_g):
        num_nodes = sub_g.number_of_nodes()
        num_edges = sub_g.number_of_edges()

        degrees = sorted([d for n, d in sub_g.degree()])
        deg_str = ",".join(map(str, degrees))

        first_node = list(sub_g.nodes())[0]
        if 'label' in sub_g.nodes[first_node]:
            labels = sorted([sub_g.nodes[n]['label'] for n in sub_g.nodes()])
            label_str = ",".join(labels)
            return f"N{num_nodes}_E{num_edges}_L[{label_str}]_D[{deg_str}]"

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

    def create_vocab(self, documents, labels, n_vocab=1000, disc_ratio=0.3):
        """
        HYBRID STRATEGY:
        Allocates (1 - disc_ratio) of the vocabulary to the most frequent shapes per class (for K-SVD reconstruction stability).
        Allocates (disc_ratio) of the vocabulary to the highest variance shapes (for classification power).
        """
        unique_classes = np.unique(labels)

        # --- PART 1: FREQUENCY COUNTING ---
        class_shape_counts = {cls: Counter() for cls in unique_classes}
        class_docs = {cls: [] for cls in unique_classes}
        global_shapes = set()

        for doc, label in zip(documents, labels):
            class_docs[label].append(set(doc))
            class_shape_counts[label].update(set(doc))
            global_shapes.update(set(doc))

        class_sizes = {cls: len(docs) for cls, docs in class_docs.items()}

        hybrid_vocab_dict = {}

        # --- PART 2: GATHER FREQUENT "BUILDING BLOCKS" (e.g., 70% of vocab) ---
        freq_vocab_size = int(n_vocab * (1 - disc_ratio))
        vocab_per_class = freq_vocab_size // len(unique_classes)

        for cls in unique_classes:
            # Sort by frequency within this specific class
            sorted_cls_freq = sorted(class_shape_counts[cls].items(), key=lambda x: x[1], reverse=True)
            top_cls_features = sorted_cls_freq[:vocab_per_class]

            for shape, count in top_cls_features:
                hybrid_vocab_dict[shape] = count  # Store highest frequency

        # --- PART 3: GATHER DISCRIMINATIVE "SPECIALTY PIECES" (e.g., 30% of vocab) ---
        disc_vocab_size = n_vocab - len(hybrid_vocab_dict)
        shape_scores = {}

        for shape in global_shapes:
            # Skip shapes we already secured in the frequent batch
            if shape in hybrid_vocab_dict:
                continue

            support_rates = []
            for cls in unique_classes:
                support = class_shape_counts[cls].get(shape, 0) / class_sizes[cls]
                support_rates.append(support)

            # Score is variance of support rates across classes
            shape_scores[shape] = np.var(support_rates)

        # Sort remaining shapes by discriminative score
        sorted_disc_shapes = sorted(shape_scores.items(), key=lambda x: x[1], reverse=True)
        top_disc_features = sorted_disc_shapes[:disc_vocab_size]

        # Add the discriminative shapes to our final dictionary
        for shape, score in top_disc_features:
            # We assign a dummy count or look up its global frequency just to store it
            hybrid_vocab_dict[shape] = sum([class_shape_counts[c].get(shape, 0) for c in unique_classes])

        # --- PART 4: FINALIZE ---
        # Sort everything by global occurrence just for a clean, consistent output structure
        final_vocab = sorted(hybrid_vocab_dict.items(), key=lambda item: item[1], reverse=True)

        self.n_vocab = len(final_vocab)
        print(f"Hybrid Vocab Built: {freq_vocab_size} Frequent Base Shapes + {disc_vocab_size} Discriminative Shapes.")

        return final_vocab

    def calc_coefficients(self, documents):
        sparse_vector = np.zeros([len(documents), self.n_vocab])
        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for j, (subgraph_sig, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[subgraph_sig]
        return sparse_vector

    def generate_training_embeddings(self, graphs, labels):
        """Now requires labels to perform class-aware trimming."""
        documents = self.extract_subgraphs(graphs)

        # Pass the labels to the updated vocab creator
        self.vocab = self.create_vocab(documents, labels)
        raw_embeddings = self.calc_coefficients(documents)

        # REDUNDANCY / COLLINEARITY FILTER
        print(f"\nFiltering Collinear Subgraphs...")
        df = pd.DataFrame(raw_embeddings)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_indices = [column for column in upper.columns if any(upper[column] > 0.95)]

        self.embeddings = df.drop(columns=to_drop_indices).values
        self.vocab = [v for i, v in enumerate(self.vocab) if i not in to_drop_indices]
        self.n_vocab = len(self.vocab)

        print(f"Dropped {len(to_drop_indices)} redundant topologies. Final Vocab Size: {self.n_vocab}")

        return self.embeddings

    def generate_inferencing_embeddings(self, graphs):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call generate_training_embeddings first.")

        documents = self.extract_subgraphs(graphs)
        return self.calc_coefficients(documents)