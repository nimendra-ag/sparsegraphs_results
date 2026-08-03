from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Get sorted degree sequence
        degrees = sorted([d for n, d in sub_g.degree()])
        deg_str = ",".join(map(str, degrees))

        # NEW: Check if the graph has node labels (like NCI chemical symbols)
        # We grab the first node to check for the 'label' attribute
        first_node = list(sub_g.nodes())[0]
        if 'label' in sub_g.nodes[first_node]:
            # Extract all labels in the subgraph and sort them alphabetically
            labels = sorted([sub_g.nodes[n]['label'] for n in sub_g.nodes()])
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

    def create_vocab(self, documents):
        global_counts = Counter()
        for doc in documents:
            global_counts.update(doc)

        frequent_subgraphs = {word: count for word, count in global_counts.items() if count >= self.min_count}
        sorted_vocab = sorted(frequent_subgraphs.items(), key=lambda item: item[1], reverse=True)
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, documents):
        sparse_vector = np.zeros([len(documents), self.n_vocab])
        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for j, (subgraph_sig, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[subgraph_sig]
        return sparse_vector

    # --- NEW MAIN BRANCH INTERFACE METHODS ---

    def generate_training_embeddings(self, graphs):
        """Extracts subgraphs, builds the locked vocabulary, and returns the Y matrix."""
        documents = self.extract_subgraphs(graphs)
        self.vocab = self.create_vocab(documents)
        self.embeddings = self.calc_coefficients(documents)
        return self.embeddings

    def generate_inferencing_embeddings(self, graphs):
        """Extracts subgraphs from new data and maps them to the EXISTING vocabulary."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call generate_training_embeddings first.")

        documents = self.extract_subgraphs(graphs)
        return self.calc_coefficients(documents)