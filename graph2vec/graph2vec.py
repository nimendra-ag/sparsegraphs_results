import hashlib
import random
from typing import Dict, List

import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class WeisfeilerLehmanHashing(object):
    """Weisfeiler-Lehman feature extractor.

    Vendored alongside Graph2Vec (rather than imported from karateclub) so this
    module is self-contained, exactly as sf/sf.py is for SF.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
        erase_base_features (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        attributed: bool,
        erase_base_features: bool,
    ):
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.features = None
        self.extracted_features = None
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """Creating the base features.

        With `attributed=True` the node's "feature" attribute is used as the
        iteration-0 label — which is the atom symbol written by
        utils.graph_data._mol_to_graph — otherwise WL falls back to degree.
        """
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, "feature")
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}

    def _erase_base_features(self):
        """Erasing the base features."""
        for k in self.extracted_features:
            del self.extracted_features[k][0]

    def _do_a_recursion(self) -> Dict[int, str]:
        """The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted(str(deg) for deg in degs)
            features = "_".join(features)
            # md5 rather than the builtin hash(), so the subtree identifiers are
            # stable across processes without a PYTHONHASHSEED caveat.
            new_features[node] = hashlib.md5(features.encode()).hexdigest()
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """The method does a series of WL recursions."""
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """Return the node level features."""
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """Return the graph level features."""
        return [
            feature
            for _, features in self.extracted_features.items()
            for feature in features
        ]


class Graph2Vec():
    r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurrence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

    def _wl_documents(self, graphs: List[nx.classes.graph.Graph]) -> List[List[str]]:
        """Turn each graph into its bag of WL subtree identifiers.

        Shared by fit() and infer() so the training corpus and the inferred
        documents can never be built two different ways.
        """
        return [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            ).get_graph_features()
            for graph in graphs
        ]

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_integrity(graph)

        return graph

    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [
            TaggedDocument(words=features, tags=[str(i)])
            for i, features in enumerate(self._wl_documents(graphs))
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        # `.dv` is gensim >= 4.0's name for what used to be `.docvecs`
        # (requirements.txt pins gensim 4.4.0, where `.docvecs` is gone).
        self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents)]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        embedding = np.array(
            [
                self.model.infer_vector(
                    document,
                    alpha=self.learning_rate,
                    min_alpha=0.00001,
                    epochs=self.epochs,
                )
                for document in self._wl_documents(graphs)
            ]
        )

        return embedding
