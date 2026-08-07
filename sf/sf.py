import numpy as np
import networkx as nx
from typing import List
from scipy.sparse.linalg import eigsh
import random

class SF():
    r"""An implementation of `"SF" <https://arxiv.org/abs/1810.09155>`_
    from the NeurIPS Relational Representation Learning Workshop '18 paper "A Simple Baseline Algorithm for Graph Classification".
    The procedure calculates the k lowest eigenvalues of the normalized Laplacian.
    If the graph has a lower number of eigenvalues than k the representation is padded with zeros.

    Args:
        dimensions (int): Number of lowest eigenvalues. Default is 128.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(self, dimensions: int = 128, seed: int = 42):
        self.dimensions = dimensions
        self.seed = seed

    def _calculate_sf(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        number_of_nodes = graph.number_of_nodes()
        L_tilde = nx.normalized_laplacian_matrix(graph, nodelist=range(number_of_nodes))
        if number_of_nodes <= self.dimensions:
            embedding = eigsh(
                L_tilde,
                k=number_of_nodes - 1,
                which="LM",
                ncv=10 * self.dimensions,
                return_eigenvectors=False,
            )

            shape_diff = self.dimensions - embedding.shape[0] - 1
            embedding = np.pad(
                embedding, (1, shape_diff), "constant", constant_values=0
            )
        else:
            embedding = eigsh(
                L_tilde,
                k=self.dimensions,
                which="LM",
                ncv=10 * self.dimensions,
                return_eigenvectors=False,
            )
        return embedding

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
    def fit(self, graphs):
        """Fitting an SF model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_sf(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Inferring the embedding vectors.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        embedding = np.array([self._calculate_sf(graph) for graph in graphs])
        return embedding
