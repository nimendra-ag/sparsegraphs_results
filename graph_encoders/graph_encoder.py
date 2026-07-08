from abc import ABC, abstractmethod
import random
import numpy as np
import networkx as nx
from typing import List

# abstract base class
class GraphEncoder(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.embeddings = None

    @abstractmethod
    def generate_training_embeddings(self, graphs):
        pass

    @abstractmethod
    def generate_inferencing_embeddings(self, graphs):
        pass

    # --- Persistence contract ------------------------------------------------
    # Each encoder owns its own on-disk representation (loose coupling): the
    # artifact store only hands a directory to save()/load() and never needs to
    # know an encoder's internals. Subclasses that are meant to be deployed must
    # override both. `load` is a classmethod so a bundle can be rebuilt without
    # an existing instance.
    def save(self, dirpath: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save(); it cannot be exported."
        )

    @classmethod
    def load(cls, dirpath: str) -> "GraphEncoder":
        raise NotImplementedError(
            f"{cls.__name__} does not implement load(); it cannot be restored."
        )

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
