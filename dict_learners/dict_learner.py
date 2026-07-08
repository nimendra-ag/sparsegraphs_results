from abc import ABC, abstractmethod


class DictLearner(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, training_graph_embeddings, y_train=None):
        """Learn the dictionary.

        `y_train` is accepted by every dict learner for a uniform call site,
        even though only supervised learners (e.g. FDDL) use it. Unsupervised
        learners (e.g. AKSVD) simply ignore it.
        """
        pass

    def infer(self, infer_graph_embeddings):
        pass

    def n_atoms(self) -> int:
        """Total number of dictionary atoms (columns of the dictionary).

        Used for dynamic output naming. Subclasses expose the count under
        different attributes, so the base resolves the common cases and
        subclasses may override for anything exotic.
        """
        D = getattr(self, "D", None)
        if D is not None:
            return int(D.shape[1])
        dimensions = getattr(self, "dimensions", None)
        if dimensions is not None:
            return int(dimensions)
        raise NotImplementedError(
            f"{type(self).__name__} cannot report its atom count; override n_atoms()."
        )

    # --- Persistence contract ------------------------------------------------
    # Same rationale as GraphEncoder: each dict learner serialises itself into a
    # directory the artifact store provides, so new learners plug in without
    # touching the store.
    def save(self, dirpath: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save(); it cannot be exported."
        )

    @classmethod
    def load(cls, dirpath: str) -> "DictLearner":
        raise NotImplementedError(
            f"{cls.__name__} does not implement load(); it cannot be restored."
        )
