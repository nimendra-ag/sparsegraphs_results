import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dict_learners.dict_learner import DictLearner
from ksvd import ApproximateKSVD

class AKSVD(DictLearner):
    def __init__(
            self,
            dimensions: int = 32,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
    ):
        super().__init__(name="AKSVD")
        self._dictionary = None
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.aksvd = ApproximateKSVD(n_components=self.dimensions, max_iter=self.max_iter, tol=self.tol,
                 transform_n_nonzero_coefs=self.n_non_zero_coefs)

    def fit(self, training_graph_embeddings):
        self._dictionary = self.aksvd.fit(training_graph_embeddings).components_

        # self._embedding = self.aksvd.transform(training_graph_embeddings)
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.aksvd.transform(infer_graph_embeddings)
        return sparse_embeddings