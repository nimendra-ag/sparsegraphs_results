import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dict_learners.dict_learner import DictLearner
from ksvd import ApproximateKSVD

class AKSVD(DictLearner):
    def __init__(
            self,
            dimensions: int = 350,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
            seed: int = 42,
    ):
        super().__init__(name="AKSVD")
        self._dictionary = None
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.seed = seed
        self.aksvd = ApproximateKSVD(n_components=self.dimensions, max_iter=self.max_iter, tol=self.tol,
                 transform_n_nonzero_coefs=self.n_non_zero_coefs)

    def fit(self, training_graph_embeddings, y_train=None):
        # y_train is ignored: AKSVD is unsupervised. It is accepted only so the
        # dict-learner call site is uniform across supervised/unsupervised types.
        # Seed injected per run (Monte Carlo CV) — only the random-init fallback in
        # ApproximateKSVD._initialize is stochastic, but we seed for reproducibility.
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self._dictionary = self.aksvd.fit(training_graph_embeddings).components_

        # self._embedding = self.aksvd.transform(training_graph_embeddings)
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.aksvd.transform(infer_graph_embeddings)
        return sparse_embeddings

    def n_atoms(self) -> int:
        return int(self.dimensions)

    # --- Persistence ---------------------------------------------------------
    # transform() only needs components_ (the dictionary) and the hyperparams,
    # so we save the dictionary as .npy and the config as JSON — no pickle of the
    # inner estimator required.
    _CONFIG_FILE = "aksvd_config.json"
    _DICT_FILE = "aksvd_dictionary.npy"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "dimensions": self.dimensions,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_non_zero_coefs": self.n_non_zero_coefs,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError("AKSVD has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)
        # _dictionary is a GPU tensor (ApproximateKSVD runs on CUDA); move it to
        # host memory before converting to a NumPy array for np.save.
        dictionary = self._dictionary
        if isinstance(dictionary, torch.Tensor):
            dictionary = dictionary.detach().cpu().numpy()
        np.save(os.path.join(dirpath, self._DICT_FILE), np.asarray(dictionary))

    @classmethod
    def load(cls, dirpath: str) -> "AKSVD":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            dimensions=config["dimensions"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            n_non_zero_coefs=config["n_non_zero_coefs"],
            seed=config.get("seed", 42),
        )
        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        # Move the dictionary back onto the estimator's device so transform()
        # (which builds a Gram matrix against GPU tensors) doesn't mix backends.
        components = torch.from_numpy(components).float().to(learner.aksvd.device)
        learner._dictionary = components
        # Restore the inner estimator's dictionary so transform() works.
        learner.aksvd.components_ = components
        return learner