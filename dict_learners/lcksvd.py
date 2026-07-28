"""
dict_learners/lcksvd.py
=======================
Pipeline adapter that wraps the core LCKSVD implementation
into the DictLearner interface used by the WL pipeline.

Interface contract
------------------
The pipeline calls:
    lcksvd = LCKSVDLearner().fit(training_graph_embeddings, labels)
    X_train = lcksvd.infer(graph_embeddings_ml_train)
    X_test  = lcksvd.infer(graph_embeddings_ml_test)

Key design decisions
--------------------
1.  fit() accepts labels alongside embeddings — LC-KSVD is a *supervised*
    dictionary learner, so labels are required at fit time.
    This is the only API difference from AKSVD; the rest of the pipeline
    (infer → Evaluator) is identical.

2.  infer() returns *sparse codes* (K,)-dimensional vectors, not class
    predictions. The downstream Evaluator trains and evaluates its own
    ML models (LR, GB, SVM, RF) on those codes, exactly as with AKSVD.
    The LC-KSVD internal classifier W_hat is therefore NOT used here —
    it remains available via .lcksvd_model.W_hat if needed externally.

3.  The input embeddings from WL are (N, n) row-major (scikit-learn
    convention). The core LCKSVD expects (n, N) column-major.
    Transposition is handled here in fit() and infer() so the core
    algorithm stays untouched.

4.  num_classes is inferred from the unique values in labels at fit time.

5.  No original LC-KSVD logic is modified. This file is purely an
    adapter layer.
"""

import numpy as np
from dict_learners.dict_learner import DictLearner
from dict_learners.lc_ksvd import LCKSVD, LCKSVDConfig


class LCKSVDLearner(DictLearner):
    """
    Supervised dictionary learner wrapping LCKSVD for the WL pipeline.

    Parameters
    ----------
    K               : int   — number of dictionary atoms.
                              Rule of thumb: (atoms_per_class * num_classes).
                              Mirroring AKSVD's 'dimensions' parameter.
    sparsity        : int   — max nonzero coefficients per sparse code (T).
                              Mirrors AKSVD's 'n_non_zero_coefs'.
    n_iter          : int   — outer LC-KSVD iterations. Mirrors AKSVD's 'max_iter'.
    alpha           : float — weight of discriminative sparse-code error term.
    beta            : float — weight of classification error term.
    lambda1         : float — ridge regularisation for classifier W.
    lambda2         : float — ridge regularisation for transform A.
    variant         : str   — 'lcksvd1' or 'lcksvd2'.
    """

    def __init__(
        self,
        K: int = 256,
        sparsity: int = 10,
        n_iter: int = 10,
        alpha: float = 32.0,
        beta: float = 2.0,
        lambda1: float = 1e-2,
        lambda2: float = 1e-2,
        variant: str = "lcksvd1",
    ):
        super().__init__(name="LCKSVD")

        self._config = LCKSVDConfig(
            K=K,
            sparsity=sparsity,
            n_iter=n_iter,
            alpha=alpha,
            beta=beta,
            lambda1=lambda1,
            lambda2=lambda2,
            variant=variant,
        )

        # The core LCKSVD model — created and populated in fit()
        self.lcksvd_model: LCKSVD = LCKSVD(self._config)
        self._dictionary = None  # populated after fit(), mirrors AKSVD pattern

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        training_graph_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> "LCKSVDLearner":
        """
        Learn the dictionary from labelled WL graph embeddings.

        Parameters
        ----------
        training_graph_embeddings : (N, n) ndarray
            Row-major embedding matrix from WL.generate_training_embeddings().
            Each row is one graph's WL feature vector.

        labels : (N,) ndarray of int
            Class labels for each training graph (0-based integer indices).
            LC-KSVD requires labels at fit time — this is the fundamental
            difference from unsupervised AKSVD.

        Returns
        -------
        self
        """
        # WL produces (N, n); LCKSVD core expects (n, N) — transpose once here.
        Y = np.array(training_graph_embeddings, dtype=float).T  # (n, N)

        labels_arr = np.array(labels, dtype=int)
        num_classes = int(np.unique(labels_arr).shape[0])

        self.lcksvd_model.fit(Y, labels_arr, num_classes=num_classes)

        # Expose dictionary in the same attribute pattern as AKSVD
        self._dictionary = self.lcksvd_model.D_hat  # (n, K)

        return self

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------

    def infer(self, infer_graph_embeddings: np.ndarray) -> np.ndarray:
        """
        Encode graphs into sparse codes using the learned dictionary.

        This mirrors AKSVD.infer() — the output is a sparse embedding
        matrix suitable for the Evaluator's downstream ML classifiers.

        Parameters
        ----------
        infer_graph_embeddings : (N, n) ndarray
            Row-major embedding matrix from WL.generate_inferencing_embeddings().

        Returns
        -------
        sparse_embeddings : (N, K) ndarray
            Each row is a K-dimensional sparse code for one graph.
            Returned in row-major form so it plugs directly into
            scikit-learn's MaxAbsScaler and the Evaluator.
        """
        Y = np.array(infer_graph_embeddings, dtype=float).T  # (n, N)

        # encode() returns (K, N); transpose back to (N, K) for sklearn
        X = self.lcksvd_model.encode(Y)   # (K, N)
        return X.T                         # (N, K)
