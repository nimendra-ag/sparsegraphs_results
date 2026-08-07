"""dict_learners/lcksvd.py
=========================
Pipeline adapter that wraps the core LCKSVD implementation
(dict_learners/lc_ksvd.py) into the DictLearner interface used by the
WL pipeline — the same relationship `dict_learners/aksvd.py` has to
`dict_learners/ksvd.py`.

Interface contract
------------------
The pipeline (utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

so this adapter uses the *same* signature as every other learner. LC-KSVD
is supervised, so it raises when `y_train` is missing (mirroring FDDLGPU)
rather than silently ignoring it the way unsupervised AKSVD does.

Key design decisions
--------------------
1.  Constructor parameters follow the AKSVD adapter's vocabulary
    (`dimensions`, `max_iter`, `tol`, `n_non_zero_coefs`, `seed`) and map
    onto the core's LCKSVDConfig fields (K, n_iter, tol, sparsity, seed).
    The LC-KSVD-specific knobs (alpha/beta/lambda1/lambda2/variant) sit
    alongside them.

2.  infer() returns *sparse codes* (N, K), not class predictions. The
    downstream Evaluator trains its own ML models (LR, GB, SVM, RF) on
    those codes, exactly as with AKSVD. The LC-KSVD internal classifier
    W_hat is therefore not used by the pipeline; it stays reachable as
    `.lcksvd_model.W_hat` (and via `predict()` / `predict_scores()`).

3.  The embeddings from WL are (N, n) row-major (scikit-learn convention);
    the core LCKSVD expects (n, N) column-major. Transposition happens
    here so the core algorithm stays untouched.

4.  Labels are remapped to contiguous 0-based class *indices* here. The
    dataset loaders emit {-1, 1}, but the core indexes rows directly
    (`H[labels, arange(N)]`) and compares against 0-based atom labels, so
    raw {-1, 1} would collapse both classes onto the same row of H and
    leave class -1 matching no atom at all. `classes_` records the
    original labels so predictions can be mapped back.

5.  No core LC-KSVD logic is modified. This file is purely an adapter
    layer plus persistence.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dict_learners.dict_learner import DictLearner
from dict_learners.lc_ksvd import LCKSVD, LCKSVDConfig


class LCKSVDLearner(DictLearner):
    """Supervised dictionary learner wrapping LCKSVD for the WL pipeline.

    Parameters
    ----------
    dimensions       : int   — number of dictionary atoms (the core's K).
                               Rule of thumb: atoms_per_class * num_classes.
                               Named to match AKSVD's `dimensions`.
    max_iter         : int   — outer LC-KSVD (Approximate K-SVD) iterations.
    tol              : float — early-stop tolerance on ||Y - DX||_F.
    n_non_zero_coefs : int   — max nonzero coefficients per sparse code (T).
    alpha            : float — weight of the discriminative sparse-code error term.
    beta             : float — weight of the classification error term (LC-KSVD2).
    lambda1          : float — ridge regularisation for classifier W.
    lambda2          : float — ridge regularisation for transform A.
    variant          : str   — 'lcksvd1' or 'lcksvd2'.
    seed             : int   — injected per run (Monte Carlo CV) for reproducibility.
    """

    def __init__(
            self,
            dimensions: int = 64,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
            alpha: float = 32.0,
            beta: float = 2.0,
            lambda1: float = 1e-2,
            lambda2: float = 1e-2,
            variant: str = "lcksvd1",
            seed: int = 42,
    ):
        super().__init__(name="LCKSVD")
        self._dictionary = None
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.variant = variant
        self.seed = seed

        # Original dataset labels, in the order the core's class indices use.
        # Set in fit(); also written into the bundle manifest by the artifact store.
        self.classes_ = None

        self.lcksvd = LCKSVD(LCKSVDConfig(
            K=self.dimensions,
            sparsity=self.n_non_zero_coefs,
            n_iter=self.max_iter,
            tol=self.tol,
            alpha=self.alpha,
            beta=self.beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            variant=self.variant,
            seed=self.seed,
        ))

    def fit(self, training_graph_embeddings, y_train=None):
        # LC-KSVD is supervised: the label-consistency term Q and the classifier
        # term H are both built from y_train, so there is nothing to fall back to.
        if y_train is None:
            raise ValueError("LCKSVDLearner.fit requires y_train (it is a supervised learner).")

        # Seed injected per run (Monte Carlo CV). The per-class SVD init calls
        # ARPACK, which draws its starting vector from the global NumPy RNG, so
        # seeding globally here matters on top of the config's own seed.
        np.random.seed(self.seed)

        # WL produces (N, n); the LCKSVD core expects (n, N) — transpose once here.
        Y = np.asarray(training_graph_embeddings, dtype=float).T  # (n, N)

        # Dataset labels are {-1, 1}; the core needs contiguous 0-based indices.
        y_arr = np.asarray(y_train)
        self.classes_, class_idx = np.unique(y_arr, return_inverse=True)
        num_classes = int(self.classes_.shape[0])

        self.lcksvd.fit(Y, class_idx.astype(int), num_classes=num_classes)

        # Expose the dictionary in the same attribute pattern as AKSVD.
        # Deliberately NOT named `D`: utils/mccv.py treats a learner exposing
        # .D/.k/.classes_ as having a class-partitioned dictionary and runs SRC
        # on it. LC-KSVD's atoms carry class labels but not necessarily equal
        # per-class block sizes (K need not divide evenly), so SRC's fixed-stride
        # slicing would not line up. Those arms stay NaN, as they do for AKSVD.
        self._dictionary = self.lcksvd.D_hat  # (n, K)

        return self

    def infer(self, infer_graph_embeddings):
        Y = np.asarray(infer_graph_embeddings, dtype=float).T  # (n, N)
        # encode() returns (K, N); transpose back to (N, K) for sklearn.
        sparse_embeddings = self.lcksvd.encode(Y).T
        return sparse_embeddings

    def n_atoms(self) -> int:
        return int(self.dimensions)

    # --- LC-KSVD native classifier -------------------------------------------
    # Not used by the pipeline (the Evaluator trains its own models on the
    # sparse codes), but the learned W_hat is the paper's own classifier, so it
    # is exposed here with the labels mapped back to the dataset's convention.

    def predict(self, infer_graph_embeddings):
        Y = np.asarray(infer_graph_embeddings, dtype=float).T  # (n, N)
        return np.asarray(self.classes_)[self.lcksvd.predict(Y)]

    def predict_scores(self, infer_graph_embeddings):
        Y = np.asarray(infer_graph_embeddings, dtype=float).T  # (n, N)
        return self.lcksvd.predict_scores(Y).T  # (N, num_classes)

    # --- Persistence ---------------------------------------------------------
    # encode() only needs D_hat and the sparsity level, so the dictionary goes to
    # .npy and the hyperparams to JSON — same shape as the AKSVD adapter. A_hat,
    # W_hat, atom labels and the class order ride along in an .npz so a restored
    # learner can still run the native LC-KSVD classifier, not just infer().
    _CONFIG_FILE = "lcksvd_config.json"
    _DICT_FILE = "lcksvd_dictionary.npy"
    _STATE_FILE = "lcksvd_state.npz"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "dimensions": self.dimensions,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_non_zero_coefs": self.n_non_zero_coefs,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "variant": self.variant,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError("LCKSVD has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # LCKSVD is pure NumPy, so D_hat is already an ndarray. Kept duck-typed
        # so a tensor-backed backend would still serialize.
        dictionary = self._dictionary
        if hasattr(dictionary, "detach"):
            dictionary = dictionary.detach().cpu().numpy()
        np.save(os.path.join(dirpath, self._DICT_FILE), np.asarray(dictionary))

        model = self.lcksvd
        np.savez(
            os.path.join(dirpath, self._STATE_FILE),
            A_hat=np.asarray(model.A_hat),
            W_hat=np.asarray(model.W_hat),
            atom_labels=np.asarray(model.atom_labels_),
            classes=np.asarray(self.classes_),
        )

    @classmethod
    def load(cls, dirpath: str) -> "LCKSVDLearner":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            dimensions=config["dimensions"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            n_non_zero_coefs=config["n_non_zero_coefs"],
            alpha=config["alpha"],
            beta=config["beta"],
            lambda1=config["lambda1"],
            lambda2=config["lambda2"],
            variant=config["variant"],
            seed=config.get("seed", 42),
        )

        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        learner._dictionary = components

        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.classes_ = state["classes"]

        # Restore the core model's learned state so infer()/predict() work.
        learner.lcksvd.D_hat = components
        learner.lcksvd.A_hat = state["A_hat"]
        learner.lcksvd.W_hat = state["W_hat"]
        learner.lcksvd.atom_labels_ = state["atom_labels"]
        learner.lcksvd.num_classes_ = int(learner.classes_.shape[0])
        return learner
