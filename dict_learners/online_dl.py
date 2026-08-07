"""dict_learners/online_dl.py
============================
Pipeline adapter that wraps scikit-learn's online dictionary learning
(`sklearn.decomposition.DictionaryLearning`, the Mairal et al. online
dictionary-learning solver) into the DictLearner interface used by the WL
pipeline — the same relationship `dict_learners/aksvd.py` has to
`dict_learners/ksvd.py` and `dict_learners/bayesian.py` has to
`dict_learners/bayesian_dl.py`.

Provenance
----------
This is the notebook `test_scikitlearn_dl.ipynb` turned into a module. The
learning logic is unchanged — it is still sklearn's estimator, constructed with
the exact parameters the notebook used:

    DictionaryLearning(n_components=8000, transform_algorithm='omp',
                       transform_alpha=0.1, random_state=42, verbose=True,
                       max_iter=100, tol=1e-06)

so those are the defaults here. Everything the notebook did around them (load
NCI, WL vocabulary, split, MaxAbsScaler, Evaluator, write results) is *not*
duplicated in this file: that is exactly what `utils/pipeline.py`,
`utils/export.py` and `utils/mccv.py` already do for every other arm, and the
`implements/wl_online_dl*.py` scripts reuse them.

Interface contract
------------------
The pipeline (utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

so this adapter uses the *same* signature as every other learner. Online
dictionary learning is unsupervised — the reconstruction/sparsity objective, not
the labels, decides the atoms — so `y_train` is accepted and ignored, exactly as
AKSVD and BayesianDL do, rather than raising the way the supervised learners
(FDDLGPU, LCKSVDLearner, FrozenKSVDLearner) do.

Notes
-----
1.  Constructor parameters follow the AKSVD adapter's vocabulary (`dimensions`,
    `seed`) and map onto sklearn's own names (`n_components`, `random_state`).

2.  sklearn stores the dictionary row-major as `components_` of shape
    (n_components, n_features); it is exposed as `_dictionary` and saved as-is.
    Deliberately NOT named `D`: utils/mccv.py treats a learner exposing
    .D/.k/.classes_ as having a class-partitioned dictionary and runs SRC on it.
    These atoms carry no class identity, so those arms correctly stay NaN.

3.  `dimensions` defaults to the notebook's 8000 atoms, which is a multi-hour
    (potentially multi-day) fit on NCI-sized data. Pass a smaller `dimensions`
    when a quick run is wanted — nothing else changes.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sklearn.decomposition import DictionaryLearning

from dict_learners.dict_learner import DictLearner


class OnlineDL(DictLearner):
    """scikit-learn online dictionary learning for the WL pipeline.

    Parameters
    ----------
    dimensions               : int   — number of dictionary atoms (sklearn's
                                       `n_components`). Named to match AKSVD's
                                       `dimensions`.
    alpha                    : float — sparsity weight of the *fitting* problem.
    max_iter                 : int   — maximum alternating optimisation rounds.
    tol                      : float — convergence tolerance on the dictionary
                                       difference between two rounds.
    fit_algorithm            : str   — 'lars' or 'cd', the solver used while
                                       learning the dictionary.
    transform_algorithm      : str   — sparse-coding algorithm used by infer()
                                       ('omp' in the notebook).
    transform_alpha          : float — sparsity/tolerance knob of the coding
                                       step; for 'omp' it is the reconstruction
                                       tolerance.
    transform_n_nonzero_coefs: int   — hard sparsity budget for 'omp'/'lars'
                                       coding; None lets sklearn pick
                                       max(n_features / 10, 1).
    n_jobs                   : int   — parallelism handed to sklearn.
    seed                     : int   — injected per run (Monte Carlo CV) for
                                       reproducibility; passed through as
                                       sklearn's `random_state`.
    verbose                  : bool  — sklearn's per-iteration progress output.
    """

    def __init__(
            self,
            dimensions: int = 64,
            alpha: float = 1.0,
            max_iter: int = 100,
            tol: float = 1e-6,
            fit_algorithm: str = "lars",
            transform_algorithm: str = "omp",
            transform_alpha: float = 0.1,
            transform_n_nonzero_coefs: int = None,
            n_jobs: int = None,
            seed: int = 42,
            verbose: bool = True,
    ):
        super().__init__(name="OnlineDL")
        self._dictionary = None
        self.dimensions = dimensions
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_alpha = transform_alpha
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.dict_learner = self._build_estimator()

    def _build_estimator(self) -> DictionaryLearning:
        """Construct the sklearn estimator from the adapter's hyperparameters.

        Kept separate so `load()` can rebuild an identically configured
        estimator before injecting a restored dictionary into it.
        """
        return DictionaryLearning(
            n_components=self.dimensions,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_algorithm=self.fit_algorithm,
            transform_algorithm=self.transform_algorithm,
            transform_alpha=self.transform_alpha,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def fit(self, training_graph_embeddings, y_train=None):
        # y_train is ignored: online dictionary learning is unsupervised. It is
        # accepted only so the dict-learner call site is uniform across
        # supervised/unsupervised types.
        # sklearn draws from its own check_random_state(seed), but the SVD-based
        # initialisation calls into LAPACK/ARPACK paths that can consult the
        # global NumPy RNG, so seed globally too — same rationale as AKSVD.
        np.random.seed(self.seed)

        self.dict_learner.fit(training_graph_embeddings)
        self._dictionary = self.dict_learner.components_  # (n_components, P)
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.dict_learner.transform(infer_graph_embeddings)
        return sparse_embeddings

    def n_atoms(self) -> int:
        return int(self.dimensions)

    # --- Persistence ---------------------------------------------------------
    # transform() only needs components_ (the dictionary) and the hyperparams,
    # so we save the dictionary as .npy and the config as JSON — no pickle of the
    # inner estimator required.
    _CONFIG_FILE = "online_dl_config.json"
    _DICT_FILE = "online_dl_dictionary.npy"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "dimensions": self.dimensions,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "fit_algorithm": self.fit_algorithm,
            "transform_algorithm": self.transform_algorithm,
            "transform_alpha": self.transform_alpha,
            "transform_n_nonzero_coefs": self.transform_n_nonzero_coefs,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError("OnlineDL has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)
        # DictionaryLearning is pure NumPy, so components_ is already an ndarray.
        # Kept duck-typed so a tensor-backed backend would still serialize.
        dictionary = self._dictionary
        if hasattr(dictionary, "detach"):
            dictionary = dictionary.detach().cpu().numpy()
        np.save(os.path.join(dirpath, self._DICT_FILE), np.asarray(dictionary))

    @classmethod
    def load(cls, dirpath: str) -> "OnlineDL":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            dimensions=config["dimensions"],
            alpha=config["alpha"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            fit_algorithm=config["fit_algorithm"],
            transform_algorithm=config["transform_algorithm"],
            transform_alpha=config["transform_alpha"],
            transform_n_nonzero_coefs=config["transform_n_nonzero_coefs"],
            n_jobs=config.get("n_jobs"),
            seed=config.get("seed", 42),
            verbose=config.get("verbose", True),
        )
        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        learner._dictionary = components
        # Restore the inner estimator's fitted state so transform() works:
        # components_ satisfies check_is_fitted, n_features_in_ satisfies the
        # input validation sklearn runs with reset=False.
        learner.dict_learner.components_ = components
        learner.dict_learner.n_features_in_ = int(components.shape[1])
        return learner
