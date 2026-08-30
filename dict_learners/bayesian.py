"""dict_learners/bayesian.py
===========================
Pipeline adapter that wraps the core Beta Process Factor Analysis
implementation (dict_learners/bayesian_dl.py) into the DictLearner interface
used by the WL pipeline — the same relationship `dict_learners/aksvd.py` has to
`dict_learners/ksvd.py` and `dict_learners/lcksvd.py` has to
`dict_learners/lc_ksvd.py`.

Interface contract
------------------
The pipeline (utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

so this adapter uses the *same* signature as every other learner. BPFA is
unsupervised — the beta process prior, not the labels, decides which atoms a
sample uses — so `y_train` is accepted and ignored, exactly as AKSVD does,
rather than raising the way the supervised learners (FDDLGPU, LCKSVDLearner,
FrozenKSVDLearner) do.

Key design decisions
--------------------
1.  Constructor parameters follow the AKSVD adapter's vocabulary
    (`dimensions`, `seed`) and map onto the core's own names
    (`n_components`, `random_state`). The BPFA-specific knobs — the Gibbs
    iteration counts and the Beta/Gamma hyper-priors — sit alongside them.

2.  `dimensions` is the *maximum* dictionary size K. The beta process prior
    infers the effective size automatically (unused atoms decay toward the
    prior), so `n_atoms()` reports K — the actual width of the sparse codes
    the Evaluator sees — while `effective_dictionary_size` reports how many
    atoms the sampler actually kept. Both are recorded at save time.

3.  The embeddings from WL are (N, P) row-major, which is already the layout
    the core's fit()/infer() expect, so — unlike the LC-KSVD adapter — no
    transposition is needed. The core stores its dictionary column-major as
    (P, K); that is left as-is and saved that way.

4.  infer() returns the posterior mean of the sparse codes z_i ⊙ s_i, shape
    (N, K). The downstream Evaluator trains its own ML models (LR, GB, SVM,
    RF) on those codes, exactly as with AKSVD.

5.  No .D/.k/.classes_ is exposed: BPFA never sees labels, so its atoms carry
    no class identity and utils/mccv.py correctly records the SRC arms as NaN
    (same as AKSVD, LC-KSVD and frozen K-SVD).

6.  No core BPFA logic is modified. This file is purely an adapter layer plus
    persistence.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dict_learners.dict_learner import DictLearner
from dict_learners.bayesian_dl import BPFA


class BayesianDL(DictLearner):
    """Non-parametric Bayesian dictionary learner (BPFA) for the WL pipeline.

    Parameters
    ----------
    dimensions   : int   — maximum number of dictionary atoms (the core's K).
                           Named to match AKSVD's `dimensions`; the effective
                           size is inferred by the beta process prior.
    n_iter       : int   — Gibbs sweeps during dictionary learning.
    n_infer_iter : int   — Gibbs sweeps when coding new data with D held fixed.
    n_burnin     : int   — inference sweeps discarded before collecting samples.
    n_collect    : int   — post-burn-in samples averaged into the final code.
    a0, b0       : float — Beta prior on the atom-usage probabilities pi_k.
                           `b0=None` defaults to N/8 inside the core (the value
                           recommended by the paper).
    c0, d0       : float — Gamma hyper-prior on the noise precision gamma_eps.
    e0, f0       : float — Gamma hyper-prior on the weight precision gamma_s.
    init_method  : str   — 'svd' (recommended) or 'random'.
    seed         : int   — injected per run (Monte Carlo CV) for reproducibility;
                           passed through as the core's `random_state`.
    verbose      : bool  — log sampler progress every 10 iterations.
    """

    def __init__(
            self,
            dimensions: int = 32,
            n_iter: int = 10,
            n_infer_iter: int = 50,
            n_burnin: int = 20,
            n_collect: int = 10,
            a0: float = 1.0,
            b0: float = None,
            c0: float = 1e-6,
            d0: float = 1e-6,
            e0: float = 1e-6,
            f0: float = 1e-6,
            init_method: str = "svd",
            seed: int = 42,
            verbose: bool = False,
    ):
        super().__init__(name="BayesianDL")
        self._dictionary = None
        self.dimensions = dimensions
        self.n_iter = n_iter
        self.n_infer_iter = n_infer_iter
        self.n_burnin = n_burnin
        self.n_collect = n_collect
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
        self.e0 = e0
        self.f0 = f0
        self.init_method = init_method
        self.seed = seed
        self.verbose = verbose

        self.bpfa = BPFA(
            n_components=self.dimensions,
            n_iter=self.n_iter,
            n_infer_iter=self.n_infer_iter,
            n_burnin=self.n_burnin,
            n_collect=self.n_collect,
            a0=self.a0,
            b0=self.b0,
            c0=self.c0,
            d0=self.d0,
            e0=self.e0,
            f0=self.f0,
            init_method=self.init_method,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def fit(self, training_graph_embeddings, y_train=None):
        # y_train is ignored: BPFA is unsupervised. It is accepted only so the
        # dict-learner call site is uniform across supervised/unsupervised types.
        # The sampler draws from its own RandomState(seed), but the SVD init
        # calls into LAPACK/ARPACK paths that can consult the global NumPy RNG,
        # so seed globally too — same rationale as the AKSVD adapter.
        np.random.seed(self.seed)

        self.bpfa.fit(training_graph_embeddings)
        # Expose the dictionary in the same attribute pattern as AKSVD.
        # Deliberately NOT named `D`: utils/mccv.py treats a learner exposing
        # .D/.k/.classes_ as having a class-partitioned dictionary and runs SRC
        # on it. BPFA's atoms carry no class identity, so those arms stay NaN.
        self._dictionary = self.bpfa.dictionary  # (P, K)
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.bpfa.infer(infer_graph_embeddings)
        return sparse_embeddings

    def n_atoms(self) -> int:
        # The code width the Evaluator sees is the full K, not the effective
        # size: atoms the prior switched off still occupy (zero) columns.
        return int(self.dimensions)

    # --- BPFA posterior summaries --------------------------------------------
    # Not used by the pipeline, but they are the reason to run a non-parametric
    # model in the first place, so they are exposed (and persisted) here.

    @property
    def effective_dictionary_size(self) -> int:
        """Atoms used more than 0.1% of the time (0 before fit)."""
        return self.bpfa.effective_dictionary_size

    @property
    def noise_std(self) -> float:
        """Posterior estimate of the observation noise standard deviation."""
        return self.bpfa.noise_std

    @property
    def usage_probabilities(self):
        """Per-atom usage probabilities pi_k (None before fit)."""
        return self.bpfa.usage_probabilities

    # --- Persistence ---------------------------------------------------------
    # infer() needs the dictionary plus the three posterior quantities the Gibbs
    # sweeps condition on (phi, alpha, pi), so the dictionary goes to .npy, the
    # hyperparams to JSON and the posterior state to an .npz — the same shape as
    # the LC-KSVD / frozen K-SVD adapters.
    _CONFIG_FILE = "bayesian_config.json"
    _DICT_FILE = "bayesian_dictionary.npy"
    _STATE_FILE = "bayesian_state.npz"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "dimensions": self.dimensions,
            "n_iter": self.n_iter,
            "n_infer_iter": self.n_infer_iter,
            "n_burnin": self.n_burnin,
            "n_collect": self.n_collect,
            "a0": self.a0,
            "b0": self.b0,
            "c0": self.c0,
            "d0": self.d0,
            "e0": self.e0,
            "f0": self.f0,
            "init_method": self.init_method,
            "seed": self.seed,
            # Provenance only — not a constructor argument. Records how many of
            # the K atoms the beta process actually kept for this fit.
            "effective_dictionary_size": self.effective_dictionary_size,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError("BayesianDL has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # BPFA is pure NumPy, so the dictionary is already an ndarray. Kept
        # duck-typed so a tensor-backed backend would still serialize.
        dictionary = self._dictionary
        if hasattr(dictionary, "detach"):
            dictionary = dictionary.detach().cpu().numpy()
        np.save(os.path.join(dirpath, self._DICT_FILE), np.asarray(dictionary))

        np.savez(
            os.path.join(dirpath, self._STATE_FILE),
            phi=np.asarray(self.bpfa._phi, dtype=np.float64),
            alpha=np.asarray(self.bpfa._alpha, dtype=np.float64),
            pi=np.asarray(self.bpfa._pi, dtype=np.float64),
        )

    @classmethod
    def load(cls, dirpath: str) -> "BayesianDL":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            dimensions=config["dimensions"],
            n_iter=config["n_iter"],
            n_infer_iter=config["n_infer_iter"],
            n_burnin=config["n_burnin"],
            n_collect=config["n_collect"],
            a0=config["a0"],
            b0=config["b0"],
            c0=config["c0"],
            d0=config["d0"],
            e0=config["e0"],
            f0=config["f0"],
            init_method=config["init_method"],
            seed=config.get("seed", 42),
        )

        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        learner._dictionary = components

        # Restore the core sampler's posterior state so infer() works: without
        # phi/alpha/pi the conditional for (Z, S) would fall back to the priors.
        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.bpfa._dictionary = components
        learner.bpfa._phi = float(state["phi"])
        learner.bpfa._alpha = float(state["alpha"])
        learner.bpfa._pi = state["pi"]
        return learner
