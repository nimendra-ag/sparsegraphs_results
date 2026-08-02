"""dict_learners/bayesian_gpu.py
===============================
Pipeline adapter for the GPU core (dict_learners/bayesian_dl_gpu.py) — the
device-side twin of `dict_learners/bayesian.py`, in the same relationship
`bayesian.py` has to `bayesian_dl.py`.

It is a separate learner class rather than a `use_gpu=True` flag on BayesianDL
for the same reason FDDLGPU is its own class: the arm it produces is its own
row in the results table (`wl_bayesian_gpu`), with its own artifact bundle and
its own saved config, and a bundle should say on disk which implementation
produced it.

Interface contract
------------------
Identical to BayesianDL — the pipeline
(utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

BPFA is unsupervised, so `y_train` is accepted and ignored, exactly as AKSVD
and BayesianDL do.

Differences from the CPU adapter
--------------------------------
1.  `device` / `dtype` / `infer_batch_size` are exposed. `device=None` selects
    CUDA when available and CPU otherwise, so this arm still *runs* without a
    GPU (it just loses the reason to exist).

2.  Seeding goes through torch's global RNG, not NumPy's, so a given seed
    reproduces this implementation exactly but does NOT reproduce the CPU
    adapter's sample path. See the parity note in the GPU core's docstring.

3.  Persistence writes the same three artifacts under different filenames, and
    stores plain NumPy — the GPU core already keeps its fitted state on the
    host — so a bundle trained on a GPU box loads on a CPU-only one.

4.  No core BPFA logic is modified. This file is purely an adapter layer plus
    persistence.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dict_learners.dict_learner import DictLearner
from dict_learners.bayesian_dl_gpu import BPFAGPU


class BayesianDLGPU(DictLearner):
    """Non-parametric Bayesian dictionary learner (BPFA) on GPU.

    Parameters
    ----------
    dimensions       : int   — maximum number of dictionary atoms (the core's K).
                               Named to match AKSVD's `dimensions`; the effective
                               size is inferred by the beta process prior.
    n_iter           : int   — Gibbs sweeps during dictionary learning.
    n_infer_iter     : int   — Gibbs sweeps when coding new data with D fixed.
    n_burnin         : int   — inference sweeps discarded before collecting.
    n_collect        : int   — post-burn-in samples averaged into the final code.
    a0, b0           : float — Beta prior on the atom-usage probabilities pi_k.
                               `b0=None` defaults to N/8 inside the core.
    c0, d0           : float — Gamma hyper-prior on the noise precision gamma_eps.
    e0, f0           : float — Gamma hyper-prior on the weight precision gamma_s.
    init_method      : str   — 'svd', 'svd_lowrank' (fast randomised init; only
                               the leading K components are ever used) or 'random'.
    device           : str   — 'cuda' / 'cpu'. None auto-selects CUDA if present.
    dtype            : str   — 'float32' (default) or 'float64' working precision
                               for the (P, N) arrays. The reductions feeding the
                               scalar conditionals are float64 either way.
    infer_batch_size : int   — encode inference data in batches of this many
                               samples to bound VRAM. Exact, not an
                               approximation: with D fixed the samples are
                               conditionally independent. None = one pass.
    seed             : int   — injected per run (Monte Carlo CV); passed through
                               as the core's `random_state`.
    verbose          : bool  — extra sampler diagnostics via logging.
    """

    _DTYPES = {"float32": torch.float32, "float64": torch.float64}

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
            device: str = None,
            dtype: str = "float32",
            infer_batch_size: int = None,
            seed: int = 42,
            verbose: bool = False,
    ):
        super().__init__(name="BayesianDLGPU")
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
        self.device = device
        self.dtype = dtype
        self.infer_batch_size = infer_batch_size
        self.seed = seed
        self.verbose = verbose

        if dtype not in self._DTYPES:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Use one of {sorted(self._DTYPES)}."
            )

        self.bpfa = BPFAGPU(
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
            device=self.device,
            dtype=self._DTYPES[self.dtype],
            infer_batch_size=self.infer_batch_size,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def fit(self, training_graph_embeddings, y_train=None):
        # y_train is ignored: BPFA is unsupervised. It is accepted only so the
        # dict-learner call site is uniform across supervised/unsupervised types.
        # The sampler seeds torch's global RNG itself (see BPFAGPU._seed_generators);
        # NumPy is seeded too because the CPU SVD fallback path goes through it.
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
    # sweeps condition on (phi, alpha, pi). Raw arrays go to .npy/.npz (no torch
    # dependency on load) and scalars to JSON — same shape as FDDLGPU.
    _CONFIG_FILE = "bayesian_gpu_config.json"
    _DICT_FILE = "bayesian_gpu_dictionary.npy"
    _STATE_FILE = "bayesian_gpu_state.npz"

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
            "dtype": self.dtype,
            "infer_batch_size": self.infer_batch_size,
            "seed": self.seed,
            # Provenance only — not constructor arguments. `device` records where
            # this fit actually ran (None means "whatever was available"), and
            # effective size records how many of the K atoms the prior kept.
            "device": str(self.bpfa.device),
            "effective_dictionary_size": self.effective_dictionary_size,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError("BayesianDLGPU has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # The GPU core keeps its fitted state on the host, so this is already a
        # NumPy array. Kept duck-typed in case that ever changes.
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
    def load(cls, dirpath: str) -> "BayesianDLGPU":
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
            # `device` is deliberately not restored: the saved value records
            # where training ran, but inference must bind to whatever hardware
            # the loading machine has. None re-runs the auto-select.
            dtype=config.get("dtype", "float32"),
            infer_batch_size=config.get("infer_batch_size"),
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
