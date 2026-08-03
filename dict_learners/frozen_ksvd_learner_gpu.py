"""dict_learners/frozen_ksvd_learner_gpu.py
==========================================
Pipeline adapter for the GPU cores (dict_learners/frozen_ksvd_gpu.py, driven by
dict_learners/incremental_frozen_dict_gpu.py) — the device-side twin of
`dict_learners/frozen_ksvd_learner.py`, in the same relationship
`bayesian_gpu.py` has to `bayesian.py`.

It is a separate learner class rather than a `use_gpu=True` flag on
FrozenKSVDLearner for the same reason FDDLGPU and BayesianDLGPU are their own
classes: the arm it produces is its own row in the results table
(`wl_frozen_ksvd_gpu`), with its own artifact bundle and its own saved config,
and a bundle should say on disk which implementation produced it.

Interface contract
------------------
Identical to FrozenKSVDLearner — the pipeline
(utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

Frozen K-SVD is supervised — the labels decide which stage each sample trains —
so this raises when `y_train` is missing, exactly as the CPU adapter does.

Differences from the CPU adapter
--------------------------------
1.  `device` / `dtype` / `batch_size` are exposed. `device=None` selects CUDA
    when available and CPU otherwise, so this arm still *runs* without a GPU
    (it just loses the reason to exist).

2.  Seeding goes through torch's global RNG as well as NumPy's. The CPU
    adapter seeds NumPy alone because its only randomness is ARPACK's start
    vector inside `scipy.sparse.linalg.svds`; the GPU cores use a deterministic
    dense SVD, so the only draw left is the random-init fallback for a stage
    with fewer samples than atoms. A fixed seed reproduces this implementation
    exactly but does NOT reproduce the CPU adapter's dictionary — see the
    parity note in dict_learners/frozen_ksvd_gpu.py.

3.  Persistence writes the same three artifacts under different filenames, and
    stores plain NumPy — so a bundle trained on a GPU box loads on a CPU-only
    one.

4.  No core frozen K-SVD logic is modified. This file is purely an adapter
    layer plus persistence.

As in the CPU arm, this learner deliberately exposes no .D/.k: its dictionary
blocks are not equal-width (base 96 vs residual 32), so utils/mccv.py's SRC
arms — which slice a dictionary at a fixed stride — stay NaN, as they do for
AKSVD and LC-KSVD.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dict_learners.dict_learner import DictLearner
from dict_learners.incremental_frozen_dict_gpu import IncrementalFrozenDictionaryGPU


class FrozenKSVDLearnerGPU(DictLearner):
    """Incremental frozen-dictionary learner for the WL pipeline, on GPU.

    Parameters
    ----------
    n_components_base     : int   — atoms for the base (majority) class dictionary.
    n_components_residual : int   — new atoms learned for each additional class.
    max_iter              : int   — maximum K-SVD iterations per stage.
    tol                   : float — early-stop tolerance on ||X - gamma D||_F.
    n_non_zero_coefs      : int   — max nonzero coefficients per sparse code (OMP).
    base_label            : int   — class label used as the base dictionary.
                                    Defaults to -1 to match the WL encoder's
                                    majority-class convention; all other labels
                                    are added incrementally via add_class.
    device                : str   — 'cuda' / 'cpu'. None auto-selects CUDA if present.
    dtype                 : str   — 'float32' (default) or 'float64' working
                                    precision. float64 matches the CPU arm's
                                    precision at a large throughput cost on
                                    consumer GPUs; the OMP solves run over
                                    supports of ~10 atoms, so float32 is ample.
    batch_size            : int   — signals per Batch-OMP chunk, to bound VRAM.
                                    Exact, not an approximation: the signals are
                                    coded independently. None = one pass.
    seed                  : int   — injected per run (Monte Carlo CV) for
                                    reproducibility.
    """

    _DTYPES = {"float32": torch.float32, "float64": torch.float64}

    def __init__(
            self,
            n_components_base: int = 96,
            n_components_residual: int = 32,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
            base_label: int = -1,
            device: str = None,
            dtype: str = "float32",
            batch_size: int = None,
            seed: int = 42,
    ):
        super().__init__(name="FrozenKSVDGPU")
        self._dictionary = None
        self.n_components_base = n_components_base
        self.n_components_residual = n_components_residual
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.base_label = base_label
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.seed = seed

        if dtype not in self._DTYPES:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Use one of {sorted(self._DTYPES)}."
            )

        # Dataset labels in stage order (base class first). Set in fit(); also
        # written into the bundle manifest by the artifact store.
        self.classes_ = None

        self.incremental = IncrementalFrozenDictionaryGPU(
            n_components_base=self.n_components_base,
            n_components_residual=self.n_components_residual,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.n_non_zero_coefs,
            device=self.device,
            dtype=self._DTYPES[self.dtype],
            batch_size=self.batch_size,
        )

    def fit(self, training_graph_embeddings, y_train=None):
        """Build the dictionary incrementally: base class first, then each
        remaining class adds residual atoms on top of the frozen ones.

        Parameters
        ----------
        training_graph_embeddings: shape = [n_samples, n_features]
        y_train: shape = [n_samples]
            Class labels used to split the data by class.

        Returns
        -------
        self
        """
        # Frozen K-SVD is supervised: the labels are what partition the data
        # into the base stage and the per-class residual stages, so there is
        # nothing to fall back to.
        if y_train is None:
            raise ValueError(
                "FrozenKSVDLearnerGPU.fit requires y_train (it is a supervised learner)."
            )

        # Seed injected per run (Monte Carlo CV). Torch's global RNG is what the
        # random-init fallback draws from; NumPy is seeded too so any host-side
        # path stays reproducible alongside it.
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        print(f"Training {self.name} on context [{self.incremental.device}] "
              f"(base={self.n_components_base}, residual={self.n_components_residual}/class, "
              f"max_iter={self.max_iter}, dtype={self.dtype})...", flush=True)

        labels = np.asarray(y_train)
        unique_labels = np.unique(labels)

        # Determine base label — use configured base_label if present,
        # otherwise fall back to the first unique label
        if self.base_label in unique_labels:
            base_label = self.base_label
        else:
            base_label = unique_labels[0]

        # Stage 1: learn base dictionary from base class
        base_mask = labels == base_label
        base_data = training_graph_embeddings[base_mask]
        self.incremental.fit_base(base_data)
        stage_order = [base_label]

        # Stage 2+: add residual atoms for each remaining class
        for cls in unique_labels:
            if cls == base_label:
                continue
            cls_mask = labels == cls
            cls_data = training_graph_embeddings[cls_mask]
            self.incremental.add_class(cls_data)
            stage_order.append(cls)

        # Stage order, not np.unique order: this is the order the dictionary
        # blocks were appended in, so it lines up with class_boundaries_.
        self.classes_ = np.asarray(stage_order)
        # Keep the exported dictionary on the host: everything outside the
        # cores (persistence, manifest, atom counts) is NumPy.
        self._dictionary = self.incremental.components_.detach().cpu().numpy()
        return self

    def infer(self, infer_graph_embeddings):
        sparse_embeddings = self.incremental.transform(infer_graph_embeddings)
        return sparse_embeddings

    def n_atoms(self) -> int:
        # The true total is only known after fit(): it depends on how many
        # classes were actually present (base + residual per extra class).
        # utils/export.py and utils/mccv.py both call this after fitting; the
        # fallback covers the binary case so an unfitted learner can still
        # report a sensible size.
        if self._dictionary is not None:
            return int(self._dictionary.shape[0])
        return int(self.n_components_base + self.n_components_residual)

    # --- Persistence ---------------------------------------------------------
    # transform() only needs components_ (the combined dictionary) and the
    # hyperparams, so we save the dictionary as .npy and the config as JSON —
    # same shape as the CPU adapter, under GPU-specific filenames so the two
    # bundles are never confusable. The stage bookkeeping (class boundaries +
    # the class order they correspond to) rides along in an .npz so a restored
    # learner can still say which atoms came from which class.
    _CONFIG_FILE = "frozen_ksvd_gpu_config.json"
    _DICT_FILE = "frozen_ksvd_gpu_dictionary.npy"
    _STATE_FILE = "frozen_ksvd_gpu_state.npz"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "n_components_base": self.n_components_base,
            "n_components_residual": self.n_components_residual,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_non_zero_coefs": self.n_non_zero_coefs,
            "base_label": self.base_label,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "seed": self.seed,
            # Provenance only — not a constructor argument. Records where this
            # fit actually ran; `device=None` means "whatever was available".
            "device": str(self.incremental.device),
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError(
                "FrozenKSVDGPU has no dictionary to save; fit the learner first."
            )
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # fit() already brought the dictionary back to the host; kept duck-typed
        # so a device tensor would still serialize.
        dictionary = self._dictionary
        if hasattr(dictionary, "detach"):
            dictionary = dictionary.detach().cpu().numpy()
        np.save(os.path.join(dirpath, self._DICT_FILE), np.asarray(dictionary))

        # class_boundaries_ is {stage: (start, end)}; flatten to rows of
        # [stage, start, end] so it survives a plain, pickle-free .npz.
        boundaries = np.asarray(
            [[stage, start, end]
             for stage, (start, end) in sorted(self.incremental.class_boundaries_.items())],
            dtype=np.int64,
        )
        np.savez(
            os.path.join(dirpath, self._STATE_FILE),
            class_boundaries=boundaries,
            classes=np.asarray(self.classes_),
        )

    @classmethod
    def load(cls, dirpath: str) -> "FrozenKSVDLearnerGPU":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            n_components_base=config["n_components_base"],
            n_components_residual=config["n_components_residual"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            n_non_zero_coefs=config["n_non_zero_coefs"],
            base_label=config.get("base_label", -1),
            # `device` is deliberately not restored: the saved value records
            # where training ran, but inference must bind to whatever hardware
            # the loading machine has. None re-runs the auto-select.
            dtype=config.get("dtype", "float32"),
            batch_size=config.get("batch_size"),
            seed=config.get("seed", 42),
        )

        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        learner._dictionary = components

        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.classes_ = state["classes"]

        # Restore the incremental learner's state so infer() works. The
        # dictionary goes back onto the working device, since transform()
        # multiplies against it there.
        learner.incremental.components_ = torch.as_tensor(
            components,
            dtype=learner._DTYPES[learner.dtype],
            device=learner.incremental.device,
        )
        learner.incremental.class_boundaries_ = {
            int(stage): (int(start), int(end))
            for stage, start, end in state["class_boundaries"]
        }
        learner.incremental._n_classes_added = max(
            len(learner.incremental.class_boundaries_) - 1, 0
        )
        return learner
