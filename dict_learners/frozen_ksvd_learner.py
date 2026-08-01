"""dict_learners/frozen_ksvd_learner.py
======================================
Pipeline adapter that wraps the incremental frozen-dictionary learner
(dict_learners/incremental_frozen_dict.py, built on dict_learners/frozen_ksvd.py)
into the DictLearner interface used by the WL pipeline — the same relationship
`dict_learners/aksvd.py` has to `dict_learners/ksvd.py`.

Interface contract
------------------
The pipeline (utils/pipeline.fit_encoder_and_dictionary) calls:

    dict_learner.fit(training_graph_embeddings=emb, y_train=y)
    X = dict_learner.infer(embeddings)
    n = dict_learner.n_atoms()

so this adapter uses the *same* signature as every other learner. Frozen K-SVD
is supervised — the labels decide which stage each sample trains — so it raises
when `y_train` is missing (mirroring LCKSVDLearner/FDDLGPU) rather than
silently ignoring it the way unsupervised AKSVD does.

Key design decisions
--------------------
1.  Constructor parameters follow the AKSVD adapter's vocabulary
    (`max_iter`, `tol`, `n_non_zero_coefs`, `seed`). AKSVD's single
    `dimensions` splits in two here because the dictionary is built in
    stages: `n_components_base` atoms for the base class, then
    `n_components_residual` further atoms per additional class.

2.  infer() returns *sparse codes* (N, total_atoms) over the full combined
    dictionary. The downstream Evaluator trains its own ML models
    (LR, GB, SVM, RF) on those codes, exactly as with AKSVD.

3.  The embeddings from WL are (N, n) row-major, which is already the layout
    ApproximateKSVD/FrozenKSVD expect, so — unlike the LC-KSVD adapter — no
    transposition is needed and `components_` stays (total_atoms, n).

4.  `classes_` records the dataset labels in *stage order* (base class first,
    then each added class), so it lines up with `class_boundaries_` and the
    block layout of the dictionary. utils/mccv.py only runs SRC on a learner
    exposing .D/.k/.classes_; this learner deliberately exposes no .D/.k
    because its blocks are not equal-width (base 96 vs residual 32), so SRC's
    fixed-stride slicing would not line up. Those arms stay NaN, as they do
    for AKSVD and LC-KSVD.

5.  No core frozen K-SVD logic is modified. This file is purely an adapter
    layer plus persistence.
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dict_learners.dict_learner import DictLearner
from dict_learners.incremental_frozen_dict import IncrementalFrozenDictionary


class FrozenKSVDLearner(DictLearner):
    """Incremental frozen-dictionary learner for the WL pipeline.

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
    seed                  : int   — injected per run (Monte Carlo CV) for
                                    reproducibility.
    """

    def __init__(
            self,
            n_components_base: int = 96,
            n_components_residual: int = 32,
            max_iter: int = 10,
            tol: float = 1e-6,
            n_non_zero_coefs: int = 10,
            base_label: int = -1,
            seed: int = 42,
    ):
        super().__init__(name="FrozenKSVD")
        self._dictionary = None
        self.n_components_base = n_components_base
        self.n_components_residual = n_components_residual
        self.max_iter = max_iter
        self.tol = tol
        self.n_non_zero_coefs = n_non_zero_coefs
        self.base_label = base_label
        self.seed = seed

        # Dataset labels in stage order (base class first). Set in fit(); also
        # written into the bundle manifest by the artifact store.
        self.classes_ = None

        self.incremental = IncrementalFrozenDictionary(
            n_components_base=self.n_components_base,
            n_components_residual=self.n_components_residual,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.n_non_zero_coefs,
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
                "FrozenKSVDLearner.fit requires y_train (it is a supervised learner)."
            )

        # Seed injected per run (Monte Carlo CV). Both stages call
        # scipy.sparse.linalg.svds for atom init, which draws its ARPACK
        # starting vector from the global NumPy RNG, so seeding globally here
        # is what makes a run reproducible.
        np.random.seed(self.seed)

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
        self._dictionary = self.incremental.components_
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
    # same shape as the AKSVD adapter. The stage bookkeeping (class boundaries
    # + the class order they correspond to) rides along in an .npz so a restored
    # learner can still say which atoms came from which class.
    _CONFIG_FILE = "frozen_ksvd_config.json"
    _DICT_FILE = "frozen_ksvd_dictionary.npy"
    _STATE_FILE = "frozen_ksvd_state.npz"

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
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self._dictionary is None:
            raise ValueError(
                "FrozenKSVD has no dictionary to save; fit the learner first."
            )
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # FrozenKSVD is pure NumPy, so components_ is already an ndarray.
        # Kept duck-typed so a tensor-backed backend would still serialize.
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
    def load(cls, dirpath: str) -> "FrozenKSVDLearner":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        learner = cls(
            n_components_base=config["n_components_base"],
            n_components_residual=config["n_components_residual"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            n_non_zero_coefs=config["n_non_zero_coefs"],
            base_label=config.get("base_label", -1),
            seed=config.get("seed", 42),
        )

        components = np.load(os.path.join(dirpath, cls._DICT_FILE))
        learner._dictionary = components

        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.classes_ = state["classes"]

        # Restore the incremental learner's state so infer() works.
        learner.incremental.components_ = components
        learner.incremental.class_boundaries_ = {
            int(stage): (int(start), int(end))
            for stage, start, end in state["class_boundaries"]
        }
        learner.incremental._n_classes_added = max(
            len(learner.incremental.class_boundaries_) - 1, 0
        )
        return learner
