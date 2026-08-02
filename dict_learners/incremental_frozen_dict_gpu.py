# coding:utf-8
"""
Incremental Frozen Dictionary Learning — GPU
============================================

PyTorch port of `dict_learners/incremental_frozen_dict.py`. Same staging, same
frozen-atom contract, same combined dictionary; only the linear algebra moves
to the device (see `dict_learners/frozen_ksvd_gpu.py` for the ported cores and
the numerical-parity note).

Incrementally learn class-specific dictionaries, freezing all previously
learned atoms before training the next class.

Based on the approach in Carroll et al. 2017 ("Outlier Learning via
Augmented Frozen Dictionaries"):

    1. fit_base(X)   — Learn a base dictionary from one class using
                       standard ApproximateKSVDGPU.  All atoms become frozen.
    2. add_class(X)  — Learn residual atoms for a new class using
                       FrozenKSVDGPU.  Previous atoms stay frozen; new atoms
                       capture what frozen atoms cannot represent.
                       Repeat for each additional class.
    3. transform(X)  — Encode signals over the full combined dictionary.

After all classes:

    D = [D_base | D_class1 | D_class2 | ...]
         frozen   frozen      frozen

`components_` is kept as a torch tensor on the working device between stages —
the next stage feeds it straight back in as `frozen_atoms`, so round-tripping
it through host memory would be pure overhead. The adapter
(`dict_learners/frozen_ksvd_learner_gpu.py`) is what moves it back to NumPy for
persistence and for the downstream Evaluator.
"""

import numpy as np
import torch

# In-repo core algorithms, not the PyPI `ksvd` package: `frozen_ksvd_gpu.py`
# ports the same ApproximateKSVD every other arm (AKSVD, LC-KSVD) builds on, so
# the base stage here is the standard K-SVD the paper compares against, on the
# device.
from dict_learners.frozen_ksvd_gpu import (
    ApproximateKSVDGPU,
    FrozenKSVDGPU,
    batch_omp_gram,
    _resolve_device,
)


class IncrementalFrozenDictionaryGPU(object):
    def __init__(
        self,
        n_components_base=96,
        n_components_residual=32,
        max_iter=10,
        tol=1e-6,
        transform_n_nonzero_coefs=10,
        device=None,
        dtype=torch.float32,
        batch_size=None,
    ):
        """
        Parameters
        ----------
        n_components_base:
            Number of dictionary atoms for the base class.

        n_components_residual:
            Number of new atoms to learn for each subsequent class.

        max_iter:
            Maximum K-SVD iterations per stage.

        tol:
            Convergence tolerance per stage.

        transform_n_nonzero_coefs:
            Sparsity level for OMP encoding.

        device:
            'cuda' / 'cpu'. None auto-selects CUDA when available.

        dtype:
            Working precision (torch.float32 by default, torch.float64 to match
            the CPU arm's precision).

        batch_size:
            Signals per Batch-OMP chunk; bounds VRAM without changing results.
        """
        self.n_components_base = n_components_base
        self.n_components_residual = n_components_residual
        self.max_iter = max_iter
        self.tol = tol
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.device = _resolve_device(device)
        self.dtype = dtype
        self.batch_size = batch_size

        self.components_ = None
        self.class_boundaries_ = {}
        self._n_classes_added = 0

    # --- helpers -------------------------------------------------------------

    def _to_device(self, X):
        """(n_samples, n_features) host or device array -> working tensor.

        The CPU core casts to float64 here; this one casts to `self.dtype`,
        which is the single place the precision trade-off is made.
        """
        if torch.is_tensor(X):
            return X.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)

    def _stage_kwargs(self):
        return dict(
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
        )

    # --- stages --------------------------------------------------------------

    def fit_base(self, X):
        """
        Learn the base dictionary from the base class data.

        Uses standard ApproximateKSVDGPU — no frozen atoms at this stage.

        Parameters
        ----------
        X: shape = [n_samples, n_features]
            Training signals for the base class only.

        Returns
        -------
        self
        """
        X = self._to_device(X)

        aksvd = ApproximateKSVDGPU(
            n_components=self.n_components_base,
            **self._stage_kwargs(),
        )
        aksvd.fit(X)

        self.components_ = aksvd.components_
        self.class_boundaries_[0] = (0, self.n_components_base)

        return self

    def add_class(self, X):
        """
        Learn residual atoms for a new class with all previous atoms frozen.

        The FrozenKSVDGPU learner sparse-codes over [D_frozen | D_new] every
        iteration but only updates D_new.  After training, D_new is
        appended to the combined dictionary and also becomes frozen for
        any future add_class call.

        Parameters
        ----------
        X: shape = [n_samples, n_features]
            Training signals for this class only.

        Returns
        -------
        self
        """
        if self.components_ is None:
            raise RuntimeError("Call fit_base() before add_class().")

        X = self._to_device(X)

        n_frozen = self.components_.shape[0]
        n_total = n_frozen + self.n_components_residual

        frozen_learner = FrozenKSVDGPU(
            n_components=n_total,
            n_frozen=n_frozen,
            **self._stage_kwargs(),
        )
        frozen_learner.fit(X, frozen_atoms=self.components_)

        # Verify the frozen-dictionary contract was honoured. The tolerance is
        # the CPU core's; the frozen block is copied, not recomputed, so this
        # is an exact-equality check in practice either way.
        if not torch.allclose(
            frozen_learner.components_[:n_frozen],
            self.components_,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Frozen atoms were modified during learning. "
                "This indicates a bug in FrozenKSVDGPU._update_dict."
            )

        self._n_classes_added += 1
        self.class_boundaries_[self._n_classes_added] = (n_frozen, n_total)
        self.components_ = frozen_learner.components_

        return self

    def transform(self, X):
        """
        Encode X over the full combined dictionary using Batch-OMP.

        Parameters
        ----------
        X: shape = [n_samples, n_features]

        Returns
        -------
        codes: shape = [n_samples, total_atoms] — NumPy, on the host, because
        the downstream Evaluator trains sklearn models on these.
        """
        if self.components_ is None:
            raise RuntimeError("Call fit_base() before transform().")

        X = self._to_device(X)
        D = self.components_.to(device=self.device, dtype=self.dtype)

        gram = D @ D.T
        alpha0 = X @ D.T

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        codes = batch_omp_gram(gram, alpha0, n_nonzero_coefs, self.batch_size)
        return codes.detach().cpu().numpy()
