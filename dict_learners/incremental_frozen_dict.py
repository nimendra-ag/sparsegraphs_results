# coding:utf-8
"""
Incremental Frozen Dictionary Learning
=======================================

Incrementally learn class-specific dictionaries, freezing all previously
learned atoms before training the next class.

Based on the approach in Carroll et al. 2017 ("Outlier Learning via
Augmented Frozen Dictionaries"):

    1. fit_base(X)   — Learn a base dictionary from one class using
                       standard ApproximateKSVD.  All atoms become frozen.
    2. add_class(X)  — Learn residual atoms for a new class using
                       FrozenKSVD.  Previous atoms stay frozen; new atoms
                       capture what frozen atoms cannot represent.
                       Repeat for each additional class.
    3. transform(X)  — Encode signals over the full combined dictionary.

After all classes:

    D = [D_base | D_class1 | D_class2 | ...]
         frozen   frozen      frozen
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp_gram

# In-repo core algorithms, not the PyPI `ksvd` package: `dict_learners/ksvd.py`
# is the same ApproximateKSVD every other arm (AKSVD, LC-KSVD) builds on, so the
# base stage here is bit-for-bit the standard K-SVD the paper compares against.
from dict_learners.ksvd import ApproximateKSVD
from dict_learners.frozen_ksvd import FrozenKSVD


class IncrementalFrozenDictionary(object):
    def __init__(
        self,
        n_components_base=96,
        n_components_residual=32,
        max_iter=10,
        tol=1e-6,
        transform_n_nonzero_coefs=10,
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
        """
        self.n_components_base = n_components_base
        self.n_components_residual = n_components_residual
        self.max_iter = max_iter
        self.tol = tol
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

        self.components_ = None
        self.class_boundaries_ = {}
        self._n_classes_added = 0

    def fit_base(self, X):
        """
        Learn the base dictionary from the base class data.

        Uses standard ApproximateKSVD — no frozen atoms at this stage.

        Parameters
        ----------
        X: shape = [n_samples, n_features]
            Training signals for the base class only.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        aksvd = ApproximateKSVD(
            n_components=self.n_components_base,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
        )
        aksvd.fit(X)

        self.components_ = aksvd.components_
        self.class_boundaries_[0] = (0, self.n_components_base)

        return self

    def add_class(self, X):
        """
        Learn residual atoms for a new class with all previous atoms frozen.

        The FrozenKSVD learner sparse-codes over [D_frozen | D_new] every
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

        X = np.asarray(X, dtype=np.float64)

        n_frozen = self.components_.shape[0]
        n_total = n_frozen + self.n_components_residual

        frozen_learner = FrozenKSVD(
            n_components=n_total,
            n_frozen=n_frozen,
            max_iter=self.max_iter,
            tol=self.tol,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
        )
        frozen_learner.fit(X, frozen_atoms=self.components_)

        # Verify the frozen-dictionary contract was honoured
        if not np.allclose(
            frozen_learner.components_[:n_frozen],
            self.components_,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Frozen atoms were modified during learning. "
                "This indicates a bug in FrozenKSVD._update_dict."
            )

        self._n_classes_added += 1
        self.class_boundaries_[self._n_classes_added] = (n_frozen, n_total)
        self.components_ = frozen_learner.components_

        return self

    def transform(self, X):
        """
        Encode X over the full combined dictionary using Gram OMP.

        Parameters
        ----------
        X: shape = [n_samples, n_features]

        Returns
        -------
        codes: shape = [n_samples, total_atoms]
        """
        if self.components_ is None:
            raise RuntimeError("Call fit_base() before transform().")

        X = np.asarray(X, dtype=np.float64)
        D = self.components_

        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T
