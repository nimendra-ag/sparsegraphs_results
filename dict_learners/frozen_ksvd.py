# coding:utf-8
"""
Frozen K-SVD: ApproximateKSVD with fixed (frozen) dictionary atoms.

This is a minimal extension of the ApproximateKSVD algorithm. The only
modifications are:

    1. ``_initialize`` places frozen atoms in the first ``n_frozen`` rows
       of D and initialises the remaining rows normally.
    2. ``_update_dict`` starts its loop from ``n_frozen`` instead of 0,
       so frozen atoms are never modified.
    3. ``fit`` accepts an optional ``frozen_atoms`` argument.

Everything else — data layout, sparse coding via Gram OMP, convergence
check, normalization — is identical to the original ApproximateKSVD.

Base algorithm reference:
    M. Aharon, M. Elad, and A. Bruckstein,
    "K-SVD: An Algorithm for Designing Overcomplete Dictionaries
     for Sparse Representation,"
    IEEE Trans. Signal Process., vol. 54, no. 11, pp. 4311-4322, 2006.
"""

import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram


class FrozenKSVD(object):
    def __init__(self, n_components, n_frozen=0, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Total number of dictionary elements (frozen + learnable).

        n_frozen:
            Number of leading dictionary atoms to keep fixed.
            Must be >= 0 and strictly less than n_components.

        max_iter:
            Maximum number of iterations.

        tol:
            Tolerance for error.

        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target.
        """
        if n_frozen < 0:
            raise ValueError("n_frozen must be >= 0.")
        if n_frozen >= n_components:
            raise ValueError("n_frozen must be strictly less than n_components.")

        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_frozen = n_frozen
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        # --- FROZEN LOGIC: start from n_frozen instead of 0 ---
        for j in range(self.n_frozen, self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X, frozen_atoms=None):
        n_features = X.shape[1]
        n_learnable = self.n_components - self.n_frozen

        # --- Initialise learnable atoms (same logic as ApproximateKSVD) ---
        if min(X.shape) < n_learnable:
            D_learnable = np.random.randn(n_learnable, n_features)
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=n_learnable)
            D_learnable = np.dot(np.diag(s), vt)
        D_learnable /= np.linalg.norm(D_learnable, axis=1)[:, np.newaxis]

        # --- FROZEN LOGIC: prepend frozen atoms ---
        if self.n_frozen > 0:
            if frozen_atoms is not None:
                if frozen_atoms.shape != (self.n_frozen, n_features):
                    raise ValueError(
                        f"frozen_atoms shape {frozen_atoms.shape} doesn't "
                        f"match expected ({self.n_frozen}, {n_features})."
                    )
                D_frozen = frozen_atoms.copy()
            else:
                raise ValueError(
                    "frozen_atoms must be provided when n_frozen > 0. "
                    "Freezing random atoms has no practical value."
                )

            D_frozen /= np.linalg.norm(D_frozen, axis=1)[:, np.newaxis]
            D = np.vstack([D_frozen, D_learnable])
        else:
            D = D_learnable

        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X, frozen_atoms=None):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]

        frozen_atoms: shape = [n_frozen, n_features], optional
            Pre-defined atoms to freeze. Required when n_frozen > 0.
        """
        D = self._initialize(X, frozen_atoms)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)
