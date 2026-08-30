# coding:utf-8
"""
Frozen K-SVD — GPU core algorithm.

PyTorch port of `dict_learners/ksvd.py` + `dict_learners/frozen_ksvd.py`. The
two CPU files are merged here because the frozen variant is a three-line delta
on ApproximateKSVD and both stages need the *same* Batch-OMP kernel, which is
the only genuinely new code on this side. The pipeline adapter (DictLearner
interface, seeding, persistence) lives in
`dict_learners/frozen_ksvd_learner_gpu.py`; the staged base/residual
orchestration in `dict_learners/incremental_frozen_dict_gpu.py`. This module
knows nothing about the WL pipeline, exactly as the NumPy cores do.

Where the time actually goes
----------------------------
Profiling the CPU arm, the K-SVD iteration splits roughly as:

  * sparse coding — `sklearn.linear_model.orthogonal_mp_gram` over all N
    training signals. sklearn solves the targets in a Python `for` loop, one
    Cholesky-updated least-squares per signal, so cost grows linearly in N with
    a large constant. This dominates, and it is what `batch_omp_gram` below
    rewrites: the same Batch-OMP recurrence (Rubinstein, Zibulevsky & Elad
    2008), but with the N signals advanced *together* — one batched triangular
    solve per OMP step instead of N sequential ones.
  * dictionary update — a K-length loop of rank-1 residual updates. It stays a
    Python loop here too (atom j's update sees the residual left by atoms
    0..j-1), but each step becomes two matmuls on the device.

Numerical parity
----------------
The math is a faithful port; the numbers will not match the CPU core
bit-for-bit, for three reasons, none of which changes what is being computed:

  1. Different SVD init. The CPU core calls `scipy.sparse.linalg.svds`, whose
     ARPACK start vector comes from the global NumPy RNG and whose k singular
     triplets come back in ascending order. This port uses a dense
     `torch.linalg.svd` and takes the leading k in descending order — the same
     subspace, different (deterministic) row order, and atom order is
     immaterial to the model.
  2. Different tie-breaking in OMP. `argmax` over equal correlations resolves
     differently in torch than in sklearn's loop.
  3. float32 by default (see `dtype`).

Run-to-run reproducibility *within* this implementation is exact for a fixed
seed, device and dtype.

Base algorithm reference:
    M. Aharon, M. Elad, and A. Bruckstein,
    "K-SVD: An Algorithm for Designing Overcomplete Dictionaries
     for Sparse Representation,"
    IEEE Trans. Signal Process., vol. 54, no. 11, pp. 4311-4322, 2006.

Batch-OMP reference:
    R. Rubinstein, M. Zibulevsky, and M. Elad,
    "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
     Matching Pursuit," Technion CS Tech. Report, 2008.
"""

from __future__ import annotations

import numpy as np
import torch

# Guard for the two places a divide-by-zero would silently poison the whole
# dictionary with NaN: the Cholesky diagonal in OMP, and the atom
# renormalisation in the dictionary update.
_EPS = 1e-10


def _resolve_device(device=None) -> torch.device:
    """CUDA when available and nothing was asked for, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_omp_gram(gram, alpha0, n_nonzero_coefs, batch_size=None):
    """Batch Orthogonal Matching Pursuit, all signals advanced in lockstep.

    Drop-in replacement for ``sklearn.linear_model.orthogonal_mp_gram`` as this
    codebase uses it, with the transpose already applied:

        sklearn : orthogonal_mp_gram(D @ D.T, D @ X.T, n_nonzero_coefs).T
        here    : batch_omp_gram(D @ D.T, X @ D.T, n_nonzero_coefs)

    Parameters
    ----------
    gram : (K, K) tensor
        D D^T. Symmetric by construction, which is what lets the residual
        correlations be refreshed with a single (N, K) @ (K, K) matmul.
    alpha0 : (N, K) tensor
        X D^T — the correlation of every signal with every atom.
    n_nonzero_coefs : int
        Atoms selected per signal. Exactly this many OMP steps are run (as
        sklearn does when given `n_nonzero_coefs`), clipped to K.
    batch_size : int or None
        Signals processed per chunk. Peak working memory is O(batch_size * K)
        plus the (batch_size, T, T) Cholesky factors, so this bounds VRAM
        without changing the result — the signals are independent.

    Returns
    -------
    (N, K) tensor of sparse codes, same dtype/device as `alpha0`.
    """
    n_samples, n_atoms = alpha0.shape
    n_nonzero = int(min(n_nonzero_coefs, n_atoms))
    if n_samples == 0 or n_nonzero <= 0:
        return torch.zeros_like(alpha0)

    if batch_size is None or batch_size >= n_samples:
        return _batch_omp_chunk(gram, alpha0, n_nonzero)

    out = torch.zeros_like(alpha0)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        out[start:stop] = _batch_omp_chunk(gram, alpha0[start:stop], n_nonzero)
    return out


def _batch_omp_chunk(gram, alpha0, n_nonzero):
    """One chunk of `batch_omp_gram`. See that function for the contract.

    Maintains, for every signal in the chunk, the Cholesky factor L of
    G[I, I] (I = atoms selected so far). Each step appends one row to L via a
    triangular solve, then re-solves for the coefficients on the enlarged
    support and refreshes the residual correlations
    ``alpha = alpha0 - G gamma``.
    """
    n_samples, n_atoms = alpha0.shape
    device, dtype = alpha0.device, alpha0.dtype

    alpha = alpha0.clone()
    # Support (indices, in selection order) and its Cholesky factor.
    support = torch.zeros(n_samples, n_nonzero, dtype=torch.long, device=device)
    chol = torch.zeros(n_samples, n_nonzero, n_nonzero, dtype=dtype, device=device)
    chol[:, 0, 0] = 1.0
    # Selected atoms are excluded from the next argmax. Their correlation is
    # ~0 by construction, but "~0" is not "0" in float32 and re-selecting an
    # atom would make G[I, I] singular.
    chosen = torch.zeros(n_samples, n_atoms, dtype=torch.bool, device=device)
    codes = torch.zeros(n_samples, n_atoms, dtype=dtype, device=device)

    for t in range(n_nonzero):
        # --- select the atom most correlated with the current residual ------
        k = alpha.abs().masked_fill(chosen, -1.0).argmax(dim=1)          # (N,)

        # --- extend the Cholesky factor: L_new = [[L, 0], [w', sqrt(1-w'w)]]
        if t > 0:
            g_new = gram[support[:, :t], k.unsqueeze(1)]                 # (N, t)
            w = torch.linalg.solve_triangular(
                chol[:, :t, :t], g_new.unsqueeze(-1), upper=False
            ).squeeze(-1)
            chol[:, t, :t] = w
            # 1 - w'w is the Schur complement; it goes non-positive only when
            # the new atom is numerically in the span of the current support.
            # Clamping keeps that signal's solve finite instead of poisoning
            # the batch with NaN — the atom simply contributes ~nothing.
            chol[:, t, t] = torch.sqrt(
                torch.clamp(1.0 - (w * w).sum(dim=1), min=_EPS)
            )

        support[:, t] = k
        chosen.scatter_(1, k.unsqueeze(1), True)

        # --- least squares on the support: (L L') gamma_I = alpha0[I] -------
        rhs = torch.gather(alpha0, 1, support[:, : t + 1]).unsqueeze(-1)
        chol_t = chol[:, : t + 1, : t + 1]
        z = torch.linalg.solve_triangular(chol_t, rhs, upper=False)
        gamma_i = torch.linalg.solve_triangular(
            chol_t.transpose(1, 2), z, upper=True
        ).squeeze(-1)

        codes.zero_()
        codes.scatter_(1, support[:, : t + 1], gamma_i)

        # --- refresh residual correlations ----------------------------------
        # alpha = alpha0 - G gamma, and G is symmetric, so the whole batch is
        # one matmul against the dense-but-mostly-zero code matrix. Cheaper in
        # memory than gathering the selected Gram columns per signal.
        if t + 1 < n_nonzero:
            alpha = alpha0 - codes @ gram

    return codes


class ApproximateKSVDGPU(object):
    """ApproximateKSVD (dict_learners/ksvd.py) with the linear algebra on device.

    Data layout is unchanged from the CPU core: X is (n_samples, n_features)
    and ``components_`` is (n_components, n_features), i.e. atoms are rows.
    """

    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None, device=None,
                 dtype=torch.float32, batch_size=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements.

        max_iter:
            Maximum number of iterations.

        tol:
            Tolerance for error.

        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target.

        device:
            'cuda' / 'cpu'. None auto-selects CUDA when available, so this
            class still runs (slowly) on a CPU-only box.

        dtype:
            Working precision. float32 by default — consumer GPUs run float64
            at a small fraction of their float32 throughput, and OMP's solves
            are over supports of ~10 atoms, well inside float32's comfort zone.
            Pass torch.float64 to match the CPU core's precision exactly.

        batch_size:
            Signals per OMP chunk; bounds VRAM without changing the result.
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.device = _resolve_device(device)
        self.dtype = dtype
        self.batch_size = batch_size

    # --- helpers -------------------------------------------------------------

    def _to_device(self, X):
        if torch.is_tensor(X):
            return X.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(
            np.asarray(X), dtype=self.dtype, device=self.device
        )

    def _n_nonzero(self, X):
        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])
        return n_nonzero_coefs

    # --- core steps ----------------------------------------------------------

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            D, gamma = _update_atom(X, D, gamma, j)
        return D, gamma

    def _initialize(self, X):
        # Same branch as the CPU core: fall back to random atoms when the data
        # cannot supply n_components singular vectors.
        if min(X.shape) < self.n_components:
            D = torch.randn(
                self.n_components, X.shape[1], device=self.device, dtype=self.dtype
            )
        else:
            D = _svd_atoms(X, self.n_components)
        return _normalize_rows(D)

    def _transform(self, D, X):
        gram = D @ D.T
        alpha0 = X @ D.T
        return batch_omp_gram(gram, alpha0, self._n_nonzero(X), self.batch_size)

    # --- public API ----------------------------------------------------------

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        X = self._to_device(X)
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = torch.linalg.norm(X - gamma @ D)
            # .item() syncs the device once per iteration — the same cost the
            # CPU core pays for free, and the only way to honour `tol`.
            if e.item() < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, self._to_device(X))


class FrozenKSVDGPU(ApproximateKSVDGPU):
    """ApproximateKSVDGPU with fixed (frozen) dictionary atoms.

    The GPU twin of `dict_learners/frozen_ksvd.py`, and the same minimal
    extension of the base algorithm:

        1. ``_initialize`` places frozen atoms in the first ``n_frozen`` rows
           of D and initialises the remaining rows normally.
        2. ``_update_dict`` starts its loop from ``n_frozen`` instead of 0,
           so frozen atoms are never modified.
        3. ``fit`` accepts an optional ``frozen_atoms`` argument.

    Everything else — data layout, sparse coding via Batch-OMP, convergence
    check, normalization — is inherited unchanged.
    """

    def __init__(self, n_components, n_frozen=0, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None, device=None,
                 dtype=torch.float32, batch_size=None):
        """
        Parameters
        ----------
        n_components:
            Total number of dictionary elements (frozen + learnable).

        n_frozen:
            Number of leading dictionary atoms to keep fixed.
            Must be >= 0 and strictly less than n_components.

        Remaining parameters: see ApproximateKSVDGPU.
        """
        if n_frozen < 0:
            raise ValueError("n_frozen must be >= 0.")
        if n_frozen >= n_components:
            raise ValueError("n_frozen must be strictly less than n_components.")

        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            transform_n_nonzero_coefs=transform_n_nonzero_coefs,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.n_frozen = n_frozen

    def _update_dict(self, X, D, gamma):
        # --- FROZEN LOGIC: start from n_frozen instead of 0 ---
        for j in range(self.n_frozen, self.n_components):
            D, gamma = _update_atom(X, D, gamma, j)
        return D, gamma

    def _initialize(self, X, frozen_atoms=None):
        n_features = X.shape[1]
        n_learnable = self.n_components - self.n_frozen

        # --- Initialise learnable atoms (same logic as ApproximateKSVDGPU) ---
        if min(X.shape) < n_learnable:
            D_learnable = torch.randn(
                n_learnable, n_features, device=self.device, dtype=self.dtype
            )
        else:
            D_learnable = _svd_atoms(X, n_learnable)
        D_learnable = _normalize_rows(D_learnable)

        # --- FROZEN LOGIC: prepend frozen atoms ---
        if self.n_frozen > 0:
            if frozen_atoms is not None:
                if tuple(frozen_atoms.shape) != (self.n_frozen, n_features):
                    raise ValueError(
                        f"frozen_atoms shape {tuple(frozen_atoms.shape)} doesn't "
                        f"match expected ({self.n_frozen}, {n_features})."
                    )
                D_frozen = self._to_device(frozen_atoms).clone()
            else:
                raise ValueError(
                    "frozen_atoms must be provided when n_frozen > 0. "
                    "Freezing random atoms has no practical value."
                )

            D_frozen = _normalize_rows(D_frozen)
            D = torch.cat([D_frozen, D_learnable], dim=0)
        else:
            D = D_learnable

        return D

    def fit(self, X, frozen_atoms=None):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]

        frozen_atoms: shape = [n_frozen, n_features], optional
            Pre-defined atoms to freeze. Required when n_frozen > 0.
        """
        X = self._to_device(X)
        D = self._initialize(X, frozen_atoms)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = torch.linalg.norm(X - gamma @ D)
            if e.item() < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self


# --- shared primitives -------------------------------------------------------
# Module-level so the base and frozen classes provably run the *same* atom
# update; the only difference between them is the range of j they run it over.


def _update_atom(X, D, gamma, j):
    """One approximate-K-SVD rank-1 atom update, in place on D and gamma.

    Line-for-line the body of the CPU core's `_update_dict` loop, including
    its `gamma[:, j] > 0` activity test (strictly positive, not nonzero — kept
    as-is so the two arms optimise the identical objective).
    """
    I = gamma[:, j] > 0
    if not bool(torch.any(I)):
        return D, gamma

    d_old = D[j, :].clone()
    D[j, :] = 0
    g = gamma[I, j]
    r = X[I, :] - gamma[I, :] @ D
    d = r.T @ g
    norm_d = torch.linalg.norm(d)
    # The CPU core divides unguarded and yields a NaN atom, which then spreads
    # through every subsequent Gram matrix. A zero-norm residual means this
    # atom's signals are already perfectly explained by the others, so the
    # informative thing to do is put the atom back and move on.
    if norm_d.item() < _EPS:
        D[j, :] = d_old
        return D, gamma
    d = d / norm_d
    g = r @ d
    D[j, :] = d
    gamma[I, j] = g
    return D, gamma


def _svd_atoms(X, k):
    """Leading `k` right singular vectors of X, scaled by their singular values.

    The CPU core's `diag(s) @ vt` from `scipy.sparse.linalg.svds(X, k=k)`.
    torch returns the full (thin) decomposition in descending order, so the
    truncation is a slice; svds returns its k triplets ascending. Same
    subspace, different row order — see the parity note in the module
    docstring.
    """
    _, s, vh = torch.linalg.svd(X, full_matrices=False)
    return torch.diag(s[:k]) @ vh[:k]


def _normalize_rows(D):
    """Scale every atom to unit L2 norm, guarding the all-zero row."""
    norms = torch.linalg.norm(D, dim=1, keepdim=True)
    return D / torch.clamp(norms, min=_EPS)
