"""
LC-KSVD: Label Consistent K-SVD  —  Approximate K-SVD variant
==============================================================
Implementation based on:
  [1] Jiang, Z., Lin, Z., Davis, L.S. (2011).
      "Learning a Discriminative Dictionary for Sparse Coding via Label Consistent K-SVD."
      CVPR 2011, pp. 1697-1704.

  [2] Jiang, Z., Lin, Z., Davis, L.S. (2013).
      "Label Consistent K-SVD: Learning a Discriminative Dictionary for Recognition."
      IEEE TPAMI, 35(11), pp. 2651-2664.

Overview of the method
----------------------
Standard K-SVD learns a dictionary D purely for signal reconstruction:
    min_{D, X}  ||Y - DX||_F^2     s.t. ||x_i||_0 <= T

LC-KSVD extends this by stacking two extra supervised terms into the
same matrix equation so that the original K-SVD solver can handle it
unchanged:

LC-KSVD1 adds a "discriminative sparse-code error":
    min_{D, A, X}  ||Y - DX||_F^2  +  alpha * ||Q - AX||_F^2
                   s.t. ||x_i||_0 <= T

LC-KSVD2 additionally adds a classification error:
    min_{D, W, A, X}  ||Y - DX||_F^2
                    + alpha * ||Q - AX||_F^2
                    + beta  * ||H - WX||_F^2
                   s.t. ||x_i||_0 <= T

The key trick (Section 3.3 in [2]) stacks everything into one K-SVD
problem on augmented matrices Y_new and D_new with the same X.

Approximate K-SVD variant
--------------------------
This version replaces the two original sub-routines with their
Approximate K-SVD equivalents from Rubinstein et al. (2008),
which is what the `ksvd` library's ApproximateKSVD class uses:

  Sparse coding   : orthogonal_mp_gram (sklearn) — Gram-space OMP.
                    Precomputes D @ D^T and D @ Y^T once per iteration,
                    then solves all N signals in projected space.
                    Much faster than signal-space OMP for large N.

  Dictionary update: power-method rank-1 update instead of truncated SVD.
                    For each atom k, the new atom is computed as a
                    weighted sum of residual rows (r^T g), then
                    normalised. Coefficients are updated as r @ d.
                    No SVD call — faster per atom.

  Initialisation  : per-class SVD-based init (scipy.sparse.linalg.svds)
                    instead of per-class random K-SVD.

  Convergence     : tolerance-based early stopping on reconstruction
                    error, plus a max iteration cap.

Everything else — the LC-KSVD augmented system, build_label_matrix,
build_discriminative_codes, ridge_regression, extract_and_renorm,
the LCKSVD class and all its methods — is identical to before.

Convention note
---------------
  D  : (n, K)  — atoms are COLUMNS  (our convention, matches the paper)
  Y  : (n, N)  — signals are COLUMNS
  X  : (K, N)  — sparse codes, one column per signal

  The AKSVD library uses the transposed convention (atoms are rows).
  We adapt the update formula here so the maths is identical but the
  array layout matches our pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse.linalg
from numpy.linalg import norm
from sklearn.linear_model import orthogonal_mp_gram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approximate K-SVD dictionary update
# ---------------------------------------------------------------------------

def aksvd_update(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    sparsity: int,
    n_iter: int,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Approximate K-SVD iterations on Y ≈ D @ X.

    Mirrors ApproximateKSVD._update_dict() from the library, adapted
    to our column-major convention (D columns, Y columns, X columns).

    For each atom k the update is:
      1. Find signals that use atom k:  I = X[k, :] != 0
      2. Compute partial residual:      R = Y[:, I] - D @ X[:, I]  +  d_k x_k_I^T
                                          = Y[:, I] - (D without d_k) @ X[:, I]
      3. New atom:     d_k = R @ g / ||R @ g||   where g = X[k, I]  (coefficients)
      4. New codes:    X[k, I] = R^T @ d_k

    This is a power-method rank-1 approximation: one step toward the
    best rank-1 factor of R, without computing a full SVD.

    Parameters
    ----------
    Y        : (m, N)  signal (or augmented) matrix
    D        : (m, K)  dictionary — columns are atoms
    X        : (K, N)  sparse codes
    sparsity : int     sparsity level T
    n_iter   : int     maximum number of outer iterations
    tol      : float   early-stop when ||Y - DX||_F < tol

    Returns
    -------
    D : (m, K)  updated dictionary
    X : (K, N)  updated sparse codes
    """
    K = D.shape[1]

    for iteration in range(n_iter):
        # ---- Sparse coding step (sklearn orthogonal_mp_gram) -------------
        # Mirrors ApproximateKSVD._transform() directly.
        # D is (m, K) columns; library uses (K, m) rows — transpositions
        # below produce the identical Gram matrix and projections.
        col_norms = norm(D, axis=0, keepdims=True)   # (1, K)
        col_norms[col_norms == 0] = 1.0
        D_normed = D / col_norms
        gram = D_normed.T @ D_normed                 # (K, K)  — mirrors D.dot(D.T)
        Xy   = D_normed.T @ Y                        # (K, N)  — mirrors D.dot(X.T)
        X    = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=sparsity)  # (K, N)

        # ---- Tolerance check (mirrors ApproximateKSVD.fit) ---------------
        # Check reconstruction error before the dictionary update.
        # If already within tolerance, stop early.
        recon_err = norm(Y - D_normed @ X)
        if recon_err < tol:
            logger.debug(
                "Early stop at iter %d / %d  (err=%.6f < tol=%.6f)",
                iteration + 1, n_iter, recon_err, tol,
            )
            D = D_normed   # keep the normalised version as the final D
            break

        # ---- Dictionary update step (Approximate K-SVD) ------------------
        # Work with normalised D so atom norms stay controlled.
        D = D_normed.copy()

        for k in range(K):
            # Boolean mask of signals that have a nonzero coefficient for atom k
            # Mirrors:  I = gamma[:, j] > 0  in the library
            # (our X[k,:] plays the role of gamma[:,j])
            I = X[k, :] != 0        # (N,) boolean mask
            if not np.any(I):
                # Unused atom — reinitialise with the worst-reconstructed signal.
                # This heuristic is not in the library but is standard practice.
                residuals = Y - D @ X
                err_norms = norm(residuals, axis=0)
                worst     = int(np.argmax(err_norms))
                new_atom  = Y[:, worst].copy()
                n_val     = norm(new_atom)
                D[:, k]   = new_atom / n_val if n_val > 1e-10 else new_atom
                continue

            # Coefficients of atom k for the signals that use it: g = X[k, I]
            # Mirrors:  g = gamma[I, j].T
            g = X[k, I]                             # (nnz,)

            # Partial residual: Y[:, I] minus contribution of ALL atoms
            # then add back atom k's own contribution so we isolate what
            # atom k should represent.
            # Mirrors:  r = X[I, :] - gamma[I, :].dot(D)
            #                         ^^^^^ this is the full reconstruction
            # In our convention (columns):
            #   r = (Y[:, I] - D @ X[:, I]).T  +  outer(g, d_k)  — then .T back
            # Equivalently, zero atom k out temporarily:
            d_k_saved = D[:, k].copy()
            D[:, k]   = 0.0
            R = Y[:, I] - D @ X[:, I]              # (m, nnz)  partial residual
            D[:, k] = d_k_saved                    # restore

            # New atom: weighted sum of residual columns, then normalise.
            # Mirrors:  d = r.T.dot(g) then d /= norm(d)
            # In our convention: R is (m, nnz), g is (nnz,)
            #   d_new = R @ g  then normalise
            d_new = R @ g                           # (m,)
            n_val = norm(d_new)
            if n_val < 1e-10:
                continue                            # degenerate — skip update
            d_new /= n_val

            # New coefficients for the signals that use atom k.
            # Mirrors:  g = r.dot(d)  in the library (r is (nnz, m), d is (m,))
            # In our convention: R is (m, nnz), d_new is (m,)
            #   new_g = R^T @ d_new
            D[:, k]  = d_new
            X[k, I]  = R.T @ d_new                 # (nnz,)

        logger.debug("AKSVD iter %d / %d done", iteration + 1, n_iter)

    return D, X


# ---------------------------------------------------------------------------
# Label / discriminative-code helpers  (unchanged)
# ---------------------------------------------------------------------------

def build_label_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build one-hot label matrix H ∈ {0,1}^{num_classes × N}.

    Parameters
    ----------
    labels      : (N,) integer class indices, 0-based
    num_classes : int

    Returns
    -------
    H : (num_classes, N)
    """
    N = labels.shape[0]
    H = np.zeros((num_classes, N))
    H[labels, np.arange(N)] = 1.0
    return H


def build_discriminative_codes(
    labels: np.ndarray,
    atom_labels: np.ndarray,
) -> np.ndarray:
    """
    Build discriminative target sparse codes Q ∈ {0,1}^{K × N}.

    Q[k, i] = 1 iff atom k and signal y_i share the same class label.
    This is the core of the label consistency constraint (Section 3.1 in [1]).

    Parameters
    ----------
    labels      : (N,) class index for each training signal
    atom_labels : (K,) class index for each dictionary atom

    Returns
    -------
    Q : (K, N)
    """
    K = atom_labels.shape[0]
    N = labels.shape[0]
    Q = np.zeros((K, N))
    for k in range(K):
        Q[k, labels == atom_labels[k]] = 1.0
    return Q


def init_atom_labels(
    labels: np.ndarray,
    num_classes: int,
    K: int,
) -> np.ndarray:
    """
    Assign a class label to each dictionary atom uniformly.

    Atoms are distributed across classes as evenly as possible.
    (Section 3.3.1 in [1])

    Parameters
    ----------
    labels      : unused here, kept for API consistency
    num_classes : int
    K           : total number of dictionary atoms

    Returns
    -------
    atom_labels : (K,) integer class indices
    """
    atoms_per_class = K // num_classes
    remainder       = K % num_classes
    atom_labels     = []
    for c in range(num_classes):
        count = atoms_per_class + (1 if c < remainder else 0)
        atom_labels.extend([c] * count)
    return np.array(atom_labels, dtype=int)


# ---------------------------------------------------------------------------
# Dictionary initialisation  (SVD-based, per class)
# ---------------------------------------------------------------------------

def init_dictionary_svd(
    Y: np.ndarray,
    labels: np.ndarray,
    atom_labels: np.ndarray,
) -> np.ndarray:
    """
    Initialise D^(0) using per-class truncated SVD.

    Mirrors ApproximateKSVD._initialize():
        u, s, vt = scipy.sparse.linalg.svds(X, k=n_components)
        D = diag(s) @ vt
        D /= row_norms

    Applied per class: for each class c, run SVDs on the subset of
    training signals Y[:, labels==c] to initialise the K_c atoms
    assigned to that class.

    If a class has fewer signals than its atom count, fall back to
    random Gaussian atoms for that class (same fallback as the library).

    Parameters
    ----------
    Y           : (n, N)  all training signals
    labels      : (N,)    class index per signal
    atom_labels : (K,)    class index per atom

    Returns
    -------
    D0 : (n, K)  initial dictionary, atoms are columns
    """
    n = Y.shape[0]
    K = atom_labels.shape[0]
    D0          = np.zeros((n, K))
    num_classes = int(atom_labels.max()) + 1
    rng         = np.random.default_rng(seed=42)

    for c in range(num_classes):
        signal_mask = labels == c
        atom_mask   = atom_labels == c
        Y_c         = Y[:, signal_mask]          # (n, N_c)
        K_c         = int(atom_mask.sum())

        if Y_c.shape[1] == 0 or K_c == 0:
            continue

        if Y_c.shape[1] < K_c:
            # Fewer signals than atoms for this class — random fallback
            # (mirrors the library's  if min(X.shape) < n_components  branch)
            D_c = rng.standard_normal((n, K_c))
        else:
            # SVD-based init — mirrors _initialize():
            #   u, s, vt = svds(X, k=K_c)     where X is (N_c, n) row-major
            #   D = diag(s) @ vt               → (K_c, n)
            # In our column convention X_c = Y_c.T is (N_c, n):
            try:
                _, s, vt = scipy.sparse.linalg.svds(Y_c.T, k=K_c)
                # svds returns singular values in ascending order — reverse
                s  = s[::-1]
                vt = vt[::-1, :]
                # D_c_rows = diag(s) @ vt  →  (K_c, n)
                D_c_rows = np.diag(s) @ vt          # (K_c, n)
                D_c      = D_c_rows.T               # (n, K_c)  — our convention
            except Exception:
                # svds can fail for very small or rank-deficient matrices
                D_c = rng.standard_normal((n, K_c))

        # Normalise columns (mirrors  D /= norm(D, axis=1)[:, newaxis]
        # in the library, which normalises rows; we normalise columns)
        col_norms = norm(D_c, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_c /= col_norms

        D0[:, atom_mask] = D_c

    return D0


# ---------------------------------------------------------------------------
# Ridge regression helper  (unchanged)
# ---------------------------------------------------------------------------

def ridge_regression(X: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve:  argmin_W  ||T - WX||_F^2  + lam * ||W||_F^2

    Closed-form:  W = T @ X^T @ (X @ X^T + lam * I)^{-1}

    Parameters
    ----------
    X   : (K, N)  sparse codes
    T   : (m, N)  target matrix (Q or H)
    lam : float   ridge regularisation parameter

    Returns
    -------
    W : (m, K)
    """
    K    = X.shape[0]
    gram = X @ X.T + lam * np.eye(K)
    return T @ X.T @ np.linalg.inv(gram)


# ---------------------------------------------------------------------------
# Extract D, A, W from joint D_new and renormalise (Eq. 23 / Eq. 15)
# ---------------------------------------------------------------------------

def extract_and_renorm(
    D_new: np.ndarray,
    n: int,
    K: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover D_hat, A_hat, W_hat from the jointly normalised D_new.

    Reference: Eq. 15 in [1] / Eq. 23 in [2].

    Parameters
    ----------
    D_new : (n + K + num_classes, K)  joint dictionary after AKSVD
    n     : feature dimensionality
    K     : number of dictionary atoms

    Returns
    -------
    D_hat : (n, K)
    A_hat : (K, K)
    W_hat : (num_classes, K)
    """
    D_block = D_new[:n, :]
    A_block = D_new[n:n + K, :] / np.sqrt(alpha)
    W_block = D_new[n + K:, :]  / np.sqrt(beta)

    d_norms          = norm(D_block, axis=0)
    d_norms[d_norms == 0] = 1.0

    D_hat = D_block / d_norms
    A_hat = A_block / d_norms
    W_hat = W_block / d_norms
    return D_hat, A_hat, W_hat


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LCKSVDConfig:
    """
    Hyperparameters for LC-KSVD (Approximate K-SVD variant).

    Attributes
    ----------
    K        : number of dictionary atoms (total, across all classes)
    sparsity : sparsity level T — each signal uses at most T atoms
    n_iter   : maximum number of outer AKSVD iterations
    tol      : early-stop tolerance on reconstruction error ||Y - DX||_F.
               Mirrors ApproximateKSVD's tol parameter.
    alpha    : weight of the discriminative sparse-code error term
    beta     : weight of the classification error term (LC-KSVD2 only)
    lambda1  : ridge regularisation for W initialisation
    lambda2  : ridge regularisation for A initialisation
    variant  : 'lcksvd1' or 'lcksvd2'
    """
    K: int        = 256
    sparsity: int = 30
    n_iter: int   = 10
    tol: float    = 1e-6       # ← new: mirrors ApproximateKSVD.tol
    alpha: float  = 16.0
    beta: float   = 4.0
    lambda1: float = 1e-4
    lambda2: float = 1e-4
    variant: str  = "lcksvd2"


# ---------------------------------------------------------------------------
# Main LC-KSVD class  (unchanged public API)
# ---------------------------------------------------------------------------

class LCKSVD:
    """
    Label Consistent K-SVD — Approximate K-SVD variant.

    Uses Gram-space OMP (sklearn) for sparse coding and the power-method
    dictionary update from ApproximateKSVD instead of truncated SVD.
    All LC-KSVD logic (label consistency, augmented system, classifier W)
    is identical to the original.

    Usage
    -----
    >>> model = LCKSVD(LCKSVDConfig(K=256, sparsity=10, variant='lcksvd2'))
    >>> model.fit(Y_train, labels_train, num_classes=2)
    >>> predictions = model.predict(Y_test)
    """

    def __init__(self, config: LCKSVDConfig = LCKSVDConfig()) -> None:
        self.cfg = config

        self.D_hat: Optional[np.ndarray] = None
        self.A_hat: Optional[np.ndarray] = None
        self.W_hat: Optional[np.ndarray] = None
        self.atom_labels_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        Y: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> "LCKSVD":
        """
        Learn dictionary D, transform A, and classifier W from labelled data.

        Parameters
        ----------
        Y           : (n, N)  training signal matrix, each column one sample
        labels      : (N,)    integer class indices in [0, num_classes)
        num_classes : int     number of classes

        Returns
        -------
        self
        """
        cfg = self.cfg
        n, N = Y.shape
        K    = cfg.K
        self.num_classes_ = num_classes

        logger.info(
            "LC-KSVD fit [AKSVD]: n=%d, N=%d, K=%d, sparsity=%d, variant=%s",
            n, N, K, cfg.sparsity, cfg.variant,
        )

        # ---- Step 1: Assign class labels to atoms -----------------------
        atom_labels = init_atom_labels(labels, num_classes, K)
        self.atom_labels_ = atom_labels

        # ---- Step 2: Build supervised targets ---------------------------
        H = build_label_matrix(labels, num_classes)         # (num_classes, N)
        Q = build_discriminative_codes(labels, atom_labels) # (K, N)

        # ---- Step 3: Initialise D^(0) via per-class SVD ----------------
        logger.info("Initialising dictionary via per-class SVD ...")
        D0 = init_dictionary_svd(Y, labels, atom_labels)

        # ---- Step 4: Initialise sparse codes X^(0) ----------------------
        col_norms  = norm(D0, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D0_normed  = D0 / col_norms
        gram       = D0_normed.T @ D0_normed          # (K, K)
        Xy         = D0_normed.T @ Y                  # (K, N)
        X0         = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=cfg.sparsity)  # (K, N)

        # ---- Step 5: Initialise A^(0) and W^(0) via ridge regression ----
        A0 = ridge_regression(X0, Q, cfg.lambda2)
        W0 = ridge_regression(X0, H, cfg.lambda1)

        # ---- Step 6: Build augmented matrices ---------------------------
        effective_beta = 0.0 if cfg.variant == "lcksvd1" else cfg.beta

        Y_new = np.vstack([
            Y,
            np.sqrt(cfg.alpha)      * Q,
            np.sqrt(effective_beta) * H,
        ])   # (n + K + num_classes, N)

        D_new = np.vstack([
            D0,
            np.sqrt(cfg.alpha)      * A0,
            np.sqrt(effective_beta) * W0,
        ])   # (n + K + num_classes, K)

        col_norms = norm(D_new, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_new /= col_norms

        # ---- Step 7: Approximate K-SVD on the augmented system ----------
        logger.info(
            "Running AKSVD on augmented system for up to %d iterations ...",
            cfg.n_iter,
        )
        D_new, X = aksvd_update(
            Y_new, D_new, X0, cfg.sparsity, cfg.n_iter, cfg.tol
        )

        # ---- Step 8: Extract and renormalise D, A, W --------------------
        D_hat, A_hat, W_hat = extract_and_renorm(
            D_new, n, K,
            cfg.alpha,
            effective_beta if effective_beta > 0 else 1.0,
        )

        # ---- Step 9: LC-KSVD1 — refit W separately ---------------------
        if cfg.variant == "lcksvd1":
            logger.info("LC-KSVD1: fitting classifier W separately ...")
            col_norms   = norm(D_hat, axis=0, keepdims=True)
            col_norms[col_norms == 0] = 1.0
            D_hat_normed = D_hat / col_norms
            gram         = D_hat_normed.T @ D_hat_normed   # (K, K)
            Xy           = D_hat_normed.T @ Y              # (K, N)
            X_final      = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=cfg.sparsity)
            W_hat        = ridge_regression(X_final, H, cfg.lambda1)

        self.D_hat = D_hat
        self.A_hat = A_hat
        self.W_hat = W_hat

        logger.info("LC-KSVD training complete.")
        return self

    # ------------------------------------------------------------------
    # encode / predict / predict_scores  (unchanged)
    # ------------------------------------------------------------------

    def encode(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute sparse codes for input signals using the learned dictionary.

        Parameters
        ----------
        Y : (n, N_test) test signals

        Returns
        -------
        X : (K, N_test) sparse codes
        """
        if self.D_hat is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        col_norms = norm(self.D_hat, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D_normed  = self.D_hat / col_norms
        gram      = D_normed.T @ D_normed     # (K, K)
        Xy        = D_normed.T @ Y            # (K, N_test)
        return orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=self.cfg.sparsity)  # (K, N_test)

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Classify test signals via the internal W_hat linear classifier.

        Parameters
        ----------
        Y : (n, N_test)

        Returns
        -------
        predicted_labels : (N_test,) integer class indices
        """
        X      = self.encode(Y)
        scores = self.W_hat @ X
        return np.argmax(scores, axis=0)

    def predict_scores(self, Y: np.ndarray) -> np.ndarray:
        """
        Return raw classifier scores (before argmax).

        Parameters
        ----------
        Y : (n, N_test)

        Returns
        -------
        scores : (num_classes, N_test)
        """
        X = self.encode(Y)
        return self.W_hat @ X


