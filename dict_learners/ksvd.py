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

Approximate K-SVD via ksvd.ApproximateKSVD
-------------------------------------------
All dictionary learning sub-routines (initialisation, sparse coding,
dictionary update, convergence check) are delegated directly to
ApproximateKSVD from ksvd.py.  No logic is duplicated here.

  _initialize  → per-class SVD-based or random initialisation of D
  _transform   → Gram-space OMP via sklearn orthogonal_mp_gram
  _update_dict → power-method rank-1 atom update
  fit loop     → alternates _transform / tolerance check / _update_dict

Convention bridging
-------------------
ApproximateKSVD uses row-major convention:
    D      : (K, n)  — atoms are rows
    signals: (N, n)  — samples are rows
    codes  : (N, K)  — one row per sample

Our pipeline uses column-major (matches the LC-KSVD paper):
    D      : (n, K)  — atoms are columns
    signals: (n, N)  — one column per sample
    codes  : (K, N)  — one column per sample

Bridging is applied at every call site with a single .T — the
ApproximateKSVD internal logic is never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.linalg import norm
from ksvd import ApproximateKSVD

logger = logging.getLogger(__name__)


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
    Hyperparameters for LC-KSVD.

    Attributes
    ----------
    K        : number of dictionary atoms (total, across all classes)
    sparsity : sparsity level T — each signal uses at most T atoms
    n_iter   : maximum number of AKSVD iterations (maps to max_iter)
    tol      : early-stop tolerance on ||Y - DX||_F (maps to tol)
    alpha    : weight of the discriminative sparse-code error term
    beta     : weight of the classification error term (LC-KSVD2 only)
    lambda1  : ridge regularisation for W initialisation
    lambda2  : ridge regularisation for A initialisation
    variant  : 'lcksvd1' or 'lcksvd2'
    """
    K: int         = 256
    sparsity: int  = 30
    n_iter: int    = 10
    tol: float     = 1e-6
    alpha: float   = 16.0
    beta: float    = 4.0
    lambda1: float = 1e-4
    lambda2: float = 1e-4
    variant: str   = "lcksvd2"


# ---------------------------------------------------------------------------
# Main LC-KSVD class
# ---------------------------------------------------------------------------

class LCKSVD:
    """
    Label Consistent K-SVD — using ApproximateKSVD from ksvd.py.

    All dictionary learning sub-routines are delegated to ApproximateKSVD.
    All LC-KSVD logic (augmented system, label consistency, classifier W)
    is implemented here and is unchanged from the paper.

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
    # Internal helper: build a per-class ApproximateKSVD instance
    # ------------------------------------------------------------------

    def _make_aksvd(self, K_c: int) -> ApproximateKSVD:
        """
        Create an ApproximateKSVD instance with config settings.

        Parameters
        ----------
        K_c : int  number of atoms for this instance

        Returns
        -------
        ApproximateKSVD
        """
        return ApproximateKSVD(
            n_components=K_c,
            max_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            transform_n_nonzero_coefs=self.cfg.sparsity,
        )

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

        # ---- Step 3: Initialise D^(0) via per-class ApproximateKSVD ----
        # ApproximateKSVD._initialize() handles SVD-based or random init.
        # Convention bridge: ApproximateKSVD expects (N_c, n) row-major,
        # so we pass Y_c.T; it returns D_c of shape (K_c, n), so we
        # take .T to get our (n, K_c) column convention.
        logger.info("Initialising dictionary via per-class ApproximateKSVD ...")
        D0 = np.zeros((n, K))
        for c in range(num_classes):
            signal_mask = labels == c
            atom_mask   = atom_labels == c
            Y_c         = Y[:, signal_mask]              # (n, N_c)
            K_c         = int(atom_mask.sum())

            if Y_c.shape[1] == 0 or K_c == 0:
                continue

            aksvd_c     = self._make_aksvd(K_c)
            D_c_rows    = aksvd_c._initialize(Y_c.T)    # (K_c, n)
            D0[:, atom_mask] = D_c_rows.T               # (n, K_c)

        # ---- Step 4: Initialise sparse codes X^(0) ----------------------
        # ApproximateKSVD._transform(D, X) expects:
        #   D : (K, n)  — pass D0.T
        #   X : (N, n)  — pass Y.T
        # returns (N, K) — transpose to get our (K, N)
        aksvd_full = self._make_aksvd(K)
        X0 = aksvd_full._transform(D0.T, Y.T).T         # (K, N)

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

        # ---- Step 7: ApproximateKSVD on the augmented system ------------
        # We call ApproximateKSVD.fit() on the augmented system.
        # Convention bridge:
        #   fit() expects X of shape (N, n_features) — pass Y_new.T
        #   after fit, components_ is (K, m) — transpose to get our (m, K)
        logger.info(
            "Running AKSVD on augmented system for up to %d iterations ...",
            cfg.n_iter,
        )
        m = Y_new.shape[0]   # augmented feature dimension
        aksvd_aug = ApproximateKSVD(
            n_components=K,
            max_iter=cfg.n_iter,
            tol=cfg.tol,
            transform_n_nonzero_coefs=cfg.sparsity,
        )
        # Provide D_new.T as the initial dictionary by setting components_
        # before calling fit(), bypassing _initialize().
        aksvd_aug.components_ = D_new.T                 # (K, m) row-major

        # Run the fit loop manually to use our pre-initialised D_new.T:
        #   mirrors ApproximateKSVD.fit() exactly, using its own methods.
        D_aug = aksvd_aug.components_                   # (K, m)
        for _ in range(cfg.n_iter):
            gamma = aksvd_aug._transform(D_aug, Y_new.T)  # (N, K)
            e = np.linalg.norm(Y_new.T - gamma.dot(D_aug))
            if e < cfg.tol:
                break
            D_aug, gamma = aksvd_aug._update_dict(Y_new.T, D_aug, gamma)
        aksvd_aug.components_ = D_aug

        # Retrieve D_new in our column convention
        D_new = aksvd_aug.components_.T                 # (m, K)

        # Retrieve final sparse codes X in our column convention
        X = aksvd_aug._transform(D_aug, Y_new.T).T     # (K, N)

        # ---- Step 8: Extract and renormalise D, A, W --------------------
        D_hat, A_hat, W_hat = extract_and_renorm(
            D_new, n, K,
            cfg.alpha,
            effective_beta if effective_beta > 0 else 1.0,
        )

        # ---- Step 9: LC-KSVD1 — refit W separately ---------------------
        if cfg.variant == "lcksvd1":
            logger.info("LC-KSVD1: fitting classifier W separately ...")
            X_final = aksvd_full._transform(D_hat.T, Y.T).T   # (K, N)
            W_hat   = ridge_regression(X_final, H, cfg.lambda1)

        self.D_hat = D_hat
        self.A_hat = A_hat
        self.W_hat = W_hat

        logger.info("LC-KSVD training complete.")
        return self

    # ------------------------------------------------------------------
    # encode
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

        aksvd = self._make_aksvd(self.cfg.K)
        # _transform expects D (K, n) and X (N, n); returns (N, K)
        return aksvd._transform(self.D_hat.T, Y.T).T    # (K, N_test)

    # ------------------------------------------------------------------
    # predict / predict_scores
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(seed=42)

    num_classes       = 3
    n_features        = 50
    n_train_per_class = 40
    n_test_per_class  = 10
    K                 = num_classes * 20
    sparsity          = 6

    centres = rng.standard_normal((num_classes, n_features))
    Y_train_list, Y_test_list = [], []
    lbl_train_list, lbl_test_list = [], []

    for c in range(num_classes):
        Y_train_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_train_per_class))
        )
        Y_test_list.append(
            centres[c, :, None] + 0.1 * rng.standard_normal((n_features, n_test_per_class))
        )
        lbl_train_list.extend([c] * n_train_per_class)
        lbl_test_list.extend([c] * n_test_per_class)

    Y_train      = np.hstack(Y_train_list)
    Y_test       = np.hstack(Y_test_list)
    labels_train = np.array(lbl_train_list)
    labels_test  = np.array(lbl_test_list)

    for variant in ("lcksvd2", "lcksvd1"):
        cfg   = LCKSVDConfig(K=K, sparsity=sparsity, n_iter=10, tol=1e-6, variant=variant)
        model = LCKSVD(cfg)
        model.fit(Y_train, labels_train, num_classes=num_classes)
        preds = model.predict(Y_test)
        acc   = float(np.mean(preds == labels_test))
        print(f"{variant} test accuracy: {acc * 100:.1f}%")