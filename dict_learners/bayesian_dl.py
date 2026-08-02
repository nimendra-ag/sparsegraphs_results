"""
Beta Process Factor Analysis (BPFA) Dictionary Learning  —  core algorithm.

This module is the standalone algorithm, in the same role `dict_learners/ksvd.py`
and `dict_learners/lc_ksvd.py` play for their arms: it knows nothing about the WL
pipeline. The pipeline adapter (DictLearner interface, seeding, persistence) lives
in `dict_learners/bayesian.py`.

Python implementation of the non-parametric Bayesian dictionary learning algorithm
from: "Non-Parametric Bayesian Dictionary Learning for Sparse Image Representations,"
Zhou et al., NIPS 2009.

The beta process prior allows the model to infer the effective dictionary size
automatically, and Gibbs sampling provides full posterior inference over all
model parameters including the noise precision.

Model:
    x_i = D @ w_i + eps_i
    w_i = z_i ⊙ s_i          (element-wise product)
    d_k ~ N(0, (1/P) I_P)    (dictionary atoms)
    z_ik ~ Bernoulli(pi_k)    (binary usage indicators)
    pi_k ~ Beta(a0/K, b0*(K-1)/K)
    s_i  ~ N(0, gamma_s^{-1} I_K)  (weights)
    eps_i ~ N(0, gamma_eps^{-1} I_P) (noise)
    gamma_s ~ Gamma(e0, f0)
    gamma_eps ~ Gamma(c0, d0)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class BPFA:
    """Beta Process Factor Analysis for dictionary learning via Gibbs sampling.

    Parameters
    ----------
    n_components : int
        Maximum dictionary size K. The effective size is inferred automatically
        via the beta process prior (unused atoms decay toward the prior).
    n_iter : int
        Number of Gibbs sampling iterations for dictionary learning.
    n_infer_iter : int
        Number of Gibbs sampling iterations when inferring sparse codes for
        new data (dictionary held fixed). Set to 0 to use the MAP-style
        posterior mean from a single conditional solve instead of sampling.
    n_burnin : int
        Number of burn-in iterations discarded before collecting samples
        during inference. Applies only when ``n_infer_iter > 0``.
    n_collect : int
        Number of post-burn-in samples to average for the final sparse code
        during inference. More samples reduce posterior variance.
    a0 : float
        First parameter of the Beta prior on pi_k. Controls expected sparsity
        together with ``b0``. In the limit K -> inf, the expected number of
        active atoms per sample is Poisson(a0 / b0).
    b0 : float or None
        Second parameter of the Beta prior on pi_k. If None, defaults to N/8
        where N is the number of training samples (recommended by the paper).
    c0, d0 : float
        Hyper-prior parameters (Gamma) for the noise precision gamma_eps.
        The paper recommends non-informative values (1e-6, 1e-6).
    e0, f0 : float
        Hyper-prior parameters (Gamma) for the weight precision gamma_s.
        The paper recommends non-informative values (1e-6, 1e-6).
    init_method : str
        Initialization strategy: ``"svd"`` (recommended) or ``"random"``.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        If True, log progress every 10 iterations.
    """

    def __init__(
        self,
        n_components: int = 32,
        n_iter: int = 10,
        n_infer_iter: int = 50,
        n_burnin: int = 20,
        n_collect: int = 10,
        a0: float = 1.0,
        b0: Optional[float] = None,
        c0: float = 1e-6,
        d0: float = 1e-6,
        e0: float = 1e-6,
        f0: float = 1e-6,
        init_method: str = "svd",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.name = "BPFA"
        self._dictionary = None
        self.n_components = n_components
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
        self.random_state = random_state
        self.verbose = verbose

        # Populated after fit()
        self._phi: float = 1.0       # noise precision
        self._alpha: float = 1.0     # weight precision
        self._pi: Optional[np.ndarray] = None  # usage probabilities per atom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, training_graph_embeddings: ArrayLike) -> "BPFA":
        """Learn a dictionary from training data using Gibbs sampling.

        Parameters
        ----------
        training_graph_embeddings : array-like of shape (N, P)
            Training data matrix where N is the number of samples and P is
            the feature dimensionality.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(training_graph_embeddings, dtype=np.float64)
        N, P = X.shape
        K = self.n_components

        b0 = self.b0 if self.b0 is not None else N / 8.0

        # --- Initialise all latent variables ---
        D, S, Z, phi, alpha, pi = self._initialise(X, P, N, K, rng)

        # Residual matrix (P x N): keeps X^T - D @ diag(alpha_i) for speed
        residual = X.T - D @ (Z * S).T

        for it in range(self.n_iter):
            residual, D, Z, S = self._sample_dzs(
                residual, D, Z, S, pi, alpha, phi, P, K, N, rng,
                sample_d=True, sample_z=True, sample_s=True,
            )
            pi = self._sample_pi(Z, N, self.a0, b0, rng)
            alpha = self._sample_alpha(S, self.e0, self.f0, Z, alpha, rng)
            phi = self._sample_phi(residual, self.c0, self.d0, rng)

            if self.verbose and (it + 1) % 10 == 0:
                n_active = int(np.sum(pi > 1e-3))
                avg_nnz = np.mean(np.sum(Z, axis=1))
                noise_std = np.sqrt(1.0 / phi)
                logger.info(
                    "iter %3d/%d | active atoms: %d | avg nnz/sample: %.1f | "
                    "noise std: %.4f",
                    it + 1, self.n_iter, n_active, avg_nnz, noise_std,
                )

        # Store learned parameters
        self._dictionary = D  # (P, K)
        self._phi = phi
        self._alpha = alpha
        self._pi = pi

        if self.verbose:
            n_active = int(np.sum(pi > 1e-3))
            logger.info(
                "BPFA fit complete. Effective dictionary size M=%d (of K=%d)",
                n_active, K,
            )

        return self

    def infer(self, infer_graph_embeddings: ArrayLike) -> np.ndarray:
        """Compute sparse representations for new data given the learned dictionary.

        The dictionary D is held fixed and Gibbs sampling runs over (Z, S) only.
        The returned codes are the posterior mean of alpha_i = z_i ⊙ s_i,
        averaged over ``n_collect`` post-burn-in samples.

        Parameters
        ----------
        infer_graph_embeddings : array-like of shape (N_new, P)
            New data to encode.

        Returns
        -------
        sparse_codes : ndarray of shape (N_new, K)
            Sparse coefficient matrix.
        """
        if self._dictionary is None:
            raise RuntimeError("Must call fit() before infer().")

        rng = np.random.RandomState(self.random_state)
        X = np.asarray(infer_graph_embeddings, dtype=np.float64)
        N, P = X.shape
        D = self._dictionary
        K = D.shape[1]
        phi = self._phi
        alpha = self._alpha
        pi = self._pi

        if P != D.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: dictionary has P={D.shape[0]} "
                f"but input has P={P}."
            )

        # Initialise Z, S to zeros (cold start)
        Z = np.zeros((N, K), dtype=np.float64)
        S = np.zeros((N, K), dtype=np.float64)
        residual = X.T.copy()  # P x N

        total_iter = self.n_burnin + self.n_collect
        if self.n_infer_iter > 0:
            total_iter = max(total_iter, self.n_infer_iter)

        collected = np.zeros((N, K), dtype=np.float64)
        n_collected = 0

        for it in range(total_iter):
            residual, _, Z, S = self._sample_dzs(
                residual, D, Z, S, pi, alpha, phi, P, K, N, rng,
                sample_d=False, sample_z=True, sample_s=True,
            )

            if it >= self.n_burnin:
                collected += Z * S
                n_collected += 1

        if n_collected > 0:
            return collected / n_collected
        else:
            return Z * S

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(
        self,
        X: np.ndarray,
        P: int,
        N: int,
        K: int,
        rng: np.random.RandomState,
    ) -> tuple:
        """Initialise D, S, Z, phi, alpha, pi.

        SVD initialisation mirrors the MATLAB ``InitMatrix`` with option 'SVD'.
        """
        if self.init_method == "svd":
            U, sigma, Vt = np.linalg.svd(X.T, full_matrices=False)  # P x N
            n_svd = min(P, N, K)
            D = np.zeros((P, K), dtype=np.float64)
            S = np.zeros((N, K), dtype=np.float64)
            D[:, :n_svd] = U[:, :n_svd] * sigma[:n_svd]  # broadcast
            S[:, :n_svd] = Vt[:n_svd, :].T
            Z = np.ones((N, K), dtype=np.float64)
            pi = 0.5 * np.ones(K, dtype=np.float64)
        elif self.init_method == "random":
            D = rng.randn(P, K) / np.sqrt(P)
            S = rng.randn(N, K)
            Z = np.zeros((N, K), dtype=np.float64)
            pi = 0.01 * np.ones(K, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. Use 'svd' or 'random'."
            )

        phi = 1.0 / (0.1 ** 2)   # initial noise precision (σ ≈ 0.1)
        alpha = 1.0               # initial weight precision

        return D, S, Z, phi, alpha, pi

    # ------------------------------------------------------------------
    # Gibbs sampling steps
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_dzs(
        residual: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        S: np.ndarray,
        pi: np.ndarray,
        alpha: float,
        phi: float,
        P: int,
        K: int,
        N: int,
        rng: np.random.RandomState,
        *,
        sample_d: bool = True,
        sample_z: bool = True,
        sample_s: bool = True,
    ) -> tuple:
        """DkZkSk strategy: sample D_k, Z_k, S_k one atom at a time.

        This is the most efficient update order from the original MATLAB code,
        as it avoids recomputing the full residual between sweeps.

        Parameters
        ----------
        residual : ndarray (P, N)
            Current residual X^T - D @ (Z ⊙ S)^T.
        """
        for k in range(K):
            active = Z[:, k] > 0.5  # boolean mask over N samples
            n_active = int(np.sum(active))

            # --- Add back atom k's contribution to the residual ---
            if n_active > 0:
                # residual[:, active] += outer(d_k, s_active_k)
                residual[:, active] += np.outer(D[:, k], S[active, k])

            # --- Sample d_k ---
            if sample_d:
                if n_active > 0:
                    s_active = S[active, k]
                    sig_dk = 1.0 / (phi * np.dot(s_active, s_active) + P)
                    mu_dk = phi * sig_dk * (residual[:, active] @ s_active)
                else:
                    sig_dk = 1.0 / P
                    mu_dk = np.zeros(P)
                D[:, k] = mu_dk + rng.randn(P) * np.sqrt(sig_dk)

            dtd = np.dot(D[:, k], D[:, k])

            # --- Sample z_k ---
            if sample_z:
                # For inactive entries, impute s from the prior N(0, 1/alpha)
                s_full = S[:, k].copy()
                inactive = ~active
                n_inactive = N - n_active
                if n_inactive > 0:
                    s_full[inactive] = rng.randn(n_inactive) * np.sqrt(
                        1.0 / alpha
                    )

                # Log-likelihood ratio for z_ik = 1 vs 0
                dk_proj = residual.T @ D[:, k]  # (N,)
                log_ratio = -0.5 * phi * (
                    s_full ** 2 * dtd - 2.0 * s_full * dk_proj
                )
                log_ratio = np.clip(log_ratio, -500.0, 500.0)

                p1 = np.exp(log_ratio) * pi[k]
                p0 = 1.0 - pi[k]
                prob = p1 / (p0 + p1)

                Z[:, k] = (rng.rand(N) < prob).astype(np.float64)

            # Refresh active mask after Z update
            active = Z[:, k] > 0.5
            n_active = int(np.sum(active))

            # --- Sample s_k ---
            if sample_s:
                S[:, k] = 0.0
                if n_active > 0:
                    sig_s = 1.0 / (alpha + phi * dtd)
                    mu_s = phi * sig_s * (residual[:, active].T @ D[:, k])
                    S[active, k] = mu_s + rng.randn(n_active) * np.sqrt(sig_s)

            # --- Subtract atom k's (updated) contribution ---
            if n_active > 0:
                residual[:, active] -= np.outer(D[:, k], S[active, k])

        return residual, D, Z, S

    @staticmethod
    def _sample_pi(
        Z: np.ndarray,
        N: int,
        a0: float,
        b0: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Sample pi_k ~ Beta(a0/K + sum_i z_ik, b0(K-1)/K + N - sum_i z_ik).

        Matches Eq. 20 of the inference document / SamplePi.m.
        """
        K = Z.shape[1]
        sum_z = np.sum(Z, axis=0)  # (K,)
        return rng.beta(sum_z + a0 / K, b0 * (K - 1) / K + N - sum_z)

    @staticmethod
    def _sample_alpha(
        S: np.ndarray,
        e0: float,
        f0: float,
        Z: np.ndarray,
        alpha: float,
        rng: np.random.RandomState,
    ) -> float:
        """Sample gamma_s (weight precision) from its Gamma posterior.

        Matches Samplealpha.m: the rate includes the marginal contribution of
        the zero entries in S that would have been drawn from N(0, 1/alpha).
        """
        n_total = Z.size
        n_active = int(np.sum(Z > 0.5))
        shape = e0 + 0.5 * n_total
        rate = f0 + 0.5 * np.sum(S ** 2) + 0.5 * (n_total - n_active) / alpha
        return rng.gamma(shape, 1.0 / rate)

    @staticmethod
    def _sample_phi(
        residual: np.ndarray,
        c0: float,
        d0: float,
        rng: np.random.RandomState,
    ) -> float:
        """Sample gamma_eps (noise precision) from its Gamma posterior.
 
        Matches Samplephi.m.
        """
        n_elements = residual.size
        shape = c0 + 0.5 * n_elements
        rate = d0 + 0.5 * np.sum(residual ** 2)
        return rng.gamma(shape, 1.0 / rate)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dictionary(self) -> Optional[np.ndarray]:
        """Learned dictionary matrix D of shape (P, K)."""
        return self._dictionary

    @property
    def effective_dictionary_size(self) -> int:
        """Number of dictionary atoms used more than 0.1% of the time."""
        if self._pi is None:
            return 0
        return int(np.sum(self._pi > 1e-3))

    @property
    def noise_std(self) -> float:
        """Estimated noise standard deviation."""
        return np.sqrt(1.0 / self._phi)

    @property
    def usage_probabilities(self) -> Optional[np.ndarray]:
        """Per-atom usage probabilities pi_k."""
        return self._pi