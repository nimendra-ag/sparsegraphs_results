"""
Beta Process Factor Analysis (BPFA) Dictionary Learning  —  GPU core algorithm.

PyTorch port of `dict_learners/bayesian_dl.py`. Same model, same Gibbs sampler,
same update order; only the linear algebra moves to the device. The pipeline
adapter (DictLearner interface, seeding, persistence) lives in
`dict_learners/bayesian_gpu.py` — this module knows nothing about the WL
pipeline, exactly as the NumPy core does.

Why the CPU core is slow (and what changed here)
------------------------------------------------
The sampler is inherently sequential over atoms — atom k's conditional depends
on the residual left by atoms 0..k-1 — so the K-loop stays a Python loop here
too. The cost is *inside* it. The NumPy core addresses the active samples with
a boolean mask (``residual[:, active]``), and NumPy fancy-indexing **copies**:
three (P, n_active) temporaries per atom per sweep, allocated and thrown away
K * n_iter times. That copying, not the arithmetic, dominates the runtime.

This port removes the copies entirely by rewriting each masked operation as a
full-width one on the vector ``w_k = z_k ⊙ s_k``, which is *by construction*
zero wherever the mask is false. That makes the masked and full-width forms
algebraically identical (see the per-step notes in ``_sample_dzs``), and leaves
four dense (P, N) operations per atom — two rank-1 updates and two matvecs —
which are exactly what a GPU is good at.

One further saving: the NumPy core computes ``residual.T @ d_k`` twice (once in
the z-step over all N, once in the s-step over the active columns). The residual
and d_k are unchanged between those two steps, so this port computes it once and
reuses it. Same numbers, one fewer pass over (P, N).

Numerical parity
----------------
The *math* is a faithful port; the *numbers* will not match the CPU core
sample-for-sample, for two unavoidable reasons:

  1. Different RNG. Torch's generator is not NumPy's RandomState, so an
     identical seed yields a different (equally valid) posterior sample path.
  2. Different draw counts. Where the CPU core draws `n_active` normals, this
     port draws N and discards the masked ones — vectorised, but a different
     position in the stream.

Both are properties of the sampler's randomness, not of the posterior: the two
implementations target the same distribution. Run-to-run reproducibility within
this implementation is exact for a fixed seed and device.

`dtype` defaults to float32 (consumer GPUs run float64 at a small fraction of
their float32 throughput). The reductions that feed the Gamma/Beta conditionals
— sums over (P, N) with ~1e7 terms — are accumulated in float64 regardless, and
those scalar conditionals are sampled in float64 on the CPU, so the precision
that matters is not the precision that costs.

Model (identical to the NumPy core):
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
import time
from typing import Optional

import numpy as np
import torch
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Scalar conditionals (pi, gamma_s, gamma_eps) are sampled here. Shapes reach
# ~0.5 * P * N, so they are drawn in float64 on the CPU: a handful of scalars per
# sweep, where a float32 rounding of a 1e7-sized shape parameter would be a real
# bias and the cost of avoiding it is nil.
_SCALAR_DTYPE = torch.float64


class BPFAGPU:
    """Beta Process Factor Analysis via Gibbs sampling, on GPU (PyTorch).

    Parameters
    ----------
    n_components : int
        Maximum dictionary size K. The effective size is inferred automatically
        via the beta process prior (unused atoms decay toward the prior).
    n_iter : int
        Number of Gibbs sampling iterations for dictionary learning.
    n_infer_iter : int
        Number of Gibbs sampling iterations when inferring sparse codes for
        new data (dictionary held fixed). Set to 0 to use only the
        burn-in + collect schedule below.
    n_burnin : int
        Number of burn-in iterations discarded before collecting samples
        during inference.
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
    e0, f0 : float
        Hyper-prior parameters (Gamma) for the weight precision gamma_s.
    init_method : str
        ``"svd"``      — exact truncated SVD init (matches the NumPy core).
        ``"svd_lowrank"`` — randomised rank-K SVD (torch.svd_lowrank). Only the
                        leading K <= min(P, N) components are ever used, so this
                        is the same init up to the randomised solver's error,
                        at a fraction of the cost on a wide (P, N) matrix.
        ``"random"``   — draw D from the prior.
    device : str or torch.device or None
        Compute device. None selects CUDA when available, else CPU.
    dtype : torch.dtype
        Working precision for the (P, N) arrays. float32 by default.
    infer_batch_size : int or None
        Encode inference data in batches of this many samples. With D held
        fixed the samples are conditionally independent — column i of the
        residual is touched only by sample i's own (z_i, s_i) — so batching
        changes nothing about the distribution each code is drawn from; it is
        not an approximation. (It does shift the RNG stream, so batched and
        unbatched runs give different draws from that same posterior, exactly
        as two seeds would.) Use it to bound the (P, batch) residual when a
        test split will not fit in VRAM. None encodes in one pass.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        If True, log sampler diagnostics every 10 iterations.
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
        device=None,
        dtype: torch.dtype = torch.float32,
        infer_batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.name = "BPFAGPU"
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
        self.dtype = dtype
        self.infer_batch_size = infer_batch_size
        self.random_state = random_state
        self.verbose = verbose

        # Check and assign GPU automatically (mirrors FDDLGPU).
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Populated after fit(). Kept as NumPy/floats — the same host-side
        # contract the CPU core exposes — so persistence is device-independent
        # and a bundle saved from a GPU run loads anywhere.
        self._phi: float = 1.0       # noise precision
        self._alpha: float = 1.0     # weight precision
        self._pi: Optional[np.ndarray] = None  # usage probabilities per atom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, training_graph_embeddings: ArrayLike) -> "BPFAGPU":
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
        self._seed_generators()

        X = torch.as_tensor(
            np.asarray(training_graph_embeddings), dtype=self.dtype
        ).to(self.device)
        N, P = X.shape
        K = self.n_components

        b0 = self.b0 if self.b0 is not None else N / 8.0

        print(f"Training {self.name} on context [{self.device}] "
              f"(K={K}, n_iter={self.n_iter}, N={N}, P={P}, "
              f"dtype={str(self.dtype).replace('torch.', '')})...", flush=True)

        # --- Initialise all latent variables ---
        D, S, Z, phi, alpha, pi = self._initialise(X, P, N, K)

        # Residual matrix (P x N): keeps X^T - D @ (Z ⊙ S)^T.
        residual = X.T.contiguous() - D @ (Z * S).T
        del X  # the residual carries all the information from here on

        for it in range(self.n_iter):
            t0 = time.perf_counter()
            residual, D, Z, S = self._sample_dzs(
                residual, D, Z, S, pi, alpha, phi, P, K, N,
                sample_d=True, sample_z=True, sample_s=True,
            )
            pi = self._sample_pi(Z, N, self.a0, b0)
            alpha = self._sample_alpha(S, self.e0, self.f0, Z, alpha)
            phi = self._sample_phi(residual, self.c0, self.d0)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            n_active = int((pi > 1e-3).sum())
            print(f"  [iter {it + 1:>3}/{self.n_iter}] {time.perf_counter() - t0:6.2f}s "
                  f"| active atoms: {n_active}/{K} "
                  f"| noise std: {float(np.sqrt(1.0 / phi)):.4f}", flush=True)

            if self.verbose and (it + 1) % 10 == 0:
                avg_nnz = float(Z.sum(dim=1).mean())
                logger.info(
                    "iter %3d/%d | active atoms: %d | avg nnz/sample: %.1f | "
                    "noise std: %.4f",
                    it + 1, self.n_iter, n_active, avg_nnz, np.sqrt(1.0 / phi),
                )

        # Store learned parameters on the host (see __init__ note).
        self._dictionary = D.detach().cpu().numpy().astype(np.float64)  # (P, K)
        self._phi = float(phi)
        self._alpha = float(alpha)
        self._pi = pi.detach().cpu().numpy().astype(np.float64)

        del residual, D, Z, S
        self._empty_cache()

        if self.verbose:
            logger.info(
                "BPFA fit complete. Effective dictionary size M=%d (of K=%d)",
                self.effective_dictionary_size, K,
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

        self._seed_generators()

        X_np = np.asarray(infer_graph_embeddings)
        N_total, P = X_np.shape
        if P != self._dictionary.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: dictionary has "
                f"P={self._dictionary.shape[0]} but input has P={P}."
            )

        batch = self.infer_batch_size or N_total
        codes = np.empty((N_total, self._dictionary.shape[1]), dtype=np.float64)
        for start in range(0, N_total, batch):
            stop = min(start + batch, N_total)
            codes[start:stop] = self._infer_batch(X_np[start:stop])
        return codes

    def _infer_batch(self, X_np: np.ndarray) -> np.ndarray:
        """Encode one batch of samples. Exact — samples are independent given D."""
        X = torch.as_tensor(X_np, dtype=self.dtype).to(self.device)
        N, P = X.shape

        D = torch.as_tensor(self._dictionary, dtype=self.dtype).to(self.device)
        K = D.shape[1]
        phi = self._phi
        alpha = self._alpha
        pi = torch.as_tensor(self._pi, dtype=self.dtype).to(self.device)

        # Initialise Z, S to zeros (cold start)
        Z = torch.zeros((N, K), dtype=self.dtype, device=self.device)
        S = torch.zeros((N, K), dtype=self.dtype, device=self.device)
        residual = X.T.contiguous()  # P x N
        del X

        total_iter = self.n_burnin + self.n_collect
        if self.n_infer_iter > 0:
            total_iter = max(total_iter, self.n_infer_iter)

        collected = torch.zeros((N, K), dtype=self.dtype, device=self.device)
        n_collected = 0

        for it in range(total_iter):
            residual, _, Z, S = self._sample_dzs(
                residual, D, Z, S, pi, alpha, phi, P, K, N,
                sample_d=False, sample_z=True, sample_s=True,
            )

            if it >= self.n_burnin:
                collected += Z * S
                n_collected += 1

        out = (collected / n_collected) if n_collected > 0 else (Z * S)
        out = out.detach().cpu().numpy().astype(np.float64)

        del residual, D, Z, S, collected
        self._empty_cache()
        return out

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self, X: torch.Tensor, P: int, N: int, K: int) -> tuple:
        """Initialise D, S, Z, phi, alpha, pi.

        SVD initialisation mirrors the NumPy core's, which mirrors the MATLAB
        ``InitMatrix`` with option 'SVD'.
        """
        if self.init_method in ("svd", "svd_lowrank"):
            n_svd = min(P, N, K)
            U, sigma, Vh = self._svd(X.T, n_svd)
            D = torch.zeros((P, K), dtype=self.dtype, device=self.device)
            S = torch.zeros((N, K), dtype=self.dtype, device=self.device)
            D[:, :n_svd] = U[:, :n_svd] * sigma[:n_svd]  # broadcast
            S[:, :n_svd] = Vh[:n_svd, :].T
            Z = torch.ones((N, K), dtype=self.dtype, device=self.device)
            pi = torch.full((K,), 0.5, dtype=self.dtype, device=self.device)
        elif self.init_method == "random":
            D = torch.randn((P, K), dtype=self.dtype, device=self.device) / np.sqrt(P)
            S = torch.randn((N, K), dtype=self.dtype, device=self.device)
            Z = torch.zeros((N, K), dtype=self.dtype, device=self.device)
            pi = torch.full((K,), 0.01, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. Use 'svd', "
                f"'svd_lowrank' or 'random'."
            )

        phi = 1.0 / (0.1 ** 2)   # initial noise precision (σ ≈ 0.1)
        alpha = 1.0               # initial weight precision

        return D, S, Z, phi, alpha, pi

    def _svd(self, A: torch.Tensor, n_svd: int) -> tuple:
        """Truncated SVD of A (P, N), returning (U, sigma, Vh).

        Only the leading `n_svd` <= K components are ever read. `svd_lowrank`
        gets them with a randomised solver; `svd` computes the full thin SVD.
        A full thin SVD of a wide matrix is heavy on cuSOLVER, so a device
        failure (OOM / solver error) falls back to the CPU rather than aborting
        a long run at its first minute.
        """
        if self.init_method == "svd_lowrank":
            # q slightly above the target rank is the standard oversampling for
            # randomised range finders; niter=4 subspace iterations tighten it.
            q = min(n_svd + 10, min(A.shape))
            U, sigma, V = torch.svd_lowrank(A, q=q, niter=4)
            return U, sigma, V.T

        try:
            U, sigma, Vh = torch.linalg.svd(A, full_matrices=False)
            return U, sigma, Vh
        except RuntimeError as exc:
            print(f"  [init][WARN] device SVD failed ({exc}); falling back to CPU SVD. "
                  f"Consider init_method='svd_lowrank' — only the leading "
                  f"{n_svd} components are used.", flush=True)
            U, sigma, Vh = np.linalg.svd(
                A.detach().cpu().numpy(), full_matrices=False
            )
            to_dev = lambda a: torch.as_tensor(a, dtype=self.dtype).to(self.device)
            return to_dev(U), to_dev(sigma), to_dev(Vh)

    # ------------------------------------------------------------------
    # Gibbs sampling steps
    # ------------------------------------------------------------------

    def _sample_dzs(
        self,
        residual: torch.Tensor,
        D: torch.Tensor,
        Z: torch.Tensor,
        S: torch.Tensor,
        pi: torch.Tensor,
        alpha: float,
        phi: float,
        P: int,
        K: int,
        N: int,
        *,
        sample_d: bool = True,
        sample_z: bool = True,
        sample_s: bool = True,
    ) -> tuple:
        """DkZkSk strategy: sample D_k, Z_k, S_k one atom at a time.

        Same update order as the NumPy core. The difference is that every step
        below runs at full width (N columns) instead of on a boolean-masked
        copy, which is what removes the per-atom allocations.

        The identity that makes those two forms equivalent: define

            w_k = z_k ⊙ s_k

        Because z_k is 0/1 and the s-step writes zeros wherever z_k = 0, w_k
        agrees with s_k on the active set and is exactly 0 off it. So
        ``outer(d_k, w_k)`` over all N columns equals the core's
        ``outer(d_k, s_k[active])`` over the active ones, ``w_k · w_k`` equals
        ``s_active · s_active``, and ``residual @ w_k`` equals
        ``residual[:, active] @ s_active``. The n_active == 0 branch of the core
        falls out for free: w_k is then the zero vector, giving sig_dk = 1/P and
        mu_dk = 0, which is precisely what that branch hard-codes.

        Parameters
        ----------
        residual : Tensor (P, N)
            Current residual X^T - D @ (Z ⊙ S)^T.
        """
        sqrt_inv_alpha = float(np.sqrt(1.0 / alpha))

        for k in range(K):
            d_k = D[:, k]
            # w_k = z_k ⊙ s_k — the atom's actual contribution to the residual.
            w_k = Z[:, k] * S[:, k]

            # --- Add back atom k's contribution to the residual ---
            residual.addr_(d_k, w_k)

            # --- Sample d_k ---
            if sample_d:
                sig_dk = 1.0 / (phi * float(torch.dot(w_k, w_k)) + P)
                mu_dk = (phi * sig_dk) * (residual @ w_k)
                d_k = mu_dk + torch.randn(
                    P, dtype=self.dtype, device=self.device
                ) * np.sqrt(sig_dk)
                D[:, k] = d_k

            dtd = float(torch.dot(d_k, d_k))

            # Projection of the residual on atom k. The core computes this twice
            # (all N in the z-step, the active columns in the s-step); residual
            # and d_k are unchanged between them, so once is enough.
            dk_proj = residual.T @ d_k  # (N,)

            # --- Sample z_k ---
            if sample_z:
                active = Z[:, k] > 0.5
                # For inactive entries, impute s from the prior N(0, 1/alpha).
                s_full = torch.where(
                    active,
                    S[:, k],
                    torch.randn(N, dtype=self.dtype, device=self.device)
                    * sqrt_inv_alpha,
                )

                # Log-likelihood ratio for z_ik = 1 vs 0
                log_ratio = (-0.5 * phi) * (
                    s_full * s_full * dtd - 2.0 * s_full * dk_proj
                )
                log_ratio.clamp_(-500.0, 500.0)

                # The core forms the probability as p1 / (p0 + p1) with
                # p1 = exp(L) * pi_k and p0 = 1 - pi_k. That is safe in float64
                # but NOT in float32: exp overflows above L ~ 88, so p1 becomes
                # inf, p1 / (p0 + p1) becomes inf/inf = nan, and `rand < nan` is
                # False — the atom would be switched OFF exactly when the
                # evidence most strongly favours switching it ON. The clamp to
                # +/-500 does not help; it is itself a float64-era bound.
                #
                # Rearranged, that same expression is a logistic:
                #     p1/(p0+p1) = 1 / (1 + e^-L * (1-pi)/pi)
                #                = sigmoid(L + logit(pi))
                # which is algebraically identical, saturates instead of
                # overflowing, and is exact in float32 at any L.
                prob = torch.sigmoid(log_ratio + torch.logit(pi[k]))

                Z[:, k] = (
                    torch.rand(N, dtype=self.dtype, device=self.device) < prob
                ).to(self.dtype)

            # --- Sample s_k ---
            if sample_s:
                # Refresh active mask after the Z update. Entries that are
                # inactive get exactly 0, as in the core's `S[:, k] = 0.0`
                # followed by a write to the active positions only.
                active = Z[:, k] > 0.5
                sig_s = 1.0 / (alpha + phi * dtd)
                mu_s = (phi * sig_s) * dk_proj
                s_new = mu_s + torch.randn(
                    N, dtype=self.dtype, device=self.device
                ) * np.sqrt(sig_s)
                S[:, k] = torch.where(
                    active, s_new, torch.zeros((), dtype=self.dtype, device=self.device)
                )

            # --- Subtract atom k's (updated) contribution ---
            residual.addr_(d_k, Z[:, k] * S[:, k], alpha=-1.0)

        return residual, D, Z, S

    def _sample_pi(
        self,
        Z: torch.Tensor,
        N: int,
        a0: float,
        b0: float,
    ) -> torch.Tensor:
        """Sample pi_k ~ Beta(a0/K + sum_i z_ik, b0(K-1)/K + N - sum_i z_ik).

        Matches Eq. 20 of the inference document / SamplePi.m.
        """
        K = Z.shape[1]
        sum_z = Z.sum(dim=0, dtype=_SCALAR_DTYPE).cpu()  # (K,)
        pi = torch.distributions.Beta(
            sum_z + a0 / K,
            b0 * (K - 1) / K + N - sum_z,
        ).sample()
        return pi.to(dtype=self.dtype, device=self.device)

    def _sample_alpha(
        self,
        S: torch.Tensor,
        e0: float,
        f0: float,
        Z: torch.Tensor,
        alpha: float,
    ) -> float:
        """Sample gamma_s (weight precision) from its Gamma posterior.

        Matches Samplealpha.m: the rate includes the marginal contribution of
        the zero entries in S that would have been drawn from N(0, 1/alpha).
        """
        n_total = Z.numel()
        n_active = int((Z > 0.5).sum())
        shape = e0 + 0.5 * n_total
        rate = (
            f0
            + 0.5 * float((S * S).sum(dtype=_SCALAR_DTYPE))
            + 0.5 * (n_total - n_active) / alpha
        )
        return self._sample_gamma(shape, rate)

    def _sample_phi(
        self,
        residual: torch.Tensor,
        c0: float,
        d0: float,
    ) -> float:
        """Sample gamma_eps (noise precision) from its Gamma posterior.

        Matches Samplephi.m.
        """
        n_elements = residual.numel()
        shape = c0 + 0.5 * n_elements
        rate = d0 + 0.5 * float((residual * residual).sum(dtype=_SCALAR_DTYPE))
        return self._sample_gamma(shape, rate)

    @staticmethod
    def _sample_gamma(shape: float, rate: float) -> float:
        """Draw one Gamma(shape, rate) in float64 on the CPU.

        NumPy's rng.gamma takes a *scale*; torch's Gamma takes a *rate*, and
        rate = 1/scale — the core's `rng.gamma(shape, 1.0 / rate)` is this same
        draw. Shapes here run to ~0.5 * P * N, hence float64.
        """
        return float(
            torch.distributions.Gamma(
                torch.tensor(shape, dtype=_SCALAR_DTYPE),
                torch.tensor(rate, dtype=_SCALAR_DTYPE),
            ).sample()
        )

    # ------------------------------------------------------------------
    # Device / RNG helpers
    # ------------------------------------------------------------------

    def _seed_generators(self) -> None:
        """Seed the global torch RNGs (mirrors FDDLGPU).

        torch.distributions draws from the global generator, so the scalar
        conditionals and the tensor draws have to share it — a private
        Generator would cover only half the sampler and give a false sense of
        reproducibility.
        """
        if self.random_state is None:
            return
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
        return float(np.sqrt(1.0 / self._phi))

    @property
    def usage_probabilities(self) -> Optional[np.ndarray]:
        """Per-atom usage probabilities pi_k."""
        return self._pi
