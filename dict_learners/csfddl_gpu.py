import os
import json

import torch
import numpy as np
from dict_learners.dict_learner import DictLearner


class CSFDDLGPU(DictLearner):
    """
    Cost-Sensitive Fisher Discrimination Dictionary Learning (CS-FDDL) — GPU version.

    Extends vanilla FDDL by injecting per-class weights into:
      1. Reconstruction gradients (global, self-reconstruction, cross-suppression)
      2. Fisher criterion (weighted within/between-class scatter + weighted global mean)

    NOT weighted (by design):
      - Sparsity threshold: sparsity level is a modeling choice, not class-dependent
      - Dictionary update: atoms are normalized regardless; codes already carry the weighting
      - Inference: test samples have no known labels

    Weighting schemes:
      - 'inverse_freq':   w_i = N / (C * n_i)           — standard balanced weighting
      - 'inverse_sqrt':   w_i = sqrt(N / (C * n_i))     — gentler rebalancing
      - 'effective_number': w_i = (1 - beta) / (1 - beta^n_i)  — Cui et al., CVPR 2019
    """

    def __init__(
            self,
            k: int = 128,
            lambda1: float = 0.1,
            lambda2: float = 0.1,
            eta: float = 1.0,
            max_iter: int = 64,
            lr: float = 0.01,
            ipm_iters: int = 15,
            weighting: str = 'inverse_freq',
            effective_number_beta: float = 0.9999,
            seed: int = 42
    ):
        super().__init__(name="CSFDDLGPU")
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.max_iter = max_iter
        self.lr = lr
        self.ipm_iters = ipm_iters
        self.weighting = weighting
        self.effective_number_beta = effective_number_beta
        self.seed = seed

        # Check and assign GPU automatically
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.D = None
        self.X_train = None
        self.classes_ = None
        self.class_sizes_ = None
        self.class_weights_ = None
        self.M_i = {}

    # ------------------------------------------------------------------ #
    #  Weighting
    # ------------------------------------------------------------------ #
    def _compute_weights(self, class_sizes, n_classes):
        """
        Compute per-class weights and normalize so they sum to n_classes.
        This preserves overall gradient magnitude while rebalancing across classes.
        """
        N = sum(class_sizes)

        if self.weighting == 'inverse_freq':
            # w_i = N / (C * n_i)  — same as sklearn class_weight='balanced'
            raw = [N / (n_classes * class_sizes[i]) for i in range(n_classes)]

        elif self.weighting == 'inverse_sqrt':
            # Gentler version — less aggressive rebalancing
            raw = [np.sqrt(N / (n_classes * class_sizes[i])) for i in range(n_classes)]

        elif self.weighting == 'effective_number':
            # Cui et al. CVPR 2019: E_n = (1 - beta^n) / (1 - beta)
            beta = self.effective_number_beta
            effective = [(1.0 - beta ** class_sizes[i]) / (1.0 - beta)
                         for i in range(n_classes)]
            raw = [1.0 / e for e in effective]

        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting}")

        # Normalize: sum of weights = n_classes (preserves gradient scale)
        total = sum(raw)
        weights = [w * n_classes / total for w in raw]

        return weights

    # ------------------------------------------------------------------ #
    #  Core building blocks
    # ------------------------------------------------------------------ #
    def _soft_threshold(self, X: torch.Tensor, tau: float) -> torch.Tensor:
        # Pytorch handles soft thresholding easily with sign and relu
        return torch.sign(X) * torch.nn.functional.relu(torch.abs(X) - tau)

    def _step_size(self, D: torch.Tensor, max_weight: float = 1.0):
        """
        Lipschitz-safe step size, adjusted for the maximum class weight.
        Weighting scales gradients by w_i, so worst-case Lipschitz is max_w * L.
        """
        # ord=2 calculates spectral norm (largest singular value).
        # GUARD: matrix_norm(ord=2) is an SVD that HANGS INDEFINITELY (and
        # uninterruptibly, since it blocks inside a CUDA C call) if D contains
        # NaN/Inf. Check first so a poisoned dictionary fails loudly here instead
        # of freezing the terminal. This .item() forces a sync, but only once
        # per call and it is the cheap price of not hanging.
        if not torch.isfinite(D).all():
            n_nan = int(torch.isnan(D).sum())
            n_inf = int(torch.isinf(D).sum())
            bad_cols = int((~torch.isfinite(D)).any(dim=0).sum())
            raise FloatingPointError(
                f"_step_size: dictionary D is non-finite (nan={n_nan}, inf={n_inf}, "
                f"{bad_cols}/{D.shape[1]} atoms affected). Aborting before the SVD, "
                f"which would hang. Most likely an atom was normalised by a zero "
                f"norm (0/0) during init — see the zero-norm warning in fit()."
            )
        L = 2.0 * (torch.linalg.matrix_norm(D, ord=2) ** 2) \
            + 2.0 * self.lambda2 * (1.0 + self.eta)
        return 1.0 / (1.05 * max_weight * L)

    # ------------------------------------------------------------------ #
    #  X-update  (cost-sensitive)
    # ------------------------------------------------------------------ #
    def _compute_gradient_Xi(self, Ai, D, Xi, class_idx, k,
                              lambda2, eta, M_weighted, w_i):
        """
        Same four gradient components as vanilla FDDL, but:
          - Uses weighted global mean M_weighted instead of plain mean
          - Entire gradient is scaled by class weight w_i
        """
        # --- Reconstruction: global fidelity ---
        grad_global = -2 * D.T @ (Ai - D @ Xi)

        # --- Reconstruction: self (class i's own sub-dictionary) ---
        grad_local = torch.zeros_like(Xi)
        start_idx = class_idx * k
        end_idx = start_idx + k
        Di = D[:, start_idx:end_idx]
        Xii = Xi[start_idx:end_idx, :]
        grad_local[start_idx:end_idx, :] = -2 * Di.T @ (Ai - Di @ Xii)

        # --- Reconstruction: cross-suppression (other sub-dicts should NOT reconstruct) ---
        grad_sabotage = torch.zeros_like(Xi)
        for j in range(D.shape[1] // k):
            if j != class_idx:
                j_start = j * k
                j_end = j_start + k
                Dj = D[:, j_start:j_end]
                Xij = Xi[j_start:j_end, :]
                grad_sabotage[j_start:j_end, :] = 2 * Dj.T @ (Dj @ Xij)

        # --- Fisher: within-class + between-class scatter ---
        # Uses M_weighted (weighted global mean) instead of plain M_global
        Mi = torch.mean(Xi, dim=1, keepdim=True)
        grad_fisher = 2 * (Xi - Mi) - 2 * (Mi - M_weighted) + 2 * eta * Xi

        # Scale entire gradient by class weight
        return w_i * (grad_global + grad_local + grad_sabotage + (lambda2 * grad_fisher))

    def _update_X(self, A, D, X, k, n_classes, class_sizes, class_weights):
        """
        Cost-sensitive X-update:
          - Computes weighted global mean (each class centroid weighted by w_i)
          - Passes w_i to gradient computation
          - Sparsity threshold is NOT weighted (intentional)
        """
        # --- Weighted global mean ---
        # Plain mean: m = (1/N) * Σ x_k  → biased toward majority
        # Weighted:   m_w = Σ w_i*m_i / Σ w_i  → equal class influence
        W_total = sum(class_weights)
        M_weighted = torch.zeros((X.shape[0], 1), device=X.device)
        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            M_i = torch.mean(X[:, col_start:col_end], dim=1, keepdim=True)
            M_weighted += class_weights[i] * M_i
            col_start = col_end
        M_weighted /= W_total

        # Step size accounts for max weight
        max_w = max(class_weights)
        t = self._step_size(D, max_weight=max_w)

        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            w_i = class_weights[i]
            Ai = A[:, col_start:col_end]
            Xi = X[:, col_start:col_end]

            for _ in range(self.ipm_iters):
                grad = self._compute_gradient_Xi(
                    Ai, D, Xi, i, k, self.lambda2, self.eta, M_weighted, w_i
                )
                Xi = Xi - t * grad
                # Sparsity threshold is UNWEIGHTED — sparsity is a modeling choice,
                # not class-dependent. Weighting it would force minority codes to be
                # sparser (higher w_i → higher threshold), which is undesirable.
                Xi = self._soft_threshold(Xi, self.lambda1 * t)

            X[:, col_start:col_end] = Xi
            col_start = col_end

        return X

    # ------------------------------------------------------------------ #
    #  D-update  (unchanged from vanilla FDDL)
    # ------------------------------------------------------------------ #
    def _update_D(self, A, D, X, k, n_classes, class_sizes):
        """
        Dictionary update is NOT weighted. Rationale:
          - Each sub-dictionary D_i updates from its own class's data (A_i, X_ii)
          - Atoms are normalized to unit length regardless of class frequency
          - The codes X already carry the cost-sensitive signal from _update_X
          - Weighting D would require scaling the stacked least-squares targets,
            adding complexity with marginal benefit
        """
        for i in range(n_classes):
            start_idx = i * k
            end_idx = start_idx + k
            Di = D[:, start_idx:end_idx]
            Xi_all = X[start_idx:end_idx, :]

            A_hat = A.clone()
            for j in range(n_classes):
                if j != i:
                    j_start = j * k
                    j_end = j_start + k
                    A_hat -= D[:, j_start:j_end] @ X[j_start:j_end, :]

            col_start = sum(class_sizes[:i])
            col_end = col_start + class_sizes[i]
            Ai = A[:, col_start:col_end]
            Xii = Xi_all[:, col_start:col_end]

            # Equivalent slicing/concats in pyTorch
            X_others = torch.cat((Xi_all[:, :col_start], Xi_all[:, col_end:]), dim=1)
            zeros = torch.zeros((A.shape[0], X_others.shape[1]), device=self.device)

            Lambda_i = torch.cat((A_hat, Ai, zeros), dim=1)
            Zi = torch.cat((Xi_all, Xii, X_others), dim=1)

            for atom_idx in range(k):
                d_l = Di[:, atom_idx].view(-1, 1)
                z_l = Zi[atom_idx, :].view(1, -1)

                Y = Lambda_i - (Di @ Zi) + (d_l @ z_l)
                d_new = Y @ z_l.T
                norm_d = torch.norm(d_new)
                Di[:, atom_idx] = (d_new / norm_d).flatten() if norm_d > 1e-10 else d_l.flatten()

            D[:, start_idx:end_idx] = Di
        return D

    # ------------------------------------------------------------------ #
    #  fit / infer
    # ------------------------------------------------------------------ #
    def fit(self, training_graph_embeddings, y_train=None):
        if y_train is None:
            raise ValueError("CSFDDLGPU.fit requires y_train (it is a supervised learner).")
        print(f"Training {self.name} on context [{self.device}] "
              f"(k={self.k}/class, max_iter={self.max_iter}, ipm_iters={self.ipm_iters}, "
              f"weighting='{self.weighting}')...",
              flush=True)

        # Ensure y_train is numpy array for logical indexing
        if torch.is_tensor(y_train): y_train = y_train.cpu().numpy()
        else: y_train = np.array(y_train)

        # Build class distributions
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)

        self.class_sizes_ = []
        A_grouped = []
        for c in self.classes_:
            A_c = training_graph_embeddings[y_train == c].T
            A_grouped.append(A_c)
            self.class_sizes_.append(A_c.shape[1])

        # --- Compute and report class weights ---
        self.class_weights_ = self._compute_weights(self.class_sizes_, n_classes)
        for i, c in enumerate(self.classes_):
            print(f"  [weights] class {c}: {self.class_sizes_[i]} samples, "
                  f"weight = {self.class_weights_[i]:.4f}", flush=True)

        # Convert concatenated data over to the GPU
        A_np = np.hstack(A_grouped)
        A = torch.tensor(A_np, dtype=torch.float32, device=self.device)

        features = A.shape[0]
        total_atoms = self.k * n_classes
        print(f"  [init] features={features}, classes={n_classes}, "
              f"class_sizes={self.class_sizes_}, total_atoms={total_atoms}", flush=True)

        # Setup weights on GPU — seed injected per run (Monte Carlo CV)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.D = torch.zeros((features, total_atoms), device=self.device)
        col_start = 0

        for i in range(n_classes):
            Ai = A[:, col_start:col_start + self.class_sizes_[i]]
            idx = torch.randint(0, self.class_sizes_[i], (self.k,), device=self.device)
            Di = Ai[:, idx]
            norms = torch.norm(Di, dim=0)
            # A sampled column with zero norm is an all-zero (degenerate) training
            # embedding. Dividing by it gives 0/0 = NaN, which later makes the
            # spectral-norm SVD hang forever. Warn loudly and leave those atoms as
            # zero columns (guarded division) so the run stays diagnosable.
            n_zero = int((norms == 0).sum())
            if n_zero:
                print(f"  [init][WARN] class {self.classes_[i]}: sampled {n_zero}/{self.k} "
                      f"zero-norm (all-zero) embeddings as atoms; leaving them as zero "
                      f"columns instead of dividing by zero. Consider dropping all-zero "
                      f"training rows upstream.", flush=True)
                norms = norms.clone()
                norms[norms == 0] = 1.0
            Di = Di / norms
            self.D[:, i * self.k:(i + 1) * self.k] = Di
            col_start += self.class_sizes_[i]

        if not torch.isfinite(self.D).all():
            print(f"  [init][WARN] initial D has "
                  f"{int((~torch.isfinite(self.D)).sum())} non-finite entries", flush=True)

        X = torch.zeros((total_atoms, sum(self.class_sizes_)), device=self.device)

        # Execute Alternate Optimizations directly on VRAM. Each iteration is
        # timed and D's health is checked so a stall or divergence is pinned to a
        # specific iteration (flush=True so the last line survives even a hang).
        import time as _time
        for it in range(self.max_iter):
            _t0 = _time.perf_counter()
            print(f"  [iter {it + 1:>2}/{self.max_iter}] update_X...", end="", flush=True)
            X = self._update_X(A, self.D, X, self.k, n_classes,
                               self.class_sizes_, self.class_weights_)
            print(" update_D...", end="", flush=True)
            self.D = self._update_D(A, self.D, X, self.k, n_classes, self.class_sizes_)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            d_nan = int(torch.isnan(self.D).sum())
            print(f" done in {_time.perf_counter() - _t0:6.2f}s"
                  f" | max|X|={X.abs().max().item():.3g} D_nan={d_nan}", flush=True)
            if d_nan:
                print(f"  [iter {it + 1}][WARN] D diverged to NaN this iteration; "
                      f"subsequent _step_size() will abort.", flush=True)

        # Transfer back to RAM for external pipelines
        self.X_train = X.cpu().numpy()
        self.D = self.D.cpu()  # Store globally decoupled from device

        col_start = 0
        for i, c in enumerate(self.classes_):
            col_end = col_start + self.class_sizes_[i]
            Xi = X[:, col_start:col_end].cpu().numpy()
            self.M_i[c] = np.mean(Xi, axis=1)
            col_start = col_end

        return self

    def infer(self, infer_graph_embeddings):
        """Uses ISTA logic on GPU quickly.

        Inference is NOT weighted — test samples have no known labels, so the
        step size falls back to the unweighted (max_weight=1) Lipschitz bound.
        """
        # Uploads items to Device
        A_test = torch.tensor(infer_graph_embeddings.T, dtype=torch.float32, device=self.device)
        D_gpu = self.D.to(self.device)
        Z = torch.zeros((D_gpu.shape[1], A_test.shape[1]), device=self.device)
        t = self._step_size(D_gpu)

        for _ in range(self.ipm_iters * 2):
            grad = -2 * D_gpu.T @ (A_test - D_gpu @ Z)
            Z = Z - t * grad
            Z = self._soft_threshold(Z, self.lambda1 * t)

        # Downloads array structure down locally.
        return Z.T.cpu().numpy()

    # --- Persistence ---------------------------------------------------------
    # Learned state needed for inference/SRC: the dictionary D (features x atoms)
    # and, for FDDL-native SRC, the per-class mean codes M_i. The fitted class
    # weights ride along for provenance (they are a property of the training
    # distribution, not of the config). X_train is a training by-product and is
    # deliberately NOT persisted. Raw arrays go to .npy/.npz (no torch/class
    # dependency on load); scalars to JSON.
    _CONFIG_FILE = "csfddl_config.json"
    _D_FILE = "csfddl_D.npy"
    _STATE_FILE = "csfddl_state.npz"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "k": self.k,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "eta": self.eta,
            "max_iter": self.max_iter,
            "lr": self.lr,
            "ipm_iters": self.ipm_iters,
            "weighting": self.weighting,
            "effective_number_beta": self.effective_number_beta,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self.D is None:
            raise ValueError("CSFDDLGPU has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # D may be a torch tensor (CPU) or numpy — normalise to numpy.
        D_np = self.D.cpu().numpy() if torch.is_tensor(self.D) else np.asarray(self.D)
        np.save(os.path.join(dirpath, self._D_FILE), D_np)

        # classes_ defines the class order; M_i and class_weights are stacked in
        # that same order.
        classes = np.asarray(self.classes_)
        class_sizes = np.asarray(self.class_sizes_ if self.class_sizes_ is not None else [])
        class_weights = np.asarray(self.class_weights_ if self.class_weights_ is not None else [])
        if self.M_i:
            M_i = np.stack([np.asarray(self.M_i[c]) for c in self.classes_], axis=0)
        else:
            M_i = np.empty((0,))
        np.savez(
            os.path.join(dirpath, self._STATE_FILE),
            classes=classes,
            class_sizes=class_sizes,
            class_weights=class_weights,
            M_i=M_i,
        )

    @classmethod
    def load(cls, dirpath: str) -> "CSFDDLGPU":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)

        learner = cls(
            k=config["k"],
            lambda1=config["lambda1"],
            lambda2=config["lambda2"],
            eta=config["eta"],
            max_iter=config["max_iter"],
            lr=config["lr"],
            ipm_iters=config["ipm_iters"],
            weighting=config["weighting"],
            effective_number_beta=config["effective_number_beta"],
            seed=config["seed"],
        )

        # Keep D on CPU; infer()/SRC move it to the active device on demand.
        D_np = np.load(os.path.join(dirpath, cls._D_FILE))
        learner.D = torch.tensor(D_np, dtype=torch.float32)

        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.classes_ = state["classes"]
        learner.class_sizes_ = state["class_sizes"].tolist()
        learner.class_weights_ = state["class_weights"].tolist()
        M_i_arr = state["M_i"]
        learner.M_i = {}
        if M_i_arr.size:
            # Key by the same class objects as classes_ (matches fit()), so the
            # SRC classifier's M_i[c] lookups resolve identically.
            for i, c in enumerate(learner.classes_):
                learner.M_i[c] = M_i_arr[i]
        return learner
