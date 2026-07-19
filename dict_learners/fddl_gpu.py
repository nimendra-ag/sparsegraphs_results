import os
import json

import torch
import numpy as np
import random
from dict_learners.dict_learner import DictLearner

class FDDLGPU(DictLearner):
    def __init__(
            self,
            k: int = 2048,
            lambda1: float = 0.1,
            lambda2: float = 0.1,
            eta: float = 1.0,
            max_iter: int = 64,
            lr: float = 0.01,
            ipm_iters: int = 15,
            seed: int = 42
    ):
        super().__init__(name="FDDLGPU")
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.max_iter = max_iter
        self.lr = lr
        self.ipm_iters = ipm_iters
        self.seed = seed
        
        # Check and assign GPU automatically
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.D = None
        self.X_train = None
        self.classes_ = None
        self.class_sizes_ = None
        self.M_i = {} 

    def _soft_threshold(self, X: torch.Tensor, tau: float) -> torch.Tensor:
        # Pytorch handles soft thresholding easily with sign and relu
        return torch.sign(X) * torch.nn.functional.relu(torch.abs(X) - tau)
    
    def _step_size(self, D: torch.Tensor):
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
        L = 2.0 * (torch.linalg.matrix_norm(D, ord=2) ** 2) + 2.0 * self.lambda2 * (1.0 + self.eta)
        return 1.0 / (1.05 * L)

    def _compute_gradient_Xi(self, Ai, D, Xi, class_idx, k, lambda2, eta, M_global):
        grad_global = -2 * D.T @ (Ai - D @ Xi)
        
        grad_local = torch.zeros_like(Xi)
        start_idx = class_idx * k
        end_idx = start_idx + k
        Di = D[:, start_idx:end_idx]
        Xii = Xi[start_idx:end_idx, :]
        grad_local[start_idx:end_idx, :] = -2 * Di.T @ (Ai - Di @ Xii)

        grad_sabotage = torch.zeros_like(Xi)
        for j in range(D.shape[1] // k):
            if j != class_idx:
                j_start = j * k
                j_end = j_start + k
                Dj = D[:, j_start:j_end]
                Xij = Xi[j_start:j_end, :]
                grad_sabotage[j_start:j_end, :] = 2 * Dj.T @ (Dj @ Xij)

        Mi = torch.mean(Xi, dim=1, keepdim=True)
        grad_fisher = 2 * (Xi - Mi) - 2 * (Mi - M_global) + 2 * eta * Xi

        return grad_global + grad_local + grad_sabotage + (lambda2 * grad_fisher)

    def _update_X(self, A, D, X, k, n_classes, class_sizes):
        M_global = torch.mean(X, dim=1, keepdim=True)
        t = self._step_size(D)
        
        col_start = 0
        for i in range(n_classes):
            col_end = col_start + class_sizes[i]
            Ai = A[:, col_start:col_end]
            Xi = X[:, col_start:col_end]

            for _ in range(self.ipm_iters):
                grad = self._compute_gradient_Xi(Ai, D, Xi, i, k, self.lambda2, self.eta, M_global)
                Xi = Xi - t * grad
                Xi = self._soft_threshold(Xi, self.lambda1 * t)

            X[:, col_start:col_end] = Xi
            col_start = col_end
        return X

    def _update_D(self, A, D, X, k, n_classes, class_sizes):
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

    def fit(self, training_graph_embeddings, y_train=None):
        if y_train is None:
            raise ValueError("FDDLGPU.fit requires y_train (it is a supervised learner).")
        print(f"Training {self.name} on context [{self.device}] "
              f"(k={self.k}/class, max_iter={self.max_iter}, ipm_iters={self.ipm_iters})...",
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
            X = self._update_X(A, self.D, X, self.k, n_classes, self.class_sizes_)
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
        """Uses ISTA logic on GPU quickly"""
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
    # and, for FDDL-native SRC, the per-class mean codes M_i. X_train is a
    # training by-product and is deliberately NOT persisted. Raw arrays go to
    # .npy/.npz (no torch/class dependency on load); scalars to JSON.
    _CONFIG_FILE = "fddl_config.json"
    _D_FILE = "fddl_D.npy"
    _STATE_FILE = "fddl_state.npz"

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
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self.D is None:
            raise ValueError("FDDLGPU has no dictionary to save; fit the learner first.")
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # D may be a torch tensor (CPU) or numpy — normalise to numpy.
        D_np = self.D.cpu().numpy() if torch.is_tensor(self.D) else np.asarray(self.D)
        np.save(os.path.join(dirpath, self._D_FILE), D_np)

        # classes_ defines the class order; M_i is stacked in that same order.
        classes = np.asarray(self.classes_)
        class_sizes = np.asarray(self.class_sizes_ if self.class_sizes_ is not None else [])
        if self.M_i:
            M_i = np.stack([np.asarray(self.M_i[c]) for c in self.classes_], axis=0)
        else:
            M_i = np.empty((0,))
        np.savez(
            os.path.join(dirpath, self._STATE_FILE),
            classes=classes,
            class_sizes=class_sizes,
            M_i=M_i,
        )

    @classmethod
    def load(cls, dirpath: str) -> "FDDLGPU":
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
            seed=config["seed"],
        )

        # Keep D on CPU; infer()/SRC move it to the active device on demand.
        D_np = np.load(os.path.join(dirpath, cls._D_FILE))
        learner.D = torch.tensor(D_np, dtype=torch.float32)

        state = np.load(os.path.join(dirpath, cls._STATE_FILE), allow_pickle=False)
        learner.classes_ = state["classes"]
        learner.class_sizes_ = state["class_sizes"].tolist()
        M_i_arr = state["M_i"]
        learner.M_i = {}
        if M_i_arr.size:
            # Key by the same class objects as classes_ (matches fit()), so the
            # SRC classifier's M_i[c] lookups resolve identically.
            for i, c in enumerate(learner.classes_):
                learner.M_i[c] = M_i_arr[i]
        return learner