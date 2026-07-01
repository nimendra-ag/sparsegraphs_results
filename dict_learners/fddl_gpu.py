import torch
import numpy as np
import random
from dict_learners.dict_learner import DictLearner

class FDDLGPU(DictLearner):
    def __init__(
            self,
            k: int = 256,
            lambda1: float = 0.1,
            lambda2: float = 0.1,
            eta: float = 1.0,
            max_iter: int = 64,
            lr: float = 0.01,
            ipm_iters: int = 15
    ):
        super().__init__(name="FDDLGPU")
        self.k = k  
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.max_iter = max_iter
        self.lr = lr
        self.ipm_iters = ipm_iters
        
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
        # ord=2 calculates spectral norm (largest singular value)
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

    def fit(self, training_graph_embeddings, y_train):
        print(f"Training {self.name} on context [{self.device}]...")
        
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

        # Setup weights on GPU
        torch.manual_seed(42)                  # <-- add this
        torch.cuda.manual_seed(42)
        self.D = torch.zeros((features, total_atoms), device=self.device)
        col_start = 0
        
        for i in range(n_classes):
            Ai = A[:, col_start:col_start + self.class_sizes_[i]]
            idx = torch.randint(0, self.class_sizes_[i], (self.k,), device=self.device)
            Di = Ai[:, idx]
            Di = Di / torch.norm(Di, dim=0) 
            self.D[:, i * self.k:(i + 1) * self.k] = Di
            col_start += self.class_sizes_[i]

        X = torch.zeros((total_atoms, sum(self.class_sizes_)), device=self.device)

        # Execute Alternate Optimizations directly on VRAM 
        for it in range(self.max_iter):
            X = self._update_X(A, self.D, X, self.k, n_classes, self.class_sizes_)
            self.D = self._update_D(A, self.D, X, self.k, n_classes, self.class_sizes_)

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