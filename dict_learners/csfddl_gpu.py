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
            k: int = 2048,
            lambda1: float = 0.1,
            lambda2: float = 0.1,
            eta: float = 1.0,
            max_iter: int = 64,
            lr: float = 0.01,
            ipm_iters: int = 15,
            weighting: str = 'inverse_freq',
            effective_number_beta: float = 0.9999,
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
        return torch.sign(X) * torch.nn.functional.relu(torch.abs(X) - tau)

    def _step_size(self, D: torch.Tensor, max_weight: float = 1.0):
        """
        Lipschitz-safe step size, adjusted for the maximum class weight.
        Weighting scales gradients by w_i, so worst-case Lipschitz is max_w * L.
        """
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
    def fit(self, training_graph_embeddings, y_train):
        print(f"Training {self.name} on [{self.device}] with weighting='{self.weighting}'...")

        if torch.is_tensor(y_train):
            y_train = y_train.cpu().numpy()
        else:
            y_train = np.array(y_train)

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
            print(f"  Class {c}: {self.class_sizes_[i]} samples, weight = {self.class_weights_[i]:.4f}")

        A_np = np.hstack(A_grouped)
        A = torch.tensor(A_np, dtype=torch.float32, device=self.device)

        features = A.shape[0]
        total_atoms = self.k * n_classes

        torch.manual_seed(42)
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

        for it in range(self.max_iter):
            X = self._update_X(A, self.D, X, self.k, n_classes,
                               self.class_sizes_, self.class_weights_)
            self.D = self._update_D(A, self.D, X, self.k, n_classes,
                                    self.class_sizes_)

        self.X_train = X.cpu().numpy()
        self.D = self.D.cpu()

        col_start = 0
        for i, c in enumerate(self.classes_):
            col_end = col_start + self.class_sizes_[i]
            Xi = X[:, col_start:col_end].cpu().numpy()
            self.M_i[c] = np.mean(Xi, axis=1)
            col_start = col_end

        return self

    def infer(self, infer_graph_embeddings):
        """
        Inference is NOT weighted — test samples have no known labels.
        Uses standard ISTA over the full learned dictionary.
        """
        A_test = torch.tensor(infer_graph_embeddings.T, dtype=torch.float32,
                              device=self.device)
        D_gpu = self.D.to(self.device)
        Z = torch.zeros((D_gpu.shape[1], A_test.shape[1]), device=self.device)
        t = self._step_size(D_gpu)

        for _ in range(self.ipm_iters * 2):
            grad = -2 * D_gpu.T @ (A_test - D_gpu @ Z)
            Z = Z - t * grad
            Z = self._soft_threshold(Z, self.lambda1 * t)

        return Z.T.cpu().numpy()
