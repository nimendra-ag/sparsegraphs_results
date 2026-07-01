import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


class SRCClassifier:
    """
    Loosely coupled SRC-style classifier for any structured dictionary learner.

    Works with any dict learner that exposes:
      - .D           (dictionary matrix — numpy array OR torch tensor)
      - .k           (atoms per class)
      - .classes_    (array of class labels)
      - .M_i         (optional dict of per-class mean codes, for FDDL-native mode)

    Automatically detects GPU (torch) vs CPU (numpy) from the dict learner.

    Usage:
        src = SRCClassifier(trained_dict_learner, gamma=0.0)   # pure SRC
        src = SRCClassifier(trained_dict_learner, gamma=0.5)   # FDDL-native
        results = src.evaluate(test_embeddings, y_test)
    """

    def __init__(self, dict_learner, gamma: float = 0.0):
        """
        gamma=0.0  -> pure SRC (residual only), works with FDDL/DPL/DLSI/any
        gamma>0.0  -> FDDL-native (residual + coefficient distance), needs M_i
        """
        self.dl = dict_learner
        self.gamma = gamma

        # Detect if dict learner uses torch
        self._use_torch = self._check_torch()

        if self._use_torch:
            import torch
            self._torch = torch
            self.device = getattr(dict_learner, 'device', torch.device('cpu'))

    def _check_torch(self):
        try:
            import torch
            return torch.is_tensor(self.dl.D)
        except ImportError:
            return False

    def _encode_torch(self, embeddings):
        """ISTA sparse coding on GPU. Input/output are numpy arrays."""
        torch = self._torch

        D_gpu = self.dl.D.to(self.device)
        A = torch.tensor(embeddings.T, dtype=torch.float32, device=self.device)

        lambda1 = getattr(self.dl, 'lambda1', 0.1)
        ipm_iters = getattr(self.dl, 'ipm_iters', 15)

        # Step size from spectral norm
        L = 2.0 * (torch.linalg.matrix_norm(D_gpu, ord=2) ** 2)
        t = 1.0 / (1.05 * L)

        Z = torch.zeros((D_gpu.shape[1], A.shape[1]), device=self.device)
        for _ in range(ipm_iters * 2):
            grad = -2 * D_gpu.T @ (A - D_gpu @ Z)
            Z = Z - t * grad
            Z = torch.sign(Z) * torch.nn.functional.relu(torch.abs(Z) - lambda1 * t)

        return Z.cpu().numpy(), D_gpu.cpu().numpy()

    def _encode_numpy(self, embeddings):
        """ISTA sparse coding on CPU. Input/output are numpy arrays."""
        D = np.array(self.dl.D)  # safe copy, handles both np and torch-cpu
        A = embeddings.T

        lambda1 = getattr(self.dl, 'lambda1', 0.1)
        ipm_iters = getattr(self.dl, 'ipm_iters', 15)

        L = 2.0 * (np.linalg.norm(D, 2) ** 2)
        t = 1.0 / (1.05 * L)

        Z = np.zeros((D.shape[1], A.shape[1]))
        for _ in range(ipm_iters * 2):
            grad = -2 * D.T @ (A - D @ Z)
            Z = np.sign(Z - t * grad) * np.maximum(
                np.abs(Z - t * grad) - lambda1 * t, 0.0
            )

        return Z, D

    def predict(self, test_embeddings):
        """
        Returns: (preds, scores)
            preds  — predicted labels, numpy array of shape (n_samples,)
            scores — per-class SRC scores, numpy array of shape (n_samples, n_classes)
                     lower score = better reconstruction = more likely that class
        """
        # --- Encode (GPU or CPU) ---
        if self._use_torch:
            Z, D = self._encode_torch(test_embeddings)
        else:
            Z, D = self._encode_numpy(test_embeddings)

        # --- Classify on CPU (always numpy from here) ---
        A = test_embeddings.T  # (features, n_samples)
        k = self.dl.k
        classes = self.dl.classes_
        n_classes = len(classes)

        has_means = (
            self.gamma > 0
            and hasattr(self.dl, 'M_i')
            and len(self.dl.M_i) > 0
        )

        # Convert M_i values to numpy if they are torch tensors
        M_i = {}
        if has_means:
            for c, m in self.dl.M_i.items():
                if self._use_torch and self._torch.is_tensor(m):
                    M_i[c] = m.cpu().numpy()
                else:
                    M_i[c] = np.asarray(m)

        n_samples = A.shape[1]
        preds = np.empty(n_samples, dtype=classes.dtype)
        all_scores = np.zeros((n_samples, n_classes))

        for col in range(n_samples):
            z = Z[:, col]
            a = A[:, col]

            for idx, c in enumerate(classes):
                s, e = idx * k, (idx + 1) * k

                z_c = np.zeros_like(z)
                z_c[s:e] = z[s:e]
                resid = np.sum((a - D @ z_c) ** 2)

                score = resid
                if has_means:
                    score += self.gamma * np.sum((z - M_i[c]) ** 2)

                all_scores[col, idx] = score

            preds[col] = classes[np.argmin(all_scores[col])]

        return preds, all_scores

    def _scores_to_proba(self, all_scores):
        """
        Convert SRC residual scores to pseudo-probabilities.
        Lower residual = better fit = higher probability, so we negate
        then softmax across classes.
        """
        neg_scores = -all_scores
        # Shift for numerical stability
        neg_scores -= neg_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(neg_scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def evaluate(self, test_embeddings, y_true, minority_label=1):
        """
        Returns Precision, Recall, F1-Score, ROC-AUC, PR-AUC
        for the minority (positive) class.
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score,
            balanced_accuracy_score
        )

        y_pred, all_scores = self.predict(test_embeddings)
        probas = self._scores_to_proba(all_scores)

        # Find which column index corresponds to the minority label
        minority_idx = np.where(self.dl.classes_ == minority_label)[0][0]
        minority_proba = probas[:, minority_idx]

        return {
            "balanced_acc":  round(balanced_accuracy_score(y_true, y_pred), 4),
            "precision":     round(precision_score(y_true, y_pred, pos_label=minority_label), 4),
            "recall":        round(recall_score(y_true, y_pred, pos_label=minority_label), 4),
            "f1_score":      round(f1_score(y_true, y_pred, pos_label=minority_label), 4),
            "roc_auc":       round(roc_auc_score(y_true, minority_proba), 4),
            "pr_auc":        round(average_precision_score(y_true, minority_proba), 4),
        }