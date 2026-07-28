import numpy as np


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
        Returns the same scalar metric set (and the same key names) as
        `utils.evaluator.Evaluator._evaluate_model`, so SRC rows and sklearn
        rows in an aggregated MC-CV table are directly comparable:

          Macro-*      both classes weighted equally
          Accuracy / ROC-AUC / MCC   symmetric over the whole test set
          Minority-*   the minority (positive) class alone

        Note `Macro-Recall` is numerically identical to balanced accuracy,
        which this method used to report under the key `balanced_acc`.

        Values are returned unrounded; the MC-CV aggregator formats them.
        """
        from sklearn.metrics import (
            accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
            roc_auc_score, average_precision_score,
        )

        y_pred, all_scores = self.predict(test_embeddings)
        probas = self._scores_to_proba(all_scores)

        classes = np.asarray(self.dl.classes_)
        minority_idx = int(np.where(classes == minority_label)[0][0])
        majority_label = classes[classes != minority_label][0]
        majority_idx = int(np.where(classes == majority_label)[0][0])
        minority_proba = probas[:, minority_idx]
        majority_proba = probas[:, majority_idx]

        # Per-class precision / recall / F1, ordered [minority, majority].
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            labels=[minority_label, majority_label],
            average=None, zero_division=0,
        )

        # Average precision ignores true negatives, so each class is ranked by
        # its own probability column rather than sharing one score vector.
        ap_minority = average_precision_score(
            y_true, minority_proba, pos_label=minority_label)
        ap_majority = average_precision_score(
            y_true, majority_proba, pos_label=majority_label)

        return {
            "Macro-Precision": float(prec.mean()),
            "Macro-Recall": float(rec.mean()),
            "Macro-F1": float(f1.mean()),
            "Macro-PR-AUC": float((ap_minority + ap_majority) / 2),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "ROC-AUC": float(roc_auc_score(y_true, minority_proba)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "Minority-Precision": float(prec[0]),
            "Minority-Recall": float(rec[0]),
            "Minority-F1": float(f1[0]),
            "Minority-PR-AUC": float(ap_minority),
        }