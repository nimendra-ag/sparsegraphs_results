"""
utils/lcksvd_evaluator.py
=========================
Evaluates LC-KSVD's built-in linear classifier W_hat on held-out test data.

This is separate from the pipeline's Evaluator class, which trains and
evaluates external ML models (LR, GB, SVM, RF) on top of the sparse codes.
Here we instead use the classifier that LC-KSVD learned jointly with its
dictionary — W_hat — and report a full set of classification metrics.

Why both evaluators matter
--------------------------
The external Evaluator tells you how good the *sparse codes* are as features
for arbitrary classifiers.
This evaluator tells you how good LC-KSVD's *own* classifier is — i.e. how
well the joint learning of D, A, and W actually worked end-to-end.
Comparing the two gives insight into whether W_hat is the bottleneck or
whether the sparse codes themselves need improvement.

Metrics reported
----------------
  - Precision    : TP / (TP + FP)
  - Recall       : TP / (TP + FN)
  - F1-Score     : harmonic mean of Precision and Recall
  - ROC-AUC      : area under the ROC curve (uses raw classifier scores)
  - PR-AUC       : area under the Precision-Recall curve
                   More informative than ROC-AUC for imbalanced datasets
                   because ROC-AUC can look deceptively high when the
                   majority class dominates.

Label convention
----------------
The NCI dataset uses labels {-1, 1} (majority=-1, minority=1), NOT {0, 1}.
This evaluator handles arbitrary label values — it does not assume
0-based integer indices.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize


class LCKSVDEvaluator:
    """
    Evaluates the LC-KSVD internal linear classifier W_hat.

    Parameters
    ----------
    lcksvd_learner : LCKSVDLearner
        A fitted LCKSVDLearner instance (from dict_learners/lcksvd.py).
        The internal model is accessed via lcksvd_learner.lcksvd_model.
    """

    def __init__(self, lcksvd_learner) -> None:
        self._model = lcksvd_learner.lcksvd_model

        if self._model.D_hat is None:
            raise RuntimeError(
                "LCKSVDLearner is not fitted yet. Call .fit() before evaluating."
            )

        self._num_classes = self._model.num_classes_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_predictions(
        self,
        graph_embeddings: np.ndarray,
        unique_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the LC-KSVD internal classifier on WL graph embeddings.

        The core LCKSVD model uses 0-based atom/class indices internally.
        W_hat has shape (num_classes, K) where row i corresponds to the
        i-th unique label in sorted order. Predictions are mapped back to
        the original label values before returning.

        Parameters
        ----------
        graph_embeddings : (N, n) ndarray
        unique_labels    : (num_classes,) sorted array of original label values

        Returns
        -------
        y_pred   : (N,) predicted labels in the original label space
        y_scores : (N, num_classes) raw W_hat scores, columns ordered to
                   match unique_labels
        """
        # Transpose to column-major for the core LCKSVD convention
        Y = np.array(graph_embeddings, dtype=float).T   # (n, N)

        # Sparse codes via OMP on D_hat
        X = self._model.encode(Y)                        # (K, N)

        # Raw classifier scores — row i = score for class index i
        raw_scores = self._model.W_hat @ X              # (num_classes, N)

        # argmax gives 0-based class indices → map back to original labels
        pred_indices = np.argmax(raw_scores, axis=0)    # (N,)
        y_pred = unique_labels[pred_indices]             # original label values

        # Transpose so columns correspond to classes (sklearn convention)
        y_scores = raw_scores.T                         # (N, num_classes)

        return y_pred, y_scores

    # ------------------------------------------------------------------
    # Public evaluation method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        graph_embeddings_test: np.ndarray,
        y_test: np.ndarray,
        positive_label: int = 1,
    ) -> dict:
        """
        Compute Precision, Recall, F1, ROC-AUC, and PR-AUC for the
        LC-KSVD internal classifier on held-out test graphs.

        Parameters
        ----------
        graph_embeddings_test : (N_test, n) ndarray
            Raw WL embeddings for test graphs — NOT scaled. The internal
            classifier encodes via OMP on D_hat, so no MaxAbsScaler needed.

        y_test : (N_test,) array-like
            True class labels in the original label space (e.g. {-1, 1}).

        positive_label : int
            The label value of the positive (minority) class.
            Default is 1, which is the minority class in the NCI dataset.

        Returns
        -------
        metrics : dict with keys:
            'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
            'y_pred', 'y_scores', 'classification_report'
        """
        y_test = np.array(y_test)

        # Recover the sorted unique label values used during fit.
        # The core model maps label → 0-based index via np.unique's sorted order,
        # which is exactly what init_atom_labels and build_label_matrix use.
        unique_labels = np.unique(y_test)   # e.g. [-1, 1] or [0, 1]

        if len(unique_labels) != self._num_classes:
            raise ValueError(
                f"y_test contains {len(unique_labels)} unique labels but the "
                f"model was trained with {self._num_classes} classes."
            )

        y_pred, y_scores = self._get_predictions(graph_embeddings_test, unique_labels)

        is_binary = (self._num_classes == 2)

        if is_binary:
            # Locate the column in y_scores that corresponds to positive_label.
            # unique_labels is sorted, so its position gives the column index.
            pos_col = int(np.where(unique_labels == positive_label)[0][0])
            pos_scores = y_scores[:, pos_col]   # (N_test,) scores for positive class

            precision = precision_score(
                y_test, y_pred,
                pos_label=positive_label,
                average="binary",
                zero_division=0,
            )
            recall = recall_score(
                y_test, y_pred,
                pos_label=positive_label,
                average="binary",
                zero_division=0,
            )
            f1 = f1_score(
                y_test, y_pred,
                pos_label=positive_label,
                average="binary",
                zero_division=0,
            )
            roc_auc = roc_auc_score(y_test, pos_scores)
            pr_auc = average_precision_score(
                y_test, pos_scores, pos_label=positive_label
            )

        else:
            # Multiclass: macro averaging for Precision, Recall, F1;
            # OvR macro for AUC metrics.
            precision = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )

            y_bin = label_binarize(y_test, classes=unique_labels)   # (N, num_classes)
            roc_auc = roc_auc_score(
                y_bin, y_scores, multi_class="ovr", average="macro"
            )
            pr_auc = average_precision_score(y_bin, y_scores, average="macro")

        report = classification_report(y_test, y_pred, zero_division=0)

        return {
            "precision":              precision,
            "recall":                 recall,
            "f1":                     f1,
            "roc_auc":                roc_auc,
            "pr_auc":                 pr_auc,
            "y_pred":                 y_pred,
            "y_scores":               y_scores,
            "classification_report":  report,
        }

    def print_results(self, metrics: dict) -> None:
        """
        Print a formatted summary of the evaluation metrics.

        Parameters
        ----------
        metrics : dict returned by .evaluate()
        """
        print("=" * 52)
        print("   LC-KSVD Internal Classifier — Evaluation")
        print("=" * 52)
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  F1-Score   : {metrics['f1']:.4f}")
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
        print("-" * 52)
        print("  Per-class breakdown:")
        print(metrics["classification_report"])
        print("=" * 52)