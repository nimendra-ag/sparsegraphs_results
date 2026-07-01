from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


class Evaluator:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42, cm_dir="./cm"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.cm_dir = cm_dir
        os.makedirs(self.cm_dir, exist_ok=True)

    def _find_optimal_threshold(self, y_true, y_scores):
        """Find the optimal threshold"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        # Avoid division by zero
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
        # Last element of precisions/recalls has no corresponding threshold
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]

    def _evaluate_model(self, model, model_name, optimize_threshold=True):
        model.fit(self.X_train, self.y_train)
        y_scores = model.predict_proba(self.X_test)[:, 1]

        if optimize_threshold:
            threshold = self._find_optimal_threshold(self.y_test, y_scores)
            y_pred = (y_scores >= threshold).astype(int)
        else:
            threshold = 0.5
            y_pred = model.predict(self.X_test)

        # Metrics
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_scores)
        pr_auc = average_precision_score(self.y_test, y_scores)

        print(f"\n===== {model_name} (threshold={threshold:.3f}) =====")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
        print("\nClassification Report")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class -1", "Class 1"],
            yticklabels=["Class -1", "Class 1"],
        )
        plt.title(
            f"{model_name} (thr={threshold:.3f})\n"
            f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.cm_dir,
                f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return {
            "Precision": precision,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Threshold": threshold,
        }

    def predict_logistic_regression(self):
        print("Predicting with Logistic Regression")
        model = LogisticRegression(class_weight="balanced", random_state=self.random_state)
        return self._evaluate_model(model, "Logistic Regression")

    def predict_gradient_boosting(self):
        # GradientBoostingClassifier doesn't support class_weight;
        # use scale_pos_weight via sample_weight instead
        print("Predicting with Gradient Boosting")
        n_neg = np.sum(self.y_train == -1)
        n_pos = np.sum(self.y_train == 1)
        sample_weights = np.where(
            self.y_train == 1, n_neg / n_pos, 1.0
        )
        model = GradientBoostingClassifier(random_state=self.random_state)
        # Override fit to pass sample_weight
        model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        # Evaluate without re-fitting
        y_scores = model.predict_proba(self.X_test)[:, 1]
        threshold = self._find_optimal_threshold(self.y_test, y_scores)
        y_pred = (y_scores >= threshold).astype(int)

        precision = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_scores)
        pr_auc = average_precision_score(self.y_test, y_scores)

        print(f"\n===== Gradient Boosting (threshold={threshold:.3f}) =====")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
        print("\nClassification Report")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class -1", "Class 1"],
            yticklabels=["Class -1", "Class 1"],
        )
        plt.title(
            f"Gradient Boosting (thr={threshold:.3f})\n"
            f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cm_dir, "confusion_matrix_gradient_boosting.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        return {
            "Precision": precision,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Threshold": threshold,
        }

    def predict_svm(self):
        print("Predicting with SVM")
        base_model = LinearSVC(class_weight="balanced", random_state=self.random_state)
        model = CalibratedClassifierCV(base_model)
        return self._evaluate_model(model, "Linear SVM")

    def predict_random_forest(self):
        print("Predicting with Random Forest")
        model = RandomForestClassifier(class_weight="balanced", random_state=self.random_state)
        return self._evaluate_model(model, "Random Forest")