from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    roc_auc_score,
    accuracy_score,
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
from datetime import datetime


class Evaluator:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        implementation="unknown_impl",
        dataset="unknown_dataset",
        n_atoms=None,
        random_state=42,
        results_dir="results",
        fixed_thresholds=None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.implementation = implementation
        self.dataset = dataset
        self.n_atoms = n_atoms
        self.results_dir = results_dir
        # Thresholds fit on a validation set and reused on the test set.
        # Maps model_name -> threshold. When a model is present here, its
        # decision threshold is taken from this dict instead of being tuned
        # on the current (test) labels, avoiding test-set leakage.
        self.fixed_thresholds = dict(fixed_thresholds) if fixed_thresholds else {}
        os.makedirs(self.results_dir, exist_ok=True)

        self._model_records = []

    def _find_optimal_threshold(self, y_true, y_scores):
        """Find the threshold that maximizes F1-score."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        with np.errstate(invalid="ignore"):
            f1_scores = np.where(
                (precisions + recalls) > 0,
                2 * precisions * recalls / (precisions + recalls),
                0.0,
            )
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]

    def _majority_minority_labels(self):
        labels, counts = np.unique(self.y_test, return_counts=True)
        majority_label = labels[np.argmax(counts)]
        minority_label = labels[np.argmin(counts)]
        return majority_label, minority_label

    def _plot_confusion_matrix(self, cm, title, tick_labels, filepath):
        """Render a single confusion matrix as its own matplotlib figure."""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _evaluate_model(self, model, model_name, optimize_threshold=True):
        model.fit(self.X_train, self.y_train)
        y_scores = model.predict_proba(self.X_test)[:, 1]

        if model_name in self.fixed_thresholds:
            # Reuse a threshold fit on the validation set — do not tune on the
            # current (test) labels.
            threshold = self.fixed_thresholds[model_name]
            y_pred = np.where(y_scores >= threshold, 1, -1)
        elif optimize_threshold:
            threshold = self._find_optimal_threshold(self.y_test, y_scores)
            y_pred = np.where(y_scores >= threshold, 1, -1)  # <-- FIX: map to -1/1, not 0/1
        else:
            threshold = 0.5
            y_pred = model.predict(self.X_test)

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_scores)
        pr_auc = average_precision_score(self.y_test, y_scores)
        report_text = classification_report(self.y_test, y_pred, zero_division=0)

        print(f"\n===== {model_name} (threshold={threshold:.3f}) =====")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
        print("\nClassification Report")
        print(report_text)

        majority_label, minority_label = self._majority_minority_labels()
        cm_global = confusion_matrix(self.y_test, y_pred)
        cm_per_class = multilabel_confusion_matrix(
            self.y_test, y_pred, labels=[majority_label, minority_label]
        )
        cm_majority, cm_minority = cm_per_class[0], cm_per_class[1]

        # Deferred to save_report(), once the run folder (named with the
        # completion timestamp) exists.
        self._model_records.append({
            "model_name": model_name,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "report_text": report_text,
            "cm_global": cm_global,
            "cm_majority": cm_majority,
            "cm_minority": cm_minority,
            "majority_label": majority_label,
            "minority_label": minority_label,
        })

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Threshold": threshold,
            "Classification Report": report_text,
            "Confusion Matrix": cm_global,
            "Confusion Matrix (Majority Class)": cm_majority,
            "Confusion Matrix (Minority Class)": cm_minority,
        }

    def get_thresholds(self):
        """Return the decision threshold used for each evaluated model as a
        {model_name: threshold} dict. Call this after running the evaluations
        on the validation set, then pass the result as `fixed_thresholds` to a
        new Evaluator built on the test set so the test metrics use the
        validation-tuned thresholds instead of tuning on the test labels."""
        return {
            record["model_name"]: record["threshold"]
            for record in self._model_records
        }

    def save_report(self):
        """Create one folder per execution (named after the implementation,
        dataset, dictionary atom count and the timestamp the execution
        completed), and save the results txt file plus every model's
        confusion matrix images (global / majority class / minority class)
        into it."""
        completed_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        atoms_part = f"_atoms{self.n_atoms}" if self.n_atoms is not None else ""
        run_folder_name = f"{self.implementation}_{self.dataset}{atoms_part}_{completed_at}"
        run_folder = os.path.join(self.results_dir, run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        header = (
            f"Implementation  : {self.implementation}\n"
            f"Dataset         : {self.dataset}\n"
            f"Dictionary Atoms: {self.n_atoms}\n"
            f"Completed At    : {completed_at}\n"
            + "=" * 60 + "\n\n"
        )
        report_blocks = []

        for record in self._model_records:
            model_slug = record["model_name"].lower().replace(" ", "_")
            majority_label = record["majority_label"]
            minority_label = record["minority_label"]

            cm_global_path = os.path.join(run_folder, f"cm_global_{model_slug}.png")
            cm_majority_path = os.path.join(run_folder, f"cm_majority_class_{model_slug}.png")
            cm_minority_path = os.path.join(run_folder, f"cm_minority_class_{model_slug}.png")

            self._plot_confusion_matrix(
                record["cm_global"],
                f"{record['model_name']} - Whole Dataset",
                ["Class -1", "Class 1"],
                cm_global_path,
            )
            self._plot_confusion_matrix(
                record["cm_majority"],
                f"{record['model_name']} - Majority Class ({majority_label})",
                ["Rest", f"Class {majority_label}"],
                cm_majority_path,
            )
            self._plot_confusion_matrix(
                record["cm_minority"],
                f"{record['model_name']} - Minority Class ({minority_label})",
                ["Rest", f"Class {minority_label}"],
                cm_minority_path,
            )

            report_blocks.append(
                f"===== {record['model_name']} (threshold={record['threshold']:.3f}) =====\n"
                f"Accuracy  : {record['accuracy']:.4f}\n"
                f"Precision : {record['precision']:.4f}\n"
                f"Recall    : {record['recall']:.4f}\n"
                f"F1-Score  : {record['f1']:.4f}\n"
                f"ROC-AUC   : {record['roc_auc']:.4f}\n"
                f"PR-AUC    : {record['pr_auc']:.4f}\n"
                f"\nClassification Report\n{record['report_text']}\n"
                f"Confusion Matrix (whole dataset)\n{record['cm_global']}\n"
                f"\nConfusion Matrix (majority class = {majority_label} vs rest)\n{record['cm_majority']}\n"
                f"\nConfusion Matrix (minority class = {minority_label} vs rest)\n{record['cm_minority']}\n"
            )

        content = header + "\n".join(report_blocks)
        results_filepath = os.path.join(run_folder, f"results_{run_folder_name}.txt")
        with open(results_filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved results to {run_folder}")
        return run_folder

    def predict_logistic_regression(self):
        print("Predicting with Logistic Regression")
        model = LogisticRegression(class_weight="balanced", random_state=self.random_state)
        return self._evaluate_model(model, "Logistic Regression")

    def predict_gradient_boosting(self):
        print("Predicting with Gradient Boosting")
        classes, counts = np.unique(self.y_train, return_counts=True)
        # Mirrors sklearn's "balanced" formula: n_samples / (n_classes * count_per_class)
        weight_map = dict(zip(classes, len(self.y_train) / (len(classes) * counts)))
        sample_weights = np.array([weight_map[y] for y in self.y_train])

        model = GradientBoostingClassifier(random_state=self.random_state)
        model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        return self._evaluate_model(model, "Gradient Boosting")

    def predict_svm(self):
        print("Predicting with SVM")
        base_model = LinearSVC(class_weight="balanced", random_state=self.random_state)
        model = CalibratedClassifierCV(base_model)
        return self._evaluate_model(model, "Linear SVM")

    def predict_random_forest(self):
        print("Predicting with Random Forest")
        model = RandomForestClassifier(class_weight="balanced", random_state=self.random_state)
        return self._evaluate_model(model, "Random Forest")
