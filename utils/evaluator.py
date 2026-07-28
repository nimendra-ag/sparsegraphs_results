from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    roc_auc_score,
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
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
    # Column widths for the per-model metric table.
    _LABEL_W = 22
    _COL_W = 15

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
        started_at=None,
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
        # Timestamp when this evaluation run began; the run folder is named with
        # both start and completion times so its duration is visible on disk.
        self.started_at = started_at or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Thresholds fit on a validation set and reused on the test set.
        # Maps model_name -> threshold. When a model is present here, its
        # decision threshold is taken from this dict instead of being tuned
        # on the current (test) labels, avoiding test-set leakage.
        self.fixed_thresholds = dict(fixed_thresholds) if fixed_thresholds else {}
        os.makedirs(self.results_dir, exist_ok=True)

        self._model_records = []
        # Fitted estimator objects, kept so the export pipeline can serialise the
        # exact models that produced these metrics. Maps model_name -> estimator.
        self.fitted_models = {}

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

    @staticmethod
    def _scores_for(y_scores, label):
        """Score vector treating `label` as the positive class.

        `y_scores` is P(y = +1) from predict_proba, so the complementary class
        is scored by 1 - y_scores. Needed because average precision is
        asymmetric — it ignores true negatives, so each class needs its own
        ranking to be scored fairly.
        """
        return y_scores if label == 1 else 1.0 - y_scores

    def _chance_baselines(self, minority_label, majority_label):
        """Scores a knowledge-free model attains on this split.

        Threshold metrics use the best trivial constant classifier; ranking
        metrics (ROC-AUC, average precision) use a random ranker. These matter
        because the metrics have wildly different floors under imbalance: at a
        5% positive rate a constant classifier already scores ~0.95 accuracy
        and ~0.49 macro-F1, but exactly 0.0 MCC.
        """
        n_total = len(self.y_test)
        n_minority = int(np.sum(self.y_test == minority_label))
        n_majority = int(np.sum(self.y_test == majority_label))
        pi_minority = n_minority / n_total
        pi_majority = n_majority / n_total

        # A random ranker's precision at any cut equals the class base rate, so
        # its PR curve is flat at that height and average precision shares the
        # same floor. The best trivial F1 for a class comes from predicting it
        # everywhere: precision = prevalence, recall = 1.
        f1_minority = 2 * pi_minority / (1 + pi_minority)
        f1_majority = 2 * pi_majority / (1 + pi_majority)

        return {
            "support": {
                "minority": n_minority,
                "majority": n_majority,
                "total": n_total,
            },
            # Per-class floors, listed under each model's table.
            "precision": (pi_minority, pi_majority),
            "f1": (f1_minority, f1_majority),
            "pr_auc": (pi_minority, pi_majority),
            # Macro-level floors — what the report's Baseline column shows.
            "macro": {
                "precision": (pi_minority + pi_majority) / 2,
                # Any constant or random classifier trades TPR against TNR one
                # for one, so macro-recall is pinned at exactly 0.5.
                "recall": 0.5,
                # All-negative scores f1_majority on one class and 0 on the
                # other; all-positive is the mirror image. Take the better one.
                "f1": max(f1_majority, f1_minority) / 2,
                "accuracy": max(pi_minority, pi_majority),
                "pr_auc": (pi_minority + pi_majority) / 2,
            },
            "roc_auc": 0.5,
            "mcc": 0.0,
        }

    @classmethod
    def _cell(cls, value):
        """Render one table cell: 4dp for floats, bare digits for counts."""
        if value is None:
            return f"{'-':>{cls._COL_W}s}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):>{cls._COL_W}d}"
        return f"{float(value):>{cls._COL_W}.4f}"

    def _format_metrics_block(self, record):
        """Render one model's metrics. Used for both the console output and the
        saved report so the two can never drift apart."""
        w = self._LABEL_W
        minority_label = int(record["minority_label"])
        majority_label = int(record["majority_label"])
        minority_header = f"Minority({minority_label:+d})"
        majority_header = f"Majority({majority_label:+d})"

        lines = [
            f"===== {record['model_name']} (threshold={record['threshold']:.4f}) =====",
            "",
            f"{'':{w}s}"
            + f"{minority_header:>{self._COL_W}s}"
            + f"{majority_header:>{self._COL_W}s}"
            + f"{'Macro':>{self._COL_W}s}"
            + f"{'Baseline':>{self._COL_W}s}",
        ]
        for label, values in record["split"].items():
            lines.append(f"{label:{w}s}" + "".join(self._cell(v) for v in values))
        lines.append(
            f"{'Support':{w}s}"
            + "".join(self._cell(v) for v in record["support"])
            + self._cell(None)
        )

        lines += [
            "",
            "Symmetric (whole test set)",
            f"{'ROC-AUC':{w}s}{self._cell(record['roc_auc'])}"
            f"   (baseline {record['baselines']['roc_auc']:.4f})",
            f"{'MCC':{w}s}{self._cell(record['mcc'])}"
            f"   (baseline {record['baselines']['mcc']:.4f})",
            "",
            "Chance baseline by class",
            f"{'':{w}s}{minority_header:>{self._COL_W}s}{majority_header:>{self._COL_W}s}",
        ]
        for label, pair in record["per_class_baseline"].items():
            lines.append(f"{label:{w}s}" + "".join(self._cell(v) for v in pair))

        return "\n".join(lines)

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

    def _evaluate_model(self, model, model_name, optimize_threshold=True,
                        sample_weight=None):
        # Single fit. `sample_weight` is threaded through rather than applied by
        # the caller, because a caller-side fit would be silently overwritten
        # here — dropping the weights it was trying to apply.
        if sample_weight is not None:
            model.fit(self.X_train, self.y_train, sample_weight=sample_weight)
        else:
            model.fit(self.X_train, self.y_train)
        # Retain the fitted estimator for export (the deployed model is exactly
        # the one benchmarked here).
        self.fitted_models[model_name] = model
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

        majority_label, minority_label = self._majority_minority_labels()
        baselines = self._chance_baselines(minority_label, majority_label)

        cm_global = confusion_matrix(self.y_test, y_pred)
        cm_per_class = multilabel_confusion_matrix(
            self.y_test, y_pred, labels=[majority_label, minority_label]
        )
        cm_majority, cm_minority = cm_per_class[0], cm_per_class[1]

        # Per-class precision / recall / F1, ordered [minority, majority].
        prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
            self.y_test, y_pred,
            labels=[minority_label, majority_label],
            average=None, zero_division=0,
        )

        # One-vs-rest accuracy per class. In a binary problem both classes give
        # the same value, and it equals global accuracy — computed per class
        # rather than asserted so the table shows that rather than claiming it.
        acc_minority = (cm_minority[0, 0] + cm_minority[1, 1]) / cm_minority.sum()
        acc_majority = (cm_majority[0, 0] + cm_majority[1, 1]) / cm_majority.sum()

        # Average precision ignores true negatives, so each class is scored
        # against its own ranking rather than sharing one score vector.
        ap_minority = average_precision_score(
            self.y_test, self._scores_for(y_scores, minority_label),
            pos_label=minority_label,
        )
        ap_majority = average_precision_score(
            self.y_test, self._scores_for(y_scores, majority_label),
            pos_label=majority_label,
        )

        # Threshold-free / symmetric over the whole split. MCC is the one
        # scalar here with a true zero baseline under any class ratio.
        roc_auc = roc_auc_score(self.y_test, y_scores)
        mcc = matthews_corrcoef(self.y_test, y_pred)

        # Scalars kept for the MC-CV aggregator's key names. Precision / recall
        # / F1 are the minority-class figures, which is what they already were
        # on data where the minority class is +1.
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, rec, f1 = float(prec_pc[0]), float(rec_pc[0]), float(f1_pc[0])
        pr_auc = float(ap_minority)
        report_text = classification_report(self.y_test, y_pred, zero_division=0)

        macro = baselines["macro"]
        # Deferred to save_report(), once the run folder (named with the
        # completion timestamp) exists.
        record = {
            "model_name": model_name,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "report_text": report_text,
            # (minority, majority, macro, macro-level chance baseline)
            "split": {
                "Precision": (prec_pc[0], prec_pc[1], prec_pc.mean(), macro["precision"]),
                "Recall": (rec_pc[0], rec_pc[1], rec_pc.mean(), macro["recall"]),
                "F1": (f1_pc[0], f1_pc[1], f1_pc.mean(), macro["f1"]),
                "Accuracy": (acc_minority, acc_majority,
                             (acc_minority + acc_majority) / 2, macro["accuracy"]),
                "PR-AUC (AP)": (ap_minority, ap_majority,
                                (ap_minority + ap_majority) / 2, macro["pr_auc"]),
            },
            "support": (int(sup_pc[0]), int(sup_pc[1]), baselines["support"]["total"]),
            # Recall and Accuracy are omitted: predicting one class everywhere
            # forces that class's recall to 1.0, so per-class recall has no
            # meaningful floor, and accuracy's floor is not class-specific.
            "per_class_baseline": {
                "Precision": baselines["precision"],
                "F1": baselines["f1"],
                "PR-AUC (AP)": baselines["pr_auc"],
            },
            "baselines": baselines,
            "cm_global": cm_global,
            "cm_majority": cm_majority,
            "cm_minority": cm_minority,
            "majority_label": majority_label,
            "minority_label": minority_label,
        }
        self._model_records.append(record)

        print()
        print(self._format_metrics_block(record))
        print()

        # Every scalar is prefixed with the averaging it uses, so a column name
        # in an aggregated CSV can never be mistaken for a different average.
        return {
            # Macro-averaged: both classes count equally regardless of size.
            "Macro-Precision": float(prec_pc.mean()),
            "Macro-Recall": float(rec_pc.mean()),
            "Macro-F1": float(f1_pc.mean()),
            "Macro-PR-AUC": float((ap_minority + ap_majority) / 2),
            # Symmetric over the whole split — these have no per-class variant
            # in a binary problem, so they are reported once.
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "MCC": mcc,
            # Minority class alone — the headline figures under imbalance.
            "Minority-Precision": precision,
            "Minority-Recall": rec,
            "Minority-F1": f1,
            "Minority-PR-AUC": pr_auc,
            "Threshold": threshold,
            "Classification Report": report_text,
            "Confusion Matrix": cm_global,
            "Confusion Matrix (Majority Class)": cm_majority,
            "Confusion Matrix (Minority Class)": cm_minority,
            "Per-Class Metrics": record["split"],
            "Chance Baselines": baselines,
        }

    def get_fitted_models(self):
        """Return {model_name: fitted_estimator} for every model evaluated so
        far. Used by the export pipeline to serialise the deployable models."""
        return dict(self.fitted_models)

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

    def save_report(self, run_folder=None):
        """Save the results txt file plus every model's confusion matrix images
        (global / majority class / minority class).

        By default a new per-execution folder is created under `results_dir`,
        named with the implementation, dataset, atom count and both the start
        and completion timestamps. Pass `run_folder` to write directly into an
        existing directory instead (used when embedding provenance metrics in an
        artifact bundle, to avoid a redundant nested folder / over-long paths).
        """
        completed_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        atoms_part = f"_atoms{self.n_atoms}" if self.n_atoms is not None else ""
        # Folder / filename stem carries both start and end timestamps.
        run_folder_name = (
            f"{self.implementation}_{self.dataset}{atoms_part}"
            f"_{self.started_at}_{completed_at}"
        )
        if run_folder is None:
            run_folder = os.path.join(self.results_dir, run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        header = (
            f"Implementation  : {self.implementation}\n"
            f"Dataset         : {self.dataset}\n"
            f"Dictionary Atoms: {self.n_atoms}\n"
            f"Started At      : {self.started_at}\n"
            f"Completed At    : {completed_at}\n"
            + "=" * 60 + "\n\n"
            + "Reading the tables\n"
            "  Minority / Majority  the metric computed with that class as the\n"
            "                       positive class.\n"
            "  Macro                unweighted mean of the two class columns, so\n"
            "                       both classes count equally regardless of size.\n"
            "  Baseline             what a knowledge-free model scores in the Macro\n"
            "                       column: the best trivial constant classifier for\n"
            "                       Precision / Recall / F1 / Accuracy, and a random\n"
            "                       ranker for PR-AUC, ROC-AUC and MCC. Metrics whose\n"
            "                       floor differs by class are listed again per class\n"
            "                       under each model.\n"
            "  Accuracy             identical across all three columns by construction\n"
            "                       in a binary problem (one-vs-rest accuracy is the\n"
            "                       same for both classes); split for a uniform layout.\n"
            "  Recall               has no per-class floor — predicting a class\n"
            "                       everywhere forces its recall to 1.0 — so only the\n"
            "                       macro floor of 0.5 is meaningful.\n"
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
                self._format_metrics_block(record) + "\n"
                f"\nConfusion Matrix (whole dataset)\n{record['cm_global']}\n"
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

        # The weights are handed to _evaluate_model rather than applied here:
        # it fits the estimator itself, so a fit at this point would be thrown
        # away and the model would end up trained unweighted.
        model = GradientBoostingClassifier(random_state=self.random_state)
        return self._evaluate_model(
            model, "Gradient Boosting", sample_weight=sample_weights
        )

    def predict_svm(self):
        print("Predicting with SVM")
        base_model = LinearSVC(class_weight="balanced", random_state=self.random_state)
        model = CalibratedClassifierCV(base_model)
        return self._evaluate_model(model, "Linear SVM")

    def predict_random_forest(self):
        print("Predicting with Random Forest")
        model = RandomForestClassifier(class_weight="balanced", random_state=self.random_state)
        return self._evaluate_model(model, "Random Forest")
