from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class Evaluator:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=0):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state

    def _evaluate_model(self, model, model_name):

        model.fit(self.X_train, self.y_train)
        y_hat = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)

        # Metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_hat)

        # PR-AUC
        pr_auc = average_precision_score(self.y_test, y_hat)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Print Metrics
        print(f"\n===== {model_name} =====")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")

        print("\nClassification Report")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix Plot

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class -1', 'Class 1'], yticklabels=['Class -1', 'Class 1'])
        plt.title(f'{model_name}\n' f'F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}')

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(
            f'./cm/confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches='tight'
        )

        # plt.show()
        return {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        }

    def predict_logistic_regression(self):

        print("Predicting with Logistic Regression")
        model = LogisticRegression(random_state=self.random_state)
        return self._evaluate_model(model, "Logistic Regression")

    def predict_gradient_boosting(self):

        print("Predicting with Gradient Boosting")
        model = GradientBoostingClassifier(random_state=self.random_state)
        return self._evaluate_model(model, "Gradient Boosting")

    def predict_svm(self):

        print("Predicting with SVM")
        base_model = LinearSVC(random_state=self.random_state)
        model = CalibratedClassifierCV(base_model)
        return self._evaluate_model(model, "Linear SVM")

    def predict_random_forest(self):

        print("Predicting with Random Forest")
        model = RandomForestClassifier(random_state=self.random_state)
        return self._evaluate_model(model, "Random Forest")