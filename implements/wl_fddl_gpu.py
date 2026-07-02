import sys
import os
# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_data import GraphDataLoader

from dict_learners.fddl_gpu import FDDLGPU
from graph_encoders.wl import WL
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np


data_loader = GraphDataLoader()

graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train_full, G_test, y_train_full, y_test = train_test_split(
    graphs, y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

G_train, G_val, y_train, y_val = train_test_split(
    G_train_full, y_train_full,
    test_size=0.15 / 0.85,  # 15% of the full dataset
    random_state=42,
    stratify=y_train_full,
)

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train,
    test_size=2 / 7,  # vocab_train : ML_train = 5 : 2 -> 50% / 20% of the full dataset
    random_state=42,
    stratify=y_train,
)

class WL_FDDLGPU:
    def __init__(self):
        self.implementation = "WL_FDDLGPU"

    def run(self, G_vocab_train, y_vocab_train, G_ML_train, y_ML_train, G_val, y_val):

        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)

        fddl_gpu = FDDLGPU()
        scaler = MaxAbsScaler()

        fddl_gpu.fit(training_graph_embeddings=graph_embeddings, y_train=y_vocab_train)
        total_atoms = fddl_gpu.D.shape[1]

        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = fddl_gpu.infer(graph_embeddings_ml_train)
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)

        # ---------------------------------------------------------------------------
        # --- Evaluate on the held-out validation split -----------------------------
        # ---------------------------------------------------------------------------

        graph_embeddings_ml_val = wl.generate_inferencing_embeddings(G_val)
        X_ML_val = fddl_gpu.infer(graph_embeddings_ml_val)
        X_ML_val_scaled = scaler.transform(X_ML_val)

        evaluator_val = Evaluator(
            X_ML_train_scaled, y_ML_train, X_ML_val_scaled, y_val,
            implementation="wl_fddl_gpu",
            dataset="nci_full",
            n_atoms=total_atoms,
        )

        # --- Native classifiers
        # --- (no external ML needed) ---
        from utils.src_classifier import SRCClassifier

        # Pure SRC (works with any structured DL: FDDL, DPL, DLSI...)
        src = SRCClassifier(fddl_gpu, gamma=0.0)
        print("Pure SRC:", src.evaluate(graph_embeddings_ml_val, y_val))

        # FDDL-native (SRC + coefficient distance to class means)
        src_fddl = SRCClassifier(fddl_gpu, gamma=0.5)
        print("FDDL-native:", src_fddl.evaluate(graph_embeddings_ml_val, y_val))

        results_logistic_reg = evaluator_val.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator_val.predict_gradient_boosting()
        print(results_gradient_boosting)

        results_svm = evaluator_val.predict_svm()
        print(results_svm)

        results_random_forest = evaluator_val.predict_random_forest()
        print(results_random_forest)

        evaluator_val.save_report()

        # Thresholds tuned on the validation split, to be reused on the test
        # split so the final test metrics are not tuned on the test labels.
        val_thresholds = evaluator_val.get_thresholds()

        # ---------------------------------------------------------------------------
        # --- Evaluate on the held-out test split (use only for the final report) ---
        # ---------------------------------------------------------------------------
        # graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        # X_ML_test = fddl_gpu.infer(graph_embeddings_ml_test)
        # X_ML_test_scaled = scaler.transform(X_ML_test)

        # evaluator_test = Evaluator(
        #     X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test,
        #     implementation="wl_fddl_gpu",
        #     dataset="nci_full",
        #     n_atoms=total_atoms,
        #     fixed_thresholds=val_thresholds,  # reuse validation-tuned thresholds
        # )

        # # --- Native classifiers
        # # --- (no external ML needed) ---
        # from utils.src_classifier import SRCClassifier

        # # Pure SRC (works with any structured DL: FDDL, DPL, DLSI...)
        # src = SRCClassifier(fddl_gpu, gamma=0.0)
        # print("Pure SRC:", src.evaluate(graph_embeddings_ml_test, y_test))

        # # FDDL-native (SRC + coefficient distance to class means)
        # src_fddl = SRCClassifier(fddl_gpu, gamma=0.5)
        # print("FDDL-native:", src_fddl.evaluate(graph_embeddings_ml_test, y_test))

        # results_logistic_reg = evaluator_test.predict_logistic_regression()
        # print(results_logistic_reg)

        # results_gradient_boosting = evaluator_test.predict_gradient_boosting()
        # print(results_gradient_boosting)

        # results_svm = evaluator_test.predict_svm()
        # print(results_svm)

        # results_random_forest = evaluator_test.predict_random_forest()
        # print(results_random_forest)

        # evaluator_test.save_report()


data_loader = GraphDataLoader()
wl_fddl_gpu = WL_FDDLGPU()

# --- Evaluate on the validation split (use during model/hyperparameter tuning) ---
wl_fddl_gpu.run(G_vocab_train, y_vocab_train, G_ML_train, y_ML_train, G_val, y_val)

# --- Evaluate on the held-out test split (use only for the final report) ---
# wl_fddl_gpu.run(G_vocab_train, y_vocab_train, G_ML_train, y_ML_train, G_test, y_test)