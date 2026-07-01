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
from datetime import datetime


data_loader = GraphDataLoader()

graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train, G_test, y_train, y_test = train_test_split(
    graphs, y,
    test_size=0.2,
    random_state=42
)

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train,
    test_size=0.75,
    random_state=42
)


# Configuration
# DATASET = "nci-full"
# MODEL_NAME = "fddl"

# data_loader = GraphDataLoader()

# # Load graph data
# graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels
# y = np.array(y)

# # First divide the data into train and test sets.
# G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

# # Divide the train set further into vocab training and ML training sets
# G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)

# # 1. Feature Extraction (WL)
# wl = WL()
# graph_embeddings = wl.generate_training_embeddings(G_vocab_train)
# graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
# graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)

# # 2. Dictionary Learning (FDDL)
# fddl = FDDL(k=10, max_iter=20)
# # Note: FDDL is supervised, so it needs labels
# fddl.fit(training_graph_embeddings=graph_embeddings_ml_train, y_train=y_ML_train)

# # 3. Infer Sparse Coefficients
# X_ML_train = fddl.infer(graph_embeddings_ml_train)
# X_ML_test = fddl.infer(graph_embeddings_ml_test)

# # Scale
# scaler = MaxAbsScaler()
# X_ML_train_scaled = scaler.fit_transform(X_ML_train)
# X_ML_test_scaled = scaler.transform(X_ML_test)

# # 4. Model Evaluation
# evaluator = Evaluator(
#     X_train=X_ML_train_scaled, 
#     y_train=y_ML_train, 
#     X_test=X_ML_test_scaled, 
#     y_test=y_test, 
#     dl_model=MODEL_NAME, 
#     dataset=DATASET
# )

# results_logistic_reg = evaluator.predict_logistic_regression()
# print("Logistic Regression:", results_logistic_reg)

# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print("Gradient Boosting:", results_gradient_boosting)


class WL_FDDLGPU:
    def __init__(self):
        self.implementation = "WL_FDDLGPU"

    def run(self, G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test):

        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)

        fddl_gpu = FDDLGPU()
        fddl_gpu.fit(training_graph_embeddings=graph_embeddings, y_train=y_vocab_train)

        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = fddl_gpu.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = fddl_gpu.infer(graph_embeddings_ml_test)

        # --- Native classifiers (no external ML needed) ---
        from utils.src_classifier import SRCClassifier

        # Pure SRC (works with any structured DL: FDDL, DPL, DLSI...)
        src = SRCClassifier(fddl_gpu, gamma=0.0)
        print("Pure SRC:", src.evaluate(graph_embeddings_ml_test, y_test))

        # FDDL-native (SRC + coefficient distance to class means)
        src_fddl = SRCClassifier(fddl_gpu, gamma=0.5)
        print("FDDL-native:", src_fddl.evaluate(graph_embeddings_ml_test, y_test))

        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # Model evaluation
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test, dl_model="wl_fddl_gpu", dataset="nci-full")
        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)

        results_svm = evaluator.predict_svm()
        print(results_svm)

        results_random_forest = evaluator.predict_random_forest()
        print(results_random_forest)

        final_output = f"""
            {results_logistic_reg}
            {results_gradient_boosting}
            {results_svm}
            {results_random_forest}
            """

        end = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"results_wl_fddl_{fddl_gpu.k}_{start}_{end}.txt"

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Saved results to {filename}")
        
data_loader = GraphDataLoader()
wl_fddl_gpu = WL_FDDLGPU()
wl_fddl_gpu.run(G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test)