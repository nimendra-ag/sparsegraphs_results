from dict_learners.lcksvd import LCKSVDLearner
from graph_encoders.wl import WL
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime
from utils.lcksvd_evaluator import LCKSVDEvaluator

data_loader = GraphDataLoader()

graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels

G_train, G_test, y_train, y_test = train_test_split(
    graphs, y,
    test_size=0.2,
    random_state=42,
)

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
    G_train, y_train,
    test_size=0.75,
    random_state=42,
)

class WL_LCKSVD:
    def __init__(self, data_loader: GraphDataLoader):
        self.implementation = "WL_LCKSVD"
        self.data_loader = data_loader

    def run( self, G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test):
        
        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- Step 1: WL graph encoding -----------------------------------
        # Build vocabulary from G_vocab_train (with labels so WL can apply
        # its imbalance-aware feature selection).  This is identical to the
        # WL_AKSVD pipeline.
        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)
        # graph_embeddings : (N_vocab, n_vocab_features)

        # ---- Step 2: LC-KSVD dictionary learning -------------------------
        # fit() requires labels because LC-KSVD is supervised.
        # y_vocab_train provides the class labels for the vocab-training graphs.
        lcksvd = LCKSVDLearner().fit(training_graph_embeddings=graph_embeddings, labels=y_vocab_train)

        # ---- Step 3: Generate sparse codes for ML training set -----------
        # Sparse codes are produced by OMP on the learned dictionary D_hat.
        # Labels are NOT passed here — infer() is identical to AKSVD.infer().
        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = lcksvd.infer(graph_embeddings_ml_train)
        # X_ML_train : (N_ml_train, K)

        # ---- Step 4: Generate sparse codes for test set ------------------
        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = lcksvd.infer(graph_embeddings_ml_test)
        # X_ML_test : (N_test, K)

        # ---- Step 5: Scale -------------------------------------------------
        # MaxAbsScaler fitted only on training sparse codes — no test leakage.
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # ---- Step 6: Evaluate downstream ML models -----------------------
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)

        results_logistic_reg = evaluator.predict_logistic_regression()
        print(results_logistic_reg)

        results_gradient_boosting = evaluator.predict_gradient_boosting()
        print(results_gradient_boosting)

        results_svm = evaluator.predict_svm()
        print(results_svm)

        results_random_forest = evaluator.predict_random_forest()
        print(results_random_forest)
        # ---- Step 7: LC-KSVD internal classifier evaluation -------------
        # Evaluate W_hat — the linear classifier learned jointly with D and A.
        # Passes raw WL test embeddings (not scaled sparse codes) because
        # LCKSVDEvaluator handles its own OMP encoding internally via
        # lcksvd_model.encode(), working directly in the learned D_hat space.
        lcksvd_eval = LCKSVDEvaluator(lcksvd)
        lcksvd_metrics = lcksvd_eval.evaluate(
            graph_embeddings_test=graph_embeddings_ml_test,
            y_test=y_test,
            positive_label=1,     # minority class index for the NCI dataset
        )
        lcksvd_eval.print_results(lcksvd_metrics)

        # ---- Step 8: Save results ----------------------------------------
        final_output = f"""
            {results_logistic_reg}
            {results_gradient_boosting}
            {results_svm}
            {results_random_forest}

            --- LC-KSVD Internal Classifier ---
                    Precision : {lcksvd_metrics['precision']:.4f}
                    Recall    : {lcksvd_metrics['recall']:.4f}
                    F1-Score  : {lcksvd_metrics['f1']:.4f}
                    ROC-AUC   : {lcksvd_metrics['roc_auc']:.4f}
                    PR-AUC    : {lcksvd_metrics['pr_auc']:.4f}
            {lcksvd_metrics['classification_report']}
            """

        end = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_wllcksvd_{start}_{end}.txt"

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Saved results to {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

wl_lcksvd = WL_LCKSVD(data_loader)
wl_lcksvd.run(G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test)
