from dict_learners.aksvd import AKSVD
from graph_encoders.wl import WL
from interpreter.wl_aksvd_interpreter import WLAKSVDInterpreter
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime


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


class WL_AKSVD:
    def __init__(self, data_loader):
        self.implementation = "WL_AKSVD"
        self.data_loader = data_loader

    def run(
        self,
        G_vocab_train, y_vocab_train,
        G_ML_train,   G_test,
        y_ML_train,   y_test,
        n_explain: int = 3,
    ):
        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 1. WL feature extraction ──────────────────────────────────────────
        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)

        # ── 2. Dictionary learning ────────────────────────────────────────────
        aksvd = AKSVD().fit(training_graph_embeddings=graph_embeddings)

        # ── 3. Sparse-code generation ─────────────────────────────────────────
        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = aksvd.infer(graph_embeddings_ml_train)

        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = aksvd.infer(graph_embeddings_ml_test)

        # ── 4. Scaling ────────────────────────────────────────────────────────
        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled  = scaler.transform(X_ML_test)

        # ── 5. Classification (existing Evaluator – unchanged) ────────────────
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)

        results_logistic_reg      = evaluator.predict_logistic_regression()
        results_gradient_boosting = evaluator.predict_gradient_boosting()
        results_svm               = evaluator.predict_svm()
        results_random_forest     = evaluator.predict_random_forest()

        for result in (results_logistic_reg, results_gradient_boosting,
                       results_svm, results_random_forest):
            print(result)

        # ── 6. Dedicated interpretable classifier ─────────────────────────────
        # We train a standalone Logistic Regression so the interpreter has clean
        # coef_ access.  This is separate from the Evaluator's internal models
        # and does not affect reported metrics.
        interp_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        interp_clf.fit(X_ML_train_scaled, y_ML_train)

        # ── 7. Build interpreter ──────────────────────────────────────────────
        interpreter = WLAKSVDInterpreter(
            wl=wl,
            aksvd=aksvd,
            classifier=interp_clf,
            scaler=scaler,
            training_graphs=G_ML_train,
            training_labels=list(y_ML_train),
            training_sparse_codes=X_ML_train,  # unscaled – used for similarity
            label_map={0: "Non-Cancerous", 1: "Cancerous"},
        )

        # ── 8. Generate interpretability reports ──────────────────────────────
        interpretation_blocks = []
        for i in range(min(n_explain, len(G_test))):
            header = (
                f"\n{'#' * 70}\n"
                f"# INTERPRETABILITY REPORT – TEST GRAPH {i}"
                f"  (true label: {y_test[i]})\n"
                f"{'#' * 70}"
            )
            report = interpreter.full_report(G_test[i])
            block  = header + "\n" + report
            interpretation_blocks.append(block)
            print(block)

        # ── 9. Persist results ────────────────────────────────────────────────
        end      = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_wlaksvd_{start}_{end}.txt"

        final_output = "\n".join([
            str(results_logistic_reg),
            str(results_gradient_boosting),
            str(results_svm),
            str(results_random_forest),
            *interpretation_blocks,
        ])

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"\nSaved results to results/{filename}")

        return interpreter   # return so callers can run ad-hoc queries


# ── Entry point ────────────────────────────────────────────────────────────────
wl_aksvd = WL_AKSVD(data_loader)
interpreter = wl_aksvd.run(
    G_vocab_train, y_vocab_train,
    G_ML_train,    G_test,
    y_ML_train,    y_test,
)

# ── Ad-hoc example: explain a specific test graph ──────────────────────────────
# interpreter.explain_prediction(G_test[42])
# interpreter.find_similar_compounds(G_test[42], top_k=10)
# interpreter.explain_dictionary_atom(atom_idx=7, top_k_features=10)
# interpreter.explain_wl_feature(feature_id=435)