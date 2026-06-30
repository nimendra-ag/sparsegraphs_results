from dict_learners.aksvd import AKSVD
from graph_encoders.wl import WL
from utils.graph_data import GraphDataLoader
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
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


#-------------------------Without overfit protection-------------------------#
# # load graph data
# graphs, y = load_data(name="nci", size=2)
#
# wl = WL(graphs = graphs)
# wl.fit()
# graph_embeddings = wl.get_embeddings()
# aksvd = AKSVD(graph_embeddings=graph_embeddings)
# X = aksvd.fit()
#
# evaluator = Evaluator(X, y, test_size=0.2)

# results_logistic_reg = evaluator.predict_logistic_regression()
# print(results_logistic_reg)
# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print(results_gradient_boosting)

#-------------------------With overfit protection-------------------------#
# # load graph data
# graphs, y = data_loader.nci_full_graphs, data_loader.nci_full_labels
#
# # First divide the data into train and test sets.
# G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)
#
# # divide the train set further into vocab training and ML training sets
# G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)
#
# wl = WL()
# graph_embeddings = wl.generate_training_embeddings(G_vocab_train)
#
# aksvd = AKSVD().fit(training_graph_embeddings=graph_embeddings)
#
# graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
# X_ML_train = aksvd.infer(graph_embeddings_ml_train)
#
# graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
# X_ML_test = aksvd.infer(graph_embeddings_ml_test)
#
# scaler = MaxAbsScaler()
# X_ML_train_scaled = scaler.fit_transform(X_ML_train)
# X_ML_test_scaled = scaler.transform(X_ML_test)
#
# # Model evaluation
# evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)
# results_logistic_reg = evaluator.predict_logistic_regression()
# print(results_logistic_reg)
#
# results_gradient_boosting = evaluator.predict_gradient_boosting()
# print(results_gradient_boosting)

# First divide the data into train and test sets.
G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

class WL_AKSVD:
    def __init__(self, data_loader):
        self.implementation = "WL_AKSVD"
        self.data_loader = data_loader

    def run(self, G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test):

        start = datetime.now().strftime("%Y%m%d_%H%M%S")

        wl = WL()
        graph_embeddings = wl.generate_training_embeddings(G_vocab_train, y_vocab_train)

        aksvd = AKSVD().fit(training_graph_embeddings=graph_embeddings)

        #generating sparse vectors for graphs for training the ml models
        graph_embeddings_ml_train = wl.generate_inferencing_embeddings(G_ML_train)
        X_ML_train = aksvd.infer(graph_embeddings_ml_train)

        #generating sparse vectors for graphs for classification(inferencing the ml model)
        graph_embeddings_ml_test = wl.generate_inferencing_embeddings(G_test)
        X_ML_test = aksvd.infer(graph_embeddings_ml_test)

        scaler = MaxAbsScaler()
        X_ML_train_scaled = scaler.fit_transform(X_ML_train)
        X_ML_test_scaled = scaler.transform(X_ML_test)

        # Model evaluation
        evaluator = Evaluator(X_ML_train_scaled, y_ML_train, X_ML_test_scaled, y_test)
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

        filename = f"results_{start}_{end}.txt"

        with open(f"results/{filename}", "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Saved results to {filename}")
        
wl_ksvd = WL_AKSVD(data_loader)
wl_ksvd.run(G_vocab_train, y_vocab_train, G_ML_train, G_test, y_ML_train, y_test)