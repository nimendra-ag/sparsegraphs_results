"""SF (Spectral Features) baseline on the NCI1 dataset.

Straight conversion of test_sf.ipynb into a runnable script for the genv12
(Python 3.12) environment. The theoretical implementation is kept exactly as in
the notebook.

Requirements (genv12): numpy, networkx, rdkit, karateclub, scikit-learn.
Run from the project root so the relative dataset path resolves:
    python implements/sf.py
"""

import numpy as np
import networkx as nx
from rdkit import Chem
from karateclub import SF
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def load_nci1():
    print('Loading NCI1 dataset')

    graphs = []
    y = []

    filepath = "datasets/NCI_full/1total-connect.sdf"

    supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue

        G = nx.Graph()

        # Add atoms as nodes
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                label=atom.GetSymbol()   # WL uses node labels
            )

        # Add bonds as edges
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=str(bond.GetBondType()),
                bond_order=bond.GetBondTypeAsDouble(),
                aromatic=bond.GetIsAromatic(),
                in_ring=bond.IsInRing(),
                conjugated=bond.GetIsConjugated(),
                stereo=str(bond.GetStereo())
            )

        # Get graph label
        # In NCI1, class label is stored as a molecule property
        label = int(float(mol.GetProp("value")))
        graphs.append(G)
        y.append(label)

    print(f"Loaded {len(graphs)} graphs")
    return graphs, y


def run(graphs, y):
    # Set embedding dimensions to the average number of nodes in the dataset
    avg_nodes = int(np.round(np.mean([G.number_of_nodes() for G in graphs])))
    print(f"Average number of nodes (k) set to: {avg_nodes}")

    print("Fitting SF model and extracting embeddings...")
    sf_model = SF(dimensions=avg_nodes, seed=42)
    sf_model.fit(graphs)

    # Extract feature matrix X and target array y
    X = sf_model.get_embedding()
    y = np.array(y)

    # 10-fold cross-validation with class proportions preserved and seed fixed to 1
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    # Metrics tracking
    lr_accuracies, lr_aucs = [], []
    gb_accuracies, gb_aucs = [], []

    print("Evaluating classifiers using 10-fold cross-validation...")

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #  Classifier 1: Logistic Regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)

        # Predict classes and probabilities
        lr_pred = lr.predict(X_test)
        lr_prob = lr.predict_proba(X_test)[:, 1]  # Get probability of the positive class

        lr_accuracies.append(accuracy_score(y_test, lr_pred))
        lr_aucs.append(roc_auc_score(y_test, lr_prob))

        # Classifier 2: Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)

        # Predict classes and probabilities
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:, 1]

        gb_accuracies.append(accuracy_score(y_test, gb_pred))
        gb_aucs.append(roc_auc_score(y_test, gb_prob))

    # Average the results over all testing sets
    print("\nFinal Results (10-Fold CV) ")
    print("Logistic Regression:")
    print(f"Average Accuracy : {np.mean(lr_accuracies) * 100:.2f}%")
    print(f"Average AUC: {np.mean(lr_aucs):.4f}")

    print("\nGradient Boosting:")
    print(f"Average Accuracy : {np.mean(gb_accuracies) * 100:.2f}%")
    print(f"Average AUC: {np.mean(gb_aucs):.4f}")


if __name__ == "__main__":
    graphs, y = load_nci1()
    run(graphs, y)
