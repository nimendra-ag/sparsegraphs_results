"""Inference-time entry point for the application.

Loads a saved artifact bundle and turns a user-supplied molecule (SMILES or an
RDKit Mol) into a classification, running the exact frozen training pipeline:

    molecule -> graph -> WL embedding (saved vocab) -> sparse code (saved D)
             -> scale (saved scaler) -> classifier proba -> saved threshold -> label

Graph construction mirrors utils/graph_data.py so inference-time graphs are
byte-for-byte compatible with how the encoder was trained (node feature = atom
symbol; bonds as edges).
"""

import numpy as np

from utils.artifact_store import load_bundle


def molecule_to_graph(mol):
    """RDKit Mol -> networkx graph, matching GraphDataLoader's construction."""
    import networkx as nx

    if mol is None:
        raise ValueError("Received a None molecule (RDKit failed to parse the input).")

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), feature=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            bond_order=bond.GetBondTypeAsDouble(),
            aromatic=bond.GetIsAromatic(),
            in_ring=bond.IsInRing(),
            conjugated=bond.GetIsConjugated(),
            stereo=str(bond.GetStereo()),
        )
    return G


def smiles_to_graph(smiles):
    """SMILES string -> networkx graph (adds explicit Hs to match SDF loading)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    return molecule_to_graph(mol)


class InferencePipeline:
    """Wraps a loaded bundle and classifies graphs / molecules."""

    def __init__(self, bundle):
        self.encoder = bundle["encoder"]
        self.dict_learner = bundle["dict_learner"]
        self.scaler = bundle["scaler"]
        self.models = bundle["models"]
        self.thresholds = bundle["thresholds"]
        self.manifest = bundle["manifest"]
        self.default_model = self.manifest.get("default_model")

    @classmethod
    def from_dir(cls, bundle_dir):
        return cls(load_bundle(bundle_dir))

    def sparse_codes(self, graphs):
        """graphs -> scaled sparse codes (the classifier's input space)."""
        embeddings = self.encoder.generate_inferencing_embeddings(graphs)
        codes = self.dict_learner.infer(embeddings)
        return self.scaler.transform(codes)

    def predict(self, graphs, model_name=None, return_proba=True):
        """Classify a list of networkx graphs.

        Returns (preds, proba) where proba is the positive-class probability and
        preds are in {-1, 1} using the saved (validation-tuned) threshold.
        """
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise KeyError(
                f"Model '{model_name}' not in bundle. Available: {sorted(self.models)}"
            )

        X_scaled = self.sparse_codes(graphs)
        model = self.models[model_name]
        proba = model.predict_proba(X_scaled)[:, 1]

        threshold = self.thresholds.get(model_name, 0.5)
        preds = np.where(proba >= threshold, 1, -1)
        return (preds, proba) if return_proba else preds

    def predict_smiles(self, smiles, model_name=None):
        """Convenience: classify a single SMILES string.

        Returns {'label': int, 'probability': float, 'threshold': float,
                 'model': str}.
        """
        model_name = model_name or self.default_model
        graph = smiles_to_graph(smiles)
        preds, proba = self.predict([graph], model_name=model_name)
        return {
            "label": int(preds[0]),
            "probability": float(proba[0]),
            "threshold": float(self.thresholds.get(model_name, 0.5)),
            "model": model_name,
        }
