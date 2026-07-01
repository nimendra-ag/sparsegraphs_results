import os
import networkx as nx
from rdkit import Chem


class GraphDataLoader:

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.nci_full_graphs, self.nci_full_labels = self.load_nci_full()
            # self.reddit10k_graphs, self.reddit10k_labels = self.load_reddit10k()
            self._initialized = True

    def load_nci_full(self, id=1):
        """
        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)
        """
        print('Loading NCI dataset')
        DATASET_DIR = "datasets/NCI_balanced"  # change this
        graphs = []
        y = []

        filename = f"{id}-balance.sdf"
        filepath = os.path.join(DATASET_DIR, filename)

        supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            G = nx.Graph()

            # Add atoms as nodes
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    feature=atom.GetSymbol()   # WL uses node labels
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
            # In NCI, class label is stored as a molecule property
            label = int(float(mol.GetProp("value")))
            graphs.append(G)
            y.append(label)

        print(f"Loaded {len(graphs)} graphs")
        return graphs, y

    def load_reddit10k(self):
        print('Loading reddit10k dataset')
        reader = GraphSetReader("reddit10k")

        graphs = reader.get_graphs()
        y = reader.get_target()
        print(f"Loaded {len(graphs)} graphs")
        return graphs, y