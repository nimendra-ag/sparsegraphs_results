import os
import gzip
import zipfile
import urllib.request

import networkx as nx
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


DATASETS_DIR = "datasets"


NCI_IDS = (1, 33, 41, 47, 81, 83, 109, 123, 145)
DEFAULT_NCI_ID = 33

# NCI ships in two variants covering the same nine screens, each with its own
# directory and file-naming convention:
#   * "nci_full"     - every molecule in the screen, heavily imbalanced
#                      (screen 33: 35555 inactive vs 1467 active, ~1:24)
#   * "nci_balanced" - a 1:1 subsample of the same screen (screen 33: 1467/1467
#                      on disk; ~1447/1396 after RDKit drops unsanitizable
#                      molecules, so treat it as near-balanced, not exactly 1:1)
# The two are separate datasets, not two views of one: a run on "nci_balanced"
# is not comparable with a run on "nci_full", so the variant name travels in
# every folder name and manifest exactly like the screen id does.
# variant name -> (directory under datasets/, file-name template)
_NCI_VARIANTS = {
    "nci_full": ("NCI_full", "{id}total-connect.sdf"),
    "nci_balanced": ("NCI_balanced", "{id}-balance.sdf"),
}
# Datasets whose identity includes a numeric id. Everything else ignores it.
_ID_DATASETS = {name: DEFAULT_NCI_ID for name in _NCI_VARIANTS}


def resolve_dataset_id(dataset, dataset_id=None):
    """Return the id that `dataset` will actually be loaded with, or None.

    Non-NCI datasets have no id, so they always resolve to None; NCI resolves a
    missing id to the loader default, so a run that never mentioned an id still
    records the screen it really used.
    """
    default = _ID_DATASETS.get(str(dataset).lower())
    if default is None:
        return None
    return int(dataset_id) if dataset_id is not None else default


def dataset_tag(dataset, dataset_id=None):
    """Self-describing name for folders/reports, e.g. "nci_full_id33".

    Datasets without an id are returned unchanged ("mutag"), so callers can use
    this unconditionally.
    """
    resolved = resolve_dataset_id(dataset, dataset_id)
    return dataset if resolved is None else f"{dataset}_id{resolved}"


def dataset_load_kwargs(dataset, dataset_id=None):
    """kwargs for `GraphDataLoader.load` that pin `dataset` to `dataset_id`."""
    resolved = resolve_dataset_id(dataset, dataset_id)
    return {} if resolved is None else {"id": resolved}

def _mol_to_graph(mol):
    """RDKit Mol -> networkx graph.

    Single source of truth for molecule->graph construction, shared by every
    molecular loader (NCI SDF, ogbg-molhiv SMILES) and mirrored in
    utils/inference.py so training and inference graphs are byte-for-byte
    compatible: node feature = atom symbol, bonds as edges with bond attributes.
    """
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

    return G

class GraphDataLoader:

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Datasets are loaded lazily and cached the first time they are
            # requested, so importing this class no longer forces every dataset
            # (or its heavy dependencies / downloads) to be read up front.
            self._cache = {}
            self._initialized = True

    # -- unified entry point --------------------------------------------------

    def load(self, dataset="nci_full", **kwargs):
        """Return (graphs, labels) for `dataset`, caching the result.

        Supported datasets:
          * "nci_full"      - full NCI screen, imbalanced (kwargs: id)
          * "nci_balanced"  - 1:1 balanced NCI screen (kwargs: id)
          * "mutag"         - MUTAG (TU Dortmund format)
          * "ptc_mr"        - PTC-MR (TU Dortmund format)
          * "ogbg_molhiv"   - ogbg-molhiv (OGB, built from SMILES via RDKit)

        Labels follow the pipeline convention: -1 (majority/negative) and
        1 (minority/positive).
        """
        dataset = dataset.lower()
        key = (dataset, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            try:
                loader = self._LOADERS[dataset]
            except KeyError:
                raise ValueError(
                    f"Unknown dataset {dataset!r}. Available: "
                    f"{sorted(self._LOADERS)}"
                )
            self._cache[key] = loader(self, **kwargs)
        return self._cache[key]

    # -- NCI ------------------------------------------------------------------

    def _load_nci(self, variant, id):
        """Shared body for the NCI variants; `variant` keys `_NCI_VARIANTS`.

        Both variants are SDF files holding the same kind of record, so only the
        directory and file name differ between them. The class label lives in
        each molecule's "value" property and already follows the pipeline's
        {-1, 1} convention in both, so no remapping is needed.
        """
        id = int(id)
        dir_name, name_template = _NCI_VARIANTS[variant]
        print(f"Loading {variant} dataset (id={id})")
        DATASET_DIR = os.path.join(DATASETS_DIR, dir_name)
        graphs = []
        y = []

        filename = name_template.format(id=id)
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"NCI screen {id} not found at {os.path.abspath(filepath)}. "
                f"Known ids: {list(NCI_IDS)}."
            )

        supplier = Chem.SDMolSupplier(filepath, removeHs=False)
        skipped = 0
        for mol in supplier:
            if mol is None:
                skipped += 1
                continue

            G = _mol_to_graph(mol)

            # Get graph label
            # In NCI, class label is stored as a molecule property
            label = int(float(mol.GetProp("value")))
            graphs.append(G)
            y.append(label)

        print(f"Loaded {len(graphs)} graphs, skipped {skipped} unsanitizable molecules")
        return graphs, y

    def load_nci_full(self, id=DEFAULT_NCI_ID):
        """Full NCI screen, as shipped (heavily imbalanced).

        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)

        The id selects which NCI screen is loaded and is carried into every
        artefact name and manifest downstream (see `dataset_tag`), so runs on
        different screens never collide or get compared by accident.
        """
        return self._load_nci("nci_full", id)

    def load_nci_balanced(self, id=DEFAULT_NCI_ID):
        """Class-balanced (1:1) subsample of the same NCI screen.

        Takes the same ids as `load_nci_full`. Use it as the imbalance control:
        the encoder, dictionary learner and split machinery are unchanged, so a
        balanced run isolates how much of a full-screen result is driven by the
        class ratio rather than by the method.
        """
        return self._load_nci("nci_balanced", id)

   
   

    # Registry used by load(); maps dataset name -> loader method.
    _LOADERS = {
        "nci_full": load_nci_full,
        "nci_balanced": load_nci_balanced,
    }

    # -- backward-compatible lazy accessors -----------------------------------

    @property
    def nci_full_graphs(self):
        return self.load("nci_full")[0]

    @property
    def nci_full_labels(self):
        return self.load("nci_full")[1]

    @property
    def nci_balanced_graphs(self):
        return self.load("nci_balanced")[0]

    @property
    def nci_balanced_labels(self):
        return self.load("nci_balanced")[1]


def available_datasets():
    """Names accepted by `GraphDataLoader.load`, for CLI choices and messages."""
    return sorted(GraphDataLoader._LOADERS)
