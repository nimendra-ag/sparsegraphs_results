import os
import gzip
import zipfile
import urllib.request

import networkx as nx
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit's per-molecule sanitization warnings/errors (e.g. "Explicit
# valence ... is greater than permitted"). We count the skipped molecules
# ourselves instead of streaming every failure to the terminal.
RDLogger.DisableLog("rdApp.*")


DATASETS_DIR = "datasets"

# Public download locations.
#   * TU Dortmund graph-kernel datasets (MUTAG, PTC_MR, ...):
#         https://chrsmrrs.github.io/datasets/docs/datasets/
#   * OGB molecule property-prediction datasets (ogbg-molhiv, ...):
#         https://ogb.stanford.edu/docs/graphprop/
# TU datasets are served from a couple of mirrors that go down often; we try
# each in turn. If they are all unreachable you can also just download the zip
# by hand (from the docs page above) and drop it into `datasets/` — the loaders
# extract a pre-placed zip / folder without touching the network.
_TU_URLS = [
    "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip",
    "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{name}.zip",
]
_OGB_MOLHIV_URLS = [
    "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
]


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


def _download_and_extract(urls, dest_dir, zip_name):
    """Fetch a dataset zip and extract it into `dest_dir`.

    `urls` is an ordered list of candidate download URLs (mirrors); the first
    one that succeeds wins. If `dest_dir/zip_name` already exists (e.g. the user
    downloaded it by hand) no network access happens at all — we just extract it.

    Raises a RuntimeError with manual-download instructions if every mirror is
    unreachable, so the user gets an actionable message instead of a raw
    getaddrinfo/URLError traceback.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, zip_name)

    if not os.path.exists(zip_path):
        errors = []
        for url in urls:
            try:
                print(f"Downloading {url}")
                # A browser-like User-Agent avoids 403s from picky mirrors.
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, \
                        open(zip_path, "wb") as out:
                    out.write(resp.read())
                break
            except Exception as exc:  # noqa: BLE001 - report all mirror failures
                errors.append(f"  {url}\n    -> {type(exc).__name__}: {exc}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        else:
            joined = "\n".join(errors)
            raise RuntimeError(
                f"Could not download the dataset from any mirror:\n{joined}\n\n"
                f"The dataset host is likely down or unreachable. To proceed "
                f"offline, download the file manually and place it at:\n"
                f"    {os.path.abspath(zip_path)}\n"
                f"(or extract it so that its folder sits inside "
                f"'{os.path.abspath(dest_dir)}'), then re-run."
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


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
          * "nci_full"      - NCI SDF files (kwargs: id, default 1)
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

    def load_nci_full(self, id=1):
        """
        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)
        """
        print('Loading NCI dataset')
        DATASET_DIR = os.path.join(DATASETS_DIR, "NCI_full")
        graphs = []
        y = []

        filename = f"{id}total-connect.sdf"
        filepath = os.path.join(DATASET_DIR, filename)

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

    # -- TU Dortmund datasets (MUTAG, PTC-MR) ---------------------------------

    def load_mutag(self):
        return self._load_tu("MUTAG")

    def load_ptc_mr(self):
        return self._load_tu("PTC_MR")

    def _load_tu(self, name):
        """Load a TU Dortmund graph-kernel dataset.

        The archive stores the whole collection in a handful of flat text files
        (https://chrsmrrs.github.io/datasets/docs/format/):

          {name}_graph_indicator.txt : node i (1-indexed) -> graph id
          {name}_A.txt               : "u, v" edges over global 1-indexed nodes
          {name}_graph_labels.txt    : graph id -> class label
          {name}_node_labels.txt     : node i -> integer node label (optional)

        WL only needs a discrete node label, so we use the integer node label as
        the node `feature`. Graph labels are remapped to {-1, 1}.
        """
        print(f"Loading {name} dataset")
        root = os.path.join(DATASETS_DIR, name)
        if not os.path.isdir(root):
            urls = [u.format(name=name) for u in _TU_URLS]
            _download_and_extract(urls, DATASETS_DIR, f"{name}.zip")

        def _read(suffix):
            path = os.path.join(root, f"{name}_{suffix}.txt")
            if not os.path.exists(path):
                return None
            with open(path) as fh:
                return [line.strip() for line in fh if line.strip()]

        graph_indicator = _read("graph_indicator")
        graph_labels = _read("graph_labels")
        node_labels = _read("node_labels")
        edges = _read("A")

        # Build one empty graph per graph id, and a global-node -> (graph, node)
        # map. Node ids are made local (0-based) per graph.
        graphs_by_id = {}
        node_to_graph = {}
        local_index = {}
        for global_node, gid in enumerate(graph_indicator, start=1):
            gid = int(gid)
            G = graphs_by_id.setdefault(gid, nx.Graph())
            local = G.number_of_nodes()
            feature = node_labels[global_node - 1] if node_labels else "0"
            G.add_node(local, feature=str(feature))
            node_to_graph[global_node] = gid
            local_index[global_node] = local

        # Add edges (the adjacency list is symmetric; nx.Graph dedupes).
        for line in edges:
            u, v = (int(x) for x in line.split(","))
            gid = node_to_graph[u]
            graphs_by_id[gid].add_edge(local_index[u], local_index[v])

        # Assemble in graph-id order and remap labels to {-1, 1}.
        graphs, y = [], []
        raw_labels = [int(v) for v in graph_labels]
        # In TU molecular datasets the positive class is the larger label
        # (e.g. {-1, 1} or {0, 1}); map the max label -> 1, everything else -> -1.
        positive = max(raw_labels)
        for gid in sorted(graphs_by_id):
            graphs.append(graphs_by_id[gid])
            y.append(1 if raw_labels[gid - 1] == positive else -1)

        print(f"Loaded {len(graphs)} graphs")
        return graphs, y

    # -- OGB (ogbg-molhiv) ----------------------------------------------------

    def load_ogbg_molhiv(self):
        """Load ogbg-molhiv (https://ogb.stanford.edu/docs/graphprop/).

        Rather than pulling in the heavy `ogb`/`torch` stack, we download OGB's
        raw CSV release, which ships the molecules as SMILES plus the binary
        HIV_active label. Each molecule is built with RDKit through the same
        `_mol_to_graph` path as NCI, so ogbg-molhiv graphs are fully compatible
        with the WL encoder and the inference pipeline. Labels are remapped from
        {0, 1} to {-1, 1}.
        """
        print("Loading ogbg-molhiv dataset")
        # OGB ships ogbg-molhiv's raw CSV as "hiv.zip", which extracts to hiv/.
        root = os.path.join(DATASETS_DIR, "hiv")
        csv_path = os.path.join(root, "mapping", "mol.csv.gz")
        if not os.path.exists(csv_path):
            _download_and_extract(_OGB_MOLHIV_URLS, DATASETS_DIR, "hiv.zip")

        graphs, y = [], []
        skipped = 0
        with gzip.open(csv_path, "rt") as fh:
            header = fh.readline().rstrip("\n").split(",")
            smiles_col = header.index("smiles")
            label_col = header.index("HIV_active")
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) <= max(smiles_col, label_col):
                    continue
                smiles = parts[smiles_col]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped += 1
                    continue
                # AddHs to mirror inference.smiles_to_graph and the explicit-H
                # atoms present in the NCI SDF graphs.
                mol = Chem.AddHs(mol)
                graphs.append(_mol_to_graph(mol))
                y.append(1 if int(parts[label_col]) == 1 else -1)

        print(f"Loaded {len(graphs)} graphs, skipped {skipped} unparseable molecules")
        return graphs, y

    # Registry used by load(); maps dataset name -> loader method.
    _LOADERS = {
        "nci_full": load_nci_full,
        "mutag": load_mutag,
        "ptc_mr": load_ptc_mr,
        "ogbg_molhiv": load_ogbg_molhiv,
    }

    # -- backward-compatible lazy accessors -----------------------------------

    @property
    def nci_full_graphs(self):
        return self.load("nci_full")[0]

    @property
    def nci_full_labels(self):
        return self.load("nci_full")[1]
