"""
gSpan: Graph-Based Substructure Pattern Mining
===============================================

Faithful implementation of:
    Yan, X. and Han, J. "gSpan: Graph-Based Substructure Pattern Mining"
    Proc. 2002 IEEE International Conference on Data Mining (ICDM'02)

This implements:
    - DFS Code representation (Definition 4, 5 in the paper)
    - DFS Lexicographic Order (Definition 5, page 10)
    - Minimum DFS Code / Canonical Form (Definition 6, page 10)
    - Algorithm 2: GraphSet_Projection (page 14)
    - Subprocedure 1: Subgraph_Mining (page 15)
    - Subprocedure 2: Enumerate (page 16)
    - Rightmost Extension (Section 3, Property 1)

Assumption:
    If your input graphs lack edge labels, a default edge label (0) is used.
    The paper's algorithm requires both vertex and edge labels.
"""

from collections import defaultdict
from copy import deepcopy
import networkx as nx

# ---------------------------------------------------------------------------
# Default label used when graphs have no edge labels.
# Paper Section 2 defines l: V ∪ E → Σ, i.e., BOTH vertex and edge labels.
# ---------------------------------------------------------------------------
VACANT_EDGE_LABEL = -1  # sentinel for "no edge"
VACANT_VERTEX_LABEL = -1


# ===========================================================================
# Data Structures
# ===========================================================================

class Edge:
    """An edge in the input graph, stored in adjacency-list style."""
    __slots__ = ('frm', 'to', 'elabel')

    def __init__(self, frm, to, elabel):
        self.frm = frm
        self.to = to
        self.elabel = elabel

    def __repr__(self):
        return f"Edge({self.frm}->{self.to}, elabel={self.elabel})"


class Vertex:
    """A vertex in the input graph."""
    __slots__ = ('vid', 'vlabel', 'edges')

    def __init__(self, vid, vlabel):
        self.vid = vid
        self.vlabel = vlabel
        self.edges = []  # list of Edge

    def __repr__(self):
        return f"Vertex({self.vid}, vlabel={self.vlabel})"


class Graph:
    """
    Internal graph representation used by gSpan.
    Sparse adjacency list as described in Section 5 of the paper.
    """
    def __init__(self, gid=None):
        self.gid = gid
        self.vertices = {}       # vid -> Vertex
        self.set_of_elabel = set()
        self.set_of_vlabel = set()

    def add_vertex(self, vid, vlabel):
        if vid in self.vertices:
            return
        v = Vertex(vid, vlabel)
        self.vertices[vid] = v
        self.set_of_vlabel.add(vlabel)

    def add_edge(self, frm, to, elabel):
        self.vertices[frm].edges.append(Edge(frm, to, elabel))
        self.vertices[to].edges.append(Edge(to, frm, elabel))
        self.set_of_elabel.add(elabel)


# ===========================================================================
# DFS Code Edge — the 5-tuple (i, j, l_vi, l_edge, l_vj)
# Paper Definition 4: "An edge e = (i, j, l_i, l_(i,j), l_j)"
# Backward edge: i > j   (going to an already-discovered vertex)
# Forward  edge: i < j   (introducing a new vertex)
# ===========================================================================

class DFSEdge:
    """
    One entry in a DFS code: the 5-tuple (frm, to, vlabel_frm, elabel, vlabel_to).
    Per the paper, a backward edge has frm > to, and a forward edge has frm < to.
    """
    __slots__ = ('frm', 'to', 'vlabel_frm', 'elabel', 'vlabel_to')

    def __init__(self, frm, to, vlabel_frm, elabel, vlabel_to):
        self.frm = frm
        self.to = to
        self.vlabel_frm = vlabel_frm
        self.elabel = elabel
        self.vlabel_to = vlabel_to

    def is_forward(self):
        return self.frm < self.to

    def is_backward(self):
        return self.frm > self.to

    def __eq__(self, other):
        return (self.frm == other.frm and self.to == other.to and
                self.vlabel_frm == other.vlabel_frm and
                self.elabel == other.elabel and
                self.vlabel_to == other.vlabel_to)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return (f"({self.frm}, {self.to}, "
                f"{self.vlabel_frm}, {self.elabel}, {self.vlabel_to})")

    # -----------------------------------------------------------------------
    # DFS Lexicographic Order on individual edges (Definition 5, page 10).
    #
    # Given two edge tuples a = (i_a, j_a, ...) and b = (i_b, j_b, ...):
    #   1. backward < forward
    #   2. Both backward:  compare j (smaller j first),
    #                      then edge label, (vertex labels are fixed by DFS indices)
    #   3. Both forward:   compare i (LARGER i first — rightmost-vertex-first),
    #                      then frm vertex label, edge label, to vertex label.
    # -----------------------------------------------------------------------
    def __lt__(self, other):
        # Case 1: backward < forward
        if self.is_backward() and other.is_forward():
            return True
        if self.is_forward() and other.is_backward():
            return False

        # Case 2: both backward
        if self.is_backward() and other.is_backward():
            if self.to != other.to:
                return self.to < other.to
            # Same target DFS index — compare edge label
            return self.elabel < other.elabel

        # Case 3: both forward
        # Paper: "a_t < b_t if i_b < i_a" — i.e., deeper source wins.
        if self.frm != other.frm:
            return self.frm > other.frm  # deeper (larger DFS index) is "smaller"
        if self.vlabel_frm != other.vlabel_frm:
            return self.vlabel_frm < other.vlabel_frm  # not used in practice (frm index fixes it)
        if self.elabel != other.elabel:
            return self.elabel < other.elabel
        return self.vlabel_to < other.vlabel_to

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return self == other or other < self

    def __hash__(self):
        return hash((self.frm, self.to, self.vlabel_frm, self.elabel, self.vlabel_to))


class DFSCode:
    """
    A DFS code: an ordered sequence of DFSEdge tuples.
    Paper Definition 4: code(G, T) = (a_0, a_1, ..., a_m).
    """
    def __init__(self):
        self.edges = []

    def push_back(self, frm, to, vl_frm, el, vl_to):
        self.edges.append(DFSEdge(frm, to, vl_frm, el, vl_to))

    def pop_back(self):
        self.edges.pop()

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return self.edges[idx]

    def __repr__(self):
        return " | ".join(str(e) for e in self.edges)

    def rightmost_vertex(self):
        """The rightmost vertex is the vertex with the highest DFS index."""
        rm = 0
        for e in self.edges:
            rm = max(rm, e.frm, e.to)
        return rm

    def rightmost_path(self):
        """
        Compute the rightmost path: the straight path from v_0 to v_n
        in the DFS tree (forward edges only).
        Returns a list of DFS vertex indices from root to rightmost vertex.
        """
        # Build the forward-edge tree
        # The rightmost path is found by tracing from the rightmost vertex
        # back to root through the forward edges.
        parent = {}
        max_v = 0
        for e in self.edges:
            if e.is_forward():
                parent[e.to] = e.frm
                if e.to > max_v:
                    max_v = e.to
        path = []
        v = max_v
        while v in parent:
            path.append(v)
            v = parent[v]
        path.append(v)  # root (v=0)
        path.reverse()
        return path

    def build_graph(self):
        """Build a Graph object from this DFS code (for canonical-form checking)."""
        g = Graph()
        for e in self.edges:
            if e.frm not in g.vertices:
                g.add_vertex(e.frm, e.vlabel_frm)
            if e.to not in g.vertices:
                g.add_vertex(e.to, e.vlabel_to)
            g.add_edge(e.frm, e.to, e.elabel)
        return g


# ===========================================================================
# PDFS: Projected DFS — tracks where a pattern occurs in the database.
# Each entry records an embedding (occurrence) of the current pattern
# in a specific database graph.
# ===========================================================================

class PDFSEntry:
    """One embedding record: which graph, which edge, and backpointer."""
    __slots__ = ('gid', 'edge', 'prev')

    def __init__(self, gid, edge, prev):
        self.gid = gid
        self.edge = edge      # the Edge in the database graph
        self.prev = prev      # PDFSEntry or None (linked list of embeddings)


class Projected(list):
    """A list of PDFSEntry — one per occurrence of the current DFS code."""

    def get_support(self):
        """Count distinct graph IDs (= support, per paper Definition in Section 2)."""
        return len({entry.gid for entry in self})


# ===========================================================================
# Helper: trace back the full embedding from a PDFSEntry chain.
# ===========================================================================

def _build_right_most_path(dfs_code, right_most_path_out):
    """Compute rightmost path indices into dfs_code.edges."""
    right_most_path_out.clear()
    # The rightmost path consists of the forward edges that form the path
    # from root to the rightmost vertex.
    old_frm = -1
    for i in range(len(dfs_code) - 1, -1, -1):
        e = dfs_code[i]
        if e.is_forward() and (old_frm == -1 or e.to == old_frm):
            right_most_path_out.append(i)
            old_frm = e.frm
    return right_most_path_out


def _get_forward_root_edges(g, v):
    """
    Get all edges from vertex v in graph g that can serve as forward root edges
    (1-edge subgraphs starting from v).
    """
    result = []
    for e in g.vertices[v].edges:
        vl_to = g.vertices[e.to].vlabel
        vl_from = g.vertices[v].vlabel
        # For 1-edge DFS codes: ensure the starting vertex label <= ending vertex label
        # (or starting < ending), otherwise we'd produce the same edge twice.
        if vl_from < vl_to or (vl_from == vl_to and v < e.to):
            result.append(e)
        elif vl_from == vl_to and v == e.to:
            # self-loop: skip (shouldn't happen in molecular graphs)
            pass
    return result


def _get_forward_pure_edges(g, v_frm, v_to, history):
    """
    Forward edges from v_frm to new vertices not in history.
    These are candidates for forward extension from a vertex on the rightmost path.
    """
    result = []
    for e in g.vertices[v_frm].edges:
        if e.to in history:
            continue
        # Must go to a vertex not yet in the embedding
        result.append(e)
    return result


def _get_backward_edge(g, dfs_edge_frm, dfs_edge_to, history):
    """
    Look for a backward edge in graph g from the rightmost vertex (mapped by dfs_edge_to)
    to a vertex on the rightmost path (mapped by dfs_edge_frm).
    """
    v_to = dfs_edge_to.to  # actual vertex in g for the rightmost vertex
    v_frm = dfs_edge_frm.frm  # actual vertex in g we want to connect back to
    for e in g.vertices[v_to].edges:
        if e.to == v_frm:
            # Must not be an edge already in the history
            if (e.frm, e.to) not in history and (e.to, e.frm) not in history:
                return e
    return None


# ===========================================================================
# History: track which vertices and edges are used in the current embedding.
# ===========================================================================

class History:
    """
    Reconstruct the embedding by walking the PDFSEntry linked list.
    Provides O(1) lookup for which vertices/edges are already used.
    """
    def __init__(self, g, pdfs_entry):
        self.edges = []    # list of Edge in order
        self.vertex_set = set()
        self.edge_set = set()  # (frm, to) pairs with frm < to

        # Walk back through the linked list
        e_list = []
        p = pdfs_entry
        while p is not None:
            e_list.append(p.edge)
            p = p.prev
        e_list.reverse()

        for e in e_list:
            self.edges.append(e)
            self.vertex_set.add(e.frm)
            self.vertex_set.add(e.to)
            key = (min(e.frm, e.to), max(e.frm, e.to))
            self.edge_set.add(key)

    def has_vertex(self, vid):
        return vid in self.vertex_set

    def has_edge(self, frm, to):
        key = (min(frm, to), max(frm, to))
        return key in self.edge_set

    def __contains__(self, vid):
        return vid in self.vertex_set


# ===========================================================================
# Minimum DFS Code Check — Definition 6 (page 10)
#
# "Given a graph G, Z(G) = {code(G,T) | ∀T, T is a DFS tree for G},
#  based on DFS lexicographic order, the minimum one, min(Z(G)),
#  is called Minimum DFS Code of G."
#
# This is the canonicality check: s ≠ min(s) → prune (Subprocedure 1, line 1).
# Implemented by rebuilding the graph from the DFS code and attempting to
# construct a DFS code that is lexicographically smaller.
# ===========================================================================

def _is_min(dfs_code):
    """
    Check if the current DFS code is the minimum DFS code for the graph
    it represents. (Paper: s ≠ min(s) → return in Subprocedure 1, line 1.)

    We reconstruct the graph and try to build a DFS code from scratch.
    At each step, if the newly chosen edge is smaller than the corresponding
    edge in dfs_code, then dfs_code is NOT minimum. If it's larger, we can
    prune that branch. If equal, we continue.

    Returns True if dfs_code IS the minimum DFS code.
    """
    if len(dfs_code) <= 1:
        return True

    g = dfs_code.build_graph()
    min_code = DFSCode()

    # Try to build the minimum DFS code via the same rightmost extension rules
    return _is_min_recursive(dfs_code, g, min_code)


def _is_min_recursive(dfs_code, g, min_code):
    """Recursive helper for minimum DFS code check."""
    idx = len(min_code)
    if idx == len(dfs_code):
        return True  # They match all the way — it IS minimal

    # Build the rightmost path of min_code so far
    rm_path = []
    _build_right_most_path(min_code, rm_path)

    rm_vertex = -1  # DFS index of the rightmost vertex in min_code
    if len(min_code) > 0:
        rm_vertex = max(max(e.frm, e.to) for e in min_code.edges)

    # Map from DFS index → actual vertex in g
    # Rebuild the mapping from the min_code built so far
    dfs_to_vertex = {}
    vertex_to_dfs = {}
    if len(min_code) > 0:
        # Walk the min_code to build the mapping
        dfs_to_vertex[min_code[0].frm] = min_code[0].frm
        dfs_to_vertex[min_code[0].to] = min_code[0].to
        vertex_to_dfs[min_code[0].frm] = min_code[0].frm
        vertex_to_dfs[min_code[0].to] = min_code[0].to
        # Actually for the canonical check, DFS index IS the vertex ID in g
        # because g was built from the DFS code with those exact indices.

    # --- Generate candidate extensions in DFS lexicographic order ---

    # 1. Try backward extensions from the rightmost vertex
    #    (to vertices on the rightmost path, smallest target DFS index first)
    if len(rm_path) > 0:
        for i in range(len(rm_path) - 1, -1, -1):
            rm_edge_idx = rm_path[i]
            rm_edge = min_code[rm_edge_idx]
            target_dfs = rm_edge.frm  # the vertex on the rightmost path

            # Check if there's an edge in g from rm_vertex to target_dfs
            for edge in g.vertices[rm_vertex].edges:
                if edge.to == target_dfs:
                    # Check this edge isn't already in min_code
                    already_in = False
                    for me in min_code.edges:
                        if ((me.frm == rm_vertex and me.to == target_dfs) or
                                (me.frm == target_dfs and me.to == rm_vertex)):
                            already_in = True
                            break
                    if already_in:
                        continue

                    # Build the candidate backward DFSEdge
                    candidate = DFSEdge(
                        rm_vertex, target_dfs,
                        g.vertices[rm_vertex].vlabel,
                        edge.elabel,
                        g.vertices[target_dfs].vlabel
                    )
                    if candidate < dfs_code[idx]:
                        return False  # Found something smaller → NOT minimum
                    if candidate == dfs_code[idx]:
                        min_code.push_back(
                            candidate.frm, candidate.to,
                            candidate.vlabel_frm, candidate.elabel,
                            candidate.vlabel_to
                        )
                        result = _is_min_recursive(dfs_code, g, min_code)
                        min_code.pop_back()
                        return result
                    # candidate > dfs_code[idx]: try next

    # 2. Try forward extensions from rightmost path vertices
    #    (from the rightmost vertex first, then up the path)
    new_to = rm_vertex + 1 if rm_vertex >= 0 else 1
    used_vertices = set()
    if len(min_code) > 0:
        for e in min_code.edges:
            used_vertices.add(e.frm)
            used_vertices.add(e.to)

    # Iterate rightmost path from bottom (rightmost vertex) to top (root)
    rm_path_vertices = []
    if len(min_code) > 0:
        # Extract the DFS indices on the rightmost path
        old_frm = -1
        for i in range(len(min_code) - 1, -1, -1):
            e = min_code[i]
            if e.is_forward() and (old_frm == -1 or e.to == old_frm):
                rm_path_vertices.append(e.to)
                old_frm = e.frm
        rm_path_vertices.append(old_frm if old_frm != -1 else 0)
        rm_path_vertices.reverse()
    else:
        rm_path_vertices = [0]

    for v_dfs in reversed(rm_path_vertices):
        # Try forward extensions from v_dfs
        for edge in sorted(g.vertices[v_dfs].edges,
                           key=lambda e: (e.elabel, g.vertices[e.to].vlabel)):
            if edge.to in used_vertices:
                continue
            candidate = DFSEdge(
                v_dfs, new_to,
                g.vertices[v_dfs].vlabel,
                edge.elabel,
                g.vertices[edge.to].vlabel
            )
            if candidate < dfs_code[idx]:
                return False
            if candidate == dfs_code[idx]:
                min_code.push_back(
                    candidate.frm, candidate.to,
                    candidate.vlabel_frm, candidate.elabel,
                    candidate.vlabel_to
                )
                result = _is_min_recursive(dfs_code, g, min_code)
                min_code.pop_back()
                return result

    # If we get here, we couldn't find a matching extension.
    # This means the current dfs_code is still potentially minimal.
    return True


# ===========================================================================
# gSpan Main Class
# ===========================================================================

class GSpan:
    """
    gSpan: Graph-based Substructure Pattern Mining.

    Parameters
    ----------
    min_support : int
        Minimum absolute support threshold.
        A subgraph is frequent if it occurs in >= min_support graphs.
    min_num_vertices : int
        Minimum number of vertices in a frequent subgraph (default 1).
    max_num_vertices : int
        Maximum number of vertices in a frequent subgraph (default inf).
        Set this to control pattern size and runtime.
    verbose : bool
        If True, print progress during mining.

    Usage
    -----
        miner = GSpan(min_support=10, max_num_vertices=10)
        miner.run(graphs)
        # miner.frequent_subgraphs is a list of (DFSCode, support, graph_ids)
    """

    def __init__(self, min_support=10, min_num_vertices=1,
                 max_num_vertices=float('inf'), verbose=False):
        self.min_support = min_support
        self.min_num_vertices = min_num_vertices
        self.max_num_vertices = max_num_vertices
        self.verbose = verbose

        self._database = []          # list of Graph
        self._dfs_code = DFSCode()   # current DFS code being extended
        self.frequent_subgraphs = [] # results: list of (DFSCode copy, support, gid_set)
        self._report_count = 0

    # -------------------------------------------------------------------
    # Convert NetworkX graphs to internal format
    # -------------------------------------------------------------------
    def _convert_graphs(self, nx_graphs, node_label_attr='feature',
                        edge_label_attr=None, default_edge_label=0):
        """
        Convert a list of NetworkX graphs to the internal Graph representation.

        Parameters
        ----------
        nx_graphs : list of nx.Graph
        node_label_attr : str
            Node attribute name holding the vertex label.
        edge_label_attr : str or None
            Edge attribute name holding the edge label. If None, use default.
        default_edge_label : int
            Edge label to use when edge_label_attr is None.

        Returns
        -------
        list of Graph, dict mapping label strings to integers
        """
        # Collect all labels and map to integers for efficient comparison
        vlabel_set = set()
        elabel_set = set()

        for g in nx_graphs:
            for _, data in g.nodes(data=True):
                vlabel_set.add(data.get(node_label_attr, 'UNK'))
            if edge_label_attr:
                for _, _, data in g.edges(data=True):
                    elabel_set.add(data.get(edge_label_attr, default_edge_label))

        # Sort labels for deterministic mapping
        vlabel_map = {lab: i for i, lab in enumerate(sorted(vlabel_set))}
        if edge_label_attr:
            elabel_map = {lab: i for i, lab in enumerate(sorted(elabel_set))}
        else:
            elabel_map = {}

        graphs = []
        for gid, nx_g in enumerate(nx_graphs):
            g = Graph(gid=gid)
            # Re-index vertices to 0..n-1
            node_list = sorted(nx_g.nodes())
            node_remap = {old: new for new, old in enumerate(node_list)}

            for old_id in node_list:
                new_id = node_remap[old_id]
                raw_label = nx_g.nodes[old_id].get(node_label_attr, 'UNK')
                g.add_vertex(new_id, vlabel_map[raw_label])

            for u, v, data in nx_g.edges(data=True):
                nu, nv = node_remap[u], node_remap[v]
                if edge_label_attr:
                    el = elabel_map[data.get(edge_label_attr, default_edge_label)]
                else:
                    el = default_edge_label
                g.add_edge(nu, nv, el)

            graphs.append(g)

        return graphs, vlabel_map, elabel_map

    # -------------------------------------------------------------------
    # Algorithm 2: GraphSet_Projection  (Paper page 14)
    # -------------------------------------------------------------------
    def run(self, nx_graphs, node_label_attr='feature',
            edge_label_attr=None, default_edge_label=0):
        """
        Run gSpan on a list of NetworkX graphs.

        Parameters
        ----------
        nx_graphs : list of nx.Graph
            The graph database.
        node_label_attr : str
            Node attribute key for vertex labels (default 'feature').
        edge_label_attr : str or None
            Edge attribute key for edge labels. None = use default_edge_label.
        default_edge_label : int
            Label to use if edge_label_attr is None.

        Returns
        -------
        self : GSpan
            Access results via self.frequent_subgraphs.
        """
        self.frequent_subgraphs = []
        self._report_count = 0

        # Convert to internal representation
        self._database, self.vlabel_map, self.elabel_map = self._convert_graphs(
            nx_graphs, node_label_attr, edge_label_attr, default_edge_label
        )
        self.inv_vlabel_map = {v: k for k, v in self.vlabel_map.items()}
        self.inv_elabel_map = {v: k for k, v in self.elabel_map.items()}

        if self.verbose:
            print(f"[gSpan] Database: {len(self._database)} graphs, "
                  f"{len(self.vlabel_map)} vertex labels, "
                  f"{len(self.elabel_map) if self.elabel_map else '1 (default)'} edge labels")

        # ----- Step 1 (lines 1-3): Sort labels by frequency, remove infrequent -----
        # Count vertex and edge label frequencies
        vlabel_freq = defaultdict(int)
        elabel_freq = defaultdict(int)
        for g in self._database:
            seen_v = set()
            seen_e = set()
            for vid, v in g.vertices.items():
                if v.vlabel not in seen_v:
                    vlabel_freq[v.vlabel] += 1
                    seen_v.add(v.vlabel)
                for e in v.edges:
                    ekey = (min(e.frm, e.to), max(e.frm, e.to), e.elabel)
                    if ekey not in seen_e:
                        elabel_freq[e.elabel] += 1
                        seen_e.add(ekey)

        # ----- Step 2 (line 4): Find all frequent 1-edge subgraphs -----
        # A 1-edge subgraph is (0, 1, l_v0, l_e, l_v1) with l_v0 <= l_v1
        # (to avoid listing the same edge twice in different directions).
        edge_counter = defaultdict(set)  # (vl_from, el, vl_to) -> set of gids

        for g in self._database:
            done_edges = set()
            for vid, v in g.vertices.items():
                for e in v.edges:
                    if (e.frm, e.to) in done_edges:
                        continue
                    done_edges.add((e.frm, e.to))
                    done_edges.add((e.to, e.frm))

                    vl_frm = v.vlabel
                    vl_to = g.vertices[e.to].vlabel
                    el = e.elabel

                    # Canonical form: ensure vl_frm <= vl_to
                    if vl_frm <= vl_to:
                        key = (vl_frm, el, vl_to)
                    else:
                        key = (vl_to, el, vl_frm)
                    edge_counter[key].add(g.gid)

        # ----- Step 3 (lines 5-6): Sort 1-edge subgraphs in DFS lexicographic order -----
        # DFS lex order for 1-edge: (0, 1, l_v0, l_e, l_v1) — just sort by the tuple.
        freq_1edges = []
        for (vl_frm, el, vl_to), gids in edge_counter.items():
            if len(gids) >= self.min_support:
                freq_1edges.append((vl_frm, el, vl_to, gids))

        freq_1edges.sort(key=lambda x: (x[0], x[1], x[2]))

        if self.verbose:
            print(f"[gSpan] Frequent 1-edge subgraphs: {len(freq_1edges)}")

        # ----- Step 4 (lines 7-12): For each frequent 1-edge, mine all descendants -----
        for vl_frm, el, vl_to, gids in freq_1edges:
            # Initialize the DFS code with this 1-edge
            self._dfs_code = DFSCode()
            self._dfs_code.push_back(0, 1, vl_frm, el, vl_to)

            # Build the initial projected database
            projected = Projected()
            for gid in gids:
                g = self._database[gid]
                for vid, v in g.vertices.items():
                    if v.vlabel != vl_frm:
                        continue
                    for e in v.edges:
                        if e.elabel != el:
                            continue
                        if g.vertices[e.to].vlabel != vl_to:
                            continue
                        # For canonical 1-edges where vl_frm == vl_to,
                        # only take one direction to avoid duplicates.
                        if vl_frm == vl_to and e.frm > e.to:
                            continue
                        projected.append(PDFSEntry(gid, e, None))

            # Subgraph_Mining (Subprocedure 1)
            self._subgraph_mining(projected)

            self._dfs_code = DFSCode()

        if self.verbose:
            print(f"[gSpan] Mining complete. Found {len(self.frequent_subgraphs)} "
                  f"frequent subgraphs.")

        return self

    # -------------------------------------------------------------------
    # Subprocedure 1: Subgraph_Mining  (Paper page 15)
    #
    # 1: if s ≠ min(s)  →  return
    # 2: S ← S ∪ {s}
    # 3: generate all s' potential children with one edge growth
    # 4: Enumerate(s)
    # 5: for each c, c is s' child do
    # 6:    if support(c) >= minSup
    # 7:       s ← c
    # 8:       Subgraph_Mining(GS, S, s)
    # -------------------------------------------------------------------
    def _subgraph_mining(self, projected):
        """
        Recursive pattern-growth mining following Subprocedure 1.

        Parameters
        ----------
        projected : Projected
            The projected database for the current DFS code.
        """
        # Line 1: Canonical form check — if s ≠ min(s), prune.
        if not _is_min(self._dfs_code):
            return

        # Line 2: Report this frequent subgraph
        support = projected.get_support()
        num_vertices = self._dfs_code.rightmost_vertex() + 1

        if num_vertices >= self.min_num_vertices:
            gid_set = {entry.gid for entry in projected}
            self.frequent_subgraphs.append(
                (deepcopy(self._dfs_code), support, gid_set)
            )
            self._report_count += 1
            if self.verbose and self._report_count % 1000 == 0:
                print(f"  [gSpan] Found {self._report_count} patterns so far "
                      f"(current code length: {len(self._dfs_code)})")

        # Check size limit
        if num_vertices >= self.max_num_vertices:
            return

        # Lines 3-4: Generate rightmost extensions and count support
        # (Enumerate — Subprocedure 2, merged with extension generation)
        rm_path = []
        _build_right_most_path(self._dfs_code, rm_path)

        rm_vertex_dfs = self._dfs_code.rightmost_vertex()

        # Collect all candidate extensions
        # backward_extensions[DFSEdge] -> Projected
        # forward_extensions[DFSEdge] -> Projected
        backward_extensions = defaultdict(Projected)
        forward_extensions = defaultdict(Projected)

        for entry in projected:
            gid = entry.gid
            g = self._database[gid]
            history = History(g, entry)

            # ---- Backward extensions ----
            # From the rightmost vertex to vertices on the rightmost path.
            # Walk the rightmost path and try connecting back to each vertex.
            for i in range(len(rm_path)):
                rm_edge_idx = rm_path[i]
                dfs_edge = self._dfs_code[rm_edge_idx]

                # The target of the backward extension in the DFS code
                backward_target_dfs = dfs_edge.frm

                # Get the actual graph vertices from the embedding
                rm_vertex_actual = history.edges[-1].to  # rightmost vertex in g
                # Walk the history backwards to find the vertex mapped to backward_target_dfs
                # We need to find which vertex in g corresponds to each DFS index.
                # Build the DFS-index-to-vertex mapping from history.
                dfs_to_actual = self._build_dfs_vertex_map(history)

                if backward_target_dfs not in dfs_to_actual:
                    continue
                target_actual = dfs_to_actual[backward_target_dfs]

                # Look for edge from rm_vertex_actual to target_actual in g
                for edge in g.vertices[rm_vertex_actual].edges:
                    if edge.to != target_actual:
                        continue
                    if history.has_edge(rm_vertex_actual, target_actual):
                        continue

                    # Valid backward extension
                    dfs_e = DFSEdge(
                        rm_vertex_dfs,
                        backward_target_dfs,
                        g.vertices[rm_vertex_actual].vlabel,
                        edge.elabel,
                        g.vertices[target_actual].vlabel
                    )
                    backward_extensions[dfs_e].append(
                        PDFSEntry(gid, edge, entry)
                    )
                    break  # Only one backward edge per target per embedding

            # ---- Forward extensions ----
            # From vertices on the rightmost path to new vertices.
            # Priority: rightmost vertex first (deepest), then up the path.
            dfs_to_actual = self._build_dfs_vertex_map(history)

            # Walk rightmost path from bottom (rightmost vertex) to top
            rm_path_dfs_vertices = []
            old_frm = -1
            for i in range(len(self._dfs_code) - 1, -1, -1):
                e = self._dfs_code[i]
                if e.is_forward() and (old_frm == -1 or e.to == old_frm):
                    rm_path_dfs_vertices.append(e.to)
                    old_frm = e.frm
            rm_path_dfs_vertices.append(old_frm if old_frm != -1 else 0)
            rm_path_dfs_vertices.reverse()

            new_to_dfs = rm_vertex_dfs + 1

            for v_dfs in reversed(rm_path_dfs_vertices):
                if v_dfs not in dfs_to_actual:
                    continue
                v_actual = dfs_to_actual[v_dfs]

                for edge in g.vertices[v_actual].edges:
                    if edge.to in history:
                        continue

                    dfs_e = DFSEdge(
                        v_dfs,
                        new_to_dfs,
                        g.vertices[v_actual].vlabel,
                        edge.elabel,
                        g.vertices[edge.to].vlabel
                    )
                    forward_extensions[dfs_e].append(
                        PDFSEntry(gid, edge, entry)
                    )

        # Lines 5-8: Recurse on each frequent child in DFS lexicographic order
        # Backward extensions first (smaller in lex order), then forward.
        all_extensions = []
        for dfs_e, proj in backward_extensions.items():
            all_extensions.append((dfs_e, proj))
        for dfs_e, proj in forward_extensions.items():
            all_extensions.append((dfs_e, proj))

        # Sort by DFS lexicographic order
        all_extensions.sort(key=lambda x: x[0])

        for dfs_e, proj in all_extensions:
            if proj.get_support() < self.min_support:
                continue

            self._dfs_code.push_back(
                dfs_e.frm, dfs_e.to,
                dfs_e.vlabel_frm, dfs_e.elabel, dfs_e.vlabel_to
            )
            self._subgraph_mining(proj)
            self._dfs_code.pop_back()

    def _build_dfs_vertex_map(self, history):
        """
        Build a mapping from DFS code vertex index → actual vertex ID in the
        database graph, by walking the DFS code edges and history edges in parallel.
        """
        dfs_to_actual = {}
        for i, (dfs_e, hist_e) in enumerate(zip(self._dfs_code.edges, history.edges)):
            if dfs_e.frm not in dfs_to_actual:
                dfs_to_actual[dfs_e.frm] = hist_e.frm
            if dfs_e.to not in dfs_to_actual:
                dfs_to_actual[dfs_e.to] = hist_e.to
        return dfs_to_actual

    # -------------------------------------------------------------------
    # Utility: Convert results to readable format
    # -------------------------------------------------------------------
    def get_frequent_subgraphs_as_nx(self):
        """
        Convert the mined frequent subgraphs to a list of dicts with:
        - 'graph': nx.Graph with original string labels
        - 'support': int
        - 'graph_ids': set of int
        - 'dfs_code': string representation of the DFS code
        """
        results = []
        for dfs_code, support, gid_set in self.frequent_subgraphs:
            G = nx.Graph()
            for e in dfs_code.edges:
                frm_label = self.inv_vlabel_map.get(e.vlabel_frm, e.vlabel_frm)
                to_label = self.inv_vlabel_map.get(e.vlabel_to, e.vlabel_to)
                if self.inv_elabel_map:
                    e_label = self.inv_elabel_map.get(e.elabel, e.elabel)
                else:
                    e_label = e.elabel

                if e.frm not in G:
                    G.add_node(e.frm, feature=frm_label)
                if e.to not in G:
                    G.add_node(e.to, feature=to_label)
                G.add_edge(e.frm, e.to, label=e_label)

            results.append({
                'graph': G,
                'support': support,
                'graph_ids': gid_set,
                'dfs_code': str(dfs_code),
                'num_vertices': dfs_code.rightmost_vertex() + 1,
                'num_edges': len(dfs_code),
            })
        return results
