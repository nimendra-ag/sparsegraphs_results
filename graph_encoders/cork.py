"""
CORK: Correspondence-based Quality Criterion for Feature Selection
===================================================================

Implementation of:
    Thoma, M. et al. "Near-optimal supervised feature selection among
    frequent subgraphs" Proc. SDM 2009

This is the **post-hoc** (un-nested) variant: given a pre-mined binary
indicator matrix and class labels, greedily select the feature subset
that minimises the number of cross-class correspondences.

A *correspondence* is a pair (i, j) where graph i belongs to class A,
graph j belongs to class B, and their binary feature vectors are
identical on every selected feature.  The quality function

    q(U) = -(number of correspondences under feature set U)

is proven submodular, so greedy forward selection achieves at least
(1 - 1/e) ≈ 63 % of the optimal quality (Nemhauser et al., 1978).

Backends
--------
The greedy scan costs O(iterations x n_features x n_graphs) elementwise work.
Two implementations of that scan are provided; they compute the *same*
quantity and differ only in how it is evaluated:

    "python"  The reference implementation (`_fit_python`), expressed with
              Python sets and scalar indexing exactly as written above. It is
              the clearest statement of the algorithm and stays here as the
              correctness oracle, but a mined pattern set of any real size makes
              it hours-slow: `X_a[i, f]` is a scalar numpy index, executed
              n_features x n_graphs times per iteration.

    "torch"   The same greedy selection with the candidate scan batched
              (`_fit_vectorized`). Groups are carried as an integer group-id
              vector instead of a dict of sets, so the per-group class-1 counts
              for *every* candidate feature at once become a single index_add
              reduction, and the correspondence reduction becomes elementwise
              arithmetic over an (n_groups x n_features) array. Runs on CUDA
              when available, otherwise on CPU torch (still far faster than the
              scalar path).

`backend="auto"` (the default) picks torch when it is importable and falls back
to the reference path otherwise.

Equivalence: both paths implement identical stopping rules, identical
correspondence bookkeeping and identical group splitting. The only behavioural
difference is TIE-BREAKING — when two candidate features eliminate exactly the
same number of correspondences, the reference path keeps whichever the Python
set happened to yield first, while the vectorised path takes the lowest feature
index. Selections can therefore differ on an exact tie (both are equally valid
greedy choices; the submodular guarantee is unaffected).
"""

import numpy as np
from collections import defaultdict


VALID_BACKENDS = ("auto", "torch", "python")


class CORK:
    """
    Greedy forward feature selection using the CORK quality criterion.

    Parameters
    ----------
    tolerance : int
        Stop early when the remaining correspondence count drops below
        this threshold.  Matches the tolerance parameter *t* in the paper
        (Section 3.5).  Default 0 means "resolve every correspondence
        possible".
    max_features : int or None
        Hard cap on the number of selected features.  None = no cap
        (selection stops only when no candidate reduces correspondences
        or *tolerance* is reached).
    verbose : bool
        Print progress per iteration.
    backend : {"auto", "torch", "python"}
        How to evaluate the candidate scan.  See the module docstring.
        "auto" uses torch (CUDA if present) when importable.
    feature_chunk : int
        Number of candidate features scored per batch in the vectorised
        backend.  Caps peak memory at roughly
        ``n_groups * feature_chunk * 8`` bytes; lower it if a very large
        pattern set exhausts VRAM.
    """

    def __init__(self, tolerance=0, max_features=None, verbose=False,
                 backend="auto", feature_chunk=4096):
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {VALID_BACKENDS}"
            )
        self.tolerance = tolerance
        self.max_features = max_features
        self.verbose = verbose
        self.backend = backend
        self.feature_chunk = feature_chunk

        # Populated after fit()
        self.selected_indices_ = []
        self.correspondence_trace_ = []  # correspondences remaining per iteration
        self.backend_used_ = None        # provenance: which path actually ran

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Run CORK greedy forward selection.

        Parameters
        ----------
        X : np.ndarray, shape (n_graphs, n_features)
            Binary indicator matrix (1 if graph contains subgraph, else 0).
        y : np.ndarray, shape (n_graphs,)
            Class labels.  Exactly two distinct values expected.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CORK requires exactly 2 classes, got {len(classes)}: {classes}"
            )

        # Partition graph indices by class
        mask_a = (y == classes[0])
        mask_b = (y == classes[1])
        X_a = X[mask_a]  # (|A|, n_features)
        X_b = X[mask_b]  # (|B|, n_features)

        backend, device = self._resolve_backend()
        self.backend_used_ = backend

        if self.verbose:
            where = f"{backend}" + (f":{device}" if device is not None else "")
            print(f"[CORK] |A|={X_a.shape[0]}, |B|={X_b.shape[0]}, "
                  f"features={X.shape[1]}, "
                  f"initial correspondences={X_a.shape[0] * X_b.shape[0]}, "
                  f"backend={where}")

        if backend == "python":
            return self._fit_python(X_a, X_b, X.shape[1])
        return self._fit_vectorized(X_a, X_b, device)

    def transform(self, X):
        """
        Reduce a binary indicator matrix to the selected feature columns.

        Parameters
        ----------
        X : np.ndarray, shape (n_graphs, n_features)

        Returns
        -------
        np.ndarray, shape (n_graphs, n_selected_features)
        """
        if not self.selected_indices_:
            raise RuntimeError("CORK has not been fit yet or selected 0 features.")
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(self, X, y):
        """Fit and transform in one call."""
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self):
        """-> (backend_name, torch device or None)."""
        if self.backend == "python":
            return "python", None

        try:
            import torch
        except ImportError:
            if self.backend == "torch":
                raise RuntimeError(
                    "backend='torch' was requested but PyTorch is not installed."
                )
            return "python", None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return "torch", device

    # ------------------------------------------------------------------
    # Vectorised backend
    # ------------------------------------------------------------------

    def _fit_vectorized(self, X_a, X_b, device):
        """Batched equivalent of `_fit_python`.

        The dict-of-sets group structure is replaced by two group-id vectors
        (`ga`, `gb`), one entry per training graph. Two graphs share a group iff
        they agree on every selected feature so far — the same invariant the
        dict keys encode — so the correspondence count within a group stays
        |A_group| * |B_group| and the reduction formula is unchanged.
        """
        import torch

        n_a, n_features = X_a.shape
        n_b = X_b.shape[0]

        A = torch.from_numpy(np.ascontiguousarray(X_a)).to(device)
        B = torch.from_numpy(np.ascontiguousarray(X_b)).to(device)

        # One group holding everyone, matching the reference path's `{(): ...}`.
        ga = torch.zeros(n_a, dtype=torch.int64, device=device)
        gb = torch.zeros(n_b, dtype=torch.int64, device=device)
        n_groups = 1

        total_corr = n_a * n_b
        self.selected_indices_ = []
        self.correspondence_trace_ = [total_corr]

        # Already-selected features are excluded from the candidate set. They
        # would score 0 anyway (a group is by construction constant on them),
        # but masking keeps that guarantee independent of the arithmetic.
        selected_mask = np.zeros(n_features, dtype=bool)

        iteration = 0
        while True:
            # Stopping conditions — identical to the reference path.
            if total_corr <= self.tolerance:
                if self.verbose:
                    print(f"[CORK] Stopped: correspondences ({total_corr}) "
                          f"<= tolerance ({self.tolerance})")
                break

            if self.max_features and len(self.selected_indices_) >= self.max_features:
                if self.verbose:
                    print(f"[CORK] Stopped: reached max_features "
                          f"({self.max_features})")
                break

            if selected_mask.all():
                if self.verbose:
                    print("[CORK] Stopped: no more candidate features")
                break

            reductions = self._candidate_reductions(A, B, ga, gb, n_groups)
            reductions[selected_mask] = 0

            # np.argmax breaks ties towards the lowest feature index (see the
            # module docstring); the reference path breaks them by set order.
            best_feature = int(np.argmax(reductions))
            best_reduction = int(reductions[best_feature])

            if best_reduction <= 0:
                if self.verbose:
                    print("[CORK] Stopped: no candidate reduces correspondences")
                break

            # --- Accept best feature ---
            self.selected_indices_.append(best_feature)
            selected_mask[best_feature] = True
            ga, gb, n_groups = self._split_group_ids(A, B, ga, gb, best_feature)
            total_corr -= best_reduction
            self.correspondence_trace_.append(total_corr)
            iteration += 1

            if self.verbose:
                print(f"  iter {iteration}: selected feature {best_feature}, "
                      f"eliminated {best_reduction} correspondences, "
                      f"{total_corr} remaining")

        if self.verbose:
            print(f"[CORK] Done. Selected {len(self.selected_indices_)} features. "
                  f"Final correspondences: {total_corr}")

        return self

    def _candidate_reductions(self, A, B, ga, gb, n_groups):
        """Correspondences eliminated by EVERY candidate feature, in one sweep.

        Per group and feature f this is the reference path's

            a0 * b1 + a1 * b0

        with a1/b1 the number of class-A/B members of the group having f = 1.
        Those counts are a segment-sum over the group-id vector, i.e. one
        index_add per class per feature chunk.

        Returns a host-side int64 array of length n_features.
        """
        import torch

        n_features = A.shape[1]
        sizes_a = torch.bincount(ga, minlength=n_groups)   # int64, (n_groups,)
        sizes_b = torch.bincount(gb, minlength=n_groups)
        sizes_a = sizes_a.unsqueeze(1)
        sizes_b = sizes_b.unsqueeze(1)

        out = np.zeros(n_features, dtype=np.int64)

        # Chunked over features so peak memory is bounded regardless of how many
        # patterns gSpan mined.
        step = max(1, int(self.feature_chunk))
        for start in range(0, n_features, step):
            stop = min(start + step, n_features)
            width = stop - start

            # float32 accumulation is EXACT here: every partial sum is an
            # integer count bounded by the class size, and any integer below
            # 2**24 is exactly representable. Chosen over an integer dtype
            # because float index_add_ is the universally supported kernel.
            a1 = torch.zeros((n_groups, width), dtype=torch.float32, device=A.device)
            a1.index_add_(0, ga, A[:, start:stop].to(torch.float32))
            b1 = torch.zeros((n_groups, width), dtype=torch.float32, device=B.device)
            b1.index_add_(0, gb, B[:, start:stop].to(torch.float32))

            # Products reach |A| * |B| and their sum can exceed float32's exact
            # range, so the arithmetic itself is done in int64.
            a1 = a1.to(torch.int64)
            b1 = b1.to(torch.int64)
            a0 = sizes_a - a1
            b0 = sizes_b - b1
            out[start:stop] = (a0 * b1 + a1 * b0).sum(dim=0).cpu().numpy()

        return out

    @staticmethod
    def _split_group_ids(A, B, ga, gb, f):
        """Split every group on feature *f*, returning compact new group ids.

        The equivalent of `_split_groups`: the reference path extends each dict
        key with the feature's value, here the group id is extended the same way
        (`g * 2 + value`) and then relabelled to a contiguous range. Relabelling
        over the ids actually observed means empty groups are dropped, matching
        the reference path's `if a0 or b0` guard.
        """
        import torch

        key_a = ga * 2 + A[:, f].to(torch.int64)
        key_b = gb * 2 + B[:, f].to(torch.int64)

        combined = torch.cat((key_a, key_b))
        uniq, inverse = torch.unique(combined, return_inverse=True)
        n_groups = int(uniq.numel())

        return inverse[:key_a.numel()], inverse[key_a.numel():], n_groups

    # ------------------------------------------------------------------
    # Reference backend (correctness oracle)
    # ------------------------------------------------------------------

    def _fit_python(self, X_a, X_b, n_features):
        """The original set-based greedy loop, kept verbatim as the oracle."""
        all_feature_indices = set(range(n_features))

        # --- Initial state: no features selected ---
        # Every A-B pair is a correspondence: |A| * |B| total.
        # We maintain *groups* of graphs that share the same selected-feature
        # vector.  Initially one group containing everyone.
        #
        # Representation:
        #   groups : dict[tuple -> (set_of_A_indices, set_of_B_indices)]
        # The key is the tuple of feature values for the selected features.
        # Within each group, #correspondences = |A_group| * |B_group|.

        # Start: single group, key = empty tuple
        groups = {
            (): (set(range(X_a.shape[0])), set(range(X_b.shape[0])))
        }

        total_corr = X_a.shape[0] * X_b.shape[0]
        self.selected_indices_ = []
        self.correspondence_trace_ = [total_corr]

        iteration = 0
        while True:
            # Check stopping conditions
            if total_corr <= self.tolerance:
                if self.verbose:
                    print(f"[CORK] Stopped: correspondences ({total_corr}) "
                          f"<= tolerance ({self.tolerance})")
                break

            if self.max_features and len(self.selected_indices_) >= self.max_features:
                if self.verbose:
                    print(f"[CORK] Stopped: reached max_features "
                          f"({self.max_features})")
                break

            candidates = all_feature_indices - set(self.selected_indices_)
            if not candidates:
                if self.verbose:
                    print("[CORK] Stopped: no more candidate features")
                break

            # --- Evaluate every candidate feature ---
            best_feature = None
            best_reduction = 0

            for f in candidates:
                reduction = self._evaluate_candidate(
                    f, groups, X_a, X_b
                )
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = f

            if best_feature is None or best_reduction == 0:
                if self.verbose:
                    print("[CORK] Stopped: no candidate reduces correspondences")
                break

            # --- Accept best feature ---
            self.selected_indices_.append(best_feature)
            groups = self._split_groups(best_feature, groups, X_a, X_b)
            total_corr -= best_reduction
            self.correspondence_trace_.append(total_corr)
            iteration += 1

            if self.verbose:
                print(f"  iter {iteration}: selected feature {best_feature}, "
                      f"eliminated {best_reduction} correspondences, "
                      f"{total_corr} remaining")

        if self.verbose:
            print(f"[CORK] Done. Selected {len(self.selected_indices_)} features. "
                  f"Final correspondences: {total_corr}")

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_candidate(f, groups, X_a, X_b):
        """
        Compute how many correspondences feature *f* would eliminate.

        For each group, splitting on feature f produces two sub-groups
        (value=0 and value=1).  The reduction is:

            old_corr - new_corr
            = |A|*|B| - (|A0|*|B0| + |A1|*|B1|)
            = |A0|*|B1| + |A1|*|B0|

        where A0/B0 have f=0 and A1/B1 have f=1 within that group.
        """
        total_reduction = 0

        for key, (a_set, b_set) in groups.items():
            if not a_set or not b_set:
                continue  # no correspondences in this group

            # Count how many A-graphs and B-graphs in this group have f=1
            a1 = sum(1 for i in a_set if X_a[i, f] == 1)
            b1 = sum(1 for i in b_set if X_b[i, f] == 1)
            a0 = len(a_set) - a1
            b0 = len(b_set) - b1

            total_reduction += a0 * b1 + a1 * b0

        return total_reduction

    @staticmethod
    def _split_groups(f, groups, X_a, X_b):
        """
        Split every existing group on feature *f*, producing new groups
        keyed by the extended feature-value tuple.
        """
        new_groups = {}

        for key, (a_set, b_set) in groups.items():
            # Split A-indices
            a0 = {i for i in a_set if X_a[i, f] == 0}
            a1 = a_set - a0
            # Split B-indices
            b0 = {i for i in b_set if X_b[i, f] == 0}
            b1 = b_set - b0

            key0 = key + (0,)
            key1 = key + (1,)

            if a0 or b0:
                new_groups[key0] = (a0, b0)
            if a1 or b1:
                new_groups[key1] = (a1, b1)

        return new_groups
