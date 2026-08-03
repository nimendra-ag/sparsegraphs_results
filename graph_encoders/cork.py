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
"""

import numpy as np
from collections import defaultdict


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
    """

    def __init__(self, tolerance=0, max_features=None, verbose=False):
        self.tolerance = tolerance
        self.max_features = max_features
        self.verbose = verbose

        # Populated after fit()
        self.selected_indices_ = []
        self.correspondence_trace_ = []  # correspondences remaining per iteration

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

        n_features = X.shape[1]
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

        if self.verbose:
            print(f"[CORK] |A|={X_a.shape[0]}, |B|={X_b.shape[0]}, "
                  f"features={n_features}, initial correspondences={total_corr}")

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
