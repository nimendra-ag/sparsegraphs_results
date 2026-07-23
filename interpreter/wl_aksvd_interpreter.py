"""
interpreter/wl_aksvd_interpreter.py
=====================================
Multi-level model interpretability for the WL + ApproximateKSVD pipeline.

Levels implemented
------------------
1  Prediction Explanation     – which dictionary atoms drove the decision
2  Dictionary Atom Analysis   – top WL features per atom + per-class statistics
4  Similar Compound Retrieval – cosine-similarity search in sparse-code space
5  Contribution Breakdown     – ASCII bar chart (SHAP-style)
6  WL Feature Statistics      – class-level prevalence of individual WL tokens

Why this pipeline is naturally interpretable
--------------------------------------------
GNN explanation requires an additional approximation layer (GNNExplainer).
Here every step is transparent:

  Graph → WL subtrees (vocab) → AKSVD sparse code → classifier
             ↑                        ↑                   ↑
         countable tokens       discrete atoms      coef_ / importances_

No approximation is needed; the explanation IS the computation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WLAKSVDInterpreter:
    """
    Multi-level interpretability for the WL + ApproximateKSVD graph
    classification pipeline.

    Parameters
    ----------
    wl
        Fitted WL instance.  Must expose:
          - .vocab         : List[Tuple[str, float]]
          - .class_df      : Dict[class_label, Counter{token: doc_freq}]
          - .class_counts  : Counter{class_label: n_docs}
    aksvd
        Fitted AKSVD instance.  Must expose:
          - ._dictionary : np.ndarray  shape (n_atoms, n_vocab_features)
    classifier
        Fitted sklearn-compatible classifier.
        Best attribution support  : LogisticRegression, LinearSVC  (.coef_)
        Partial attribution support: RandomForest, GradientBoosting (.feature_importances_)
    scaler
        Fitted sklearn scaler (e.g. MaxAbsScaler) used before the classifier.
    training_graphs
        Graphs used to train the ML model (G_ML_train).
        Stored for Level 4 compound retrieval.
    training_labels
        Labels corresponding to training_graphs (y_ML_train).
    training_sparse_codes
        **Unscaled** AKSVD sparse codes for training_graphs (X_ML_train).
        Used as the reference database for cosine-similarity search (Level 4).
    label_map
        Optional human-readable mapping, e.g. {0: "Non-Cancerous", 1: "Cancerous"}.
    """

    def __init__(
        self,
        wl: Any,
        aksvd: Any,
        classifier: Any,
        scaler: Any,
        training_graphs: List,
        training_labels: List,
        training_sparse_codes: np.ndarray,
        label_map: Optional[Dict[Any, str]] = None,
    ) -> None:
        _required = ("vocab", "class_df", "class_counts")
        missing = [attr for attr in _required if not hasattr(wl, attr)]
        if missing:
            raise AttributeError(
                f"WL instance is missing required attributes: {missing}. "
                "Ensure you are using the updated WL class that stores "
                "class_df and class_counts inside create_vocab()."
            )

        self.wl = wl
        self.aksvd = aksvd
        self.classifier = classifier
        self.scaler = scaler
        self.training_graphs = training_graphs
        self.training_labels = list(training_labels)
        self.training_sparse_codes = training_sparse_codes
        self.label_map = label_map or {}

        # Derived references – share memory with the fitted WL, no copies
        self.vocab: List[Tuple[str, float]] = wl.vocab
        self.vocab_words: List[str] = [w for w, _ in self.vocab]
        self.vocab_scores: Dict[str, float] = {w: s for w, s in self.vocab}
        self.dictionary: np.ndarray = aksvd._dictionary  # (n_atoms, n_vocab)
        self.class_df: Dict = wl.class_df
        self.class_counts: Counter = wl.class_counts
        self.unique_classes: List = sorted(set(training_labels))

    # Private helpers

    def _label_name(self, label: Any) -> str:
        """Return human-readable class name from label_map, or str(label)."""
        return self.label_map.get(label, str(label))

    def _embed_graph(self, graph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute WL bag-of-features and AKSVD sparse code for a single graph.

        Returns
        -------
        wl_embedding  : np.ndarray  shape (1, n_vocab_features)
        sparse_code   : np.ndarray  shape (1, n_atoms)

        Notes
        -----
        ApproximateKSVD.transform() squeezes single-sample output to 1D.
        We always reshape to (1, n_atoms) so every caller receives a
        consistent 2D array and sklearn's scaler/predict accept it without
        manual reshaping at the call site.
        """
        wl_emb = self.wl.generate_inferencing_embeddings([graph])
        sparse_code = self.aksvd.infer(wl_emb)

        # Guard against ApproximateKSVD squeezing single-sample output to 1D
        if sparse_code.ndim == 1:
            sparse_code = sparse_code.reshape(1, -1)

        return wl_emb, sparse_code

    def _atom_contributions(
        self, scaled_code: np.ndarray, predicted_class: Any
    ) -> np.ndarray:
        """
        Return a signed per-atom contribution vector aligned to *predicted_class*.

        Attribution strategy (in priority order)
        -----------------------------------------
        1. coef_                – LogisticRegression, LinearSVC
           contribution_k = scaled_code_k × coef[predicted_class][k]
        2. feature_importances_ – RandomForest, GradientBoosting
           contribution_k = scaled_code_k × global_importance_k  (non-negative)
        3. Raw activation       – fallback when neither attribute exists
        """
        if hasattr(self.classifier, "coef_"):
            coef = self.classifier.coef_
            if coef.shape[0] == 1:
                # Binary LR: single row represents the positive class
                sign = (
                    1.0 if predicted_class == self.classifier.classes_[1] else -1.0
                )
                return sign * scaled_code[0] * coef[0]

            cls_idx = list(self.classifier.classes_).index(predicted_class)
            return scaled_code[0] * coef[cls_idx]

        if hasattr(self.classifier, "feature_importances_"):
            return scaled_code[0] * self.classifier.feature_importances_

        # Fallback: magnitude of activation (direction unknown)
        return np.abs(scaled_code[0])

    def _predict_with_proba(
        self, scaled_code: np.ndarray
    ) -> Tuple[Any, Optional[float]]:
        """
        Return (prediction, confidence).
        confidence is None when predict_proba is unavailable (e.g. LinearSVC).
        """
        pred = self.classifier.predict(scaled_code)[0]
        try:
            proba = self.classifier.predict_proba(scaled_code)[0]
            return pred, float(proba[list(self.classifier.classes_).index(pred)])
        except AttributeError:
            return pred, None

    @staticmethod
    def _ascii_bar(pct: float, width: int = 30) -> str:
        filled = max(0, min(round(pct / 100 * width), width))
        return "█" * filled + "░" * (width - filled)

    def _build_explanation(
        self,
        sparse_code: np.ndarray,
        scaled_code: np.ndarray,
        top_k_atoms: int,
    ) -> Dict:
        """
        Internal helper that builds the core explanation dict.
        Called by both explain_prediction() and full_report() so that
        embeddings are never recomputed unnecessarily.
        """
        prediction, confidence = self._predict_with_proba(scaled_code)
        contribs = self._atom_contributions(scaled_code, prediction)

        active_mask = sparse_code[0] != 0
        contribs_active = np.where(active_mask, contribs, 0.0)
        total_abs = float(np.sum(np.abs(contribs_active)))

        top_idx = np.argsort(np.abs(contribs_active))[::-1][:top_k_atoms]
        top_atoms = [
            {
                "atom_idx": int(i),
                "raw_contribution": float(contribs_active[i]),
                "percentage": round(
                    abs(contribs_active[i]) / total_abs * 100, 2
                ) if total_abs > 0 else 0.0,
                "direction": (
                    "supporting" if contribs_active[i] >= 0 else "opposing"
                ),
                "activation": float(sparse_code[0][i]),
            }
            for i in top_idx
            if contribs_active[i] != 0.0
        ]

        return {
            "prediction": prediction,
            "prediction_label": self._label_name(prediction),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "n_active_atoms": int(np.sum(active_mask)),
            "top_atoms": top_atoms,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Level 1 – Prediction Explanation
    # ─────────────────────────────────────────────────────────────────────────

    def explain_prediction(self, graph, top_k_atoms: int = 5) -> Dict:
        """
        Level 1 – Predict the class and rank the dictionary atoms that
        contributed most to that decision.

        Returns
        -------
        dict
            prediction        : raw class label
            prediction_label  : human-readable name
            confidence        : float in [0, 1] or None
            n_active_atoms    : number of non-zero atoms in the sparse code
            top_atoms         : list[dict], each with keys:
                atom_idx, raw_contribution, percentage, direction, activation
        """
        _, sparse_code = self._embed_graph(graph)
        scaled = self.scaler.transform(sparse_code)
        return self._build_explanation(sparse_code, scaled, top_k_atoms)

    # ─────────────────────────────────────────────────────────────────────────
    # Level 2 – Dictionary Atom Composition
    # ─────────────────────────────────────────────────────────────────────────

    def explain_dictionary_atom(
        self, atom_idx: int, top_k_features: int = 10
    ) -> Dict:
        """
        Level 2 – Decompose one dictionary atom into its highest-weight WL
        features, each annotated with per-class prevalence statistics.

        Returns
        -------
        dict
            atom_idx             : int
            n_nonzero_features   : number of WL features with non-zero weight
            sparsity_ratio       : fraction of features that are zero
            top_features         : list[dict], each with keys:
                feature_id, wl_token, atom_weight, discriminative_score,
                class_prevalence_pct
        """
        weights = self.dictionary[atom_idx]
        top_fi = np.argsort(np.abs(weights))[::-1][:top_k_features]

        features = []
        for fi in top_fi:
            word = self.vocab_words[fi]
            prevalence = {
                self._label_name(cls): round(
                    self.class_df[cls].get(word, 0)
                    / max(self.class_counts[cls], 1) * 100,
                    2,
                )
                for cls in self.unique_classes
            }
            features.append(
                {
                    "feature_id": int(fi),
                    "wl_token": word,
                    "atom_weight": round(float(weights[fi]), 6),
                    "discriminative_score": round(
                        float(self.vocab_scores.get(word, 0.0)), 6
                    ),
                    "class_prevalence_pct": prevalence,
                }
            )

        n_nonzero = int(np.count_nonzero(weights))
        return {
            "atom_idx": atom_idx,
            "n_nonzero_features": n_nonzero,
            "sparsity_ratio": round(1.0 - n_nonzero / len(weights), 4),
            "top_features": features,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Level 4 – Similar Training Compounds
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_compounds(self, graph, top_k: int = 5) -> List[Dict]:
        """
        Level 4 – Retrieve the closest training compounds by cosine similarity
        over unscaled AKSVD sparse codes.

        Similarity is computed in sparse-code space because that is the
        representation the classifier actually uses, making retrieved neighbours
        directly relevant to the prediction.

        Returns
        -------
        list[dict]
            rank, train_index, label, label_name, similarity_pct
        """
        _, sparse_code = self._embed_graph(graph)
        sims = cosine_similarity(sparse_code, self.training_sparse_codes)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "train_index": int(i),
                "label": self.training_labels[i],
                "label_name": self._label_name(self.training_labels[i]),
                "similarity_pct": round(float(sims[i]) * 100, 2),
            }
            for rank, i in enumerate(top_idx, start=1)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Level 5 – ASCII Contribution Breakdown
    # ─────────────────────────────────────────────────────────────────────────

    def contribution_breakdown(
        self, explanation: Dict, bar_width: int = 30
    ) -> str:
        """
        Level 5 – Render the dictionary atom contributions as an ASCII bar
        chart (SHAP-style).  Accepts the dict returned by explain_prediction().

        [S] = atom is supporting the predicted class
        [O] = atom is opposing the predicted class (pushes toward another class)
        """
        atoms = explanation["top_atoms"]
        shown_pct = sum(a["percentage"] for a in atoms)
        other_pct = max(0.0, 100.0 - shown_pct)
        sep = "═" * 65

        lines = [
            sep,
            f"  PREDICTION   : {explanation['prediction_label']}",
            (
                f"  CONFIDENCE   : {explanation['confidence'] * 100:.1f}%"
                if explanation["confidence"] is not None
                else "  CONFIDENCE   : N/A  (model does not expose predict_proba)"
            ),
            f"  ACTIVE ATOMS : {explanation['n_active_atoms']}",
            sep,
            "  CONTRIBUTION BREAKDOWN          [S = Supporting | O = Opposing]",
            f"  {'Dictionary Atom':<26} {'Bar':<{bar_width + 2}} {'%':>6}",
            "─" * 65,
        ]

        for a in atoms:
            tag = "S" if a["direction"] == "supporting" else "O"
            label = f"Dict Atom {a['atom_idx']:>4d} [{tag}]"
            bar = self._ascii_bar(a["percentage"], bar_width)
            lines.append(f"  {label:<26} {bar}  {a['percentage']:>5.1f}%")

        if other_pct > 0.5:
            bar = self._ascii_bar(other_pct, bar_width)
            lines.append(f"  {'Others':<26} {bar}  {other_pct:>5.1f}%")

        lines.append(sep)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Level 6 – WL Feature Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def explain_wl_feature(self, feature_id: int) -> Dict:
        """
        Level 6 – Characterise a single WL feature token by its discriminative
        score and per-class prevalence statistics.

        Since WL tokens are structural hashes, there is no human-readable
        SMILES string to display.  Instead, we ground each feature in
        statistical language: how often it appears in each class, and how
        strongly it separates them.

        Returns
        -------
        dict
            feature_id           : int
            wl_token             : the raw WL hash string
            discriminative_score : Hellinger-based score from vocab creation
            class_statistics     : dict[class_name, {doc_frequency, prevalence_pct}]
            dominant_class       : class in which this feature appears most often
        """
        word = self.vocab_words[feature_id]
        class_stats = {
            self._label_name(cls): {
                "doc_frequency": int(self.class_df[cls].get(word, 0)),
                "prevalence_pct": round(
                    self.class_df[cls].get(word, 0)
                    / max(self.class_counts[cls], 1) * 100,
                    2,
                ),
            }
            for cls in self.unique_classes
        }
        dominant = max(
            class_stats, key=lambda c: class_stats[c]["prevalence_pct"]
        )
        return {
            "feature_id": feature_id,
            "wl_token": word,
            "discriminative_score": round(
                float(self.vocab_scores.get(word, 0.0)), 6
            ),
            "class_statistics": class_stats,
            "dominant_class": dominant,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Level 3 – Node-level attribution (WL token → graph node tracing)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_token_node_map(self, graph) -> Dict[str, List]:
        """
        Re-run WL hashing on a single graph and return {wl_token: [node_ids]}.

        The pipeline's create_wl_hash() calls get_graph_features(), which
        flattens all (node × iteration) tokens into one list and loses the
        node correspondence.  Here we read extracted_features *before* that
        flattening step:

            extracted_features = {
                iteration_0: [token_node_0, token_node_1, ...],
                iteration_1: [token_node_0, token_node_1, ...],
                ...
            }

        We call WeisfeilerLehmanHashing with the exact same parameters as the
        pipeline, so the token strings are byte-for-byte identical to those in
        the learned vocabulary.
        """
        from graph_encoders.wlkernalsubtree import WeisfeilerLehmanHashing

        g = self.wl._check_graph(graph)
        wl_hash = WeisfeilerLehmanHashing(
            g,
            self.wl.wl_iterations,
            self.wl.attributed,
            self.wl.erase_base_features,
        )

        nodes = list(g.nodes())
        token_node_map: Dict[str, List] = defaultdict(list)

        # extracted_features: {iter_idx: [token_for_node_0, token_for_node_1, ...]}
        for node_features in wl_hash.extracted_features.values():
            for node_pos, token in enumerate(node_features):
                token_str = str(token)
                node_id = nodes[node_pos]
                # A node may produce the same token at different iterations;
                # keep it listed only once per token.
                if node_id not in token_node_map[token_str]:
                    token_node_map[token_str].append(node_id)

        return dict(token_node_map)

    def get_node_importance(
        self,
        graph,
        top_k_atoms: int = 5,
    ) -> Dict:
        """
        Trace prediction contributions back to individual graph nodes.

        Attribution chain (no approximations)
        --------------------------------------
        prediction
          └─ Σ_k  atom_contribution(k)            ← sparse_code[k] × coef[k]
               └─ Σ_{t ∈ vocab ∩ tokens_at_v}  dictionary[k][t]
                                                   ← atom k's weight for token t

        For every node v in the graph:

            node_importance(v) = Σ_k { atom_contribution(k)
                                        × Σ_t { dictionary[k][t]
                                                 ∀ vocab token t that v generated } }

        This chains each term directly from the trained model — no GNNExplainer,
        no masking, no post-hoc approximation.

        Returns
        -------
        dict
            prediction          : str
            confidence          : float | None
            node_importance     : Dict[node_id, float]  – signed scores
            node_importance_pct : Dict[node_id, float]  – % of total |importance|
            sorted_nodes        : List[(node_id, float)] – ranked by |score|
            node_sources        : Dict[node_id, List[dict]] – full traceability
        """
        _, sparse_code = self._embed_graph(graph)
        scaled = self.scaler.transform(sparse_code)
        explanation = self._build_explanation(sparse_code, scaled, top_k_atoms)

        token_node_map = self._build_token_node_map(graph)
        vocab_index = {w: i for i, w in enumerate(self.vocab_words)}

        node_importance: Dict = defaultdict(float)
        node_sources: Dict = defaultdict(list)

        for atom_info in explanation["top_atoms"]:
            k = atom_info["atom_idx"]
            atom_contrib = atom_info["raw_contribution"]
            atom_weights = self.dictionary[k]   # shape (n_vocab,)

            for token_str, node_ids in token_node_map.items():
                if token_str not in vocab_index:
                    continue   # token not in learned vocabulary; skip

                feat_idx = vocab_index[token_str]
                w = float(atom_weights[feat_idx])
                if w == 0.0:
                    continue   # this atom does not use this WL feature; skip

                path_importance = atom_contrib * w

                for node_id in node_ids:
                    node_importance[node_id] += path_importance
                    node_sources[node_id].append({
                        "atom_idx": k,
                        "token": token_str,
                        "atom_contribution": round(atom_contrib, 6),
                        "atom_weight": round(w, 6),
                        "path_importance": round(path_importance, 6),
                    })

        sorted_nodes = sorted(
            node_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        total_abs = sum(abs(s) for _, s in sorted_nodes) or 1.0

        return {
            "prediction": explanation["prediction_label"],
            "confidence": explanation["confidence"],
            "node_importance": dict(node_importance),
            "node_importance_pct": {
                nid: round(abs(score) / total_abs * 100, 2)
                for nid, score in sorted_nodes
            },
            "sorted_nodes": sorted_nodes,
            "node_sources": dict(node_sources),
        }

    def node_highlight_report(
        self,
        graph,
        top_k_atoms: int = 5,
        top_n_nodes: Optional[int] = None,
    ) -> str:
        """
        Text report listing which nodes to highlight and why.

        For each node the report shows:
          • Importance score as a % of total attribution
          • Direction: [+] supports the predicted class, [-] opposes it
          • The node's graph attributes (atom type / feature vector)
          • Traceability: which dictionary atoms drove the importance score

        Parameters
        ----------
        graph        : single networkx graph to explain
        top_k_atoms  : number of dictionary atoms to trace through
        top_n_nodes  : cap the list to the N most important nodes
                       (default: all nodes that received any importance)
        """
        result = self.get_node_importance(graph, top_k_atoms=top_k_atoms)
        g = self.wl._check_graph(graph)

        sorted_nodes = result["sorted_nodes"]
        if top_n_nodes is not None:
            sorted_nodes = sorted_nodes[:top_n_nodes]

        total_abs = (
            sum(abs(s) for _, s in result["sorted_nodes"]) or 1.0
        )
        sep = "═" * 68

        lines = [
            sep,
            f"  PREDICTION   : {result['prediction']}",
            (
                f"  CONFIDENCE   : {result['confidence'] * 100:.1f}%"
                if result["confidence"] is not None
                else "  CONFIDENCE   : N/A"
            ),
            sep,
            "  NODE HIGHLIGHTING     [+] promotes prediction   [-] opposes it",
            f"  {'Node':<8} {'Feature / Attrs':<22} {'Importance':<30} {'%':>6}  Dir",
            "─" * 68,
        ]

        for node_id, score in sorted_nodes:
            attrs = dict(g.nodes[node_id])
            # Probe common molecular-graph attribute keys
            feat_val = attrs.get(
                "feature",
                attrs.get("label", attrs.get("x", list(attrs.values())[:1] or "?")),
            )
            feat_str = str(feat_val)
            if len(feat_str) > 20:
                feat_str = feat_str[:17] + "..."

            direction = "[+]" if score >= 0 else "[-]"
            pct = abs(score) / total_abs * 100
            bar = self._ascii_bar(pct, 28)
            lines.append(
                f"  {str(node_id):<8} {feat_str:<22} {bar}  {pct:>5.1f}%  {direction}"
            )

        # ── Traceability block ────────────────────────────────────────────────
        lines += [
            "─" * 68,
            "  TRACEABILITY: top nodes → source dictionary atoms",
            "─" * 68,
        ]
        for node_id, score in sorted_nodes[:min(5, len(sorted_nodes))]:
            sources = result["node_sources"].get(node_id, [])

            # Aggregate path_importance per atom
            atom_totals: Dict[int, float] = defaultdict(float)
            for rec in sources:
                atom_totals[rec["atom_idx"]] += rec["path_importance"]

            top_atoms = sorted(
                atom_totals.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]

            direction = "[+]" if score >= 0 else "[-]"
            atom_str = "  ".join(
                f"Dict Atom {k} ({v:+.4f})" for k, v in top_atoms
            )
            lines.append(
                f"  Node {node_id} {direction}  ←  {atom_str or 'no source'}"
            )

        lines.append(sep)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Full Report (combines all levels)
    # ─────────────────────────────────────────────────────────────────────────

    def full_report(
        self,
        graph,
        top_k_atoms: int = 5,
        top_k_features_per_atom: int = 5,
        top_k_similar: int = 5,
    ) -> str:
        """
        Generate the complete multi-level interpretability report for one graph.

        Embeddings are computed exactly once and reused across all levels,
        so this is no slower than calling explain_prediction() alone.

        Parameters
        ----------
        graph                  : a single networkx graph
        top_k_atoms            : atoms to show in Levels 1 & 5
        top_k_features_per_atom: WL features to show per atom in Levels 2 & 6
        top_k_similar          : training compounds to retrieve in Level 4

        Returns
        -------
        str  – formatted multi-section text report
        """
        # Compute embeddings once ─────────────────────────────────────────────
        _, sparse_code = self._embed_graph(graph)
        scaled = self.scaler.transform(sparse_code)
        explanation = self._build_explanation(sparse_code, scaled, top_k_atoms)

        lines: List[str] = [self.contribution_breakdown(explanation)]  # Level 5

        # ── Level 1 ───────────────────────────────────────────────────────────
        conf_str = (
            f"  (Confidence: {explanation['confidence'] * 100:.1f}%)"
            if explanation["confidence"] is not None else ""
        )
        lines += [
            "",
            "── LEVEL 1: Prediction Reasoning "
            "─────────────────────────────────────────",
            f"  Prediction : {explanation['prediction_label']}{conf_str}",
            "  Atoms driving this prediction:",
        ]
        for a in explanation["top_atoms"]:
            lines.append(
                f"    • Dict Atom {a['atom_idx']:>4d}"
                f"  {a['percentage']:>5.1f}%  [{a['direction']}]"
                f"  activation={a['activation']:.4f}"
            )

        # ── Level 2 ───────────────────────────────────────────────────────────
        lines += [
            "",
            "── LEVEL 2: Dictionary Atom Composition "
            "───────────────────────────────────",
        ]
        for a in explanation["top_atoms"]:
            info = self.explain_dictionary_atom(
                a["atom_idx"], top_k_features=top_k_features_per_atom
            )
            lines.append(
                f"\n  Dict Atom {a['atom_idx']:>4d}"
                f"  |  non-zero WL features: {info['n_nonzero_features']}"
                f"  |  sparsity: {info['sparsity_ratio']:.3f}"
            )
            for feat in info["top_features"]:
                prev = "  ".join(
                    f"{cls}: {pct}%"
                    for cls, pct in feat["class_prevalence_pct"].items()
                )
                lines.append(
                    f"    Feature {feat['feature_id']:>6d}"
                    f"  weight={feat['atom_weight']:>+.4f}"
                    f"  disc={feat['discriminative_score']:.4f}"
                    f"  prevalence [{prev}]"
                )

        # ── Level 4 ───────────────────────────────────────────────────────────
        # Reuse sparse_code computed above (no second embedding call)
        sims = cosine_similarity(sparse_code, self.training_sparse_codes)[0]
        top_sim_idx = np.argsort(sims)[::-1][:top_k_similar]

        lines += [
            "",
            "── LEVEL 4: Similar Training Compounds "
            "────────────────────────────────────",
        ]
        for rank, i in enumerate(top_sim_idx, start=1):
            lines.append(
                f"  #{rank}"
                f"  idx={i:>5d}"
                f"  label={self._label_name(self.training_labels[i]):<16}"
                f"  similarity={sims[i] * 100:.1f}%"
            )

        # ── Level 6 (from the most contributing atom) ─────────────────────────
        if explanation["top_atoms"]:
            top_atom = explanation["top_atoms"][0]
            info = self.explain_dictionary_atom(
                top_atom["atom_idx"], top_k_features=top_k_features_per_atom
            )
            lines += [
                "",
                f"── LEVEL 6: WL Feature Statistics"
                f"  (top atom: Dict Atom {top_atom['atom_idx']})"
                f" ────────────────────────",
            ]
            for feat in info["top_features"]:
                fi = self.explain_wl_feature(feat["feature_id"])
                lines += [
                    f"\n  WL Feature {feat['feature_id']:>6d}"
                    f"  token = {fi['wl_token']}",
                    f"    Discriminative score : {fi['discriminative_score']:.6f}",
                    f"    Dominant class       : {fi['dominant_class']}",
                ]
                for cls_name, stats in fi["class_statistics"].items():
                    lines.append(
                        f"    [{cls_name:<16}]"
                        f"  present in {stats['prevalence_pct']:.1f}%"
                        f" of compounds  ({stats['doc_frequency']} docs)"
                    )

        # ── Level 3: Node Highlighting ────────────────────────────────────────
        lines += [
            "",
            "── LEVEL 3: Node Highlighting ──────────────────────────────────────────",
        ]
        lines.append(self.node_highlight_report(graph, top_k_atoms=top_k_atoms))

        return "\n".join(lines)