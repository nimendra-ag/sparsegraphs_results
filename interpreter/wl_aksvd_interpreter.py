"""
interpreter/wl_aksvd_interpreter.py
=====================================
Multi-level model interpretability for the WL + ApproximateKSVD pipeline.

Levels
------
1  Prediction Explanation   – which dictionary atoms drove the decision
2  Dictionary Atom Analysis – top WL features per atom + per-class statistics
5  Contribution Breakdown   – ASCII bar chart (SHAP-style)

Token-level and substructure highlighting is handled by the notebook's
visualise_subtrees / visualise_prediction functions via get_node_importance().
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class WLAKSVDInterpreter:

    def __init__(
        self,
        wl: Any,
        aksvd: Any,
        classifier: Any,
        scaler: Any,
        training_graphs: List = None,
        training_labels: List = None,
        training_sparse_codes: np.ndarray = None,
        label_map: Optional[Dict[Any, str]] = None,
    ) -> None:
        _required = ("vocab", "class_df", "class_counts")
        missing = [a for a in _required if not hasattr(wl, a)]
        if missing:
            raise AttributeError(
                f"WL instance is missing: {missing}. "
                "Ensure the updated WL class is used."
            )

        self.wl = wl
        self.aksvd = aksvd
        self.classifier = classifier
        self.scaler = scaler
        self.label_map = label_map or {}

        self.vocab: List[Tuple[str, float]] = wl.vocab
        self.vocab_words: List[str] = [w for w, _ in self.vocab]
        self.vocab_scores: Dict[str, float] = {w: s for w, s in self.vocab}
        self.dictionary: np.ndarray = aksvd._dictionary
        self.class_df: Dict = wl.class_df
        self.class_counts: Counter = wl.class_counts
        self.unique_classes: List = sorted(set(training_labels)) if training_labels else []

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _label_name(self, label: Any) -> str:
        return self.label_map.get(label, str(label))

    def _embed_graph(self, graph) -> Tuple[np.ndarray, np.ndarray]:
        wl_emb = self.wl.generate_inferencing_embeddings([graph])
        sparse_code = self.aksvd.infer(wl_emb)
        if sparse_code.ndim == 1:
            sparse_code = sparse_code.reshape(1, -1)
        return wl_emb, sparse_code

    def _atom_contributions(self, scaled_code: np.ndarray, predicted_class: Any) -> np.ndarray:
        if hasattr(self.classifier, "coef_"):
            coef = self.classifier.coef_
            if coef.shape[0] == 1:
                sign = 1.0 if predicted_class == self.classifier.classes_[1] else -1.0
                return sign * scaled_code[0] * coef[0]
            cls_idx = list(self.classifier.classes_).index(predicted_class)
            return scaled_code[0] * coef[cls_idx]
        if hasattr(self.classifier, "feature_importances_"):
            return scaled_code[0] * self.classifier.feature_importances_
        return np.abs(scaled_code[0])

    def _predict_with_proba(self, scaled_code: np.ndarray) -> Tuple[Any, Optional[float]]:
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
        self, sparse_code: np.ndarray, scaled_code: np.ndarray, top_k_atoms: int
    ) -> Dict:
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
                "percentage": round(abs(contribs_active[i]) / total_abs * 100, 2)
                              if total_abs > 0 else 0.0,
                "direction": "supporting" if contribs_active[i] >= 0 else "opposing",
                "activation": float(sparse_code[0][i]),
            }
            for i in top_idx if contribs_active[i] != 0.0
        ]
        return {
            "prediction": prediction,
            "prediction_label": self._label_name(prediction),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "n_active_atoms": int(np.sum(active_mask)),
            "top_atoms": top_atoms,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Level 1
    # ─────────────────────────────────────────────────────────────────────────

    def explain_prediction(self, graph, top_k_atoms: int = 5) -> Dict:
        _, sparse_code = self._embed_graph(graph)
        scaled = self.scaler.transform(sparse_code)
        return self._build_explanation(sparse_code, scaled, top_k_atoms)

    # ─────────────────────────────────────────────────────────────────────────
    # Level 2
    # ─────────────────────────────────────────────────────────────────────────

    def explain_dictionary_atom(self, atom_idx: int, top_k_features: int = 10) -> Dict:
        weights = self.dictionary[atom_idx]
        top_fi = np.argsort(np.abs(weights))[::-1][:top_k_features]
        features = []
        for fi in top_fi:
            word = self.vocab_words[fi]
            prevalence = {
                self._label_name(cls): round(
                    self.class_df[cls].get(word, 0)
                    / max(self.class_counts[cls], 1) * 100, 2
                )
                for cls in self.unique_classes
            }
            features.append({
                "feature_id": int(fi),
                "wl_token": word,
                "atom_weight": round(float(weights[fi]), 6),
                "discriminative_score": round(float(self.vocab_scores.get(word, 0.0)), 6),
                "class_prevalence_pct": prevalence,
            })
        n_nonzero = int(np.count_nonzero(weights))
        return {
            "atom_idx": atom_idx,
            "n_nonzero_features": n_nonzero,
            "sparsity_ratio": round(1.0 - n_nonzero / len(weights), 4),
            "top_features": features,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Level 5
    # ─────────────────────────────────────────────────────────────────────────

    def contribution_breakdown(self, explanation: Dict, bar_width: int = 30) -> str:
        atoms = explanation["top_atoms"]
        shown_pct = sum(a["percentage"] for a in atoms)
        other_pct = max(0.0, 100.0 - shown_pct)
        sep = "=" * 65
        lines = [
            sep,
            f"  PREDICTION   : {explanation['prediction_label']}",
            (f"  CONFIDENCE   : {explanation['confidence'] * 100:.1f}%"
             if explanation["confidence"] is not None
             else "  CONFIDENCE   : N/A"),
            f"  ACTIVE ATOMS : {explanation['n_active_atoms']}",
            sep,
            "  CONTRIBUTION BREAKDOWN          [S = Supporting | O = Opposing]",
            f"  {'Dictionary Atom':<26} {'Bar':<{bar_width + 2}} {'%':>6}",
            "-" * 65,
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
    # explain_wl_feature – used by notebook tables
    # ─────────────────────────────────────────────────────────────────────────

    def explain_wl_feature(self, feature_id: int) -> Dict:
        word = self.vocab_words[feature_id]
        class_stats = {
            self._label_name(cls): {
                "doc_frequency": int(self.class_df[cls].get(word, 0)),
                "prevalence_pct": round(
                    self.class_df[cls].get(word, 0)
                    / max(self.class_counts[cls], 1) * 100, 2
                ),
            }
            for cls in self.unique_classes
        }
        dominant = max(class_stats, key=lambda c: class_stats[c]["prevalence_pct"])
        return {
            "feature_id": feature_id,
            "wl_token": word,
            "discriminative_score": round(float(self.vocab_scores.get(word, 0.0)), 6),
            "class_statistics": class_stats,
            "dominant_class": dominant,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # get_node_importance – feeds visualise_subtrees / visualise_prediction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_token_node_map(self, graph) -> Dict[str, List]:
        """Re-run WL hashing and return {token_hash: [node_ids]}."""
        from graph_encoders.wlkernalsubtree import WeisfeilerLehmanHashing
        g = self.wl._check_graph(graph)
        wl_hash = WeisfeilerLehmanHashing(
            g, self.wl.wl_iterations, self.wl.attributed, self.wl.erase_base_features
        )
        nodes = list(g.nodes())
        token_node_map: Dict[str, List] = defaultdict(list)
        for node_features in wl_hash.extracted_features.values():
            for node_pos, token in enumerate(node_features):
                token_str = str(token)
                node_id = nodes[node_pos]
                if node_id not in token_node_map[token_str]:
                    token_node_map[token_str].append(node_id)
        return dict(token_node_map)

    def get_node_importance(self, graph, top_k_atoms: int = 5) -> Dict:
        """
        Compute per-node importance (Scores A and B) and the node_sources
        traceability table consumed by the notebook visualisation functions.

        Score A  –  atom_contribution x dictionary_weight
        Score B  –  atom_contribution x dictionary_weight x wl_feature_count
        """
        wl_emb, sparse_code = self._embed_graph(graph)
        wl_vec = wl_emb[0]
        scaled = self.scaler.transform(sparse_code)
        explanation = self._build_explanation(sparse_code, scaled, top_k_atoms)

        token_node_map = self._build_token_node_map(graph)
        vocab_index = {w: i for i, w in enumerate(self.vocab_words)}

        node_imp_a: Dict = defaultdict(float)
        node_imp_b: Dict = defaultdict(float)
        node_sources: Dict = defaultdict(list)

        for atom_info in explanation["top_atoms"]:
            k = atom_info["atom_idx"]
            atom_contrib = atom_info["raw_contribution"]
            atom_weights = self.dictionary[k]

            for token_str, node_ids in token_node_map.items():
                if token_str not in vocab_index:
                    continue
                feat_idx = vocab_index[token_str]
                w = float(atom_weights[feat_idx])
                if w == 0.0:
                    continue
                wl_coef = float(wl_vec[feat_idx])
                path_a = atom_contrib * w
                path_b = atom_contrib * w * wl_coef

                for node_id in node_ids:
                    node_imp_a[node_id] += path_a
                    node_imp_b[node_id] += path_b
                    node_sources[node_id].append({
                        "atom_idx": k,
                        "token": token_str,
                        "atom_contribution": round(atom_contrib, 6),
                        "atom_weight": round(w, 6),
                        "wl_coef": round(wl_coef, 6),
                        "path_importance_a": round(path_a, 6),
                        "path_importance_b": round(path_b, 6),
                    })

        sorted_a = sorted(node_imp_a.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_b = sorted(node_imp_b.items(), key=lambda x: abs(x[1]), reverse=True)
        total_a = sum(abs(s) for _, s in sorted_a) or 1.0
        total_b = sum(abs(s) for _, s in sorted_b) or 1.0

        return {
            "prediction": explanation["prediction_label"],
            "confidence": explanation["confidence"],
            "sorted_nodes":        sorted_a,
            "node_importance":     dict(node_imp_a),
            "node_importance_pct": {nid: round(abs(s)/total_a*100, 2) for nid, s in sorted_a},
            "sorted_nodes_b":         sorted_b,
            "node_importance_b":      dict(node_imp_b),
            "node_importance_b_pct":  {nid: round(abs(s)/total_b*100, 2) for nid, s in sorted_b},
            "node_sources": dict(node_sources),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Full text report  (Level 5 -> Level 1 -> Level 2)
    # ─────────────────────────────────────────────────────────────────────────

    def full_report(
        self, graph, top_k_atoms: int = 5, top_k_features_per_atom: int = 5
    ) -> str:
        """Levels 5 + 1 + 2 as a formatted text report."""
        _, sparse_code = self._embed_graph(graph)
        scaled = self.scaler.transform(sparse_code)
        explanation = self._build_explanation(sparse_code, scaled, top_k_atoms)

        lines: List[str] = [self.contribution_breakdown(explanation)]

        conf_str = (f"  (Confidence: {explanation['confidence'] * 100:.1f}%)"
                    if explanation["confidence"] is not None else "")
        lines += [
            "",
            "-- LEVEL 1: Prediction Reasoning --",
            f"  Prediction : {explanation['prediction_label']}{conf_str}",
            "  Atoms driving this prediction:",
        ]
        for a in explanation["top_atoms"]:
            lines.append(
                f"    * Dict Atom {a['atom_idx']:>4d}"
                f"  {a['percentage']:>5.1f}%  [{a['direction']}]"
                f"  activation={a['activation']:.4f}"
            )

        lines += ["", "-- LEVEL 2: Dictionary Atom Composition --"]
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

        return "\n".join(lines)
