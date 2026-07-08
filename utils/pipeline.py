"""Shared representation pipeline.

Both the Monte-Carlo-CV benchmark and the single-pass export script must build
the WL->dictionary->sparse-code representation *identically* — otherwise the
deployed model would not match the numbers reported in the paper. To guarantee
that, the expensive, order-sensitive core lives here once and is called by both.

Everything is written against the base-class contracts
(`GraphEncoder.generate_*_embeddings`, `DictLearner.fit/infer`), so it works
unchanged for any encoder/dict-learner combination (WL+FDDL, WL+AKSVD,
FSM+FDDL, ...).
"""

import os
import json

import numpy as np
import joblib


def fit_encoder_and_dictionary(encoder, dict_learner, G_vocab_train, y_vocab_train):
    """Fit the encoder's vocabulary and then the dictionary on that vocabulary.

    Returns the training graph embeddings (encoder space) in case a caller wants
    them; the encoder and dict_learner are mutated in place.
    """
    train_emb = encoder.generate_training_embeddings(G_vocab_train, y_vocab_train)
    dict_learner.fit(training_graph_embeddings=train_emb, y_train=y_vocab_train)
    return train_emb


def sparse_codes(encoder, dict_learner, graphs):
    """Encode graphs -> WL embedding -> sparse codes via the trained dictionary."""
    embeddings = encoder.generate_inferencing_embeddings(graphs)
    return dict_learner.infer(embeddings)


def save_sparse_code_cache(dirpath, splits, scaler=None, metadata=None):
    """Persist raw (pre-scale) sparse codes + labels so downstream ML models can
    be re-trained/compared without recomputing WL+dictionary.

    Parameters
    ----------
    dirpath : str
        Destination directory (created if missing).
    splits : dict[str, tuple(np.ndarray, array-like)]
        e.g. {"ml_train": (X, y), "val": (X, y), "test": (X, y)}. Store the
        *raw* FDDL codes (not scaled) so future experiments can choose their own
        preprocessing; the scaler is saved separately for reproducibility.
    scaler : fitted transformer, optional
        Saved alongside so the exact training-time scaling can be reproduced.
    metadata : dict, optional
        Provenance (implementation, dataset, seed, dictionary atom count, ...).
        These codes are valid ONLY for the dictionary that produced them.
    """
    os.makedirs(dirpath, exist_ok=True)

    arrays = {}
    for split_name, (X, y) in splits.items():
        arrays[f"X_{split_name}"] = np.asarray(X)
        arrays[f"y_{split_name}"] = np.asarray(y)
    np.savez_compressed(os.path.join(dirpath, "sparse_codes.npz"), **arrays)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(dirpath, "scaler.joblib"))

    if metadata is not None:
        meta = dict(metadata)
        meta.setdefault(
            "note",
            "Raw pre-scale sparse codes. Valid only for the dictionary that "
            "produced them; re-encode if WL/dictionary params change.",
        )
        with open(os.path.join(dirpath, "cache_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return dirpath
