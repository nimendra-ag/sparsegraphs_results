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


def _drop_zero_embeddings(embeddings, labels):
    """Drop rows whose embedding is the all-zero vector, keeping labels aligned.

    An all-zero encoder embedding is a degenerate sample: none of its features
    survived the encoder's vocabulary (e.g. a tiny molecule whose WL subtrees
    were all trimmed by the feature-selection cut). Such a row carries no signal
    for dictionary learning, and — critically — it is what lets a dictionary
    learner sample a zero-norm atom and divide by zero (0/0 -> NaN). Removing it
    here, at the encoder->dictionary boundary, keeps the training matrix clean.

    This is applied to the TRAINING path only. Inference (val/test) goes through
    `sparse_codes`, which never calls this — every val/test graph must still get
    a code/prediction, even if its embedding happens to be all-zero.

    Returns (embeddings, labels, n_dropped) with the same types as the inputs
    (labels stays None if it was None, i.e. the unsupervised case).
    """
    keep = np.any(embeddings != 0, axis=1)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  [filter] dropped {n_dropped}/{embeddings.shape[0]} all-zero "
              f"training embeddings before dictionary fit "
              f"(no signal; would poison atom init).", flush=True)
        embeddings = embeddings[keep]
        if labels is not None:
            labels = np.asarray(labels)[keep]
    return embeddings, labels, n_dropped


def fit_encoder_and_dictionary(encoder, dict_learner, G_vocab_train, y_vocab_train):
    """Fit the encoder's vocabulary and then the dictionary on that vocabulary.

    All-zero training embeddings are dropped between the two steps (see
    `_drop_zero_embeddings`) so the dictionary learner never sees a degenerate,
    signal-free sample. This affects TRAINING only; the val/test inference path
    (`sparse_codes`) keeps every row.

    Returns the (filtered) training graph embeddings in case a caller wants
    them; the encoder and dict_learner are mutated in place.
    """
    train_emb = encoder.generate_training_embeddings(G_vocab_train, y_vocab_train)
    train_emb, y_vocab_train, _ = _drop_zero_embeddings(train_emb, y_vocab_train)
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
