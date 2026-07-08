"""Deployable artifact bundle: save/load a complete inference pipeline.

A bundle is a single self-describing directory holding everything the app needs
to classify a new graph:

    <root>/<implementation>_<dataset>_atoms<N>_<timestamp>/
        manifest.json          # what this is + how to rebuild it
        encoder/               # encoder.save() writes here (e.g. WL vocab+config)
        dict_learner/          # dict_learner.save() writes here (e.g. FDDL D+state)
        scaler.joblib          # fitted feature scaler
        models/model_<slug>.joblib   # trained classifiers
        thresholds.json        # per-model decision thresholds
        eval/                  # (optional) provenance metrics for this instance

Loose coupling: the store never touches component internals. It delegates to
each component's own save()/load() and uses the registry to map the manifest's
class names back to classes. New encoders/learners plug in with zero changes
here.
"""

import os
import json
import subprocess
from datetime import datetime

import numpy as np
import joblib

from utils.registry import get_encoder_class, get_dict_learner_class

ENCODER_SUBDIR = "encoder"
DICT_LEARNER_SUBDIR = "dict_learner"
MODELS_SUBDIR = "models"
EVAL_SUBDIR = "eval"
SCALER_FILE = "scaler.joblib"
THRESHOLDS_FILE = "thresholds.json"
MANIFEST_FILE = "manifest.json"


def _model_slug(model_name):
    return model_name.lower().replace(" ", "_")


def build_bundle_name(implementation, dataset, n_atoms, started_at=None, ended_at=None):
    """Dynamic, self-describing bundle name — mirrors the results/ convention.

    Carries both the start and end timestamps so a bundle's build duration is
    visible from its folder name alone.
    """
    ended_at = ended_at or datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = started_at or ended_at
    atoms_part = f"_atoms{n_atoms}" if n_atoms is not None else ""
    return f"{implementation}_{dataset}{atoms_part}_{started_at}_{ended_at}"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _lib_versions():
    versions = {}
    for mod in ("numpy", "scipy", "sklearn", "torch", "joblib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def save_bundle(
    root,
    implementation,
    dataset,
    encoder,
    dict_learner,
    scaler,
    models,
    thresholds,
    default_model=None,
    extra_metadata=None,
    started_at=None,
):
    """Serialise a full pipeline into a new versioned bundle directory.

    `models` is {model_name: fitted_estimator}; `thresholds` is
    {model_name: float}. `started_at` (a YYYYmmdd_HHMMSS string) is the run's
    start time; the bundle folder is named "<...>_<start>_<end>". Returns the
    bundle path.
    """
    n_atoms = dict_learner.n_atoms()
    ended_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = os.path.join(
        root,
        build_bundle_name(implementation, dataset, n_atoms, started_at, ended_at),
    )
    os.makedirs(bundle_dir, exist_ok=True)

    # --- delegate component serialisation (loose coupling) ---
    encoder.save(os.path.join(bundle_dir, ENCODER_SUBDIR))
    dict_learner.save(os.path.join(bundle_dir, DICT_LEARNER_SUBDIR))

    # --- generic pieces ---
    joblib.dump(scaler, os.path.join(bundle_dir, SCALER_FILE))

    models_dir = os.path.join(bundle_dir, MODELS_SUBDIR)
    os.makedirs(models_dir, exist_ok=True)
    model_files = {}
    for name, estimator in models.items():
        fname = f"model_{_model_slug(name)}.joblib"
        joblib.dump(estimator, os.path.join(models_dir, fname))
        model_files[name] = os.path.join(MODELS_SUBDIR, fname)

    with open(os.path.join(bundle_dir, THRESHOLDS_FILE), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in thresholds.items()}, f, indent=2)

    # --- manifest: everything needed to reconstruct + validate ---
    classes = getattr(dict_learner, "classes_", None)
    manifest = {
        "implementation": implementation,
        "dataset": dataset,
        "started_at": started_at,
        "ended_at": ended_at,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "encoder_class": type(encoder).__name__,
        "dict_learner_class": type(dict_learner).__name__,
        "n_atoms": int(n_atoms),
        "feature_dim": int(encoder.n_vocab) if hasattr(encoder, "n_vocab") else None,
        "classes": np.asarray(classes).tolist() if classes is not None else None,
        "models": model_files,
        "default_model": default_model or (next(iter(models)) if models else None),
        "git_commit": _git_commit(),
        "library_versions": _lib_versions(),
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata
    with open(os.path.join(bundle_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return bundle_dir


def load_bundle(bundle_dir):
    """Reconstruct a pipeline from a bundle directory.

    Returns a dict:
        {encoder, dict_learner, scaler, models, thresholds, manifest}
    """
    with open(os.path.join(bundle_dir, MANIFEST_FILE), encoding="utf-8") as f:
        manifest = json.load(f)

    encoder_cls = get_encoder_class(manifest["encoder_class"])
    dict_learner_cls = get_dict_learner_class(manifest["dict_learner_class"])

    encoder = encoder_cls.load(os.path.join(bundle_dir, ENCODER_SUBDIR))
    dict_learner = dict_learner_cls.load(os.path.join(bundle_dir, DICT_LEARNER_SUBDIR))
    scaler = joblib.load(os.path.join(bundle_dir, SCALER_FILE))

    models = {}
    for name, rel_path in manifest.get("models", {}).items():
        models[name] = joblib.load(os.path.join(bundle_dir, rel_path))

    with open(os.path.join(bundle_dir, THRESHOLDS_FILE), encoding="utf-8") as f:
        thresholds = json.load(f)

    return {
        "encoder": encoder,
        "dict_learner": dict_learner,
        "scaler": scaler,
        "models": models,
        "thresholds": thresholds,
        "manifest": manifest,
    }
