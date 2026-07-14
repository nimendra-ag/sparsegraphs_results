"""Single-pass export: turn a trained pipeline into a deployable artifact bundle.

Generic over the encoder/dict-learner (base-class contract only), so every
implementation — WL+FDDL, WL+AKSVD, FSM+FDDL, ... — reuses it and an
`implements/<impl>.py` export script is only a few lines.

Split policy mirrors the MC-CV benchmark exactly (train/val/test, where train
subdivides into vocab_train + ML_train), so the exported model is built the same
way it was benchmarked and the bundle carries an honest held-out **test**
evaluation (not just a validation slice):

    vocab_train (50%) -> encoder vocabulary + dictionary
    ML_train    (20%) -> classifiers
    val         (15%) -> decision-threshold tuning
    test        (15%) -> held-out evaluation saved under eval/

This file is also where every intermediate component is persisted: the
deployable bundle AND the raw sparse-code cache (ml_train/val/test codes +
labels) so different downstream classifiers can be compared later without
recomputing WL+dictionary.
"""

import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from utils.evaluator import Evaluator
from utils.seeding import seed_everything
from utils import pipeline
from utils.artifact_store import save_bundle

# Split proportions — identical to wl_fddl_gpu_mccv.py.
TEST_SIZE = 0.15                 # 15% of full -> held-out test
VAL_SIZE_OF_REMAINDER = 0.15 / 0.85   # -> 15% of full -> validation
VOCAB_ML_RATIO = 2 / 7           # vocab_train:ML_train = 5:2 -> 50% / 20% of full


def export_pipeline(
    encoder,
    dict_learner,
    graphs,
    y,
    implementation,
    dataset,
    artifacts_root="artifacts",
    split_seed=42,
    save_eval=True,
    save_code_cache=True,
):
    """Fit the full pipeline on one fixed split and write a deployable bundle.

    Returns the bundle directory path.
    """
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(split_seed)

    # --- 3-tier split (mirrors the MC-CV benchmark) --------------------------
    G_train_full, G_test, y_train_full, y_test = train_test_split(
        graphs, y, test_size=TEST_SIZE, random_state=split_seed, stratify=y
    )
    G_train, G_val, y_train, y_val = train_test_split(
        G_train_full, y_train_full,
        test_size=VAL_SIZE_OF_REMAINDER, random_state=split_seed, stratify=y_train_full
    )
    G_vocab, G_ml, y_vocab, y_ml = train_test_split(
        G_train, y_train,
        test_size=VOCAB_ML_RATIO, random_state=split_seed, stratify=y_train
    )

    # --- representation (shared with MC-CV, so no drift) ---------------------
    scaler = MaxAbsScaler()
    pipeline.fit_encoder_and_dictionary(encoder, dict_learner, G_vocab, y_vocab)
    n_atoms = dict_learner.n_atoms()

    X_ml = pipeline.sparse_codes(encoder, dict_learner, G_ml)
    X_val = pipeline.sparse_codes(encoder, dict_learner, G_val)
    X_test = pipeline.sparse_codes(encoder, dict_learner, G_test)
    X_ml_s = scaler.fit_transform(X_ml)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # --- classifiers on ML split; thresholds tuned on the validation split ---
    evaluator_val = Evaluator(
        X_ml_s, y_ml, X_val_s, y_val,
        implementation=implementation, dataset=dataset, n_atoms=n_atoms,
        random_state=split_seed, started_at=started_at,
    )
    evaluator_val.predict_logistic_regression()
    evaluator_val.predict_gradient_boosting()
    evaluator_val.predict_svm()
    evaluator_val.predict_random_forest()

    models = evaluator_val.get_fitted_models()       # trained on the ML split
    thresholds = evaluator_val.get_thresholds()      # tuned on the val split

    # --- honest held-out TEST evaluation (val-tuned thresholds reused) -------
    evaluator_test = Evaluator(
        X_ml_s, y_ml, X_test_s, y_test,
        implementation=implementation, dataset=dataset, n_atoms=n_atoms,
        random_state=split_seed, fixed_thresholds=thresholds, started_at=started_at,
    )
    evaluator_test.predict_logistic_regression()
    evaluator_test.predict_gradient_boosting()
    evaluator_test.predict_svm()
    evaluator_test.predict_random_forest()

    # --- write the bundle ----------------------------------------------------
    bundle_dir = save_bundle(
        artifacts_root,
        implementation,
        dataset,
        encoder=encoder,
        dict_learner=dict_learner,
        scaler=scaler,
        models=models,
        thresholds=thresholds,
        started_at=started_at,
        extra_metadata={
            "split_seed": split_seed,
            "split": {"vocab_train": 0.50, "ml_train": 0.20, "val": 0.15, "test": 0.15},
            "note": "Headline performance is the MC-CV mean +/- std; the eval/ "
                    "metrics here are this single instance's held-out test.",
        },
    )

    # --- analytics: feature-selection elbow plot ------------------------------
    # The WL encoder stashes the full pre-trim score curve; render the knee/elbow
    # diagnostic into the bundle so the vocab-size decision is auditable.
    selection_scores = getattr(encoder, "selection_scores_", None)
    if selection_scores is not None:
        from utils.knee import plot_knee_curve
        analytics_dir = os.path.join(bundle_dir, "analytics")
        plot_path = plot_knee_curve(
            selection_scores,
            os.path.join(analytics_dir, "wl_feature_selection_knee.png"),
            title=f"{implementation} / {dataset}",
        )
        print(f"Saved feature-selection elbow plot -> {plot_path}")

    # --- sparse-code cache: raw (pre-scale) codes for classifier experiments -
    if save_code_cache:
        pipeline.save_sparse_code_cache(
            os.path.join(bundle_dir, "code_cache"),
            splits={
                "ml_train": (X_ml, y_ml),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
            },
            scaler=scaler,
            metadata={
                "implementation": implementation,
                "dataset": dataset,
                "split_seed": split_seed,
                "total_atoms": int(n_atoms),
            },
        )

    # --- provenance: held-out test metrics inside the bundle -----------------
    if save_eval:
        evaluator_test.save_report(run_folder=os.path.join(bundle_dir, "eval"))

    print(f"Exported deployable bundle -> {bundle_dir}")
    return bundle_dir
