# PCA comparison arm

Isolated experiment directory. Nothing here runs unless you invoke it, and the
production pipeline (`implements/wl_fddl_gpu.py`, `implements/wl_fddl_gpu_mccv.py`)
is unchanged by its presence.

## The question

The baseline spends the labels on **feature selection**: it scores every WL
subtree hash with `total_presence * |sqrt(p_maj) - sqrt(p_min)|` and keeps the
top prefix covering 99% of that score mass. On `nci_full` that is
**10130 -> 1751 hashes** (the elbow cut would keep 224).

Arm A asks whether that supervised selection beats simply compressing the *whole*
vocabulary with an unsupervised linear projection:

```
baseline   WL -> score -> energy cut (10130 -> 1751 hashes) -> FDDL -> classifiers
Arm A      WL -> score -> no cut     (10130 hashes) -> PCA (-> d) -> FDDL -> classifiers
```

Scoring, hashing, splits, FDDL hyperparameters, scaler, classifiers, and
threshold protocol are identical. The only difference is how ~10k scored hashes
become FDDL's input: a supervised **subset**, or an unsupervised **rotation**.

### What PCA does not give you

Worth being explicit, because it is the honest framing for the paper: PCA is not
a filter. Every one of its `d` output columns is a dense combination of all 10130
hashes, so **nothing is discarded**. At inference time Arm A must still compute
every WL subtree hash for every incoming graph; the energy cut genuinely lets you
skip 8379 of them. If Arm A ties on AUC, the baseline still wins on encoder cost
and on interpretability.

## Files

| File | Role |
|---|---|
| `wl_full.py` | `WLFull` — WL with the cut disabled (`selection="none"`), verified |
| `reducer.py` | `PCAReducer` — PCA / TruncatedSVD, fixed width or variance target |
| `pca_dict_learner.py` | `PCAThenDictLearner` — composes reducer + FDDL as one `DictLearner` |
| `run_arm_a.py` | Single-split export — the entry point |

The reduction slots in as a `DictLearner`, which is why `utils/pipeline.py`,
`utils/export.py`, and `utils/artifact_store.py` needed no changes:

```
encoder -> [ PCAReducer -> FDDLGPU ] -> classifiers
           \___ PCAThenDictLearner ___/
```

Three edits were made outside this directory, all additive:

- `graph_encoders/wl.py` — new `selection="none"` branch. Defaults untouched
  (`selection="energy"`, `energy=0.99`).
- `utils/registry.py` — registers `WLFull` and `PCAThenDictLearner` so exported
  bundles can be re-loaded.
- `utils/elbow.py` — accepts `selection="none"` so the analytics figures do not
  claim a cut was applied when none was.

### Why `selection="none"` and not `energy=1.0`

An energy target of 1.0 looks like it keeps everything, but features scoring
exactly 0 (equally present in both classes) make the cumulative-score curve
plateau before the last rank, so `find_energy_cut` stops early. MUTAG has such
features — 139 of 143 kept on seed 41 — while `nci_full` has none. The guard in
`WLFull.create_vocab` exists because this trap is invisible on the main dataset.

## Running it

Use the project environment (`genv12`, which matches the versions in existing
manifests):

```
C:\Users\Puldith CE\miniconda3\envs\genv12\python.exe
```

### 1. Wiring check (~1 minute)

```bash
python pca/run_arm_a.py --dataset mutag --dim 32 --atoms 16
```

### 2. Single split on the real dataset

```bash
python pca/run_arm_a.py --dataset nci_full --dim 1751 --atoms 256
```

Writes an artifact bundle to `artifacts/wl_pca1751_fddl_gpu_nci_full_atoms*_*/`
containing the encoder vocabulary, the fitted reducer, the FDDL dictionary, the
trained classifiers, the elbow analytics, and a held-out test report under
`eval/`.

`--dim` matched to the baseline's keep count (1751 on `nci_full`) and `--atoms`
matched to the baseline run are what make the arms comparable; changing either
adds a second uncontrolled variable.

### Letting the variance spectrum pick the width

```bash
python pca/run_arm_a.py --dataset nci_full --pca-energy 0.99 --atoms 256
```

This is the PCA analogue of the encoder's energy cut — same rule (sort a
non-negative spectrum, accumulate, stop at the target), applied to the eigenvalue
spectrum instead of the supervised discriminative-score spectrum. It resolves `d`
from the data, so use a fixed `--dim` when you want the arms matched exactly.

### Resource expectations on `nci_full`

The whole point of Arm A is that the cut is gone, so the encoder builds the full
matrix. `vocab_train` is 50% of ~36,900 graphs:

- `calc_coefficients` allocates **18,470 x 10,130 float64 ≈ 1.5 GB**, and its
  Python loop does ~5.8x the work it does in the baseline.
- The reducer casts to float32 (+0.75 GB) and sklearn centers a copy (+0.75 GB).
- Budget **~5 GB of free RAM** and expect the encode phase to dominate.

If memory is tight, drop `--dim` first (it does not change the encoder cost) or
run with `--no-center` (TruncatedSVD, no centering) — but then report it as SVD,
not PCA.

## Reading the result

### A single split is not a reportable comparison

`run_arm_a.py` fits **one** partition. Its metrics carry split luck as well as
any real effect, so treat the output as a wiring check and a first signal — not
as a number to put beside the baseline's MC-CV mean ± std. There is deliberately
no MC-CV driver in this directory; if a reportable number is ever needed, the
comparison has to be *paired* (both arms share `MASTER_SEEDS`, so each seed
derives the identical partition, and most of the spread cancels in the
per-seed difference). Comparing two independent mean ± std summaries by eye
proves nothing — the error bars overlap even when the paired difference is
tight.

### Read AUC, not accuracy

The datasets are imbalanced and thresholds are tuned on validation. `ROC-AUC` and
`PR-AUC` are threshold-free and are the honest comparison; accuracy and F1 move
with the tuned threshold and are noisier.

### One width is not the experiment

A single `--dim` answers "is PCA at width d better than the energy cut", which
confounds *method* with *width*. If PCA at 256 beats a 1751-hash cut, that may
just be dimensionality. Sweep it:

```bash
for d in 64 128 256 512 1024 1751; do
  python pca/run_arm_a.py --dataset nci_full --dim $d --atoms 256
done
```

Then plot test AUC against `d` with the elbow (224) and energy (1751) keep counts
marked as vertical lines. That figure — not a pair of scalars — is the result,
and it is the direct analogue of the existing elbow suite.

## Not implemented

Arm B (cut, *then* PCA) and Arm C (PCA **instead of** FDDL, which tests whether
the dictionary learning earns its keep) both reuse `PCAReducer`. Arm C needs only
a thin `DictLearner` exposing `fit`/`infer` as `fit_transform`/`transform`.
