# Full Result Analysis — `analysis/all_results.csv`

*Sparse dictionary learning for imbalanced whole-graph classification on the NCI1 (`nci_full`) corpus.*

**Scope of this document.** This is an independent, end-to-end re-analysis of every row in
`analysis/all_results.csv` (556 records). It reproduces the findings reported in Chapter 4 of the
project report, and then goes further: it audits the evaluation protocol, tests the project's two
imbalance-specific contributions head-to-head, decomposes where performance actually comes from,
adds a cost analysis, and documents several places where numbers in the report do not survive a
careful re-derivation. Every number below is computed directly from the CSV and is reproducible.

---

## 0. Executive summary

| # | Finding | Strength |
| --- | --- | --- |
| 1 | **Supervision in the dictionary objective is the single strongest predictor of quality.** FDDL and CS-FDDL take ranks 1–2 in essentially every one of 36 conditions; the gap to third place is ~1.9 rank positions. | Very strong, reproduces the report |
| 2 | **Frozen K-SVD — the cheap imbalance trick — works.** It beats vanilla AKSVD on 46/64 matched cells (+0.019 Minority-F1, Wilcoxon *p* < 0.0001), with the gain concentrated at 128–1024 atoms. | Strong, **new** |
| 3 | **CS-FDDL — the expensive imbalance trick — does not.** It *loses* to vanilla FDDL on 51/72 matched cells (−0.012 Minority-F1, *p* = 0.0002; −0.024 on the WL encoder). Its #2 rank comes from FDDL's family advantage, not from the cost-sensitive weighting. | Strong, **new; contradicts the report's framing** |
| 4 | **The baseline verdict inverts with the classifier, cleanly and at every matched width.** Under LogReg/LinearSVM, the best WL pipeline beats graph2vec at all 7 matched widths (+0.056 … +0.118 Minority-F1). Under GradientBoosting/RandomForest, graph2vec wins at 13 of 14 matched widths. | Very strong, sharpens the report |
| 5 | **Only 4 of 25 pipelines ever beat the best baseline**, and only under a linear classifier: `wl+fddl`, `wl+csfddl`, `wl+frozen_ksvd` (and `wl+aksvd` in one cell). 28 of 500 pipeline cells (5.6 %) clear the bar on Minority-F1. | Strong, **new** |
| 6 | **Dictionary size has three regimes** — scales (FDDL ρ = +0.79, CS-FDDL +0.67), saturates (AKSVD, Frozen, LC-KSVD), collapses (Bayesian ρ = −0.74, Online-DL −0.30). | Strong, reproduces the report |
| 7 | **Cost and quality are uncorrelated (ρ = −0.16); the entire cost/quality Pareto frontier is Frozen K-SVD.** 14 s of fitting buys Minority-F1 0.4745 — within 0.015 of the best pipeline number in the study, which costs 100–700× more. | Strong, **new** |
| 8 | **Macro-F1, Minority-F1 and MCC are the same measurement** (Spearman ρ ≥ 0.994). Only ROC-AUC carries independent information (ρ ≈ 0.87). The report's "three metrics" are really two. | Strong, **new** |
| 9 | **Which design factor matters most depends on the metric.** ROC-AUC: encoder (η² = 0.34) ≈ learner (0.31) ≫ classifier (0.08). Minority-F1: learner (0.31) ≈ classifier (0.29) ≫ encoder (0.10). | **New** |
| 10 | **The evaluation protocol is not uniform.** Only 38 % of pipeline rows are 5-seed MC-CV; 30 % are single-split. `gspan_cork` is entirely single-split and `wl_edge` is 90 % single-split. Several headline comparisons are therefore between estimators of very different variance. | **Methodological, important** |

---

## 1. What is actually in the file (inventory and protocol audit)

### 1.1 Inventory

556 rows × 32 columns; `status = ok` for all; `dataset = nci_full` for all.

| Group | Rows | Levels |
| --- | --- | --- |
| Pipelines | 520 | 4 encoders × 7 dictionary learners = 25 realised combinations (of 28) |
| Baselines | 36 | `sf` (4), `graph2vec` (28), `gcn` (4) |
| Encoders | — | `wl` 224, `fsm` 140, `gspan_cork` 116, `wl_edge` 40 |
| Learners | — | `fddl` 104, `aksvd` 84, `csfddl` 72, `online_dl` 68, `lcksvd` / `frozen_ksvd` / `bayesian` 64 each |
| Classifiers | — | LogReg / LinearSVM / GradientBoosting / RandomForest, 138 rows each; `GCN` 4 |
| Dictionary sizes | — | 16, 26, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20000 |

Five combinations were never run: `wl_edge` × {`csfddl`, `lcksvd`, `online_dl`, `frozen_ksvd`,
`bayesian`} — only FDDL and AKSVD exist for the edge encoder.

### 1.2 The protocol is *not* uniform — and this matters

The report states that "for each graph encoder + dictionary learner combination we executed
Monte-Carlo cross validation with 5 seeds." The CSV says otherwise:

| Evaluation protocol | Pipeline rows | Share |
| --- | --- | --- |
| MC-CV, 5 seeds | 196 | 37.7 % |
| MC-CV, 2 seeds | 148 | 28.5 % |
| MC-CV, 3 seeds | 20 | 3.8 % |
| Single split (`source = artifact`) | 156 | 30.0 % |

Concretely:

- **`gspan_cork` is 100 % single-split** (all 116 rows). Every gspan_cork number in this study is a
  one-draw estimate with no variance information.
- **`wl_edge` is 90 % single-split**; only the 32-atom FDDL cell is 5-seed.
- **`wl+csfddl` is 2-seed throughout**, while its direct comparator `wl+fddl` is 5-seed throughout.
  The two arms of the project's own FDDL-vs-CS-FDDL comparison were not measured at equal precision.
- **`wl+lcksvd` mixes three protocols inside one curve**: 5 seeds at 32 and 2048 atoms, 3 seeds at
  64–1024, and a single split at 4096. Its dictionary-size curve is partly an artefact of changing
  measurement noise along the x-axis.
- `wl+aksvd` and `wl+frozen_ksvd` use 2 seeds at 32–64 atoms and 5 seeds from 128 up — the *low* end
  of both curves is the noisiest part.

**Consequence.** Wherever a single-split number appears in a comparison, it is both noisier and
plausibly optimistic (one favourable partition is not averaged away). This is not fatal, but it
means the `wl_edge` and `gspan_cork` results must be read as provisional, and it is the main reason
the nominal "best overall" row of the study should not be quoted as a headline.

### 1.3 Missing data

20 rows — all `wl_edge+fddl` at 2048–20000 atoms — have `macro_f1`, `roc_auc` and `accuracy` but
**no minority-class metrics at all**. These happen to be the highest Macro-F1 and ROC-AUC rows in
the entire study (§9.2), which is exactly the wrong place to be missing the metric the project
cares about most. Re-running these five widths with minority metrics recorded is the single
highest-value piece of missing work in the results set.

---

## 2. The ground truth of this dataset: a 4.55 % base rate

The GCN baseline gives us the class balance for free, because **it failed completely**:

| GCN hidden size | Accuracy | Minority precision | Minority recall | Minority-F1 | ROC-AUC | Minority PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.9545 | 0.000 | 0.000 | 0.000 | 0.5527 | 0.0564 |
| 32 | 0.9524 | 0.000 | 0.000 | 0.000 | 0.5000 | 0.0476 |
| 64 | 0.9531 | 0.000 | 0.000 | 0.000 | 0.5000 | 0.0469 |
| 128 | 0.9553 | 0.000 | 0.000 | 0.000 | 0.5000 | 0.0447 |

**Observation.** The GCN predicted "inactive" for every molecule at every width. Its ROC-AUC is
0.500 (pure chance) at three of four widths, and its minority PR-AUC equals the base rate. Its
accuracy is 95.2–95.6 %.

**Analysis.** This pins the minority class at **4.47 – 4.76 % (mean 4.55 %)** of the corpus, and it
is the most useful single fact in the file: *a model that does nothing scores 95.4 % accuracy here.*

**Conclusion.**

1. Accuracy is uninformative on this corpus and is excluded from all comparisons below. Its Spearman
   correlation with Minority-F1 is 0.82 — high enough to look meaningful, low enough to mislead —
   and the highest-accuracy row in the whole file (0.9579) is a row with *no recorded* Minority-F1.
2. The GCN baseline is **not a competitive comparison, it is a degenerate control.** Reporting "our
   method beats GCN" is close to meaningless; reporting "GCN collapses at 4.6 % imbalance without
   class balancing, our pipelines do not" is the real and genuinely useful statement. The report's
   figures already mark GCN "off scale"; that framing is correct and should be kept.

### 2.1 The metric set is smaller than it looks

Spearman correlations across all 500 pipeline rows with complete metrics:

| | ROC-AUC | Macro-F1 | Minority-F1 | MCC | Minority PR-AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| **ROC-AUC** | 1.000 | 0.860 | 0.874 | 0.876 | 0.851 | 0.668 |
| **Macro-F1** | | 1.000 | **0.996** | **0.994** | 0.980 | 0.889 |
| **Minority-F1** | | | 1.000 | **0.999** | 0.976 | 0.852 |
| **MCC** | | | | 1.000 | 0.976 | 0.849 |

**Observation.** Macro-F1, Minority-F1 and MCC are rank-identical to within 0.6 %. On a
4.6 %-minority binary problem the majority-class F1 sits at ≈ 0.96 and barely moves, so Macro-F1 is
an affine shadow of Minority-F1.

**Conclusion.** The study effectively has **two** metrics, not three: a *threshold-free ranking*
metric (ROC-AUC) and a *thresholded decision* metric (any of Minority-F1 / Macro-F1 / MCC).
Reporting all three inflates apparent corroboration. Recommendation: report **Minority-F1** as the
decision metric (it is the operationally meaningful one) and **ROC-AUC** as the ranking metric, and
demote Macro-F1 / MCC to an appendix. Minority **PR-AUC** (ρ = 0.85 with ROC-AUC, 0.98 with
Minority-F1) is a more appropriate ranking metric at a 4.6 % positive rate; it is already in the
file but is never used in the report, and is worth promoting.

---

## 3. The noise floor: how large must a difference be to be real?

Across the 372 rows carrying a standard deviation:

| Metric | Median SD across seeds | Mean | Max |
| --- | --- | --- | --- |
| Macro-F1 | **0.0119** | 0.0123 | 0.0424 |
| ROC-AUC | **0.0147** | 0.0155 | 0.0745 |
| Minority PR-AUC | 0.0206 | 0.0223 | 0.0957 |
| Minority-F1 | **0.0215** | 0.0223 | 0.0742 |
| MCC | 0.0230 | 0.0236 | 0.0770 |

Per-learner reproducibility (5-seed rows, coefficient of variation on Minority-F1):

| Learner | Median Minority-F1 | Median SD | CV |
| --- | --- | --- | --- |
| `fddl` | 0.395 | 0.021 | **0.056** |
| `frozen_ksvd` | 0.384 | 0.027 | 0.071 |
| `aksvd` | 0.313 | 0.023 | 0.081 |
| `online_dl` | 0.322 | 0.025 | 0.083 |
| `bayesian` | 0.204 | 0.021 | 0.101 |
| `lcksvd` | 0.282 | 0.026 | 0.105 |

**Analysis.** FDDL is not only the most accurate learner, it is the **most reproducible** — its
relative seed-to-seed wobble is half that of LC-KSVD and Bayesian. This is a second, independent
argument for the discriminative objective: a Fisher term constrains the solution, and a constrained
solution is a stable one.

**Decision rule used throughout this document:** a Minority-F1 gap below **0.022**, or a ROC-AUC gap
below **0.015**, is *not* a result unless it survives a paired test across many matched cells. The
worst individual cell in the file (`wl+csfddl` @64, LogReg) has SD 0.074 — larger than most of the
effects the study is trying to measure.

---

## 4. Finding 1 — What the objective optimises decides everything

Method: for each of the 36 conditions (3 encoders that have all 7 learners × 4 classifiers ×
3 metrics), score each learner at *its own best* dictionary size and rank 1 (best) … 7 (worst). A
learner with no real advantage averages 4.00.

| Learner | Objective type | Mean rank | SD |
| --- | --- | --- | --- |
| **`fddl`** | Fisher discrimination (supervised) | **1.36** | 0.54 |
| **`csfddl`** | Fisher discrimination + class weights | **1.92** | 0.69 |
| `frozen_ksvd` | Reconstruction, class-staged | 3.83 | 1.44 |
| `aksvd` | Reconstruction | 4.43 | 1.34 |
| `online_dl` | Reconstruction (stochastic) | 4.78 | 1.33 |
| `lcksvd` | Reconstruction + label consistency | 5.79 | 0.94 |
| `bayesian` | Generative likelihood (BPFA) | 5.89 | 1.43 |

*(This reproduces Table 4.3 of the report exactly.)*

The ordering is invariant to how you slice it:

| Learner | by metric: Macro-F1 / Min-F1 / ROC-AUC | by encoder: fsm / gspan_cork / wl | by classifier: GB / SVM / LR / RF |
| --- | --- | --- | --- |
| `fddl` | 1.42 / 1.33 / 1.33 | 1.50 / 1.50 / **1.08** | 1.56 / 1.11 / 1.33 / 1.44 |
| `csfddl` | 1.92 / 2.08 / 1.75 | 1.58 / 2.00 / 2.17 | 1.44 / 2.11 / 2.33 / 1.78 |
| `frozen_ksvd` | 3.67 / 3.67 / 4.17 | 3.42 / **5.17** / 2.92 | 3.89 / 4.33 / 3.78 / 3.33 |
| `aksvd` | 4.33 / 4.33 / 4.62 | 4.83 / 3.96 / 4.50 | 4.67 / 4.39 / 4.11 / 4.56 |
| `online_dl` | 4.75 / 4.75 / 4.83 | 5.42 / 4.58 / 4.33 | 4.89 / 4.56 / 4.44 / 5.22 |
| `lcksvd` | 5.75 / 5.83 / 5.79 | 5.33 / 5.96 / 6.08 | 5.89 / 5.72 / 5.56 / 6.00 |
| `bayesian` | 6.17 / 6.00 / 5.50 | 5.92 / 4.83 / **6.92** | 5.67 / 5.78 / 6.44 / 5.67 |

**Observations.**

1. The two Fisher-objective learners are 1st and 2nd under *every* metric, *every* encoder and
   *every* classifier. There is no slice in which this ordering breaks.
2. **LC-KSVD is the study's most interesting failure.** It is the only other learner with an
   explicit label term, yet it ranks 6/7 — behind three purely unsupervised methods. The mechanism
   is visible in the objective: LC-KSVD bolts a label-consistency penalty onto a reconstruction
   objective and solves the *stacked* K-SVD problem, so labels enter as an additional reconstruction
   target rather than as a separation constraint. Bolting labels on is not the same as optimising
   for separation.
3. `frozen_ksvd` is strongly encoder-dependent: rank 2.92 on WL but 5.17 on gspan_cork. Its
   two-stage majority-then-minority construction needs an encoder whose feature space actually
   separates the two stages; gspan_cork's CORK-selected binary indicators apparently do not.
4. `bayesian` is worst on WL (6.92) and mid-pack on gspan_cork (4.83), for the reason in §6: BPFA's
   failure is a function of dictionary size, and only WL was swept out to 4096 atoms.

**Conclusion.** The headline of this study is not "dictionary learning helps." It is: **a dictionary
helps exactly to the extent that its objective contains a discrimination term.** The four
unsupervised learners average rank 4.9 — collectively no better than picking among them at random.

---

## 5. Findings 2 & 3 — The project's two imbalance contributions, tested head-to-head

The report presents two class-imbalance mechanisms as contributions: **Frozen K-SVD** (learn on the
majority, freeze, augment for the minority) and **CS-FDDL** (per-class gradient weighting inside
FDDL). Ranking tables cannot answer whether either *mechanism* works, because they compare across
learner families. The right test is **paired against the same learner without the mechanism**, at
matched (encoder, atoms, classifier).

| Comparison | Metric | n cells | Wins | Mean Δ | Median Δ | Wilcoxon *p* |
| --- | --- | --- | --- | --- | --- | --- |
| **`frozen_ksvd` − `aksvd`** | Minority-F1 | 64 | 46 | **+0.0190** | +0.0149 | **< 0.0001** |
| | ROC-AUC | 64 | 49 | +0.0079 | +0.0075 | < 0.0001 |
| | Macro-F1 | 64 | 45 | +0.0111 | +0.0061 | < 0.0001 |
| **`csfddl` − `fddl`** | Minority-F1 | 72 | 21 | **−0.0122** | −0.0114 | **0.0002** |
| | ROC-AUC | 72 | 24 | −0.0052 | −0.0043 | 0.0028 |
| | Macro-F1 | 72 | 22 | −0.0071 | −0.0060 | 0.0001 |

### 5.1 Frozen K-SVD works, and works where you would predict

Per encoder (Minority-F1, `frozen_ksvd` − `aksvd`):

| Encoder | n | Frozen wins | Mean Δ |
| --- | --- | --- | --- |
| `wl` | 32 | **28** | **+0.0324** |
| `fsm` | 16 | 11 | +0.0136 |
| `gspan_cork` | 16 | 7 | −0.0025 |

And by dictionary size (WL, averaged over classifiers):

| Atoms | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Δ Minority-F1 | +0.004 | +0.025 | **+0.065** | +0.048 | **+0.069** | +0.046 | +0.015 | −0.013 |

**Analysis.** The freezing benefit has a clear, theory-consistent shape: ≈ 0 when the dictionary is
too small to hold both a majority basis *and* a minority residual (32 atoms), peaking at 128–1024
atoms where the split has room to matter, then vanishing and reversing at 4096, where an
unconstrained K-SVD has enough capacity to cover the minority anyway and freezing is only a
constraint. A noise artefact would not produce this shape.

**Conclusion.** Frozen K-SVD is a validated contribution: **+0.032 Minority-F1 over vanilla AKSVD on
the WL encoder (28/32 matched wins)** — and, as §10 shows, at *lower* cost than the method it beats.
Recommended operating range: 128–1024 atoms.

### 5.2 CS-FDDL does not beat plain FDDL on this corpus

Per encoder (Minority-F1, `csfddl` − `fddl`):

| Encoder | n | CS-FDDL wins | Mean Δ |
| --- | --- | --- | --- |
| `wl` | 32 | **4** | **−0.0243** |
| `fsm` | 24 | 9 | −0.0067 |
| `gspan_cork` | 16 | 8 | +0.0036 |

**Observation.** On the study's primary encoder, cost-sensitive weighting loses 28 of 32 matched
comparisons, and the mean loss (−0.024) exceeds the Minority-F1 noise floor (0.0215). On the other
two encoders it is a wash. There is no slice of this data in which CS-FDDL is better than FDDL.

**Analysis — three candidate explanations, in order of plausibility.**

1. **The step-size penalty.** By the report's own §2.10 description, CS-FDDL divides the IPM step
   size by `max(w_i)` to preserve convergence under the inflated Lipschitz constant. At a 4.55 %
   minority rate, inverse-frequency weighting gives `max(w_i) ≈ 11`, so CS-FDDL takes steps roughly
   an order of magnitude smaller than FDDL for the same iteration budget. **The most likely reading
   is that CS-FDDL is simply under-converged at matched iteration count** — not a worse objective,
   the same objective solved less far. This is directly testable.
2. **FDDL is already class-balanced by construction.** FDDL's discriminative fidelity term is
   written *per class* (`r(A_i, D, X_i)` summed over classes, each class owning a sub-dictionary),
   so per-class residuals already enter the objective with equal structural weight regardless of
   `n_i`. Re-weighting a term that is already per-class adds variance without adding signal.
3. **Under-measurement.** `wl+csfddl` is 2-seed while `wl+fddl` is 5-seed. That inflates CS-FDDL's
   variance but should not bias its mean; it makes individual cell comparisons noisy, not the
   72-cell paired test. It cannot explain a 28-of-32 losing streak.

**Conclusion — and this matters for how the report is written.** CS-FDDL earns its rank-2 position
because it is an FDDL, not because it is cost-sensitive. The defensible claim is:

> *"The Fisher-discrimination family (FDDL / CS-FDDL) dominates. Within that family, cost-sensitive
> weighting produced no measurable benefit at the 4.55 % imbalance level tested, and a small but
> statistically consistent loss on the WL encoder; we attribute this primarily to the step-size
> reduction the weighting requires, and identify it as the priority follow-up."*

This is a *stronger* contribution statement than the current one, because it is a falsifiable
negative result with a diagnosed mechanism — and it is exactly what the report's own Threats to
Validity section already anticipates ("no theoretical convergence proof for the weighted
formulation … dividing the step size by max(w_i) slows convergence").

### 5.3 The other paired contrasts, for completeness

| Comparison | Minority-F1 mean Δ | Wins / n | *p* |
| --- | --- | --- | --- |
| `fddl` − `aksvd` | **+0.0642** | 65 / 68 | < 0.0001 |
| `fddl` − `lcksvd` | **+0.0941** | 63 / 64 | < 0.0001 |
| `csfddl` − `aksvd` | +0.0502 | 60 / 64 | < 0.0001 |
| `frozen_ksvd` − `online_dl` | +0.0442 | 44 / 64 | < 0.0001 |
| `aksvd` − `bayesian` | +0.0828 | 55 / 64 | < 0.0001 |
| `aksvd` − `online_dl` | +0.0252 | 38 / 64 | 0.0088 |

The supervised-vs-unsupervised gap (+0.064 FDDL over AKSVD, 65/68 wins) is **three times the
Minority-F1 noise floor** and is the most robust effect in the entire dataset.

---

## 6. Finding 4 — Dictionary size: three regimes, set by the objective

Mean Spearman ρ(atoms, metric) computed within each (encoder, classifier) curve, then averaged:

| Learner | ρ ROC-AUC | ρ Macro-F1 | ρ Minority-F1 | Mean ΔAUC (max − min atoms) | Regime |
| --- | --- | --- | --- | --- | --- |
| `fddl` | **+0.79** | +0.74 | +0.69 | **+0.042** | scales |
| `csfddl` | **+0.67** | +0.60 | +0.60 | +0.028 | scales |
| `lcksvd` | +0.18 | +0.28 | +0.18 | +0.010 | scales weakly |
| `aksvd` | −0.36 | +0.03 | +0.02 | −0.013 | saturates |
| `frozen_ksvd` | −0.31 | −0.27 | −0.23 | −0.016 | saturates / decays |
| `online_dl` | −0.30 | −0.01 | −0.05 | −0.042 | degrades |
| `bayesian` | **−0.74** | −0.75 | **−0.79** | **−0.091** | collapses |

WL encoder, mean ROC-AUC over the four classifiers (bold = each learner's peak):

| Atoms | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fddl` | .8110 | .8271 | .8396 | .8495 | .8323 | .8466 | .8400 | **.8516** |
| `csfddl` | .8059 | .8077 | .8168 | .8364 | .8220 | **.8411** | .8410 | .8265 |
| `frozen_ksvd` | .7723 | .8112 | **.8335** | .8233 | .8299 | .8242 | .8209 | .7921 |
| `aksvd` | .7767 | .8047 | .8085 | .8089 | .8118 | **.8188** | .8045 | .7876 |
| `lcksvd` | .7342 | .7621 | .7553 | .7767 | .7693 | **.7965** | .7882 | .7820 |
| `online_dl` | .7888 | .7919 | .8141 | **.8156** | .7980 | .7626 | .7380 | .7064 |
| `bayesian` | .7590 | .7742 | **.7824** | .7485 | .7212 | .6755 | .6488 | .6432 |

**Observations.**

1. The Bayesian collapse is dramatic and monotone: **−0.174 ROC-AUC from 32 to 4096 atoms** on
   WL/RandomForest (0.7830 → 0.6087). It is not noise; every intermediate point lies on the slope.
2. Online-DL degrades almost as badly (0.8398 → 0.6883 on WL/RF), beginning after 256 atoms.
3. Each learner's peak width is the same under all three metrics — the optimum is a property of the
   learner, not of the metric.
4. **In *relative* terms the effect is smaller than it looks.** Expressed as a fraction of each
   learner's own peak, every learner is within 2 % of peak somewhere in 128–1024 atoms; only
   Bayesian (0.82 of peak at 4096) and Online-DL (0.87) are badly off. **Dictionary size is a
   robustness parameter more than a performance parameter** — for a well-chosen learner, anything in
   128–2048 is within noise of the best.

**Analysis.** The mechanism is the objective, again. A reconstruction or likelihood objective given
more atoms spends them on whatever dominates the input distribution — in a 95.4 %-majority corpus,
the majority class and the encoder's high-frequency structural noise. Extra atoms improve
reconstruction while *diluting* the discriminative content of the code. A Fisher term constrains
capacity allocation to preserve class scatter, so extra atoms continue to pay.

**Conclusion.** "More atoms is better" is true **only for FDDL and CS-FDDL.** For BPFA it is
actively harmful — an important warning, since BPFA's advertised selling point is that it *infers*
dictionary size automatically. On this corpus the inferred size is the wrong thing to trust, because
the posterior is fitted to reconstruction, not to separation.

### 6.1 Overcompleteness is not the explanation

The WL energy cut retains ≈ 1751 features, so dictionaries above ~1751 atoms are overcomplete. If
overcompleteness drove the collapse, all learners would turn down in the same place. They do not:
Bayesian turns down at 256 atoms (far *under*complete) while FDDL is still rising at 4096
(overcomplete). **The regime boundary is set by the objective, not by the completeness ratio.**

---

## 7. Finding 5 — The baseline verdict inverts with the classifier

This is the study's most important comparative result, and it is cleaner than the report states.

### 7.1 Best pipeline vs best baseline, per classifier (Minority-F1)

| Classifier | Best pipeline | Minority-F1 | Best baseline | Minority-F1 | Δ |
| --- | --- | --- | --- | --- | --- |
| **LogisticRegression** | `wl+fddl` @2048 | **0.4478** | graph2vec @1024 | 0.3709 | **+0.0769** |
| **LinearSVM** | `wl+fddl` @4096 | **0.4507** | graph2vec @2048 | 0.3837 | **+0.0670** |
| GradientBoosting | `wl+fddl` @4096 | 0.4103 | graph2vec @2048 | 0.4430 | −0.0327 |
| RandomForest | `wl_edge+aksvd` @4096 | 0.4895 | graph2vec @512 | 0.5195 | −0.0300 |

Both linear wins are **3–4× the Minority-F1 noise floor**; both losses are ≈ 1.5× it.

### 7.2 The matched-width duel (the fair version)

Comparing pipeline and graph2vec at *identical* embedding width and *identical* classifier removes
the "best-of-many-widths" advantage entirely.

**LogisticRegression** — best pipeline minus graph2vec at each width:

| Width | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Δ Minority-F1 | **+0.084** | **+0.081** | +0.060 | +0.064 | **+0.085** | +0.070 | **+0.080** |
| best pipeline | wl+fddl | wl+fddl | wl+frozen | wl+frozen | wl+frozen | wl+csfddl | wl+fddl |

**LinearSVM:**

| Width | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Δ Minority-F1 | +0.076 | **+0.118** | +0.064 | +0.056 | +0.057 | +0.081 | +0.065 |

**GradientBoosting:** +0.018, −0.033, −0.020, −0.007, −0.008, −0.021, −0.045 → loses at 6 of 7
**RandomForest:** −0.075, −0.061, −0.037, −0.029, −0.045, −0.029, −0.054 → loses at 7 of 7

**Conclusion.** The win under linear classifiers is not a cherry-picked width: it holds at **all
seven widths, for both linear classifiers, with the smallest margin (+0.056) still 2.6× the noise
floor.** Symmetrically, the loss under tree ensembles holds at 13 of 14 widths. This is a genuine,
symmetric, mechanism-driven result and should carry the report's headline exactly as stated.

### 7.3 Per-learner ledger vs graph2vec (WL encoder, matched width, Minority-F1)

Mean Δ, with matched-width win counts (out of 7) in parentheses:

| Learner | LogReg | LinearSVM | GradientBoosting | RandomForest |
| --- | --- | --- | --- | --- |
| `fddl` | **+0.069 (7/7)** | **+0.067 (7/7)** | −0.034 (0/7) | −0.057 (0/7) |
| `frozen_ksvd` | **+0.042 (6/7)** | +0.023 (6/7) | −0.067 (0/7) | −0.067 (0/7) |
| `csfddl` | **+0.039 (6/7)** | **+0.043 (7/7)** | −0.045 (0/7) | −0.083 (0/7) |
| `aksvd` | −0.018 (1/7) | −0.027 (0/7) | −0.088 (0/7) | −0.093 (0/7) |
| `online_dl` | −0.024 (0/7) | −0.033 (0/7) | −0.101 (0/7) | −0.136 (0/7) |
| `lcksvd` | −0.075 (0/7) | −0.081 (0/7) | −0.140 (0/7) | −0.173 (0/7) |
| `bayesian` | −0.103 (0/7) | −0.107 (0/7) | −0.203 (0/7) | −0.282 (0/7) |

### 7.4 How many pipeline cells beat the best baseline at all?

| Metric | LogReg | LinearSVM | GradientBoosting | RandomForest |
| --- | --- | --- | --- | --- |
| Minority-F1 | 16 / 125 (12.8 %) | 12 / 125 (9.6 %) | 0 / 125 | 0 / 125 |
| ROC-AUC | 8 / 130 (6.2 %) | 6 / 130 (4.6 %) | 0 / 130 | 1 / 130 (0.8 %) |
| Macro-F1 | 21 / 130 (16.2 %) | 18 / 130 (13.8 %) | 5 / 130 (3.8 %) | 5 / 130 (3.8 %) |

**Only four pipelines ever clear the bar on Minority-F1 — `wl+fddl`, `wl+csfddl`, `wl+frozen_ksvd`,
and `wl+aksvd` (once, at 2048/LogReg).** All are WL-encoded. 28 of 500 pipeline cells (5.6 %) beat
the best baseline available to their classifier.

**Conclusion.** The correct claim is narrow and defensible: *"Three WL-based sparse pipelines beat
every baseline under linear classifiers, at every matched width. No pipeline beats graph2vec under
a tree ensemble."* Any broader claim is not supported by this file.

### 7.5 The SF baseline is weak under linear models and strong under RandomForest

| Metric | SF value | % of pipeline cells beating it |
| --- | --- | --- |
| Minority-F1, LogReg | 0.1759 | 98 % (122/125) |
| Minority-F1, LinearSVM | 0.1818 | 98 % (122/125) |
| Minority-F1, GradientBoosting | 0.2833 | 70 % (88/125) |
| **Minority-F1, RandomForest** | **0.4405** | **15 % (19/125)** |

**Analysis.** SF is a 26-dimensional Laplacian-eigenvalue descriptor. Under a linear model it is
nearly useless (ROC-AUC 0.68); under a RandomForest it reaches Minority-F1 0.4405 — **better than
the best configuration of 18 of the 25 pipelines.** SF is also the only baseline evaluated with a
proper repeated protocol (10-fold), so it is the most trustworthy number in the baseline set.

**Conclusion.** SF+RandomForest is a much harder baseline than the report's framing suggests, and it
is essentially free to compute. Any claim of practical superiority must clear SF+RF (0.4405), not
just SF+linear (0.176). Seven pipelines do, at their best configuration: `wl_edge+aksvd` 0.4895,
`wl+fddl` 0.4839, `wl+frozen_ksvd` 0.4745, `fsm+csfddl` 0.4537, `wl+csfddl` 0.4535, `fsm+fddl`
0.4489, `wl+online_dl` 0.4448.

---

## 8. Finding 6 — What the dictionary actually buys, and a needed correction

### 8.1 The linearisation diagnostic

The gap between a linear classifier and a RandomForest on the *same* codes measures how much class
structure is present but not linearly accessible. WL encoder, each learner at its own best width:

| Learner | GradBoost | LinearSVM | LogReg | RandForest | RF − LogReg |
| --- | --- | --- | --- | --- | --- |
| `csfddl` | 0.3973 | 0.4474 | 0.4405 | 0.4535 | **+0.013** |
| `bayesian` | 0.2468 | 0.2789 | 0.2756 | 0.2999 | +0.024 |
| `fddl` | 0.4103 | 0.4507 | 0.4478 | 0.4839 | +0.036 |
| `frozen_ksvd` | 0.3706 | 0.4159 | 0.4352 | 0.4745 | +0.039 |
| `aksvd` | 0.3381 | 0.3669 | 0.3805 | 0.4335 | +0.053 |
| `lcksvd` | 0.3081 | 0.3046 | 0.3227 | 0.3896 | +0.067 |
| `online_dl` | 0.3706 | 0.3386 | 0.3443 | 0.4448 | **+0.101** |

**Observation.** CS-FDDL's codes are read essentially as well by a linear model as by a forest
(+0.013, below the noise floor); Online-DL's are not (+0.101 — nearly a quarter of its final
performance supplied by the classifier rather than by the representation).

**Analysis.** This is the mechanism behind Finding 5. The dictionary's job is to *linearise* the
class structure. A RandomForest performs that linearisation for free on any representation — which
is why graph2vec's dense embedding wins under RF but loses under LogReg. Where the downstream model
is linear, a discriminative dictionary supplies a function nothing else in the pipeline provides.

**Caveat (important).** The pattern is *not* monotone in quality: `bayesian`, the worst learner, has
the second-smallest gap (+0.024), simply because it is uniformly bad and there is little structure
for a forest to recover. **RF − LogReg measures entanglement, not quality**, and must be read
alongside the absolute level.

### 8.2 Correction: the "RandomForest compresses the ranking" claim does not hold

Report §4.2.3 states that under RandomForest the best-worst learner spread is 0.0789 while under
LogisticRegression it is 0.1673, "more than twice as large," concluding that a forest can "partially
compensate for a weak representation."

Re-deriving from the CSV:

| Way of computing | RF spread | LogReg spread |
| --- | --- | --- |
| Report's figure | 0.0789 | 0.1673 |
| Best over **all encoders** (report's method, correct min/max) | 0.0999 | 0.1673 |
| **WL only (encoder-consistent)** | **0.1840** | **0.1722** |

Two issues:

1. **Table 4.6 silently mixes encoders between the two columns of the same row.** Provenance of each
   cell, recomputed:

   | Learner | LogReg value from | RandomForest value from |
   | --- | --- | --- |
   | `csfddl` | 0.4405 — **wl** @1024 | 0.4537 — **fsm** @1024 |
   | `fddl` | 0.4478 — wl @2048 | 0.4839 — wl @4096 |
   | `frozen_ksvd` | 0.4352 — wl @512 | 0.4745 — wl @512 |
   | `lcksvd` | 0.3227 — wl @2048 | 0.3896 — wl @1024 |
   | `online_dl` | 0.3443 — wl @512 | 0.4448 — wl @64 |
   | `aksvd` | 0.3805 — **wl** @2048 | 0.4895 — **wl_edge** @4096 |
   | `bayesian` | 0.2805 — **fsm** @256 | 0.4106 — **gspan_cork** @64 |

   Three of seven rows compare a learner on one encoder against the same learner on a *different*
   encoder. The `bayesian` row is the worst case: it compares fsm/LogReg against
   gspan_cork/RandomForest, so its "+0.130 gap" is partly an encoder difference.

2. **The RF spread was taken between the wrong two learners** (0.4895 − 0.4106); the actual minimum
   under RF is `lcksvd` at 0.3896, giving 0.0999.

**Corrected conclusion.** Holding the encoder fixed at WL, the best-worst spread is **0.184 under
RandomForest and 0.172 under LogisticRegression** — statistically indistinguishable. **A
RandomForest does *not* compress the learner ranking.** What survives, and is well supported, is the
narrower per-learner statement in §8.1: *discriminative dictionaries produce codes whose class
information is already linearly accessible, so they lose little when the classifier is linear;
reconstruction-only dictionaries lose a great deal.* That claim should replace the spread argument.

### 8.3 LogisticRegression and LinearSVM are interchangeable

Across 125 matched (encoder, learner, atoms) cells: Pearson *r* = 0.967, Spearman ρ = 0.950, mean
|difference| = **0.0121** Minority-F1 — below the noise floor.

**Conclusion.** They are one experimental arm, not two. Reporting both doubles the table size and
adds no information; keep LinearSVM (marginally higher ceiling: 0.4507 vs 0.4478) and move LogReg to
an appendix. It also means the "linear classifier" finding rests on two *correlated* observations,
not two independent confirmations — worth stating honestly.

---

## 9. Finding 7 — Encoders: WL wins, and the `wl_edge` result is confounded

### 9.1 Paired comparison at matched (learner, atoms, classifier)

| Comparison | n cells | Wins | Mean Δ Minority-F1 | Median Δ |
| --- | --- | --- | --- | --- |
| `wl` vs `fsm` | 140 | 104 | **+0.0489** | +0.0441 |
| `wl` vs `gspan_cork` | 116 | 76 | **+0.0215** | +0.0206 |
| `fsm` vs `gspan_cork` | 80 | 43 | +0.0020 | +0.0037 |
| `wl` vs `wl_edge` | 16 | 9 | **+0.0006** | +0.0010 |

**Observations.** WL beats FSM by more than twice the noise floor and gspan+CORK by about one noise
floor. FSM and gspan+CORK are statistically tied. And — critically — **WL and WL-Edge are tied at
matched configuration.**

### 9.2 The `wl_edge` confound, quantified

Unconditioned means make `wl_edge` look like the best encoder by a wide margin (mean ROC-AUC 0.845
vs WL 0.792; best-cell ROC-AUC 0.8912 vs 0.8616). But:

- `wl_edge` was run with **only 2 of 7 learners** (FDDL, AKSVD) — the two strongest of their
  families, with the four weakest never run. Its mean is computed over a *pre-filtered, favourable*
  learner subset.
- `wl_edge` is the **only encoder swept out to 8192–20000 atoms**; WL stops at 4096. Its top rows are
  at widths WL never attempted.
- 36 of 40 `wl_edge` rows are **single-split, n_seeds = 1**.
- The 20 highest-width `wl_edge+fddl` rows have **no minority metrics at all**.

At the four widths where both encoders exist, the picture is far less impressive:

| Learner @ width | Classifier | wl_edge Min-F1 | wl Min-F1 | wl_edge ROC-AUC | wl ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| fddl @32 | LogReg | 0.2712 | **0.3205** | 0.7941 | **0.8079** |
| fddl @32 | RandomForest | 0.4129 | 0.4133 | **0.8350** | 0.8277 |
| aksvd @1024 | RandomForest | **0.4812** | 0.4335 | **0.8646** | 0.8397 |
| aksvd @2048 | LogReg | 0.3277 | **0.3805** | 0.8265 | 0.8191 |
| aksvd @4096 | RandomForest | **0.4895** | 0.4058 | **0.8614** | 0.8205 |

**Conclusion.** `wl_edge` shows a real advantage in exactly one place — **paired with AKSVD under a
RandomForest at large dictionaries** (+0.05 to +0.08 Minority-F1). Everywhere else it is level with
or behind plain WL. The apparent encoder-level superiority is a selection artefact of the learner
subset, the width grid and the single-split protocol. The report's conclusion-validity section
already acknowledges the compute constraint; this quantifies its consequence. **`wl_edge` should be
presented as a promising but under-evaluated encoder, not as the best one.**

### 9.3 Where the encoder actually matters

Encoder means restricted to a common grid (atoms ≤ 256, all learners) — Minority-F1:

| Encoder | GradBoost | LinearSVM | LogReg | RandForest |
| --- | --- | --- | --- | --- |
| `wl` | 0.3108 | **0.2895** | **0.2884** | 0.3882 |
| `wl_edge` | 0.3074 | 0.2918 | 0.2712 | **0.4129** |
| `gspan_cork` | **0.3161** | 0.2574 | 0.2578 | 0.3684 |
| `fsm` | 0.3023 | 0.2792 | 0.2741 | 0.3603 |

**Analysis.** WL's advantage is concentrated in the **linear** columns (+0.031 over gspan_cork under
LogReg) and largely disappears under GradientBoosting. This is the §8.1 mechanism one level up: WL's
high-dimensional count vector is *already* closer to linearly separable, so its benefit appears
wherever the downstream model cannot bend the space itself.

---

## 10. Finding 8 — Cost is uncorrelated with quality, and the frontier is Frozen K-SVD

336 rows carry `fit_seconds` (WL 192, FSM 140, wl_edge 4; gspan_cork has no timings).

### 10.1 Cost by learner

| Learner | Median (s) | Min | Max |
| --- | --- | --- | --- |
| **`frozen_ksvd`** | **16.0** | 10.8 | 157.7 |
| `bayesian` | 79.2 | 12.1 | 192.1 |
| `fddl` | 169.7 | 29.2 | 681.3 |
| `csfddl` | 176.8 | 29.7 | 6060.7 |
| `aksvd` | 242.1 | 47.3 | 9048.1 |
| `lcksvd` | 585.6 | 113.9 | **10505.8** |
| `online_dl` | 805.5 | 21.5 | 5659.6 |

Cost by dictionary size (WL, seconds):

| Learner | 32 | 128 | 512 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- |
| `frozen_ksvd` | 11 | 12 | **14** | 17 | **33** |
| `bayesian` | 12 | 16 | 35 | 102 | 192 |
| `online_dl` | 22 | 35 | 100 | 1096 | 4070 |
| `csfddl` | 30 | 67 | 275 | 1879 | **6061** |
| `aksvd` | 47 | 80 | 242 | 2208 | **9048** |
| `lcksvd` | 114 | 268 | 1683 | **10506** | — |

**Range: 700× between the cheapest and most expensive fit in the study.**

### 10.2 Cost buys nothing

Spearman correlation between fit time and best achieved quality (WL, per unique fit):
**ρ(cost, ROC-AUC) = −0.155; ρ(cost, Minority-F1) = −0.104.**

Spending more compute is, if anything, mildly *counter*productive — because the two most expensive
learners (LC-KSVD 10 506 s, Online-DL 5 660 s) are also two of the three worst-ranked.

### 10.3 The Pareto frontier is a single method

Every non-dominated point in (fit seconds, Minority-F1) belongs to `wl+frozen_ksvd`:

| Config | Fit (s) | Minority-F1 | ROC-AUC |
| --- | --- | --- | --- |
| `wl+frozen_ksvd` @32, RF | 10.8 | 0.3829 | 0.8045 |
| `wl+frozen_ksvd` @64, RF | 11.4 | 0.4133 | 0.8216 |
| `wl+frozen_ksvd` @128, RF | 11.9 | 0.4704 | 0.8520 |
| **`wl+frozen_ksvd` @512, RF** | **14.1** | **0.4745** | 0.8511 |

**Conclusion — the practical recommendation of this study.** `wl + frozen_ksvd @512` reaches
Minority-F1 **0.4745** in **14 seconds**. The best pipeline number in the whole file is 0.4895
(`wl_edge+aksvd` @4096, single split, untimed); the best 5-seed number is 0.4839 (`wl+fddl` @4096,
whose CS-FDDL sibling at the same width costs 6061 s). **Frozen K-SVD gives up ~0.010–0.015
Minority-F1 — below the noise floor — for a 100–700× reduction in fit cost.** For any deployment,
and certainly for the Molytica local-training server (where the report itself flags CPU training as
a limitation), this is the configuration to ship.

*Caveat:* `wl+fddl` timings exist only at 32 atoms (29.2 s), so FDDL's cost at 4096 atoms is
inferred from its CS-FDDL sibling. Recording timings for the remaining FDDL widths would close this
gap.

---

## 11. Finding 9 — Where the performance actually comes from (variance decomposition)

On the fully-crossed block — 3 encoders × 7 learners × 2 widths (128, 256) × 4 classifiers = 168
rows — the share of total variance explained by each factor (η²):

| Factor | η² for **Minority-F1** | η² for **ROC-AUC** |
| --- | --- | --- |
| Dictionary learner | **0.308** | **0.313** |
| Classifier | **0.294** | 0.084 |
| Encoder | 0.097 | **0.339** |
| Dictionary size (128 vs 256) | 0.001 | 0.001 |
| learner + classifier (2-way) | 0.645 | 0.439 |
| encoder + learner (2-way) | 0.512 | **0.754** |

On the full WL sweep (7 learners × 8 widths × 4 classifiers = 224 rows):

| Factor | η² Minority-F1 | η² ROC-AUC |
| --- | --- | --- |
| Dictionary learner | **0.555** | **0.538** |
| Classifier | 0.120 | 0.052 |
| Dictionary size | 0.056 | 0.064 |

**Observations and conclusions.**

1. **The learner is the dominant factor under every metric** — 31 % on the mixed block, 56 % once the
   encoder is held fixed. Choosing the right dictionary learner matters more than anything else you
   can do.
2. **The encoder governs *ranking* quality; the classifier governs *decision* quality.** Encoder
   explains 34 % of ROC-AUC variance but only 10 % of Minority-F1; classifier explains 29 % of
   Minority-F1 but only 8 % of ROC-AUC. This decomposition is new and directly actionable: *if your
   ROC-AUC is disappointing, change the encoder; if your F1 at the operating point is disappointing,
   change the classifier or the threshold, not the encoder.*
3. **Dictionary size explains ≈ 0 %** within the 128–256 band and only 5–6 % across the full 32–4096
   sweep. It is the least important of the four knobs. The report's Finding 2 is really about the
   *shape* of the size response (which is real, and diagnostic of the objective), not about size
   being a large lever — two different claims that should be worded differently.

### 11.1 How much does tuning inside one pipeline buy?

Range of Minority-F1 within each pipeline (across all widths and classifiers):

| Widest-ranging | Range | Narrowest-ranging | Range |
| --- | --- | --- | --- |
| `wl+online_dl` | 0.309 | `fsm+lcksvd` | 0.103 |
| `wl+frozen_ksvd` | 0.247 | `gspan_cork+online_dl` | 0.131 |
| `fsm+csfddl` | 0.222 | `fsm+bayesian` | 0.132 |
| `wl+aksvd` | 0.217 | `gspan_cork+lcksvd` | 0.136 |

Fixing the width at each pipeline's best and varying only the classifier still moves Minority-F1 by
0.03 – 0.18 (median 0.10). **Classifier choice alone is worth more than the entire
supervised-vs-unsupervised learner gap in many cells** — which is precisely why the classifier must
be held fixed in any baseline comparison, as §7.2 does.

---

## 12. Finding 10 — Every method lands at nearly the same operating point

Median minority precision / recall by classifier, across all 556 rows:

| Classifier | Minority precision | Minority recall | P/R ratio | Tuned threshold (median, range) |
| --- | --- | --- | --- | --- |
| LinearSVM | 0.231 | 0.377 | 0.64 | 0.116 (0.079 – 0.166) |
| LogisticRegression | 0.237 | 0.380 | 0.67 | 0.752 (0.662 – 0.868) |
| GradientBoosting | 0.297 | 0.350 | 0.86 | 0.650 (0.465 – 0.743) |
| RandomForest | 0.405 | 0.388 | 1.00 | 0.333 (0.220 – 0.498) |

**Observations.**

1. **Recall is nearly constant across every classifier and every method (0.35 – 0.39).** The
   threshold is tuned to maximise minority-F1 on the validation split, and that objective lands
   every pipeline at essentially the same recall. Differences between methods appear almost
   *entirely* as differences in **precision** (0.23 → 0.41).
2. Precision and recall are only weakly correlated across pipelines (*r* = 0.37) — methods are not
   simply sliding along one shared PR curve; better methods have genuinely better curves.
3. **The tuned thresholds are informative about calibration.** LogisticRegression's optimal
   threshold sits at **0.66 – 0.87, well above 0.5** — the opposite of the naive expectation for
   imbalanced data. This is the signature of `class_weight='balanced'`: balanced weighting already
   inflates minority probabilities, so the F1-optimal cut has to be pushed back *up*. RandomForest,
   which carries no such reweighting here, needs the threshold pushed *down* to 0.22 – 0.50, and
   LinearSVM's sigmoid-mapped margins need 0.08 – 0.17.

**Conclusion.** Two practical consequences. (a) In a screening deployment the choice among these
pipelines is a **precision** decision at roughly fixed recall — i.e. "how many wasted assays per
true hit," exactly what a wet-lab budget cares about. Reporting minority precision at a fixed recall
(say 0.38) would communicate this far better than F1 alone. (b) **The default 0.5 threshold is wrong
in both directions depending on the classifier** — threshold tuning is not an optional refinement
here, it is load-bearing, and the decision to tune it on a dedicated validation split is well
justified.

---

## 13. Finding 11 — Metric-driven model selection is nearly free

For each (encoder, learner, classifier) curve with ≥ 4 widths, does the ROC-AUC-optimal dictionary
size equal the Minority-F1-optimal one?

- 88 curves examined; the two criteria agree in **43 (49 %)**.
- Median Minority-F1 lost by selecting width on ROC-AUC instead: **0.0038**; mean 0.0112; worst case
  0.0499.

Loss by learner: `csfddl` 0.006 < `bayesian` 0.009 < `aksvd` 0.010 < `fddl` 0.010 < `lcksvd` 0.013 <
`frozen_ksvd` 0.013 < `online_dl` 0.018.

**Conclusion.** Although the two criteria disagree about *which* width is best about half the time,
the cost of that disagreement is below the noise floor in the median case. **Tuning on ROC-AUC is a
safe proxy** — convenient, because ROC-AUC is the more stable metric (SD 0.0147 vs 0.0215). The
exception is `online_dl`, where selecting on ROC-AUC costs 0.018 on average, because its ROC-AUC and
F1 curves genuinely peak in different places (256 vs 64 atoms).

---

## 14. Discrepancies between this analysis and the report

Listed so they can be fixed rather than defended.

| # | Report says | The CSV says | Severity |
| --- | --- | --- | --- |
| 1 | "For each graph encoder + dictionary learner combination we executed Monte-Carlo cross validation with 5 seeds" (§4.1) | 38 % of pipeline rows are 5-seed; 30 % are single-split; gspan_cork and 90 % of wl_edge have **no** repeated evaluation | **High** — affects every comparison involving those encoders |
| 2 | Table 4.6 compares LogReg and RandomForest "per learner at its best dictionary size" | 3 of 7 rows take the two columns from **different encoders** (`csfddl`, `aksvd`, `bayesian`) | **High** — invalidates the derived spread claim |
| 3 | "With RandomForest the best-worst spread is 0.0789 … with logistic regression 0.1673, more than twice as large" (§4.2.3) | Correct mixed-encoder values are 0.0999 / 0.1673; **encoder-consistent (WL) values are 0.1840 / 0.1722 — indistinguishable** | **High** — the conclusion ("RF compensates for weak representations") is not supported |
| 4 | CS-FDDL presented as a working imbalance contribution, ranked 2nd | Paired against plain FDDL it **loses**: 21/72 wins, −0.0122 Minority-F1, *p* = 0.0002 (−0.0243 on WL) | **High** — a real negative result that should be reported as such |
| 5 | "in a 6 %-minority corpus" (§4.2.2) | Base rate is **4.55 %** (4.47 – 4.76 %), consistent with §4.1's own "approximately 4.6 %" | Low — internal inconsistency |
| 6 | Figures 4.4 / 4.5 described as showing WL+FDDL / WL+CS-FDDL "outperform the baselines" under linear classifiers | Confirmed, and stronger than stated: they win at **all 7 matched widths**, for both linear classifiers | None — an under-claim worth strengthening |
| 7 | "Among the combinations tested, WL-Edge with FDDL achieved the best classification performance … Graph2Vec achieved the strongest overall" (§4.3.4) | True for ROC-AUC / Macro-F1 only; **the `wl_edge+fddl` rows at 2048–20000 atoms have no minority metrics recorded**, so no Minority-F1 comparison exists for them | Medium — the comparison is metric-limited and should say so |

**A gap, not a discrepancy.** The report opens by asking whether the pipelines beat "what you would
get without a dictionary at all," but **`all_results.csv` contains no no-dictionary ablation** — no
row with raw (or energy-cut) WL counts fed straight to the four classifiers. The graph2vec / SF /
GCN baselines are different *encoders*, not a dictionary ablation. Until that row exists, the study
cannot answer its own opening question. It is a cheap experiment — one encode, four classifier fits
— and would materially strengthen the contribution.

---

## 15. Threats to validity, restated from the data

1. **Protocol heterogeneity (§1.2).** Single-split rows are noisier and plausibly optimistic. Every
   `gspan_cork` conclusion and most `wl_edge` conclusions rest on one draw.
2. **Unequal precision inside the key contrast.** `wl+csfddl` (2 seeds) vs `wl+fddl` (5 seeds) is the
   project's own FDDL / CS-FDDL comparison, run at unequal precision.
3. **Incomplete factorial.** 5 of 28 encoder×learner combinations are missing, and the width grid
   differs by encoder (gspan_cork stops at 512, WL at 4096, wl_edge runs to 20000). Unconditioned
   encoder means are therefore not interpretable; only the matched-cell comparisons of §9.1 are.
4. **Missing minority metrics on the top-scoring rows (§1.3).**
5. **Single dataset.** All 556 rows are `nci_full`. Nothing here speaks to a different imbalance
   ratio, a different domain, or a multi-class problem — and the imbalance-specific findings
   (Frozen K-SVD's benefit, CS-FDDL's non-benefit) are precisely the ones most likely to move with
   the base rate.
6. **No hyperparameter tuning.** The report states this explicitly; it applies symmetrically to
   pipelines and baselines, so the *comparison* is fair, but every absolute number is a lower bound.
7. **Correlated metrics (§2.1) and correlated classifiers (§8.3)** mean that apparent agreement
   "across three metrics and four classifiers" is really agreement across two metrics and three
   classifiers.
8. **Multiple comparisons.** 520 pipeline cells were searched for the best. The matched-width duel
   (§7.2) is the analysis least exposed to this, which is why it should carry the headline.

---

## 16. Conclusions

### 16.1 What is established

1. **Supervision in the dictionary objective is the mechanism.** FDDL / CS-FDDL rank 1–2 in all 36
   conditions; FDDL beats AKSVD by +0.064 Minority-F1 on 65 of 68 matched cells. Sparsity alone does
   nothing — LC-KSVD and BPFA are sparse too, and they finish last.
2. **Three WL pipelines beat every baseline under linear classifiers, at every matched width**
   (+0.056 … +0.118 Minority-F1): `wl+fddl`, `wl+csfddl`, `wl+frozen_ksvd`. The mechanism is that a
   discriminative dictionary linearises the class structure — which is also why the advantage
   inverts under a RandomForest, which performs that linearisation itself.
3. **Frozen K-SVD is a validated, essentially free imbalance mechanism** (+0.032 Minority-F1 over
   AKSVD on WL, 28/32 matched wins, 14 s fit time), with a benefit profile across dictionary size
   that matches its theory.
4. **CS-FDDL's weighting is not validated on this corpus** and costs a small, consistent amount
   relative to plain FDDL; the leading hypothesis is under-convergence from the `1/max(w_i)`
   step-size reduction.
5. **BPFA is contraindicated for imbalanced graph data at scale** (ρ = −0.74 with atoms; −0.174
   ROC-AUC from 32 to 4096 atoms). Its automatic size selection optimises reconstruction, which is
   the wrong target here.
6. **Cost and quality are decoupled**, and the entire Pareto frontier is Frozen K-SVD.
7. **The GCN baseline is a degenerate control, not a competitor**, and SF+RandomForest (0.4405) is a
   far stronger bar than SF+linear (0.176) — it is the number to clear.

### 16.2 The four recommendations that follow

| Situation | Ship this | Why |
| --- | --- | --- |
| Linear / interpretable model required (Molytica's interpretability pipeline needs one) | **`wl` + `fddl` @2048–4096 + LinearSVM** | Minority-F1 0.4507; beats every baseline at every matched width |
| Compute- or latency-constrained (local trainer, CPU) | **`wl` + `frozen_ksvd` @512 + RandomForest** | 0.4745 in 14 s — within noise of the study's best |
| Maximum accuracy, classifier unconstrained | **graph2vec @512 + RandomForest** (0.5195) | The honest answer: no sparse pipeline beats it under a tree ensemble |
| Cheapest respectable baseline | **SF @26 + RandomForest** (0.4405) | 26 features, 10-fold validated, beats the best configuration of 18 of 25 pipelines |

### 16.3 The five highest-value pieces of missing work

1. Re-run the five `wl_edge+fddl` widths (2048–20000) **recording minority metrics** — these are the
   study's highest-scoring rows and currently cannot enter the primary comparison.
2. Add the **no-dictionary ablation** (energy-cut WL features → the four classifiers). Without it the
   study cannot answer its own framing question.
3. Re-run **CS-FDDL with a larger IPM iteration budget** to test the under-convergence hypothesis
   behind the negative result in §5.2. If the gap closes, the contribution is rehabilitated; if it
   does not, the negative result is confirmed and publishable.
4. Bring `gspan_cork` and `wl_edge` onto the **5-seed MC-CV protocol** so their numbers are
   comparable to the rest.
5. Extend `wl_edge` to the **remaining five learners** at matched widths, so the encoder comparison
   is not conditioned on a favourable learner subset.

---

*Generated from `analysis/all_results.csv` (556 rows). Every table above is directly recomputable
from that file; paired comparisons use the Wilcoxon signed-rank test over matched
(encoder, atoms, classifier) cells.*
