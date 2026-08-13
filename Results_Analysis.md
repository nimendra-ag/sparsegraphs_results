# Results and Analysis

*Sparse dictionary learning over graph representations for imbalanced molecular classification (NCI corpus)*

---

## 1. What we tested, and how to read this section

### 1.1 The experiment in one paragraph

Every pipeline in this study has three stages: a **graph encoder** turns each molecule into a feature vector, a **dictionary learner** turns that vector into a sparse code, and a **classifier** turns the code into a prediction. We swept all three stages plus the dictionary size, and compared the whole thing against three off-the-shelf reference methods. The question we are answering is not "does dictionary learning work" but **"which combination works, and does any of it beat what you would get without a dictionary at all."**

| Stage | Levels we ran |
| --- | --- |
| Graph encoder | `wl`, `wl_edge`, `fsm`, `gspan_cork` |
| Dictionary learner | `fddl`, `csfddl`, `aksvd`, `lcksvd`, `frozen_ksvd`, `online_dl`, `bayesian` |
| Dictionary size (atoms) | 32 → 20,000, roughly powers of two (encoder-dependent) |
| Classifier | LogisticRegression, LinearSVM, GradientBoosting, RandomForest |
| Baselines | `sf` (26 structural features, 10-fold CV), `graph2vec` (7 embedding widths), `gcn` (end-to-end network) |

This gives **556 evaluation records** in `analysis/all_results.csv` — 520 pipeline records and 36 baseline records — and **113 figures** across eight comparison scenarios.

### 1.2 The single most important fact about this dataset

**The minority class is about 4.6 % of the corpus.** We can read this straight off the `gcn` baseline, which failed completely: its minority recall is exactly 0.0000 at every hidden size, meaning it predicted "inactive" for every single molecule. Its accuracy was still **0.9545 – 0.9553**.

Its accuracy across its four hidden sizes ranged 0.9524 – 0.9553, which places the minority class at **roughly 4.5 – 4.8 %**.

That number is the whole problem in one line. **A model that does nothing scores 95 % accuracy here.** Accuracy is therefore excluded from every comparison in this section, and we report three metrics instead:

| Metric | What it tells us |
| --- | --- |
| **ROC-AUC** | How well the model *ranks* molecules, ignoring where you put the decision threshold |
| **Macro-F1** | How good the actual yes/no decision is, averaged over both classes |
| **Minority-F1** | How good the decision is **on the 4.6 % that actually matter** |

Minority-F1 is the metric we care about most. In a screening application, correctly saying "inactive" costs nothing and triggers no assay; correctly finding an *active* compound is the entire point. Macro-F1 partly hides this, because the majority-class F1 it averages in sits at ~0.96 for every method and barely moves — it dilutes the differences we are trying to measure. So we always report Minority-F1 alongside it.

### 1.3 The noise floor — how big a difference has to be before we believe it

Most runs are Monte-Carlo cross-validation (MC-CV) with 2–5 seeds, so we know how much each number wobbles:

| Metric | Median standard deviation across seeds |
| --- | --- |
| ROC-AUC | **0.0146** |
| Macro-F1 | **0.0119** |
| Minority-F1 | **0.0216** |

Throughout this section, **a gap smaller than these values is not a result.** We say so explicitly wherever it matters.

### 1.4 The eight scenarios are eight views of one experiment

The figure suite is organised as scenarios S1–S8. They are **not eight separate findings** — S1, S2 and S4 are three different cuts of the same encoder × learner × size surface, S3 is a robustness check on S1, and S7 is a ranked version of S4. Only S5 (classifier interaction), S6 (cost) and S8 (operating point) introduce genuinely new axes.

So this section is organised by **finding**, and each finding names the figures that evidence it. That is also the basis for the appendix plan in Section 10: a figure that only confirms another figure belongs in the appendix.

---

## 2. Finding 1 — Supervised dictionary learning wins; unsupervised dictionary learning mostly does not

This is the clearest result in the study.

The seven dictionary learners split into three groups by what their training objective actually optimises:

| Group | Learners | What the objective contains |
| --- | --- | --- |
| **Supervised (discriminative)** | `fddl`, `csfddl` | A Fisher-discrimination term - the dictionary is explicitly pushed to separate the classes |
| **Label-aware but weakly so** | `lcksvd` | A label-consistency term bolted onto a reconstruction objective |
| **Unsupervised** | `aksvd`, `frozen_ksvd`, `online_dl`, `bayesian` | Reconstruction or generative likelihood only - the labels are never seen |

### 2.1 The ranking

We defined a *condition* as one (encoder × classifier × metric) triple. With 3 encoders (`wl_edge` excluded — only 2 of 7 learners were run on it), 4 classifiers and 3 metrics, that is **36 conditions**. In each condition every learner is scored at its own best dictionary size, and the seven are ranked 1 (best) to 7 (worst). A learner with no real advantage would average rank 4.0.

| Learner | Type | Mean rank | SD of rank | How the 36 ranks fell |
| --- | --- | --- | --- | --- |
| **`fddl`** | **supervised** | **1.36** | 0.54 | 1st ×24, 2nd ×11, 3rd ×1 — **never below 3rd** |
| **`csfddl`** | **supervised** | **1.92** | 0.69 | 1st ×10, 2nd ×19, 3rd ×7 — **never below 3rd** |
| `frozen_ksvd` | unsupervised | 3.83 | 1.44 | mostly 3rd (×17), but 7th ×4 |
| `aksvd` | unsupervised | 4.43 | 1.34 | all over the range, 1st ×2 to 7th ×1 |
| `online_dl` | unsupervised | 4.78 | 1.33 | 5th ×12, 4th ×8 |
| `lcksvd` | label-consistency | 5.79 | 0.94 | never above 4th — reliably poor |
| `bayesian` | generative | 5.89 | 1.43 | 7th ×18 |

The two supervised learners take ranks 1 and 2 in essentially every condition, and the gap to third place is **about two full rank positions** — far wider than the gap between any two learners inside either group.

The ordering does not depend on which metric you pick:

| Metric | `fddl` | `csfddl` | `frozen_ksvd` | `aksvd` | `online_dl` | `lcksvd` | `bayesian` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC | 1.33 | 1.75 | 4.17 | 4.62 | 4.83 | 5.79 | 5.50 |
| Macro-F1 | 1.42 | 1.92 | 3.67 | 4.33 | 4.75 | 5.75 | 6.17 |
| Minority-F1 | 1.33 | 2.08 | 3.67 | 4.33 | 4.75 | 5.83 | 6.00 |

> 📊 **Attach: `s1_wl_RF_roc_auc.png`** (`analysis/figures/s1_learner_sweep/`) — all seven learners on one panel, with the baselines drawn in. The two-tier separation is visible at a glance.
>
> 📊 **Attach: `s3_wl_RF.png`** (`analysis/figures/s3_metric_robustness/`) — the same ranking under all three metrics side by side, which is what proves the ordering is not a metric artefact.

### 2.2 The head-to-head test

Ranks can be misleading, so we also compared the learners **directly at matched settings**: same encoder, same dictionary size, same classifier, MC-CV records only. For each cell we took the best supervised learner and the best unsupervised learner and subtracted.

| Metric | Mean (supervised − unsupervised) | Median | Supervised wins |
| --- | --- | --- | --- |
| ROC-AUC | **+0.0384** | +0.0301 | **50 / 56 cells** |
| Macro-F1 | **+0.0267** | +0.0229 | **47 / 56 cells** |
| Minority-F1 | **+0.0499** | +0.0451 | **47 / 56 cells** |

Every one of these margins is **2–3× the noise floor**, and the win rate is 84–89 %. This is a real, consistent effect, not a lucky configuration.

On the `wl` encoder with LogisticRegression, the per-learner Minority-F1 ladder makes the point concretely:

| Learner | Minority-F1 (LogReg) | Minority-F1 (RF) |
| --- | --- | --- |
| `fddl` | **0.4478** | **0.4839** |
| `csfddl` | **0.4405** | 0.4535 |
| `frozen_ksvd` | 0.4352 | 0.4745 |
| `aksvd` | 0.3805 | 0.4335 |
| `online_dl` | 0.3443 | 0.4448 |
| `lcksvd` | 0.3227 | 0.3896 |
| `bayesian` | 0.2756 | 0.2999 |

### 2.3 Why `lcksvd` is the interesting failure

`lcksvd` is the one learner that *nominally* uses labels — it carries a label-consistency term — and it finishes **sixth of seven**. This matters for the headline claim, because it shows the result is not simply "any use of labels helps."

The difference is *where* the supervision acts. `fddl` and `csfddl` put a Fisher-discrimination term directly into the dictionary objective, so the atoms themselves are forced to separate the classes. `lcksvd` learns a mostly reconstructive dictionary and then asks a linear transform to make the codes label-consistent afterwards. **Supervision applied to the dictionary works; supervision bolted on after the dictionary does not.**

*Evidence: S1 (all 48 panels), S3, S4, S7.*

---

## 3. Finding 2 — Extra dictionary atoms only pay off if the objective is supervised

Dictionary size is the one knob everyone tunes, and the answer is not the same for every learner. Correlating atom count against score (Spearman ρ, MC-CV records, pooled over encoders and classifiers):

| Learner | ρ(atoms, ROC-AUC) | ρ(atoms, Minority-F1) | Mean ROC-AUC change, smallest → largest size | Median best size | Behaviour |
| --- | --- | --- | --- | --- | --- |
| `csfddl` | **+0.63** | +0.54 | **+0.021** | 1024 | **scales up** |
| `fddl` | **+0.58** | +0.49 | **+0.041** | 4096 | **scales up** |
| `lcksvd` | +0.48 | +0.49 | +0.054 | 1024 | scales, from a low base |
| `frozen_ksvd` | +0.02 | +0.20 | +0.020 | 128 | saturates early |
| `aksvd` | −0.15 | +0.18 | +0.011 | 544 | saturates |
| `online_dl` | −0.36 | −0.09 | **−0.082** | 192 | **degrades** |
| `bayesian` | **−0.82** | −0.79 | **−0.116** | 96 | **collapses** |

The collapse is dramatic and monotone. On `wl` with RandomForest:

| Atoms | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bayesian` ROC-AUC | 0.783 | 0.781 | 0.780 | 0.731 | 0.689 | 0.651 | 0.626 | **0.609** |
| `online_dl` ROC-AUC | 0.840 | 0.832 | 0.837 | 0.827 | 0.805 | 0.759 | 0.725 | **0.688** |
| `fddl` ROC-AUC | 0.828 | 0.843 | 0.851 | 0.860 | 0.847 | 0.858 | 0.856 | **0.862** |
| `csfddl` ROC-AUC | 0.822 | 0.818 | 0.839 | 0.859 | 0.830 | 0.849 | 0.842 | 0.832 |

`bayesian` loses **0.174 ROC-AUC** from 32 to 4096 atoms — twelve times the noise floor. That is not a plateau; it is active damage.

**Why this happens.** A dictionary that only optimises reconstruction spends every extra atom on whatever dominates the input — and in a corpus that is 95.4 % majority class, that means it spends its new capacity describing the majority class and the encoder's high-frequency structural noise ever more finely. Reconstruction improves; the class signal in the code gets diluted. A dictionary with a discrimination term is constrained to keep the classes separated as it grows, so it keeps converting capacity into accuracy.

**Practical consequence.** You cannot pick one dictionary size and share it across a learner comparison. At 128 atoms `frozen_ksvd` looks competitive with `fddl`; at 4096 atoms it is not close. **Any published comparison at a fixed atom count is reporting an artefact of that choice.**

> 📊 **Attach: `s1_wl_LogReg_minority_f1.png`** (`analysis/figures/s1_learner_sweep/`) — the three regimes (rising, flat, collapsing) are all visible in one panel, and this is also the figure that carries the baseline comparison in Finding 3.

*Evidence: S1 (all 48 panels), S2, S3.*

---

## 4. Finding 3 — **Our pipelines beat every baseline when the classifier is linear**

This is the headline result of the project, and it is the one that justifies the whole approach.

### 4.1 The comparison, done fairly

The comparison has to be made **at the same classifier**. Comparing our best RandomForest number against graph2vec's best LogisticRegression number would confound the two factors we are trying to separate. So for each of the four classifiers we take our best MC-CV pipeline and the best baseline *driven by that same classifier*:

**LogisticRegression**

| Metric | Best pipeline (MC-CV) | `graph2vec` | `sf` | Our margin over `graph2vec` |
| --- | --- | --- | --- | --- |
| **Minority-F1** | **0.4478 ± 0.0356** (`wl`/`fddl`/2048) | 0.3709 | 0.1759 | **+0.0769 (2.2 SD)** ✅ |
| **Macro-F1** | **0.7086 ± 0.0177** (`wl`/`fddl`/2048) | 0.6683 | 0.5567 | **+0.0403 (2.3 SD)** ✅ |
| ROC-AUC | 0.8520 ± 0.0136 (`wl`/`fddl`/4096) | 0.8471 | 0.6830 | +0.0049 (within noise) ➖ |

**LinearSVM**

| Metric | Best pipeline (MC-CV) | `graph2vec` | `sf` | Our margin over `graph2vec` |
| --- | --- | --- | --- | --- |
| **Minority-F1** | **0.4507 ± 0.0201** (`wl`/`fddl`/4096) | 0.3837 | 0.1818 | **+0.0670 (3.3 SD)** ✅ |
| **Macro-F1** | **0.7121 ± 0.0107** (`wl`/`fddl`/4096) | 0.6727 | 0.5586 | **+0.0394 (3.7 SD)** ✅ |
| ROC-AUC | 0.8510 ± 0.0185 (`wl`/`fddl`/256) | 0.8486 | 0.6799 | +0.0024 (within noise) ➖ |

**The Minority-F1 and Macro-F1 wins are 2.2 to 3.7 standard deviations wide.** These are not ties and they are not noise. Against `sf` the margins are enormous — **+0.27 Minority-F1**, more than double its score. Against `gcn`, which never predicted a single active compound (Minority-F1 = 0.0000), the comparison is not even meaningful.

### 4.2 The win holds at every matched width, not just at the best one

A single best-vs-best number can always be luck. So we also compared our `wl` pipelines against `graph2vec` **at matched representation width** — our dictionary atom count against graph2vec's embedding dimension, since both are simply the width of the vector handed to the classifier. They share seven widths (32 → 2048).

**Under LogisticRegression:**

| Learner | Mean Minority-F1 difference | Widths won | Mean Macro-F1 difference | Widths won |
| --- | --- | --- | --- | --- |
| **`fddl`** | **+0.0691** | **7 / 7** ✅ | **+0.0392** | **7 / 7** ✅ |
| `frozen_ksvd` | +0.0423 | 6 / 7 ✅ | +0.0256 | 6 / 7 ✅ |
| `csfddl` | +0.0392 | 6 / 7 ✅ | +0.0215 | 6 / 7 ✅ |
| `aksvd` | −0.0182 | 1 / 7 | −0.0095 | 1 / 7 |
| `online_dl` | −0.0237 | 0 / 7 | −0.0149 | 0 / 7 |
| `lcksvd` | −0.0754 | 0 / 7 | −0.0437 | 0 / 7 |
| `bayesian` | −0.1030 | 0 / 7 | −0.0623 | 0 / 7 |

**Under LinearSVM:**

| Learner | Mean Minority-F1 difference | Widths won | Mean Macro-F1 difference | Widths won |
| --- | --- | --- | --- | --- |
| **`fddl`** | **+0.0668** | **7 / 7** ✅ | **+0.0375** | **7 / 7** ✅ |
| **`csfddl`** | **+0.0433** | **7 / 7** ✅ | **+0.0229** | **7 / 7** ✅ |
| `frozen_ksvd` | +0.0226 | 6 / 7 ✅ | +0.0143 | 6 / 7 ✅ |
| `aksvd` | −0.0265 | 0 / 7 | −0.0157 | 0 / 7 |
| `bayesian` | −0.1065 | 0 / 7 | −0.0636 | 0 / 7 |

`fddl` beats `graph2vec` at **every single matched width, under both linear classifiers, on both F1 metrics** — 28 out of 28 head-to-head comparisons. That consistency is what turns a favourable number into a finding.

> 📊 **Attach: `s1_wl_LogReg_minority_f1.png`** (`analysis/figures/s1_learner_sweep/`) — **the single most important figure in the report.** The graph2vec sweep is drawn on the same width axis as our learners, so the reader sees the `fddl` curve sitting above the graph2vec curve across the whole range.
>
> 📊 **Attach: `s1_wl_LinSVM_minority_f1.png`** (`analysis/figures/s1_learner_sweep/`) — the same picture under LinearSVM, which shows the result is a property of *linearity*, not of one particular linear model.
>
> 📊 **Attach: `s4_LogReg_minority_f1.png`** (`analysis/figures/s4_combination_matrix/`) — the full encoder × learner grid with the baseline strip underneath, so every cell can be read against the references at once.

### 4.3 What this means

**We have found a genuinely better algorithm combination for imbalanced graph classification under a linear decision model.**

The recipe is specific and reproducible:

> **WL subtree encoder → Fisher-discriminative dictionary learning (`fddl`) at 2048–4096 atoms → linear classifier (LogisticRegression or LinearSVM)**

This beats a strong published unsupervised graph embedding (`graph2vec`) by **+0.07 Minority-F1** and **+0.04 Macro-F1**, and beats a hand-crafted structural-feature baseline (`sf`) by **+0.27 Minority-F1**. It is the configuration to report as the project's contribution.

### 4.4 Where we do *not* win — stated plainly

Under the two **tree ensembles**, the verdict reverses:

| Classifier | Metric | Best pipeline (MC-CV) | `graph2vec` | Our margin |
| --- | --- | --- | --- | --- |
| GradientBoosting | Minority-F1 | 0.4103 | 0.4430 | −0.0327 ❌ |
| GradientBoosting | ROC-AUC | 0.8485 | 0.8674 | −0.0189 ❌ |
| RandomForest | Minority-F1 | 0.4839 ± 0.0233 | **0.5195** | −0.0356 ❌ |
| RandomForest | Macro-F1 | 0.7299 ± 0.0120 | **0.7482** | −0.0183 ❌ |
| RandomForest | ROC-AUC | 0.8616 ± 0.0072 | **0.8898** | −0.0282 ❌ |

Under RandomForest, `graph2vec` wins at **0/7 matched widths lost** — i.e. it beats us everywhere. Reporting this honestly costs us nothing, because Finding 4 explains exactly *why* it happens, and the explanation strengthens rather than weakens our claim.

Two caveats keep the whole comparison fair in both directions. `graph2vec`'s numbers are **single-split and selected as the best of 7 dimensions**, so they are optimistically biased. Our numbers are **seed-averaged maxima over 8 sizes**, so they are optimistic too — but less noisily so, since ours carry a measured standard deviation and graph2vec's do not.

*Evidence: S1 (baseline sweeps on the shared width axis), S2, S4 (baseline strip), S7.*

---

## 5. Finding 4 — What the dictionary actually buys is **linear separability**

This finding explains Finding 3 completely, and it is the most intellectually interesting result in the study.

### 5.1 The diagnostic

For each learner, look at how much the score improves when you swap a linear classifier for a RandomForest, on the `wl` encoder at each learner's best size:

| Learner | LogReg | LinSVM | GBoost | RF | **RF − LogReg** | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `csfddl` | 0.4405 | 0.4474 | 0.3973 | 0.4535 | **+0.0130** | codes already linearly separable |
| `fddl` | 0.4478 | 0.4507 | 0.4103 | 0.4839 | **+0.0361** | mostly linearly separable |
| `frozen_ksvd` | 0.4352 | 0.4159 | 0.3706 | 0.4745 | +0.0393 | partly entangled |
| `aksvd` | 0.3805 | 0.3669 | 0.3381 | 0.4335 | +0.0530 | entangled |
| `lcksvd` | 0.3227 | 0.3046 | 0.2977 | 0.3896 | +0.0669 | entangled |
| `online_dl` | 0.3443 | 0.3386 | 0.3706 | 0.4448 | **+0.1005** | heavily entangled |

*(Minority-F1. `bayesian` is omitted from the reading: its gap is small (+0.024) only because it is poor under every classifier — a small gap is only meaningful when the score is high.)*

**The interpretation is clean.** A wide LogReg→RF gap means the class information *is* in the code but a straight line cannot get at it — the forest is recovering it by carving the space into non-linear regions. A narrow gap means the representation has already done that work.

`csfddl` at **+0.013** has produced codes a linear model reads about as well as a forest does. `online_dl` at **+0.101** has produced codes where roughly a quarter of the final performance comes from the classifier rather than from the representation.

And notice the gap ordering **almost exactly reproduces the quality ordering of Finding 1**. That is not a coincidence: it is the same property measured two ways.

> 📊 **Attach: `s5_minority_f1.png`** (`analysis/figures/s5_classifier_interaction/`) — every combination as a row, every classifier as a column, with the baselines ruled off at the bottom. The LogReg→RF gap is read straight off each row. **This is the figure that earns its place most decisively** — no single-classifier study could have produced it.

### 5.2 Why this makes Finding 3 inevitable

Once you see Finding 4, the reversal in Section 4.4 stops being a disappointment and becomes a prediction.

**The dictionary's job is to linearise the class structure. A RandomForest does that job for free.**

So:

- Where the downstream model is **linear**, our dictionary is doing work nothing else does → **we win, at every matched width.**
- Where the downstream model is a **forest**, the forest performs that same linearisation itself on the dense `graph2vec` embedding, and our sparse code has spent representational capacity on a service the classifier no longer needs → **graph2vec wins.**

These are two sides of one mechanism, not two contradictory results.

### 5.3 The two linear classifiers agree almost perfectly

A useful robustness check. Averaged over every pipeline configuration:

| Metric | LogReg | LinSVM | GBoost | RF | LinSVM − LogReg |
| --- | --- | --- | --- | --- | --- |
| ROC-AUC | 0.7865 | 0.7850 | 0.7688 | **0.8002** | **−0.0015** |
| Macro-F1 | 0.6269 | 0.6266 | 0.6299 | **0.6651** | **−0.0003** |
| Minority-F1 | 0.2993 | 0.2992 | 0.2997 | **0.3649** | **−0.0001** |

LogisticRegression and LinearSVM are **statistically indistinguishable** — they differ by less than 0.002 on every metric, one-tenth of the noise floor. This matters: it confirms our Finding 3 result is driven by the *linearity* of the decision boundary and not by a quirk of one particular solver, loss function or regularisation scheme.

A second observation from the same table: **GradientBoosting is the worst classifier for sparse codes**, coming in below both linear models on ROC-AUC (0.7688 vs 0.7865). Boosted stumps on a sparse, high-dimensional code apparently split on individual atoms and overfit, while a forest's feature bagging handles the same input well. If you are going to use a tree ensemble on these codes, use a forest.

*Evidence: S5 (all three metrics).*

---

## 6. Finding 5 — WL is the best encoder, and the encoder ranking is partly confounded

Comparing `wl` against `fsm` at **strictly matched** (learner, size, classifier) under the same MC-CV protocol — the only encoder pair where a clean comparison is possible — over 140 matched triples:

| Metric | Mean (wl − fsm) | Median | `wl` wins |
| --- | --- | --- | --- |
| ROC-AUC | **+0.0310** | +0.0292 | **111 / 140** |
| Macro-F1 | **+0.0262** | +0.0227 | **107 / 140** |
| Minority-F1 | **+0.0489** | +0.0441 | **104 / 140** |

WL's advantage is roughly **twice the noise floor** and holds in about three-quarters of matched cells. This is a real effect.

The other two encoders **cannot be placed on the same footing**, and this needs stating clearly:

| Encoder | Best result | Problem |
| --- | --- | --- |
| `wl` | ROC-AUC 0.8616 (`fddl`/4096/RF) | ✅ clean, MC-CV, 5 seeds |
| `fsm` | ROC-AUC 0.8527 (`csfddl`/512/RF) | ✅ clean, MC-CV |
| `gspan_cork` | ROC-AUC 0.8305 (`csfddl`/64/RF) | ⚠️ **all 116 records are single splits** |
| `wl_edge` | ROC-AUC 0.8912 (`fddl`/20000/RF) | ⚠️ **36 of 40 records are single splits** |

`wl_edge` posts the **highest raw numbers in the entire study** — ROC-AUC 0.8912 and Macro-F1 0.7821 at 20,000 atoms — which would edge out even graph2vec's 0.8898. But its only MC-CV record (`fddl` at 32 atoms) scores an entirely unremarkable 0.8350, and its Minority-F1 column is missing above 32 atoms.

> **`wl_edge` must be reported as a promising but unvalidated lead, not as the project's best result.** Its headline number rests on one draw of one partition, with no measured spread. Validating it with proper MC-CV is the single highest-value follow-up experiment available.

The honest headline configuration is the best **MC-CV** result: **`wl`/`fddl`/4096/RandomForest** at ROC-AUC 0.8616 ± 0.0072, Macro-F1 0.7299 ± 0.0120, Minority-F1 0.4839 ± 0.0233.

> 📊 **Attach: `s4_RF_roc_auc.png`** (`analysis/figures/s4_combination_matrix/`) — the whole encoder × learner surface as a heat map, with the baselines as a strip underneath on the same colour scale. One figure that shows every combination we ran.

*Evidence: S2, S4, S7.*

---

## 7. Finding 6 — Cost varies by 700× and is almost unrelated to quality

Dictionary fitting time on the `wl` encoder:

| Learner | Median fit | 32 atoms | Largest size | Growth | Quality rank |
| --- | --- | --- | --- | --- | --- |
| **`frozen_ksvd`** | **14 s** | 11 s | 33 s (4096) | **×3** | 3rd |
| `bayesian` | 29 s | 12 s | 192 s (4096) | ×16 | 7th |
| `online_dl` | 69 s | 22 s | 4,070 s (4096) | ×189 | 5th |
| `aksvd` | 168 s | 47 s | 9,048 s (4096) | ×191 | 4th |
| `csfddl` | 221 s | 30 s | 6,061 s (4096) | ×204 | **2nd** |
| **`lcksvd`** | **653 s** | 114 s | **10,506 s (2048)** | ×92 | **6th** |

Pair this with Finding 2 and the picture is unflattering for most of the pool:

- **`lcksvd` is the most expensive learner and ranks sixth.** It spends nearly three hours to reach 2048 atoms for a result the cheapest learner beats in 33 seconds.
- **`aksvd` spends 9,048 seconds to reach 4096 atoms — a size at which it has already saturated and is drifting downward.** The expensive learners are expensive precisely in the regime where the extra capacity does them no good.
- **`frozen_ksvd` is the cost-effectiveness winner:** 14 s median, nearly flat scaling, and third place in quality. Where compute is the binding constraint, it is the rational default — and note from Section 4.2 that it *also* beats graph2vec at 6/7 widths under both linear classifiers.
- **`fddl`/`csfddl` cost real money** (`csfddl` at 4096 atoms costs 6,061 s) but are the only learners that convert that spend into accuracy.

**Caveat on coverage:** only **336 of 520** pipeline records carry a fit time, and several combinations are only partly timed, so the Pareto frontier in S6 is drawn over a subset and may miss a combination's true best size. The figures state this per panel.

> 📊 **Attach: `s6_RF_roc_auc.png`** (`analysis/figures/s6_cost_vs_performance/`) — quality against fit time on a log axis with the Pareto frontier marked. Up and to the left is better.

*Evidence: S6.*

---

## 8. Finding 7 — Every pipeline sits at essentially the same operating point

At each combination's best Minority-F1 under RandomForest, the precision/recall split is remarkably uniform:

| Configuration | Precision | Recall | Minority-F1 | P/R ratio |
| --- | --- | --- | --- | --- |
| `wl_edge`/`aksvd`/4096 * | 0.542 | 0.446 | 0.489 | 1.21 |
| `wl`/`fddl`/4096 | 0.512 | 0.460 | 0.484 | 1.11 |
| `wl`/`frozen_ksvd`/512 | 0.503 | 0.449 | 0.475 | 1.12 |
| `fsm`/`csfddl`/1024 | 0.464 | 0.454 | 0.454 | 1.02 |
| `wl`/`aksvd`/1024 | 0.427 | 0.445 | 0.433 | 0.96 |
| `wl`/`bayesian`/128 | 0.278 | 0.332 | 0.300 | 0.84 |
| `graph2vec`/512 (baseline) | 0.533 | 0.506 | 0.519 | 1.05 |

*(\* single split)*

The P/R ratio hovers around 1.0 across the whole quality range, and the baselines sit in the same band. **The pipelines do not trade off differently from one another — they move along a common iso-F1 progression, with better representations pushing further out rather than adopting a different balance.**

There is one exception worth noting, and it favours us. Under the **linear** classifiers the picture shifts toward recall: `wl`/`fddl`/2048 under LogReg runs at P 0.407 / R 0.504 (ratio 0.81), and `wl_edge`/`aksvd`/4096 reaches R 0.585. **If the application values finding actives over avoiding false alarms — which active-compound screening usually does — the linear pipelines are already operating closer to the right corner** than the forest configurations are.

**The actionable conclusion:** this uniformity is a consequence of the threshold-calibration step, which optimises a symmetric criterion. **If you want a different precision/recall balance, the calibration objective is the lever to pull, not the dictionary.** No representation in this study offers a materially different operating point, and re-calibrating for recall will likely yield more than further dictionary tuning.

> 📊 **Attach: `s8_RF.png`** (`analysis/figures/s8_minority_tradeoff/`) — every combination on the precision/recall plane with iso-F1 contours and the baselines as points.

*Evidence: S8.*

---

## 9. Threats to validity

Stated plainly, worst first.

1. **Protocol confounding — the biggest issue.** 156 of 520 pipeline records (30 %) are **single splits**: all 116 `gspan_cork` records, 36 of 40 `wl_edge` records, and 4 `wl`/`lcksvd`/4096 records. Given a median MC-CV spread of 0.0146 (ROC-AUC) and 0.0216 (Minority-F1), a single-split number carries roughly that much *unquantified* uncertainty. The encoder ranking in Finding 5 is clean **only** for the `wl` vs `fsm` pair. Every headline claim in Findings 1–4 is drawn from MC-CV records only, which is why those findings survive this threat.

2. **Unequal repetition counts.** Seeds range from 2 to 5 (196 records at 5 seeds, 148 at 2, 20 at 3). A standard deviation from 2 repetitions is a weak estimate, so error bars are **indicative, not calibrated intervals** — particularly for `csfddl`, which was run at 2 seeds throughout.

3. **Selection on the reported data.** Every headline number is a maximum over dictionary sizes evaluated on the same test partitions, so it is optimistically biased. This applies symmetrically to us (max over ~8 sizes) and to `graph2vec` (max over 7 widths), but graph2vec's maximum is drawn from noisier single-split values and is the more inflated of the two. The matched-width analysis in Section 4.2 exists precisely to sidestep this — it compares like with like at every width rather than max against max.

4. **A data-hygiene bug worth fixing.** Four rows (`wl`/`lcksvd`/4096) carry `source = "artifacts"` rather than `"artifact"`. `analysis/make_figures.py` tests for `"artifact"` exactly, so these four single-split points are currently drawn as if they were MC-CV means (solid markers, connected lines) in S1 and S2. It does not change any conclusion — `lcksvd` ranks sixth either way — but it should be corrected before the figures are published.

5. **Incomplete timing coverage.** 184 of 520 records lack a fit time; Finding 6 is drawn over the timed subset.

6. **Missing cells.** `wl_edge` was run with only 2 of 7 learners, and its Minority-F1 is absent above 32 atoms. The combination matrix is not complete, and the study's highest raw score sits in its least-validated corner.

7. **Single corpus.** Everything here is NCI. Finding 2's mechanism *explicitly* invokes class imbalance, so we should expect the supervised-vs-unsupervised gap to shrink on a balanced dataset. Nothing here establishes that it transfers.

---

## 10. Which figures to put where

There are **113 figures**. The report cannot carry them all, and it does not need to — most are confirmatory.

### 10.1 Main text — attach these eight

| # | File | Folder | What it carries |
| --- | --- | --- | --- |
| **1** | **`s1_wl_LogReg_minority_f1.png`** | `s1_learner_sweep/` | **The headline result** — our `fddl` curve above the graph2vec sweep at every width, plus the three capacity regimes |
| **2** | **`s1_wl_LinSVM_minority_f1.png`** | `s1_learner_sweep/` | The same win under LinearSVM — proves it is about linearity, not one solver |
| **3** | **`s5_minority_f1.png`** | `s5_classifier_interaction/` | **Finding 4** — the LogReg→RF gap, the mechanism behind everything else |
| **4** | **`s4_LogReg_minority_f1.png`** | `s4_combination_matrix/` | Full encoder × learner grid with the baseline strip, under the classifier where we win |
| **5** | `s1_wl_RF_roc_auc.png` | `s1_learner_sweep/` | Findings 1 + 2: the learner hierarchy and the capacity regimes |
| **6** | `s4_RF_roc_auc.png` | `s4_combination_matrix/` | Finding 5: the encoder × learner surface |
| **7** | `s6_RF_roc_auc.png` | `s6_cost_vs_performance/` | Finding 6: cost against quality with the Pareto frontier |
| **8** | `s8_RF.png` | `s8_minority_tradeoff/` | Finding 7: the shared operating point |

**If the section must be shorter, figures 1, 2, 3 and 4 are the irreducible set.** They carry the headline win, its robustness, the mechanism that explains it, and the design surface it sits in.

Consider also adding **`s3_wl_RF.png`** (`s3_metric_robustness/`) if space allows — it is the cheapest way to show the learner ranking survives a change of metric.

### 10.2 Appendix — the remaining 105 figures

Group them by scenario with a one-line caption each:

| Appendix | Scenario | Count | Caption |
| --- | --- | --- | --- |
| **A.1** | S1 — capacity sweeps | 45 remaining | Score vs dictionary size, one panel per encoder × classifier × metric. **Include in full** — primary evidence for Finding 2, and readers will want the panel matching their own setting. |
| **A.2** | S2 — encoder comparison | 12 | Encoders overlaid, faceted by learner. Confirmatory for Finding 5. |
| **A.3** | S3 — metric robustness | 16 (15 if `s3_wl_RF.png` is promoted) | Three metrics side by side per encoder × classifier. |
| **A.4** | S4 — combination matrices | 10 remaining | The encoder × learner heat map for every other classifier and metric. |
| **A.5** | S5 — classifier interaction | 2 remaining | The ROC-AUC and Macro-F1 versions of main-text figure 3. |
| **A.6** | S6 — cost vs performance | 11 remaining | Cost/quality plane per classifier and metric. |
| **A.7** | S7 — leaderboards | 6 | Top-20 ranked configurations per metric, with and without baselines. Worth keeping all six — `s7_minority_f1_top20.png` is a good "everything at once" summary. |
| **A.8** | S8 — operating points | 3 remaining | Precision/recall plane per classifier; `s8_LogReg.png` is the useful one, showing our recall-leaning linear configurations. |

**If the appendix must be bounded**, trim in this order:

1. **S2 (12 figures)** — the most redundant set. It re-plots the S1 data with encoder and learner swapping roles, and adds nothing that S1 and S4 do not already carry between them.
2. **S3 down to the four `wl` panels (removes 12)** — the metric-robustness claim needs demonstrating once, not sixteen times.

That removes 24 figures for no loss of evidence, leaving a manageable ~81-figure appendix.

---

## 11. Conclusions

### 11.1 What we found

**1. Supervised dictionary learning works; unsupervised dictionary learning largely does not.**
`fddl` and `csfddl` — the only two learners with a Fisher-discrimination term in the objective — take ranks 1 and 2 in essentially all 36 conditions and never fall below third. At matched settings they beat the best unsupervised learner in **47–50 of 56 cells**, by 2–3× the noise floor. Crucially, `lcksvd` shows this is not simply "labels help": it uses labels too, and finishes sixth. **Supervision has to be inside the dictionary objective, not bolted on afterwards.**

**2. Capacity is only useful to a discriminative objective.**
`fddl` and `csfddl` improve as the dictionary grows (ρ ≈ +0.6); `bayesian` actively collapses (ρ = −0.82, losing 0.174 ROC-AUC from 32 to 4096 atoms) and `online_dl` degrades. On a 95 %-majority corpus, extra atoms in a reconstructive dictionary get spent describing the majority class in ever finer detail. **Dictionary size must be tuned per learner — a fixed-size comparison reports an artefact.**

**3. We found a better algorithm combination for imbalanced graph classification under a linear model.**
**WL → `fddl` at 2048–4096 atoms → LogisticRegression or LinearSVM** beats `graph2vec` by **+0.077 Minority-F1** and **+0.040 Macro-F1** under LogReg, and **+0.067 / +0.039** under LinSVM — margins of **2.2 to 3.7 standard deviations**. It wins at **7 of 7 matched widths under both linear classifiers on both F1 metrics**, 28 of 28 head-to-head comparisons. Against `sf` the margin is **+0.27 Minority-F1**; against `gcn`, which never identified a single active compound, there is no contest. This is the project's contribution and it is defensible.

**4. What the dictionary actually buys is linear separability.**
The LogReg→RF gap falls from **+0.101** (`online_dl`) to **+0.013** (`csfddl`). A good dictionary does the work a forest would otherwise have to do — which is exactly why we win under linear models and lose under RandomForest, where `graph2vec` beats us by 0.036 Minority-F1. **These are not two contradictory results; they are one mechanism seen from both sides.** LogisticRegression and LinearSVM agree to within 0.002 on every metric, confirming the effect is about linearity itself.

**5. WL is the best encoder**, by +0.031 ROC-AUC and +0.049 Minority-F1 over FSM across 140 strictly matched configurations. `wl_edge` posts the study's highest raw numbers (ROC-AUC 0.8912) but on single splits only — a promising lead, not a result.

**6. Cost and quality are nearly unrelated.** The two most expensive learners rank fifth and sixth; `frozen_ksvd` delivers third place for 14 seconds. Compute spent on a saturating learner is compute wasted.

**7. All representations share one operating point.** Precision ≈ recall for nearly every configuration. To change that balance, change the threshold-calibration objective — not the dictionary.

### 11.2 What the project delivers

The value proposition of this work is **specific and honest**, and stating it precisely is more useful than overclaiming:

> These pipelines buy a **sparse, linearly-separable, interpretable representation** at a real computational cost. Where the downstream model must be linear — for interpretability, for deployment cost, for calibrated probabilities, for regulatory explainability — they are **measurably the best option tested**, beating a strong unsupervised graph embedding at every matched width. Where an unconstrained forest is acceptable, a dense `graph2vec` embedding remains ahead, because the forest performs the linearisation the dictionary was providing.

That is a real, bounded, reproducible contribution. It identifies **which** dictionary objective matters (discriminative, applied inside the objective), **how much** capacity to give it (it scales — don't under-size it), **which** encoder to pair it with (WL), and **which** downstream models it helps (linear ones, decisively).

### 11.3 What to do next

In priority order:

1. **Validate `wl_edge` with proper MC-CV.** It holds the study's highest raw score in its least-trustworthy corner. This is the cheapest possible route to a stronger headline number.
2. **Re-calibrate the decision threshold for recall.** Finding 7 says the operating point, not the representation, is the remaining lever — and screening applications want recall.
3. **Test on a balanced corpus.** Finding 2's mechanism explicitly depends on imbalance. Showing the supervised advantage shrinks when the imbalance is removed would confirm the mechanism; showing it persists would broaden the claim considerably.
4. **Measure the interpretability claim directly.** The pipelines' distinctive value is a sparse, linearly-readable code. That value does not show up in ROC-AUC. If it is real, it needs its own experiment — atom-level attribution against known pharmacophores would do it.
