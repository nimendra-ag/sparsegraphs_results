# Results and Analysis

## 1. Preliminaries: What the Experimental Surface Supports

### 1.1 Inventory

The analysis draws on 556 evaluation records in `analysis/all_results.csv`, comprising 520 pipeline records and 36 baseline records on the NCI corpus (N ≈ 36,937).

The class balance can be read directly off the `gcn` baseline, which is degenerate: its minority recall is exactly 0.0000 at every hidden size, so it predicts the majority class for every graph and its accuracy is therefore the majority rate. That rate is 0.9545–0.9553, placing the **minority class at roughly 4.6 %**. This is the essential context for everything below — a trivial all-negative classifier scores 95 % accuracy on this corpus, which is why accuracy is excluded from every comparison in this section.

The pipeline records span a four-factor design:

| Factor | Levels |
| --- | --- |
| Graph encoder | `wl`, `wl_edge`, `fsm`, `gspan_cork` |
| Dictionary learner | `fddl`, `csfddl`, `aksvd`, `lcksvd`, `online_dl`, `frozen_ksvd`, `bayesian` |
| Dictionary size | 32 … 20,000 atoms (powers of two, encoder-dependent) |
| Downstream classifier | LogReg, LinearSVM, GradientBoosting, RandomForest |

Three reference methods are carried throughout: `sf` (a fixed 26-dimensional structural-feature descriptor, 10-fold CV), `graph2vec` (7 embedding dimensions, single split), and `gcn` (an end-to-end network, 2 seeds).

Evaluation is restricted to **ROC-AUC**, **Macro-F1** and **Minority-F1**. These answer different questions and are reported together deliberately: ROC-AUC scores the ranking irrespective of the decision threshold, Macro-F1 scores the thresholded decision on both classes, and **Minority-F1 isolates the active class — the 4.6 % of compounds the screen exists to identify, and the only class on which a prediction changes what gets tested.** A correct "inactive" call triggers no assay and costs nothing to get right; the `gcn` baseline demonstrates the consequence, scoring 95.4 % accuracy while never once identifying an active compound.

Minority-F1 is reported *alongside* Macro-F1 rather than folded into it because the two per-class F1 scores Macro-F1 averages are wildly unequal in informativeness. Decomposing all 500 pipeline records (majority-F1 = 2·macro − minority):

| Component | Mean | Range | Spread |
| --- | --- | --- | --- |
| Majority-class F1 | 0.9580 | 0.8945 – 0.9771 | 0.083 |
| Minority-class F1 | — | 0.1286 – 0.4895 | **0.361** |

Half of every Macro-F1 figure is a near-constant term around 0.96 on which no method distinguishes itself. Averaging it against the informative half **halves the visible spread between methods**, which is precisely why the minority column is carried separately throughout.

### 1.2 The protocol confound, stated before any comparison

Evaluation protocol is **not** uniform across the design, and it is partially confounded with the encoder factor:

| Encoder | Records | Protocol | Seeds |
| --- | --- | --- | --- |
| `wl` | 224 | MC-CV (except `lcksvd`/4096) | 2–5 |
| `fsm` | 140 | MC-CV | 2–5 |
| `gspan_cork` | 116 | single split | 1 |
| `wl_edge` | 40 | single split (except `fddl`/32) | 1 |

156 of the 520 pipeline records — 30 % — are single splits. Every claim about `gspan_cork` or `wl_edge` therefore rests on one draw of the partition, with no spread. Given that the median MC-CV standard deviation is 0.0146 for ROC-AUC and 0.0216 for Minority-F1, a single-split number carries roughly that much unquantified uncertainty. This is the single largest threat to the conclusions and is revisited in Section 9.

A second, subtler issue: repetition counts vary between 2 and 5 across pipelines. A standard deviation computed from two repetitions is a very weak estimate of spread, and error bars in the figures should be read as indicative rather than as calibrated intervals.

### 1.3 A note on the eight scenarios

The figure suite is organised into eight comparison scenarios, but these are **eight views of one design, not eight independent findings**. They collapse to six substantive results. Scenarios S1, S2 and S4 are three cuts of the same encoder × learner × capacity surface; S3 is a robustness check confirming that S1's ordering survives a change of metric; S7 is a ranking of what S4 already shows cell by cell. Only S5 (classifier interaction), S6 (cost) and S8 (operating point) introduce genuinely new axes.

The analysis below is therefore organised by **finding**, with each finding citing whichever scenarios evidence it. This is also the basis for the figure-placement recommendation in Section 10: a scenario that merely confirms another belongs in the appendix.

---

## 2. Finding 1 — The learner hierarchy is stable, and it tracks discriminativeness

**How the ranking is constructed.** A *condition* is one (encoder × classifier × metric) triple: 3 encoders × 4 classifiers × 3 metrics = **36**. (`wl_edge` is excluded — only 2 of the 7 learners were run on it, and a field of seven cannot be ranked from two entrants; this drops 12 would-be conditions.) Within each condition every learner is reduced to its best score across dictionary sizes, and the seven results are sorted, 1 = best through 7 = worst. Each learner therefore accumulates 36 ranks, summarised below by their mean and standard deviation.

Read the two columns as follows. **Mean rank is typical competitive position**, on a scale where 1 is always-best, 7 is always-worst, and **4.0 is the null** — the average a learner with no systematic advantage would post. **SD of rank is consistency**: a small SD means the learner holds the same position regardless of setting, a large one means its position is setting-dependent.

Ranking the seven learners this way produces a strikingly consistent ordering:

| Learner | Mean rank | SD of rank | Rank distribution across the 36 conditions |
| --- | --- | --- | --- |
| **`fddl`** | **1.36** | 0.54 | 1st ×24, 2nd ×11, 3rd ×1 — never below 3rd |
| **`csfddl`** | **1.92** | 0.69 | 1st ×10, 2nd ×19, 3rd ×7 — never below 3rd |
| `frozen_ksvd` | 3.83 | 1.44 | 3rd ×17, but 7th ×4 |
| `aksvd` | 4.43 | 1.34 | spans the full range, 1st ×2 to 7th ×1 |
| `online_dl` | 4.78 | 1.33 | 5th ×12, 4th ×8 |
| `lcksvd` | 5.79 | 0.94 | never above 4th — reliably poor |
| `bayesian` | 5.89 | 1.43 | 7th ×18, yet reaches 3rd ×4 |

The distribution column earns its place: `aksvd`'s mean of 4.43 sits almost exactly at the null, but with an SD of 1.34 and a full 1-to-7 range it is not *consistently mediocre* — it has been both the best and the worst learner in the pool depending on the condition. That is a different claim from `online_dl`'s similar mean, and only the spread distinguishes them.

Two observations carry the weight.

First, the separation between the leading pair and the rest is far larger than the separation within either group. `fddl` and `csfddl` occupy ranks 1–2 in almost every condition; the gap to `frozen_ksvd` in third is roughly two full rank positions. A rank SD of 0.54 for `fddl` means it is essentially never outside the top two, regardless of encoder, classifier or metric.

Second, **the ordering is the discriminativeness ordering**. `fddl` (Fisher discrimination dictionary learning) and `csfddl` (its class-specific variant) are the only two learners in the pool whose objective contains a supervised discrimination term. `lcksvd` nominally carries a label-consistency term but ranks sixth; `frozen_ksvd`, `aksvd` and `online_dl` are purely reconstructive; `bayesian` is generative. The result is not that dictionary learning helps, but that **supervised dictionary learning helps and unsupervised dictionary learning largely does not** — a conclusion Section 5 sharpens considerably.

This ordering is invariant to the choice of metric (mean ranks under ROC-AUC, Macro-F1 and Minority-F1 agree to within 0.1 rank for the top two), which is what S3 exists to verify.

**What rank cannot tell you, and one place it matters.** A rank discards magnitude: a win by 0.003 counts exactly as much as a win by 0.05. This is harmless for the top-two-versus-the-rest separation, which is two full rank positions wide and backed by raw gaps far above the noise floor. It is *not* harmless for the ordering **within** the leading pair:

> `fddl` beats `csfddl` in 25 of 36 conditions, with a mean gap of only **+0.0055** — and that gap is smaller than the ROC-AUC noise floor of 0.0146 in **31 of the 36**.

No single condition therefore establishes that `fddl` outperforms `csfddl`. What supports the ordering is the consistency of the sign across 36 conditions, not its size in any one of them. The two are best reported as a jointly leading pair whose internal order is suggestive rather than resolved; the claim the data does settle is that both outrank the remaining five.

*Evidence: S1 (all encoders), S3, S4, S7.*

---

## 3. Finding 2 — Only discriminative dictionaries convert capacity into accuracy

The dictionary size sweep separates the learners into three regimes. Spearman correlation between atom count and score, pooled across encoders and classifiers over MC-CV records only:

| Learner | ρ(atoms, ROC-AUC) | ρ(atoms, Minority-F1) | Mean Δ (largest − smallest) ROC-AUC | Regime |
| --- | --- | --- | --- | --- |
| `csfddl` | **+0.63** | +0.55 | +0.036 | scales |
| `fddl` | **+0.58** | +0.50 | +0.039 | scales |
| `lcksvd` | +0.48 | +0.49 | +0.019 | scales weakly |
| `frozen_ksvd` | +0.02 | +0.20 | −0.015 | saturates |
| `aksvd` | −0.15 | +0.18 | −0.019 | saturates |
| `online_dl` | −0.36 | −0.09 | −0.059 | degrades |
| `bayesian` | **−0.82** | −0.79 | −0.116 | collapses |

The `bayesian` collapse is severe and monotone. On `wl` with RandomForest, ROC-AUC falls from 0.783 at 32 atoms to 0.609 at 4096 — a loss of 0.174, an order of magnitude larger than the MC-CV noise floor of 0.015. `online_dl` shows the same shape more gently (0.840 → 0.688). These are not plateaus; they are active deterioration with added capacity.

The interpretation is a straightforward consequence of the objectives. A purely reconstructive or generative dictionary given more atoms spends them on reconstructing whatever dominates the input distribution — which, in a 6 %-minority corpus, is the majority class and the encoder's high-frequency structural noise. The added atoms improve reconstruction while diluting the discriminative signal in the code. A dictionary with a Fisher-discrimination term is constrained to allocate capacity in a way that keeps class scatter separated, and so continues to profit from additional atoms.

Median peak sizes reflect this directly: `bayesian` peaks at 128 atoms, `online_dl` at 128, `frozen_ksvd` at 256, `aksvd` at 256, while `csfddl` peaks at 1024 and `fddl` at 1536 (median across conditions), with `fddl` still improving at 4096 on `wl`.

`lcksvd` is the one ambiguous case. Its MC-CV records reach only 2048 atoms, over which it scales positively (ρ = +0.48); the recently added single-split records at 4096 fall back to 0.8080 ROC-AUC under RandomForest, below its 2048 MC-CV value of 0.8120. On one draw this is not decisive, but it is consistent with `lcksvd` turning over rather than continuing to scale, which would place it with the saturating group despite its nominal label-consistency term.

**Practical consequence.** Dictionary size cannot be tuned once and shared across learners. Any study that fixes a single atom count across a learner comparison will misrepresent the ranking — at 128 atoms `frozen_ksvd` is competitive with `fddl`; at 4096 it is not remotely.

*Evidence: S1 (all 48 panels), S2, S3.*

---

## 4. Finding 3 — The classifier gap is a direct measurement of code separability

This is the most informative result in the study, and it is the one no single-classifier analysis would have surfaced.

Measuring the Minority-F1 improvement from LogisticRegression to RandomForest, per learner at its best dictionary size:

| Learner | LogReg | RandomForest | RF − LogReg |
| --- | --- | --- | --- |
| `csfddl` | 0.4405 | 0.4537 | **+0.013** |
| `fddl` | 0.4478 | 0.4839 | **+0.036** |
| `frozen_ksvd` | 0.4352 | 0.4745 | +0.039 |
| `lcksvd` | 0.3227 | 0.3896 | +0.067 |
| `online_dl` | 0.3443 | 0.4448 | +0.101 |
| `aksvd` | 0.3805 | 0.4895 | +0.109 |
| `bayesian` | 0.2805 | 0.4106 | **+0.130** |

The gap orders the learners almost exactly as Finding 1 does, and it admits a clean reading. **A wide LogReg→RF gap means the class information is present in the code but not linearly accessible; the forest is recovering it through non-linear partitioning.** A narrow gap means the representation has already done that work.

`csfddl` at +0.013 has produced codes that a linear model reads essentially as well as a forest. `bayesian` at +0.130 has produced codes whose class information is entirely entangled — nearly a third of its final performance is contributed by the classifier, not the representation.

This reframes the entire comparison. Under RandomForest the spread between the best and worst learner on Minority-F1 is 0.4895 − 0.4106 = 0.079. Under LogisticRegression it is 0.4478 − 0.2805 = 0.167 — **more than double**. A forest is a partial substitute for a good representation, and evaluating dictionaries through a forest alone compresses the very differences the study exists to measure.

The methodological recommendation follows: **the linear-classifier result is the more honest measure of representation quality**, and the RF−LogReg gap should be reported as a derived diagnostic in its own right.

*Evidence: S5 (all three metrics) — this is the scenario that earns its place most decisively.*

---

## 5. Finding 4 — WL dominates at matched configuration; the encoder ranking is partly confounded

Comparing `wl` against `fsm` at strictly matched (learner, atoms, classifier) under the same MC-CV protocol — the only encoder pair for which a clean comparison exists — over 140 matched triples:

| Metric | Mean (wl − fsm) | Median | wl wins |
| --- | --- | --- | --- |
| ROC-AUC | **+0.031** | +0.029 | 111 / 140 |
| Minority-F1 | **+0.049** | +0.044 | 104 / 140 |

The WL advantage is roughly twice the MC-CV noise floor and holds in about three-quarters of matched cells. This is a real effect, not a selection artefact.

The remaining two encoders cannot be placed on the same footing:

- **`gspan_cork`** is uniformly the weakest (best RF ROC-AUC 0.8305 against `wl`'s 0.8616), but every one of its 116 records is a single split. Some of that deficit may be protocol.
- **`wl_edge`** posts the highest raw numbers in the entire study (ROC-AUC 0.8912, Macro-F1 0.7821 at `fddl`/20,000 atoms/RF), but 36 of its 40 records are single splits, and its only MC-CV record — `fddl` at 32 atoms — scores 0.8350, entirely unremarkable. Its Minority-F1 column is absent above 32 atoms.

**`wl_edge` should be reported as a promising but unvalidated lead, not as the study's best result.** The honest headline is the best MC-CV configuration: `wl`/`fddl`/4096/RF at ROC-AUC 0.8616 ± 0.0072, Macro-F1 0.7299 ± 0.0120, Minority-F1 0.4839 ± 0.0233.

*Evidence: S2, S4 (with baseline strip), S7 (pipelines-only cut).*

---

## 6. Finding 5 — The baseline verdict inverts with the classifier

The comparison against `graph2vec` is the study's most consequential result, and it is not a single verdict. Comparing each `wl` pipeline against `graph2vec` at **matched representation width** (dictionary atoms against embedding dimension — both are the width of the vector handed to the classifier), over the 7 widths they share:

**Under LogisticRegression, Minority-F1:**

| Learner | Mean difference | Beats graph2vec at |
| --- | --- | --- |
| `fddl` | **+0.069** | **7 / 7 widths** |
| `frozen_ksvd` | +0.042 | 6 / 7 |
| `csfddl` | +0.039 | 6 / 7 |
| `aksvd` | −0.018 | 1 / 7 |
| `bayesian` | −0.103 | 0 / 7 |

**Under RandomForest, Minority-F1:**

| Learner | Mean difference | Beats graph2vec at |
| --- | --- | --- |
| `fddl` | **−0.057** | **0 / 7 widths** |
| `csfddl` | −0.083 | 0 / 7 |
| `bayesian` | −0.282 | 0 / 7 |

The sign flips completely. **Sparse discriminative codes are linearly more informative than graph2vec embeddings at every matched width; dense graph2vec embeddings are more informative to a forest at every matched width.**

This is exactly Finding 3 seen from the other side. The dictionary's contribution is to linearise the class structure — which is precisely the work a RandomForest performs for free on the dense embedding. Where a linear model is required (for interpretability, for deployment cost, for calibrated probabilities), `fddl` codes are the better representation by a wide and consistent margin. Where a forest is acceptable, `graph2vec` is unbeaten.

And it is unbeaten in absolute terms too. No MC-CV pipeline in the study surpasses it on any metric:

| Metric | Best MC-CV pipeline | Best baseline | Gap |
| --- | --- | --- | --- |
| ROC-AUC | 0.8616 ± 0.0072 (`wl`/`fddl`/4096/RF) | **0.8898** (graph2vec/128/RF) | −0.028 |
| Macro-F1 | 0.7299 ± 0.0120 | **0.7482** (graph2vec/512/RF) | −0.018 |
| Minority-F1 | 0.4839 ± 0.0233 | **0.5195** (graph2vec/512/RF) | −0.036 |

Each gap is 2–4 noise floors wide. The only pipeline number that exceeds a baseline is the single-split `wl_edge`/`fddl`/20,000 at 0.8912 ROC-AUC, which edges graph2vec's 0.8898 by 0.0014 — a difference far below the uncertainty of either.

Two caveats keep this fair. `graph2vec`'s figures are single-split and selected as the best of 7 dimensions, so they are optimistically biased; the pipeline figures are seed-averaged maxima over 8 sizes, so they are optimistic too, but less noisily so. And the remaining two baselines are comfortably beaten: `sf` reaches only 0.8185 ROC-AUC, and `gcn` fails outright — Minority-F1 of exactly 0.0000 at every hidden size, and ROC-AUC of exactly 0.5000 at three of its four, meaning it never learned to predict the minority class at all.

*Evidence: S1 and S2 (baseline sweeps drawn on the shared width axis), S4 (baseline strip), S7.*

---

## 7. Finding 6 — Cost varies by three orders of magnitude and is uncorrelated with quality

Median dictionary fit time, and the scaling with atoms on the `wl` encoder:

| Learner | Median fit (s) | 32 atoms | 4096 atoms | Growth |
| --- | --- | --- | --- | --- |
| `frozen_ksvd` | **16** | 11 s | 33 s | ×3 |
| `bayesian` | 79 | 12 s | 192 s | ×16 |
| `fddl` | 170 | 29 s | — | — |
| `csfddl` | 177 | 30 s | 6,061 s | ×202 |
| `aksvd` | 242 | 47 s | 9,048 s | ×193 |
| `lcksvd` | 586 | 114 s | 10,506 s (2048) | ×92 |
| `online_dl` | 806 | 22 s | 4,070 s | ×185 |

The pairing of Finding 2 with this table is unflattering to most of the pool. `lcksvd` and `online_dl` are the two most expensive learners and rank sixth and fifth. `aksvd` spends 9,048 seconds to reach 4096 atoms, a size at which it has already saturated and is drifting downward. **The expensive learners are expensive precisely in the regime where the extra capacity does them no good.**

`frozen_ksvd` deserves specific note as the cost-effectiveness winner: 16 s median, near-flat scaling, and third place in the quality ranking. Where compute is the binding constraint it is the rational default.

The honest caveat is coverage: only 336 of 520 pipeline records carry a fit time, and several combinations are only partly timed, so the Pareto frontier in S6 is drawn over a subset and may omit a combination's true best size. The figures state this per panel.

*Evidence: S6.*

---

## 8. Finding 7 — Every pipeline sits at the same operating point

At each combination's best Minority-F1 under RandomForest, the precision/recall decomposition is remarkably uniform:

| Configuration | Precision | Recall | Minority-F1 |
| --- | --- | --- | --- |
| `wl_edge`/`aksvd`/4096 | 0.542 | 0.446 | 0.490 |
| `wl`/`fddl`/4096 | 0.512 | 0.460 | 0.484 |
| `wl`/`frozen_ksvd`/512 | 0.503 | 0.449 | 0.475 |
| `fsm`/`csfddl`/1024 | 0.464 | 0.454 | 0.454 |
| `wl`/`bayesian`/128 | 0.278 | 0.332 | 0.300 |

Precision exceeds recall for nearly every configuration, and the ratio is stable across the whole quality range. The pipelines do not trade off differently — they move along a common iso-F1 progression, with better representations pushing further out rather than sitting at a different balance.

This is a consequence of the threshold-calibration step (Section 6 of the protocol document), which optimises a symmetric criterion on the validation tier. **If the application values recall over precision — as active-compound screening typically does — the calibration objective, not the dictionary, is the lever to pull.** No representation in this study offers a materially different operating point, and re-calibrating for recall is likely to yield more than any further dictionary tuning.

*Evidence: S8.*

---

## 9. Threats to Validity

Stated plainly, in order of severity.

1. **Protocol confounding.** 156 of 520 pipeline records are single-split: all 116 `gspan_cork` records, 36 of 40 `wl_edge` records, and the 4 `wl`/`lcksvd`/4096 records. The encoder ranking in Finding 4 is clean only for the `wl`/`fsm` pair. `wl_edge`'s leading numbers are unvalidated.
2. **Unequal repetition counts.** MC-CV seeds range from 2 to 5. Standard deviations from 2 repetitions are weak estimates, and the error bars are not comparable across pipelines.
3. **Selection on the reported data.** Every headline number is a maximum over dictionary sizes evaluated on the same test partitions, and is therefore optimistically biased. This applies symmetrically to pipelines (max over ~8 sizes) and to `graph2vec` (max over 7 dimensions), but `graph2vec`'s maximum is drawn from noisier single-split values and is the more inflated of the two.
4. **Incomplete timing coverage.** 184 of 520 pipeline records lack a fit time; the cost analysis is drawn over the timed subset.
5. **Missing cells.** `wl_edge` was run with only 2 of 7 learners, and its Minority-F1 is absent above 32 atoms. The combination matrix is not complete.
6. **Single corpus.** All conclusions are for NCI. Nothing here establishes that the discriminative-vs-reconstructive ordering transfers to a balanced corpus, and Finding 2's mechanism explicitly invokes class imbalance.

---

## 10. Figure Placement

### 10.1 Main text — eight figures

Chosen so that each carries a distinct finding and none merely restates another.

| # | Figure | Carries |
| --- | --- | --- |
| 1 | `s7_leaderboard/s7_roc_auc_top20.png` | Headline ranking with baselines in the race and error bars visible |
| 2 | `s1_learner_sweep/s1_wl_RF_roc_auc.png` | Findings 1 + 2: the hierarchy and the three capacity regimes in one panel |
| 3 | `s5_classifier_interaction/s5_minority_f1.png` | **Finding 3** — the central result; the LogReg→RF gap read off directly |
| 4 | `s1_learner_sweep/s1_wl_LogReg_minority_f1.png` | Finding 5: the matched-width crossover where dictionaries beat graph2vec |
| 5 | `s4_combination_matrix/s4_RF_roc_auc.png` | Finding 4: encoder × learner surface with the baseline strip |
| 6 | `s6_cost_vs_performance/s6_RF_roc_auc.png` | Finding 6: cost against quality |
| 7 | `s8_minority_tradeoff/s8_RF.png` | Finding 7: the shared operating point |
| 8 | `s3_metric_robustness/s3_wl_RF.png` | Robustness: the ordering survives the metric change |

If the section must be shorter, figures 1, 3, 4 and 5 are the irreducible set — they carry the ranking, the central mechanism, the baseline verdict and the design surface.

### 10.2 Appendix — the remaining 105 figures

Grouped by scenario, each with a one-line caption stating what it varies:

- **A.1 — Capacity sweeps (S1, remaining 45).** The full encoder × classifier × metric grid. Include in full: they are the primary evidence for Finding 2 and readers will want the condition matching their own setting.
- **A.2 — Encoder comparison (S2, all 12).** Faceted by learner. Confirmatory for Finding 4.
- **A.3 — Metric robustness (S3, remaining 15).**
- **A.4 — Combination matrices (S4, remaining 11).** Per classifier and metric.
- **A.5 — Classifier interaction (S5, remaining 2).** ROC-AUC and Macro-F1 versions of the Finding 3 figure.
- **A.6 — Cost (S6, remaining 11).**
- **A.7 — Leaderboards (S7, remaining 5).** Including the pipelines-only cuts.
- **A.8 — Operating points (S8, remaining 3).**

**Recommended trimming.** If the appendix must be bounded, S2 (12 figures) is the most redundant — it re-plots the S1 data with encoder and learner swapping roles and adds no finding that S1 and S4 do not carry between them. S3 can be reduced to the four `wl` panels, since the metric-robustness claim needs demonstrating once, not sixteen times. That removes 24 figures for no loss of evidence.

---

## 11. Conclusions

1. **Supervised dictionary learning is the only variety that works here.** `fddl` and `csfddl` occupy the top two ranks in essentially every condition; the reconstructive and generative learners are two full rank positions behind.
2. **Capacity is only useful to a discriminative objective.** `fddl` and `csfddl` improve monotonically with atoms (ρ ≈ +0.6); `bayesian` actively collapses (ρ = −0.82, losing 0.17 ROC-AUC from 32 to 4096 atoms). Dictionary size must be tuned per learner.
3. **The dictionary's real contribution is linearisation.** The LogReg→RF gap falls from 0.130 (`bayesian`) to 0.013 (`csfddl`). A good dictionary does the work a forest would otherwise have to do — and evaluating through a forest hides more than half of the between-learner spread.
4. **WL is the better encoder**, by +0.031 ROC-AUC over FSM across 140 matched configurations.
5. **Against graph2vec the verdict depends on the downstream model.** `fddl` codes beat graph2vec at 7 of 7 matched widths under a linear classifier (+0.069 Minority-F1) and lose at 7 of 7 under a forest (−0.057). No MC-CV pipeline beats graph2vec outright on any metric.
6. **The pipelines' value proposition is therefore specific, not general:** they buy a linearly separable, sparse, interpretable representation at a real computational cost. They do not buy raw predictive performance over a dense unsupervised embedding fed to a forest. Stating this plainly is more useful than the alternative reading, and it points directly at the next experiment — whether the sparse codes' interpretability and linear accessibility can be shown to have value that ROC-AUC does not capture.
