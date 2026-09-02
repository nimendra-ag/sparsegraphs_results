# Results Section — Presentation Guide

*What to include, how to display it, what to conclude. Built from `analysis/all_results.csv` and
`analysis/ALL_RESULTS_ANALYSIS.md`. All figure paths are files that already exist in
`analysis/figures/`.*

---

## 0. Verdict on your four picks

| Your pick | Keep? | Why |
| --- | --- | --- |
| 1. Supervision is the strongest predictor (ranks 1–2 in all 36 conditions) | **Keep — slide 2** | This is the thesis of the whole project. Strongest, cleanest, most defensible result in the file. |
| 2. Three dictionary-size regimes (scales / saturates / collapses) | **Keep — slide 3** | It is the *mechanism* behind pick 1. It turns "FDDL is better" into "we know why FDDL is better." Also the most visually striking figure you own. |
| 3. Baseline verdict inverts with the classifier | **Keep — slide 5** | This is your honest headline and the answer to "did you beat the state of the art?" Panels reward a two-sided answer far more than a one-sided one. |
| 4. Variance decomposition (η²) | **Demote to a strip on the conclusions slide** | Genuinely novel, but it costs 90 seconds to explain η² to a panel, and it is a *meta*-result — it says which knob to turn, not what you discovered. Reduce to one plain-English line + a 3-bar chart on the final slide. Full version → backup slide. |

### The gap in your list

All four of your picks are **findings about dictionary learning in general**. None of them is about
**your own contributions** — Frozen K-SVD and CS-FDDL, the two imbalance mechanisms this project
actually built. An evaluation panel will ask, explicitly, *"what did you contribute, and did it
work?"* You need a slide that answers that with a statistical test, not a ranking table.

**So I recommend adding two slides:**

| Added | Why it earns a slide |
| --- | --- |
| **A. Your two imbalance mechanisms, tested head-to-head** (Frozen K-SVD ✓, CS-FDDL ✗) | Directly answers "what did *you* contribute?" It is a matched-pair Wilcoxon test, so it is the most rigorous evidence in the deck. It includes an honest negative result with a diagnosed cause — that reads as scientific maturity, not failure. |
| **B. Cost / practical recommendation** (Frozen K-SVD: 0.4745 Minority-F1 in **14 seconds**) | Connects the research to your two deliverables (the pip package and the Molytica local trainer, which the report itself flags as CPU-limited). It converts "we ran a big sweep" into "here is what you should ship." Panels remember this slide. |

---

## 1. The recommended arc — 8 slides

If you have room for only 5, use the **Minimum cut** column at the end of this section.

| # | Slide title (use these) | Core message |
| --- | --- | --- |
| **R0** | *How to read these numbers* | 4.55 % minority ⇒ accuracy is meaningless; here is our noise floor |
| **R1** | *What the objective optimises decides everything* | Supervised dictionaries rank 1–2 in all 36 conditions |
| **R2** | *More atoms only pays off if the objective is supervised* | Three regimes; BPFA actively collapses |
| **R3** | *Our two imbalance mechanisms, tested head-to-head* | Frozen K-SVD works (+0.019, p<0.0001); CS-FDDL does not (−0.012, p=0.0002) |
| **R4** | *Do we beat the baselines? Yes — if the classifier is linear* | +0.077 LogReg, +0.067 SVM; −0.033 GB, −0.030 RF |
| **R5** | *Why the verdict inverts* | The dictionary's job is linearisation; a forest does it for free |
| **R6** | *Cost: 700× spread, and it buys nothing* | Frozen K-SVD @512 = 0.4745 in 14 s |
| **R7** | *Conclusions and design guidance* | 5 bullets + "what to use when" table + η² strip |

---

## R0 — How to read these numbers *(framing slide — do not skip)*

Without this slide, every later number looks arbitrary and the panel will ask "why no accuracy?"
mid-presentation. Spend 45 seconds here and you buy credibility for the rest.

**Show — three small blocks, no big figure:**

**(a) Evidence base**

> 556 evaluation records · 520 pipelines + 36 baselines
> 4 encoders × 7 dictionary learners × 8 dictionary sizes × 4 classifiers
> Monte-Carlo cross-validation, up to 5 seeds per configuration

**(b) The GCN row — this is your most persuasive single table**

| GCN hidden size | Accuracy | Minority-F1 | ROC-AUC |
| --- | --- | --- | --- |
| 16 | 0.9545 | **0.000** | 0.553 |
| 32 | 0.9524 | **0.000** | 0.500 |
| 64 | 0.9531 | **0.000** | 0.500 |
| 128 | 0.9553 | **0.000** | 0.500 |

**(c) Noise floor**

> Median seed-to-seed SD: **ROC-AUC ± 0.015**, **Minority-F1 ± 0.022**
> *Any gap smaller than this is not a result — and we say so.*

**What to say (≈40 s).**
> "The minority class here is 4.55 % of the corpus. We can read that straight off the GCN baseline,
> which predicted 'inactive' for every single molecule at every width — and still scored 95.4 %
> accuracy. So accuracy is excluded from every comparison in this section. We report two things: an
> ROC-AUC for ranking quality and a Minority-F1 for decision quality, and we treat anything below
> ±0.022 Minority-F1 as noise."

**Interpretation note to deliver explicitly:** GCN is a **degenerate control, not a competitor.**
Do *not* claim "we beat GCN" — it collapsed. Claim: *"GCN collapses at this imbalance without class
balancing; every one of our pipelines does not."* That is the honest and stronger statement, and it
pre-empts the obvious panel question.

---

## R1 — What the objective optimises decides everything *(your pick #1)*

**Figure:** `analysis/figures/s3_metric_robustness/s3_wl_RF.png`
(3 panels: ROC-AUC, Macro-F1, Minority-F1 vs dictionary size — the report's Figure 4.2)

**Why this figure and not the single-metric one:** it shows the learner ordering is *identical under
all three metrics*. That kills the "you cherry-picked a metric" question before it is asked.

**Table — put this beside the figure, it is your centrepiece:**

| Learner | Objective contains | Mean rank | SD |
| --- | --- | --- | --- |
| **FDDL** | Fisher discrimination (supervised) | **1.36** | 0.54 |
| **CS-FDDL** | Fisher discrimination + class weights | **1.92** | 0.69 |
| Frozen K-SVD | reconstruction, class-staged | 3.83 | 1.44 |
| AKSVD | reconstruction | 4.43 | 1.34 |
| Online DL | reconstruction (stochastic) | 4.78 | 1.33 |
| LC-KSVD | reconstruction + label consistency | 5.79 | 0.94 |
| Bayesian (BPFA) | generative likelihood | 5.89 | 1.43 |

**Display rules:**
- **Colour-code the "objective" column** — one colour for supervised (rows 1–2), one for
  reconstruction/generative (rows 3–7). The visual should make the point before you speak.
- Add a dashed reference line or footnote: **"a learner with no advantage averages 4.00."** Without
  this, "1.36" means nothing to the panel.
- Do **not** put all three slice-tables (by metric / by encoder / by classifier) on the slide. Put
  one line instead: *"the ordering is unchanged under every metric, every encoder and every
  classifier."* Keep the slice tables as a backup slide.

**One extra number to say aloud (do not put on the slide):**
> "Paired at matched encoder, dictionary size and classifier, FDDL beats AKSVD by 0.064 Minority-F1
> and wins 65 of 68 matched comparisons — three times our noise floor."

**What to say (≈60 s).**
> "We ranked all seven learners in 36 conditions — every encoder, classifier and metric combination —
> each at its own best dictionary size. The two learners whose objective contains a Fisher
> discrimination term take ranks one and two in essentially every condition. The gap to third place
> is about two full rank positions, which is wider than the gap between any two learners *inside*
> either group. The interesting case is LC-KSVD: it is the only other learner that sees labels, and
> it finishes sixth of seven — behind three purely unsupervised methods. Bolting a label-consistency
> term onto a reconstruction objective is not the same as optimising for separation."

**Conclusion line for the slide:**
> **A dictionary helps exactly to the extent that its objective contains a discrimination term.
> Sparsity alone does nothing — LC-KSVD and BPFA are sparse too, and they finish last.**

---

## R2 — More atoms only pays off if the objective is supervised *(your pick #2)*

**Figure:** `analysis/figures/s1_learner_sweep/s1_wl_RF_roc_auc.png`
(the report's Figure 4.1 — the seven-curve fan-out, with the Bayesian and Online-DL curves diving)

This is the best *visual* in the whole project. It is self-explanatory from ten metres away: two
curves rise, three flatten, two fall off a cliff. Let it be large — half the slide minimum.

**Table — the compact version. Do not show all three ρ columns; one is enough:**

| Learner | ρ(atoms, ROC-AUC) | Regime |
| --- | --- | --- |
| CS-FDDL | **+0.67** | **scales** |
| FDDL | **+0.79** | **scales** |
| LC-KSVD | +0.18 | scales weakly |
| Frozen K-SVD | −0.31 | saturates |
| AKSVD | −0.36 | saturates |
| Online DL | −0.30 | degrades |
| Bayesian | **−0.74** | **collapses** |

**Display rules:**
- Sort the table to match the visual order of the curves on the figure at the right-hand edge, so the
  eye can move between them.
- Use a **traffic-light colour** on the Regime column (green / amber / red). Regime is the message;
  the ρ value is the evidence.
- Annotate the figure directly (a text box on the slide, over the plot): **"Bayesian: −0.174 ROC-AUC
  from 32 → 4096 atoms."** One arrow, one number.

**What to say (≈60 s).**
> "Dictionary size is the one knob everyone tunes, and the answer is not the same for every learner.
> A purely reconstructive or generative objective, given more atoms, spends them reconstructing
> whatever dominates the input — and in a 95 %-majority corpus that is the majority class and the
> encoder's high-frequency noise. Reconstruction improves while the discriminative content of the
> code is diluted. A Fisher term constrains where capacity goes, so extra atoms keep paying. BPFA is
> the extreme case: it loses 0.174 ROC-AUC going from 32 to 4096 atoms — and that matters, because
> automatic dictionary-size inference is BPFA's advertised selling point. On this corpus the inferred
> size optimises reconstruction, which is the wrong target."

**Add this nuance — it is honest and it pre-empts a sharp question:**
> "In *relative* terms the effect is smaller than the plot suggests: every learner sits within 2 % of
> its own peak somewhere between 128 and 1024 atoms. Dictionary size is a robustness parameter more
> than a performance parameter — the shape of the curve is diagnostic of the objective, but the
> height of the peak is not very sensitive to where you sit."

**Conclusion line for the slide:**
> **"More atoms is better" is true only for FDDL and CS-FDDL. For BPFA it is actively harmful.**

---

## R3 — Our two imbalance mechanisms, tested head-to-head *(added — your contribution slide)*

This is the slide that answers *"what did you build, and did it work?"*

**The framing sentence that makes this slide land (say it first):**
> "A ranking table cannot test a *mechanism*, because it compares across different learner families.
> To test our two imbalance mechanisms we paired each against the same learner *without* the
> mechanism, at matched encoder, dictionary size and classifier, and ran a Wilcoxon signed-rank test."

**Table — the whole slide in six numbers:**

| Mechanism tested | vs. baseline learner | Matched cells | Wins | Mean Δ Minority-F1 | *p* |
| --- | --- | --- | --- | --- | --- |
| **Frozen K-SVD** (majority-freeze + minority augment) | AKSVD | 64 | **46** | **+0.019** | **< 0.0001** |
| **CS-FDDL** (per-class gradient weighting) | FDDL | 72 | 21 | **−0.012** | 0.0002 |

**Second table (or a small bar chart) — the part that proves Frozen K-SVD is real, not noise:**

Frozen K-SVD advantage over AKSVD, by dictionary size (WL encoder):

| Atoms | 32 | 64 | **128** | 256 | **512** | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Δ Minority-F1 | +0.004 | +0.025 | **+0.065** | +0.048 | **+0.069** | +0.046 | +0.015 | −0.013 |

**Display rule:** make the second table a **small bar chart** (bars above/below zero), not a table.
The inverted-U shape *is* the argument, and a shape reads instantly where a row of numbers does not.
On WL specifically Frozen K-SVD wins **28 of 32** matched cells, mean **+0.032** — put that number
on the slide.

**What to say (≈75 s).**
> "Frozen K-SVD works. It beats vanilla AKSVD on 46 of 64 matched comparisons, and on the WL encoder
> it wins 28 of 32 with a mean gain of 0.032 Minority-F1. More importantly, the *shape* is what the
> mechanism predicts: near zero at 32 atoms, where the dictionary is too small to hold both a majority
> basis and a minority residual; peaking around 128 to 1024 where the split has room to matter; and
> vanishing at 4096 where an unconstrained K-SVD has enough capacity to cover the minority anyway, so
> freezing is only a constraint. Noise does not produce that shape.
>
> CS-FDDL is the honest negative. Paired against plain FDDL it loses 51 of 72 matched cells. Our
> leading explanation is convergence, not the objective: the cost-sensitive formulation divides the
> IPM step size by max of the class weights, and at a 4.55 % minority rate that factor is about
> eleven — so at matched iteration count CS-FDDL is taking steps an order of magnitude smaller. We
> believe it is under-converged rather than wrong, and that is a directly testable follow-up."

**Do not apologise for the negative result.** Present it as designed rigour. If it helps, add a
one-line footer:

> *A ranking table would have shown CS-FDDL at rank 2 and hidden this. The paired test is why we ran it.*

**Conclusion line for the slide:**
> **Of our two imbalance mechanisms, the cheap one is validated (+0.032 Minority-F1 on WL, at lower
> cost than the method it beats); the expensive one is not, and we have a diagnosed, testable cause.**

---

## R4 — Do we beat the baselines? Yes — if the classifier is linear *(your pick #3)*

**Figures — use two, side by side. The contrast IS the finding:**

| Left panel | Right panel |
| --- | --- |
| `analysis/figures/s3_metric_robustness/s3_wl_LinSVM.png` | `analysis/figures/s3_metric_robustness/s3_wl_RF.png` |
| *"Linear SVM — we win"* | *"Random Forest — graph2vec wins"* |

These are the report's Figures 4.4 and 4.6, and they already carry the SF / graph2vec / GCN baseline
lines. Crop each to the **Minority-F1 panel only** if the 3-panel version is too dense at
projection size — one metric, two classifiers, one message.

**Table — the four-row summary. This is the number the panel will write down:**

| Classifier | Best pipeline | Minority-F1 | Best baseline | Minority-F1 | Δ |
| --- | --- | --- | --- | --- | --- |
| **LogisticRegression** | WL + FDDL @2048 | **0.4478** | graph2vec @1024 | 0.3709 | **+0.077** |
| **LinearSVM** | WL + FDDL @4096 | **0.4507** | graph2vec @2048 | 0.3837 | **+0.067** |
| GradientBoosting | WL + FDDL @4096 | 0.4103 | graph2vec @2048 | 0.4430 | −0.033 |
| RandomForest | WL_edge + AKSVD @4096 | 0.4895 | graph2vec @512 | 0.5195 | −0.030 |

**Display rules:**
- Green the top two Δ cells, red the bottom two. The inversion should be visible without reading.
- Put the noise floor on the slide as a footnote: *"noise floor ± 0.022 — the two wins are 3–4×
  it, the two losses ≈1.5×."* This is what converts the table from "numbers" into "evidence."

**The strongest supporting number you have — say it, and consider a small strip chart:**
> Compared at **identical embedding width and identical classifier**, the best WL pipeline beats
> graph2vec at **all 7 widths** under LogReg (+0.060 … +0.085) and **all 7** under LinearSVM
> (+0.056 … +0.118). Under RandomForest it loses at **all 7**.

That kills the "you just picked your best width against their average" objection outright. If you
make one new chart for this deck, make this one (see §3).

**Also state the limit — volunteering it is far better than being caught by it:**
> "Only four of our twenty-five pipelines ever clear the best baseline, and only under a linear
> classifier: WL+FDDL, WL+CS-FDDL, WL+Frozen-K-SVD, and WL+AKSVD in a single cell. That is 28 of 500
> pipeline cells."

**What to say (≈70 s).**
> "Against the baselines the verdict is not one-sided, and it inverts cleanly with the classifier.
> With Logistic Regression or a Linear SVM, our WL plus FDDL pipeline beats every baseline — by 0.077
> and 0.067 Minority-F1, three to four times our noise floor — and it does so at every matched
> embedding width, not just at its best one. With Gradient Boosting or Random Forest, graph2vec wins,
> by about 0.03. We report both directions because the mechanism explains both, and that is the next
> slide."

**Conclusion line for the slide:**
> **Where the downstream model is linear, a discriminative dictionary does something nothing else in
> the pipeline does. Where it is a tree ensemble, the ensemble already does it.**

---

## R5 — Why the verdict inverts *(mechanism — merge into R4 if short on time)*

**Table — WL encoder only, each learner at its best width:**

| Learner | LogReg | RandomForest | **RF − LogReg** |
| --- | --- | --- | --- |
| **CS-FDDL** | 0.4405 | 0.4535 | **+0.013** |
| FDDL | 0.4478 | 0.4839 | +0.036 |
| Frozen K-SVD | 0.4352 | 0.4745 | +0.039 |
| AKSVD | 0.3805 | 0.4335 | +0.053 |
| LC-KSVD | 0.3227 | 0.3896 | +0.067 |
| **Online DL** | 0.3443 | 0.4448 | **+0.101** |

**Figure (optional):** `analysis/figures/s5_classifier_interaction/s5_minority_f1.png`

**What to say (≈45 s).**
> "The gap between a linear model and a forest, on the *same* codes, measures how much class
> structure is present but not linearly reachable. CS-FDDL's codes are read essentially as well by a
> linear model as by a forest — a gap of 0.013, below our noise floor. Online-DL's are not: a tenth
> of a point of Minority-F1, roughly a quarter of its final performance, is supplied by the
> classifier rather than by the representation. So the dictionary's real job is to linearise the
> class structure — which is exactly why our advantage appears under linear classifiers and
> disappears under a forest that performs that linearisation itself."

**Caveat you must state (30 seconds, and it protects you):**
> "This gap measures entanglement, not quality. Bayesian has the second-smallest gap simply because
> it is uniformly bad and there is little structure left for a forest to recover — so we read this
> column alongside the absolute level, never on its own."

### ⚠ Do not present the report's Table 4.6 spread argument

Report §4.2.3 states the best-worst learner spread is 0.0789 under RandomForest vs 0.1673 under
LogisticRegression, "more than twice as large," concluding a forest compensates for a weak
representation. **That does not survive re-derivation.** Three of the seven rows in Table 4.6 take
the LogReg and RandomForest values from *different encoders*, and the RF spread was taken between
the wrong two learners. Computed encoder-consistently on WL, the spreads are **0.184 (RF) and 0.172
(LogReg)** — indistinguishable.

Use the per-learner **RF − LogReg** table above instead. It is correct, it makes the same point
better, and if a panel member has the CSV open, you are safe. (Full derivation: `ALL_RESULTS_ANALYSIS.md` §8.2.)

---

## R6 — Cost: 700× spread, and it buys nothing *(added — the practical slide)*

**Figure:** `analysis/figures/s6_cost_vs_performance/s6_RF_minority_f1.png`

**Table:**

| Config | Fit time | Minority-F1 |
| --- | --- | --- |
| **WL + Frozen K-SVD @512** | **14 s** | **0.4745** |
| WL + FDDL @4096 (best 5-seed result) | ~10³ s scale¹ | 0.4839 |
| WL + LC-KSVD @2048 (most expensive) | 10 506 s | 0.3227 |

¹ *state as "the same-width FDDL-family fit costs 6 061 s" — FDDL timings were only recorded at 32
atoms, so use its CS-FDDL sibling and say so.*

**Headline numbers for the slide:**
> ρ(fit time, quality) = **−0.16** — spending more compute is, if anything, mildly counterproductive.
> The **entire cost/quality Pareto frontier is Frozen K-SVD.**

**What to say (≈45 s).**
> "Fit cost across this study spans a factor of seven hundred, and it is essentially uncorrelated with
> quality — the correlation is minus 0.16, because our two most expensive learners are also two of the
> three worst. Every non-dominated point on the cost-versus-quality frontier belongs to Frozen K-SVD.
> Fourteen seconds of fitting gets you 0.4745 Minority-F1 — within our noise floor of the best number
> any pipeline in this study produced, for two to three orders of magnitude less compute. That is the
> configuration we ship in the local training server, where the report already flags CPU training as
> a limitation."

**Why this slide matters for your evaluation:** it is the only slide that connects the research
sweep to your two engineering deliverables (`graph_dictlearn` and the Molytica local trainer) and to
your stated extended objective on computational efficiency. It converts a benchmark study into an
engineering recommendation.

---

## R7 — Conclusions and design guidance

### The five conclusion bullets (slide-ready, use close to verbatim)

1. **Supervision, not sparsity, is the mechanism.** FDDL and CS-FDDL rank 1–2 in all 36 conditions;
   FDDL beats AKSVD by +0.064 Minority-F1 on 65 of 68 matched cells. LC-KSVD and BPFA are sparse too
   and finish last.
2. **Capacity only pays under a discriminative objective.** FDDL improves with dictionary size
   (ρ = +0.79); BPFA collapses (ρ = −0.74, −0.174 ROC-AUC from 32 to 4096 atoms).
3. **Our Frozen K-SVD imbalance mechanism is validated** (+0.032 Minority-F1 over AKSVD on WL,
   28/32 matched wins, *p* < 0.0001) — and it is the cheapest learner in the study.
   **CS-FDDL is not validated** at this imbalance level; we attribute it to the step-size reduction
   the weighting requires and identify it as the priority follow-up.
4. **Three WL pipelines beat every baseline under a linear classifier, at every matched width**
   (+0.056 … +0.118 Minority-F1). Under a tree ensemble graph2vec wins by ~0.03 — because a forest
   performs the linearisation the dictionary was providing.
5. **The practical recommendation is WL + Frozen K-SVD @512**: 0.4745 Minority-F1 in 14 seconds,
   within the noise floor of the study's best result at 1/400 of the cost.

### The "what to use when" table — put this on the slide

| If you need… | Use | Minority-F1 |
| --- | --- | --- |
| An interpretable / linear model (Molytica's interpretability pipeline requires one) | **WL + FDDL @2048–4096 + LinearSVM** | **0.451** |
| Speed / CPU / local training | **WL + Frozen K-SVD @512 + RandomForest** | **0.475** |
| Maximum accuracy, classifier unconstrained | graph2vec @512 + RandomForest | 0.520 |
| A cheap, strong reference point | SF @26 + RandomForest | 0.441 |

**Say this about row 3 — it is the most credible thing you can do in the whole talk:**
> "We include the honest answer in our own recommendation table. If you are free to use a Random
> Forest, graph2vec is still the better encoder on this corpus. Our contribution is specific: sparse
> discriminative dictionaries win where interpretability or a linear model is required — which is
> exactly the setting our interpretability pipeline needs."

### The η² strip *(your pick #4, in its demoted form)*

One line of three horizontal bars, no η² symbol on the slide, captioned:

> **Which knob matters most?**
> Ranking quality (ROC-AUC) → **the encoder** · Decision quality (Minority-F1) → **the classifier**
> Both → **the dictionary learner** · Dictionary size → **least of the four**

Only if asked, give the numbers: encoder η² = 0.34 on ROC-AUC vs 0.10 on Minority-F1; classifier
0.08 vs 0.29; learner 0.31 on both; dictionary size 0.001–0.06.

---

## 2. Minimum cut — if you only get 5 slides

| Keep | Drop / merge |
| --- | --- |
| R0 *(compress to a 3-line header on R1)* | — |
| **R1** Supervision decides everything | — |
| **R2** Three regimes | — |
| **R3** Our two mechanisms, tested | — |
| **R4** Baseline verdict inverts (+ one mechanism line from R5) | **R5** merged in |
| **R7** Conclusions (+ the Frozen K-SVD 14 s number pulled in from R6) | **R6** merged in |

That is 5 content slides + conclusions. **Never drop R3** — it is the only slide that is about your
own work.

---

## 3. New figures worth making (3, all small)

Everything else is already rendered in `analysis/figures/`. These three do not exist yet and each
carries a slide:

| # | Chart | For | Spec |
| --- | --- | --- | --- |
| 1 | **Matched-width duel** | R4 | Two panels (Linear / Tree). x = dictionary width (32…2048, log₂). y = *best pipeline Minority-F1 − graph2vec Minority-F1* at the same width. Bars above zero in green, below in red, zero line bold. Left panel is all-green, right panel all-red. **This is the highest-value new chart in the deck.** |
| 2 | **Frozen K-SVD advantage vs dictionary size** | R3 | Single row of bars, Δ Minority-F1 (Frozen − AKSVD) at 32…4096, zero line. The inverted-U is the argument. |
| 3 | **Which knob matters** | R7 | 3 horizontal bars (encoder / learner / classifier) × 2 metrics, or one grouped pair. Keep it tiny — it is a footnote made visual. |

---

## 4. Three things to *not* say

| Do not say | Say instead |
| --- | --- |
| "We ran 5-seed Monte-Carlo CV for every combination." | "The WL sweep is 5-seed MC-CV; gspan_cork and WL-Edge are single-split and we report them as provisional." *(Only 38 % of pipeline rows are 5-seed; 30 % are single-split — a panel member with the CSV can check.)* |
| "WL-Edge + FDDL is our best result" (ROC-AUC 0.891 @20 000 atoms) | "WL-Edge is promising but under-evaluated." *Those rows are single-split and have **no minority-class metrics recorded at all**, so they cannot enter the primary comparison.* |
| "Random Forest compresses the learner ranking" (report Table 4.6 / §4.2.3) | Use the per-learner **RF − LogReg** table in R5. The spread claim does not survive an encoder-consistent re-derivation. |

---

## 5. Likely panel questions, and the answer

| Question | Answer |
| --- | --- |
| *"Why no accuracy?"* | R0's GCN table. A do-nothing model scores 95.4 %. |
| *"Did you beat the state of the art?"* | "Under a linear classifier, yes, at every matched width. Under a Random Forest, no — graph2vec wins by 0.03, and we explain why on the mechanism slide." |
| *"Your CS-FDDL contribution didn't work?"* | "Not at this imbalance level. We tested it properly rather than reporting its rank, and we have a diagnosed cause — the step-size division by max class weight, roughly a factor of eleven here — which makes it a convergence problem, not an objective problem. That is our first follow-up." |
| *"Is 0.077 a big difference?"* | "Our noise floor is 0.022 Minority-F1, measured across seeds. So it is three to four times noise — and it reproduces at all seven matched widths." |
| *"Why is Minority-F1 only ~0.45? That seems low."* | "At a 4.55 % base rate with no hyperparameter tuning, on a scale where a do-nothing model scores zero. The best baseline in the literature setup reaches 0.52. Also: recall is nearly constant at 0.38 across all methods, so these differences are almost entirely differences in *precision* — i.e. wasted assays per true hit." |
| *"Why is graph2vec better with Random Forest?"* | R5, in one sentence: "The dictionary's job is to linearise class structure; a forest does that for free, so the sparse code spends capacity on a function the classifier already provides." |
| *"How many configurations did you actually beat the baseline in?"* | "28 of 500 pipeline cells, all under linear classifiers, from four pipelines — all WL-encoded. We state that limit explicitly." |

---

## 6. One-page cheat sheet of every number you need

```
Base rate             4.55 %  (GCN accuracy 95.2–95.6 %, Minority-F1 0.000)
Noise floor           ROC-AUC ±0.015   Minority-F1 ±0.022
Records               556 = 520 pipelines + 36 baselines

RANKS (36 conditions, random = 4.00)
  FDDL 1.36 | CS-FDDL 1.92 | Frozen 3.83 | AKSVD 4.43 | OnlineDL 4.78 | LC-KSVD 5.79 | BPFA 5.89

SIZE REGIMES  rho(atoms, ROC-AUC)
  FDDL +0.79 | CS-FDDL +0.67 | LC-KSVD +0.18 | Frozen -0.31 | AKSVD -0.36 | OnlineDL -0.30 | BPFA -0.74
  BPFA loses 0.174 ROC-AUC from 32 -> 4096 atoms

OUR MECHANISMS (paired Wilcoxon, matched encoder x atoms x classifier)
  Frozen - AKSVD : +0.019 Min-F1, 46/64 wins, p < 0.0001   (WL: +0.032, 28/32)
  CS-FDDL - FDDL : -0.012 Min-F1, 21/72 wins, p = 0.0002   (WL: -0.024, 4/32)

BASELINE DUEL (best pipeline - best baseline, Minority-F1)
  LogReg   +0.077   |  LinearSVM +0.067   |  GBoost -0.033  |  RandomForest -0.030
  Matched-width: linear wins 7/7 at both classifiers; RF loses 7/7
  Cells beating best baseline: 28 / 500 (5.6 %), all linear, 4 pipelines

LINEARISATION GAP (RF - LogReg, WL)
  CS-FDDL +0.013 | FDDL +0.036 | Frozen +0.039 | AKSVD +0.053 | LC-KSVD +0.067 | OnlineDL +0.101

COST
  Frozen K-SVD @512 = 14 s -> 0.4745 Min-F1   |  LC-KSVD @2048 = 10 506 s -> 0.3227
  rho(cost, quality) = -0.16 ;  700x spread ; entire Pareto frontier = Frozen K-SVD

RECOMMENDATIONS
  linear/interpretable : WL + FDDL @2048-4096 + LinearSVM     0.451
  fast / CPU           : WL + Frozen K-SVD @512 + RF          0.475
  unconstrained        : graph2vec @512 + RF                  0.520
  cheap reference      : SF @26 + RF                          0.441
```

---

*Sources: `analysis/all_results.csv`, `analysis/ALL_RESULTS_ANALYSIS.md`, `analysis/figures/`.*
