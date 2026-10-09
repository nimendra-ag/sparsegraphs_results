# WL + FDDL: Result Analysis across NCI1 / NCI33 / NCI41 / NCI47

**Scope.** 23 Monte-Carlo-CV executions of the WL + FDDL pipeline (5 seeds each), covering
4 datasets × dictionary sizes 128–2048 atoms (plus 4096 for NCI1 and NCI33). Each execution
was scored with 4 learned classifiers (2 linear, 2 tree ensembles) and the dictionary's own
sparse-representation classifier (SRC). Every number below is the **mean over the 5 seeds of an
execution**. Data: [all_results.csv](all_results.csv) (one row per execution × classifier).

### TL;DR: four findings

1. **Random Forest is the best classifier in every one of the 22 dataset × atom settings**, on
   every threshold metric. The other tree ensemble, Gradient Boosting, is the *worst* of the four,
   so the advantage comes from the Random Forest itself, not from being tree-based.
2. **Dictionary size matters for linear models, not for Random Forest.** Linear classifiers
   gain about +0.03 Macro-F1 from 128 → 2048 atoms; Random Forest gains about +0.003. At 2048+ atoms
   the gap shrinks to seed-noise level. Random Forest with only 128 atoms beats the best linear
   model at 2048 atoms and is about 4× cheaper.
3. **The FDDL dictionary is a good feature extractor but a poor classifier.** SRC gets *worse* as
   the dictionary grows (−0.093 Macro-F1), and the FDDL Fisher term in SRC makes no measurable
   difference (≤ 0.001).
4. **Accuracy and ROC-AUC hide these differences.** With only 4–5 % actives, every learned
   classifier scores about 0.95 accuracy (no better than predicting "inactive" for everything). MCC and
   minority-class metrics are what separate the methods.

---

## 1. Setup in brief

| Item | Value |
|---|---|
| Pipeline | WL subtree features (2 iterations) → energy-based feature selection (99 % energy, ~1,650–1,800 features kept) → FDDL dictionary → sparse codes (MaxAbs-scaled) → classifier |
| Splits per seed | 50 % dictionary/vocabulary · 20 % classifier training · 15 % validation (decision-threshold tuning, F1-optimal) · 15 % test (reported) |
| Classifiers | **Linear:** Logistic Regression, Linear SVM (calibrated) · **Tree ensembles:** Random Forest, Gradient Boosting · **SRC:** residual-based (γ = 0) and FDDL-native (γ = 0.5). All learned classifiers are class-balanced and use default hyper-parameters |
| Class imbalance (test) | NCI1 4.7 % · NCI33 3.9 % · NCI41 5.3 % · NCI47 4.6 % actives |
| Coverage | 128, 256, 512, 1024, 2048 atoms on all 4 datasets; 4096 on NCI1 and NCI33 only. NCI1 @ 2048 was run twice and the two runs are averaged (they differ by ≤ 0.009 Macro-F1, less than the seed std) |
| Seed noise | Typical seed-to-seed std: Macro-F1 ≈ 0.012–0.015, MCC ≈ 0.023–0.030 |

**Table 1. Mean over the four datasets (test set). Best value per column in bold.**

| Classifier | Macro-F1 @128 | @512 | @1024 | @2048 | MCC @128 | @512 | @1024 | @2048 |
|---|---|---|---|---|---|---|---|---|
| Random Forest | **0.713** | **0.720** | **0.726** | **0.716** | **0.428** | **0.442** | **0.455** | **0.437** |
| Linear SVM | 0.676 | 0.684 | 0.699 | 0.701 | 0.356 | 0.371 | 0.400 | 0.404 |
| Logistic Regression | 0.666 | 0.682 | 0.696 | 0.701 | 0.337 | 0.368 | 0.394 | 0.403 |
| Gradient Boosting | 0.672 | 0.679 | 0.686 | 0.684 | 0.345 | 0.360 | 0.374 | 0.371 |
| SRC (dictionary) | 0.636 | 0.596 | 0.573 | 0.542 | 0.309 | 0.278 | 0.267 | 0.236 |

![Fig 1](figures/fig1_overview_macro_f1.png)

Fig. 1 shows the full picture per dataset. The same pattern repeats on all four datasets: Random Forest
on top and nearly flat, linear models rising with dictionary size, Gradient Boosting lagging, and SRC
falling steeply. Because all four datasets agree, the findings below are not artefacts of one dataset.

---

## 2. Insight 1: Random Forest wins everywhere; "tree-based" alone is not the reason

- **Random Forest has the highest Macro-F1, MCC, Minority-F1 and Minority-PR-AUC in 22 / 22 settings**
  (ROC-AUC: 21 / 22). Its mean rank is 1.00. The others rank Linear SVM 2.59, Logistic Regression 2.86
  and Gradient Boosting 3.55 (Macro-F1).
- Average lead over the *best* linear model in each setting:
  **+0.028 Macro-F1** (range +0.010 to +0.044), **+0.056 MCC**, **+0.068 Minority-PR-AUC**.
- The differences are statistically reliable. The Friedman test across the four classifiers gives
  p = 5.7 × 10⁻¹⁰, and a Wilcoxon signed-rank test of Random Forest against the best linear model gives
  p = 4 × 10⁻⁵ (n = 22 settings).
- **Gradient Boosting, also a tree ensemble, ranks last** among the learned classifiers in most
  settings, and it is the slowest (≈ 150 s per seed at 2048 atoms). So the results do not support
  "trees beat linear models"; they support **"Random Forest beats everything else"**.

**Why this is plausible.** The sparse codes are high-dimensional, non-negative and mostly zero. A
class-balanced forest of deep, bagged trees can combine a few active atoms non-linearly, and its
averaging gives it low variance. Gradient Boosting is run with default settings (shallow trees and
100 rounds) and weights only through `sample_weight`, so it is under-powered here. Tuning it might
narrow its gap, but nothing in these runs suggests it would overtake Random Forest.

---

## 3. Insight 2: Dictionary size matters for linear models, not for Random Forest

![Fig 2](figures/fig2_dictionary_size_effect.png)

- **(a) Gain from 128 → 2048 atoms (Macro-F1, mean of 4 datasets):** Logistic Regression **+0.034**,
  Linear SVM **+0.025**, Gradient Boosting +0.012, **Random Forest +0.003**, SRC **−0.093**.
  The size–performance correlation is strong for the linear models (Spearman ρ = 0.86 and 0.85,
  p < 10⁻⁶) and not significant for Random Forest (ρ = 0.33, p = 0.13).
- **(b) The Random Forest lead shrinks steadily:** +0.037 at 128 atoms → +0.026 at 1024 → +0.014 at 2048 →
  ≈ +0.01 at 4096 (NCI1, NCI33). At 2048+ atoms the lead is about one seed-std, so the classifiers
  are close to tied there. At ≤ 1024 atoms the lead is about 1.8–2.7 times the seed noise.
- **(c) Cost matters as well.** Runtime grows roughly in proportion to the number of atoms (dictionary fit
  on NCI1: 18 s at 128 → 101 s at 2048 → 180 s at 4096). **Random Forest at 128 atoms (Macro-F1 0.713,
  ≈ 27 s/seed) beats Logistic Regression at 2048 atoms (0.701, ≈ 107 s/seed)** at about a quarter of the cost.
- **Random Forest's best size is about 1024 atoms** (best on NCI33, NCI41 and NCI47). Its whole
  128–2048 curve is within about one seed-std, so this is a soft optimum. On NCI41 *every* classifier drops
  from 1024 to 2048. That is a single execution, so treat it as a hint of over-sizing, not a
  confirmed effect.

**Interpretation.** A linear model needs the classes to be separable along individual atoms, so it
benefits when more, finer atoms are available. Random Forest can build the same decision rules from
combinations of a few coarse atoms. In practice, a small dictionary + Random Forest is the efficient
choice, and a large dictionary is only worth its cost if a linear (interpretable) model is required.

---

## 4. Insight 3: The FDDL dictionary works as a feature extractor, not as a classifier

![Fig 3](figures/fig3_src_degradation.png)

- **SRC is the only method that gets worse with more atoms**, and it does so on all 4 datasets (Spearman
  ρ = −0.97). Its Macro-F1 falls from 0.636 to 0.542 and its MCC from 0.309 to 0.236 (128 → 2048). It is the worst
  method in every setting.
- **The failure mode is a recall/precision collapse (Fig. 3a).** Minority recall *rises* (0.54 → 0.72),
  but minority precision falls (0.24 → 0.13, against ≈ 0.49 for Random Forest). As the dictionary
  grows, SRC labels more and more graphs as "active".
- **The Fisher term does nothing at prediction time.** SRC with γ = 0.5 (FDDL-native) and γ = 0
  (pure residual) differ by at most 0.001 Macro-F1 in every setting.

**Plausible mechanism (consistent with the data, not yet tested directly).** FDDL gives each class
the *same* number of atoms (total / 2), but the dictionary split contains only about 670–870 active
graphs per dataset. At 1024 total atoms the active sub-dictionary already has 512 atoms. At 2048 it has
1024 atoms, **more atoms than there are active training graphs**. Such an over-complete
sub-dictionary can reconstruct almost *any* graph well, so the minimum-residual rule increasingly picks
"active". That explains recall rising while precision collapses. A direct test would be a run with
class-proportional atom allocation.

**Implication for the thesis.** The discriminative training of FDDL pays off *indirectly*: its codes
are strong features for a downstream classifier (Insights 1–2). Its built-in classification rule
should not be used, especially for large dictionaries on imbalanced data.

---

## 5. Insight 4: Under 4–5 % actives, accuracy and ROC-AUC compress the differences

![Fig 4](figures/fig4_metric_sensitivity.png)

Fig. 4 shows each classifier's score as a percentage of Random Forest's score on the same metric,
averaged over all settings:

- **Accuracy:** all learned classifiers fall between 0.92 and 0.96. Predicting "inactive" for every
  graph already scores 0.947–0.961, so even Random Forest (0.947–0.957) does not beat this trivial
  baseline. The F1-tuned thresholds deliberately trade some accuracy for finding actives.
- **ROC-AUC:** every method, *including SRC*, reaches ≥ 96 % of Random Forest. On NCI41 @ 128 atoms SRC
  even has a higher ROC-AUC than Random Forest (0.850 vs 0.848), yet a much lower MCC (0.340 vs 0.433).
- **MCC, Minority-F1 and Minority-PR-AUC** spread the methods out: linear models reach 82–88 % of Random Forest,
  Gradient Boosting 82–84 %, SRC only 58–68 %.

**Implication.** Use **MCC and Minority-PR-AUC** as headline metrics, with Macro-F1 as the
familiar summary. Treat ROC-AUC as secondary, and do not report accuracy as evidence of quality on
these datasets. A Minority-PR-AUC of 0.41 (Random Forest) should be read against its chance level, which equals
the prevalence (0.04–0.05): it is about 9× better than random.

---

## 6. Caveats

- **One execution per setting.** Error bars show seed-to-seed variation inside one execution. The
  one repeated setting (NCI1 @ 2048) reproduces within 0.009 Macro-F1, which is reassuring but is a
  single check.
- **4096 atoms was run only on NCI1 and NCI33**, so statements about 4096 come from two datasets.
- **All classifiers use default hyper-parameters.** The ranking of Gradient Boosting in particular
  could change with tuning. The Random Forest-vs-linear conclusion is less likely to change, because the gap is
  consistent across all 22 settings.
- **The statistical tests treat the 22 settings as independent**, although settings from the same dataset are
  related. The fact that every dataset shows the same direction (Fig. 1) is the stronger evidence.
- All four datasets are NCI anticancer screens with similar chemistry and imbalance, so the results
  may not generalise to other graph domains.

## 7. Recommendations

1. **Default configuration: WL + FDDL with 512–1024 atoms + Random Forest.** It is the best or tied-best on every
   dataset at moderate cost. Use 128–256 atoms if runtime matters: the loss against Random Forest's best size is only 0.008–0.016 Macro-F1.
2. If a **linear model is required** (interpretability), use **≥ 2048 atoms**. Below that the linear
   models give up 0.02–0.04 Macro-F1.
3. **Do not report SRC as the FDDL classifier.** Present it as a baseline that shows why the codes need a
   downstream classifier. If SRC matters, test class-proportional atom allocation.
4. **Report MCC and Minority-PR-AUC** (with the prevalence baseline) next to Macro-F1. Drop accuracy.

---

*Reproduce:* `python test_analysis/wl_fddl/build_all_results.py` (→ `all_results.csv`), then
`python test_analysis/wl_fddl/make_figures.py` (→ `figures/`, `key_numbers.txt` with all test
statistics quoted above).
