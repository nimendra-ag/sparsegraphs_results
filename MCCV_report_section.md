# Monte Carlo Cross-Validation: Protocol and Implementation

## 1. Motivation and Choice of Resampling Scheme

Every quantitative claim made in this project rests on a comparison between representation pipelines — a graph encoder paired with a dictionary learner — evaluated through a battery of downstream classifiers. Such comparisons are only meaningful if the reported figures are stable: a difference of two or three points in macro-F1 between two pipelines is worth discussing only if it is large relative to the variation that the same pipeline would exhibit merely by being handed a different random partition of the same corpus.

A single fixed train/test split cannot supply that context. It yields one number per metric with no associated spread, and on a heavily imbalanced molecular corpus that number is unusually sensitive to the particular molecules that happen to land in the test set, since a comparatively small number of minority-class graphs dominates recall, precision and the Matthews correlation coefficient. Reporting such a point estimate invites the reader to attribute to the method what may in fact be attributable to the draw.

The natural remedy is a resampling scheme, and two candidates present themselves. Stratified *k*-fold cross-validation is the conventional choice, but it is a poor fit for this pipeline for three reasons. First, *k*-fold rigidly couples the number of repetitions to the train/test proportion: choosing five folds is simultaneously choosing an 80/20 split, whereas this study requires a deliberately structured four-way partition in which only 15 % of the corpus is held out for testing. Second, the folds are not independent — they are constrained to be mutually exclusive and exhaustive — which complicates the interpretation of the spread across folds. Third, and decisively in practice, the fitting cost is dominated by unsupervised representation learning (vocabulary construction followed by dictionary learning), which for the larger dictionaries runs for hours per repetition; a scheme whose repetition count is dictated by geometry rather than by the available compute budget is operationally unattractive.

This work therefore adopts **Monte Carlo cross-validation** (MCCV), also known in the literature as repeated random subsampling validation. Instead of partitioning the corpus once into mutually exclusive folds, MCCV draws a *fresh, independent, stratified partition* of the entire dataset for each repetition, refits the complete pipeline from scratch on that partition, and evaluates on the corresponding held-out test split. The reported figure for a given metric is then the mean over repetitions, accompanied by the sample standard deviation and a confidence interval that together characterise how much of the observed performance is attributable to the pipeline and how much to the accident of the split.

Three properties recommend this scheme in the present setting:

- **The split proportions and the repetition count are independent design choices.** The four-tier partition described in Section 3 can be specified on methodological grounds and held fixed, while the number of repetitions is chosen purely on the basis of the compute available.
- **Repetitions are mutually independent.** Each is generated from its own master seed and shares no state with any other, which makes the scheme embarrassingly parallel, trivially resumable, and safe to extend after the fact: additional repetitions can be appended to an existing run without invalidating those already completed.
- **The estimator degrades gracefully.** If a repetition fails — an out-of-memory condition on the GPU, a convergence failure in a classifier — the remaining repetitions still constitute a valid, if slightly less precise, MCCV estimate, and the recorded sample size makes that reduction explicit.

The corresponding cost must be stated plainly. Because each repetition draws its test split from the full corpus, the test splits of different repetitions overlap. The repetition-level metrics are therefore *not* statistically independent observations, and the confidence interval reported in Section 8 should be read as a descriptive summary of split-induced variability rather than as a formally exact interval for the generalisation error. This is a well-documented property of MCCV and is shared, in different form, by every resampling scheme that reuses a finite dataset; it is disclosed here so that the intervals are interpreted for what they are.

## 2. Formal Statement of the Estimator

Let the corpus be $\mathcal{D} = \{(G_i, y_i)\}_{i=1}^{N}$, where $G_i$ is a molecular graph and $y_i \in \{-1, +1\}$ its class label, with $+1$ denoting the minority (active) class. Let $\mathcal{A}$ denote the complete pipeline — encoder vocabulary construction, dictionary learning, sparse coding, feature scaling, classifier fitting and threshold calibration — and let $m(\cdot)$ denote a scalar evaluation metric.

For each repetition $b = 1, \dots, B$, governed by a master seed $s_b$:

1. Draw a stratified partition
$$\pi_b : \mathcal{D} \longrightarrow (\mathcal{D}^{\text{voc}}_b,\; \mathcal{D}^{\text{ml}}_b,\; \mathcal{D}^{\text{val}}_b,\; \mathcal{D}^{\text{test}}_b)$$
with the fixed proportions $(0.50,\, 0.20,\, 0.15,\, 0.15)$ of $N$, drawn independently of all other repetitions.
2. Fit the pipeline, $\hat{f}_b = \mathcal{A}\!\left(\mathcal{D}^{\text{voc}}_b, \mathcal{D}^{\text{ml}}_b, \mathcal{D}^{\text{val}}_b; s_b\right)$, where the validation split enters only through decision-threshold calibration.
3. Record $m_b = m\!\left(\hat{f}_b, \mathcal{D}^{\text{test}}_b\right)$.

The reported quantities are the empirical mean, the unbiased sample standard deviation, and the half-width of a 95 % confidence interval based on Student's *t* distribution:

$$\bar{m} = \frac{1}{B}\sum_{b=1}^{B} m_b, \qquad
s_m = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(m_b - \bar{m}\right)^2}, \qquad
h_{95} = t_{0.975,\, B-1}\cdot \frac{s_m}{\sqrt{B}} .$$

Every metric, for every classifier, is summarised by this triple together with the effective sample size $B$.

## 3. The Four-Tier Stratified Partition

The distinguishing feature of the protocol is that the partition has four tiers rather than the customary two or three. This is a direct consequence of the pipeline's structure: unlike a conventional classifier that consumes fixed features, the pipeline *learns its own representation* in two successive unsupervised or weakly supervised stages before any classifier is fitted. Each stage that consumes data is therefore given data of its own.

| Tier | Share of corpus | Size (NCI, $N \approx 36{,}937$) | Consumed by |
| --- | --- | --- | --- |
| `vocab_train` | 50 % | 18,467 | Encoder vocabulary (WL subtree / subgraph feature selection) **and** dictionary learning |
| `ML_train` | 20 % | 7,388 | Fitting the downstream classifiers on sparse codes |
| `val` | 15 % | 5,541 | Decision-threshold calibration |
| `test` | 15 % | 5,541 | Held-out final evaluation |

The rationale for each boundary is as follows.

**`vocab_train` (50 %).** Both the encoder's feature vocabulary and the learned dictionary are estimated here. These are the most data-hungry components of the pipeline — the encoder must observe enough graphs for its adaptive feature-selection cut to be stable, and the dictionary must be fitted on enough embeddings that its atoms are not artefacts of a handful of molecules — so this tier receives the largest allocation. Crucially, these components are fitted *once per repetition* and then frozen; nothing downstream is permitted to modify them.

**`ML_train` (20 %).** The classifiers are fitted on sparse codes computed from graphs the dictionary has never seen. This separation is essential rather than cosmetic. A dictionary reconstructs its own training embeddings better than it reconstructs unseen ones, so sparse codes computed on `vocab_train` would be systematically cleaner and more discriminative than codes the model will encounter at deployment. Fitting the classifiers on such codes would produce an optimistic and, more damagingly, an *inconsistent* picture: the representation would appear stronger in training than it is in operation. Placing the classifiers on a disjoint tier ensures they are trained on codes drawn from the same distribution as those they will be scored on.

**`val` (15 %).** Under class imbalance the default decision threshold of 0.5 is arbitrary and generally far from optimal. The threshold is therefore treated as a fitted parameter (Section 6) and estimated here, on data disjoint from both the representation-learning tiers and the test split.

**`test` (15 %).** Touched exactly once per repetition, after every parameter, hyperparameter and threshold has been fixed. No quantity derived from this tier feeds back into any fitting decision.

Operationally the partition is constructed as three nested stratified splits, each stratified on the class label so that the minority proportion is preserved in every tier — indispensable on a corpus where the minority class accounts for only a small percentage of the graphs, and where an unstratified draw could plausibly starve a tier of positive examples. The first split removes 15 % for `test`; the second removes $0.15/0.85 \approx 17.65\,\%$ of the remainder, which is exactly 15 % of the corpus, for `val`; the third divides the surviving 70 % in the ratio $5:2$, yielding 50 % and 20 % of the corpus for `vocab_train` and `ML_train` respectively. All three splits are driven by the same split sub-seed, so the entire partition is a deterministic function of the master seed.

An identical split policy is used by the single-pass export pipeline that produces deployable artefacts. This is deliberate: the model that is exported is built by exactly the procedure that was benchmarked, so the reported figures describe the artefact that is actually shipped.

## 4. Seed Management and Reproducibility

A repetition is uniquely identified by an integer **master seed**, and reproducibility is defined at that granularity: re-running a given master seed reproduces its partition, its encoder vocabulary, its dictionary initialisation, its classifier fits and therefore its metrics.

Seeding proceeds in two complementary steps. First, `seed_everything` sets every *global* random number generator that third-party libraries may consult implicitly — Python's `random`, NumPy's legacy global generator, the `PYTHONHASHSEED` environment variable, and, where PyTorch is present, both the CPU and CUDA generators. On the GPU path, cuDNN is additionally placed in deterministic mode with autotuning disabled, accepting a modest throughput penalty in exchange for run-to-run reproducibility.

Second, and more importantly, the master seed is expanded into four statistically independent sub-seeds through NumPy's `SeedSequence` entropy-mixing facility:

$$s \;\longmapsto\; \left(s_{\text{split}},\; s_{\text{enc}},\; s_{\text{dict}},\; s_{\text{clf}}\right)$$

which govern the partition, the encoder, the dictionary learner and the classifiers respectively. Deriving sub-seeds in this way, rather than by the common expedient of offsetting a base seed by small integers, guarantees that the four streams are well separated and that no unintended correlation is induced between, for example, the partition and the dictionary initialisation. It also confers a practical benefit for ablation: because the sub-seeds are independent, a change in the dictionary learner's initialisation cannot silently alter the data partition, so two configurations compared under the same master seed are genuinely compared on the same split.

The encoder and dictionary learner are supplied to the harness as *factories* — callables that accept a seed and return a freshly constructed object — rather than as pre-built instances. This is a deliberate defence against state leakage across repetitions: a fitted encoder or dictionary carried between seeds would allow one repetition's vocabulary or atoms to contaminate the next. Constructing them anew inside each repetition makes such leakage structurally impossible.

Master seeds themselves are drawn without replacement from $\{0, \dots, 100\}$. Sampling without replacement matters, because a repeated master seed would reproduce an identical partition and thus contribute a duplicate row that inflates the apparent sample size while contributing no new information about split-induced variance.

## 5. Execution Model: One Operating-System Process per Repetition

The harness does not execute repetitions in a loop within a single process. Instead, an **orchestrator** process spawns one fresh child process per master seed, sequentially, and waits for each to terminate before launching the next. The child — the **worker** — receives its master seed and the shared output directory on the command line, executes exactly one repetition, appends its results to disk, and exits.

This design is motivated by resource hygiene rather than parallelism. Dictionary learning on the GPU allocates large intermediate tensors; even with careful deallocation, a long-lived Python process accumulates heap fragmentation, retains cached CUDA allocations, and may leave the CUDA context in a state where the driver spills into shared host memory. The empirical consequence in earlier in-process implementations was a monotonic slowdown across repetitions, with the last repetition running substantially slower than the first — an artefact of the harness that would have contaminated the timing analysis and, in the limit, caused later repetitions to fail outright. Because the operating system reclaims a terminated process's entire address space and destroys its CUDA context unconditionally, running each repetition as a complete process lifetime restores the machine to a clean state between repetitions. Every repetition consequently observes identical resource conditions, and the per-phase timings reported in Section 9 are comparable across seeds.

The same script therefore supports three modes of invocation, selected by command-line flags:

```
python implements/<impl>_mccv.py                              # orchestrate a full run
python implements/<impl>_mccv.py --seed 17 --out-dir <run>    # worker: one repetition
python implements/<impl>_mccv.py --aggregate --out-dir <run>  # re-aggregate an existing run
```

The third mode is what makes a run repairable: a repetition that failed can be re-executed on its own, into the same output directory, and the run re-aggregated afterwards without disturbing the repetitions that succeeded.

The protocol itself is implemented once, in a shared harness module, and is parameterised only by an encoder factory, a dictionary-learner factory, an implementation name and a dataset name. Each concrete experiment script is consequently a few lines long. Because every arm of the study — across all encoder and dictionary-learner combinations — is produced by the same machinery, differences between arms can be attributed to the components under comparison rather than to divergent evaluation code, which is precisely the property a comparative study requires.

## 6. Decision-Threshold Calibration

The corpus is heavily imbalanced, and on such data the choice of decision threshold is not a detail but a first-order determinant of every threshold-dependent metric. A classifier that would appear worthless at the default threshold of 0.5 may be competitive at 0.2, and reporting the former would confound the quality of the representation with an arbitrary convention.

The threshold is therefore treated as a fitted quantity. For each classifier, the harness constructs the precision–recall curve on the **validation** split and selects the threshold maximising F1 on that split. The four thresholds so obtained — one per scikit-learn classifier — are then *frozen* and supplied to a second evaluation pass on the test split, which applies them verbatim rather than re-optimising.

This ordering is the crux. Selecting a threshold on the test labels and then reporting metrics computed at that threshold would be a form of test-set leakage: the reported F1 would be a maximum over thresholds rather than the performance of a deployable decision rule, and the resulting figures would be optimistically biased by an amount that grows with the imbalance. Calibrating on a disjoint validation split and freezing the result yields a threshold that is fixed before the test data is consulted — the same situation as deployment, where the operating point must be chosen in advance.

Beyond thresholding, class imbalance is addressed at the classifier level through balanced class weighting: logistic regression, the linear support-vector machine and the random forest are fitted with `class_weight="balanced"`, while gradient boosting, which exposes no such parameter, receives explicit per-sample weights computed by the same $n/(k \cdot n_c)$ formula. Threshold calibration and class weighting thus address the imbalance at two distinct points — the decision rule and the loss — and are applied consistently across every repetition.

## 7. Classifiers and Metric Suite

Each repetition evaluates six classifiers on the held-out test split: four conventional discriminative models fitted on the sparse codes (logistic regression, gradient boosting, a calibrated linear SVM, and a random forest), together with two sparse-representation classifiers (SRC) that operate directly on the encoder-space embeddings and classify by reconstruction residual, in a pure variant and in a Fisher-discriminative variant.

The SRC arms require one qualification, which the harness enforces explicitly. Sparse-representation classification assigns a sample to the class whose block of dictionary atoms reconstructs it best, and is therefore defined only for a *class-partitioned* dictionary in which each block of atoms carries a known class identity. Supervised learners of the FDDL family construct such a dictionary by design. An unsupervised learner never observes the labels, so its atoms carry no class identity, and slicing them into blocks by index would impose a labelling the learner never inferred, yielding residuals with no interpretation. For such configurations the harness detects the absence of a class-partitioned dictionary and records the SRC metrics as `NaN` rather than fabricating them. The columns are retained rather than dropped so that every implementation produces a table with an identical schema and results remain aligned column-for-column across arms.

Eleven scalar metrics are recorded per classifier, grouped into three families whose names state their own averaging so that a column can never be misread:

- **Macro-averaged** (`Macro-Precision`, `Macro-Recall`, `Macro-F1`, `Macro-PR-AUC`), in which both classes count equally regardless of size;
- **Symmetric over the whole split** (`Accuracy`, `ROC-AUC`, `MCC`), which have no per-class variant in a binary problem;
- **Minority-class only** (`Minority-Precision`, `Minority-Recall`, `Minority-F1`, `Minority-PR-AUC`), which are the headline figures under imbalance.

Recording all three families is a response to a specific hazard: on this corpus a trivial majority-class classifier already attains roughly 0.95 accuracy while attaining exactly 0.0 MCC, and macro-F1 and minority-F1 can tell markedly different stories about the same predictions. The Matthews correlation coefficient is given particular weight in the discussion precisely because it is the only scalar in the suite whose chance baseline is 0.0 irrespective of the class ratio. To keep this comparison explicit, the harness also computes the chance baseline attainable by a knowledge-free model on each split — the best trivial constant classifier for threshold metrics, a random ranker for ranking metrics — and reports it alongside the achieved value.

Missing metrics are treated as errors rather than silently omitted: if a result dictionary lacks an expected key the repetition fails loudly. Given that a repetition may run for hours, discovering an absent column only after the full run has completed would be an expensive failure mode.

## 8. Aggregation and Reported Statistics

Once every repetition has terminated, the aggregator reads the per-repetition table, de-duplicates by master seed retaining the most recent row for each — so that a re-executed repetition cleanly supersedes its earlier result instead of being double-counted — and computes, for each of the $6 \times 11$ metric columns, the mean, the sample standard deviation with one degree of freedom removed, and the 95 % confidence half-width defined in Section 2.

Two choices in this computation merit comment. The standard deviation uses $\text{ddof}=1$ because the repetitions are a sample rather than a population, and the population form would understate the spread appreciably at the small repetition counts that a multi-hour pipeline permits. The confidence interval uses the Student's *t* critical value with $B-1$ degrees of freedom rather than the normal approximation, for the same reason: at $B = 5$ the *t* quantile is 2.776 against the normal's 1.96, so using the normal value would narrow the interval by roughly 30 % on no statistical grounds whatever. Where SciPy is unavailable the implementation falls back to the normal quantile, and this substitution is noted rather than concealed.

Each summary row carries its own effective sample size $n$. This is not decorative: if repetitions have failed and been skipped, $n$ is the number that actually contributed, and the interval is to be read against that number rather than against the number of repetitions originally planned.

The number of repetitions $B$ is a compute-bound parameter rather than a statistical one. Five repetitions were used for the initial studies, and two for the broader encoder × dictionary-learner grid, where the number of configurations multiplied the per-repetition cost by more than an order of magnitude. This trade-off should be stated frankly when interpreting the results: at $B = 2$ the confidence half-widths are wide — the *t* quantile alone is 12.71 — and the intervals for such runs are best read as an indication that two independent partitions produced consistent results, not as a precise bound on the generalisation error. The mean and standard deviation remain informative, and any configuration whose reported spread is large relative to the difference under discussion is flagged as such rather than reported as a clean win.

## 9. Artefacts, Provenance and Auditability

Each run writes a self-describing directory whose name encodes the implementation, the dataset, the resulting dictionary size and both the start and completion timestamps, for example `mc_cv_fsm_lcksvd_nci_full_atoms512_20260805_210537_20260805_213119`. The elapsed duration of a run is therefore legible from its directory name alone. Five artefacts are produced:

- **`per_run_metrics.csv`** — one row per repetition, keyed by master seed, carrying the dictionary size and all sixty-six metric columns under self-describing `Classifier/Metric` names. This file is the single source of truth from which every summary is derived.
- **`per_run_timings.csv`** — one row per repetition giving the wall-clock seconds spent in each pipeline phase: data loading, partitioning, encoder and dictionary fitting, sparse coding of each split, and each classifier's validation and test passes, measured individually.
- **`summary_mean_std.csv`** — the aggregated mean, standard deviation, confidence half-width and sample size for each metric.
- **`summary_timings.csv`** — per-phase mean, standard deviation, minimum and maximum across repetitions. This view answers a question the metric tables cannot: *which* phase is responsible for a slow repetition, distinguishing, for instance, a support-vector machine that failed to converge on an unlucky partition from a genuinely expensive dictionary fit.
- **`manifest.json`** — the run's provenance record.

The manifest deserves particular emphasis, because the CSV files record *how well the pipeline scored* while the manifest records *what was actually built*. It carries the run-level header (implementation, dataset, planned seeds, split policy, start and end times) together with one entry per repetition giving that repetition's derived sub-seeds, the realised sizes of its four splits, the encoder and dictionary-learner classes, the total number of atoms, and — the field that motivated the manifest's existence — `n_selected_features`, the size of the vocabulary that the encoder's adaptive feature-selection cut retained for that particular partition.

That last quantity is resampled with the partition and therefore differs from repetition to repetition; without the manifest it would be visible only in the workers' console output and would be lost once the terminal scrolled. Recording it makes the vocabulary size a first-class experimental observable: the aggregator folds the per-repetition counts into a run-level mean, standard deviation, minimum and maximum, so the stability of the encoder's feature selection can be inspected across partitions in the same way as any performance metric. In a representative run, for instance, the two repetitions selected 451 and 479 features respectively — a spread that is itself part of the experimental finding, and one that would otherwise be invisible.

Encoder attributes are read defensively when the manifest entry is assembled: an encoder that trims on a fixed budget and exposes no score curve simply records `null` for the fields that do not apply to it, rather than aborting a multi-hour run over a missing attribute. Where the dictionary learner exposes a notion of effective dictionary size — as a non-parametric Bayesian learner does, this being the entire point of its prior — that value is recorded alongside the nominal atom count.

## 10. Fault Tolerance and Resumability

Because a single run may occupy a machine for many hours, the harness is designed on the assumption that it *will* be interrupted, and its persistence strategy follows accordingly.

Writes are append-only and occur at repetition granularity. A worker appends its metrics row, its timings row and its manifest entry immediately upon completing, and then exits. If the machine fails during the fourth repetition, the first three are already durable on disk and only the missing repetitions need to be re-executed. Re-execution is idempotent at the repetition level: both the CSV reader and the manifest apply a last-one-wins rule keyed on the master seed, so a re-run repetition supersedes its earlier entry rather than duplicating it.

The manifest, which is read-modify-written by every worker in turn, is committed atomically through a temporary file followed by an atomic rename. A worker killed mid-write therefore leaves the previous manifest intact rather than a truncated file that would break every subsequent worker's read.

Repetition failures are, by default, logged and skipped rather than fatal, so that the surviving repetitions still yield a summary; the per-metric sample size records how many actually contributed, and a prominent warning names the failed seeds together with the exact command needed to fill them in. A `--fail-fast` flag inverts this behaviour for debugging. One case is treated as unconditionally fatal: if *every* repetition fails, the run exits with a non-zero status rather than reporting a hollow success, on the reasoning that a uniform failure indicates a deterministic defect in the code rather than a transient environmental fault, and should not be allowed to masquerade as a completed experiment. The final directory rename, by contrast, is best-effort: a rename blocked by an open handle or an antivirus lock is reported as a warning and never permitted to destroy results that have already been computed.

## 11. Summary of the Protocol

The procedure executed for each configuration under study can be stated compactly:

```
Input : dataset D, encoder factory E, dictionary-learner factory L,
        master seeds s_1..s_B
for b = 1..B:                                  # one OS process per repetition
    seed all global RNGs with s_b
    (s_split, s_enc, s_dict, s_clf) <- SeedSequence(s_b)
    partition D into (vocab_train, ML_train, val, test)
        = (50%, 20%, 15%, 15%), stratified, using s_split
    encoder    <- E(s_enc);   dict <- L(s_dict)
    fit encoder vocabulary and dictionary on vocab_train
    compute sparse codes for ML_train, val, test; fit MaxAbs scaler on ML_train
    fit classifiers on (ML_train codes, s_clf)
    calibrate one decision threshold per classifier on val   # F1-optimal
    evaluate on test with those thresholds frozen            # 6 classifiers x 11 metrics
    append metrics, timings and provenance to the run directory
aggregate: per metric, report mean, sample std (ddof = 1), 95% t-CI and n
```

The design principle running through every element of this protocol is that each stage of the pipeline is fitted on data disjoint from the data used to evaluate it, that the only source of variation between repetitions is the master seed, and that everything required to reproduce or audit a reported figure — the seeds, the realised split sizes, the selected vocabulary size, the atom count, the per-phase timings and the library environment — is written to disk alongside the figure itself.
