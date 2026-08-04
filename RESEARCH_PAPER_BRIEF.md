# Brief: LAPAS Research Paper (Springer LNCS Format)

This file is an instruction brief for an AI (or a human) that will write the
final research paper describing this project. It is not the paper itself. It
exists so the writer has every fact, constraint, and structural decision in
one place and does not need to re-derive them from the codebase or invent
numbers. Every metric quoted below was pulled directly from the code and
data in this repository on 2026-07-31; do not alter, round differently, or
fabricate any number not present here without re-running the underlying
script and citing the new source file.

---

## 1. Task Definition

Produce a complete academic paper describing the LAPAS (Loan Approval
Prediction & Advisory System) Honours project, formatted for Springer's
**Lecture Notes in Computer Science (LNCS)** proceedings template. The paper
must read as a genuine conference/workshop paper: abstract, introduction,
related work, methodology, system architecture, experiments, results,
discussion, conclusion, references. It should function as the write-up of
the Honours research, not as marketing copy or a user manual.

## 2. Hard Constraints (non-negotiable)

1. **Format**: Springer LNCS LaTeX class (`llncs.cls`, `\documentclass[runningheads]{llncs}`), single-column, Springer reference style (`splncs04.bst` or equivalent numbered style). See Section 9 for exact deliverable structure.
2. **Length**: **Maximum 27 pages**, including references and figures. Target 22-25 pages of body content so the bibliography and any appendix fit inside the limit. See Section 8 for the page budget.
3. **No em dashes anywhere in the text.** Use commas, colons, semicolons, or parentheses instead. This applies to prose, captions, and table content. (Mirrors a real constraint already enforced in `src/ai_advisor/advisor.py`'s system prompt — the paper should hold itself to the same standard it documents.)
4. **No emojis anywhere.**
5. **Diagrams**: clean, minimalist, warm, neutral colour palette only. **No blue.** See Section 7 for the exact palette and rules.
6. **All figures must be regenerated for the paper**, not lifted as-is from `reports/*.png`. Those PNGs were produced by `scripts/tfidf_chroma.py` using default matplotlib/seaborn styling and likely include blue and other saturated colours that violate constraint 5. Re-plot from `reports/tfidf_chroma_metrics.csv` and `models/model_comparison.csv` using the palette in Section 7.
7. **All data, metrics, and hyperparameters must be factual**, sourced from this repository. Do not invent benchmark numbers, dataset sizes, or citation details. Where a genuinely unresolved gap exists (e.g. an experiment that is designed but not yet executed), say so explicitly rather than inventing a result. Section 6 flags every such gap.
8. Every external work cited in Related Work must be a real, verifiable publication. Do not fabricate citations, DOIs, or author names. If uncertain whether a specific paper exists, prefer citing the well-established, easily-verifiable foundational papers listed in Section 6.7 over an obscure or uncertain one.

## 3. Audience and Voice

Honours-level (undergraduate research, University of Johannesburg) computer
science paper. Formal, precise, third person. Confident about what was
built and measured; honest and specific about limitations. No hedging
filler ("it is important to note that..."), no marketing adjectives
("cutting-edge", "state-of-the-art" used loosely), no first-person "we
believe" without evidence attached.

## 4. Project One-Line Summary

LAPAS is a loan approval prediction and advisory system combining (1) a
multi-classifier supervised learning pipeline for binary loan approval
prediction, (2) a retrieval-augmented generation (RAG) pipeline that
produces natural-language advisory reports grounded in real regulatory and
lending-policy documents, and (3) a controlled benchmark comparing sparse
(TF-IDF) and dense (sentence-embedding) retrieval methods for that RAG
pipeline. All three are exposed through a FastAPI backend and a five-page
Streamlit frontend backed by PostgreSQL.

---

## 5. Paper Structure and Content Brief

Write the sections in this order. Each subsection below states what must be
covered and supplies the exact facts to use. Do not add sections beyond
this list (no separate "Future Work" chapter distinct from Conclusion,
for instance) unless doing so measurably improves clarity within the page
budget.

### 5.1 Title

Should name the system and its two core contributions: prediction and
advisory generation. Example direction (do not copy verbatim, refine it):
"LAPAS: A Loan Approval Prediction and Retrieval-Augmented Advisory System
with a Comparative Study of Sparse and Dense Retrieval". Keep it under
~20 words per LNCS convention.

### 5.2 Abstract (150-250 words, LNCS convention)

Must state: the problem (opaque, single-outcome loan approval systems
give applicants no actionable path forward), the approach (three
components: classifier comparison, RAG-based advisory generation over real
lending-policy and regulatory documents, and a sparse-vs-dense retrieval
benchmark), the headline results (best classifier ROC-AUC, retrieval
latency/quality trade-off), and the contribution (an end-to-end reproducible
system plus an empirical comparison others can build on). Follow with 4-6
keywords, LNCS style, e.g.: credit scoring, retrieval-augmented generation,
XGBoost, TF-IDF, dense retrieval, explainable AI.

### 5.3 Introduction (~2 pages)

Cover, in order:

1. **Motivation.** Loan approval/rejection systems typically return a
   binary or probabilistic outcome with no explanation the applicant can
   act on. Regulatory pressure (e.g. South Africa's National Credit Act,
   Basel credit risk principles, CFPB guidance in the US) increasingly
   expects explainability in consumer credit decisions. Cite the actual
   regulatory source documents used in this project (Section 6.5).
2. **Problem statement.** Two coupled problems: (a) which classical ML
   algorithm best predicts approval on a realistic, imbalanced applicant
   dataset, and (b) how to generate a grounded, non-hallucinated advisory
   explanation for the applicant, and which retrieval strategy (sparse
   TF-IDF vs dense embeddings) best supports that generation step.
3. **Contributions**, stated as a numbered list:
   - A comparative evaluation of four classifiers (Random Forest, XGBoost,
     Gaussian Naive Bayes, and a from-scratch logistic regression
     implementation) on a 45,000-row applicant dataset, ranked by a
     composite of eight metrics.
   - A RAG advisory pipeline grounded in nine real regulatory and
     lending-policy source documents, using per-applicant local feature
     attribution (LIME) plus a Mistral-7B instruction-tuned LLM.
   - A controlled, 20-query benchmark of TF-IDF versus ChromaDB
     dense-embedding retrieval, measuring latency, relevance, and
     rank agreement.
   - A working, deployed reference implementation (FastAPI + PostgreSQL +
     Streamlit) rather than an offline notebook study.
4. **Paper organisation** paragraph (standard LNCS closer).

### 5.4 Related Work (~2.5-3 pages)

Structure into three subsections mirroring the three contributions. For
each, cite genuine, verifiable literature (see Section 6.7 for a safe
starter list) and then state concretely how this project's approach
differs or what gap it addresses. Do not just summarise other papers;
every paragraph should end with a sentence connecting it back to a design
decision in this project (e.g. "unlike X, which evaluates only accuracy,
this work reports Brier score and MCC specifically because the dataset is
imbalanced at 22% positive class").

1. **Credit scoring / loan approval prediction.** Classical ML approaches
   to credit scoring (logistic regression as an industry baseline, tree
   ensembles as the modern standard). Position this project's four-way
   comparison and its inclusion of a hand-implemented logistic regression
   as a transparent baseline against the field's use of ensemble methods.
2. **Explainable AI for credit decisions.** LIME and SHAP as local
   attribution methods; note this project uses LIME throughout (not SHAP)
   for macOS/dependency reasons documented in Section 6.4, and discuss the
   trade-off honestly.
3. **Retrieval-augmented generation and retrieval method comparison.**
   RAG as a technique for grounding LLM output in source documents; prior
   work comparing sparse lexical retrieval (BM25/TF-IDF) against dense
   embedding retrieval in general domains. Position this project's
   contribution as a small, domain-specific (consumer lending) empirical
   comparison rather than a general-IR benchmark, and note the sample size
   (20 queries) as a scope limitation to be acknowledged, not hidden.

### 5.5 System Architecture (~2.5 pages, at least one full diagram)

Describe the end-to-end system as three pipelines converging into shared
serving infrastructure. Use the data flow already documented in
`CLAUDE.md`'s "Data Flow" diagram as the source of truth, redrawn as a
proper figure (Section 7 for style):

```
data/raw/loan_data.csv
  -> feature engineering (database/feature_eng.py)
  -> data/processed/loan_features.csv
  -> model training (scripts/train_model.py)
  -> models/best_model.joblib
  -> [ LoanContextBuilder ] <- data/loan_strategy_docs/ (TF-IDF store + dense vector store)
  -> LoanAdvisor.advise() -> Markdown advisory report
  -> FastAPI backend -> Streamlit frontend
```

Cover:

- **Data layer**: PostgreSQL via SQLAlchemy ORM, 8 tables:
  `loan_applicants`, `engineered_features`, `ml_models`,
  `model_predictions`, `retrieval_documents`, `rag_explanations`,
  `rag_explanation_chunks`, `evaluator_assessments`. Note the
  `evaluator_assessments` table exists to support a planned third
  experiment (structured human rubric scoring, four dimensions on a
  1-5 Likert scale, from a panel of evaluators, intended for Cohen's
  Kappa analysis) but, as of this writing, **no evaluator data has been
  collected** (see Section 6.6, do not report a kappa value).
- **Serving layer**: FastAPI backend (`backend/main.py`) with three
  routers (`applicants`, `advisory`, `comparisons`) and process-wide
  singletons for the classifier and both retrieval stores, warmed up in a
  `startup` event specifically to control import order (see Section 6.4).
- **Presentation layer**: Streamlit, five pages (application submission,
  applicant browser, aggregate dashboard, per-applicant AI advisory
  report with LIME attribution and PDF export, and a findings page that
  surfaces the classifier and retrieval comparison results produced by
  this paper's experiments).
- **Deployment/runtime environment**: Python 3.10, macOS Intel (x86_64),
  which imposed several concrete engineering constraints described in
  Section 6.4. This is worth a short paragraph: production ML systems
  frequently must be engineered around platform-specific runtime
  conflicts (here, competing OpenMP runtimes between XGBoost and
  PyTorch/sentence-transformers), and documenting the resolution is itself
  a reproducibility contribution.

### 5.6 Methodology (~5-6 pages, the largest section)

#### 5.6.1 Dataset

- Source file: `data/raw/loan_data.csv`.
- **45,000 rows, 14 raw columns.**
- Target: `loan_status`, binary (1 = approved, 0 = rejected).
- **Class balance: 10,000 approved (22.2%) vs 35,000 rejected (77.8%).**
  State explicitly that this is an imbalanced binary classification
  problem and that this motivated the choice of evaluation metrics
  (Section 5.6.3) beyond raw accuracy.
- Raw feature list (14): `person_age`, `person_gender`,
  `person_education`, `person_income`, `person_emp_exp`,
  `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_int_rate`,
  `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`,
  `previous_loan_defaults_on_file`, `loan_status`.
- Light cleaning performed: enum normalisation (e.g. "Doctorate" to
  "Doctor", "HOMEIMPROVEMENT" to "HOME_IMPROVEMENT"), boolean coercion of
  the previous-default flag, and capping of implausible ages (>100) to
  100.

#### 5.6.2 Feature Engineering

Implemented in `database/feature_eng.py`. The 14 raw columns are expanded
to **36 columns** (14 raw/passthrough + 21 engineered + 1 pipeline-version
metadata column). Present as a table grouped by category, each with the
exact derivation:

| Category | Feature | Definition |
|---|---|---|
| Financial ratios | `monthly_income` | `person_income / 12` |
| Financial ratios | `debt_to_income_ratio`, `loan_to_income_ratio` | both set to the source `loan_percent_income` (dual-named for schema completeness) |
| Financial ratios | `monthly_loan_burden` | `loan_amnt * (1 + loan_int_rate/100) / 12` (simple-interest annualisation) |
| Financial ratios | `affordability_ratio` | `1 - (monthly_loan_burden / monthly_income)`, clipped at 0 |
| Financial ratios | `credit_history_to_age_ratio` | `cb_person_cred_hist_length / person_age` |
| Age/employment | `emp_to_age_ratio` | `person_emp_exp / person_age` |
| Age/employment | `loan_per_age` | `loan_amnt / person_age` |
| Age/employment | `young_inexperienced` | boolean: age < 25 AND employment experience = 0 |
| Credit quality | `credit_score_tier` | standard FICO-style bands: Exceptional 800-850, Very Good 740-799, Good 670-739, Fair 580-669, Poor 300-579 |
| Credit quality | `thin_credit_file` | boolean: credit history length < 2 years |
| Credit quality | `score_per_history_year` | `credit_score / cb_person_cred_hist_length` |
| Credit quality | `credit_risk_interaction` | boolean: interest rate above the dataset median AND credit score below the dataset median |
| Income/burden | `income_bucket` | four buckets from dataset-derived quartile-like cut points: low <= 47,204; mid_low <= 67,048; medium <= 95,789; high above |
| Income/burden | `high_loan_burden_flag` | boolean: `loan_percent_income` > 0.30 |
| Employment | `employment_stability` | "stable" if employment experience >= 2 years, else "unstable" |
| Risk | `composite_risk_score` | weighted sum of seven binary risk signals (see below), weights sum to 1.0 |
| Risk | `is_high_risk` | boolean: `composite_risk_score` at or above the dataset's 75th percentile |
| Homeownership | `homeownership_score` | ordinal 0-3: OTHER=0, RENT=1, MORTGAGE=2, OWN=3 |
| Homeownership | `stability_income_interaction` | `homeownership_score * log1p(person_income)` |
| Intent | `intent_risk_score` | ordinal 0 (safest, EDUCATION) to 5 (riskiest, VENTURE); HOME_IMPROVEMENT=1, PERSONAL=2, MEDICAL=3, DEBTCONSOLIDATION=4 |

`composite_risk_score` weights (must be reproduced exactly, they sum to
1.0): debt-to-income ratio above 0.40 (0.20), loan-to-income ratio above
0.40 (0.15), thin credit file (0.10), credit risk interaction (0.15), high
loan burden flag (0.10), previous default on file (0.20), young and
inexperienced (0.10).

Note for the paper: `credit_risk_interaction` and `is_high_risk` are
computed from **dataset-wide statistics** (medians and a quantile), not
per-row, which means single-applicant inference at serving time must reuse
precomputed reference statistics rather than recomputing them live. This
is a genuine design detail worth one sentence in the methodology as it
affects reproducibility of any single prediction in isolation.

Two implementation notes safe to mention as scope/honesty statements: the
raw dataset contains no `loan_grade` column (some downstream schema fields
that reference a grade are unused for this dataset), and the codebase
lists eight target algorithms as aspirational (`svm`, `gradient_boosting`,
`lightgbm`, `catboost` alongside the four implemented ones); only report on
the four that are actually implemented.

#### 5.6.3 Classifiers

Implemented in `src/classifier/classifier.py` via a unified `LoanClassifier`
wrapper around four algorithms, each behind a common
`fit`/`predict`/`predict_proba`/`tune`/`evaluate`/`save`/`load` API using a
scikit-learn `Pipeline`:

1. **Random Forest** (`sklearn.ensemble.RandomForestClassifier`),
   `class_weight="balanced"`, parallelised (`n_jobs=-1`).
2. **XGBoost** (`xgboost.XGBClassifier`), imbalance handled via
   `scale_pos_weight` computed at fit time as `n_negative / n_positive`
   (rather than sklearn's `class_weight`, which XGBoost does not support
   natively), `eval_metric="logloss"`.
3. **Gaussian Naive Bayes** (`sklearn.naive_bayes.GaussianNB`), used with
   feature scaling (`StandardScaler`) since it assumes normally
   distributed features.
4. **Custom Logistic Regression**: a from-scratch implementation
   (`src/classifier/logistic_regression.py`, gradient descent with L2
   regularisation), used as a transparent, dependency-free baseline
   against the three library-backed models, also scaled.

State explicitly that scaling (`StandardScaler`) is applied only for
logistic regression and naive Bayes, and withheld for the two tree-based
models, since they are scale-invariant. Optional probability calibration
(`CalibratedClassifierCV`, sigmoid method) exists in the pipeline but was
**not** used for the champion model reported in Section 5.7 (see below).

**Hyperparameter search.** `LoanClassifier.tune()` supports both
`GridSearchCV` (exhaustive) and `RandomizedSearchCV` (sampled), both using
`StratifiedKFold` cross-validation (default 5 folds) scored by
`roc_auc` by default. Each algorithm has a defined search space (report
the random-forest and XGBoost spaces as representative examples in a
table or listing, since they are the two top performers):

- Random Forest random-search space: `n_estimators` in {50, 100, 200,
  400}; `max_depth` in {None, 5, 10, 20, 30}; `min_samples_split` in {2,
  5, 10, 20}; `min_samples_leaf` in {1, 2, 4, 8}; `max_features` in
  {"sqrt", "log2", 0.5}.
- XGBoost random-search space: `n_estimators` in {50, 100, 200, 400};
  `max_depth` in {3..9}; `learning_rate` in {0.005, 0.01, 0.05, 0.1, 0.2,
  0.3}; `subsample` and `colsample_bytree` each in {0.5..1.0 step 0.1};
  `reg_alpha` in {0, 0.01, 0.1, 1.0, 5.0}; `reg_lambda` in {0.5, 1.0, 2.0,
  5.0, 10.0}; `min_child_weight` in {1, 3, 5, 7}.
- The training entry point exposes this via CLI flags
  (`scripts/train_model.py --tune --tune-n-iter 30`).

**Important factual caveat, state this honestly rather than implying an
exhaustively tuned model:** the currently promoted champion model
(`models/best_model.joblib`, metadata in `models/best_model.json`) was
trained with `best_params: null`, `calibrate: false`,
`scale_features: false`, i.e. **using default hyperparameters**, not the
output of the tuning search described above. The tuning capability exists
and is fully wired into the training script, but the specific artefact
reported as the champion in this paper's results was not itself the
product of a hyperparameter search run. This should be framed as a
documented direction for immediate future work, not concealed.

#### 5.6.4 Evaluation Metrics

Defined in `src/classifier/evaluate.py`. Motivate the metric set with the
class imbalance fact from 5.6.1: accuracy alone is misleading at a 78/22
split, so eight metrics are reported per model:

- **Accuracy**
- **Precision** and **Recall** (positive class = approved)
- **F1 (macro)** and **F1 (weighted)**
- **ROC-AUC**
- **Average precision** (area under the precision-recall curve, more
  informative than ROC-AUC under class imbalance)
- **Brier score** (mean squared error of the predicted probability
  against the outcome; measures probability calibration quality, lower is
  better)
- **Matthews Correlation Coefficient (MCC)** (a single balanced measure,
  range -1 to +1, robust to class imbalance)

**Composite ranking methodology.** Each of five higher-is-better metrics
(ROC-AUC, F1-weighted, average precision, accuracy, MCC) is independently
rank-ordered across the candidate models; Brier score is rank-ordered
separately (lower is better). The composite rank is the mean of these six
per-metric ranks, and models are sorted ascending by that mean (rank 1.0
would mean a model won on every metric). Separately, one designated metric
(configurable, default **ROC-AUC**, `MODEL_SELECTION_METRIC` in
`config/settings.py`) is used to promote a single "champion" model that
gets saved to `models/best_model.joblib` and served in production. State
explicitly in the results/discussion that these two selection procedures
can disagree (they do here, see 5.7).

#### 5.6.5 Retrieval-Augmented Advisory Pipeline

Describe as four stages:

1. **Document ingestion** (`src/ai_advisor/document_loader.py`): source
   corpus of **nine real documents** in `data/loan_strategy_docs/`: five
   PDFs (Basel credit risk principles, a CFPB consumer credit score guide,
   South Africa's National Credit Regulator consumer debt-management
   guide, the National Credit Act consumer guide, and a World Bank
   responsible digital credit report) and four plain-text policy files
   (credit guidelines, an improvement roadmap, a lending policy, and a
   risk assessment reference). Documents are chunked into overlapping
   word-level windows, chunk size 400 words with 50-word overlap
   (the production default in `document_loader.py`; note this differs
   from the `RAG_CONFIG` dictionary in `config/settings.py`, which
   specifies 512/64 but is not the value actually applied by the loader,
   an internal inconsistency worth naming honestly rather than silently
   reconciling).
2. **Two parallel retrieval indexes** built over the same chunked corpus:
   - **Sparse (TF-IDF)**: `src/tf_idf/tf_idf_store.py`, scikit-learn
     `TfidfVectorizer`, vocabulary capped at 20,000 terms, unigrams and
     bigrams (`ngram_range=(1,2)`), sublinear term-frequency scaling,
     English stop-word removal, cosine similarity for ranking.
   - **Dense**: `src/ai_advisor/vector_store.py`, ChromaDB
     `PersistentClient` with `sentence-transformers/all-MiniLM-L6-v2`
     embeddings, cosine similarity (HNSW index), embeddings explicitly
     computed and passed to ChromaDB (not delegated to ChromaDB's own
     embedding function). Both stores expose an identical query interface
     so they are drop-in interchangeable, which is what makes a controlled
     comparison possible (Section 5.7.2).
3. **Context construction** (`src/ai_advisor/loan_context_builder.py`):
   given an applicant (by database id or raw feature row), the training-time
   feature vector is reconstructed with one-hot encoding aligned to the
   model's stored feature list, the champion classifier scores it, and the
   top-15 globally important features plus a risk tier are packaged into a
   structured context.
4. **Generation** (`src/ai_advisor/advisor.py`): retrieves the top-`n_docs`
   (default 5) most relevant chunks from whichever store is selected,
   builds a Mistral instruction-format prompt
   (`<s>[INST] {system + user} [/INST]`), and calls a hosted
   `mistralai/Mistral-7B-Instruct-v0.2` model through the HuggingFace
   Inference API at `temperature=0.4` and `max_new_tokens=1800`. The
   system prompt explicitly instructs the model never to use em dashes.
   A deterministic, rule-based Markdown fallback report generator exists
   for when the LLM endpoint is unavailable, ensuring the system degrades
   gracefully rather than failing outright.
5. **Per-applicant local explainability**: separate from the RAG pipeline,
   `backend/services/lime_service.py` uses `lime.lime_tabular` to produce
   per-applicant local feature attribution (15 features, 5,000 perturbation
   samples). Note explicitly that this project uses **LIME, not SHAP**,
   throughout, a deliberate choice made to avoid a `numba`/`llvmlite`
   dependency conflict on the macOS Intel development environment. Frame
   this as a pragmatic engineering trade-off in the methodology, and revisit
   it honestly as a limitation in Section 5.9.

### 5.7 Experiments and Results (~4-5 pages, figures live here)

State up front that this section reports two completed, data-backed
experiments and names one designed-but-not-yet-executed experiment rather
than fabricating its outcome.

#### 5.7.1 Experiment 1: Classifier Comparison

Setup: all four classifiers trained on the same processed dataset (train/
test split, `test_size=0.2`, `random_state=42`, stratified), evaluated with
the metric suite from 5.6.4. Source: `models/model_comparison.csv`.
Reproduce this exact table (round to 3-4 decimal places consistently):

| Rank | Model | Accuracy | Precision | Recall | F1 (macro) | F1 (weighted) | ROC-AUC | Avg. Precision | Brier | MCC | Composite rank | Champion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest | 0.9289 | 0.8922 | 0.7735 | 0.8919 | 0.9270 | 0.9757 | 0.9313 | 0.0512 | 0.7872 | 1.33 | No |
| 2 | XGBoost | 0.9193 | 0.7762 | 0.8950 | 0.8892 | 0.9213 | 0.9776 | 0.9384 | 0.0555 | 0.7821 | 1.67 | **Yes** |
| 3 | Logistic Regression (custom) | 0.8358 | 0.5905 | 0.8515 | 0.7923 | 0.8451 | 0.9269 | 0.8132 | 0.1498 | 0.6083 | 3.17 | No |
| 4 | Gaussian Naive Bayes | 0.7632 | 0.4838 | 0.9785 | 0.7346 | 0.7830 | 0.9343 | 0.7864 | 0.2277 | 0.5685 | 3.83 | No |

Discuss, with actual numbers, not vague adjectives:

- The two tree ensembles (Random Forest, XGBoost) clearly outperform the
  linear and probabilistic baselines across nearly every metric, both
  above 0.975 ROC-AUC versus 0.93 and 0.93 for logistic regression and
  naive Bayes respectively.
- **Random Forest wins the composite multi-metric rank (1.33)** but
  **XGBoost wins on ROC-AUC alone (0.9776 vs 0.9757) and is therefore the
  model actually promoted to production**, since the system's champion
  selection uses ROC-AUC as its single criterion
  (`MODEL_SELECTION_METRIC`). This is a genuine, reportable finding: the
  two model-selection philosophies (single-metric versus composite
  multi-metric ranking) disagree on this dataset, and the paper should
  discuss what each optimises for (Random Forest has a notably better
  Brier score, 0.0512 vs 0.0555, i.e. better-calibrated probabilities,
  which is not what ROC-AUC rewards).
- Naive Bayes shows the classic high-recall, low-precision pattern for
  a generative model on correlated features (recall 0.9785 but precision
  only 0.4838), explain this as an artefact of its conditional-independence
  assumption not holding for the engineered feature set (several features
  are deliberately correlated, e.g. `debt_to_income_ratio` and
  `loan_to_income_ratio` are identical by construction).
- The custom, from-scratch logistic regression baseline (recall 0.8515,
  ROC-AUC 0.9269) performs credibly close to library-grade linear models,
  which is worth stating as a validation of the from-scratch
  implementation's correctness, not just a footnote.

#### 5.7.2 Experiment 2: Sparse vs Dense Retrieval Benchmark

Setup: 20 hand-written, domain-specific loan and credit queries (e.g.
"applicant with poor credit score and high debt to income ratio", "steps
to improve credit score before reapplying for a loan"), each run against
both the TF-IDF store and the ChromaDB dense store over the identical
9-document, chunked corpus. Metrics per query: retrieval latency (ms),
top-1 similarity score, mean similarity score across returned results,
Jaccard overlap between the two result sets, and Spearman rank correlation
between the two rankings where both returned overlapping documents.
Source: `reports/tfidf_chroma_metrics.csv` (all 20 rows, exact values,
already computed as means below):

| Metric | TF-IDF (sparse) | ChromaDB (dense) |
|---|---|---|
| Mean latency | 2.85 ms | 33.47 ms |
| Mean top-1 similarity score | 0.193 | 0.598 |
| Mean similarity score (all results) | 0.134 | 0.504 |

Additional cross-method metrics, averaged over the 20 queries:

- **Mean Jaccard overlap of top-k result sets: 0.366** (moderate-low
  agreement; the two retrieval methods often surface substantially
  different documents for the same query).
- **Mean Spearman rank correlation: 0.441** (computed over the 17 of 20
  queries where the correlation was defined; 3 queries produced an
  undefined/blank correlation in the raw data, most likely because the
  overlapping result set for that query was too small or had tied ranks
  to compute a meaningful coefficient, state this as a data-quality
  footnote rather than omitting it).

Discuss:

- **Dense retrieval is roughly an order of magnitude slower** than
  TF-IDF (about 11.7x on a ratio-of-means basis; the mean of the
  per-query speed ratios is close to 12.1x), consistent with the added
  cost of a neural forward pass per query versus a sparse dot product.
- **Dense retrieval returns substantially higher similarity scores**
  on both top-1 and mean-of-results, consistent with dense embeddings
  capturing semantic/paraphrastic matches (e.g. "steps to improve credit
  score" matching a policy passage that never uses those exact words)
  that a lexical method like TF-IDF cannot.
- The **moderate-low Jaccard overlap (0.366) and weak-to-moderate rank
  correlation (0.441)** together indicate the two methods are not
  interchangeable: they frequently retrieve different, only partially
  overlapping sets of relevant passages. This motivates a genuinely open
  practical question for the deployed system: TF-IDF is cheap and fast
  enough for interactive use but appears to under-match paraphrased,
  natural-language queries relative to dense retrieval, while dense
  retrieval is an order of magnitude more expensive per query. State the
  system's current default choice honestly by checking which store the
  running advisory pipeline actually defaults to, rather than assuming.
- Acknowledge the scope limitation directly: **n = 20 queries** is a small
  benchmark. Treat the findings as indicative rather than statistically
  conclusive, and say so in these words or similar, not just in the
  Discussion/Limitations section.

#### 5.7.3 Experiment 3: Human Evaluation of Advisory Quality (designed, not executed)

Report this honestly as **infrastructure that exists but has not yet
produced data**, do not report a Cohen's Kappa value. The database schema
(`EvaluatorAssessment` table) is already designed for a three-evaluator
panel rating each generated advisory report on four dimensions (policy
traceability, factual accuracy, completeness, actionability, each on a
1-5 Likert scale), comparing the full RAG-grounded output against a
local-attribution-only baseline, with inter-rater agreement intended to be
computed via Cohen's Kappa once data is collected. Present this as a
concrete, near-term next step (it belongs partly here as "what the system
is built to measure" and partly in Section 5.9/Conclusion as future work),
not as a completed experiment.

### 5.8 System Optimisations and Engineering Trade-offs (~1.5-2 pages)

This section answers "what did we tune to make this work" at the systems
level, distinct from the ML hyperparameter tuning already covered in
5.6.3. Present as a short table or list, each with cause and effect, drawn
from real, encountered engineering constraints on the macOS Intel (x86_64)
development platform:

- **Sequential (not batched) sentence embedding.** `SentenceTransformer.encode()`
  is called one text at a time rather than in batches, because batch
  encoding triggers a segmentation fault on macOS x86_64 due to OpenMP/
  libdispatch thread contention inside the tokenizer. This trades raw
  embedding throughput for stability.
- **Import and load ordering.** The XGBoost-backed classifier must be
  deserialised via `joblib.load()` **before** PyTorch is ever imported in
  the same process; the reverse order causes a segmentation fault during
  deserialisation (competing OpenMP runtimes). This is why the FastAPI
  backend warms up the classifier in a `startup` event and keeps the
  dense `VectorStore` import lazy.
- **Thread-count pinning.** `OMP_NUM_THREADS=1` and
  `TOKENIZERS_PARALLELISM=false` are set at the top of
  `config/settings.py`, before any other import, to prevent an indefinite
  hang (not a crash) on the first dense-retrieval call in a process that
  has already loaded XGBoost.
- **ChromaDB collection creation without an `embedding_function`.**
  ChromaDB 1.x validates the callable signature of any
  `embedding_function` strictly and rejects a plain Python function;
  explicit embeddings are computed and passed to `add()`/`query()`
  instead, sidestepping the validator entirely.
- **Class imbalance handling differs per algorithm.** `class_weight="balanced"`
  for Random Forest and the custom logistic regression; XGBoost instead
  uses `scale_pos_weight` computed at fit time from the actual class
  ratio in the training fold (`n_negative / n_positive`), since XGBoost
  does not support scikit-learn's `class_weight` API.
- **Selective feature scaling.** `StandardScaler` is applied only ahead of
  logistic regression and naive Bayes, both scale-sensitive; the two
  tree-based models are left unscaled, since splits are scale-invariant
  and unnecessary scaling would only add pipeline overhead.
- **Dataset-wide reference statistics for row-level inference.** Two
  engineered features (`credit_risk_interaction`, `is_high_risk`) are
  defined relative to dataset-wide medians/quantiles at training time; the
  backend's single-row feature engineering path
  (`backend/services/feature_engineering.py`) reuses precomputed reference
  statistics rather than recomputing a median from a single new row, which
  would be undefined.
- **LIME over SHAP** for all local explainability, to avoid a
  `numba`/`llvmlite` dependency conflict on this platform, a deliberate
  and named trade-off rather than an oversight.

Frame the section's closing paragraph around a single thesis: a meaningful
share of the engineering effort in this project went into making a
multi-framework ML/NLP stack (scikit-learn, XGBoost, PyTorch,
sentence-transformers, ChromaDB) coexist reliably on one platform, and
documenting those constraints is itself a reusable contribution for anyone
reproducing this stack.

### 5.9 Discussion and Limitations (~1.5 pages)

Be concrete and specific, referencing the actual numbers already reported,
not generic caveats. Suggested points, all grounded in fact:

- Class imbalance (22.2% positive) is real and was addressed via metric
  choice and class weighting, but no resampling (SMOTE, undersampling)
  was tried; state this as an open comparison point.
- The reported champion model (XGBoost) was **not** the product of the
  hyperparameter search infrastructure that exists in the codebase; a
  tuned run is a concrete, immediately actionable next step, not
  speculative future work.
- The retrieval benchmark (5.7.2) is small (20 queries, 9 source
  documents); results should be read as a controlled pilot, not a
  general claim about sparse versus dense retrieval.
- The advisory generation pipeline depends on a hosted third-party
  inference API (HuggingFace) for its primary LLM path; the rule-based
  fallback exists precisely because this is a single point of failure,
  and that dependency itself is worth naming as a limitation.
- The human-evaluation experiment (5.7.3) that would validate advisory
  *quality*, as opposed to retrieval mechanics, has not yet produced data.
- Only four of the eight algorithms named in the project's own
  configuration (`AVAILABLE_CLASSIFIERS`) are actually implemented;
  SVM, gradient boosting, LightGBM, and CatBoost remain unimplemented and
  would be natural extensions of Experiment 1.
- The system is built and validated on a single dataset; generalisation
  to other lenders' data distributions is untested.

### 5.10 Conclusion (~0.5-1 page)

Summarise the three contributions and their headline numbers in two or
three sentences each (best ROC-AUC 0.9776 by XGBoost; ~12x latency
difference and moderate-low result overlap between TF-IDF and dense
retrieval; a working, reproducible reference system). Close with the two
or three most concrete next steps, drawn directly from Section 5.9 (run
the existing hyperparameter search on the champion algorithm; collect
human evaluator data to compute Cohen's Kappa; extend the retrieval
benchmark's query set).

### 5.11 References

Numbered, Springer LNCS style (`splncs04` produces `[1]`, `[2]`, ...
citations). Two categories:

1. **Academic literature** supporting Related Work (Section 5.4). Use
   Section 6.7's starter list, verify each before including it, and add
   any others the writer can independently verify are real.
2. **Software and tooling citations**, standard practice for a systems
   paper: scikit-learn, XGBoost, ChromaDB, Sentence-Transformers/
   Sentence-BERT, LIME, FastAPI, Streamlit, PostgreSQL. Cite the papers
   behind these tools where one exists (scikit-learn, XGBoost, Sentence-
   BERT, and LIME all have a canonical paper), and a URL/version citation
   otherwise.
3. **Regulatory/primary source documents** actually used to ground the
   RAG pipeline (Section 6.5): cite these as grey literature/technical
   reports (Basel Committee, CFPB, South African National Credit
   Regulator, World Bank), since they are real source documents the
   system retrieves from, not just background reading.

---

## 6. Facts Reference Sheet (for the writer, not for direct inclusion)

Use this section to sanity-check any number that appears in the paper.
Do not copy this section into the paper verbatim; it is scaffolding.

### 6.1 Repository facts

- Language/runtime: Python 3.10, macOS Intel (x86_64), dependency manager `uv`.
- Backend: FastAPI. Frontend: Streamlit (5 pages). Database: PostgreSQL via SQLAlchemy.
- Key dependencies and pinned constraints (`pyproject.toml`): `scikit-learn>=1.7.2`, `xgboost>=3.2.0`, `chromadb>=0.5.0`, `sentence-transformers>=3.0.0`, `torch>=2.0.0,<2.5.0` (no macOS Intel wheels beyond 2.5), `transformers>=4.40.0,<5.0.0`, `lime>=0.2.0.1`, `fastapi>=0.115.0`, `streamlit>=1.58.0`.

### 6.2 Dataset facts

- `data/raw/loan_data.csv`: 45,000 rows x 14 columns.
- Approval rate: 22.22% (10,000 of 45,000).
- `data/processed/loan_features.csv`: 45,000 rows x 36 columns after feature engineering.

### 6.3 Model artefacts on disk

- `models/best_model.joblib` / `.json`: champion is XGBoost, selection metric ROC-AUC = 0.977612, `best_params: null` (default hyperparameters, not tuned), `calibrate: false`, `scale_features: false`, saved 2026-07-14.
- All four trained models also individually persisted: `random_forest`, `xgboost`, `naive_bayes`, `logistic_regression` (`.joblib` + `.json` pairs in `models/`).
- Full comparison table: `models/model_comparison.csv` (reproduced in full in Section 5.7.1).

### 6.4 Platform constraint facts (all confirmed in `CLAUDE.md` and source comments)

- Batch sentence-embedding causes SIGSEGV on macOS Intel; must encode one text at a time.
- `chromadb.create_collection()` must not receive an `embedding_function` argument (ChromaDB 1.x signature validation rejects it).
- `torch<2.5`, `transformers<5`, `sentence-transformers<4` required, no macOS x86_64 wheels exist for newer versions.
- XGBoost must be `joblib.load()`-ed before torch is imported in the same process, or deserialisation SIGSEGVs (reverse order is safe).
- `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` must be set before any other import, or the first dense-retrieval call after the classifier loads hangs indefinitely.

### 6.5 RAG source corpus facts

Nine files in `data/loan_strategy_docs/`:

- `basel_credit_risk_principles_2025.pdf`
- `cfpb_understand_your_credit_score.pdf`
- `ncr_consumer_guide_managing_debt.pdf`
- `ncr_national_credit_act_consumer_guide.pdf`
- `worldbank_responsible_digital_credit.pdf`
- `credit_guidelines.txt`
- `improvement_roadmap.txt`
- `lending_policy.txt`
- `risk_assessment.txt`

Chunking (`document_loader.py` production default): 400-word chunks, 50-word overlap. Note the `RAG_CONFIG` dict in `config/settings.py` states 512/64, which is not what the loader actually uses; call this out as a documented inconsistency rather than silently picking one.

### 6.6 Retrieval benchmark facts

Full per-query data in `reports/tfidf_chroma_metrics.csv`, 20 queries. Aggregate means (computed directly from that file):

- TF-IDF mean latency: 2.851 ms. Chroma mean latency: 33.472 ms.
- TF-IDF mean top-1 score: 0.1931. Chroma mean top-1 score: 0.5976.
- TF-IDF mean (all-results) score: 0.1343. Chroma mean (all-results) score: 0.5042.
- Mean Jaccard overlap: 0.3661.
- Mean Spearman rho (over the 17 queries with a defined value): 0.4412; 3 of 20 queries have a blank/undefined value in the source CSV.
- Existing chart files (`reports/tfidf_chroma_*.png`) were generated by `scripts/tfidf_chroma.py` with default matplotlib/seaborn colours and must be regenerated for the paper under the palette in Section 7, do not reuse them unmodified.

### 6.7 Safe starter citation list (verify before use, do not invent additional details about these)

Only include ones the writer can independently confirm are real, correctly attributed publications:

- Breiman, L. "Random Forests." Machine Learning, 2001. (Random Forest.)
- Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.
- Ribeiro, M. T., Singh, S., Guestrin, C. "Why Should I Trust You? Explaining the Predictions of Any Classifier." KDD, 2016. (LIME.)
- Reimers, N., Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP, 2019. (Sentence-Transformers / basis for `all-MiniLM-L6-v2`.)
- Robertson, S., Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval, 2009. (Sparse lexical retrieval background for the TF-IDF comparison.)
- Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020. (RAG.)
- Lessmann, S. et al. "Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research." European Journal of Operational Research, 2015. (Credit-scoring model benchmarking, directly relevant to Related Work 5.4.1.)
- Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." JMLR, 2011. (scikit-learn citation.)
- Basel Committee on Banking Supervision. Credit risk principles publication (cite the actual document bundled at `data/loan_strategy_docs/basel_credit_risk_principles_2025.pdf`, use its real title/date from the PDF itself).
- Consumer Financial Protection Bureau. Understanding your credit score consumer guide (cite the actual document at `data/loan_strategy_docs/cfpb_understand_your_credit_score.pdf`).
- National Credit Regulator (South Africa). Consumer guide to the National Credit Act, and consumer guide to managing debt (cite the two actual `ncr_*.pdf` documents by their real titles).
- World Bank. Responsible digital credit report (cite the actual document at `data/loan_strategy_docs/worldbank_responsible_digital_credit.pdf`).

Before finalising the reference list, open each PDF in `data/loan_strategy_docs/` and extract its real title, author/publisher, and publication date for an accurate citation rather than guessing from the filename.

---

## 7. Visual Design System for Figures and Diagrams

All figures (architecture diagrams, data-flow diagrams, bar charts, line
charts, tables rendered as images) must share one consistent, warm,
neutral, minimalist visual language. This applies to figures built with
matplotlib/seaborn (for the two experiment sections) and to any
architecture/flow diagrams built as vector graphics (TikZ within the LNCS
document is preferred for architecture diagrams, since it stays crisp at
any size and matches LaTeX's font automatically; plotted charts from the
CSV data should be regenerated in matplotlib and exported as vector PDF,
not raster PNG, for print quality).

### 7.1 Colour palette (fixed, do not deviate, no blue anywhere)

| Role | Colour name | Hex |
|---|---|---|
| Primary ink / text / strongest series | Warm charcoal | `#2B2825` |
| Secondary ink / gridlines / muted series | Warm grey | `#6B6259` |
| Background (if not pure white) | Warm ivory | `#FAF7F2` |
| Primary accent (the "best" series, headline bars) | Sepia tan | `#A9784C` |
| Secondary accent | Muted bronze | `#8B7355` |
| Tertiary accent (fourth series when needed) | Warm taupe | `#B5A48C` |
| Light structural lines / axis ticks | Pale warm grey | `#D8D2C4` |
| Negative/caution emphasis (use sparingly, not a saturated red) | Muted rust | `#9C5B3E` |

Rules:

- Never use blue, cyan, or any default matplotlib/seaborn colour cycle
  colour without overriding it first.
- Do not use hue alone to encode meaning (e.g. "green means good, red
  means bad"); use value/shade and position/labelling instead, both for
  the warm-neutral aesthetic and for print/greyscale legibility, since
  LNCS papers are frequently printed or read in black and white.
- Maximum four colours in any single figure. If a chart needs a fifth
  category, split it into two figures instead.
- White or warm-ivory background, never a dark/black chart background.
- Minimal chart junk: no drop shadows, no 3D bar effects, no gradient
  fills, no heavy gridlines. Light, thin gridlines only where they aid
  reading a value (e.g. horizontal gridlines behind a bar chart), always
  in the pale warm grey.
- Typography inside figures should match the body font (serif, matching
  LNCS's default Times-like font) at a size legible when the figure is
  scaled to column width, roughly 8-10pt effective.
- Every figure needs a caption stating what it shows and, where relevant,
  the sample size (e.g. "n = 20 queries").

### 7.2 Required figures (minimum set)

1. **System architecture diagram** (Section 5.5): the three pipelines and
   shared serving layer, boxes and arrows only, warm charcoal outlines on
   ivory background, sepia accent fill reserved for the three "pipeline"
   boxes to visually group them.
2. **Data flow diagram** (Section 5.5): linear pipeline from raw CSV
   through to the advisory report, mirroring the ASCII diagram in Section
   5.5 above but as a clean vector figure.
3. **Classifier comparison bar chart** (Section 5.7.1): grouped bars, one
   group per model, bars for at minimum ROC-AUC, F1-weighted, and MCC
   (three metrics keeps it under the 4-colour rule). Sort bars by the
   value being shown, not alphabetically.
4. **Composite rank vs single-metric (ROC-AUC) comparison** (Section
   5.7.1): a small, clear visual (e.g. a slope chart or simple two-column
   ranked list) making the Random-Forest-vs-XGBoost disagreement legible
   at a glance.
5. **Retrieval latency comparison** (Section 5.7.2): TF-IDF vs Chroma,
   either a paired bar chart or a box/strip plot over the 20 queries,
   log-scaled y-axis if needed given the ~12x gap.
6. **Retrieval score/quality comparison** (Section 5.7.2): TF-IDF vs
   Chroma top-1 and mean similarity scores, same chart style as #5 for
   visual consistency.
7. **Jaccard overlap / rank correlation summary** (Section 5.7.2): a
   single clean figure (not a busy heatmap) summarising the 0.366 mean
   Jaccard and 0.441 mean Spearman findings, e.g. a simple annotated
   strip/dot plot across the 20 queries.

Optional but encouraged if space allows within the page budget: a
confusion-matrix figure for the champion model, rendered as a 2x2 grid
using only the warm charcoal/sepia palette (no default red/green heatmap).

---

## 8. Page Budget (target 22-25 body pages, hard ceiling 27 total)

| Section | Target pages |
|---|---|
| Title, authors, abstract, keywords | 0.5 |
| 1. Introduction | 2 |
| 2. Related Work | 2.5-3 |
| 3. System Architecture (incl. 2 figures) | 2.5 |
| 4. Methodology (dataset, features, classifiers, metrics, RAG pipeline) | 5-6 |
| 5. Experiments and Results (incl. 5-6 figures) | 4-5 |
| 6. System Optimisations and Engineering Trade-offs | 1.5-2 |
| 7. Discussion and Limitations | 1.5 |
| 8. Conclusion | 0.5-1 |
| References | 1.5-2 |
| **Total** | **~22-26, ceiling 27** |

If the draft runs over, cut in this order: reduce the number of
hyperparameter-search-space tables in 5.6.3 to one representative example
instead of two, tighten Related Work prose (keep the citations, shorten
the summaries), and combine figures 5 and 6 from Section 7.2 into a single
two-panel figure. Do not cut the two "honesty" sections (5.6.3's tuning
caveat and 5.7.3's un-executed-experiment framing); those are load-bearing
for the paper's credibility.

---

## 9. Deliverable Packaging

Produce the paper as a self-contained LaTeX project so it compiles to a
PDF directly:

```
paper/
  main.tex              # \documentclass[runningheads]{llncs}, all sections
  llncs.cls             # Springer LNCS class file (obtain from Springer's
                         # official template package, do not hand-write it)
  splncs04.bst          # Springer's numbered bibliography style
  references.bib         # BibTeX entries for every citation in Section 5.11
  figures/               # all regenerated vector figures (PDF preferred),
                          # named by section, e.g. fig_architecture.pdf,
                          # fig_classifier_comparison.pdf,
                          # fig_retrieval_latency.pdf, etc.
```

`main.tex` should use standard LNCS sectioning
(`\section`, `\subsection`), `\begin{figure}...\end{figure}` with
`\caption` and `\label` for every figure, and `\begin{table}...\end{table}`
for the two data tables in Section 5.7 (do not embed tables as images).
Compile with `pdflatex` -> `bibtex` -> `pdflatex` -> `pdflatex` and verify
the final PDF page count against the 27-page ceiling before considering
the paper complete. If `llncs.cls`/`splncs04.bst` are not available in the
environment, state that clearly rather than silently substituting a
different document class, since a different class will not produce a
true LNCS-formatted PDF.
