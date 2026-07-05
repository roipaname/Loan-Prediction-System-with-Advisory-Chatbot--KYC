# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**LAPAS** — Loan Approval Prediction & Advisory System. An Honours project at the University of Johannesburg. Three interconnected ML pipelines: (1) multi-classifier loan approval prediction, (2) RAG-powered advisory report generation, (3) retrieval method benchmarking (TF-IDF vs dense embeddings).

**Runtime**: Python 3.10, macOS Intel x86_64. Package manager: `uv`. Always use `.venv/bin/python` directly — `uv run` re-resolves the lock file on every invocation and can fail.

---

## Commands

```bash
# Install / sync dependencies
uv pip install -e .

# Run the Streamlit frontend (4-page app)
.venv/bin/python -m streamlit run frontend/Home.py

# Feature engineering: data/raw/loan_data.csv → data/processed/loan_features.csv
.venv/bin/python database/feature_eng.py

# Train all classifiers, evaluate, save best to models/best_model.joblib
.venv/bin/python -m scripts.train_model
.venv/bin/python -m scripts.train_model --metric f1_weighted
.venv/bin/python -m scripts.train_model --tune --tune-n-iter 30

# Bulk-insert processed CSV into PostgreSQL
.venv/bin/python -m database.insert_processed
.venv/bin/python -m database.insert_processed --dry-run

# Benchmark TF-IDF vs Dense Embedding retrieval (outputs charts to reports/)
.venv/bin/python -m scripts.tfidf_chroma
.venv/bin/python -m scripts.tfidf_chroma --no-dense   # TF-IDF only, no model loading
```

---

## Architecture

### `config/settings.py`
Single source of truth for all paths, environment variables, and training defaults. Imported by every module. Sets up `loguru` file sinks at import time. Key exports: `BASE_DIR`, `DATA_DIR`, `MODELS_DIR`, `BEST_MODEL_PATH`, `HF_TOKEN`, `HF_MODEL`.

### `database/`
- `schemas.py` — SQLAlchemy ORM models. Seven tables: `loan_applicants`, `engineered_features`, `ml_models`, `model_predictions`, `retrieval_documents`, `rag_explanations`, `rag_explanation_chunks`, `evaluator_assessments`. The `EvaluatorAssessment` table holds human rubric scores (Experiment 3) for Cohen's Kappa computation.
- `feature_eng.py` — standalone ETL script; reads raw CSV, derives all engineered columns (ratios, flags, buckets), writes processed CSV. Column names mirror `EngineeredFeatures` ORM schema 1-to-1.
- `connection.py` — SQLAlchemy connection pool wrapper.
- `insert_processed.py` (in `scripts/`) — chunk-based idempotent bulk insert into `loan_applicants` + `engineered_features`.

### `src/classifier/`
- `classifier.py` — `LoanClassifier` wraps four algorithms (random_forest, xgboost, naive_bayes, logistic_regression) behind a common API: `train()`, `predict()`, `predict_proba()`, `save()`, `load()`. Each uses a sklearn `Pipeline` with `StandardScaler` + the classifier; XGBoost is `CalibratedClassifierCV`-wrapped.
- `logistic_regression.py` — `CustomLogisticRegression`: from-scratch implementation (gradient descent + L2) used as a comparison baseline.
- `evaluate.py` — `evaluate_classifier()` returns a metrics dict (accuracy, F1, ROC-AUC, MCC, Brier score, avg precision); `compare_classifiers()` builds a composite rank table.

### `src/tf_idf/`
Custom sparse vector store. `TFIDFStore` uses sklearn `TfidfVectorizer` (bigrams, sublinear_tf, 20k vocab) and cosine similarity. Persistence: `joblib` (vectorizer) + `scipy.sparse.save_npz` (matrix) + JSON (documents/metadata/ids). Returns ChromaDB-format dicts from `.query()`.

### `src/ai_advisor/`
Four modules wired together:

1. **`document_loader.py`** — chunks `.txt`, `.md`, `.pdf` (pypdf), `.docx` (python-docx) files into overlapping word-level windows. Used by both vector stores.

2. **`vector_store.py`** — `VectorStore` backs ChromaDB's `PersistentClient` with sentence-transformers (`all-MiniLM-L6-v2`) for dense retrieval. **Critical**: texts are encoded ONE AT A TIME (`self._embed(text)` in a loop) — batch encoding causes SIGSEGV on macOS Intel due to OMP/libdispatch thread contention. `chromadb.create_collection()` must NOT receive an `embedding_function` argument; ChromaDB 1.x validates the signature and rejects plain Python functions. Explicit embeddings are always passed to `collection.add()` / `collection.query()`. Index persists at `data/chroma_db/`.

3. **`loan_context_builder.py`** — `LoanContextBuilder.build()` accepts either a DB `applicant_id` (UUID) or a raw `feature_row` dict. Reconstructs the training-time feature vector using `pd.get_dummies(drop_first=False)` + `df.reindex(clf.feature_names_, fill_value=0)` to align columns. Loads `models/best_model.joblib`, runs inference, and returns a structured context dict with prediction, risk tier, and top-15 feature importances.

4. **`advisor.py`** — `LoanAdvisor.advise(context, n_docs=5)` retrieves relevant policy chunks from either vector store, builds a Mistral instruction-format prompt (`<s>[INST]...[/INST]`), and calls `InferenceClient.text_generation()` at `temperature=0.4`. Falls back to `_build_fallback_report()` (rule-based Markdown) when LLM is unavailable. Output must not contain em dashes (enforced in the system prompt).

### `scripts/tfidf_chroma.py`
Benchmarks TF-IDF vs Dense Embedding retrieval over 20 domain-specific loan queries. Metrics: latency, throughput, top-1 score, Jaccard overlap, Spearman rho. Outputs 5 PNG charts + a metrics CSV to `reports/`.

### `frontend/`
Streamlit multipage app. Entry point: `frontend/Home.py`. Pages under `frontend/pages/`:
- `1_Customer_Application.py` — submit new loan application
- `2_Customers.py` — browse all applicants
- `3_Dashboard.py` — aggregate KPIs and charts
- `4_AI_Advisory.py` — per-applicant SHAP waterfall + AI advisory report + PDF export

`frontend/utils/mock_data.py` provides synthetic data when the database is unavailable. `frontend/styles/theme.py` defines the gold/dark brand palette and injects global CSS.

---

## macOS Intel (x86_64) Constraints

These are hard constraints on this machine that cannot be changed by editing code:

| Constraint | Cause | Effect |
|---|---|---|
| `SentenceTransformer.encode()` must loop one text at a time | OMP/libdispatch thread contention in Rust tokenizer | SIGSEGV on batch encode |
| `chromadb.create_collection()` — no `embedding_function=` arg | ChromaDB 1.x validates callable signature strictly | `ValueError` at collection creation |
| `torch<2.5`, `transformers<5`, `sentence-transformers<4` | No macOS x86_64 wheels exist for newer versions | install failure |
| Use `.venv/bin/python`, not `uv run` | `uv run` re-resolves and may fail | command error |

---

## Data Flow

```
data/raw/loan_data.csv
    └─ database/feature_eng.py ──────────────────────────────► data/processed/loan_features.csv
                                                                         │
                                                                         ▼
                                                          scripts/train_model.py
                                                                         │
                                                                         ▼
                                                          models/best_model.joblib
                                                                         │
                                          ┌──────────────────────────────┘
                                          │
data/loan_strategy_docs/                  │
    └─ VectorStore.from_directory()       │
    └─ TFIDFStore.from_directory() ───────┤
                                          ▼
                                  LoanContextBuilder.build()
                                          │
                                          ▼
                                  LoanAdvisor.advise() ──► Markdown report
```

## Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `HF_API_TOKEN` | HuggingFace Inference API token |
| `HF_MODEL` | Defaults to `mistralai/Mistral-7B-Instruct-v0.2` |
| `DEFAULT_CLASSIFIERS` | Default algorithm for training (default: `random_forest`) |
| `MODEL_SELECTION_METRIC` | Metric used to crown best model (default: `roc_auc`) |
