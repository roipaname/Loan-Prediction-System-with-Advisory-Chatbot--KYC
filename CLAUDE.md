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

# Run the FastAPI backend (must be running for the frontend to show real data)
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000

# Run the Streamlit frontend (5-page app)
.venv/bin/python -m streamlit run frontend/Home.py

# Feature engineering: data/raw/loan_data.csv → data/processed/loan_features.csv
.venv/bin/python database/feature_eng.py

# Train all classifiers, evaluate, save best to models/best_model.joblib
# (also writes models/model_comparison.csv, the full ranked metrics table)
.venv/bin/python -m scripts.train_model
.venv/bin/python -m scripts.train_model --metric f1_weighted
.venv/bin/python -m scripts.train_model --tune --tune-n-iter 30

# Bulk-insert processed CSV into PostgreSQL
.venv/bin/python -m scripts.insert_processed
.venv/bin/python -m scripts.insert_processed --dry-run

# One-time: reset the DB and populate it with a ~2,000-row sample + real
# predictions (LIME attribution included) for the frontend/backend to serve
.venv/bin/python -m scripts.populate_db
.venv/bin/python -m scripts.populate_db --sample-size 500   # smaller/faster

# Benchmark TF-IDF vs Dense Embedding retrieval (outputs charts to reports/)
.venv/bin/python -m scripts.tfidf_chroma
.venv/bin/python -m scripts.tfidf_chroma --no-dense   # TF-IDF only, no model loading
```

---

## Architecture

### `config/settings.py`
Single source of truth for all paths, environment variables, and training defaults. Imported by every module. Sets up `loguru` file sinks at import time. Key exports: `BASE_DIR`, `DATA_DIR`, `MODELS_DIR`, `BEST_MODEL_PATH`, `HF_TOKEN`, `HF_MODEL`.

### `database/`
- `schemas.py` — SQLAlchemy ORM models. Seven tables: `loan_applicants`, `engineered_features`, `ml_models`, `model_predictions`, `retrieval_documents`, `rag_explanations`, `rag_explanation_chunks`, `evaluator_assessments`. The `EvaluatorAssessment` table holds human rubric scores (Experiment 3) for Cohen's Kappa computation. `LoanApplicant.display_code` is a short customer-facing id (`APP-XXXXXXXX` in the UI) — the real primary key is a UUID used only internally for joins/FKs. **Gotcha**: SQLAlchemy's `Enum()` column type stores the Python enum *member name* in Postgres, not `.value` (e.g. `PersonEducationEnum.master` → stored as `"master"`, not `"Master"`). The ORM translates this transparently on every read; raw SQL does not — see `operations.py`'s `_ENUM_NAME_TO_VALUE` below.
- `feature_eng.py` — standalone ETL script; reads raw CSV, derives all engineered columns (ratios, flags, buckets), writes processed CSV. Column names mirror `EngineeredFeatures` ORM schema 1-to-1. Two columns (`credit_risk_interaction`, `is_high_risk`) are computed from **dataset-wide** medians/quantiles, not per-row — single-row (re)computation must reuse precomputed reference stats instead (see `backend/services/reference_stats.py`). The raw dataset has no `loan_grade` column at all (feature_eng.py never produces one).
- `connection.py` — SQLAlchemy connection pool wrapper.
- `operations.py` — CRUD helpers (`save`, `get_by_id`, `create_applicant`, `get_applicant_by_code`, etc.) plus `get_applicants_flat()` — a single raw-SQL join (applicants + engineered_features + latest prediction) returned as a flat DataFrame for the frontend/backend; `_ENUM_NAME_TO_VALUE` remaps the raw enum-name strings back to their intended display values there.
- `insert_processed.py` lives in `scripts/`, not `database/` (its own docstring says otherwise) — chunk-based idempotent bulk insert into `loan_applicants` + `engineered_features`, with enum-coercion maps (`GENDER_MAP`, `INTENT_MAP`, etc.) reused by `scripts/populate_db.py`.

### `src/classifier/`
- `classifier.py` — `LoanClassifier` wraps four algorithms (random_forest, xgboost, naive_bayes, logistic_regression) behind a common API: `train()`, `predict()`, `predict_proba()`, `save()`, `load()`. Each uses a sklearn `Pipeline` with `StandardScaler` + the classifier; XGBoost is `CalibratedClassifierCV`-wrapped. Note: `config/settings.py`'s `AVAILABLE_CLASSIFIERS` list also names `svm`, `gradient_boosting`, `lightgbm`, `catboost` — these are aspirational/unimplemented; only the four above exist in code.
- `logistic_regression.py` — `CustomLogisticRegression`: from-scratch implementation (gradient descent + L2) used as a comparison baseline.
- `evaluate.py` — `evaluate_classifier()` returns a metrics dict (accuracy, F1, ROC-AUC, MCC, Brier score, avg precision); `compare_classifiers()` builds a composite rank table.

### `src/tf_idf/`
Custom sparse vector store. `TFIDFStore` uses sklearn `TfidfVectorizer` (bigrams, sublinear_tf, 20k vocab) and cosine similarity. Persistence: `joblib` (vectorizer) + `scipy.sparse.save_npz` (matrix) + JSON (documents/metadata/ids). Returns ChromaDB-format dicts from `.query()`.

### `src/ai_advisor/`
Four modules wired together:

1. **`document_loader.py`** — chunks `.txt`, `.md`, `.pdf` (pypdf), `.docx` (python-docx) files into overlapping word-level windows. Used by both vector stores.

2. **`vector_store.py`** — `VectorStore` backs ChromaDB's `PersistentClient` with sentence-transformers (`all-MiniLM-L6-v2`) for dense retrieval. **Critical**: texts are encoded ONE AT A TIME (`self._embed(text)` in a loop) — batch encoding causes SIGSEGV on macOS Intel due to OMP/libdispatch thread contention. `chromadb.create_collection()` must NOT receive an `embedding_function` argument; ChromaDB 1.x validates the signature and rejects plain Python functions. Explicit embeddings are always passed to `collection.add()` / `collection.query()`. Index persists at `data/chroma_db/`. Importing this module pulls in `torch` — see the macOS Intel constraint below about load order relative to the xgboost classifier.

3. **`loan_context_builder.py`** — `LoanContextBuilder.build()` accepts either a DB `applicant_id` (UUID) or a raw `feature_row` dict. Reconstructs the training-time feature vector using `pd.get_dummies(drop_first=False)` + `df.reindex(clf.feature_names_, fill_value=0)` to align columns. Loads `models/best_model.joblib`, runs inference, and returns a structured context dict with prediction, risk tier, and top-15 feature importances (global, not per-applicant — see `backend/services/lime_service.py` for per-applicant attribution).

4. **`advisor.py`** — `LoanAdvisor.advise(context, n_docs=5)` retrieves relevant policy chunks from either vector store, builds a Mistral instruction-format prompt (`<s>[INST]...[/INST]`), and calls `InferenceClient.text_generation()` at `temperature=0.4`. Falls back to `_build_fallback_report()` (rule-based Markdown) when LLM is unavailable. Output must not contain em dashes (enforced in the system prompt).

`src/ai_advisor/__init__.py` intentionally has no re-exports (no code imports from the package root — always `from src.ai_advisor.<module> import ...`). Adding eager re-exports there reintroduces the SIGSEGV/hang described below, since importing `vector_store` at package-init time forces `torch` to load before anything else in the package gets a chance to.

### `scripts/tfidf_chroma.py`
Benchmarks TF-IDF vs Dense Embedding retrieval over 20 domain-specific loan queries. Metrics: latency, throughput, top-1 score, Jaccard overlap, Spearman rho. Outputs 5 PNG charts + a metrics CSV to `reports/` (`tfidf_chroma_metrics.csv`, served by `backend`'s `/comparisons/retrieval`).

### `backend/`
FastAPI app (`main.py`) serving the frontend — the DB, trained model, and RAG pipeline are otherwise inert without it. `deps.py` holds process-wide singletons (classifier, context builder, TF-IDF/vector stores) built once and reused; `get_classifier()` is warmed up in a `startup` event specifically to enforce the load order described below. `routers/` has `applicants` (list/detail/submit), `advisory` (per-applicant LIME attribution + on-demand advisory generation), `comparisons` (serves `models/model_comparison.csv` and `reports/tfidf_chroma_metrics.csv` as JSON). `services/` has `feature_engineering.py` (single-row mirror of `database/feature_eng.py`, using `reference_stats.py`'s precomputed medians/quantiles instead of live ones), `scoring_service.py` (orchestrates a new application end to end), and `lime_service.py` (per-applicant local feature attribution via `lime.lime_tabular` — this project uses LIME instead of SHAP throughout; there is no real SHAP anywhere in `src/classifier/`).

### `frontend/`
Streamlit multipage app. Entry point: `frontend/Home.py`. Pages under `frontend/pages/`:
- `1_Customer_Application.py` — submit new loan application (real backend scoring)
- `2_Customers.py` — browse all applicants
- `3_Dashboard.py` — aggregate KPIs and charts
- `4_AI_Advisory.py` — per-applicant LIME feature attribution + AI advisory report + PDF export
- `5_Findings.py` — TF-IDF vs Dense retrieval comparison and classifier comparison, both sourced from `backend`'s `/comparisons/*` endpoints

`frontend/utils/api_client.py` is the only thing frontend pages should call for data — `get_applicants_safe()` returns `(df, used_mock_fallback)` and falls back to `frontend/utils/mock_data.get_data()` if the backend/DB is unreachable, so the app still runs standalone for a demo. `frontend/styles/theme.py` defines the gold/dark brand palette and injects global CSS.

---

## macOS Intel (x86_64) Constraints

These are hard constraints on this machine that cannot be changed by editing code:

| Constraint | Cause | Effect |
|---|---|---|
| `SentenceTransformer.encode()` must loop one text at a time | OMP/libdispatch thread contention in Rust tokenizer | SIGSEGV on batch encode |
| `chromadb.create_collection()` — no `embedding_function=` arg | ChromaDB 1.x validates callable signature strictly | `ValueError` at collection creation |
| `torch<2.5`, `transformers<5`, `sentence-transformers<4` | No macOS x86_64 wheels exist for newer versions | install failure |
| Use `.venv/bin/python`, not `uv run` | `uv run` re-resolves and may fail | command error |
| `torch` must never be imported into a process *before* the xgboost-backed `LoanClassifier` is deserialized via `joblib.load()` (the reverse order — xgboost first, torch after — is safe, confirmed by direct testing) | Competing OpenMP runtimes | SIGSEGV during `joblib.load()` deserialization. This is why `backend/deps.py` keeps the `VectorStore` import lazy (inside `get_vector_store()`) and `backend/main.py` warms up the classifier in a `startup` event before any request can trigger it |
| Set `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` (done in `config/settings.py`, before any other import) whenever xgboost and sentence-transformers/torch are both loaded in one process | Same OpenMP contention as above, but manifests as an indefinite hang rather than a crash during `SentenceTransformer.encode()` | Without it, the first dense-retrieval query in a process that already loaded the classifier hangs forever instead of returning |

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
                                          │
                                          ▼
                                  backend/ (FastAPI) ──► frontend/ (Streamlit)
```

`scripts/populate_db.py` is the one-time bridge from the CSV pipeline above to a queryable Postgres database: it samples `data/processed/loan_features.csv`, bulk-inserts applicants + engineered features, and runs every sampled row through `LoanContextBuilder` + LIME to populate real `model_predictions` rows — this is what `backend/`'s `/applicants` endpoint actually serves.

## Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `HF_API_TOKEN` | HuggingFace Inference API token |
| `HF_MODEL` | Defaults to `mistralai/Mistral-7B-Instruct-v0.2` |
| `DEFAULT_CLASSIFIERS` | Default algorithm for training (default: `random_forest`) |
| `MODEL_SELECTION_METRIC` | Metric used to crown best model (default: `roc_auc`) |
| `API_BASE_URL` | Frontend → backend base URL (default: `http://localhost:8000`) |
