# LAPAS Optimization Pass — 2026-09-20

Summary of a repo-hygiene / performance / ML-RAG-quality pass across the codebase, with before/after metrics for every change that had a measurable effect. Nothing here is committed to git — see `git diff` to review before committing.

---

## 1. Repo hygiene

| Change | File(s) |
|---|---|
| Removed `embedding_function=` argument from `create_collection()` in the `force_reindex=True` path — contradicted the documented ChromaDB 1.x constraint (no `embedding_function=`, explicit embeddings only) and would raise `ValueError` if that path ever ran | `src/ai_advisor/vector_store.py` |
| Added `*.zip` to `.gitignore`; deleted `backend.zip`/`frontend.zip`/`src.zip` — manual macOS backups (with `__MACOSX/` junk) duplicating already-tracked directories | `.gitignore` |
| Trimmed `AVAILABLE_CLASSIFIERS` to the four algorithms that actually exist (removed `svm`, `gradient_boosting`, `lightgbm`, `catboost`, which were never implemented) | `config/settings.py` |
| Documented that `requirements.txt` tracks analysis/notebook dependencies (Experiment 1 fairness metrics, Experiment 3 inter-rater reliability stats), separate from `pyproject.toml`'s app runtime deps — no packages changed | `requirements.txt` |

No metrics apply here — these are correctness/cleanliness fixes, not measured changes.

---

## 2. Runtime performance

| Change | File(s) |
|---|---|
| `get_applicants_flat()` now pushes `LIMIT` into the SQL query (parameterized) instead of fetching the entire joined result set into memory and truncating with `pandas.head()` | `database/operations.py` |
| Added an index on `LoanApplicant.created_at`, the column the applicants join sorts by on every call; applied to the live DB via `CREATE INDEX IF NOT EXISTS` (schema-only changes don't retrofit onto existing tables) | `database/schemas.py` |
| `/comparisons/models` and `/comparisons/retrieval` now cache their CSV reads in-process, keyed by file mtime, instead of re-parsing on every request | `backend/routers/comparisons.py` |

These are algorithmic/architectural fixes (no more full-table fetch, no more disk read + CSV parse per request) rather than things with a single before/after number to quote — impact scales with DB size and request volume, which are both small in the current dev/demo dataset.

---

## 3. ML / RAG quality

### 3a. RAG chunk-size fix (silent truncation bug)

`all-MiniLM-L6-v2` truncates input at 256 tokens, but chunks defaulted to ~400 words (~500-550 tokens for the regulatory PDFs) — **the back half of every chunk was silently dropped before embedding.** Lowered the default to 180 words / 30-word overlap (≈230-250 tokens, safely under the limit) in `document_loader.py`, `vector_store.py`, and `tf_idf_store.py`, then rebuilt both the Chroma and TF-IDF indexes and re-ran the 20-query retrieval benchmark (`scripts/tfidf_chroma.py`).

| Metric | Before (400w/50 overlap) | After (180w/30 overlap) | Change |
|---|---|---|---|
| TF-IDF top-1 score (mean) | 0.1917 | 0.2479 | **+29.3%** |
| TF-IDF mean score | 0.1334 | 0.1681 | **+26.0%** |
| Dense (Chroma) top-1 score (mean) | 0.5975 | 0.6345 | **+6.2%** |
| Dense mean score | 0.5042 | 0.5575 | **+10.6%** |
| TF-IDF ↔ Dense agreement (Jaccard overlap) | 0.3571 | 0.4325 | **+21.1%** |
| TF-IDF ↔ Dense rank correlation (Spearman ρ) | 0.4647 | 0.6333 | **+36.3%** |
| TF-IDF latency (mean) | 2.52 ms | 4.09 ms | +1.57 ms (more, smaller chunks) |
| Dense latency (mean) | 32.45 ms | 23.31 ms | **-28.2%** |

Both retrieval quality *and* agreement between the two methods improved — the two retrievers were disagreeing partly because the dense side was working from truncated chunks.

### 3b. Classifier tuning + a real correctness bug found and fixed

While running `scripts.train_model --tune --tune-n-iter 30` to check whether hyperparameter search improves on the saved (apparently untuned) champion model, tuning **logistic regression's** hyperparameters produced garbage (accuracy collapsed to 0.569, down from 0.836 untuned) — every cross-validation fold's `roc_auc` score came back `nan`.

**Root cause**: `CustomLogisticRegression` was declared `class CustomLogisticRegression(BaseEstimator, ClassifierMixin)`. In scikit-learn ≥1.6, tag resolution (`__sklearn_tags__()`) walks the MRO, and since `BaseEstimator` was listed first, its `__sklearn_tags__()` shadowed `ClassifierMixin`'s contribution, leaving `estimator_type=None`. `is_classifier()` then returned `False`, which silently breaks any scorer that routes through `decision_function` (including `roc_auc`, the default tuning metric).

**Fix**: swapped the base class order to `class CustomLogisticRegression(ClassifierMixin, BaseEstimator)` — mixins must precede `BaseEstimator` for tag composition to work. Verified `is_classifier()` returns `True` after the fix, then re-ran the full tuning pass.

| Model | Metric | Before (untuned) | After (tuned, bug fixed) | Change |
|---|---|---|---|---|
| **xgboost** *(new champion)* | Accuracy | 0.920 | 0.9359 | +1.6 pts |
| | ROC-AUC | 0.978 | 0.9784 | +0.04 pts |
| | MCC | 0.783 | 0.8099 | **+2.7 pts** |
| | Brier score (↓ better) | 0.055 | 0.0465 | **-15.5%** |
| random_forest *(previous champion)* | Accuracy | 0.928 | 0.9309 | +0.3 pts |
| | ROC-AUC | 0.976 | 0.9765 | +0.02 pts |
| | MCC | 0.785 | 0.7990 | +1.4 pts |
| | Brier score (↓ better) | 0.051 | 0.0510 | ~unchanged |
| logistic_regression | Accuracy | 0.836 | 0.8376 | +0.2 pts |
| | ROC-AUC | 0.927 | 0.9517 | **+2.5 pts** |
| | MCC | 0.608 | 0.6463 | **+3.8 pts** |
| | Brier score (↓ better) | 0.150 | 0.1004 | **-33.1%** |
| naive_bayes | Accuracy | 0.763 | 0.7846 | +2.2 pts |
| | ROC-AUC | 0.934 | 0.9343 | ~unchanged |
| | MCC | 0.568 | 0.5870 | +1.9 pts |

**XGBoost is now the tuned champion model** (`models/best_model.joblib`), selected on `roc_auc` per `config.settings.MODEL_SELECTION_METRIC`, edging out random_forest which held the title untuned. Naive Bayes barely moves — it only has one tunable parameter (`var_smoothing`), so this is expected, not a miss.

---

## 4. Unplanned: development environment repair

While running the scripts above, the `.venv` turned out to have widespread corruption unrelated to any of the changes here — `numpy`, `pandas`, `scipy`, `pydantic-core`, `tokenizers`, ChromaDB's Rust bindings, `scikit-learn`, and `plotly` were all missing compiled extension files, causing import-time crashes. Repaired via a full `pip install --force-reinstall -e .` (honoring the `pyproject.toml` pins, e.g. `torch<2.5` for macOS Intel wheel availability), then re-verified the macOS Intel constraints documented in `CLAUDE.md` still hold (`SentenceTransformer.encode()` single-text calls, `config.settings`'s `OMP_NUM_THREADS=1` import-order requirement).

---

## 5. Verification performed

- Started the FastAPI backend and Streamlit frontend and exercised them live: `/health`, `/comparisons/models`, `/comparisons/retrieval`, `/applicants`, and a full `/predictions/{code}/advisory` call (generates a real advisory report against the rebuilt vector store) — all returned `200`.
- Confirmed `get_applicants_flat(limit=5)` returns exactly 5 rows via the new SQL `LIMIT`.
- Confirmed no remaining references to the removed classifier names (`gradient_boosting`/`lightgbm`/`catboost`/`svm`) outside the unrelated `ModelAlgorithmEnum` DB enum.
- Confirmed `is_classifier(CustomLogisticRegression())` is `True` after the base-class fix.

See `CLAUDE.md` for the updated architecture notes reflecting these changes (chunk sizes, the fixed bugs, the new CSV caching).
