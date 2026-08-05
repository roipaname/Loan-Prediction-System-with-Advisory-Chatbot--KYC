"""
backend/deps.py
================
Process-wide singletons (classifier, context builder, retrieval stores) —
built once per uvicorn worker and reused across requests.

Import order matters on macOS Intel: torch (pulled in by VectorStore) must
not be imported before the xgboost classifier is joblib.load()'d, or it
segfaults. Vector store import stays lazy, and get_classifier() has to run
first — backend/main.py does that at startup.
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import DATA_DIR, TF_IDF_DIR
from src.ai_advisor.loan_context_builder import LoanContextBuilder
from src.classifier.classifier import LoanClassifier
from src.tf_idf.tf_idf_store import TFIDFStore

_STRATEGY_DOCS_DIR = DATA_DIR / "loan_strategy_docs"


@lru_cache(maxsize=1)
def get_context_builder() -> LoanContextBuilder:
    return LoanContextBuilder()


@lru_cache(maxsize=1)
def get_classifier() -> LoanClassifier:
    return get_context_builder().clf


@lru_cache(maxsize=1)
def get_tfidf_store() -> TFIDFStore:
    # reuse persisted index if present; rm -rf TF_IDF_DIR to force a reindex
    try:
        return TFIDFStore.load(TF_IDF_DIR)
    except FileNotFoundError:
        store = TFIDFStore.from_directory(_STRATEGY_DOCS_DIR)
        store.persist(TF_IDF_DIR)
        return store


@lru_cache(maxsize=1)
def get_vector_store():
    # lazy import (pulls in torch) — must run after get_classifier(), see above
    from src.ai_advisor.vector_store import VectorStore
    return VectorStore.from_directory(_STRATEGY_DOCS_DIR, force_reindex=False)


def get_retriever(name: str):
    if name == "vector":
        return get_vector_store()
    return get_tfidf_store()
