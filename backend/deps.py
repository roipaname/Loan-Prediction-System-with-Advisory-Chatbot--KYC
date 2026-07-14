"""
backend/deps.py
================
Process-wide singletons: the trained classifier, the RAG context builder, the
two retrieval stores, and the advisor. All are expensive to construct (model
load, embedding model load, index load), so each is built once per uvicorn
worker and reused across requests.

Import-order constraint (macOS Intel): loading the xgboost-backed
LoanClassifier via joblib AFTER `torch` has been imported into the process
segfaults (confirmed: `import torch` then `joblib.load(xgboost_model)` ->
SIGSEGV during deserialization; the reverse order is safe, and xgboost
inference keeps working fine even if torch is imported afterwards). `torch`
is a transitive import of `src.ai_advisor.vector_store.VectorStore`
(sentence-transformers), so that import is kept lazy here, and
backend/main.py warms up the classifier at startup — before any request can
possibly trigger the vector-store import — to guarantee the safe order.
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import DATA_DIR
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
    return TFIDFStore.from_directory(_STRATEGY_DOCS_DIR)


@lru_cache(maxsize=1)
def get_vector_store():
    # Lazy import: pulls in torch/sentence-transformers. Must never run
    # before get_classifier() has already loaded the xgboost model once —
    # see module docstring.
    from src.ai_advisor.vector_store import VectorStore
    return VectorStore.from_directory(_STRATEGY_DOCS_DIR, force_reindex=False)


def get_retriever(name: str):
    if name == "vector":
        return get_vector_store()
    return get_tfidf_store()
