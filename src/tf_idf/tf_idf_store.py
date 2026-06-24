"""
src/tf_idf/tf_idf_store.py
==========================
Custom TF-IDF vector database for loan strategy document retrieval.

This module provides a lightweight, self-contained vector store that uses
TF-IDF (Term Frequency-Inverse Document Frequency) as the embedding strategy
instead of dense neural embeddings.  The interface mirrors ChromaDB's
collection API so the two stores are interchangeable in retrieval experiments.

Architecture
------------
Documents are encoded as sparse TF-IDF vectors using sklearn's
TfidfVectorizer.  Queries are transformed into the same vector space at
search time, and the top-k most similar documents are retrieved via cosine
similarity.  The full index (vectorizer + matrix + documents) is persisted to
disk using joblib (vectorizer), scipy NPZ (matrix), and JSON (metadata).

Public API
----------
  TFIDFStore(name)
    .add(documents, metadatas, ids)
    .query(query_texts, n_results)  -> ChromaDB-compatible dict
    .count()                        -> int
    .persist(directory)
    TFIDFStore.load(directory)      -> TFIDFStore
    TFIDFStore.from_directory(docs_dir, **kwargs)  -> TFIDFStore

Usage
-----
    from src.tf_idf import TFIDFStore

    store = TFIDFStore.from_directory("data/loan_strategy_docs")
    results = store.query(["high debt to income ratio"], n_results=3)
    for doc, score in zip(results["documents"][0], results["distances"][0]):
        print(f"({score:.4f})  {doc[:120]}...")
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import scipy.sparse as sp
from loguru import logger as log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFStore:
    """
    TF-IDF backed vector store with a ChromaDB-compatible query interface.

    Parameters
    ----------
    name : str
        Logical name for the collection (used in persistence filenames).
    max_features : int
        Maximum vocabulary size.  Larger values improve recall at the cost
        of memory and search latency.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.  (1, 2) captures both unigrams
        and bigrams, which improves matching for multi-word financial terms
        (e.g. "debt consolidation", "credit score").
    sublinear_tf : bool
        Apply log(1 + tf) instead of raw tf.  Reduces the dominance of very
        frequent terms and generally improves retrieval quality.
    """

    _VECTORIZER_FILE = "tfidf_vectorizer.joblib"
    _MATRIX_FILE     = "tfidf_matrix.npz"
    _DATA_FILE       = "tfidf_data.json"

    def __init__(
        self,
        name: str = "tfidf_store",
        max_features: int = 20_000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self.name = name
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )
        self._matrix: Optional[sp.csr_matrix] = None
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> "TFIDFStore":
        """
        Add documents to the store and (re)fit the TF-IDF index.

        Calling add() multiple times is supported: the index is rebuilt from
        the full corpus each time so that IDF weights remain globally correct.
        For large corpora, add all documents in a single call where possible.

        Parameters
        ----------
        documents : list of raw text strings
        metadatas : list of dicts with any metadata per document (optional)
        ids       : list of unique string IDs (auto-generated if omitted)

        Returns
        -------
        self
        """
        if not documents:
            return self

        n = len(documents)
        metadatas = metadatas or [{} for _ in range(n)]
        ids       = ids       or [str(uuid.uuid4()) for _ in range(n)]

        if len(metadatas) != n or len(ids) != n:
            raise ValueError("documents, metadatas, and ids must have the same length")

        # Extend corpus; skip documents with duplicate IDs
        existing_ids = set(self._ids)
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if doc_id in existing_ids:
                log.debug("TFIDFStore: skipping duplicate id=%s", doc_id)
                continue
            self._documents.append(doc)
            self._metadatas.append(meta)
            self._ids.append(doc_id)
            existing_ids.add(doc_id)

        # Refit on the full corpus so IDF weights account for all documents
        self._matrix = self._vectorizer.fit_transform(self._documents)
        self._fitted = True

        log.info(
            "TFIDFStore '%s': %d docs indexed  vocab=%d",
            self.name, len(self._documents), len(self._vectorizer.vocabulary_),
        )
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> Dict[str, List[List[Any]]]:
        """
        Retrieve the top-n_results most relevant documents for each query.

        The return format mirrors ChromaDB's collection.query() output so the
        two stores are drop-in replacements in retrieval experiments.

        Parameters
        ----------
        query_texts : list of query strings (one result set per query)
        n_results   : number of documents to return per query

        Returns
        -------
        dict with keys:
            "documents" : list[list[str]]   — retrieved text per query
            "metadatas" : list[list[dict]]  — metadata per retrieved doc
            "distances" : list[list[float]] — 1 - cosine_sim (lower = better)
            "ids"       : list[list[str]]   — document IDs
        """
        if not self._fitted:
            raise RuntimeError(
                "TFIDFStore is empty. Call .add() before querying."
            )

        n_results = min(n_results, len(self._documents))
        q_matrix = self._vectorizer.transform(query_texts)

        # cosine_similarity returns shape (n_queries, n_docs)
        sims = cosine_similarity(q_matrix, self._matrix)

        all_docs, all_metas, all_dists, all_ids = [], [], [], []
        for row in sims:
            # argsort descending — top-k most similar
            top_k = int(np.argsort(row)[::-1][:n_results])
            top_indices = np.argsort(row)[::-1][:n_results]
            top_scores  = row[top_indices]

            all_docs.append([self._documents[i] for i in top_indices])
            all_metas.append([self._metadatas[i] for i in top_indices])
            all_dists.append([float(1.0 - s) for s in top_scores])  # distance = 1 - similarity
            all_ids.append([self._ids[i]      for i in top_indices])

        return {
            "documents": all_docs,
            "metadatas": all_metas,
            "distances": all_dists,
            "ids":       all_ids,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently in the store."""
        return len(self._documents)

    def vocab_size(self) -> int:
        """Return the number of unique terms in the fitted vocabulary."""
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)

    def get_feature_names(self) -> List[str]:
        """Return the list of vocabulary terms in feature-index order."""
        if not self._fitted:
            return []
        return self._vectorizer.get_feature_names_out().tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, directory: Union[str, Path]) -> Path:
        """
        Save the fitted store to disk.

        Three files are written:
          <directory>/tfidf_vectorizer.joblib  — fitted TfidfVectorizer
          <directory>/tfidf_matrix.npz         — sparse TF-IDF matrix
          <directory>/tfidf_data.json          — documents, metadatas, ids

        Parameters
        ----------
        directory : path to the output directory (created if absent)

        Returns
        -------
        Path to the directory containing the saved files.
        """
        if not self._fitted:
            raise RuntimeError("Cannot persist an empty TFIDFStore. Call .add() first.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._vectorizer, directory / self._VECTORIZER_FILE)
        sp.save_npz(str(directory / self._MATRIX_FILE), self._matrix)

        data = {
            "name":      self.name,
            "documents": self._documents,
            "metadatas": self._metadatas,
            "ids":       self._ids,
        }
        (directory / self._DATA_FILE).write_text(json.dumps(data, indent=2, default=str))

        log.info("TFIDFStore persisted to %s (%d docs)", directory, len(self._documents))
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "TFIDFStore":
        """
        Load a previously persisted TFIDFStore from disk.

        Parameters
        ----------
        directory : path that was passed to .persist()

        Returns
        -------
        Fully initialised TFIDFStore ready for querying.
        """
        directory = Path(directory)

        vectorizer_path = directory / cls._VECTORIZER_FILE
        matrix_path     = directory / cls._MATRIX_FILE
        data_path       = directory / cls._DATA_FILE

        for p in (vectorizer_path, matrix_path, data_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"TFIDFStore persistence file not found: {p}\n"
                    f"Make sure you persisted the store to '{directory}' first."
                )

        data = json.loads(data_path.read_text())
        store = cls(name=data.get("name", "tfidf_store"))
        store._vectorizer  = joblib.load(vectorizer_path)
        store._matrix      = sp.load_npz(str(matrix_path))
        store._documents   = data["documents"]
        store._metadatas   = data["metadatas"]
        store._ids         = data["ids"]
        store._fitted      = True

        log.info("TFIDFStore loaded from %s (%d docs)", directory, store.count())
        return store

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        docs_dir: Union[str, Path],
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        name: str = "tfidf_store",
        **kwargs: Any,
    ) -> "TFIDFStore":
        """
        Build a TFIDFStore from all text/PDF/DOCX files in a directory.

        Files are split into overlapping chunks before indexing so that
        long documents are represented by multiple, more specific entries.

        Parameters
        ----------
        docs_dir      : directory containing loan strategy documents
        chunk_size    : target chunk length in words
        chunk_overlap : words of overlap between consecutive chunks
        name          : logical store name
        **kwargs      : forwarded to TFIDFStore.__init__

        Returns
        -------
        Fitted TFIDFStore
        """
        from src.ai_advisor.document_loader import load_documents

        docs_dir = Path(docs_dir)
        log.info("TFIDFStore.from_directory: scanning %s", docs_dir)

        chunks = load_documents(docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError(f"No documents found in {docs_dir}")

        texts    = [c["text"]     for c in chunks]
        metas    = [c["metadata"] for c in chunks]
        ids      = [c["id"]       for c in chunks]

        store = cls(name=name, **kwargs)
        store.add(texts, metadatas=metas, ids=ids)
        return store

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"TFIDFStore(name={self.name!r}, "
            f"docs={self.count()}, "
            f"vocab={self.vocab_size()}, "
            f"fitted={self._fitted})"
        )
