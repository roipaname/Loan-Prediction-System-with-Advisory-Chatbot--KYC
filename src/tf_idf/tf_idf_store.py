"""
Sparse TF-IDF vector store for loan strategy document retrieval, with a
ChromaDB-compatible interface (see VectorStore) so the two are interchangeable
in retrieval experiments. Documents are encoded via sklearn's TfidfVectorizer;
queries are matched by cosine similarity. Persists as joblib (vectorizer),
scipy NPZ (matrix), and JSON (documents/metadata/ids).
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
from config.settings import TF_IDF_DIR

class TFIDFStore:
    """TF-IDF backed vector store with a ChromaDB-compatible query interface.
    ngram_range=(1,2) captures bigrams like "debt consolidation" that unigrams
    alone would miss; sublinear_tf dampens very frequent terms."""

    # filenames only, joined onto whatever directory persist()/load() get —
    # must stay relative. TF_IDF_DIR/... here would make them absolute, and
    # Path.__truediv__ discards the left operand when the right one already
    # is, silently ignoring the caller's `directory` argument.
    _VECTORIZER_FILE = "tfidf_vectorizer.joblib"
    _MATRIX_FILE     = "tfidf_matrix.npz"
    _DATA_FILE       = "tfidf_data.json"

    _DEFAULT_PERSIST_DIR = TF_IDF_DIR

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

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> "TFIDFStore":
        """Add documents and (re)fit the TF-IDF index. Rebuilds from the
        full corpus each call so IDF weights stay globally correct — for
        large corpora, prefer one call with everything."""
        if not documents:
            return self

        n = len(documents)
        metadatas = metadatas or [{} for _ in range(n)]
        ids       = ids       or [str(uuid.uuid4()) for _ in range(n)]

        if len(metadatas) != n or len(ids) != n:
            raise ValueError("documents, metadatas, and ids must have the same length")

        # skip duplicate IDs
        existing_ids = set(self._ids)
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if doc_id in existing_ids:
                log.debug("TFIDFStore: skipping duplicate id=%s", doc_id)
                continue
            self._documents.append(doc)
            self._metadatas.append(meta)
            self._ids.append(doc_id)
            existing_ids.add(doc_id)

        self._matrix = self._vectorizer.fit_transform(self._documents)
        self._fitted = True

        log.info(
            "TFIDFStore '%s': %d docs indexed  vocab=%d",
            self.name, len(self._documents), len(self._vectorizer.vocabulary_),
        )
        return self

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> Dict[str, List[List[Any]]]:
        """Top-n_results most relevant documents per query, in ChromaDB's
        collection.query() output format (distances = 1 - cosine similarity)."""
        if not self._fitted:
            raise RuntimeError(
                "TFIDFStore is empty. Call .add() before querying."
            )

        n_results = min(n_results, len(self._documents))
        q_matrix = self._vectorizer.transform(query_texts)

        sims = cosine_similarity(q_matrix, self._matrix)  # shape (n_queries, n_docs)

        all_docs, all_metas, all_dists, all_ids = [], [], [], []
        for row in sims:
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

    def persist(self, directory: Union[str, Path, None] = None) -> Path:
        """Save the fitted store to <directory> (created if absent; defaults
        to TF_IDF_DIR) as tfidf_vectorizer.joblib + tfidf_matrix.npz +
        tfidf_data.json. Returns the directory."""
        if not self._fitted:
            raise RuntimeError("Cannot persist an empty TFIDFStore. Call .add() first.")

        directory = Path(directory or self._DEFAULT_PERSIST_DIR)
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
    def load(cls, directory: Union[str, Path, None] = None) -> "TFIDFStore":
        """Load a store previously saved via .persist() (same directory,
        or TF_IDF_DIR if omitted)."""
        directory = Path(directory or cls._DEFAULT_PERSIST_DIR)

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

    @classmethod
    def from_directory(
        cls,
        docs_dir: Union[str, Path],
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        name: str = "tfidf_store",
        **kwargs: Any,
    ) -> "TFIDFStore":
        """Build a TFIDFStore from all text/PDF/DOCX files in a directory,
        chunked so long documents become multiple, more specific entries."""
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

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"TFIDFStore(name={self.name!r}, "
            f"docs={self.count()}, "
            f"vocab={self.vocab_size()}, "
            f"fitted={self._fitted})"
        )
