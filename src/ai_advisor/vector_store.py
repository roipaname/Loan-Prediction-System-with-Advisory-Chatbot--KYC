"""
src/ai_advisor/vector_store.py
==============================
Dense embedding vector store backed by ChromaDB and sentence-transformers.

Architecture follows the pattern that has been verified to work on macOS Intel
(x86_64) with Python 3.10:

  - SentenceTransformer encodes each text INDIVIDUALLY (not in batches).
    Batch encoding crashes on macOS x86_64 due to OMP/libdispatch thread
    contention inside the Rust tokenizer when batch_size > 1.
  - Explicit embeddings are passed to ChromaDB's add() and query() so that
    ChromaDB never calls its own ONNX-based embedding pipeline.

The query interface is identical to TFIDFStore so both stores are
drop-in replacements in scripts/tfidf_chroma.py.

Persistence
-----------
ChromaDB's PersistentClient writes its own SQLite + HNSW index under
  <persist_directory>/

Public API
----------
  VectorStore(collection_name, persist_directory, embedding_model)
    .add(documents, metadatas, ids)
    .query(query_texts, n_results)   -> ChromaDB-compatible dict
    .count()                         -> int
    VectorStore.from_directory(docs_dir, **kwargs)  -> VectorStore

Usage
-----
    from src.ai_advisor.vector_store import VectorStore

    vs = VectorStore.from_directory("data/loan_strategy_docs")
    r  = vs.query(["high debt loan rejection poor credit"], n_results=3)
    for doc, dist in zip(r["documents"][0], r["distances"][0]):
        print(f"({1-dist:.4f})  {doc[:120]}...")
"""
from __future__ import annotations

import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from loguru import logger as log
from sentence_transformers import SentenceTransformer

from config.settings import BASE_DIR

_DEFAULT_PERSIST_DIR    = BASE_DIR / "data" / "chroma_db"
_DEFAULT_COLLECTION     = "loan_strategy"
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    """
    Dense vector store using sentence-transformers + ChromaDB.

    Texts are encoded ONE AT A TIME to avoid the OMP/libdispatch thread
    contention that causes segfaults on macOS Intel when encoding large
    batches with sentence-transformers >= 3.x.

    Parameters
    ----------
    collection_name   : ChromaDB collection name
    persist_directory : directory for ChromaDB's SQLite + HNSW index
    embedding_model   : sentence-transformers model name
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        persist_directory: Union[str, Path, None] = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._collection_name  = collection_name
        self._embedding_model  = embedding_model
        persist_dir = Path(persist_directory or _DEFAULT_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        log.info("VectorStore: loading encoder '%s' …", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # get_collection raises ValueError if not found; create_collection raises
        # if it already exists — so we try get first, then create.
        try:
            self._collection = self._client.get_collection(name=collection_name)
            log.info(
                "VectorStore '%s': loaded existing collection (%d docs)",
                collection_name, self._collection.count(),
            )
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("VectorStore '%s': created new collection.", collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Encode a single text string to a normalised embedding list."""
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> "VectorStore":
        """
        Embed and upsert documents into the collection.

        Uses upsert so the call is idempotent: re-adding an ID overwrites it
        rather than raising an error.  Each text is encoded individually to
        avoid batch-threading crashes.

        Parameters
        ----------
        documents : raw text strings
        metadatas : per-document metadata dicts (optional)
        ids       : unique string IDs (auto-generated UUIDs if omitted)
        """
        if not documents:
            return self

        n         = len(documents)
        metadatas = metadatas or [{} for _ in range(n)]
        ids       = ids       or [str(_uuid.uuid4()) for _ in range(n)]

        existing_ids = set(self._collection.get(include=[])["ids"])
        new_docs, new_metas, new_ids = [], [], []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_metas.append(meta)
                new_ids.append(doc_id)

        if not new_docs:
            log.info(
                "VectorStore '%s': all %d docs already indexed — skipping.",
                self._collection_name, n,
            )
            return self

        log.info(
            "VectorStore '%s': encoding and adding %d docs …",
            self._collection_name, len(new_docs),
        )
        embeddings = [self._embed(doc) for doc in new_docs]

        self._collection.add(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_metas,
            embeddings=embeddings,
        )
        log.info(
            "VectorStore '%s': collection now contains %d docs",
            self._collection_name, self.count(),
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
        Retrieve the top-n_results most similar documents for each query.

        Parameters
        ----------
        query_texts : list of query strings (one result list per query)
        n_results   : documents to return per query

        Returns
        -------
        dict with keys "documents", "metadatas", "distances", "ids"
        matching the ChromaDB collection.query() output format.
        """
        if self.count() == 0:
            raise RuntimeError(
                "VectorStore is empty. Call .add() or .from_directory() first."
            )

        k = min(n_results, self.count())
        embeddings = [self._embed(q) for q in query_texts]

        return self._collection.query(
            query_embeddings=embeddings,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        docs_dir: Union[str, Path],
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        collection_name: str = _DEFAULT_COLLECTION,
        persist_directory: Union[str, Path, None] = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        force_reindex: bool = False,
    ) -> "VectorStore":
        """
        Build (or reuse) a VectorStore from all documents in a directory.

        On the first call documents are chunked, embedded, and stored in
        ChromaDB.  On subsequent calls the saved collection is loaded instantly.

        Parameters
        ----------
        docs_dir         : directory with .txt / .md / .pdf / .docx files
        chunk_size       : target chunk length in words
        chunk_overlap    : word overlap between consecutive chunks
        collection_name  : ChromaDB collection name
        persist_directory: root directory for ChromaDB index
        embedding_model  : sentence-transformers model name
        force_reindex    : drop and rebuild even if the collection exists
        """
        from src.ai_advisor.document_loader import load_documents

        store = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )

        if force_reindex and store.count() > 0:
            log.info("VectorStore: force_reindex=True — dropping collection.")
            store._client.delete_collection(name=collection_name)
            store._collection = store._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=store._embedder.encode,
            )

        if store.count() > 0:
            log.info(
                "VectorStore '%s': reusing %d indexed docs "
                "(pass force_reindex=True to rebuild).",
                collection_name, store.count(),
            )
            return store

        docs_dir = Path(docs_dir)
        chunks   = load_documents(docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError(f"No readable documents found in {docs_dir}")

        store.add(
            documents=[c["text"]     for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=      [c["id"]       for c in chunks],
        )
        return store

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection={self._collection_name!r}, "
            f"docs={self.count()}, "
            f"model={self._embedding_model!r})"
        )
