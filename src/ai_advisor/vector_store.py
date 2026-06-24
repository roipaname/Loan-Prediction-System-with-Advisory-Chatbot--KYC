"""
src/ai_advisor/vector_store.py
==============================
ChromaDB-backed vector store for loan strategy document retrieval.

Documents from data/loan_strategy_docs/ are embedded using a local
Sentence-Transformers model (all-MiniLM-L6-v2) and stored in a persistent
ChromaDB collection.  Subsequent runs reuse the persisted collection,
avoiding re-embedding on every startup.

The query interface mirrors the TFIDFStore in src/tf_idf/tf_idf_store.py so
that both stores are interchangeable in scripts/tfidf_chroma.py.

Public API
----------
  VectorStore(collection_name, persist_directory, embedding_model)
    .add(documents, metadatas, ids)
    .query(query_texts, n_results)  -> ChromaDB-compatible dict
    .count()                        -> int
    .reset()                        — drop and recreate the collection
    VectorStore.from_directory(docs_dir, **kwargs)  -> VectorStore

Usage
-----
    from src.ai_advisor import VectorStore

    vs = VectorStore.from_directory("data/loan_strategy_docs")
    results = vs.query(["high debt to income ratio loan rejection"], n_results=3)
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"({dist:.4f})  {doc[:120]} ...")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger as log

from config.settings import BASE_DIR

# Default ChromaDB persistence directory — lives alongside the models
_DEFAULT_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"

# Sentence-Transformers model used for embedding.  all-MiniLM-L6-v2 is fast,
# lightweight (80 MB), and scores well on semantic retrieval benchmarks.
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    """
    ChromaDB-backed vector store with a TFIDFStore-compatible interface.

    Parameters
    ----------
    collection_name   : ChromaDB collection name (acts as the namespace)
    persist_directory : directory where ChromaDB persists its data
    embedding_model   : Sentence-Transformers model name for encoding
    """

    def __init__(
        self,
        collection_name: str = "loan_strategy",
        persist_directory: Union[str, Path, None] = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise ImportError(
                "chromadb is required.  Install it with: pip install chromadb"
            ) from exc

        persist_directory = Path(persist_directory or _DEFAULT_PERSIST_DIR)
        persist_directory.mkdir(parents=True, exist_ok=True)

        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model

        self._client = chromadb.PersistentClient(path=str(persist_directory))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # get_or_create_collection is idempotent — reuses existing collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "VectorStore '%s': %d docs in collection  persist=%s  model=%s",
            collection_name, self._collection.count(), persist_directory, embedding_model,
        )

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
        Add documents to the ChromaDB collection.

        Existing documents with the same IDs are skipped, making this call
        idempotent.  This prevents re-embedding on every startup when the
        collection is already populated.

        Parameters
        ----------
        documents : list of raw text strings
        metadatas : list of metadata dicts (optional)
        ids       : list of unique string IDs (required — used for dedup check)

        Returns
        -------
        self
        """
        if not documents:
            return self

        n = len(documents)
        import uuid as _uuid
        metadatas = metadatas or [{} for _ in range(n)]
        ids       = ids       or [str(_uuid.uuid4()) for _ in range(n)]

        # Only add IDs that are not yet in the collection
        existing = set(self._collection.get(ids=ids)["ids"])
        new_mask = [(doc_id not in existing) for doc_id in ids]

        new_docs  = [d for d, keep in zip(documents,  new_mask) if keep]
        new_metas = [m for m, keep in zip(metadatas,  new_mask) if keep]
        new_ids   = [i for i, keep in zip(ids,        new_mask) if keep]

        if not new_docs:
            log.info("VectorStore '%s': all %d documents already present — skipping add.", self._collection_name, n)
            return self

        # ChromaDB upserts in batches to avoid exceeding request limits
        _BATCH = 100
        for start in range(0, len(new_docs), _BATCH):
            self._collection.add(
                documents=new_docs [start : start + _BATCH],
                metadatas=new_metas[start : start + _BATCH],
                ids=new_ids        [start : start + _BATCH],
            )

        log.info(
            "VectorStore '%s': added %d new docs (%d already existed)  total=%d",
            self._collection_name, len(new_docs), n - len(new_docs), self.count(),
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

        Returns the native ChromaDB response dict, which the TFIDFStore also
        mirrors, making the two stores interchangeable.

        Parameters
        ----------
        query_texts : list of query strings
        n_results   : number of results per query

        Returns
        -------
        dict with keys: "documents", "metadatas", "distances", "ids"
        """
        n_results = min(n_results, self.count())
        if n_results == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        result = self._collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self._collection.count()

    def reset(self) -> "VectorStore":
        """
        Drop the current collection and recreate it empty.

        Use this to force a full re-index of the document corpus.
        """
        self._client.delete_collection(self._collection_name)
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorStore '%s': collection reset.", self._collection_name)
        return self

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        docs_dir: Union[str, Path],
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        collection_name: str = "loan_strategy",
        persist_directory: Union[str, Path, None] = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        force_reindex: bool = False,
    ) -> "VectorStore":
        """
        Build (or reuse) a VectorStore from documents in a directory.

        On the first call the documents are chunked and embedded.  On
        subsequent calls the persisted ChromaDB collection is reused directly,
        so startup is fast even for large document corpora.

        Parameters
        ----------
        docs_dir         : directory containing strategy documents
        chunk_size       : target chunk length in words
        chunk_overlap    : word overlap between consecutive chunks
        collection_name  : ChromaDB collection name
        persist_directory: override for the ChromaDB persistence path
        embedding_model  : Sentence-Transformers model name
        force_reindex    : if True, drop and rebuild the collection

        Returns
        -------
        VectorStore instance with all documents indexed
        """
        from src.ai_advisor.document_loader import load_documents

        store = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )

        if force_reindex and store.count() > 0:
            log.info("VectorStore: force_reindex requested — dropping existing collection.")
            store.reset()

        if store.count() > 0:
            log.info(
                "VectorStore '%s': reusing %d existing documents.",
                collection_name, store.count(),
            )
            return store

        docs_dir = Path(docs_dir)
        log.info("VectorStore.from_directory: loading from %s", docs_dir)

        chunks = load_documents(docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError(f"No documents found in {docs_dir}")

        texts = [c["text"]     for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids   = [c["id"]       for c in chunks]

        store.add(texts, metadatas=metas, ids=ids)
        return store

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection={self._collection_name!r}, "
            f"docs={self.count()}, "
            f"model={self._embedding_model!r}, "
            f"persist={self._persist_directory})"
        )
