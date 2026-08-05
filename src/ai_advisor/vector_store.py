"""
Dense embedding vector store backed by ChromaDB and sentence-transformers.
Same query interface as TFIDFStore so the two are drop-in replacements for
each other in scripts/tfidf_chroma.py.

Texts are encoded ONE AT A TIME, not batched — batch encoding crashes on
macOS Intel due to OMP/libdispatch thread contention in the tokenizer.
Embeddings are passed explicitly to add()/query() so Chroma never invokes
its own ONNX embedding pipeline.
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
    """Dense vector store using sentence-transformers + ChromaDB (see module
    docstring for the one-at-a-time encoding constraint)."""

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

    def _embed(self, text: str) -> List[float]:
        """Encode a single text string to a normalised embedding list."""
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> "VectorStore":
        """Embed and add documents to the collection. Skips IDs already
        present, so calling this again with the same docs is a no-op."""
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

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> Dict[str, List[List[Any]]]:
        """Top-n_results most similar documents per query, in ChromaDB's
        collection.query() output format."""
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

    def count(self) -> int:
        return self._collection.count()

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
        """Build (or reuse) a VectorStore from all documents in a directory.
        First call chunks/embeds/stores; later calls just load the saved
        collection unless force_reindex=True."""
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

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection={self._collection_name!r}, "
            f"docs={self.count()}, "
            f"model={self._embedding_model!r})"
        )
