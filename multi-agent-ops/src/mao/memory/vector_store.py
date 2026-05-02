"""ChromaDB-based vector memory store for semantic search."""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """ChromaDB-backed vector store for long-term semantic memory."""

    def __init__(self, persist_path: str = "./data/vectors/", collection_name: str = "mao_memory"):
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> str:
        """Add a document to the vector store."""
        doc_id = doc_id or uuid.uuid4().hex
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search for similar documents."""
        results = self._collection.query(query_texts=[query], n_results=top_k)
        items: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else 0,
                })
        return items

    def delete(self, doc_id: str) -> None:
        """Remove a document from the store."""
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Return total document count."""
        return self._collection.count()
