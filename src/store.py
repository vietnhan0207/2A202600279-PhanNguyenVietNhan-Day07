from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            self._collection = client.create_collection(collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _chroma_metadata(self, doc: Document) -> dict:
        """Build ChromaDB metadata: always non-empty, includes original doc id."""
        return {**doc.metadata, "__original_id": doc.id}

    def _denormalize_metadata(self, metadata: dict) -> dict:
        """Strip internal keys added for ChromaDB compatibility."""
        return {k: v for k, v in metadata.items() if not k.startswith("__")}

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata if doc.metadata else {},
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
        query_vec = self._embedding_fn(query)
        scored = []
        for rec in records:
            score = _dot(query_vec, rec["embedding"])
            scored.append({**rec, "score": score})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                # Use a unique internal ID to avoid ChromaDB deduplication on repeated doc.id
                internal_id = f"{doc.id}__{self._next_index}"
                self._next_index += 1
                ids.append(internal_id)
                documents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append(self._chroma_metadata(doc))
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_vec = self._embedding_fn(query)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            res = self._collection.query(query_embeddings=[query_vec], n_results=n_results)
            results = []
            for i, doc_content in enumerate(res["documents"][0]):
                raw_meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                results.append({
                    "content": doc_content,
                    "score": 1 - res["distances"][0][i] if res.get("distances") else 0.0,
                    "metadata": self._denormalize_metadata(raw_meta),
                })
            return results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_vec = self._embedding_fn(query)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            res = self._collection.query(
                query_embeddings=[query_vec],
                n_results=n_results,
                where=metadata_filter,
            )
            results = []
            for i, doc_content in enumerate(res["documents"][0]):
                raw_meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                results.append({
                    "content": doc_content,
                    "score": 1 - res["distances"][0][i] if res.get("distances") else 0.0,
                    "metadata": self._denormalize_metadata(raw_meta),
                })
            return results
        else:
            # Filter in-memory records
            filtered = [
                rec for rec in self._store
                if all(rec.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                existing = self._collection.get(where={"__original_id": doc_id})
                if existing and existing.get("ids") and len(existing["ids"]) > 0:
                    self._collection.delete(ids=existing["ids"])
                    return True
                return False
            except Exception:
                return False
        else:
            original_len = len(self._store)
            self._store = [rec for rec in self._store if rec.get("metadata", {}).get("doc_id") != doc_id and rec.get("id") != doc_id]
            return len(self._store) < original_len
