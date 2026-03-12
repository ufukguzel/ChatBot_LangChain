"""
RAG (Retrieval-Augmented Generation) service.

Design decisions:
  - All mutable state is encapsulated inside RAGService instances (no module globals).
  - A threading.Lock protects _docs / _doc_index / _faiss_index from concurrent mutations.
  - FAISS is optional: if unavailable (or api_key missing), a simple keyword retriever is used.
  - Exceptions in FAISS operations are caught and logged; keyword fallback is always available.
"""
import logging
import threading
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

_SPLITTER_AVAILABLE = True
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:
    _SPLITTER_AVAILABLE = False

_FAISS_AVAILABLE = True
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InMemoryDoc:
    __slots__ = ("id", "text", "source")

    def __init__(self, doc_id: str, text: str, source: str) -> None:
        self.id = doc_id
        self.text = text
        self.source = source


class SimpleKeywordRetriever:
    """Fallback retriever using word-overlap scoring — no external dependencies."""

    def __init__(self, docs: List[InMemoryDoc]) -> None:
        self._docs = docs

    @staticmethod
    def _score(query: str, text: str) -> int:
        q_words = {w.lower() for w in query.split() if len(w) > 2}
        t_words = {w.lower() for w in text.split() if len(w) > 2}
        return len(q_words & t_words)

    def get_relevant(self, query: str, k: int = 3) -> List[InMemoryDoc]:
        ranked = sorted(self._docs, key=lambda d: self._score(query, d.text), reverse=True)
        return ranked[:k]


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class RAGService:
    """
    Thread-safe service for ingesting documents and retrieving relevant chunks.

    Usage:
        rag = RAGService(api_key="sk-...")
        n = rag.add_document("doc.pdf", text_content)
        context, sources = rag.build_context(query, use_rag=True)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._docs: List[InMemoryDoc] = []
        self._doc_index: Dict[str, InMemoryDoc] = {}
        self._faiss_index = None   # FAISS index (optional)
        self._emb = None           # OpenAIEmbeddings (optional)
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_embeddings(self) -> bool:
        """Lazily initialise OpenAI embeddings. Returns True on success."""
        if not _FAISS_AVAILABLE or not self._api_key:
            return False
        if self._emb is None:
            try:
                self._emb = OpenAIEmbeddings(api_key=self._api_key)
            except Exception as exc:
                logger.warning("Could not initialise OpenAIEmbeddings: %s", exc)
                return False
        return True

    def _split_text(self, text: str) -> List[str]:
        if _SPLITTER_AVAILABLE:
            try:
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                return [c.page_content for c in splitter.create_documents([text])]
            except Exception as exc:
                logger.warning("Text splitter failed (%s) — using naive chunking.", exc)
        # naive fallback
        chunks = [text[i : i + 700] for i in range(0, len(text), 700)]
        return chunks or [text]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, name: str, content: str) -> int:
        """
        Ingest document content.  Returns the number of chunks stored.
        Thread-safe.
        """
        chunks = self._split_text(content)
        added = 0
        texts: List[str] = []
        metadatas: List[dict] = []

        with self._lock:
            for ch in chunks:
                if not ch.strip():
                    continue
                doc_id = str(uuid.uuid4())
                doc = InMemoryDoc(doc_id, ch, source=name)
                self._docs.append(doc)
                self._doc_index[doc_id] = doc
                texts.append(ch)
                metadatas.append({"source": name, "id": doc_id})
                added += 1

            if self._ensure_embeddings() and texts:
                try:
                    if self._faiss_index is None:
                        self._faiss_index = FAISS.from_texts(texts, self._emb, metadatas=metadatas)
                    else:
                        self._faiss_index.add_texts(texts, metadatas=metadatas)
                    logger.debug("FAISS index updated with %d chunks for '%s'.", added, name)
                except Exception as exc:
                    logger.warning("FAISS indexing failed (%s) — keyword fallback active.", exc)

        logger.info("Document '%s' ingested: %d chunks.", name, added)
        return added

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, str, str]]:
        """
        Return up to *k* (id, text, source) tuples most relevant to *query*.
        Uses FAISS when available, otherwise falls back to keyword matching.
        Thread-safe.
        """
        with self._lock:
            if self._faiss_index is not None and self._emb is not None:
                try:
                    results = self._faiss_index.similarity_search(query, k=k)
                    return [
                        (
                            str(r.metadata.get("id") or ""),
                            r.page_content,
                            r.metadata.get("source", "unknown"),
                        )
                        for r in results
                    ]
                except Exception as exc:
                    logger.warning("FAISS search failed (%s) — falling back to keyword.", exc)
            # keyword fallback — take a snapshot to avoid holding the lock during scoring
            docs_snapshot = list(self._docs)

        retr = SimpleKeywordRetriever(docs_snapshot)
        hits = retr.get_relevant(query, k=k)
        return [(d.id, d.text, d.source) for d in hits]

    def get_doc(self, doc_id: str) -> Optional[InMemoryDoc]:
        """Return a document chunk by ID, or None if not found."""
        with self._lock:
            return self._doc_index.get(doc_id)

    def has_docs(self) -> bool:
        with self._lock:
            return bool(self._docs)

    def build_context(
        self, query: str, use_rag: bool, k: int = 4
    ) -> Tuple[str, List[dict]]:
        """
        Build context string and source metadata for injection into LLM prompt.

        Returns (context_str, sources_list).
        sources_list items: {"id", "source", "snippet"}
        """
        if not use_rag or not self.has_docs():
            return "", []
        try:
            hits = self.retrieve(query, k=k)
        except Exception as exc:
            logger.error("retrieve() failed: %s", exc, exc_info=True)
            return "", []

        context = "\n\n".join(text for (_id, text, _src) in hits)
        sources = [
            {"id": _id, "source": src, "snippet": text[:240]}
            for (_id, text, src) in hits
        ]
        return context, sources
