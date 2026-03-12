"""Unit tests for rag_service.py — all offline (no FAISS / OpenAI calls)."""
import pytest

from app.services.rag_service import (
    InMemoryDoc,
    RAGService,
    SimpleKeywordRetriever,
)


# ---------------------------------------------------------------------------
# InMemoryDoc
# ---------------------------------------------------------------------------

class TestInMemoryDoc:
    def test_fields(self):
        doc = InMemoryDoc("id1", "text body", "file.txt")
        assert doc.id == "id1"
        assert doc.text == "text body"
        assert doc.source == "file.txt"


# ---------------------------------------------------------------------------
# SimpleKeywordRetriever
# ---------------------------------------------------------------------------

class TestSimpleKeywordRetriever:
    def test_returns_most_relevant_first(self):
        docs = [
            InMemoryDoc("1", "Python programming language tutorial", "py.txt"),
            InMemoryDoc("2", "Java coffee brewing guide", "java.txt"),
            InMemoryDoc("3", "Python data science pandas numpy", "ds.txt"),
        ]
        retr = SimpleKeywordRetriever(docs)
        results = retr.get_relevant("Python tutorial", k=1)
        assert results[0].id == "1"

    def test_top_k_limit(self):
        docs = [InMemoryDoc(str(i), f"word{i} shared common", f"doc{i}.txt") for i in range(10)]
        retr = SimpleKeywordRetriever(docs)
        results = retr.get_relevant("shared common", k=3)
        assert len(results) == 3

    def test_empty_docs(self):
        retr = SimpleKeywordRetriever([])
        results = retr.get_relevant("anything", k=3)
        assert results == []

    def test_short_words_ignored_in_scoring(self):
        # Words <= 2 chars should not affect score
        docs = [InMemoryDoc("1", "a an is", "a.txt"), InMemoryDoc("2", "big large huge words", "b.txt")]
        retr = SimpleKeywordRetriever(docs)
        results = retr.get_relevant("big large", k=1)
        assert results[0].id == "2"


# ---------------------------------------------------------------------------
# RAGService — keyword fallback (api_key=None → no FAISS)
# ---------------------------------------------------------------------------

class TestRAGService:
    def test_has_no_docs_initially(self):
        rag = RAGService(api_key=None)
        assert not rag.has_docs()

    def test_add_document_returns_chunk_count(self):
        rag = RAGService(api_key=None)
        n = rag.add_document("test.txt", "Hello world content here.")
        assert n >= 1

    def test_has_docs_after_add(self):
        rag = RAGService(api_key=None)
        rag.add_document("x.txt", "some content")
        assert rag.has_docs()

    def test_retrieve_returns_relevant_chunks(self):
        rag = RAGService(api_key=None)
        rag.add_document("py.txt", "LangChain is a Python framework for building LLM applications.")
        hits = rag.retrieve("LangChain Python", k=1)
        assert len(hits) == 1
        _id, text, source = hits[0]
        assert "LangChain" in text
        assert source == "py.txt"

    def test_get_doc_by_id(self):
        rag = RAGService(api_key=None)
        rag.add_document("f.txt", "hello world content")
        doc_id = rag._docs[0].id
        doc = rag.get_doc(doc_id)
        assert doc is not None
        assert doc.source == "f.txt"

    def test_get_doc_missing_returns_none(self):
        rag = RAGService(api_key=None)
        assert rag.get_doc("non-existent-id") is None

    def test_build_context_no_docs(self):
        rag = RAGService(api_key=None)
        ctx, sources = rag.build_context("query", use_rag=True)
        assert ctx == ""
        assert sources == []

    def test_build_context_rag_disabled(self):
        rag = RAGService(api_key=None)
        rag.add_document("x.txt", "some content here for testing")
        ctx, sources = rag.build_context("query", use_rag=False)
        assert ctx == ""
        assert sources == []

    def test_build_context_with_docs_and_rag_enabled(self):
        rag = RAGService(api_key=None)
        rag.add_document("doc.txt", "LangChain Python framework RAG retrieval")
        ctx, sources = rag.build_context("LangChain", use_rag=True)
        assert len(ctx) > 0
        assert len(sources) >= 1
        assert "id" in sources[0]
        assert "source" in sources[0]
        assert "snippet" in sources[0]

    def test_sources_snippet_max_240_chars(self):
        long_text = "word " * 200  # 1000 chars
        rag = RAGService(api_key=None)
        rag.add_document("long.txt", long_text)
        _, sources = rag.build_context("word", use_rag=True)
        assert all(len(s["snippet"]) <= 240 for s in sources)

    def test_multiple_documents(self):
        rag = RAGService(api_key=None)
        rag.add_document("a.txt", "Apple fruit red")
        rag.add_document("b.txt", "Banana fruit yellow")
        ctx, sources = rag.build_context("fruit", use_rag=True)
        source_names = {s["source"] for s in sources}
        assert source_names  # at least one source returned

    def test_thread_safety_concurrent_add(self):
        """Concurrent document ingestion must not corrupt the index."""
        import threading
        rag = RAGService(api_key=None)
        errors = []

        def add_docs(i):
            try:
                rag.add_document(f"file{i}.txt", f"Content for document {i} with unique words delta{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_docs, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert rag.has_docs()
