"""
Integration tests for FastAPI endpoints.

All tests use a TestClient configured with FORCE_MOCK=1 (set in conftest.py).
No real OpenAI calls are made.
"""
import json
import pytest


# ---------------------------------------------------------------------------
# GET / — HTML index
# ---------------------------------------------------------------------------

class TestIndexEndpoint:
    def test_returns_200_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_html_contains_app_title(self, client):
        r = client.get("/")
        assert "LangChain Chatbot" in r.text

    def test_sets_session_cookie(self, client):
        # Fresh client without any existing cookies
        from fastapi.testclient import TestClient
        from app.main import create_app
        import os
        os.environ["FORCE_MOCK"] = "1"
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/")
        assert "sid" in r.cookies


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    def test_returns_session_id(self, client):
        r = client.post("/session")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_idempotent_with_existing_cookie(self, client):
        r1 = client.post("/session")
        sid1 = r1.json()["session_id"]
        # Second call reuses the cookie set by first
        r2 = client.post("/session")
        sid2 = r2.json()["session_id"]
        # Same client → same session
        assert sid1 == sid2


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

class TestResetEndpoint:
    def test_reset_returns_ok(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.text == "OK"

    def test_reset_without_session_does_not_error(self, client):
        from fastapi.testclient import TestClient
        from app.main import create_app
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/reset")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_basic_chat(self, client):
        r = client.post(
            "/chat",
            json={"message": "Hello bot", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    def test_reply_is_string(self, client):
        r = client.post(
            "/chat",
            json={"message": "Test", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert isinstance(r.json()["reply"], str)

    def test_sources_empty_when_no_docs(self, client):
        r = client.post(
            "/chat",
            json={"message": "Test RAG", "model": "gpt-4o-mini", "temperature": 0.4, "rag": True},
        )
        assert r.json()["sources"] == []

    def test_empty_message_rejected(self, client):
        r = client.post(
            "/chat",
            json={"message": "", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 422

    def test_message_too_long_rejected(self, client):
        r = client.post(
            "/chat",
            json={"message": "x" * 9000, "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 422

    def test_invalid_temperature_rejected(self, client):
        r = client.post(
            "/chat",
            json={"message": "Hi", "model": "gpt-4o-mini", "temperature": 5.0, "rag": False},
        )
        assert r.status_code == 422

    def test_session_cookie_set_on_new_session(self, client):
        from fastapi.testclient import TestClient
        from app.main import create_app
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.post(
            "/chat",
            json={"message": "Hello", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 200
        assert "sid" in r.cookies


# ---------------------------------------------------------------------------
# POST /chat_stream
# ---------------------------------------------------------------------------

class TestChatStreamEndpoint:
    def test_returns_200(self, client):
        r = client.post(
            "/chat_stream",
            json={"message": "Stream test", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 200

    def test_response_is_non_empty(self, client):
        r = client.post(
            "/chat_stream",
            json={"message": "Hello", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert len(r.text) > 0

    def test_sources_sentinel_present(self, client):
        r = client.post(
            "/chat_stream",
            json={"message": "Hello", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert "\n__SOURCES__:" in r.text

    def test_sources_sentinel_valid_json(self, client):
        r = client.post(
            "/chat_stream",
            json={"message": "Hello", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        idx = r.text.rfind("\n__SOURCES__:")
        assert idx != -1
        json_part = r.text[idx + len("\n__SOURCES__:"):].strip()
        sources = json.loads(json_part)  # must not raise
        assert isinstance(sources, list)

    def test_session_cookie_set_on_new_session(self, client):
        from fastapi.testclient import TestClient
        from app.main import create_app
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)
        r = c.post(
            "/chat_stream",
            json={"message": "Hello", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 200
        # Cookie must be present on new-session streaming response
        assert "sid" in r.cookies

    def test_empty_message_rejected(self, client):
        r = client.post(
            "/chat_stream",
            json={"message": "", "model": "gpt-4o-mini", "temperature": 0.4, "rag": False},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    def test_upload_txt(self, client):
        r = client.post(
            "/upload",
            files={"file": ("hello.txt", b"Hello world content for RAG testing.", "text/plain")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "hello.txt" in data["message"]

    def test_upload_md(self, client):
        r = client.post(
            "/upload",
            files={"file": ("readme.md", b"# Title\nContent here.", "text/markdown")},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_upload_invalid_extension(self, client):
        r = client.post(
            "/upload",
            files={"file": ("malware.exe", b"\x4d\x5a binary", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_upload_invalid_extension_js(self, client):
        r = client.post(
            "/upload",
            files={"file": ("script.js", b"alert(1)", "text/javascript")},
        )
        assert r.status_code == 400

    def test_upload_empty_file_rejected(self, client):
        r = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        # Empty file → 422 (no extractable content)
        assert r.status_code in (400, 413, 422)

    def test_upload_invalid_pdf_fallback(self, client):
        # Not a real PDF but extension is .pdf → should still not crash
        r = client.post(
            "/upload",
            files={"file": ("fake.pdf", b"not a real pdf content here for testing", "application/pdf")},
        )
        # Either succeeds (raw decode fallback) or 422 if no extractable text
        assert r.status_code in (200, 422)


# ---------------------------------------------------------------------------
# GET /source
# ---------------------------------------------------------------------------

class TestSourceEndpoint:
    def test_source_not_found(self, client):
        r = client.get("/source?id=does-not-exist")
        assert r.status_code == 404

    def test_source_found_after_upload(self, client):
        from fastapi.testclient import TestClient
        from app.main import create_app
        import os
        os.environ["FORCE_MOCK"] = "1"
        app = create_app()
        c = TestClient(app, raise_server_exceptions=False)

        upload_r = c.post(
            "/upload",
            files={"file": ("sample.txt", b"This is sample document content for source test.", "text/plain")},
        )
        assert upload_r.status_code == 200

        # We need to get the doc ID from the RAG service; do it via a chat + sources round
        # Instead, access the RAG service directly through the app state
        from app.api import routes as r_module
        rag = r_module._rag
        assert rag is not None and rag.has_docs()
        doc_id = rag._docs[0].id

        source_r = c.get(f"/source?id={doc_id}")
        assert source_r.status_code == 200
        data = source_r.json()
        assert data["source"] == "sample.txt"
        assert len(data["text"]) > 0
