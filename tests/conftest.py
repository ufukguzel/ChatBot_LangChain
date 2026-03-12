"""
Shared pytest fixtures and configuration.

Key principle: ALL tests must run offline (no real OpenAI calls).
  - Tests that exercise mock-mode ChatEngine directly are fully offline by design.
  - Tests that exercise LangChain paths patch ChatOpenAI so no HTTP is made.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_app():
    """Return a configured FastAPI app instance for testing (mock mode)."""
    import os
    # Force mock mode so the test app never calls OpenAI
    os.environ["FORCE_MOCK"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)

    from app.main import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(test_app):
    """Synchronous TestClient wrapping the test app."""
    from fastapi.testclient import TestClient
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# LangChain patch fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_lc_chain():
    """
    Patch ChatOpenAI and the full LangChain chain so no HTTP calls are made.
    The patched chain returns a canned response string.

    If LangChain is not installed, the symbols don't exist in the module;
    we use create=True so the patch creates them anyway, letting the tests
    that need this fixture still run (they will skip internally if needed).
    """
    canned = "LangChain mock response"

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = canned
    fake_chain.stream.return_value = iter([canned])

    with patch("app.services.chat_engine.ChatOpenAI", create=True) as mock_openai, \
         patch("app.services.chat_engine.RunnableWithMessageHistory", create=True) as mock_rwmh:
        mock_openai.return_value = MagicMock()
        mock_rwmh.return_value = fake_chain
        yield fake_chain
