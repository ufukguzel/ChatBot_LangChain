"""
FastAPI application factory.

Usage
-----
Development:
    uvicorn app.main:app --reload

Production:
    python server.py
    (or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2)
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import init_services, router
from app.core.config import settings
from app.core.logging_setup import setup_logging
from app.services.chat_engine import ChatEngineRegistry
from app.services.rag_service import RAGService

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="LangChain Chatbot API",
        version="2.0.0",
        description="CLI + FastAPI chatbot with streaming, RAG, and PDF support.",
    )

    # --- CORS (configurable via ALLOWED_ORIGINS env var) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # --- Shared services (one instance per process) ---
    registry = ChatEngineRegistry()
    rag = RAGService(api_key=settings.openai_api_key)

    # Inject into router
    init_services(registry, rag, _TEMPLATE_PATH)

    app.include_router(router)

    mock_reason = []
    if settings.force_mock:
        mock_reason.append("FORCE_MOCK=1")
    if not settings.openai_api_key:
        mock_reason.append("no OPENAI_API_KEY")

    if mock_reason:
        logger.info("Mock mode active (%s).", ", ".join(mock_reason))
    else:
        logger.info("LangChain mode: model=%s temperature=%s", settings.model_name, settings.temperature)

    return app


# Module-level instance for uvicorn / gunicorn
app = create_app()
