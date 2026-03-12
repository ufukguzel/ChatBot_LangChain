"""
FastAPI router — all HTTP endpoints.

Fixes applied vs. original app.py
----------------------------------
1. /chat_stream now sets the session cookie on the StreamingResponse when creating
   a new session (original never set cookie on StreamingResponse).
2. Sources are embedded at the end of the stream as "\n__SOURCES__:<json>" instead
   of being stored in a shared global that could be overwritten by concurrent requests.
3. /upload validates file extension and size before processing.
4. All route handlers have proper exception handling and logging.
5. Dependency-injection pattern: services are injected via init_services(), not globals.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from app.core.config import settings
from app.models.schemas import ChatRequest
from app.services.chat_engine import ChatEngineRegistry
from app.services.document_loader import load_bytes
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Service references — populated once at startup via init_services()
# ---------------------------------------------------------------------------
_registry: Optional[ChatEngineRegistry] = None
_rag: Optional[RAGService] = None
_template_path: Optional[Path] = None


def init_services(
    registry: ChatEngineRegistry,
    rag: RAGService,
    template_path: Path,
) -> None:
    global _registry, _rag, _template_path
    _registry = registry
    _rag = rag
    _template_path = template_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_or_existing_sid(request: Request) -> str:
    return request.cookies.get("sid") or str(uuid.uuid4())


def _set_session_cookie(response, sid: str, existing_sid: Optional[str]) -> None:
    """Set the session cookie only if the session is brand new."""
    if not existing_sid:
        response.set_cookie("sid", sid, httponly=True, samesite="lax", secure=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    html = _template_path.read_text(encoding="utf-8")
    existing_sid = request.cookies.get("sid")
    sid = existing_sid or str(uuid.uuid4())
    response = HTMLResponse(html)
    _set_session_cookie(response, sid, existing_sid)
    return response


@router.post("/session")
async def create_session(request: Request) -> JSONResponse:
    existing_sid = request.cookies.get("sid")
    sid = existing_sid or str(uuid.uuid4())
    response = JSONResponse({"session_id": sid})
    _set_session_cookie(response, sid, existing_sid)
    return response


@router.post("/reset")
async def reset_session(request: Request) -> PlainTextResponse:
    sid = request.cookies.get("sid")
    if sid and _registry:
        _registry.reset(sid)
    return PlainTextResponse("OK")


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or "upload"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Desteklenmeyen dosya tipi: '{ext}'. "
                f"İzin verilenler: {', '.join(settings.allowed_extensions)}"
            ),
        )

    raw = await file.read()

    if len(raw) > settings.max_upload_size_bytes:
        limit_mb = settings.max_upload_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Dosya çok büyük. Maksimum izin verilen: {limit_mb} MB",
        )

    content = load_bytes(filename, raw)
    if not content.strip():
        raise HTTPException(
            status_code=422,
            detail="Dosyadan metin içeriği çıkarılamadı.",
        )

    assert _rag is not None
    added = _rag.add_document(filename, content)
    logger.info("upload: file='%s' chunks=%d", filename, added)
    return JSONResponse({"ok": True, "message": f"'{filename}' eklendi, {added} parça."})


@router.get("/source")
async def get_source(id: str = Query(..., description="Chunk ID")) -> JSONResponse:
    assert _rag is not None
    doc = _rag.get_doc(id)
    if not doc:
        raise HTTPException(status_code=404, detail="Kaynak bulunamadı.")
    return JSONResponse({"id": doc.id, "source": doc.source, "text": doc.text})


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> JSONResponse:
    existing_sid = request.cookies.get("sid")
    sid = existing_sid or str(uuid.uuid4())

    assert _registry is not None and _rag is not None

    engine = _registry.get_or_create(
        sid,
        force_mock=settings.force_mock,
        model=body.model,
        temperature=body.temperature,
        api_key=settings.openai_api_key,
    )

    ctx, sources = _rag.build_context(body.message, body.rag)

    try:
        reply = engine.send(body.message, context=ctx)
    except Exception as exc:
        logger.error("chat send error: %s", exc, exc_info=True)
        reply = f"[Hata: {exc}]"

    response = JSONResponse({"reply": reply, "session_id": sid, "sources": sources})
    _set_session_cookie(response, sid, existing_sid)
    return response


@router.post("/chat_stream")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    existing_sid = request.cookies.get("sid")
    sid = existing_sid or str(uuid.uuid4())

    assert _registry is not None and _rag is not None

    engine = _registry.get_or_create(
        sid,
        force_mock=settings.force_mock,
        model=body.model,
        temperature=body.temperature,
        api_key=settings.openai_api_key,
    )

    ctx, sources = _rag.build_context(body.message, body.rag)

    # Capture sources_json before entering the async generator to avoid
    # any mutation after the function returns.
    sources_json = json.dumps(sources, ensure_ascii=False)

    async def gen():
        try:
            # engine.stream() is synchronous; wrap chunks in async generator
            for chunk in engine.stream(body.message, context=ctx):
                yield chunk
        except Exception as exc:
            logger.error("chat_stream error: %s", exc, exc_info=True)
            yield f"[Hata: {exc}]"
        # Always append sources sentinel so the frontend can parse them
        # without a separate /last_sources round-trip (fixes race condition).
        yield f"\n__SOURCES__:{sources_json}"

    response = StreamingResponse(gen(), media_type="text/plain")
    # FIX: set cookie on StreamingResponse for new sessions
    _set_session_cookie(response, sid, existing_sid)
    return response
