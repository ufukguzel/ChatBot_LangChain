"""
Document loading utilities.

Supports:
  - Plain text / Markdown  → UTF-8 decode with error replacement
  - PDF                    → pypdf extraction with graceful fallback to raw bytes
"""
import io
import logging

logger = logging.getLogger(__name__)

_PDF_AVAILABLE = True
try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    _PDF_AVAILABLE = False
    logger.warning("pypdf not installed — PDF files will be decoded as raw bytes.")


def load_bytes(filename: str, data: bytes) -> str:
    """
    Extract text from uploaded file bytes.

    Returns an empty string on unrecoverable errors rather than raising.
    """
    if not data:
        return ""
    lower = (filename or "").lower()
    if lower.endswith(".pdf"):
        return _read_pdf(data)
    return _decode_text(data)


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error("Failed to decode text bytes: %s", exc)
        return ""


def _read_pdf(data: bytes) -> str:
    """Extract text from PDF bytes; fall back to raw decode if pypdf is absent or fails."""
    if not _PDF_AVAILABLE:
        logger.debug("pypdf unavailable — falling back to raw byte decode for PDF.")
        return _decode_text(data)

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                parts.append(text)
            except Exception as exc:
                logger.warning("Could not extract page %d: %s", i, exc)
        return "\n".join(parts)
    except Exception as exc:
        logger.warning("pypdf failed to read PDF (%s) — falling back to raw decode.", exc)
        return _decode_text(data)
