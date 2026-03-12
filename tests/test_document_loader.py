"""Unit tests for document_loader.py — fully offline."""
import pytest

from app.services.document_loader import load_bytes, _decode_text, _read_pdf


class TestDecodeText:
    def test_utf8_text(self):
        result = _decode_text("Merhaba dünya".encode("utf-8"))
        assert "Merhaba" in result

    def test_empty_bytes(self):
        assert _decode_text(b"") == ""

    def test_invalid_utf8_replaced(self):
        result = _decode_text(b"\xff\xfe invalid utf8 here")
        assert isinstance(result, str)  # no exception


class TestReadPdf:
    def test_invalid_pdf_fallback_to_bytes(self):
        result = _read_pdf(b"this is not a PDF")
        assert isinstance(result, str)  # no exception, returns string

    def test_empty_bytes_fallback(self):
        result = _read_pdf(b"")
        assert isinstance(result, str)


class TestLoadBytes:
    def test_txt_file(self):
        content = load_bytes("readme.txt", b"Hello world content")
        assert "Hello world content" in content

    def test_md_file(self):
        content = load_bytes("notes.md", b"# Heading\n\nSome content.")
        assert "Heading" in content

    def test_pdf_fallback_on_invalid(self):
        content = load_bytes("report.pdf", b"not a real pdf binary")
        assert isinstance(content, str)

    def test_empty_bytes_returns_empty(self):
        content = load_bytes("empty.txt", b"")
        assert content == ""

    def test_unknown_extension_treated_as_text(self):
        content = load_bytes("data.csv", b"col1,col2\n1,2")
        assert "col1" in content

    def test_uppercase_extension(self):
        content = load_bytes("doc.TXT", b"Case insensitive extension")
        assert "Case insensitive" in content

    def test_no_extension(self):
        content = load_bytes("noextfile", b"plain text content")
        assert "plain text" in content
