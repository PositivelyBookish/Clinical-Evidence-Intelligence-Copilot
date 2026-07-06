"""Tests for document loading and extraction behavior."""

from __future__ import annotations

import base64
from pathlib import Path

from src.document_loader import OCR_UNAVAILABLE_WARNING, load_document


def test_load_document_reads_txt_sample() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data" / "synthetic_documents" / "clean_supported_claim.txt"

    result = load_document(sample_path)

    assert result["success"] is True
    assert result["file_type"] == "txt"
    assert result["extraction_method"] == "utf-8 text"
    assert "Hypertension" in result["raw_text"]
    assert result["text_quality_score"] >= 0.5


def test_load_document_handles_missing_ocr_gracefully(tmp_path: Path) -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5X2ioAAAAASUVORK5CYII="
    )
    image_path = tmp_path / "tiny_test.png"
    image_path.write_bytes(png_bytes)

    result = load_document(image_path)

    if result["success"] is False and result["extraction_method"] == "none":
        assert OCR_UNAVAILABLE_WARNING in result["warnings"]
