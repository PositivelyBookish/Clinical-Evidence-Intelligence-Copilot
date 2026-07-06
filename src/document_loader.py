"""Document loading and text extraction utilities for the demo app."""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any


SUPPORTED_FILE_TYPES = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}
CLINICAL_HEADINGS = [
    "Chief Complaint",
    "Assessment",
    "Plan",
    "Procedure",
    "Medication",
    "Provider",
    "Signature",
    "Diagnosis",
]
OCR_UNAVAILABLE_WARNING = (
    "OCR is not available in this local environment. Please use TXT/PDF samples "
    "for the demo or install Tesseract."
)


def load_document(file_or_path: Any) -> dict:
    """Load text from a file path or uploaded file-like object."""
    warnings: list[str] = []
    errors: list[str] = []
    raw_text = ""
    extraction_method = "none"

    file_name, suffix, file_bytes, path_obj = _normalize_input(file_or_path)
    file_type = suffix.lstrip(".")

    if suffix not in SUPPORTED_FILE_TYPES:
        errors.append(
            "Unsupported file type. Please use TXT, PDF, PNG, JPG, or JPEG files."
        )
        return _build_result(
            success=False,
            file_name=file_name,
            file_type=file_type,
            extraction_method=extraction_method,
            raw_text=raw_text,
            warnings=warnings,
            errors=errors,
        )

    try:
        if suffix == ".txt":
            raw_text = _extract_txt(file_bytes, path_obj)
            extraction_method = "utf-8 text"
        elif suffix == ".pdf":
            raw_text, extraction_method, pdf_warnings, pdf_errors = _extract_pdf(
                file_bytes=file_bytes,
                path_obj=path_obj,
            )
            warnings.extend(pdf_warnings)
            errors.extend(pdf_errors)
        else:
            raw_text, extraction_method, image_warnings, image_errors = _extract_image(
                file_bytes=file_bytes,
                path_obj=path_obj,
            )
            warnings.extend(image_warnings)
            errors.extend(image_errors)
    except UnicodeDecodeError:
        errors.append("TXT file could not be decoded as UTF-8.")
    except Exception as exc:  # pragma: no cover - defensive stability layer
        errors.append(f"Unexpected document loading error: {exc}")

    cleaned_text = raw_text.strip()
    if not cleaned_text and not errors:
        warnings.append("No text could be extracted from the document.")

    return _build_result(
        success=bool(cleaned_text),
        file_name=file_name,
        file_type=file_type,
        extraction_method=extraction_method,
        raw_text=cleaned_text,
        warnings=warnings,
        errors=errors,
    )


def _normalize_input(file_or_path: Any) -> tuple[str, str, bytes | None, Path | None]:
    """Normalize a supported input into filename, suffix, bytes, and path."""
    if isinstance(file_or_path, (str, Path)):
        path_obj = Path(file_or_path)
        return path_obj.name, path_obj.suffix.lower(), None, path_obj

    file_name = getattr(file_or_path, "name", "uploaded_document")
    suffix = Path(file_name).suffix.lower()
    file_bytes: bytes | None = None

    if hasattr(file_or_path, "getvalue"):
        file_bytes = file_or_path.getvalue()
    elif hasattr(file_or_path, "read"):
        file_bytes = file_or_path.read()

    return file_name, suffix, file_bytes, None


def _extract_txt(file_bytes: bytes | None, path_obj: Path | None) -> str:
    """Load UTF-8 text directly from bytes or a local path."""
    if file_bytes is not None:
        return file_bytes.decode("utf-8")
    if path_obj is None:
        raise FileNotFoundError("TXT input could not be resolved.")
    return path_obj.read_text(encoding="utf-8")


def _extract_pdf(
    file_bytes: bytes | None,
    path_obj: Path | None,
) -> tuple[str, str, list[str], list[str]]:
    """Extract text from PDF using PyMuPDF first, then pdfplumber."""
    warnings: list[str] = []
    errors: list[str] = []

    pymupdf_error = None
    try:
        import fitz  # type: ignore

        if file_bytes is not None:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        elif path_obj is not None:
            pdf_doc = fitz.open(path_obj)
        else:
            raise FileNotFoundError("PDF input could not be resolved.")

        try:
            pages = [page.get_text("text") for page in pdf_doc]
            text = "\n".join(pages).strip()
        finally:
            pdf_doc.close()

        if text:
            return text, "pymupdf", warnings, errors
        warnings.append("PyMuPDF loaded the PDF but returned very little text.")
    except ImportError:
        pymupdf_error = "PyMuPDF is not installed."
    except Exception as exc:  # pragma: no cover - environment dependent
        pymupdf_error = f"PyMuPDF extraction failed: {exc}"

    if pymupdf_error:
        warnings.append(pymupdf_error)

    try:
        import pdfplumber  # type: ignore

        if file_bytes is not None:
            handle = io.BytesIO(file_bytes)
            pdf_context = pdfplumber.open(handle)
        elif path_obj is not None:
            pdf_context = pdfplumber.open(path_obj)
        else:
            raise FileNotFoundError("PDF input could not be resolved.")

        with pdf_context as pdf:
            pages = [(page.extract_text() or "") for page in pdf.pages]
        text = "\n".join(pages).strip()

        if text:
            return text, "pdfplumber", warnings, errors
        warnings.append("pdfplumber loaded the PDF but returned very little text.")
    except ImportError:
        errors.append(
            "PDF extraction is not available in this local environment. "
            "Install PyMuPDF or pdfplumber to extract PDF text."
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        errors.append(f"pdfplumber extraction failed: {exc}")

    if not errors:
        errors.append(
            "Unable to extract text from the PDF. Try a text-based PDF sample or "
            "install additional PDF extraction support."
        )
    return "", "none", warnings, errors


def _extract_image(
    file_bytes: bytes | None,
    path_obj: Path | None,
) -> tuple[str, str, list[str], list[str]]:
    """Extract OCR text from images when pytesseract and Tesseract are available."""
    warnings: list[str] = []
    errors: list[str] = []

    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError:
        warnings.append(OCR_UNAVAILABLE_WARNING)
        return "", "none", warnings, errors

    if shutil.which("tesseract") is None:
        warnings.append(OCR_UNAVAILABLE_WARNING)
        return "", "none", warnings, errors

    try:
        if file_bytes is not None:
            image = Image.open(io.BytesIO(file_bytes))
        elif path_obj is not None:
            image = Image.open(path_obj)
        else:
            raise FileNotFoundError("Image input could not be resolved.")
        text = pytesseract.image_to_string(image).strip()
        return text, "pytesseract", warnings, errors
    except Exception as exc:  # pragma: no cover - environment dependent
        errors.append(f"Image OCR extraction failed: {exc}")
        return "", "pytesseract", warnings, errors


def _build_result(
    success: bool,
    file_name: str,
    file_type: str,
    extraction_method: str,
    raw_text: str,
    warnings: list[str],
    errors: list[str],
) -> dict:
    """Construct the standard document loading payload."""
    return {
        "success": success,
        "file_name": file_name,
        "file_type": file_type,
        "extraction_method": extraction_method,
        "raw_text": raw_text,
        "warnings": warnings,
        "errors": errors,
        "text_quality_score": _score_text_quality(raw_text),
    }


def _score_text_quality(text: str) -> float:
    """Assign a lightweight quality score to extracted text."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return 0.0

    text_length = len(cleaned_text)
    has_clinical_heading = any(
        heading.lower() in cleaned_text.lower() for heading in CLINICAL_HEADINGS
    )

    if text_length > 500 and has_clinical_heading:
        return 1.0
    if text_length > 200:
        return 0.8
    if text_length > 50:
        return 0.5
    return 0.2
