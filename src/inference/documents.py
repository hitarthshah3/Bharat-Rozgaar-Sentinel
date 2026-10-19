"""In-memory extraction of text from uploaded job-posting documents."""

from __future__ import annotations

import io
from functools import lru_cache
import re
from pathlib import Path

import pymupdf
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_DOCUMENT_PAGES = 12
MAX_IMAGE_PIXELS = 25_000_000
MIN_TEXT_CHARS_FOR_NATIVE_PAGE = 24
MAX_EXTRACTED_CHARS = 60_000
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


class DocumentUploadError(ValueError):
    """An uploaded file is invalid or could not produce usable text."""


class OCRUnavailableError(RuntimeError):
    """OCR was needed but its runtime is unavailable."""


@lru_cache(maxsize=1)
def _ocr_engine():
    """Load bundled English OCR models once per API process."""
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params={
            "Det.lang_type": "en",
            "Rec.lang_type": "en",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })
    except Exception as exc:
        raise OCRUnavailableError(
            "The OCR runtime is unavailable. Install the project's requirements and restart the API."
        ) from exc


def _validate_image(contents: bytes) -> None:
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise DocumentUploadError("Choose a PNG or JPEG image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise DocumentUploadError("This image is too large to process. Choose one under 25 megapixels.")
            image.verify()
    except DocumentUploadError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentUploadError("The image could not be opened. Try a PNG or JPEG file.") from exc


def _ocr_image(image: Image.Image) -> str:
    try:
        result = _ocr_engine()(image.convert("RGB"))
        return "\n".join(line.strip() for line in (result.txts or ()) if line.strip())
    except OCRUnavailableError:
        raise
    except Exception as exc:
        raise DocumentUploadError("OCR could not read this file. Try a clearer image or paste the text manually.") from exc


def _ocr_page(page: pymupdf.Page) -> str:
    scale = 220 / 72
    estimated_pixels = page.rect.width * page.rect.height * scale * scale
    if estimated_pixels > MAX_IMAGE_PIXELS:
        scale *= (MAX_IMAGE_PIXELS / estimated_pixels) ** 0.5
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return _ocr_image(image)


def extract_posting_text(filename: str, contents: bytes) -> dict:
    """Extract native or OCR text and return it without writing the upload to disk."""
    extension = Path(filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentUploadError("Upload a PDF, PNG, or JPEG file.")
    if not contents:
        raise DocumentUploadError("The selected file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise DocumentUploadError("The file is over the 15 MB upload limit.")
    if extension == ".pdf" and not contents.startswith(b"%PDF-"):
        raise DocumentUploadError("The file does not appear to be a valid PDF.")
    if extension in {".png", ".jpg", ".jpeg"}:
        _validate_image(contents)
        try:
            with Image.open(io.BytesIO(contents)) as image:
                extracted_text = _ocr_image(image)
        except (UnidentifiedImageError, OSError) as exc:
            raise DocumentUploadError("The image could not be opened. Try a PNG or JPEG file.") from exc
        if not extracted_text:
            raise DocumentUploadError("No readable text was found. Try a clearer image or paste the text manually.")
        return {
            "filename": Path(filename).name[:200],
            "page_count": 1,
            "extraction_method": "Image OCR",
            "text": extracted_text[:MAX_EXTRACTED_CHARS],
            "warning": "Review and correct the extracted text before assessment; OCR can make mistakes.",
        }

    try:
        document = pymupdf.open(stream=contents, filetype="pdf")
    except Exception as exc:
        raise DocumentUploadError("The file could not be opened. Try exporting it again or use another format.") from exc

    with document:
        if document.needs_pass:
            raise DocumentUploadError("This PDF is password-protected. Upload an unlocked copy.")
        page_count = document.page_count
        if not 1 <= page_count <= MAX_DOCUMENT_PAGES:
            raise DocumentUploadError(f"Upload a document with between 1 and {MAX_DOCUMENT_PAGES} pages.")

        extracted_pages = []
        used_ocr = False
        for page in document:
            text = page.get_text("text").strip()
            if extension != ".pdf" or len(re.sub(r"\s", "", text)) < MIN_TEXT_CHARS_FOR_NATIVE_PAGE:
                text = _ocr_page(page)
                used_ocr = True
            if text:
                extracted_pages.append(text)

    extracted_text = "\n\n".join(extracted_pages).strip()
    if not extracted_text:
        raise DocumentUploadError("No readable text was found. Try a clearer file or paste the text manually.")
    if len(extracted_text) > MAX_EXTRACTED_CHARS:
        extracted_text = extracted_text[:MAX_EXTRACTED_CHARS]

    if extension == ".pdf":
        method = "PDF text + OCR" if used_ocr else "PDF text extraction"
    else:
        method = "Image OCR"
    return {
        "filename": Path(filename).name[:200],
        "page_count": page_count,
        "extraction_method": method,
        "text": extracted_text,
        "warning": "Review and correct the extracted text before assessment; OCR can make mistakes." if used_ocr else "",
    }
