"""
PDF text extraction + chunking.
Both are called directly on the uploaded file — no folder scanning needed.
"""

from pypdf import PdfReader
import config


def extract_pdf_text(filepath: str) -> str:
    """Extracts all text from a PDF file on disk (e.g. a temp file)."""
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks by character count.
    Overlap keeps context from being severed at a chunk boundary.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide window forward with overlap

    return chunks
