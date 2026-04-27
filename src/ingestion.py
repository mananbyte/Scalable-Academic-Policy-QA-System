"""
Data Ingestion Module
- Extract text from UG Handbook PDF
- Clean text (remove noise)
- Chunk into 200-500 word segments with metadata (page, section)
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

from config import (
    UG_HANDBOOK_PDF,
    PROCESSED_DIR,
    CHUNKS_FILE,
    CHUNK_MIN_WORDS,
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each page of the PDF, returning list of {page, text}."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """Clean extracted text: remove excessive whitespace, control chars, etc."""
    # Remove form-feed and other control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Collapse multiple newlines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def detect_section_title(text: str) -> str:
    """Try to detect the chapter/section title from the beginning of a chunk."""
    # Look for "Chapter N:" pattern
    match = re.search(r"(Chapter\s+\d+[^:\n]*:?[^\n]*)", text[:300])
    if match:
        return match.group(1).strip()
    # Look for numbered headings like "1. Something"
    match = re.search(r"^(\d+\.\s+[A-Z][^\n]{5,80})", text[:300], re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page texts into chunks of CHUNK_MIN_WORDS to CHUNK_MAX_WORDS.
    Each chunk records: chunk_id, source_page(s), section_title, text.
    """
    # First, merge all pages into segments split by chapter headings
    all_text = ""
    page_boundaries = []  # (char_offset, page_number)
    for p in pages:
        start = len(all_text)
        cleaned = clean_text(p["text"])
        if not cleaned:
            continue
        all_text += cleaned + "\n\n"
        page_boundaries.append((start, p["page"]))

    # Helper: find page number for a character offset
    def get_page(offset: int) -> int:
        page_num = 1
        for start, pn in page_boundaries:
            if start <= offset:
                page_num = pn
            else:
                break
        return page_num

    # Split by paragraphs first
    paragraphs = re.split(r"\n{2,}", all_text)
    para_offsets = []
    offset = 0
    for para in paragraphs:
        idx = all_text.find(para, offset)
        para_offsets.append(idx)
        offset = idx + len(para)

    # Now group paragraphs into chunks of appropriate size
    chunks = []
    chunk_id = 0
    i = 0
    while i < len(paragraphs):
        chunk_start_i = i  # track where this chunk started
        current_words = []
        current_text_parts = []
        start_offset = para_offsets[i]
        end_offset = start_offset

        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if not para:
                i += 1
                continue
            words = para.split()
            # If adding this paragraph would exceed max, and we already have enough, stop
            if len(current_words) + len(words) > CHUNK_MAX_WORDS and len(current_words) >= CHUNK_MIN_WORDS:
                break
            current_words.extend(words)
            current_text_parts.append(para)
            end_offset = para_offsets[i] + len(paragraphs[i])
            i += 1

            # If we've reached a good size, break
            if len(current_words) >= CHUNK_MIN_WORDS:
                if len(current_words) >= CHUNK_MAX_WORDS:
                    break

        if not current_text_parts:
            i += 1
            continue

        text = "\n\n".join(current_text_parts)
        start_page = get_page(start_offset)
        end_page = get_page(end_offset)
        section = detect_section_title(text)

        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "word_count": len(current_words),
            "start_page": start_page,
            "end_page": end_page,
            "section_title": section,
            "source": "UG Handbook",
        })
        chunk_id += 1

        # No overlap backtracking — just move forward to avoid infinite loops

    return chunks


def run_ingestion() -> list[dict]:
    """Run the full ingestion pipeline. Returns the list of chunks."""
    print(f"[Ingestion] Extracting text from {UG_HANDBOOK_PDF.name}...")
    pages = extract_text_from_pdf(UG_HANDBOOK_PDF)
    print(f"[Ingestion] Extracted {len(pages)} pages.")

    print("[Ingestion] Chunking...")
    chunks = chunk_pages(pages)
    print(f"[Ingestion] Created {len(chunks)} chunks.")

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[Ingestion] Saved chunks to {CHUNKS_FILE}")

    # Print stats
    word_counts = [c["word_count"] for c in chunks]
    print(f"[Ingestion] Word count stats: min={min(word_counts)}, max={max(word_counts)}, "
          f"avg={sum(word_counts)/len(word_counts):.0f}")

    return chunks


if __name__ == "__main__":
    run_ingestion()
