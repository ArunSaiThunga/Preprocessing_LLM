from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class PageRecord:
    source_id: str
    file_name: str
    page_number: int
    section: str | None
    timestamp_or_version: str | None
    domain_category: str | None
    raw_text: str
    cleaned_text: str
    image_count: int
    has_numeric_content: bool
    metadata: dict[str, Any]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_broken_linebreaks(text: str) -> str:
    # Join lines that were broken in the middle of a sentence
    text = re.sub(r"(?<![.!?:\n])\n(?![\n•\-\d])", " ", text)
    return text


def remove_repeated_headers_footers(page_texts: list[str]) -> list[str]:
    """
    Very simple heuristic:
    - detect first/last non-empty lines that repeat across many pages
    - remove them
    """
    first_lines = []
    last_lines = []

    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            first_lines.append("")
            last_lines.append("")
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])

    def repeated_candidates(lines: list[str], min_ratio: float = 0.5) -> set[str]:
        counts = {}
        for line in lines:
            if line:
                counts[line] = counts.get(line, 0) + 1
        threshold = max(2, int(len(lines) * min_ratio))
        return {line for line, cnt in counts.items() if cnt >= threshold}

    repeated_first = repeated_candidates(first_lines)
    repeated_last = repeated_candidates(last_lines)

    cleaned_pages = []
    for text in page_texts:
        lines = [ln for ln in text.splitlines()]
        stripped = [ln.strip() for ln in lines if ln.strip()]

        if not stripped:
            cleaned_pages.append(text)
            continue

        output_lines = []
        first_removed = False
        last_nonempty_index = max(
            (i for i, ln in enumerate(lines) if ln.strip()), default=-1
        )

        for i, line in enumerate(lines):
            s = line.strip()
            if not first_removed and s in repeated_first:
                first_removed = True
                continue
            if i == last_nonempty_index and s in repeated_last:
                continue
            output_lines.append(line)

        cleaned_pages.append("\n".join(output_lines))

    return cleaned_pages


def detect_numeric_content(text: str) -> bool:
    # crude but useful for KPI-heavy pages
    return bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", text))


def extract_images_from_page(page: fitz.Page, output_dir: Path, page_number: int) -> int:
    """
    Saves displayed image blocks from the page (when available).
    Uses page.get_text('dict'), where image blocks appear with type == 1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    page_dict = page.get_text("dict")
    image_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 1]

    saved = 0
    for idx, block in enumerate(image_blocks):
        image_bytes = block.get("image")
        ext = block.get("ext", "png")
        if not image_bytes:
            continue

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_path = output_dir / f"page_{page_number+1:03d}_img_{idx+1:02d}.{ext}"
            img.save(img_path)
            saved += 1
        except Exception:
            # skip malformed image blocks
            continue

    return saved


def extract_page_text(page: fitz.Page, ocr_if_needed: bool = False) -> str:
    """
    Start with sorted text for better reading order.
    If almost no text exists and OCR is enabled, OCR the page.
    """
    text = page.get_text("text", sort=True)

    if ocr_if_needed and len(text.strip()) < 30:
        try:
            textpage = page.get_textpage_ocr()
            text = page.get_text("text", textpage=textpage)
        except Exception:
            pass

    return text


def ingest_pdf(
    pdf_path: str | Path,
    output_jsonl: str | Path,
    image_dir: str | Path,
    source_id: str,
    section: str | None = None,
    timestamp_or_version: str | None = None,
    domain_category: str | None = None,
    ocr_if_needed: bool = False,
) -> list[PageRecord]:
    pdf_path = Path(pdf_path)
    output_jsonl = Path(output_jsonl)
    image_dir = Path(image_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    raw_page_texts: list[str] = []
    page_meta: list[dict[str, Any]] = []
    page_images: list[int] = []

    for page_number, page in enumerate(doc):
        text = extract_page_text(page, ocr_if_needed=ocr_if_needed)
        raw_page_texts.append(text)

        img_count = extract_images_from_page(page, image_dir, page_number)
        page_images.append(img_count)

        page_meta.append(
            {
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "rotation": page.rotation,
            }
        )

    cleaned_page_texts = remove_repeated_headers_footers(raw_page_texts)

    records: list[PageRecord] = []
    for i, text in enumerate(cleaned_page_texts):
        cleaned = normalize_whitespace(fix_broken_linebreaks(text))

        record = PageRecord(
            source_id=source_id,
            file_name=pdf_path.name,
            page_number=i + 1,
            section=section,
            timestamp_or_version=timestamp_or_version,
            domain_category=domain_category,
            raw_text=raw_page_texts[i],
            cleaned_text=cleaned,
            image_count=page_images[i],
            has_numeric_content=detect_numeric_content(cleaned),
            metadata=page_meta[i],
        )
        records.append(record)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    return records


if __name__ == "__main__":
    pdf_path = Path(r"D:\30day git\raw_data_preprocessing_for_LLM\data\raw\Arun Sai Thunga Master Thesis.pdf")
    output_jsonl = Path(r"D:\30day git\raw_data_preprocessing_for_LLM\data\extracted\pages.jsonl")
    image_dir = Path(r"D:\30day git\raw_data_preprocessing_for_LLM\data\extracted\images")

    records = ingest_pdf(
        pdf_path=pdf_path,
        output_jsonl=output_jsonl,
        image_dir=image_dir,
        source_id="doc_001",
        section="full_document",
        timestamp_or_version="v1",
        domain_category="enterprise_pdf",
        ocr_if_needed=False,
    )

    print(f"Ingested {len(records)} pages")