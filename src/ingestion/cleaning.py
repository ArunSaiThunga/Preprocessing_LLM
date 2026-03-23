from __future__ import annotations

import re


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
    Detect first/last non-empty lines that repeat across many pages
    and remove them.
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


def clean_text(text: str) -> str:
    text = fix_broken_linebreaks(text)
    text = normalize_whitespace(text)
    return text