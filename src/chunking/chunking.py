from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pages(jsonl_path: str | Path) -> list[dict]:
    jsonl_path = Path(jsonl_path)

    pages = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    return pages


def convert_to_documents(pages: list[dict]) -> List[Document]:
    documents = []

    for page in pages:
        doc = Document(
            page_content=page["cleaned_text"],
            metadata={
                "source_id": page["source_id"],
                "file_name": page["file_name"],
                "page_number": page["page_number"],
                "section": page["section"],
                "domain_category": page["domain_category"],
                "has_numeric_content": page["has_numeric_content"],
            },
        )
        documents.append(doc)

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []

    for doc in documents:
        parent_id = f"{doc.metadata['source_id']}_p{doc.metadata['page_number']}"

        split_chunks = splitter.split_documents([doc])

        for idx, chunk in enumerate(split_chunks):
            chunk_id = f"{parent_id}_c{idx+1:03d}"

            # 👉 enrich metadata
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "chunk_index": idx + 1,
            })

            chunks.append(chunk)

    return chunks


def save_chunks(chunks: List[Document], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.metadata["chunk_id"],
                "parent_id": chunk.metadata["parent_id"],
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    input_path = Path("data/extracted/pages.jsonl")
    output_path = Path("data/processed/chunks.jsonl")

    print("Loading pages...")
    pages = load_pages(input_path)

    print(f"Loaded {len(pages)} pages")

    print("Converting to documents...")
    documents = convert_to_documents(pages)

    print(f"Created {len(documents)} documents")

    print("Chunking...")
    chunks = chunk_documents(documents)

    print(f"Generated {len(chunks)} chunks")

    print("Saving chunks...")
    save_chunks(chunks, output_path)

    print("Done.")