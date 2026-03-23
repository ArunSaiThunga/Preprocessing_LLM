from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_chunks(jsonl_path: str | Path) -> list[dict]:
    jsonl_path = Path(jsonl_path)

    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def convert_chunks_to_documents(chunks: list[dict]) -> List[Document]:
    documents = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)

    return documents


def build_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(documents: List[Document], embeddings) -> FAISS:
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def save_faiss_index(vectorstore: FAISS, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))


if __name__ == "__main__":
    input_path = Path("data/processed/chunks.jsonl")
    output_dir = Path("data/vectorstore/faiss_index")

    print("Loading chunks...")
    chunks = load_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks")

    print("Converting chunks to documents...")
    documents = convert_chunks_to_documents(chunks)
    print(f"Created {len(documents)} documents")

    print("Loading embedding model...")
    embeddings = build_embeddings_model()

    print("Building FAISS index...")
    vectorstore = build_faiss_index(documents, embeddings)

    print("Saving FAISS index...")
    save_faiss_index(vectorstore, output_dir)

    print("Done.")