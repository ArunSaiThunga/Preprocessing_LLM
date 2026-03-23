from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =========================
# LOAD CROSS-ENCODER
# =========================
def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    print("Loading CrossEncoder...")
    return CrossEncoder(model_name)


# =========================
# LOAD LOCAL LLM
# =========================
def load_llm(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print("Loading LLM (TinyLlama)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.2,
    )

    return pipe


# =========================
# CROSS-ENCODER RERANKING
# =========================
def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:

    if not documents:
        return []

    model = load_reranker()

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = []
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append(doc)

    scored_docs.sort(
        key=lambda d: d.metadata["rerank_score"],
        reverse=True,
    )

    return scored_docs[:top_k]



# ==========================
# LLM-BASED SELECTION
# ==========================
def llm_select_best_chunks(
    query: str,
    documents: List[Document],
    top_k: int = 3,
) -> List[Document]:
    """
    Uses local LLM to select most relevant chunks
    """

    if not documents:
        return []

    llm = load_llm()

    # Prepare context
    context = ""
    for i, doc in enumerate(documents):
        context += f"\n[{i}]\n{doc.page_content[:300]}\n"

    prompt = f"""
You are a document relevance evaluator.

Query:
{query}

Below are document chunks:

{context}

Select the {top_k} most relevant chunks.
Return ONLY the indices as a comma-separated list.

Example:
0,2,3
"""

    output = llm(prompt)[0]["generated_text"]

    # Extract indices
    selected_indices = []
    for token in output.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token)
            if idx < len(documents):
                selected_indices.append(idx)

    # fallback if parsing fails
    if not selected_indices:
        return documents[:top_k]

    unique_indices = []
    seen = set()

    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    selected_docs = [documents[i] for i in unique_indices[:top_k]]

    return selected_docs


# =========================
# COMBINED PIPELINE
# =========================
def rerank_and_select(
    query: str,
    documents: List[Document],
    rerank_top_k: int = 8,
    final_top_k: int = 3,
) -> List[Document]:
    """
    Full pipeline:
    1. Cross-encoder reranking
    2. LLM-based selection
    """

    # Step 1: rerank
    reranked = rerank_documents(
        query=query,
        documents=documents,
        top_k=rerank_top_k,
    )

    # Step 2: LLM selection
    final_docs = llm_select_best_chunks(
        query=query,
        documents=reranked,
        top_k=final_top_k,
    )

    return final_docs


# =========================
# DEBUG PRINT
# =========================
def print_reranked(results: List[Document], title: str = "FINAL SELECTED RESULTS"):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Rank {i} ---")
        print("Score:", round(doc.metadata.get("rerank_score", 0), 4))
        print("Metadata:", doc.metadata)
        print("Text:", doc.page_content[:400].replace("\n", " "))
        print()



if __name__ == "__main__":
    print("Starting reranker test...\n")

    from pathlib import Path
    from query_faiss import (
        load_embeddings_model,
        load_faiss_index,
        multiquery_retrieve,
    )

    # Setup
    index_dir = Path("data/vectorstore/faiss_index")
    query = "what is the software knowledge  of arunsai"

    embeddings = load_embeddings_model()
    vectorstore = load_faiss_index(index_dir, embeddings)

    # Step 1: Retrieve documents
    docs = multiquery_retrieve(
        vectorstore=vectorstore,
        query=query,
        mode="mmr",
        k_per_query=3,
    )

    print(f"\nRetrieved {len(docs)} docs BEFORE reranking\n")

    # Step 2: Rerank + LLM select
    final_docs = rerank_and_select(
        query=query,
        documents=docs,
        rerank_top_k=8,
        final_top_k=3,
    )

    print_reranked(final_docs)