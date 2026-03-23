from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Import your modules
from src.retrieval.query_faiss import (
    load_embeddings_model,
    load_faiss_index,
    multiquery_retrieve,
)

from src.retrieval.query_faiss import (
    multiquery_retrieve,
)

from src.retrieval.reranker import rerank_and_select

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =========================
# LOAD GENERATOR LLM
# =========================
def load_generator(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.3,
    )

    return pipe


# =========================
# BUILD CONTEXT
# =========================
def build_context(documents: List[Document]) -> str:
    """
    Convert selected chunks into clean context
    """

    seen_chunks = set()
    context_parts = []

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)

        text = doc.page_content.strip().replace("\n", " ")
        context_parts.append(text)

    return "\n\n".join(context_parts)

# =========================
# FINAL DEDUPLICATION
# =========================
def deduplicate_final_docs(docs):
    seen = set()
    unique = []

    for doc in docs:
        cid = doc.metadata.get("chunk_id")

        if cid not in seen:
            seen.add(cid)
            unique.append(doc)

    return unique


# =========================
# GENERATE FINAL ANSWER
# =========================
def generate_answer(query: str, context: str) -> str:
    llm = load_generator()

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    output = llm(prompt)[0]["generated_text"]

    # Clean output (remove prompt if repeated)
    answer = output.split("Answer:")[-1].strip()

    return answer


# =========================
# FULL RAG PIPELINE
# =========================
def run_rag(query: str):
    index_dir = Path("data/vectorstore/faiss_index")

    print("\n🔍 Query:", query)

    # Load
    embeddings = load_embeddings_model()
    vectorstore = load_faiss_index(index_dir, embeddings)

    # Step 1: MultiQuery Expansion
    all_docs = multiquery_retrieve(
        vectorstore=vectorstore,
        query=query,
        mode="mmr",
        k_per_query=2,
        fetch_k=10,
        lambda_mult=0.7,
    )

    print(f"\n📄 Retrieved {len(all_docs)} chunks")

    # Step 3: Rerank + Select
    final_docs = rerank_and_select(
        query=query,
        documents=all_docs,
        rerank_top_k=10,
        final_top_k=3,
    )

    final_docs = deduplicate_final_docs(final_docs)


    print(f"\n✅ Selected {len(final_docs)} best chunks")

    # Step 4: Build context
    context = build_context(final_docs)

    # Step 5: Generate answer
    answer = generate_answer(query, context)

    return answer


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    query = "tell me the skills of the arunsai"

    answer = run_rag(query)

    print("\n" + "=" * 80)
    print("🧠 FINAL ANSWER")
    print("=" * 80)
    print(answer)