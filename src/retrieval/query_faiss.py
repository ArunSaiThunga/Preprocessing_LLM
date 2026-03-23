from __future__ import annotations

from pathlib import Path
from typing import Literal, List, Set

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


SearchMode = Literal["similarity", "mmr"]


# =========================
# LOAD MODELS
# =========================
def load_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_faiss_index(index_dir: str | Path, embeddings) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# =========================
# BASE RETRIEVAL
# =========================
def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    mode: SearchMode = "mmr",
    k: int = 4,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
) -> list[Document]:

    if mode == "similarity":
        return vectorstore.similarity_search(query, k=k)

    if mode == "mmr":
        return vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    raise ValueError(f"Unsupported retrieval mode: {mode}")


# =========================
# MULTIQUERY (EMBEDDING EXPANSION)
# =========================
def expand_query(query: str) -> List[str]:
    """
    Simple semantic expansion without LLM
    """
    variations = [
        query,
        query + " title",
        query + " topic",
        query + " subject",
        query.replace("thesis", "research work"),
        query.replace("title", "name"),
        "title of the thesis",
        "thesis topic",
    ]

    return list(set(variations))


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen: Set[str] = set()
    unique_docs = []

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")

        key = chunk_id or (
            doc.metadata.get("source_id"),
            doc.metadata.get("page_number"),
            doc.page_content[:100],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def multiquery_retrieve(
    vectorstore: FAISS,
    query: str,
    mode: SearchMode = "mmr",
    k_per_query: int = 2,
    fetch_k: int = 10,
    lambda_mult: float = 0.7,
) -> List[Document]:

    queries = expand_query(query)

    print("\nExpanded Queries:")
    for q in queries:
        print("-", q)

    all_docs = []

    for q in queries:
        docs = retrieve_documents(
            vectorstore=vectorstore,
            query=q,
            mode=mode,
            k=k_per_query,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        all_docs.extend(docs)

    return deduplicate_docs(all_docs)


# =========================
# PRINTING
# =========================
def print_results(results: list[Document], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Metadata:", doc.metadata)
        print("Text:", doc.page_content[:500].replace("\n", " "))
        print()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    index_dir = Path("data/vectorstore/faiss_index")

    query = "What are the technical skills of arunsai?"

    embeddings = load_embeddings_model()
    vectorstore = load_faiss_index(index_dir, embeddings)

    # 🔹 1. Similarity Search
    similarity_results = retrieve_documents(
        vectorstore=vectorstore,
        query=query,
        mode="similarity",
        k=3,
    )

    # 🔹 2. MMR Search
    mmr_results = retrieve_documents(
        vectorstore=vectorstore,
        query=query,
        mode="mmr",
        k=3,
        fetch_k=8,
        lambda_mult=0.9,
    )

    # 🔹 3. MultiQuery + MMR (BEST)
    multiquery_results = multiquery_retrieve(
        vectorstore=vectorstore,
        query=query,
        mode="mmr",
        k_per_query=2,
        fetch_k=10,
        lambda_mult=0.8,
    )

    print_results(similarity_results, "SIMILARITY SEARCH RESULTS")
    print_results(mmr_results, "MMR SEARCH RESULTS")
    print_results(multiquery_results, "MULTIQUERY + MMR RESULTS")