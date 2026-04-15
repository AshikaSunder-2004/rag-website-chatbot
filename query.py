"""
query.py — Embed a user question, retrieve relevant chunks from ChromaDB,
rerank with a CrossEncoder, and stream an answer from Groq (Llama 3.1 70B).

Accuracy upgrades over v1:
- Embedding model: BAAI/bge-base-en-v1.5 (matches ingest.py)
- Retrieval: fetch top-20, then rerank with cross-encoder/ms-marco-MiniLM-L-6-v2
  and keep only the top-5 most relevant passages before sending to the LLM.
- LLM: llama-3.1-70b-versatile (stronger reasoning & synthesis than 8B)
- System prompt: more explicit grounding + citation instructions
- BGE prefix: "Represent this sentence for searching relevant passages: "
  boosts recall for BGE retrieval models.
"""

import os
import shutil

import chromadb
from dotenv import load_dotenv
from groq import Groq
import groq as groq_module
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
_EMBED_MODEL       = "all-MiniLM-L6-v2"          # already installed; upgrade to BGE-small when disk freed
_RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CHROMA_PATH       = "./chroma_db"
_GROQ_MODEL        = "llama-3.1-70b-versatile"   # upgraded from 8B
_FETCH_K           = 20   # retrieve more candidates for the reranker to score
_TOP_K             = 5    # final passages sent to the LLM after reranking

# BGE prefix (keep for future BGE upgrade): prepend this to queries when using BGE family models
_BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "

_SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. Your task is to answer the user's "
    "question EXCLUSIVELY using the context passages provided below.\n"
    "Rules:\n"
    "1. Base every statement on the provided context. Do NOT use prior knowledge.\n"
    "2. If the answer is not clearly supported by the context, respond with: "
    "\"I don't have enough information from the scraped content to answer that.\"\n"
    "3. Be concise and direct. Avoid unnecessary filler or repetition.\n"
    "4. When relevant, quote or paraphrase specific details from the context."
)

# Cosine-distance threshold: ChromaDB returns *distance* (lower = better).
# distance = 1 − cosine_similarity → distance > 0.7 ≈ similarity < 0.3
_LOW_CONFIDENCE_DIST = 0.7


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a working ChromaDB PersistentClient, auto-recovering from
    incompatible on-disk schemas (e.g., 'no such table: tenants').
    """
    try:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        client.list_collections()
        return client
    except Exception:
        shutil.rmtree(_CHROMA_PATH, ignore_errors=True)
        return chromadb.PersistentClient(path=_CHROMA_PATH)


# ── Lazy-loaded reranker (shared across calls in same process) ────────────────
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANK_MODEL)
    return _reranker


def answer_stream(
    question: str,
    collection_name: str = "rag",
    top_k: int = _TOP_K,
    model: SentenceTransformer | None = None,
):
    """
    Generator that yields answer tokens one-by-one (for st.write_stream).

    Yields
    ------
    str
        Individual text deltas from Groq's streaming response.

    The final item yielded is a dict with metadata:
        {"sources": [...], "low_confidence": bool}
    so the caller can extract sources after streaming completes.
    """
    chunks, sources, low_conf = _retrieve(question, collection_name, top_k, model)

    if not chunks:
        yield "No relevant content found. Please ingest a website first."
        yield {"sources": [], "low_confidence": False}
        return

    # Build the context block from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        yield "❌ GROQ_API_KEY not found. Please add it to your `.env` file."
        yield {"sources": sources, "low_confidence": low_conf}
        return

    client = Groq(api_key=api_key)

    # Prepend low-confidence warning before the LLM response
    if low_conf:
        yield "⚠️ **Low confidence:** "

    try:
        stream = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            stream=True,
            max_tokens=1024,
            temperature=0.1,   # lower = more faithful to context
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except groq_module.APIError as exc:
        yield f"\n\n❌ Groq API error: {exc.message}"
    except Exception as exc:
        yield f"\n\n❌ Unexpected error: {exc}"

    # Sentinel dict so the caller can extract sources + metadata
    yield {"sources": sources, "low_confidence": low_conf}


def answer(
    question: str,
    collection_name: str = "rag",
    top_k: int = _TOP_K,
    model: SentenceTransformer | None = None,
) -> tuple[str, list[str]]:
    """
    Non-streaming answer — kept for convenience / testing.

    Returns
    -------
    tuple[str, list[str]]
        (answer_text, deduplicated_source_urls)
    """
    chunks, sources, low_conf = _retrieve(question, collection_name, top_k, model)
    if not chunks:
        return ("No relevant content found. Please ingest a website first.", [])

    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    text = response.choices[0].message.content
    if low_conf:
        text = "⚠️ **Low confidence:** " + text
    return (text, sources)


# ── Private helpers ──────────────────────────────────────────────────────────

def _retrieve(
    question: str,
    collection_name: str,
    top_k: int,
    model: SentenceTransformer | None,
) -> tuple[list[str], list[str], bool]:
    """
    Embed the question → vector-search ChromaDB (top 20) → CrossEncoder rerank
    → return the best top_k passages.

    Returns (chunks, deduplicated_sources, low_confidence_flag).
    """
    if model is None:
        model = SentenceTransformer(_EMBED_MODEL)

    # Use BGE prefix only when using BGE family models
    prefixed_question = question  # switch to: _BGE_QUERY_PREFIX + question when using BGE
    q_embedding = model.encode([prefixed_question], show_progress_bar=False).tolist()

    client = _get_chroma_client()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist yet (first run before any ingestion)
        return ([], [], False)

    # Fetch a larger candidate pool for the reranker to work with
    fetch_k = max(_FETCH_K, top_k * 4)
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(fetch_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs  = results.get("documents",  [[]])[0]
    metas = results.get("metadatas",  [[]])[0]
    dists = results.get("distances",  [[]])[0]

    if not docs:
        return ([], [], False)

    # ── CrossEncoder reranking ────────────────────────────────────────────────
    reranker = _get_reranker()
    pairs = [(question, doc) for doc in docs]
    scores = reranker.predict(pairs)        # higher = more relevant

    # Sort by descending score and keep the best top_k
    ranked = sorted(zip(scores, docs, metas, dists), key=lambda x: x[0], reverse=True)
    ranked = ranked[:top_k]

    top_docs  = [r[1] for r in ranked]
    top_metas = [r[2] for r in ranked]
    top_dists = [r[3] for r in ranked]

    # Deduplicate sources while preserving order
    seen: set[str] = set()
    sources: list[str] = []
    for m in top_metas:
        src = m.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    # Low confidence if the *best* (post-rerank) chunk's original vector distance
    # was already far from the query (reranker can't conjure relevance from noise)
    low_conf = top_dists[0] > _LOW_CONFIDENCE_DIST if top_dists else False

    return (top_docs, sources, low_conf)
