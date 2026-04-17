import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "website_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REWRITE_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL = "llama-3.3-70b-versatile"
TOP_K = 8
MAX_HISTORY_MESSAGES = 8

# BGE query prefix (required for retrieval queries with this model)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Confidence thresholds
CONFIDENCE_HIGH = 0.45
CONFIDENCE_MEDIUM = 0.30

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_chroma_collection = None
_groq_client: Groq | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found in environment / .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Query Rewriting ────────────────────────────────────────────────────────────
def rewrite_query(original_query: str, chat_history: list[dict]) -> str:
    """
    Use llama-3.1-8b-instant to rewrite the user query for better retrieval,
    taking conversation history into account.
    """
    history_snippet = ""
    if chat_history:
        recent = chat_history[-MAX_HISTORY_MESSAGES:]
        history_snippet = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )

    system_prompt = (
        "You are a query optimization assistant. "
        "Rewrite the user's query to be more specific and retrieval-friendly, "
        "resolving any pronouns or references using the conversation history. "
        "Output ONLY the rewritten query, nothing else."
    )
    user_prompt = (
        f"Conversation history:\n{history_snippet}\n\n"
        f"Original query: {original_query}\n\n"
        "Rewritten query:"
    ) if history_snippet else f"Original query: {original_query}\n\nRewritten query:"

    groq = _get_groq()
    response = groq.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else original_query


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    """Embed query with BGE prefix and retrieve top-K chunks from ChromaDB."""
    model = _get_embed_model()
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True,
    ).tolist()

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": dist,
        })
    return chunks


# ── Reranking ──────────────────────────────────────────────────────────────────
def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank retrieved chunks using CrossEncoder and attach rerank scores."""
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# ── Confidence ─────────────────────────────────────────────────────────────────
def compute_confidence(reranked_chunks: list[dict]) -> str:
    """Derive confidence level from the top reranked chunk's score."""
    if not reranked_chunks:
        return "low"
    top_score = reranked_chunks[0]["rerank_score"]
    if top_score >= CONFIDENCE_HIGH:
        return "high"
    elif top_score >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


# ── Answer Generation (streaming) ─────────────────────────────────────────────
def generate_answer(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
) -> str:
    """
    Stream an answer from llama-3.3-70b-versatile using retrieved context
    and conversation history. Returns the full assembled answer string.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a helpful, accurate assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Be concise and cite sources where relevant."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Append trimmed conversation history
    if chat_history:
        messages.extend(chat_history[-MAX_HISTORY_MESSAGES:])

    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}"
        ),
    })

    groq = _get_groq()
    stream = groq.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_answer += delta
    print()  # newline after streaming ends

    return full_answer


# ── Public Interface ───────────────────────────────────────────────────────────
def query(
    user_query: str,
    chat_history: list[dict] | None = None,
) -> tuple[str, list[str], str]:
    """
    Full RAG pipeline: rewrite → retrieve → rerank → generate.

    Args:
        user_query:   The user's question.
        chat_history: List of {"role": "user"|"assistant", "content": str} dicts.
                      Pass the last N messages; internally trimmed to 8.

    Returns:
        tuple: (answer_text, source_urls, confidence_level)
               confidence_level ∈ {"high", "medium", "low"}
    """
    if chat_history is None:
        chat_history = []

    # 1. Rewrite query for better retrieval
    rewritten = rewrite_query(user_query, chat_history)
    print(f"[DEBUG] Rewritten query: {rewritten}")

    # 2. Retrieve top-K chunks
    chunks = retrieve(rewritten)
    if not chunks:
        return "I couldn't find relevant information to answer your question.", [], "low"

    # 3. Rerank with CrossEncoder
    reranked = rerank(rewritten, chunks)

    # 4. Compute confidence from reranker scores
    confidence = compute_confidence(reranked)

    # 5. Collect unique source URLs
    source_urls = list(dict.fromkeys(c["source"] for c in reranked if c["source"]))

    # 6. Generate streaming answer using top chunks
    answer = generate_answer(user_query, reranked, chat_history)

    return answer, source_urls, confidence


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history: list[dict] = []

    print("RAG Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        answer, sources, confidence = query(user_input, history)

        # Update conversation memory
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        # Keep only last 8 messages
        history = history[-MAX_HISTORY_MESSAGES:]

        print(f"\n[Confidence: {confidence}]")
        if sources:
            print(f"[Sources]: {', '.join(sources)}")
        print()
