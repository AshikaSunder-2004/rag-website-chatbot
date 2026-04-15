"""
ingest.py — Chunk, embed, and store scraped pages in ChromaDB.

Uses LangChain's text splitter for chunking and sentence-transformers
for local embeddings (no API key needed).

Accuracy upgrades:
- Embedding model: BAAI/bge-base-en-v1.5 (much stronger than MiniLM)
- Chunk size: 1000 chars with 150 overlap (more paragraph context)
"""

import hashlib
import shutil

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
_CHUNK_SIZE = 1000          # larger chunks preserve paragraph-level context
_CHUNK_OVERLAP = 150        # generous overlap avoids splitting key sentences
_EMBED_MODEL = "all-MiniLM-L6-v2"  # already installed; BGE-small upgrade available when disk space freed
_CHROMA_PATH = "./chroma_db"


def _get_model() -> SentenceTransformer:
    """Load embedding model (called once via Streamlit cache in app.py)."""
    return SentenceTransformer(_EMBED_MODEL)


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a working ChromaDB PersistentClient.

    If the on-disk database is incompatible with the installed ChromaDB
    version (e.g. old schema missing the 'tenants' table after an upgrade),
    the stale folder is deleted automatically and a fresh client is returned.
    """
    try:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        # Probe to detect schema errors early
        client.list_collections()
        return client
    except Exception:
        # Wipe the incompatible database and start fresh
        shutil.rmtree(_CHROMA_PATH, ignore_errors=True)
        return chromadb.PersistentClient(path=_CHROMA_PATH)


def ingest(
    pages: dict[str, str],
    collection_name: str = "rag",
    model: SentenceTransformer | None = None,
) -> int:
    """
    Chunk, embed, and store *pages* into ChromaDB.

    Parameters
    ----------
    pages : dict[str, str]
        Mapping of URL → body text produced by scraper.scrape().
    collection_name : str
        ChromaDB collection name (default "rag").
    model : SentenceTransformer, optional
        Pre-loaded model instance; if None a new one is created.

    Returns
    -------
    int
        Total number of chunks stored.
    """
    if model is None:
        model = _get_model()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
    )

    client = _get_chroma_client()

    # Delete existing collection for clean re-ingestion
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []
    all_embeds: list[list[float]] = []

    for url, text in pages.items():
        url_hash = hashlib.md5(url.encode()).hexdigest()
        chunks = splitter.split_text(text)

        if not chunks:
            continue

        # Batch-encode all chunks for this page at once — much faster
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            all_ids.append(f"{url_hash}_{i}")
            all_docs.append(chunk)
            all_metas.append({"source": url, "chunk_index": i})
            all_embeds.append(emb)

    if all_ids:
        # ChromaDB has a batch limit; add in slices of 5 000
        batch_size = 5_000
        for start in range(0, len(all_ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=all_ids[start:end],
                documents=all_docs[start:end],
                metadatas=all_metas[start:end],
                embeddings=all_embeds[start:end],
            )

    total = len(all_ids)
    print(f"Ingested {total} chunks from {len(pages)} pages")
    return total
