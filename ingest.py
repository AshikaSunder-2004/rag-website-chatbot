import os
import re
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "website_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Remove lines with fewer than 5 words and strip excessive whitespace."""
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped.split()) >= 5:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def load_and_chunk_urls(urls: list[str]) -> list[dict]:
    """Load URLs, clean text, and split into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
        except Exception as e:
            print(f"[WARNING] Failed to load {url}: {e}")
            continue

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        splits = splitter.split_documents(docs)
        for idx, split in enumerate(splits):
            all_chunks.append({
                "text": split.page_content,
                "source": url,
                "chunk_index": idx,
                "word_count": len(split.page_content.split()),
            })

    return all_chunks


def ingest(urls: list[str]) -> int:
    """
    Ingest documents from URLs into ChromaDB with cosine similarity.

    Returns:
        int: Total number of chunks stored.
    """
    print(f"[INFO] Loading and chunking {len(urls)} URL(s)...")
    chunks = load_and_chunk_urls(urls)

    if not chunks:
        print("[ERROR] No chunks produced. Aborting ingestion.")
        return 0

    print(f"[INFO] Total chunks to embed: {len(chunks)}")

    # Load embedding model
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Batch encode all chunks in a single call
    texts = [c["text"] for c in chunks]
    print("[INFO] Encoding chunks (batch)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # Required for cosine similarity with BGE
    )

    # Set up ChromaDB with cosine similarity
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists to avoid duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data for upsert
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "word_count": c["word_count"],
        }
        for c in chunks
    ]

    # Insert in batches to avoid memory issues
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"[INFO] Stored chunks {start}–{end - 1}.")

    total_stored = collection.count()
    print(f"[SUCCESS] Ingestion complete. Total chunks stored: {total_stored}")
    return total_stored


if __name__ == "__main__":
    # Replace with your target URLs
    TARGET_URLS = [
        "https://example.com",
    ]
    total = ingest(TARGET_URLS)
    print(f"[DONE] {total} chunks in ChromaDB.")
