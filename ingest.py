import torch
import os
import re
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def chunk_pages(pages: dict[str, str]) -> list[dict]:
    """Split pre-scraped pages into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for url, content in pages.items():
        cleaned = clean_text(content)
        splits = splitter.split_text(cleaned)
        for idx, split in enumerate(splits):
            all_chunks.append({
                "text": split,
                "source": url,
                "chunk_index": idx,
                "word_count": len(split.split()),
            })

    return all_chunks


def ingest(pages: dict[str, str], model: SentenceTransformer) -> int:
    """
    Ingest documents into ChromaDB.
    
    Args:
        pages: Dict of {url: text}
        model: Pre-loaded SentenceTransformer model
    """
    print(f"[INFO] Chunking {len(pages)} page(s)...")
    chunks = chunk_pages(pages)

    if not chunks:
        print("[ERROR] No chunks produced. Aborting ingestion.")
        return 0

    print(f"[INFO] Total chunks to embed: {len(chunks)}")

    # Batch encode all chunks
    texts = [c["text"] for c in chunks]
    print("[INFO] Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "word_count": c["word_count"],
        }
        for c in chunks
    ]

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    return collection.count()


if __name__ == "__main__":
    # Simple test URLs
    from sentence_transformers import SentenceTransformer
    test_model = SentenceTransformer(EMBEDDING_MODEL)
    test_pages = {
        "https://example.com": "This is a test page content for the RAG chatbot to ingest."
    }
    total = ingest(test_pages, test_model)
    print(f"[DONE] {total} chunks in ChromaDB.")
