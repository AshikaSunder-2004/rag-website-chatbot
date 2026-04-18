# 🤖 RAG-Powered Website Chatbot

A production-ready chatbot that scrapes any website, builds a local vector knowledge base, and answers your questions using **Retrieval-Augmented Generation (RAG)** — powered by **Groq** (free LLM) and local embeddings.

---

## What It Does

1. **You paste a URL** — the chatbot crawls up to 100 pages on that domain
2. **Content is chunked and embedded** — stored in a local ChromaDB vector database
3. **You ask questions** — the bot retrieves relevant chunks and generates answers using Groq's Llama 3.1 model
4. **Every answer includes sources** — so you can verify the information

No paid API keys for embeddings. Groq's free tier handles the LLM.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         STREAMLIT UI (app.py)                      │
  │   URL Input → Scrape Button → Chat Input → Streamed Answers       │
  └───────┬──────────────────────────────────────┬────────────────────┘
          │                                      │
          ▼                                      ▼
  ┌───────────────┐                     ┌────────────────┐
  │  scraper.py   │                     │   query.py     │
  │  BFS Crawler  │                     │  RAG Pipeline  │
  │               │                     │                │
  │ URL → HTML    │                     │ Question       │
  │ → Clean Text  │                     │   ↓            │
  └───────┬───────┘                     │ Embed (MiniLM) │
          │                             │   ↓            │
          ▼                             │ Search ChromaDB│
  ┌───────────────┐                     │   ↓            │
  │  ingest.py    │                     │ Build Prompt   │
  │               │                     │   ↓            │
  │ Text → Chunks │                     │ Groq LLM      │
  │ → Embeddings  │──────────┐          │   ↓            │
  │ → ChromaDB    │          │          │ Stream Answer  │
  └───────────────┘          │          └───────┬────────┘
                             │                  │
                             ▼                  │
                     ┌──────────────┐           │
                     │   ChromaDB   │◀──────────┘
                     │  (local DB)  │
                     │ ./chroma_db  │
                     └──────────────┘
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher |
| **Groq API key** | Free at [console.groq.com](https://console.groq.com) |
| **Disk space** | ~500 MB for the embedding model (downloaded once) |
| **Internet** | Required for scraping websites and calling Groq API |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## Getting Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account (Google/GitHub SSO available)
3. Navigate to **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key (starts with `gsk_`)
6. Paste it into your `.env` file

Groq's free tier includes **14,400 requests/day** with the `llama-3.1-8b-instant` model — more than enough for personal use.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to Use

1. **Paste a URL** into the sidebar (e.g., `https://docs.python.org/3/tutorial/`)
2. **Adjust max pages** with the slider (default: 50)
3. Click **🚀 Scrape & Ingest** — watch the progress bar as pages are crawled and embedded
4. Once done, the sidebar shows page/chunk stats and a list of scraped URLs
5. **Ask questions** in the chat input — answers stream in real-time with source citations
6. Each answer shows response time and estimated token count
7. Click **🗑️ Clear & Reset** to wipe the database and start fresh

---

## How RAG Works

**RAG (Retrieval-Augmented Generation)** is a technique that combines search with language model generation:

1. **Index** — Website content is split into small chunks (~500 characters each) and converted into numerical vectors (embeddings) that capture meaning. These are stored in a vector database.

2. **Retrieve** — When you ask a question, it's also converted to an embedding. The vector database finds the chunks whose embeddings are closest to your question — these are the most semantically relevant passages.

3. **Generate** — The retrieved chunks are injected into a prompt as "context," and a language model (Llama 3.1 via Groq) generates an answer grounded in that context. This prevents hallucination because the model can only use what was actually on the website.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Scraping | `requests` + `BeautifulSoup4` |
| Text Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, free |
| Vector Database | `ChromaDB` (persistent, local storage) |
| LLM | Groq API (`llama-3.1-8b-instant`) — free tier |
| UI | `Streamlit` (chat interface, sidebar, session state) |
| Config | `python-dotenv` |

---

## File Structure

```
rag-chatbot/
├── app.py              # Streamlit UI — main entry point
├── scraper.py          # BFS web crawler
├── ingest.py           # Chunking + embedding + ChromaDB storage
├── query.py            # Vector search + Groq LLM answer generation
├── requirements.txt    # All dependencies
├── .env.example        # Template for GROQ_API_KEY
├── .gitignore          # Ignore .env, chroma_db/, __pycache__/
└── README.md           # This file
```

---

## Limitations

- **JavaScript-rendered sites** — The scraper uses `requests` (no browser), so SPAs (React, Angular, Vue) that render content via JS will yield little useful text.
- **Rate limiting** — Some websites may block or throttle the crawler. No built-in delay between requests.
- **Groq rate limits** — Free tier allows ~14,400 requests/day. Heavy use may hit limits.
- **Context window** — `llama-3.1-8b-instant` has an 128K context window, but we limit to 5 chunks (~2,500 chars) for speed and relevance.
- **No authentication** — The scraper cannot access pages behind logins or paywalls.
- **Single collection** — Re-ingesting replaces the previous data. One website at a time.
- **CPU embeddings** — Embedding runs on CPU. Large sites (100 pages) may take 1–2 minutes to embed.

---

## License

MIT
