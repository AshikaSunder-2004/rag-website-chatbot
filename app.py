"""
app.py — Streamlit UI for the RAG-Powered Website Chatbot.

Run with:  streamlit run app.py
"""

import shutil
import time

import torch
import streamlit as st
from sentence_transformers import SentenceTransformer

from scraper import scrape
from ingest import ingest
from query import answer_stream

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _load_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

embed_model = _load_model()

# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "messages": [],
    "ingested": False,
    "pages_scraped": 0,
    "chunks_stored": 0,
    "scraped_urls": [],
    "collection_name": "rag",
    "last_token_count": 0,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #E2E8F0;
}

.stApp {
    background-color: #070D1A;
    background-image: radial-gradient(ellipse 70% 55% at 65% 38%, rgba(30,64,175,0.22) 0%, transparent 68%);
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 5rem;
    max-width: 100%;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0D1526 !important;
    border-right: 1px solid rgba(59,130,246,0.12) !important;
}
[data-testid="stSidebar"] .stMarkdown p { margin: 0; }
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

.sidebar-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.sidebar-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(59,130,246,0.4);
}
.sidebar-title { font-size: 17px; font-weight: 700; color: #F0F6FF; letter-spacing: -0.01em; }
.sidebar-subtitle { font-size: 12px; color: #4B6FA8; padding-left: 52px; margin-bottom: 2px; }

.label-upper {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #38BDF8; margin-bottom: 6px;
}

hr { border: none; border-top: 1px solid rgba(59,130,246,0.12); margin: 14px 0; }

.stat-row { display: flex; justify-content: space-between; align-items: baseline; padding: 3px 0; }
.stat-row .stat-label { font-size: 12px; color: #4B6FA8; }
.stat-row .stat-value { font-size: 13px; font-weight: 600; color: #60A5FA; }

.pages-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.pages-badge {
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.28);
    color: #93C5FD; font-size: 12px; font-weight: 600;
    padding: 2px 10px; border-radius: 6px;
}

/* Primary button */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: transparent !important;
    color: #93C5FD !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 16px !important; width: 100% !important;
    box-shadow: none !important; transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: rgba(59,130,246,0.1) !important;
    border-color: rgba(59,130,246,0.55) !important;
    color: #BFDBFE !important;
}

/* Secondary button */
div[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important; color: #EF4444 !important;
    border: none !important; font-size: 12px !important;
    padding: 4px 0 !important; box-shadow: none !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover { color: #FCA5A5 !important; }

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 5px rgba(59,130,246,0.2) !important;
    width: 18px !important; height: 18px !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
    font-size: 14px !important; color: #0F172A !important;
    padding: 10px 14px !important; box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #94A3B8 !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important; border-radius: 12px !important;
    font-size: 15px !important; color: #0F172A !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
    padding: 14px 18px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15), 0 4px 24px rgba(0,0,0,0.35) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background-color: #1E40AF !important; border-radius: 8px !important;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }
[data-testid="stChatMessage"] {
    background-color: transparent !important; border: none !important; padding: 0 !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(59,130,246,0.15) !important;
    background: rgba(13,21,38,0.8) !important;
    border-radius: 8px !important; box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important; color: #60A5FA !important; font-weight: 500 !important;
}

[data-testid="stSpinner"] { color: #60A5FA; font-size: 13px; }
[data-testid="stProgress"] > div > div > div { background-color: #3B82F6 !important; }

/* Top status bar */
.top-bar {
    display: flex; justify-content: flex-end; align-items: center;
    gap: 8px; padding: 14px 0 16px;
    border-bottom: 1px solid rgba(59,130,246,0.1); margin-bottom: 0;
}
.top-pill {
    font-size: 12px; font-weight: 500; color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.32); border-radius: 20px;
    padding: 5px 14px; background: rgba(59,130,246,0.06);
}

/* User bubble */
.msg-user-wrap { display: flex; flex-direction: column; align-items: flex-end; margin-bottom: 18px; }
.msg-user-sender { font-size: 11px; color: #4B6FA8; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.msg-user-bubble {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    color: #FFFFFF; border-radius: 14px 14px 3px 14px;
    padding: 11px 15px; max-width: 75%; font-size: 14px;
    line-height: 1.65; word-wrap: break-word;
    box-shadow: 0 4px 16px rgba(29,78,216,0.35);
}

/* Assistant bubble */
.msg-asst-wrap { display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 24px; max-width: 88%; }
.msg-asst-sender {
    font-size: 11px; color: #38BDF8; margin-bottom: 6px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
    display: flex; align-items: center; gap: 6px;
}
.msg-asst-sender::before {
    content: ''; display: inline-block; width: 5px; height: 5px;
    border-radius: 50%; background-color: #38BDF8; box-shadow: 0 0 6px #38BDF8;
}
.msg-asst-body {
    font-size: 15px; line-height: 1.8; color: #CBD5E1;
    background: rgba(15,25,50,0.75); border: 1px solid rgba(59,130,246,0.15);
    border-radius: 3px 14px 14px 14px; padding: 14px 18px; width: 100%;
    backdrop-filter: blur(8px);
}
.msg-asst-meta { margin-top: 7px; display: flex; justify-content: space-between; align-items: center; width: 100%; flex-wrap: wrap; gap: 4px; }
.msg-asst-sources { font-size: 11px; color: #4B6FA8; flex: 1; min-width: 0; }
.msg-asst-sources a { color: #60A5FA; text-decoration: underline; text-decoration-color: rgba(96,165,250,0.3); font-size: 11px; }
.msg-asst-sources a:hover { color: #93C5FD; }
.msg-asst-timing {
    font-size: 11px; color: #4B6FA8; white-space: nowrap; flex-shrink: 0;
    background: rgba(59,130,246,0.08); padding: 2px 8px;
    border-radius: 12px; border: 1px solid rgba(59,130,246,0.15);
}

.low-conf { color: #FBBF24; font-weight: 500; }

/* Empty state */
.empty-state { text-align: center; padding: 60px 20px 40px; }
.empty-label {
    display: flex; align-items: center; justify-content: center;
    gap: 10px; margin-bottom: 22px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #38BDF8;
}
.empty-label::before, .empty-label::after {
    content: ''; flex: 1; max-width: 60px;
    height: 1px; background: rgba(56,189,248,0.35);
}
.empty-state h2 {
    font-size: 44px; font-weight: 800; color: #F0F6FF;
    margin-bottom: 14px; letter-spacing: -0.03em; line-height: 1.1;
}
.empty-state h2 span { color: #3B82F6; }
.empty-state > p { font-size: 16px; color: #4B6FA8; line-height: 1.65; max-width: 480px; margin: 0 auto; }
.empty-steps { display: flex; justify-content: center; gap: 12px; margin-top: 36px; flex-wrap: wrap; }
.empty-step {
    display: flex; align-items: center; gap: 12px;
    background: rgba(13,21,38,0.85); border: 1px solid rgba(59,130,246,0.18);
    border-radius: 12px; padding: 14px 20px; min-width: 140px; text-align: left;
    backdrop-filter: blur(8px);
}
.step-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4);
    color: #60A5FA; font-size: 13px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-text { font-size: 13px; color: #93C5FD; font-weight: 500; line-height: 1.4; }

/* Sidebar status */
.sidebar-status { display: flex; align-items: flex-start; gap: 8px; padding: 14px 0 4px; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22C55E; box-shadow: 0 0 8px rgba(34,197,94,0.6);
    flex-shrink: 0; margin-top: 3px;
}
.status-text { font-size: 11px; color: #4B6FA8; line-height: 1.55; }
.status-text strong { color: #93C5FD; font-weight: 500; display: block; }

/* Footer */
.chat-footer { text-align: center; font-size: 12px; color: #2D4A7A; margin-top: 8px; }
.chat-footer span { color: #3B82F6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">👤</div>
        <span class="sidebar-title">RAG Chatbot</span>
    </div>
    <p class="sidebar-subtitle">Semantic search &amp; AI-powered Q&amp;A</p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # URL input
    st.markdown('<p class="label-upper">Website URL</p>', unsafe_allow_html=True)
    url_input = st.text_input(
        label="url_input",
        label_visibility="collapsed",
        placeholder="https://example.com",
    )

    # Pages slider
    page_val = st.session_state.get("_slider_val", 50)
    st.markdown(
        f'<div class="pages-row">'
        f'<p class="label-upper" style="margin:0;">Pages</p>'
        f'<span class="pages-badge">{page_val}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    max_pages = st.slider(
        label="max_pages_slider",
        label_visibility="collapsed",
        min_value=10, max_value=100, value=50, step=10,
    )
    st.session_state["_slider_val"] = max_pages

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    scrape_btn = st.button("⬇  Scrape & Ingest", type="primary", use_container_width=True)

    # Post-ingestion stats
    if st.session_state["ingested"]:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stat-row">
                <span class="stat-label">Pages scraped</span>
                <span class="stat-value">{st.session_state["pages_scraped"]}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Chunks stored</span>
                <span class="stat-value">{st.session_state["chunks_stored"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state["scraped_urls"]:
            with st.expander(f"View scraped pages ({len(st.session_state['scraped_urls'])})"):
                for u in st.session_state["scraped_urls"]:
                    st.markdown(
                        f'<a href="{u}" target="_blank" style="font-size:11px;'
                        f'color:#4B6FA8;text-decoration:underline;">{u}</a><br>',
                        unsafe_allow_html=True,
                    )

    # Last token count
    if st.session_state["last_token_count"] > 0:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-row">'
            f'<span class="stat-label">Last response</span>'
            f'<span class="stat-value">{st.session_state["last_token_count"]} tokens</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Spacer + clear button
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("Clear and Reset", type="secondary"):
        shutil.rmtree("./chroma_db", ignore_errors=True)
        for key, val in _DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

    # Model status at bottom
    docs_count = st.session_state["chunks_stored"]
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="status-dot"></div>
            <div class="status-text">
                <strong>Model ready · Groq + LLaMA 3.1 70B</strong>
                CrossEncoder Reranking enabled
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SCRAPE & INGEST LOGIC
# ═════════════════════════════════════════════════════════════════════════════
if scrape_btn:
    if not url_input.strip():
        st.sidebar.markdown(
            '<p style="font-size:12px;color:#EF4444;">Please enter a URL.</p>',
            unsafe_allow_html=True,
        )
    else:
        progress_bar = st.sidebar.progress(0)
        status_ph = st.sidebar.empty()

        def _progress(current: int, total: int, url: str):
            progress_bar.progress(min(current / total, 1.0))
            status_ph.markdown(
                f'<p style="font-size:11px;color:#4B6FA8;margin:2px 0;">'
                f'{current}/{total} pages scraped</p>',
                unsafe_allow_html=True,
            )

        with st.spinner("Scraping website..."):
            pages = scrape(url_input.strip(), max_pages=max_pages, progress_callback=_progress)

        progress_bar.empty()
        status_ph.empty()

        if not pages:
            st.sidebar.markdown(
                '<p style="font-size:12px;color:#EF4444;">No pages found. Check the URL.</p>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Chunking and embedding content..."):
                chunk_count = ingest(pages, model=embed_model)

            st.session_state["ingested"] = True
            st.session_state["pages_scraped"] = len(pages)
            st.session_state["chunks_stored"] = chunk_count
            st.session_state["scraped_urls"] = list(pages.keys())
            st.session_state["messages"] = []
            st.session_state["last_token_count"] = 0
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════

# Top status bar
docs_indexed = st.session_state["chunks_stored"]
st.markdown(
    f"""
    <div class="top-bar">
        <span class="top-pill">CrossEncoder Reranking</span>
        <span class="top-pill">{docs_indexed} docs indexed</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state["ingested"] and not st.session_state["messages"]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-label">AI-Powered Research</div>
            <h2>Chat with <span>any website</span></h2>
            <p>Paste a URL in the sidebar to scrape, embed, and start asking questions in seconds.</p>
            <div class="empty-steps">
                <div class="empty-step">
                    <div class="step-num">1</div>
                    <span class="step-text">Paste a URL</span>
                </div>
                <div class="empty-step">
                    <div class="step-num">2</div>
                    <span class="step-text">Scrape &amp; embed</span>
                </div>
                <div class="empty-step">
                    <div class="step-num">3</div>
                    <span class="step-text">Ask anything</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f"""
            <div class="msg-user-wrap">
                <span class="msg-user-sender">You</span>
                <div class="msg-user-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        sources = msg.get("sources", [])
        resp_time = msg.get("response_time")
        token_count = msg.get("token_count")

        if sources:
            links = " &middot; ".join(
                f'<a href="{s}" target="_blank">{s}</a>' for s in sources
            )
            sources_html = f'<span>Sources: {links}</span>'
        else:
            sources_html = ""

        timing_parts = []
        if resp_time is not None:
            timing_parts.append(f"{resp_time:.1f}s")
        if token_count:
            timing_parts.append(f"{token_count} tokens")
        timing_html = (
            f'<span class="msg-asst-timing">{" &nbsp; ".join(timing_parts)}</span>'
            if timing_parts else ""
        )

        meta_html = (
            f'<div class="msg-asst-meta">'
            f'<span class="msg-asst-sources">{sources_html}</span>'
            f'{timing_html}</div>'
        ) if sources_html or timing_html else ""

        st.markdown(
            f"""
            <div class="msg-asst-wrap">
                <span class="msg-asst-sender">Assistant</span>
                <div class="msg-asst-body">{content}</div>
                {meta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    _render_message(msg)


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about this website...")

st.markdown(
    '<div class="chat-footer">Powered by <span>Groq + LLaMA 3.1 70B</span> &nbsp;·&nbsp; <span>CrossEncoder Reranking</span></div>',
    unsafe_allow_html=True,
)

if prompt:
    if not st.session_state["ingested"]:
        st.markdown(
            '<p style="font-size:13px;color:#FBBF24;margin:8px 0;">'
            'Please ingest a URL first using the sidebar.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.markdown(
            f"""
            <div class="msg-user-wrap">
                <span class="msg-user-sender">You</span>
                <div class="msg-user-bubble">{prompt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        t_start = time.time()
        stream = answer_stream(prompt, chat_history=st.session_state["messages"], model=embed_model)
        collected: list[str] = []
        sources: list[str] = []

        def _token_gen():
            for token in stream:
                if isinstance(token, dict):
                    sources.extend(token.get("sources", []))
                else:
                    collected.append(token)
                    yield token

        st.markdown('<span class="msg-asst-sender">Assistant</span>', unsafe_allow_html=True)
        st.write_stream(_token_gen())

        t_elapsed = time.time() - t_start
        full_answer = "".join(collected)
        token_est = len(full_answer.split())

        if sources:
            links = " &middot; ".join(
                f'<a href="{s}" target="_blank">{s}</a>' for s in sources
            )
            sources_html = f'<span>Sources: {links}</span>'
        else:
            sources_html = ""

        timing_html = f'<span class="msg-asst-timing">{t_elapsed:.1f}s &nbsp; {token_est} tokens</span>'

        if sources_html or timing_html:
            st.markdown(
                f"""
                <div class="msg-asst-meta">
                    <span class="msg-asst-sources">{sources_html}</span>
                    {timing_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.session_state["last_token_count"] = token_est
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "response_time": t_elapsed,
            "token_count": token_est,
        })
