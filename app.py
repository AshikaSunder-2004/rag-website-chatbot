"""
app.py — Streamlit UI for the RAG-Powered Website Chatbot.

Run with:  streamlit run app.py
"""

import shutil
import time

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
    return SentenceTransformer("all-MiniLM-L6-v2")

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
/* ── Reset & base ─────────────────────────────────── */
#MainMenu       { visibility: hidden; }
footer          { visibility: hidden; }
header          { visibility: hidden; }

/* Remove default Streamlit top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 860px;
}

/* Page background */
.stApp {
    background-color: #FAFAFA;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #F4F4F5;
    border-right: 1px solid #E4E4E7;
}

[data-testid="stSidebar"] .stMarkdown p {
    margin: 0;
}

/* ── Global text defaults ─────────────────────────── */
html, body, [class*="css"] {
    color: #18181B;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif;
}

/* ── Labels ───────────────────────────────────────── */
.label-upper {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #71717A;
    margin-bottom: 4px;
}

/* ── Sidebar stat rows ────────────────────────────── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 3px 0;
}
.stat-row .stat-label {
    font-size: 12px;
    color: #71717A;
}
.stat-row .stat-value {
    font-size: 13px;
    font-weight: 500;
    color: #18181B;
}

/* ── Primary button (Scrape) ──────────────────────── */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #18181B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #27272A !important;
}

/* ── Secondary / reset button ─────────────────────── */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: transparent !important;
    color: #DC2626 !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 4px 0 !important;
    box-shadow: none !important;
    text-align: left !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    color: #B91C1C !important;
    background-color: transparent !important;
}

/* ── Slider ───────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #18181B !important;
    border-color: #18181B !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: #71717A;
    font-size: 11px;
}

/* ── Text input ───────────────────────────────────── */
[data-testid="stTextInput"] input {
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    color: #18181B !important;
    background-color: #FFFFFF !important;
    padding: 8px 10px !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #A1A1AA !important;
    box-shadow: none !important;
}

/* ── Chat input ───────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border: 1px solid #E4E4E7 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    background-color: #FFFFFF !important;
    color: #18181B !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #A1A1AA !important;
    box-shadow: none !important;
}

/* Hide default chat avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* Remove green highlight from st.chat_message */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Expander (scraped pages) ─────────────────────── */
[data-testid="stExpander"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #71717A !important;
    font-weight: 400 !important;
}

/* ── Divider ──────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #E4E4E7;
    margin: 10px 0;
}

/* ── Spinner ──────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #71717A;
    font-size: 13px;
}

/* ── Progress bar ─────────────────────────────────── */
[data-testid="stProgress"] > div > div > div {
    background-color: #18181B !important;
}

/* ── User message bubble ──────────────────────────── */
.msg-user-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin-bottom: 20px;
}
.msg-user-sender {
    font-size: 11px;
    color: #71717A;
    margin-bottom: 4px;
}
.msg-user-bubble {
    background-color: #18181B;
    color: #FFFFFF;
    border-radius: 12px;
    padding: 10px 14px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
}

/* ── Assistant message ────────────────────────────── */
.msg-asst-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 24px;
    max-width: 85%;
}
.msg-asst-sender {
    font-size: 11px;
    color: #71717A;
    margin-bottom: 6px;
}
.msg-asst-body {
    font-size: 15px;
    line-height: 1.75;
    color: #18181B;
    font-weight: 400;
}
.msg-asst-meta {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #E4E4E7;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    flex-wrap: wrap;
    gap: 4px;
}
.msg-asst-sources {
    font-size: 11px;
    color: #71717A;
    flex: 1;
    min-width: 0;
}
.msg-asst-sources a {
    color: #71717A;
    text-decoration: underline;
    text-decoration-color: #D4D4D8;
    font-size: 11px;
}
.msg-asst-sources a:hover {
    color: #18181B;
    text-decoration-color: #71717A;
}
.msg-asst-timing {
    font-size: 11px;
    color: #A1A1AA;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Low confidence text ──────────────────────────── */
.low-conf {
    color: #B45309;
    font-weight: 500;
}

/* ── Empty state ──────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 80px 20px;
}
.empty-state h2 {
    font-size: 28px;
    font-weight: 600;
    color: #18181B;
    margin-bottom: 10px;
}
.empty-state p {
    font-size: 15px;
    color: #71717A;
    margin: 0;
}

/* ── Sidebar page title ───────────────────────────── */
.sidebar-title {
    font-size: 16px;
    font-weight: 600;
    color: #18181B;
    margin-bottom: 2px;
}
.sidebar-subtitle {
    font-size: 13px;
    color: #71717A;
    margin-bottom: 0;
}

/* ── Footer hint ──────────────────────────────────── */
.chat-footer {
    text-align: center;
    font-size: 11px;
    color: #A1A1AA;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">Paste a URL, scrape it, ask questions.</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # URL input
    st.markdown('<p class="label-upper">Website URL</p>', unsafe_allow_html=True)
    url_input = st.text_input(
        label="url_input",
        label_visibility="collapsed",
        placeholder="https://example.com",
    )

    # Pages slider — label + current value on same row
    page_val = st.session_state.get("_slider_val", 50)
    cols_lbl = st.columns([1, 1])
    with cols_lbl[0]:
        st.markdown('<p class="label-upper" style="margin-top:10px;">Pages</p>', unsafe_allow_html=True)
    with cols_lbl[1]:
        st.markdown(
            f'<p style="font-size:12px;font-weight:500;color:#18181B;'
            f'text-align:right;margin-top:10px;">{page_val}</p>',
            unsafe_allow_html=True,
        )
    max_pages = st.slider(
        label="max_pages_slider",
        label_visibility="collapsed",
        min_value=10, max_value=100, value=50, step=10,
    )
    # Keep the displayed label value in sync
    st.session_state["_slider_val"] = max_pages

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    scrape_btn = st.button("Scrape and Ingest", type="primary", use_container_width=True)

    # ── Post-ingestion stats ──────────────────────────────────────────────────
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
                        f'color:#71717A;text-decoration:underline;">{u}</a><br>',
                        unsafe_allow_html=True,
                    )

    # ── Last response token count ─────────────────────────────────────────────
    if st.session_state["last_token_count"] > 0:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<p class="label-upper">Last response tokens</p>'
            f'<p style="font-size:14px;font-weight:500;color:#18181B;margin:0;">'
            f'{st.session_state["last_token_count"]}</p>',
            unsafe_allow_html=True,
        )

    # ── Reset — pushed to bottom ──────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Clear and Reset", type="secondary"):
        shutil.rmtree("./chroma_db", ignore_errors=True)
        for key, val in _DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SCRAPE & INGEST LOGIC
# ═════════════════════════════════════════════════════════════════════════════
if scrape_btn:
    if not url_input.strip():
        st.sidebar.markdown(
            '<p style="font-size:12px;color:#DC2626;">Please enter a URL.</p>',
            unsafe_allow_html=True,
        )
    else:
        progress_bar = st.sidebar.progress(0)
        status_ph = st.sidebar.empty()

        def _progress(current: int, total: int, url: str):
            progress_bar.progress(min(current / total, 1.0))
            status_ph.markdown(
                f'<p style="font-size:11px;color:#71717A;margin:2px 0;">'
                f'{current}/{total} pages scraped</p>',
                unsafe_allow_html=True,
            )

        with st.spinner("Scraping website..."):
            pages = scrape(url_input.strip(), max_pages=max_pages, progress_callback=_progress)

        progress_bar.empty()
        status_ph.empty()

        if not pages:
            st.sidebar.markdown(
                '<p style="font-size:12px;color:#DC2626;">No pages found. Check the URL.</p>',
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

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state["ingested"] and not st.session_state["messages"]:
    st.markdown(
        """
        <div class="empty-state">
            <h2>Chat with any website</h2>
            <p>Enter a URL in the sidebar to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_message(msg: dict) -> None:
    """Render a single chat message with the custom HTML layout."""
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        # Right-aligned dark bubble
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
        # Assistant — plain text, left-aligned, with metadata below
        sources = msg.get("sources", [])
        resp_time = msg.get("response_time")
        token_count = msg.get("token_count")

        # Build source links string
        if sources:
            links = " &middot; ".join(
                f'<a href="{s}" target="_blank">{s}</a>' for s in sources
            )
            sources_html = f'<span>Sources: {links}</span>'
        else:
            sources_html = ""

        # Timing string
        timing_parts = []
        if resp_time is not None:
            timing_parts.append(f"{resp_time:.1f}s")
        if token_count:
            timing_parts.append(f"{token_count} tokens")
        timing_html = (
            f'<span class="msg-asst-timing">{" &nbsp; ".join(timing_parts)}</span>'
            if timing_parts else ""
        )

        # Meta row (only shown if there's something to show)
        if sources_html or timing_html:
            meta_html = f"""
            <div class="msg-asst-meta">
                <span class="msg-asst-sources">{sources_html}</span>
                {timing_html}
            </div>
            """
        else:
            meta_html = ""

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
    '<div class="chat-footer">Powered by Groq + LLaMA 3.1 70B &nbsp;·&nbsp; CrossEncoder Reranking</div>',
    unsafe_allow_html=True,
)

if prompt:
    if not st.session_state["ingested"]:
        st.markdown(
            '<p style="font-size:13px;color:#B45309;margin:8px 0;">'
            'Please ingest a URL first using the sidebar.</p>',
            unsafe_allow_html=True,
        )
    else:
        # Append and render user message immediately
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

        # Stream assistant response using st.write_stream in a temporary block
        t_start = time.time()
        stream = answer_stream(prompt, model=embed_model)
        collected: list[str] = []
        sources: list[str] = []

        def _token_gen():
            for token in stream:
                if isinstance(token, dict):
                    sources.extend(token.get("sources", []))
                else:
                    # Strip the raw low-confidence prefix so we handle it ourselves
                    collected.append(token)
                    yield token

        # Render "Assistant" label then stream tokens below it
        st.markdown('<span class="msg-asst-sender">Assistant</span>', unsafe_allow_html=True)
        st.write_stream(_token_gen())

        t_elapsed = time.time() - t_start
        full_answer = "".join(collected)
        token_est = len(full_answer.split())

        # Show sources + timing under the streamed answer
        if sources:
            links = " &middot; ".join(
                f'<a href="{s}" target="_blank">{s}</a>' for s in sources
            )
            sources_html = f'<span>Sources: {links}</span>'
        else:
            sources_html = ""

        timing_html = (
            f'<span class="msg-asst-timing">{t_elapsed:.1f}s &nbsp; {token_est} tokens</span>'
        )

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
