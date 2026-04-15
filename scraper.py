"""
scraper.py — BFS web crawler for the RAG chatbot.

Crawls a start URL and all same-domain links up to max_pages,
returning a dict of {url: cleaned_text}.
"""

from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# Extensions to skip — binary/non-HTML resources
_SKIP_EXT = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".css", ".js", ".xml", ".zip", ".tar", ".gz", ".mp4",
    ".mp3", ".wav", ".webp", ".woff", ".woff2", ".ttf", ".eot",
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Bot/1.0)"}

# Tags whose content is boilerplate / non-informational
_BOILERPLATE_TAGS = ["nav", "footer", "header", "script", "style", "noscript", "aside"]


def _normalize_url(url: str) -> str:
    """Lowercase scheme+host, strip fragment (#) and trailing slash for dedup."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def _should_skip(url: str) -> bool:
    """Return True if the URL points to a non-HTML resource."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _SKIP_EXT)


def _extract_text(html: str) -> str:
    """Parse HTML, strip boilerplate tags, return clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _extract_links(html: str, base_url: str, target_domain: str) -> list[str]:
    """Return a list of normalized same-domain links found in the page."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full = urljoin(base_url, href)
        if urlparse(full).netloc.lower() == target_domain and not _should_skip(full):
            links.append(_normalize_url(full))
    return links


def scrape(
    start_url: str,
    max_pages: int = 50,
    progress_callback=None,
) -> dict[str, str]:
    """
    BFS-crawl *start_url* (same domain only).

    Parameters
    ----------
    start_url : str
        The seed URL to begin crawling.
    max_pages : int
        Maximum number of pages to fetch (default 50).
    progress_callback : callable, optional
        Called with (current_count, max_pages, url) after each successful page.

    Returns
    -------
    dict[str, str]
        Mapping of URL → cleaned body text (pages with <100 chars are dropped).
    """
    start_url = _normalize_url(start_url)
    target_domain = urlparse(start_url).netloc.lower()

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    pages: dict[str, str] = {}

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            resp.raise_for_status()
            # Only process HTML responses
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
        except Exception:
            # Network / HTTP errors — skip silently
            continue

        text = _extract_text(resp.text)

        # Skip thin pages (< 100 chars of real content)
        if len(text) < 100:
            continue

        pages[url] = text
        count = len(pages)

        if progress_callback:
            progress_callback(count, max_pages, url)
        print(f"Scraped {count}/{max_pages}: {url}")

        # Discover new links for BFS frontier
        for link in _extract_links(resp.text, url, target_domain):
            if link not in visited:
                queue.append(link)

    return pages
