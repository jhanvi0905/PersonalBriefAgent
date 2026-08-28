from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

import feedparser
import httpx

from assistant.config import AI_KEYWORDS, MAX_RAW_PER_SOURCE, NEWS_FEEDS
from assistant.guardrails import item_id, sanitize
from assistant.models import BriefItem, Source

_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)
_TAG = re.compile(r"<[^>]+>")


def looks_like_ai(text: str) -> bool:
    blob = text.lower()
    return any(k in blob for k in AI_KEYWORDS)


def _ts(entry: dict, fallback: datetime) -> datetime:
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (TypeError, ValueError):
            continue
    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct:
        return datetime(*parsed_struct[:6], tzinfo=timezone.utc)
    return fallback


def parse_rss(body: str, source_id: str, as_of: datetime, *, ai_only: bool) -> list[BriefItem]:
    parsed = feedparser.parse(body)
    items: list[BriefItem] = []
    for entry in parsed.entries[:MAX_RAW_PER_SOURCE]:
        title = sanitize(entry.get("title") or "Untitled", limit=200)
        summary = sanitize(_TAG.sub(" ", html.unescape(entry.get("summary") or "")))
        link = entry.get("link") or ""
        if ai_only and not looks_like_ai(f"{title} {summary} {link}"):
            continue
        if not link:
            continue
        items.append(
            BriefItem(
                id=item_id("news", link),
                source=Source.news,
                title=f"[{source_id}] {title}",
                summary=summary or title,
                timestamp=_ts(entry, as_of),
                url=link,
            )
        )
    return items


def parse_html_listing(
    body: str,
    page_url: str,
    source_id: str,
    link_prefix: str,
    as_of: datetime,
) -> list[BriefItem]:
    items: list[BriefItem] = []
    seen: set[str] = set()
    for href in _HREF.findall(body):
        url = urljoin(page_url, html.unescape(href)).split("#")[0].rstrip("/")
        if not url.startswith(link_prefix.rstrip("/")):
            continue
        if url.rstrip("/") == link_prefix.rstrip("/"):
            continue
        path = urlparse(url).path.strip("/")
        if path.count("/") < 1:
            continue
        if url in seen:
            continue
        seen.add(url)
        slug = path.split("/")[-1].replace("-", " ")
        items.append(
            BriefItem(
                id=item_id("news", url),
                source=Source.news,
                title=f"[{source_id}] {sanitize(slug, 200)}",
                summary=sanitize(slug),
                timestamp=as_of,
                url=url,
            )
        )
        if len(items) >= MAX_RAW_PER_SOURCE:
            break
    return items


def fetch_feed(client: httpx.Client, spec: dict, as_of: datetime) -> list[BriefItem]:
    resp = client.get(spec["url"], follow_redirects=True, timeout=15)
    resp.raise_for_status()
    if spec["kind"] == "rss":
        return parse_rss(
            resp.text,
            spec["id"],
            as_of,
            ai_only=spec.get("ai_only") == "true",
        )
    return parse_html_listing(
        resp.text,
        spec["url"],
        spec["id"],
        spec["link_prefix"],
        as_of,
    )


def fetch_news(
    as_of: datetime,
    *,
    client: httpx.Client | None = None,
    feeds: tuple[dict[str, str], ...] | None = None,
) -> tuple[list[BriefItem], list[str]]:
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": "personal-brief/0.1"})
    items: list[BriefItem] = []
    skipped: list[str] = []
    try:
        for spec in feeds or NEWS_FEEDS:
            try:
                items.extend(fetch_feed(client, spec, as_of))
            except Exception as exc:  # noqa: BLE001 — per-source isolation
                skipped.append(f"news:{spec['id']}:{exc.__class__.__name__}")
    finally:
        if own:
            client.close()
    return items, skipped
