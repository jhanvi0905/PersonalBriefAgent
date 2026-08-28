from __future__ import annotations

from datetime import datetime, timedelta, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from assistant.config import GMAIL_LOOKBACK_HOURS, MAX_RAW_PER_SOURCE
from assistant.guardrails import item_id, sanitize
from assistant.models import BriefItem, Source

_HEADERS = ["From", "Subject"]


def fetch_recent_emails(as_of: datetime, creds: Credentials) -> list[BriefItem]:
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    after = int((as_of - timedelta(hours=GMAIL_LOOKBACK_HOURS)).timestamp())
    listed = (
        service.users()
        .messages()
        .list(userId="me", q=f"in:inbox after:{after}", maxResults=MAX_RAW_PER_SOURCE)
        .execute()
    )
    items: list[BriefItem] = []
    for ref in listed.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="metadata", metadataHeaders=_HEADERS)
            .execute()
        )
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        labels = msg.get("labelIds", [])
        signals = [s for s, on in (("unread", "UNREAD" in labels),
                                   ("important", "IMPORTANT" in labels)) if on]
        items.append(
            BriefItem(
                id=item_id("email", ref["id"]),
                source=Source.email,
                title=sanitize(headers.get("subject") or "(no subject)", 200),
                summary=sanitize(f"From {headers.get('from', 'unknown')}. {msg.get('snippet', '')}"),
                timestamp=datetime.fromtimestamp(int(msg["internalDate"]) / 1000, tz=timezone.utc),
                urgency_signals=signals,
            )
        )
    return items
