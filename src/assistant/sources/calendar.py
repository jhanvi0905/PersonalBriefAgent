from __future__ import annotations

from datetime import datetime, timedelta, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from assistant.config import CALENDAR_LOOKAHEAD_HOURS, MAX_RAW_PER_SOURCE
from assistant.guardrails import item_id, sanitize
from assistant.models import BriefItem, Source


def _start_ts(start: dict) -> datetime:
    raw = start.get("dateTime") or start.get("date")
    ts = datetime.fromisoformat(raw)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def fetch_upcoming_events(as_of: datetime, creds: Credentials) -> list[BriefItem]:
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    listed = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=as_of.isoformat(),
            timeMax=(as_of + timedelta(hours=CALENDAR_LOOKAHEAD_HOURS)).isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=MAX_RAW_PER_SOURCE,
        )
        .execute()
    )
    items: list[BriefItem] = []
    for event in listed.get("items", []):
        items.append(
            BriefItem(
                id=item_id("calendar", event["id"]),
                source=Source.calendar,
                title=sanitize(event.get("summary") or "(no title)", 200),
                summary=sanitize(event.get("location") or event.get("description") or "No details"),
                timestamp=_start_ts(event.get("start", {})),
                url=event.get("htmlLink"),
                urgency_signals=["today"],
            )
        )
    return items
