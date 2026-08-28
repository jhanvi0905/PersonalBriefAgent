from __future__ import annotations

from datetime import datetime, timedelta

from assistant.guardrails import item_id
from assistant.models import BriefItem, Source


def sample_emails(as_of: datetime) -> list[BriefItem]:
    return [
        BriefItem(
            id=item_id("email", "msg-cfo"),
            source=Source.email,
            title="Q3 numbers from CFO",
            summary="Please review the attached deck before 6pm.",
            timestamp=as_of - timedelta(hours=2),
            urgency_signals=["vip", "due_today"],
        ),
        BriefItem(
            id=item_id("email", "msg-newsletter"),
            source=Source.email,
            title="Weekly industry roundup",
            summary="Skimmable newsletter, no action.",
            timestamp=as_of - timedelta(hours=5),
        ),
    ]
