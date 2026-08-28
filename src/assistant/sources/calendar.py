from __future__ import annotations

from datetime import datetime, timedelta, timezone

from assistant.guardrails import item_id
from assistant.models import BriefItem, Source


def sample_events(as_of: datetime) -> list[BriefItem]:
    start = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    return [
        BriefItem(
            id=item_id("calendar", "evt-standup"),
            source=Source.calendar,
            title="Team standup",
            summary="30 min. Camera optional.",
            timestamp=start + timedelta(hours=1),
            urgency_signals=["today"],
        ),
    ]
