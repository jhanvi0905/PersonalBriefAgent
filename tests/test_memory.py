from datetime import datetime, timezone

from assistant.memory import empty_store, load_memory_view, persist_brief, seed_defaults
from assistant.models import BriefDocument, BriefSection


def test_memory_roundtrip_seen_and_digest():
    store = empty_store()
    seed_defaults(store, "u1")
    as_of = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    brief = BriefDocument(
        brief_id="b1",
        generated_at=as_of,
        headline="Focus on the CFO deck",
        sections=[BriefSection(title="Now", bullets=["Review Q3"])],
        item_ids=["email:msg-cfo"],
        model="test",
        status="ok",
    )
    persist_brief(store, "u1", brief, ["email:msg-cfo"])
    view = load_memory_view(store, "u1")
    assert "email:msg-cfo" in view.seen_ids
    assert view.last_brief_digest.headlines == ["Focus on the CFO deck"]
    assert view.policy.max_items == 12
