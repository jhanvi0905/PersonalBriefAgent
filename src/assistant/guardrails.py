from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from assistant.config import (
    ITEM_SUMMARY_CHARS,
    MAX_BRIEF_ITEMS,
    MAX_CANDIDATES_TO_LLM,
    NEWS_WINDOW_HOURS,
)
from assistant.models import (
    BriefItem,
    ComposePack,
    ItemCard,
    MemoryView,
    RankedItem,
    Source,
)

_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior) (instructions|prompts)|system prompt)",
    re.I,
)


def sanitize(text: str, limit: int = ITEM_SUMMARY_CHARS) -> str:
    cleaned = _INJECTION.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def item_id(source: str, native: str) -> str:
    return f"{source}:{native}"


def in_window(ts: datetime, as_of: datetime, hours: int = NEWS_WINDOW_HOURS) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return as_of - timedelta(hours=hours) <= ts <= as_of + timedelta(hours=1)


def rule_filter(items: list[BriefItem], memory: MemoryView, as_of: datetime) -> list[BriefItem]:
    seen = set(memory.seen_ids)
    handled = set(memory.handled_ids)
    mutes = {m.lower() for m in memory.policy.mute_senders}
    out: list[BriefItem] = []
    for item in items:
        if item.id in handled or item.id in seen:
            continue
        blob = f"{item.title} {item.summary}".lower()
        if any(m and m in blob for m in mutes):
            continue
        if item.source == Source.news and not in_window(item.timestamp, as_of):
            continue
        item.seen = item.id in seen
        item.handled = item.id in handled
        out.append(item)
    return out


def salience(item: BriefItem) -> tuple:
    return (
        len(item.urgency_signals),
        item.timestamp.timestamp(),
    )


def cap_for_llm(items: list[BriefItem], limit: int = MAX_CANDIDATES_TO_LLM) -> list[BriefItem]:
    return sorted(items, key=salience, reverse=True)[:limit]


def to_card(item: BriefItem) -> ItemCard:
    return ItemCard(
        id=item.id,
        source=item.source.value,
        when=item.timestamp.strftime("%Y-%m-%d %H:%M"),
        signals=",".join(item.urgency_signals) or "-",
        text=sanitize(f"{item.title}. {item.summary}"),
    )


def pack_prioritize(items: list[BriefItem]) -> list[ItemCard]:
    return [to_card(i) for i in cap_for_llm(items)]


def pack_compose(
    items: list[BriefItem],
    ranked: list[RankedItem],
    memory: MemoryView,
) -> ComposePack:
    by_id = {i.id: i for i in items}
    winners: list[ItemCard] = []
    reasons: dict[str, str] = {}
    included = [r for r in ranked if r.include]
    included.sort(key=lambda r: r.rank)
    for row in included[: memory.policy.max_items]:
        item = by_id.get(row.item_id)
        if not item:
            continue
        winners.append(to_card(item))
        reasons[item.id] = row.reason
    return ComposePack(
        profile=memory.profile,
        open_loops=[loop.text for loop in memory.open_loops],
        last_headlines=memory.last_brief_digest.headlines,
        followups=memory.last_brief_digest.promised_followups,
        winners=winners,
        reasons=reasons,
    )


def fallback_rank(items: list[BriefItem], max_items: int = MAX_BRIEF_ITEMS) -> list[RankedItem]:
    ranked: list[RankedItem] = []
    for i, item in enumerate(sorted(items, key=salience, reverse=True), start=1):
        ranked.append(
            RankedItem(
                item_id=item.id,
                score=max(0.1, 1.0 - (i - 1) * 0.05),
                rank=i,
                reason="rule-order fallback",
                include=i <= max_items,
            )
        )
    return ranked
