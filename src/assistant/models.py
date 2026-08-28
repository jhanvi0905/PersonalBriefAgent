from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Source(str, Enum):
    email = "email"
    calendar = "calendar"
    news = "news"


class BriefItem(BaseModel):
    id: str
    source: Source
    title: str
    summary: str
    timestamp: datetime
    url: str | None = None
    urgency_signals: list[str] = Field(default_factory=list)
    seen: bool = False
    handled: bool = False


class ItemCard(BaseModel):
    """Compact, untrusted card for the prioritizer."""

    id: str
    source: str
    when: str
    signals: str
    text: str


class RankedItem(BaseModel):
    item_id: str
    score: float = Field(ge=0, le=1)
    rank: int
    reason: str
    include: bool = True


class RankedList(BaseModel):
    items: list[RankedItem]


class PolicyBlock(BaseModel):
    timezone: str = "America/New_York"
    vip_senders: list[str] = Field(default_factory=list)
    mute_senders: list[str] = Field(default_factory=list)
    max_items: int = 12


class Loop(BaseModel):
    item_id: str | None = None
    text: str
    since: str | None = None


class BriefDigest(BaseModel):
    date: str | None = None
    headlines: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    promised_followups: list[str] = Field(default_factory=list)


class MemoryView(BaseModel):
    policy: PolicyBlock = Field(default_factory=PolicyBlock)
    profile: str = ""
    open_loops: list[Loop] = Field(default_factory=list)
    last_brief_digest: BriefDigest = Field(default_factory=BriefDigest)
    seen_ids: list[str] = Field(default_factory=list)
    handled_ids: list[str] = Field(default_factory=list)


class BriefSection(BaseModel):
    title: str
    bullets: list[str]


class BriefLink(BaseModel):
    title: str
    url: str


class BriefDocument(BaseModel):
    brief_id: str
    generated_at: datetime
    headline: str  # Donna's greeting line
    sections: list[BriefSection]  # ordered: Needs you, Good to know, In AI news
    item_ids: list[str]
    model: str
    status: Literal["ok", "partial", "failed"] = "ok"
    links: list[BriefLink] = Field(default_factory=list)
    signoff: str = ""


class ComposePack(BaseModel):
    owner: str = ""
    profile: str = ""
    open_loops: list[str] = Field(default_factory=list)
    last_headlines: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    winners: list[ItemCard] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    links: list[BriefLink] = Field(default_factory=list)
