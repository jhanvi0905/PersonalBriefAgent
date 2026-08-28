from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from assistant.models import BriefDocument, ItemCard, MemoryView, RankedItem


def merge_by_source(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


def add_list(left: list | None, right: list | None) -> list:
    return [*(left or []), *(right or [])]


@dataclass
class RuntimeCtx:
    user_id: str
    request_id: str
    as_of: datetime
    model: str = "grok-3-mini"


class BriefState(TypedDict, total=False):
    source_results: Annotated[dict[str, Any], merge_by_source]
    items: list[dict]
    candidates: list[dict]
    discarded: list[dict]
    ranked: list[dict]
    memory_view: dict
    prioritize_pack: list[dict]
    compose_pack: dict
    brief: dict | None
    status: Literal["ok", "partial", "failed"]
    errors: Annotated[list[str], add_list]
    skipped: Annotated[list[str], add_list]


class BriefOutput(TypedDict, total=False):
    brief: dict | None
    status: str
    errors: list[str]


def memory_from_state(state: BriefState) -> MemoryView:
    return MemoryView.model_validate(state.get("memory_view") or {})


def cards_from_state(state: BriefState) -> list[ItemCard]:
    return [ItemCard.model_validate(c) for c in state.get("prioritize_pack") or []]


def ranked_from_state(state: BriefState) -> list[RankedItem]:
    return [RankedItem.model_validate(r) for r in state.get("ranked") or []]


def brief_from_state(state: BriefState) -> BriefDocument | None:
    raw = state.get("brief")
    return BriefDocument.model_validate(raw) if raw else None
