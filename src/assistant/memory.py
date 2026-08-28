from __future__ import annotations

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from assistant.models import BriefDigest, BriefDocument, Loop, MemoryView, PolicyBlock

_PREFS = "prefs"
_LEDGER = "ledger"
_DIGEST = "digest"
_LOOPS = "loops"
_LATEST = "latest_brief"


def empty_store() -> InMemoryStore:
    return InMemoryStore()


def load_memory_view(store: BaseStore, user_id: str) -> MemoryView:
    prefs = _val(store, (_PREFS,), user_id) or {}
    ledger = _val(store, (_LEDGER,), user_id) or {}
    digest = _val(store, (_DIGEST,), user_id) or {}
    loops = _val(store, (_LOOPS,), user_id) or []
    return MemoryView(
        policy=PolicyBlock.model_validate(prefs.get("policy") or {}),
        profile=prefs.get("profile") or "",
        open_loops=[Loop.model_validate(x) for x in loops],
        last_brief_digest=BriefDigest.model_validate(digest),
        seen_ids=list(ledger.get("seen_ids") or []),
        handled_ids=list(ledger.get("handled_ids") or []),
    )


def persist_brief(store: BaseStore, user_id: str, brief: BriefDocument, seen_ids: list[str]) -> None:
    store.put((_LATEST,), user_id, brief.model_dump(mode="json"))
    store.put(
        (_DIGEST,),
        user_id,
        BriefDigest(
            date=brief.generated_at.date().isoformat(),
            headlines=[brief.headline],
            item_ids=brief.item_ids,
            promised_followups=[],
        ).model_dump(),
    )
    prev = _val(store, (_LEDGER,), user_id) or {}
    merged = sorted(set(prev.get("seen_ids") or []) | set(seen_ids) | set(brief.item_ids))
    store.put(
        (_LEDGER,),
        user_id,
        {"seen_ids": merged, "handled_ids": list(prev.get("handled_ids") or [])},
    )


def seed_defaults(store: BaseStore, user_id: str) -> None:
    if store.get((_PREFS,), user_id) is None:
        store.put(
            (_PREFS,),
            user_id,
            {"policy": PolicyBlock().model_dump(), "profile": ""},
        )


def hydrate_from_view(store: BaseStore, user_id: str, view: MemoryView) -> None:
    store.put(
        (_PREFS,),
        user_id,
        {"policy": view.policy.model_dump(), "profile": view.profile},
    )
    store.put(
        (_LEDGER,),
        user_id,
        {"seen_ids": view.seen_ids, "handled_ids": view.handled_ids},
    )
    store.put((_DIGEST,), user_id, view.last_brief_digest.model_dump())
    store.put((_LOOPS,), user_id, [loop.model_dump() for loop in view.open_loops])


def _val(store: BaseStore, ns: tuple[str, ...], key: str):
    item = store.get(ns, key)
    return item.value if item else None
