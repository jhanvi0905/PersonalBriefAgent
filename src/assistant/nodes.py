from __future__ import annotations

from datetime import datetime, timezone

from langgraph.runtime import Runtime

from assistant.guardrails import pack_compose, pack_prioritize, rule_filter
from assistant.llm import BriefLLM
from assistant.memory import load_memory_view, persist_brief, seed_defaults
from assistant.models import BriefItem
from assistant.sources.calendar import sample_events
from assistant.sources.email import sample_emails
from assistant.sources.news import fetch_news
from assistant.state import BriefState, RuntimeCtx, memory_from_state


def _as_of(runtime: Runtime[RuntimeCtx]) -> datetime:
    ts = runtime.context.as_of
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def load_memory(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    store = runtime.store
    if store is None:
        raise RuntimeError("store required")
    seed_defaults(store, runtime.context.user_id)
    view = load_memory_view(store, runtime.context.user_id)
    return {"memory_view": view.model_dump(mode="json"), "status": "ok"}


def fetch_emails(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    items = sample_emails(_as_of(runtime))
    return {
        "source_results": {
            "email": {"ok": True, "items": [i.model_dump(mode="json") for i in items]}
        }
    }


def fetch_events(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    items = sample_events(_as_of(runtime))
    return {
        "source_results": {
            "calendar": {"ok": True, "items": [i.model_dump(mode="json") for i in items]}
        }
    }


def fetch_ai_news(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    try:
        items, skipped = fetch_news(_as_of(runtime))
        update: dict = {
            "source_results": {
                "news": {
                    "ok": not skipped,
                    "items": [i.model_dump(mode="json") for i in items],
                }
            }
        }
        if skipped:
            update["skipped"] = skipped
            update["errors"] = skipped
            update["status"] = "partial"
        return update
    except Exception as exc:  # noqa: BLE001
        return {
            "source_results": {"news": {"ok": False, "items": []}},
            "skipped": ["news"],
            "errors": [f"news:{exc.__class__.__name__}"],
        }


def normalize(state: BriefState) -> dict:
    results = state.get("source_results") or {}
    items: list[BriefItem] = []
    for payload in results.values():
        for row in payload.get("items") or []:
            items.append(BriefItem.model_validate(row))
    return {"items": [i.model_dump(mode="json") for i in items], "source_results": {}}


def apply_rules(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    memory = memory_from_state(state)
    items = [BriefItem.model_validate(r) for r in state.get("items") or []]
    kept = rule_filter(items, memory, _as_of(runtime))
    return {"candidates": [i.model_dump(mode="json") for i in kept]}


def pack_for_rank(state: BriefState) -> dict:
    items = [BriefItem.model_validate(r) for r in state.get("candidates") or []]
    return {"prioritize_pack": [c.model_dump() for c in pack_prioritize(items)]}


def make_prioritize(llm: BriefLLM):
    def prioritize(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
        items = [BriefItem.model_validate(r) for r in state.get("candidates") or []]
        if not items:
            return {"ranked": [], "status": state.get("status") or "ok"}
        from assistant.models import ItemCard

        cards = [ItemCard.model_validate(c) for c in state.get("prioritize_pack") or []]
        try:
            ranked = llm.prioritize(items, cards)
            return {"ranked": [r.model_dump() for r in ranked]}
        except Exception as exc:  # noqa: BLE001
            from assistant.guardrails import fallback_rank

            ranked = fallback_rank(items)
            return {
                "ranked": [r.model_dump() for r in ranked],
                "status": "partial",
                "skipped": ["llm_prioritizer"],
                "errors": [f"prioritize:{exc.__class__.__name__}"],
            }

    return prioritize


def pack_for_write(state: BriefState) -> dict:
    memory = memory_from_state(state)
    items = [BriefItem.model_validate(r) for r in state.get("candidates") or []]
    from assistant.models import RankedItem

    ranked = [RankedItem.model_validate(r) for r in state.get("ranked") or []]
    pack = pack_compose(items, ranked, memory)
    return {"compose_pack": pack.model_dump()}


def make_compose(llm: BriefLLM):
    def compose(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
        from assistant.models import ComposePack

        pack = ComposePack.model_validate(state.get("compose_pack") or {})
        status = state.get("status") or "ok"
        try:
            brief = llm.compose(pack, _as_of(runtime), runtime.context.model)
            if status == "partial":
                brief.status = "partial"
            return {"brief": brief.model_dump(mode="json"), "status": brief.status}
        except Exception as exc:  # noqa: BLE001
            from assistant.llm import HeuristicLLM

            brief = HeuristicLLM().compose(pack, _as_of(runtime), runtime.context.model)
            brief.status = "partial"
            return {
                "brief": brief.model_dump(mode="json"),
                "status": "partial",
                "skipped": ["llm_compose"],
                "errors": [f"compose:{exc.__class__.__name__}"],
            }

    return compose


def persist(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    from assistant.models import BriefDocument

    raw = state.get("brief")
    if not raw or runtime.store is None:
        return {}
    brief = BriefDocument.model_validate(raw)
    persist_brief(runtime.store, runtime.context.user_id, brief, brief.item_ids)
    return {}
