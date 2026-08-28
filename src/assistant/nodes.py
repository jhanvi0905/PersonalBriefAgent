from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.runtime import Runtime

from assistant.config import get_settings
from assistant.guardrails import pack_compose, pack_prioritize, rule_filter
from assistant.llm import BriefLLM, describe_llm
from assistant.logs import logger


def _log(step: str, msg: str) -> None:
    logger.info("%s: %s", step, msg)
from assistant.memory import load_memory_view, persist_brief, seed_defaults
from assistant.models import BriefItem
from assistant.sources.calendar import fetch_upcoming_events
from assistant.sources.email import fetch_recent_emails
from assistant.sources.google import CredentialsMissing, load_credentials
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


GoogleFetch = Callable[[datetime, object], list[BriefItem]]


def _google_source(key: str, fetch: GoogleFetch, as_of: datetime) -> dict:
    settings = get_settings()
    try:
        creds = load_credentials(
            Path(settings.google_client_secrets), Path(settings.google_token_file)
        )
        items = fetch(as_of, creds)
        _log(key, f"{len(items)} item(s) from Google ({settings.google_token_file})")
        return {
            "source_results": {
                key: {"ok": True, "items": [i.model_dump(mode="json") for i in items]}
            }
        }
    except CredentialsMissing as exc:
        _log(key, f"skipped — {exc} (source will be empty)")
        return {"source_results": {key: {"ok": True, "items": []}}}
    except Exception as exc:  # noqa: BLE001 — per-source isolation
        _log(key, f"skipped — {exc.__class__.__name__}: {exc}")
        return {
            "source_results": {key: {"ok": False, "items": []}},
            "skipped": [key],
            "errors": [f"{key}:{exc.__class__.__name__}"],
            "status": "partial",
        }


def fetch_emails(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    return _google_source("email", fetch_recent_emails, _as_of(runtime))


def fetch_events(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    return _google_source("calendar", fetch_upcoming_events, _as_of(runtime))


def fetch_ai_news(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    try:
        items, skipped = fetch_news(_as_of(runtime))
        _log("news", f"{len(items)} item(s) from RSS/HTML feeds"
             + (f", skipped {skipped}" if skipped else ""))
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
        _log("news", f"failed — {exc.__class__.__name__}: {exc}")
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
    by_src: dict[str, int] = {}
    for i in items:
        by_src[i.source.value] = by_src.get(i.source.value, 0) + 1
    _log("normalize", f"{len(items)} item(s) merged "
         + ", ".join(f"{v} {k}" for k, v in sorted(by_src.items())))
    return {"items": [i.model_dump(mode="json") for i in items], "source_results": {}}


def apply_rules(state: BriefState, runtime: Runtime[RuntimeCtx]) -> dict:
    from assistant.guardrails import in_window
    from assistant.models import Source

    memory = memory_from_state(state)
    as_of = _as_of(runtime)
    items = [BriefItem.model_validate(r) for r in state.get("items") or []]
    kept = rule_filter(items, memory, as_of)

    already = set(memory.seen_ids) | set(memory.handled_ids)
    n_seen = sum(1 for i in items if i.id in already)
    n_stale = sum(
        1 for i in items
        if i.source == Source.news and i.id not in already and not in_window(i.timestamp, as_of)
    )
    _log("rule_filter", f"kept {len(kept)}/{len(items)} "
         f"(dropped {n_seen} already briefed, {n_stale} news older than 48h, "
         f"{len(items) - len(kept) - n_seen - n_stale} muted/other)")
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
            _log("prioritize", f"ranked {len(items)} candidate(s) via {describe_llm(llm)}")
            return {"ranked": [r.model_dump() for r in ranked]}
        except Exception as exc:  # noqa: BLE001
            from assistant.guardrails import fallback_rank

            _log("prioritize", f"{describe_llm(llm)} failed ({exc.__class__.__name__}); "
                 "using heuristic rule-order fallback")
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
            _log("compose", f"wrote brief via {describe_llm(llm)}")
            if status == "partial":
                brief.status = "partial"
            return {"brief": brief.model_dump(mode="json"), "status": brief.status}
        except Exception as exc:  # noqa: BLE001
            from assistant.llm import HeuristicLLM

            _log("compose", f"{describe_llm(llm)} failed ({exc.__class__.__name__}); "
                 "using heuristic composer")
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
