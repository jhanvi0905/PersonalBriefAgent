from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from assistant.config import get_settings
from assistant.graph import build_graph
from assistant.llm import build_llm, describe_llm
from assistant.logs import configure_logging, logger
from assistant.memory import empty_store, hydrate_from_view, load_memory_view, seed_defaults
from assistant.models import MemoryView
from assistant.state import RuntimeCtx


def _memory_path(data_dir: Path) -> Path:
    return data_dir / "memory.json"


def _hydrate(store, user_id: str, data_dir: Path) -> None:
    path = _memory_path(data_dir)
    if path.exists():
        hydrate_from_view(store, user_id, MemoryView.model_validate_json(path.read_text()))
    else:
        seed_defaults(store, user_id)


def _snapshot(store, user_id: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    view = load_memory_view(store, user_id)
    _memory_path(data_dir).write_text(view.model_dump_json(indent=2))


def run_brief(*, as_of: datetime | None = None) -> dict:
    load_dotenv()
    configure_logging()
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    store = empty_store()
    _hydrate(store, settings.brief_user_id, data_dir)
    llm = build_llm()
    logger.info(
        "config: llm=%s | keys read from .env (XAI_API_KEY %s)",
        describe_llm(llm),
        "set" if settings.xai_api_key else "unset",
    )
    graph = build_graph(llm, store=store)
    as_of = as_of or datetime.now(timezone.utc)
    request_id = str(uuid4())
    result = graph.invoke(
        {},
        config={"configurable": {"thread_id": request_id}},
        context=RuntimeCtx(
            user_id=settings.brief_user_id,
            request_id=request_id,
            as_of=as_of,
            model=settings.xai_model,
        ),
    )
    _snapshot(store, settings.brief_user_id, data_dir)
    if result.get("brief"):
        (data_dir / "latest.json").write_text(json.dumps(result["brief"], indent=2, default=str))
    return result


def render(result: dict) -> str:
    brief = result.get("brief") or {}
    lines = [
        brief.get("headline") or "(no brief)",
        f"status={result.get('status')} model={brief.get('model')}",
        "",
    ]
    for section in brief.get("sections") or []:
        lines.append(f"## {section.get('title')}")
        for bullet in section.get("bullets") or []:
            lines.append(f"- {bullet}")
        lines.append("")
    if result.get("errors"):
        lines.append("notes: " + "; ".join(result["errors"]))
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce the latest personal brief.")
    parser.parse_args()
    print(render(run_brief()), end="")


def authorize() -> None:
    """One-time Google OAuth consent; caches a token for Gmail + Calendar."""
    load_dotenv()
    settings = get_settings()
    from assistant.sources.google import authorize as run_flow

    run_flow(Path(settings.google_client_secrets), Path(settings.google_token_file))
    print(f"Saved Google token to {settings.google_token_file}")


if __name__ == "__main__":
    main()
