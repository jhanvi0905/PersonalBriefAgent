"""Donna — the brief as a web page.

`brief-web` -> http://127.0.0.1:8765 : the page loads the cached brief from
the last run (`/api/last`, never triggers work). The graph runs once per day
at/after BRIEF_MORNING_HOUR (scheduler thread), or on demand via `/api/run`
(SSE, node-by-node). Same graph, store, and .env as the CLI.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from assistant.cli import _hydrate, _snapshot
from assistant.config import get_settings
from assistant.graph import build_graph
from assistant.llm import HeuristicLLM, build_llm, describe_llm
from assistant.logs import configure_logging, logger
from assistant.memory import empty_store
from assistant.state import RuntimeCtx

NODES = [
    "load_memory", "fetch_emails", "fetch_events", "fetch_news", "normalize",
    "rule_filter", "pack_prioritize", "prioritize", "pack_compose", "compose", "persist",
]

# Named phases over the raw nodes, for the explainer.
STAGES = [
    {"id": "memory", "name": "Memory",
     "blurb": "Load prefs, the seen-ledger, and yesterday's digest from the store.",
     "nodes": ["load_memory"]},
    {"id": "collect", "name": "Collect",
     "blurb": "Fan out: Gmail, Calendar, and the news feeds run in parallel.",
     "nodes": ["fetch_emails", "fetch_events", "fetch_news"]},
    {"id": "filter", "name": "Filter",
     "blurb": "Merge sources, then drop what's already briefed, muted, or stale.",
     "nodes": ["normalize", "rule_filter"]},
    {"id": "prioritize", "name": "Prioritize",
     "blurb": "Pack compact cards; the model ranks them and marks what belongs.",
     "nodes": ["pack_prioritize", "prioritize"]},
    {"id": "compose", "name": "Compose",
     "blurb": "Pack the winners; the model writes the brief in themed sections.",
     "nodes": ["pack_compose", "compose"]},
    {"id": "save", "name": "Save",
     "blurb": "Persist the brief and add its items to the seen-ledger.",
     "nodes": ["persist"]},
]

_INDEX = Path(__file__).parent / "web" / "index.html"
app = FastAPI(title="Personal Brief")

# One graph run at a time. A dropped EventSource auto-reconnects, and each
# reconnect would otherwise start a fresh run — i.e. spend LLM credits.
_run_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX)


@app.get("/api/graph")
def graph_spec() -> dict:
    """The compiled LangGraph — structure only, no LLM, no run."""
    g = build_graph(HeuristicLLM()).get_graph()
    stage_of = {n: s for s in STAGES for n in s["nodes"]}
    return {
        "nodes": [
            {"id": n, "stage": stage_of.get(n, {}).get("id", "")}
            for n in g.nodes
        ],
        "edges": [[e.source, e.target] for e in g.edges],
        "stages": STAGES,
        "mermaid": g.draw_mermaid(),
    }


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class _QueueHandler(logging.Handler):
    def __init__(self, sink: "queue.Queue[str]") -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.put(self.format(record))


def _cache_path() -> Path:
    return Path(get_settings().data_dir) / "brief_cache.json"


def load_cache() -> dict | None:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _execute() -> "object":
    """Run the graph once. Yields (event, data) tuples, writes data/brief_cache.json.

    Shared by the SSE endpoint and the morning scheduler. One run at a time.
    """
    if not _run_lock.acquire(blocking=False):
        yield ("error", {"message": "a run is already in progress"})
        return
    handler = None
    try:
        load_dotenv()
        configure_logging()
        settings = get_settings()
        data_dir = Path(settings.data_dir)
        store = empty_store()
        _hydrate(store, settings.brief_user_id, data_dir)
        llm = build_llm()
        graph = build_graph(llm, store=store)

        logs: "queue.Queue[str]" = queue.Queue()
        handler = _QueueHandler(logs)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        def drain():
            while not logs.empty():
                yield ("log", logs.get())

        as_of = datetime.now(timezone.utc)
        config = {
            "nodes": NODES,
            "llm": describe_llm(llm),
            "xai_api_key": bool(settings.xai_api_key),
            "xai_api_base": settings.xai_api_base or "https://api.x.ai/v1",
            "google_token": Path(settings.google_token_file).exists(),
        }
        yield ("config", config)

        request_id = str(uuid4())
        ctx = RuntimeCtx(
            user_id=settings.brief_user_id, request_id=request_id,
            as_of=as_of, model=settings.xai_model,
        )
        steps: list[dict] = []
        for chunk in graph.stream(
            {}, config={"configurable": {"thread_id": request_id}},
            context=ctx, stream_mode="updates",
        ):
            yield from drain()
            for node, delta in chunk.items():
                steps.append({"node": node, "delta": delta})
                yield ("step", {"node": node, "delta": delta})
        yield from drain()
        _snapshot(store, settings.brief_user_id, data_dir)

        brief = next(
            (s["delta"]["brief"] for s in reversed(steps)
             if s["delta"] and s["delta"].get("brief")),
            None,
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(json.dumps({
            "date": _local_now().date().isoformat(),
            "generated_at": as_of.isoformat(),
            "config": config, "steps": steps, "brief": brief,
        }, default=str, indent=2))
        yield ("done", {})
    except Exception as exc:  # noqa: BLE001 — surface it in the UI
        logger.exception("run failed")
        yield ("error", {"message": f"{exc.__class__.__name__}: {exc}"})
    finally:
        if handler is not None:
            logger.removeHandler(handler)
        _run_lock.release()


def _run() -> "object":
    for event, data in _execute():
        yield _sse(event, data)


def _replay(path: Path) -> "object":
    """Re-emit a captured SSE file (set BRIEF_REPLAY=run.sse) — UI work, no credits."""
    frame: list[str] = []
    for line in path.read_text().splitlines(keepends=True):
        frame.append(line)
        if line.strip() == "":
            yield "".join(frame)
            frame = []
            time.sleep(0.3)
    if frame:
        yield "".join(frame)


@app.get("/api/run")
def run() -> StreamingResponse:
    replay = os.environ.get("BRIEF_REPLAY")
    if replay and Path(replay).exists():
        return StreamingResponse(_replay(Path(replay)), media_type="text/event-stream")
    return StreamingResponse(_run(), media_type="text/event-stream")


@app.get("/api/last")
def last() -> dict:
    """The cached brief from the last run. The page loads this; it never
    triggers a run on its own."""
    replay = os.environ.get("BRIEF_REPLAY")
    if replay and Path(replay).exists():
        return _cache_from_sse(Path(replay))
    return load_cache() or {}


def _cache_from_sse(path: Path) -> dict:
    steps, config, brief = [], {}, None
    event = None
    for line in path.read_text().splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            if event == "config":
                config = data
            elif event == "step":
                steps.append(data)
                if (data.get("delta") or {}).get("brief"):
                    brief = data["delta"]["brief"]
    return {"date": _local_now().date().isoformat(), "generated_at": None,
            "config": config, "steps": steps, "brief": brief}


def _morning_run_if_due() -> None:
    """Run once per day, on or after the configured morning hour."""
    if os.environ.get("BRIEF_REPLAY"):
        return
    hour = get_settings().brief_morning_hour
    cache = load_cache()
    today = _local_now().date().isoformat()
    if _local_now().hour >= hour and (cache or {}).get("date") != today:
        logger.info("scheduler: producing this morning's brief")
        for _ in _execute():
            pass


def _scheduler() -> None:
    while True:
        try:
            _morning_run_if_due()
        except Exception:  # noqa: BLE001
            logger.exception("scheduled run failed")
        time.sleep(1800)


def main() -> None:
    import uvicorn

    load_dotenv()
    configure_logging()
    threading.Thread(target=_scheduler, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
