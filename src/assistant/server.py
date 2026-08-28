"""Donna — a web explainer for the LangGraph brief workflow.

`brief-web` -> http://127.0.0.1:8765 : shows today's brief as cards, the
actual compiled LangGraph beside it, and streams a run node-by-node over
SSE. Same graph, store, and .env as the CLI.
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


def _run() -> "object":
    if not _run_lock.acquire(blocking=False):
        yield _sse("error", {"message": "a run is already in progress — ignoring duplicate request"})
        return
    try:
        yield from _run_locked()
    finally:
        _run_lock.release()


def _run_locked() -> "object":
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

    def drain() -> "object":
        while not logs.empty():
            yield _sse("log", logs.get())

    try:
        yield _sse("config", {
            "nodes": NODES,
            "llm": describe_llm(llm),
            "xai_api_key": bool(settings.xai_api_key),
            "xai_api_base": settings.xai_api_base or "https://api.x.ai/v1",
            "google_token": (data_dir / Path(settings.google_token_file).name).exists()
            or Path(settings.google_token_file).exists(),
        })
        request_id = str(uuid4())
        ctx = RuntimeCtx(
            user_id=settings.brief_user_id,
            request_id=request_id,
            as_of=datetime.now(timezone.utc),
            model=settings.xai_model,
        )
        for chunk in graph.stream(
            {},
            config={"configurable": {"thread_id": request_id}},
            context=ctx,
            stream_mode="updates",
        ):
            yield from drain()
            for node, delta in chunk.items():
                yield _sse("step", {"node": node, "delta": delta})
        yield from drain()
        _snapshot(store, settings.brief_user_id, data_dir)
        yield _sse("done", {})
    except Exception as exc:  # noqa: BLE001 — surface it in the UI
        logger.exception("run failed")
        yield from drain()
        yield _sse("error", {"message": f"{exc.__class__.__name__}: {exc}"})
    finally:
        logger.removeHandler(handler)


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


def main() -> None:
    import uvicorn

    load_dotenv()
    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
