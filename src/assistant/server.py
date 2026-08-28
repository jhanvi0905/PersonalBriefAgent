"""Tiny FastAPI wrapper that streams a brief run to the browser.

`brief-web` -> http://127.0.0.1:8765 : one page that runs the graph, lights
up each DAG node as it finishes, shows the data each node produced, and
tails the `[dag]` log live. Same graph, store, and .env as the CLI.
"""

from __future__ import annotations

import json
import logging
import queue
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from assistant.cli import _hydrate, _snapshot
from assistant.config import get_settings
from assistant.graph import build_graph
from assistant.llm import build_llm, describe_llm
from assistant.logs import configure_logging, logger
from assistant.memory import empty_store
from assistant.state import RuntimeCtx

NODES = [
    "load_memory", "fetch_emails", "fetch_events", "fetch_news", "normalize",
    "rule_filter", "pack_prioritize", "prioritize", "pack_compose", "compose", "persist",
]

_INDEX = Path(__file__).parent / "web" / "index.html"
app = FastAPI(title="Personal Brief")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class _QueueHandler(logging.Handler):
    def __init__(self, sink: "queue.Queue[str]") -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.put(self.format(record))


def _run() -> "object":
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


@app.get("/api/run")
def run() -> StreamingResponse:
    return StreamingResponse(_run(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    load_dotenv()
    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
