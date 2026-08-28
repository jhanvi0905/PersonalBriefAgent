import json

from fastapi.testclient import TestClient

from assistant import server


def test_run_stream_covers_every_node(monkeypatch):
    monkeypatch.setattr(server, "_hydrate", lambda *a, **k: None)
    monkeypatch.setattr(server, "_snapshot", lambda *a, **k: None)
    monkeypatch.setattr("assistant.nodes.fetch_news", lambda as_of: ([], []))

    lines = TestClient(server.app).get("/api/run").text.splitlines()
    events = [ln.removeprefix("event: ") for ln in lines if ln.startswith("event: ")]
    nodes = {
        json.loads(ln.removeprefix("data: "))["node"]
        for ln in lines
        if ln.startswith("data: ") and '"node"' in ln
    }

    assert events[0] == "config"
    assert events[-1] == "done"
    assert {"load_memory", "normalize", "compose", "persist"} <= nodes
