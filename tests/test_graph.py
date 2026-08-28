from datetime import datetime, timezone

from assistant import graph as graph_mod
from assistant.graph import build_graph
from assistant.guardrails import pack_compose, pack_prioritize
from assistant.llm import HeuristicLLM
from assistant.memory import empty_store, seed_defaults
from assistant.models import BriefItem, MemoryView, RankedItem, Source
from assistant.state import RuntimeCtx


def test_pack_compose_uses_included_winners_only():
    as_of = datetime(2026, 8, 27, 15, tzinfo=timezone.utc)
    items = [
        BriefItem(
            id="email:a",
            source=Source.email,
            title="A",
            summary="keep",
            timestamp=as_of,
        ),
        BriefItem(
            id="email:b",
            source=Source.email,
            title="B",
            summary="drop",
            timestamp=as_of,
        ),
    ]
    ranked = [
        RankedItem(item_id="email:a", score=0.9, rank=1, reason="vip", include=True),
        RankedItem(item_id="email:b", score=0.1, rank=2, reason="noise", include=False),
    ]
    pack = pack_compose(items, ranked, MemoryView())
    assert [c.id for c in pack.winners] == ["email:a"]
    assert {c.id for c in pack_prioritize(items)} == {"email:a", "email:b"}


def test_graph_produces_brief_and_dedups_on_second_run(monkeypatch):
    def fake_news(state, runtime):
        return {
            "source_results": {
                "news": {
                    "ok": True,
                    "items": [
                        BriefItem(
                            id="news:gemini",
                            source=Source.news,
                            title="[google] Gemini API",
                            summary="LLM update",
                            timestamp=datetime(2026, 8, 27, 14, tzinfo=timezone.utc),
                            url="https://developers.googleblog.com/gemini",
                        ).model_dump(mode="json")
                    ],
                }
            }
        }

    monkeypatch.setattr(graph_mod, "fetch_ai_news", fake_news)
    store = empty_store()
    seed_defaults(store, "u")
    graph = build_graph(HeuristicLLM(), store=store)
    as_of = datetime(2026, 8, 27, 16, tzinfo=timezone.utc)
    ctx = RuntimeCtx(user_id="u", request_id="t1", as_of=as_of, model="heuristic")
    result = graph.invoke({}, config={"configurable": {"thread_id": "t1"}}, context=ctx)
    assert result["brief"]["headline"]
    assert result["brief"]["item_ids"]

    graph2 = build_graph(HeuristicLLM(), store=store)
    result2 = graph2.invoke(
        {},
        config={"configurable": {"thread_id": "t2"}},
        context=RuntimeCtx(user_id="u", request_id="t2", as_of=as_of, model="heuristic"),
    )
    assert set(result["brief"]["item_ids"]).isdisjoint(set(result2["brief"]["item_ids"]))
