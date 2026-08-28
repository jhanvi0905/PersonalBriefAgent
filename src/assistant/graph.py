from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from assistant.llm import BriefLLM, HeuristicLLM
from assistant.nodes import (
    apply_rules,
    fetch_ai_news,
    fetch_emails,
    fetch_events,
    load_memory,
    make_compose,
    make_prioritize,
    normalize,
    pack_for_rank,
    pack_for_write,
    persist,
)
from assistant.state import BriefOutput, BriefState, RuntimeCtx


def build_graph(
    llm: BriefLLM | None = None,
    *,
    store=None,
    checkpointer=None,
):
    llm = llm or HeuristicLLM()
    builder = StateGraph(
        BriefState,
        context_schema=RuntimeCtx,
        output_schema=BriefOutput,
    )
    builder.add_node("load_memory", load_memory)
    builder.add_node("fetch_emails", fetch_emails)
    builder.add_node("fetch_events", fetch_events)
    builder.add_node("fetch_news", fetch_ai_news)
    builder.add_node("normalize", normalize)
    builder.add_node("rule_filter", apply_rules)
    builder.add_node("pack_prioritize", pack_for_rank)
    builder.add_node("prioritize", make_prioritize(llm))
    builder.add_node("pack_compose", pack_for_write)
    builder.add_node("compose", make_compose(llm))
    builder.add_node("persist", persist)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "fetch_emails")
    builder.add_edge("load_memory", "fetch_events")
    builder.add_edge("load_memory", "fetch_news")
    builder.add_edge("fetch_emails", "normalize")
    builder.add_edge("fetch_events", "normalize")
    builder.add_edge("fetch_news", "normalize")
    builder.add_edge("normalize", "rule_filter")
    builder.add_edge("rule_filter", "pack_prioritize")
    builder.add_edge("pack_prioritize", "prioritize")
    builder.add_edge("prioritize", "pack_compose")
    builder.add_edge("pack_compose", "compose")
    builder.add_edge("compose", "persist")
    builder.add_edge("persist", END)

    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )
