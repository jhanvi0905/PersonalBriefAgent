# Personal brief workflow

LangGraph **workflow** (fixed DAG). Not a tool-calling agent. Grok ranks and writes; Python fetches, filters, packs, and persists.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional: XAI_API_KEY for Grok; otherwise heuristic LLM
brief
# or: python -m assistant.cli
pytest
```

Without `XAI_API_KEY`, compose still runs via `HeuristicLLM`. With a key, `ChatXAI` (`XAI_MODEL`, default `grok-3-mini`) is used.

## Layout

| Path | Role |
|---|---|
| `src/assistant/graph.py` | `StateGraph` wiring |
| `src/assistant/state.py` | `RuntimeCtx` (immutable) vs `BriefState` (reducers) |
| `src/assistant/memory.py` | LangGraph `Store`: prefs, ledger, digest |
| `src/assistant/models.py` | Pydantic contracts (`BriefItem`, packs, brief) |
| `src/assistant/guardrails.py` | Sanitize, rule filter, salience cap, packs, fallback rank |
| `src/assistant/nodes.py` | One function per DAG step |
| `src/assistant/llm.py` | `BriefLLM` protocol: `GrokLLM` / `HeuristicLLM` |
| `src/assistant/sources/email.py` | Stub inbox (swap for Gmail later) |
| `src/assistant/sources/calendar.py` | Stub events (swap for Calendar later) |
| `src/assistant/sources/news.py` | OpenAI RSS, Google Developers RSS (AI-only), Anthropic/Meta HTML |
| `src/assistant/cli.py` | Entry + `data/memory.json` snapshot |
| `src/assistant/config.py` | Budgets, feed list, settings |
| `tests/` | Mirrors the above; graph tests mock live news |

## Data layers

- **Runtime context:** `user_id`, `request_id`, `as_of`, model — not checkpointed as working memory.
- **Graph state:** pipeline only (`source_results` merge, then items → candidates → ranked → packs → brief). Raw fetches cleared in `normalize`.
- **Store:** seen/handled ids, last digest, prefs. New `thread_id` per run.
- **LLM packs:** item cards only; bodies never go to Grok.

## DAG

`load_memory` → parallel `fetch_emails` / `fetch_events` / `fetch_news` → `normalize` → `rule_filter` → `pack_prioritize` → `prioritize` → `pack_compose` → `compose` → `persist`

## Extend later

- Real mail/calendar: replace the two stub modules; keep `BriefItem`.
- More news: append `NEWS_FEEDS` in `config.py`.
- Chat/agent: new graph that **reads** the store; do not turn this DAG into ReAct.
- Delivery: Slack/email after `persist`, not inside Grok.
