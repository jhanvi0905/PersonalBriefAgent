# Personal brief workflow

LangGraph **workflow** (fixed DAG). Not a tool-calling agent. Grok ranks and writes; Python fetches, filters, packs, and persists.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional: XAI_API_KEY for Grok; otherwise heuristic LLM
brief-auth             # optional: one-time Google consent for Gmail + Calendar
brief
# or: python -m assistant.cli
pytest
```

Without `XAI_API_KEY`, compose still runs via `HeuristicLLM`. With a key, `ChatXAI` (`XAI_MODEL`, default `grok-3-mini`) is used.

### Google (Gmail + Calendar, read-only)

1. Google Cloud console → new project → enable the Gmail API and Google Calendar API.
2. OAuth consent screen: External, add your own address as a test user, scopes
   `gmail.readonly` + `calendar.readonly`.
3. Credentials → Create OAuth client ID → **Desktop app** → download JSON to
   `credentials.json` (repo root).
4. `brief-auth` opens a browser once and caches `data/google_token.json`.

No token → `brief` still runs; email/calendar just come back empty.

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
| `src/assistant/sources/google.py` | OAuth: `brief-auth` flow + token-only `load_credentials` |
| `src/assistant/sources/email.py` | Gmail: inbox messages in the last 24h |
| `src/assistant/sources/calendar.py` | Calendar: `primary` events in the next 24h |
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

- More news: append `NEWS_FEEDS` in `config.py`.
- Chat/agent: new graph that **reads** the store; do not turn this DAG into ReAct.
- Delivery: Slack/email after `persist`, not inside Grok.
