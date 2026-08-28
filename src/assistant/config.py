from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_RAW_PER_SOURCE = 8
MAX_CANDIDATES_TO_LLM = 30
MAX_BRIEF_ITEMS = 12
ITEM_SUMMARY_CHARS = 400
NEWS_WINDOW_HOURS = 48
GMAIL_LOOKBACK_HOURS = 24
CALENDAR_LOOKAHEAD_HOURS = 24

GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
)

AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "gemini",
    "gemma",
    "llm",
    "gpt",
    "claude",
    "llama",
    "machine learning",
    "deep learning",
    "neural",
    "tpu",
    "gpu",
    "model",
    "openai",
    "anthropic",
    "vertex",
)

NEWS_FEEDS: tuple[dict[str, str], ...] = (
    {
        "id": "openai",
        "kind": "rss",
        "url": "https://openai.com/news/rss.xml",
    },
    {
        "id": "google",
        "kind": "rss",
        "url": "https://developers.googleblog.com/feeds/posts/default",
        "ai_only": "true",
    },
    {
        "id": "anthropic",
        "kind": "html",
        "url": "https://www.anthropic.com/news",
        "link_prefix": "https://www.anthropic.com/news/",
    },
    {
        "id": "meta",
        "kind": "html",
        "url": "https://ai.meta.com/blog/",
        "link_prefix": "https://ai.meta.com/blog/",
    },
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    xai_api_key: str = ""
    xai_model: str = "grok-3-mini"
    xai_api_base: str = ""  # override for an OpenAI-compatible gateway (e.g. aimlapi.com)
    brief_user_id: str = "default"
    data_dir: str = "data"
    google_client_secrets: str = "credentials.json"
    google_token_file: str = "data/google_token.json"


def get_settings() -> Settings:
    return Settings()
