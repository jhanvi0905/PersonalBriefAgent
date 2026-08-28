from assistant.sources.calendar import fetch_upcoming_events
from assistant.sources.email import fetch_recent_emails
from assistant.sources.news import fetch_news

__all__ = ["fetch_recent_emails", "fetch_upcoming_events", "fetch_news"]
