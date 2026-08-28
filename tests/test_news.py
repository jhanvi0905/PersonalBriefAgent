from datetime import datetime, timezone

from assistant.sources.news import looks_like_ai, parse_html_listing, parse_rss


RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>OpenAI</title>
<item>
  <title>New model launch</title>
  <link>https://openai.com/news/new-model</link>
  <description>We shipped a model.</description>
  <pubDate>Wed, 26 Aug 2026 12:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

GOOGLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Android 17 beta</title>
  <link>https://developers.googleblog.com/android</link>
  <description>Platform APIs</description>
  <pubDate>Wed, 26 Aug 2026 12:00:00 GMT</pubDate>
</item>
<item>
  <title>Gemini API updates</title>
  <link>https://developers.googleblog.com/gemini</link>
  <description>LLM tooling for developers</description>
  <pubDate>Wed, 26 Aug 2026 13:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

HTML = """
<html><body>
<a href="/news/claude-opus">Claude Opus</a>
<a href="https://www.anthropic.com/news/">index</a>
<a href="https://example.com/x">other</a>
</body></html>
"""


def test_parse_rss_openai():
    as_of = datetime(2026, 8, 27, tzinfo=timezone.utc)
    items = parse_rss(RSS, "openai", as_of, ai_only=False)
    assert len(items) == 1
    assert items[0].url.endswith("/new-model")
    assert items[0].id.startswith("news:")


def test_google_rss_keeps_ai_only():
    as_of = datetime(2026, 8, 27, tzinfo=timezone.utc)
    items = parse_rss(GOOGLE_RSS, "google", as_of, ai_only=True)
    assert [i.url for i in items] == ["https://developers.googleblog.com/gemini"]


def test_html_listing_filters_prefix():
    as_of = datetime(2026, 8, 27, tzinfo=timezone.utc)
    items = parse_html_listing(
        HTML,
        "https://www.anthropic.com/news",
        "anthropic",
        "https://www.anthropic.com/news/",
        as_of,
    )
    assert len(items) == 1
    assert "claude-opus" in items[0].url


def test_looks_like_ai():
    assert looks_like_ai("Gemini API")
    assert not looks_like_ai("Android widget gallery")
