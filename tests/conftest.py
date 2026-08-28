import pytest

from assistant.sources.google import CredentialsMissing


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep the suite offline: no Google, no LLM API. Tests opt back in explicitly."""
    def no_google(*a, **k):
        raise CredentialsMissing("offline test")

    monkeypatch.setattr("assistant.nodes.load_credentials", no_google, raising=False)
