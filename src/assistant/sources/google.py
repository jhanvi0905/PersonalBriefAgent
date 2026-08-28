"""Google OAuth for the Gmail and Calendar read-only sources.

`brief-auth` runs the one-time browser consent flow and caches a token.
`load_credentials` is what the graph nodes call: token file only, with silent
refresh. It never opens a browser — missing/invalid tokens raise
`CredentialsMissing` so a run degrades to a partial brief instead of hanging.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from assistant.config import GOOGLE_SCOPES

SCOPES = list(GOOGLE_SCOPES)


class CredentialsMissing(RuntimeError):
    """No usable cached token; run `brief-auth`."""


def load_credentials(client_secrets: Path, token_file: Path) -> Credentials:
    if not token_file.exists():
        raise CredentialsMissing(f"{token_file} not found; run `brief-auth`")
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json())
        return creds
    raise CredentialsMissing(f"{token_file} is stale; run `brief-auth`")


def authorize(client_secrets: Path, token_file: Path) -> Credentials:
    if not client_secrets.exists():
        raise FileNotFoundError(
            f"{client_secrets} not found; download an OAuth desktop client "
            "from Google Cloud console and save it there"
        )
    creds = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets), SCOPES
    ).run_local_server(port=0)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json())
    return creds
