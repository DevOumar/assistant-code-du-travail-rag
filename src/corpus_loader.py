"""Fetches the raw Code du travail corpus from the Legifrance sandbox API.

This module only retrieves and filters the raw article tree returned by
`consult/legiPart`. It does not normalize, chunk, or reshape documents into the
id/text/metadata contract; that stays the responsibility of the future
`document-parser` branch and `chunking.py`.
"""

from __future__ import annotations

import time

import requests


OAUTH_TOKEN_URL = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
API_BASE_URL = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"

DEFAULT_REQUEST_TIMEOUT = 30
TOKEN_EXPIRY_SAFETY_MARGIN = 30


class CorpusLoaderError(Exception):
    """Base error for corpus loading failures."""


class LegifranceAuthError(CorpusLoaderError):
    """Raised when OAuth2 authentication with the Legifrance sandbox fails."""


class LegifranceApiError(CorpusLoaderError):
    """Raised when consult/legiPart cannot be reached or returns an unexpected response."""


class LegifranceTokenManager:
    """Fetches an OAuth2 client-credentials token and refreshes it before it expires."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        if self._access_token is None or time.monotonic() >= self._expires_at:
            self._refresh()
        assert self._access_token is not None
        return self._access_token

    def _refresh(self) -> None:
        try:
            response = requests.post(
                OAUTH_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "openid",
                },
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise LegifranceAuthError("Timed out while requesting an OAuth2 token.") from exc
        except requests.RequestException as exc:
            raise LegifranceAuthError(f"Failed to reach the OAuth2 token endpoint: {exc}") from exc

        if response.status_code != 200:
            raise LegifranceAuthError(
                f"OAuth2 authentication failed with HTTP {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
            access_token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 0))
        except (ValueError, KeyError, TypeError) as exc:
            raise LegifranceAuthError("Unexpected OAuth2 token response structure.") from exc

        self._access_token = access_token
        self._expires_at = time.monotonic() + max(expires_in - TOKEN_EXPIRY_SAFETY_MARGIN, 0)
