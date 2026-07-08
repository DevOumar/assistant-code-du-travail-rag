"""Fetches the raw Code du travail corpus from the Legifrance sandbox API.

This module only retrieves and filters the raw article tree returned by
`consult/legiPart`. It does not normalize, chunk, or reshape documents into the
id/text/metadata contract; that stays the responsibility of the future
`document-parser` branch and `chunking.py`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from config import AppConfig, load_config


OAUTH_TOKEN_URL = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
API_BASE_URL = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"
LEGI_PART_ENDPOINT = f"{API_BASE_URL}/consult/legiPart"

CODE_DU_TRAVAIL_TEXT_ID = "LEGITEXT000006072050"
DEFAULT_OUTPUT_FILENAME = "code_du_travail_raw.json"
DEFAULT_REQUEST_TIMEOUT = 30
TOKEN_EXPIRY_SAFETY_MARGIN = 30

_ARTICLE_NUM_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)((?:-\d+)*)$")


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


def fetch_legi_part(
    token: str,
    text_id: str = CODE_DU_TRAVAIL_TEXT_ID,
    query_date: str | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Call consult/legiPart and return the parsed JSON article tree for a code."""

    active_date = query_date or _today_iso()

    try:
        response = requests.post(
            LEGI_PART_ENDPOINT,
            json={"textId": text_id, "date": active_date},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise LegifranceApiError("Timed out while calling consult/legiPart.") from exc
    except requests.RequestException as exc:
        raise LegifranceApiError(f"Failed to reach consult/legiPart: {exc}") from exc

    if response.status_code != 200:
        raise LegifranceApiError(
            f"consult/legiPart returned HTTP {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise LegifranceApiError("consult/legiPart returned a non-JSON response.") from exc

    if not isinstance(payload, dict) or "sections" not in payload:
        raise LegifranceApiError(
            "Unexpected consult/legiPart response structure: missing 'sections'."
        )

    return payload


def extract_articles(legi_part_response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Walk the sections/articles tree and flatten it, keeping the parent section path."""

    articles = [_build_article(article, []) for article in legi_part_response.get("articles") or []]

    for section in legi_part_response.get("sections") or []:
        articles.extend(_walk_sections(section, []))

    return articles


def _walk_sections(node: Mapping[str, Any], parent_path: list[str]) -> list[dict[str, Any]]:
    title = (node.get("title") or "").strip()
    path = [*parent_path, title] if title else parent_path

    articles = [_build_article(article, path) for article in node.get("articles") or []]

    for sub_section in node.get("sections") or []:
        articles.extend(_walk_sections(sub_section, path))

    return articles


def _build_article(article: Mapping[str, Any], section_path: list[str]) -> dict[str, Any]:
    return {
        "num": article.get("num"),
        "id": article.get("id"),
        "content": article.get("content"),
        "section_path": list(section_path),
    }


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")


@dataclass(frozen=True)
class ArticleRange:
    start: str
    end: str
    theme: str


THEME_RANGES: tuple[ArticleRange, ...] = (
    ArticleRange("L3121-1", "L3121-36", theme="duree_travail"),
    ArticleRange("L3141-1", "L3141-32", theme="conges_payes"),
    ArticleRange("L1221-1", "L1248-11", theme="contrat_travail"),
    ArticleRange("L1231-1", "L1237-20", theme="rupture_contrat"),
    ArticleRange("L1237-11", "L1237-19", theme="rupture_conventionnelle"),
)


def is_article_in_scope(num: str) -> bool:
    """Return True if an article number falls in one of the covered theme ranges."""

    key = _parse_article_num(num)
    if key is None:
        return False

    for article_range in THEME_RANGES:
        start_key = _parse_article_num(article_range.start)
        end_key = _parse_article_num(article_range.end)
        if key[0] == start_key[0] == end_key[0] and start_key[1:] <= key[1:] <= end_key[1:]:
            return True

    return False


def _parse_article_num(num: str) -> tuple[str, int, tuple[int, ...]] | None:
    if not isinstance(num, str):
        return None

    match = _ARTICLE_NUM_PATTERN.match(num.strip())
    if not match:
        return None

    prefix, major, rest = match.groups()
    rest_parts = tuple(int(part) for part in rest.split("-") if part)
    return prefix.upper(), int(major), rest_parts


def load_corpus(
    config: AppConfig | None = None,
    query_date: str | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Path:
    """Fetch, filter, and persist the raw Code du travail corpus for the covered themes."""

    active_config = config or load_config()
    client_id = active_config.legifrance.client_id
    client_secret = active_config.legifrance.client_secret

    if not client_id or not client_secret:
        raise LegifranceAuthError(
            "LEGIFRANCE_CLIENT_ID and LEGIFRANCE_CLIENT_SECRET must be set to load the corpus."
        )

    token_manager = LegifranceTokenManager(client_id, client_secret)
    token = token_manager.get_token()

    legi_part = fetch_legi_part(token, query_date=query_date)
    articles = extract_articles(legi_part)
    filtered = [article for article in articles if is_article_in_scope(article["num"])]

    output = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "legifrance-sandbox",
        "text_id": CODE_DU_TRAVAIL_TEXT_ID,
        "endpoint": LEGI_PART_ENDPOINT,
        "article_count": len(filtered),
        "articles": filtered,
    }

    output_path = active_config.paths.raw_data_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path


if __name__ == "__main__":
    saved_path = load_corpus()
    print(f"Corpus saved to {saved_path}")
