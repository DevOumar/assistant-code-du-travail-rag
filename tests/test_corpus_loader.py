import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from config import load_config
from corpus_loader import (
    LegifranceApiError,
    LegifranceAuthError,
    LegifranceTokenManager,
    extract_articles,
    fetch_legi_part,
    is_article_in_scope,
    load_corpus,
)


def _oauth_response(status_code: int = 200, expires_in: int = 1200, body: dict | None = None) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = (
        body if body is not None else {"access_token": "fake-token", "expires_in": expires_in}
    )
    response.text = json.dumps(response.json.return_value)
    return response


def _legi_part_response(status_code: int = 200, body: dict | None = None) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = body if body is not None else _sample_tree()
    response.text = json.dumps(response.json.return_value)
    return response


def _sample_tree() -> dict:
    return {
        "cid": "LEGITEXT000006072050",
        "title": "Code du travail",
        "sections": [
            {
                "title": "Partie legislative",
                "sections": [
                    {
                        "title": "Livre Ier : Durree du travail",
                        "sections": [],
                        "articles": [
                            {
                                "num": "L3121-1",
                                "id": "LEGIARTI000018487817",
                                "content": "<p>Duree du travail effectif...</p>",
                            },
                            {
                                "num": "L3121-99",
                                "id": "LEGIARTI000000000000",
                                "content": "<p>Hors champ.</p>",
                            },
                        ],
                    }
                ],
                "articles": [],
            }
        ],
        "articles": [],
    }


class TestLegifranceTokenManager:
    def test_fetches_and_caches_token(self) -> None:
        with patch("corpus_loader.requests.post", return_value=_oauth_response()) as mock_post:
            manager = LegifranceTokenManager("client-id", "client-secret")

            first = manager.get_token()
            second = manager.get_token()

            assert first == "fake-token"
            assert second == "fake-token"
            mock_post.assert_called_once()

    def test_refreshes_after_expiry(self) -> None:
        responses = [
            _oauth_response(expires_in=60),
            _oauth_response(expires_in=60, body={"access_token": "renewed-token", "expires_in": 60}),
        ]

        with patch("corpus_loader.requests.post", side_effect=responses) as mock_post:
            with patch("corpus_loader.time.monotonic", side_effect=[0.0, 1000.0, 1000.0]):
                manager = LegifranceTokenManager("client-id", "client-secret")

                first = manager.get_token()
                second = manager.get_token()

        assert first == "fake-token"
        assert second == "renewed-token"
        assert mock_post.call_count == 2

    def test_raises_on_auth_failure(self) -> None:
        with patch("corpus_loader.requests.post", return_value=_oauth_response(status_code=400)):
            manager = LegifranceTokenManager("client-id", "bad-secret")

            with pytest.raises(LegifranceAuthError, match="400"):
                manager.get_token()

    def test_raises_on_timeout(self) -> None:
        with patch("corpus_loader.requests.post", side_effect=requests.Timeout("boom")):
            manager = LegifranceTokenManager("client-id", "client-secret")

            with pytest.raises(LegifranceAuthError, match="Timed out"):
                manager.get_token()

    def test_raises_on_unexpected_token_body(self) -> None:
        with patch("corpus_loader.requests.post", return_value=_oauth_response(body={"nope": True})):
            manager = LegifranceTokenManager("client-id", "client-secret")

            with pytest.raises(LegifranceAuthError, match="Unexpected"):
                manager.get_token()


class TestFetchLegiPart:
    def test_success_returns_payload(self) -> None:
        with patch("corpus_loader.requests.post", return_value=_legi_part_response()):
            payload = fetch_legi_part("token", query_date="2026-07-08T00:00:00Z")

        assert payload["cid"] == "LEGITEXT000006072050"
        assert "sections" in payload

    def test_raises_on_timeout(self) -> None:
        with patch("corpus_loader.requests.post", side_effect=requests.Timeout("boom")):
            with pytest.raises(LegifranceApiError, match="Timed out"):
                fetch_legi_part("token")

    def test_raises_on_non_200(self) -> None:
        with patch("corpus_loader.requests.post", return_value=_legi_part_response(status_code=500)):
            with pytest.raises(LegifranceApiError, match="500"):
                fetch_legi_part("token")

    def test_raises_on_unexpected_structure(self) -> None:
        with patch(
            "corpus_loader.requests.post",
            return_value=_legi_part_response(body={"cid": "LEGITEXT000006072050"}),
        ):
            with pytest.raises(LegifranceApiError, match="missing 'sections'"):
                fetch_legi_part("token")


class TestExtractArticles:
    def test_walks_nested_sections_and_keeps_path(self) -> None:
        articles = extract_articles(_sample_tree())

        assert len(articles) == 2
        first = articles[0]
        assert first["num"] == "L3121-1"
        assert first["id"] == "LEGIARTI000018487817"
        assert first["content"] == "<p>Duree du travail effectif...</p>"
        assert first["section_path"] == ["Partie legislative", "Livre Ier : Durree du travail"]

    def test_handles_missing_sections_key(self) -> None:
        assert extract_articles({"articles": []}) == []


class TestIsArticleInScope:
    @pytest.mark.parametrize(
        "num",
        [
            "L3121-1",
            "L3121-36",
            "L3121-15",
            "L3141-1",
            "L3141-32",
            "L1221-1",
            "L1248-11",
            "L1225-5",
            "L1231-1",
            "L1237-20",
            "L1237-11",
            "L1237-19",
            "L1237-21",  # outside the L1231-1/L1237-20 sub-range, but inside the broader L1221-1/L1248-11 range
        ],
    )
    def test_articles_inside_ranges_are_in_scope(self, num: str) -> None:
        assert is_article_in_scope(num) is True

    @pytest.mark.parametrize(
        "num",
        [
            "L3121-37",
            "L3121-0",
            "L3141-33",
            "L1248-12",
            "L1220-1",
            "R1111-1",
            "L1249-1",
            "not-a-number",
        ],
    )
    def test_articles_outside_ranges_are_excluded(self, num: str) -> None:
        assert is_article_in_scope(num) is False

    def test_non_string_num_is_excluded(self) -> None:
        assert is_article_in_scope(None) is False  # type: ignore[arg-type]


class TestLoadCorpus:
    def test_end_to_end_writes_filtered_articles(self, tmp_path: Path) -> None:
        config = load_config(
            env_file=None,
            environ={
                "RAW_DATA_DIR": str(tmp_path),
                "LEGIFRANCE_CLIENT_ID": "client-id",
                "LEGIFRANCE_CLIENT_SECRET": "client-secret",
            },
        )

        with patch(
            "corpus_loader.requests.post",
            side_effect=[_oauth_response(), _legi_part_response()],
        ):
            output_path = load_corpus(config=config, query_date="2026-07-08T00:00:00Z")

        assert output_path == tmp_path / "code_du_travail_raw.json"
        saved = json.loads(output_path.read_text(encoding="utf-8"))

        assert saved["text_id"] == "LEGITEXT000006072050"
        assert saved["article_count"] == 1
        assert saved["articles"][0]["num"] == "L3121-1"
        assert "retrieved_at" in saved

    def test_raises_when_credentials_missing(self, tmp_path: Path) -> None:
        config = load_config(env_file=None, environ={"RAW_DATA_DIR": str(tmp_path)})

        with pytest.raises(LegifranceAuthError, match="LEGIFRANCE_CLIENT_ID"):
            load_corpus(config=config)
