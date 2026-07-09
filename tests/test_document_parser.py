import json
from pathlib import Path

import pytest

from config import load_config
from document_parser import (
    RawCorpusFormatError,
    RawCorpusNotFoundError,
    build_document,
    clean_html,
    determine_theme,
    load_raw_articles,
    parse_corpus,
    parse_documents,
)


class TestCleanHtml:
    def test_strips_simple_paragraph_tags(self) -> None:
        raw = "<p></p>   Le contrat de travail est soumis aux regles du droit commun.<p></p><p></p>"

        assert clean_html(raw) == "Le contrat de travail est soumis aux regles du droit commun."

    def test_handles_nested_tags(self) -> None:
        raw = (
            "<p>En application de l'article "
            '<a href="/affichCodeArticle.do?idArticle=LEGIARTI000006900783">L. 1111-2</a>'
            ", les salaries mis a disposition ne sont pas pris en compte.</p>"
        )

        assert clean_html(raw) == (
            "En application de l'article L. 1111-2, les salaries mis a disposition ne "
            "sont pas pris en compte."
        )

    def test_converts_table_rows_and_cells(self) -> None:
        raw = "<table><tbody><tr><td>A1</td><td>A2</td></tr><tr><td>B1</td><td>B2</td></tr></tbody></table>"

        assert clean_html(raw) == "A1 A2\nB1 B2"

    def test_unescapes_html_entities(self) -> None:
        assert clean_html("<p>Salari&eacute; &amp; employeur</p>") == "Salarie & employeur".replace(
            "Salarie", "Salarié"
        )

    def test_collapses_excess_blank_lines(self) -> None:
        raw = "<p>Premier</p><br><br><br><p>Second</p>"

        assert clean_html(raw) == "Premier\n\nSecond"


class TestDetermineTheme:
    @pytest.mark.parametrize(
        ("num", "expected"),
        [
            ("L3121-1", "duree_travail"),
            ("L3121-36", "duree_travail"),
            ("L3141-1", "conges_payes"),
            ("L1237-11", "rupture_conventionnelle"),
            ("L1237-19", "rupture_conventionnelle"),
            ("L1231-1", "licenciement"),
            ("L1235-1", "licenciement"),
            ("L1221-1", "contrat_travail"),
            ("L1248-11", "contrat_travail"),
            ("L1237-21", "contrat_travail"),  # outside licenciement's sub-range, inside the broader contrat_travail range
        ],
    )
    def test_maps_article_number_to_theme(self, num: str, expected: str) -> None:
        assert determine_theme(num) == expected

    def test_returns_none_for_unrecognized_number(self) -> None:
        assert determine_theme("R1111-1") is None
        assert determine_theme("not-a-number") is None


class TestBuildDocument:
    def test_builds_document_from_valid_article(self) -> None:
        article = {
            "num": "L3121-1",
            "id": "LEGIARTI000018487817",
            "content": "<p>La duree du travail effectif est le temps de travail.</p>",
            "section_path": ["Partie legislative", "Livre Ier", "Chapitre Ier"],
        }

        document = build_document(article, corpus_source="legifrance-sandbox", corpus_date="2026-07-08")

        assert document is not None
        assert document.id == "article-L3121-1"
        assert document.text == "Article L3121-1. La duree du travail effectif est le temps de travail."
        assert document.metadata == {
            "article": "L3121-1",
            "num": "L3121-1",
            "legiarti": "LEGIARTI000018487817",
            "theme": "duree_travail",
            "source": "legifrance-sandbox",
            "corpus_date": "2026-07-08",
            "title": "Chapitre Ier",
        }

    def test_returns_none_when_num_missing(self) -> None:
        article = {"id": "LEGIARTI1", "content": "<p>Texte.</p>"}

        assert build_document(article, corpus_source=None, corpus_date=None) is None

    def test_returns_none_when_content_missing(self) -> None:
        article = {"num": "L3121-1", "id": "LEGIARTI1"}

        assert build_document(article, corpus_source=None, corpus_date=None) is None

    def test_returns_none_when_content_cleans_to_empty(self) -> None:
        article = {"num": "L3121-1", "id": "LEGIARTI1", "content": "<p></p>   <br>  "}

        assert build_document(article, corpus_source=None, corpus_date=None) is None

    def test_title_is_none_when_section_path_missing(self) -> None:
        article = {"num": "L3121-1", "id": "LEGIARTI1", "content": "<p>Texte.</p>"}

        document = build_document(article, corpus_source=None, corpus_date=None)

        assert document is not None
        assert document.metadata["title"] is None


class TestParseDocuments:
    def test_skips_malformed_articles_and_keeps_valid_ones(self) -> None:
        articles = [
            {"num": "L3121-1", "id": "LEGIARTI1", "content": "<p>Valide.</p>", "section_path": ["Titre"]},
            {"id": "LEGIARTI2", "content": "<p>Sans numero.</p>"},
            {"num": "L3141-1", "id": "LEGIARTI3"},
        ]

        documents, skipped = parse_documents(articles, corpus_source="legifrance-sandbox", corpus_date="2026-07-08")

        assert [document.id for document in documents] == ["article-L3121-1"]
        assert len(skipped) == 2

    def test_skips_duplicate_article_numbers(self) -> None:
        articles = [
            {"num": "L3121-1", "id": "LEGIARTI1", "content": "<p>Valide.</p>", "section_path": ["Titre"]},
            {"num": "L3121-1", "id": "LEGIARTI1", "content": "<p>Valide.</p>", "section_path": ["Titre"]},
        ]

        documents, skipped = parse_documents(articles, corpus_source="legifrance-sandbox", corpus_date="2026-07-08")

        assert [document.id for document in documents] == ["article-L3121-1"]
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "duplicate document id"


class TestLoadRawArticles:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(RawCorpusNotFoundError):
            load_raw_articles(tmp_path)

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / "code_du_travail_raw.json").write_text("{not-json", encoding="utf-8")

        with pytest.raises(RawCorpusFormatError, match="not valid JSON"):
            load_raw_articles(tmp_path)

    def test_raises_when_articles_key_missing(self, tmp_path: Path) -> None:
        (tmp_path / "code_du_travail_raw.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        with pytest.raises(RawCorpusFormatError, match="'articles'"):
            load_raw_articles(tmp_path)

    def test_loads_valid_articles_list(self, tmp_path: Path) -> None:
        payload = {"articles": [{"num": "L3121-1", "id": "LEGIARTI1", "content": "<p>Texte.</p>"}]}
        (tmp_path / "code_du_travail_raw.json").write_text(json.dumps(payload), encoding="utf-8")

        articles = load_raw_articles(tmp_path)

        assert articles == payload["articles"]


class TestParseCorpus:
    def test_end_to_end_persists_documents(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()

        payload = {
            "articles": [
                {
                    "num": "L3121-1",
                    "id": "LEGIARTI1",
                    "content": "<p>Duree du travail.</p>",
                    "section_path": ["Partie legislative", "Chapitre Ier"],
                },
                {"num": "L9999-1", "id": "LEGIARTI2", "content": None},
            ]
        }
        (raw_dir / "code_du_travail_raw.json").write_text(json.dumps(payload), encoding="utf-8")

        config = load_config(
            env_file=None,
            environ={
                "RAW_DATA_DIR": str(raw_dir),
                "PROCESSED_DATA_DIR": str(processed_dir),
                "CORPUS_SOURCE": "legifrance-sandbox",
                "CORPUS_DATE": "2026-07-08",
            },
        )

        output_path, documents, skipped = parse_corpus(config=config)

        assert output_path == processed_dir / "code_du_travail_documents.json"
        assert len(documents) == 1
        assert len(skipped) == 1

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["document_count"] == 1
        assert saved["skipped_count"] == 1
        assert saved["documents"][0]["id"] == "article-L3121-1"
        assert saved["documents"][0]["metadata"]["source"] == "legifrance-sandbox"
