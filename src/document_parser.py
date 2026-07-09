"""Normalizes the raw Legifrance corpus into documents ready for chunking.

This module reads the raw article tree produced by the `corpus-loader` branch
(data/raw/code_du_travail_raw.json), cleans the HTML content, and builds the
id/text/metadata documents expected by `chunking.py`. It does not chunk or
index anything; that stays the responsibility of `chunking.py` and later
branches.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from config import AppConfig, load_config


DEFAULT_RAW_FILENAME = "code_du_travail_raw.json"
DEFAULT_OUTPUT_FILENAME = "code_du_travail_documents.json"

_ARTICLE_NUM_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)((?:-\d+)*)$")

_PARAGRAPH_BREAK_TAGS = re.compile(r"</(p|div)\s*>", re.IGNORECASE)
_LINE_BREAK_TAGS = re.compile(r"<br\s*/?>|</tr\s*>", re.IGNORECASE)
_CELL_BREAK_TAGS = re.compile(r"</(td|th)\s*>", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class DocumentParserError(Exception):
    """Base error for document parsing failures."""


class RawCorpusNotFoundError(DocumentParserError):
    """Raised when the raw corpus JSON file cannot be found."""


class RawCorpusFormatError(DocumentParserError):
    """Raised when the raw corpus JSON file is malformed or has an unexpected shape."""


@dataclass(frozen=True)
class ThemeRange:
    start: str
    end: str
    theme: str


# Mirrors the theme ranges used by the corpus-loader branch to filter articles.
# Ordered narrowest-first so that overlapping ranges (rupture_conventionnelle and
# licenciement both sit inside contrat_travail's broad span) resolve to the most
# specific theme rather than being swallowed by the broader one.
THEME_RANGES: tuple[ThemeRange, ...] = (
    ThemeRange("L1237-11", "L1237-19", theme="rupture_conventionnelle"),
    ThemeRange("L1231-1", "L1237-20", theme="licenciement"),
    ThemeRange("L1221-1", "L1248-11", theme="contrat_travail"),
    ThemeRange("L3121-1", "L3121-36", theme="duree_travail"),
    ThemeRange("L3141-1", "L3141-32", theme="conges_payes"),
)


@dataclass(frozen=True)
class ParsedDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_html(raw_html: str) -> str:
    """Strip HTML tags/entities from an article's raw content and normalize whitespace."""

    text = _PARAGRAPH_BREAK_TAGS.sub("\n\n", raw_html)
    text = _LINE_BREAK_TAGS.sub("\n", text)
    text = _CELL_BREAK_TAGS.sub(" ", text)
    text = _TAG_PATTERN.sub("", text)
    text = html.unescape(text)
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def determine_theme(num: str) -> str | None:
    """Return the thematic section for an article number, or None if unrecognized."""

    key = _parse_article_num(num)
    if key is None:
        return None

    for theme_range in THEME_RANGES:
        start_key = _parse_article_num(theme_range.start)
        end_key = _parse_article_num(theme_range.end)
        if key[0] == start_key[0] == end_key[0] and start_key[1:] <= key[1:] <= end_key[1:]:
            return theme_range.theme

    return None


def _parse_article_num(num: str) -> tuple[str, int, tuple[int, ...]] | None:
    if not isinstance(num, str):
        return None

    match = _ARTICLE_NUM_PATTERN.match(num.strip())
    if not match:
        return None

    prefix, major, rest = match.groups()
    rest_parts = tuple(int(part) for part in rest.split("-") if part)
    return prefix.upper(), int(major), rest_parts


def load_raw_articles(raw_data_dir: Path, filename: str = DEFAULT_RAW_FILENAME) -> list[dict[str, Any]]:
    """Load the raw article list produced by the corpus-loader branch."""

    path = raw_data_dir / filename
    if not path.is_file():
        raise RawCorpusNotFoundError(f"Raw corpus file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawCorpusFormatError(f"Raw corpus file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict) or "articles" not in payload:
        raise RawCorpusFormatError(f"Raw corpus file is missing the 'articles' key: {path}")

    articles = payload["articles"]
    if not isinstance(articles, list):
        raise RawCorpusFormatError(f"'articles' must be a list in: {path}")

    return articles


def build_document(
    article: Mapping[str, Any],
    corpus_source: str | None,
    corpus_date: str | None,
) -> ParsedDocument | None:
    """Build a ParsedDocument from one raw article, or None if it cannot be parsed."""

    num = article.get("num")
    if not isinstance(num, str) or not num.strip():
        return None

    content = article.get("content")
    if not isinstance(content, str):
        return None

    cleaned_content = clean_html(content)
    if not cleaned_content:
        return None

    text = f"Article {num}. {cleaned_content}"

    section_path = article.get("section_path")
    title = section_path[-1] if isinstance(section_path, list) and section_path else None

    metadata = {
        "article": num,
        "num": num,
        "legiarti": article.get("id"),
        "theme": determine_theme(num),
        "source": corpus_source,
        "corpus_date": corpus_date,
        "title": title,
    }

    return ParsedDocument(id=f"article-{num}", text=text, metadata=metadata)


def parse_documents(
    articles: list[Mapping[str, Any]],
    corpus_source: str | None,
    corpus_date: str | None,
) -> tuple[list[ParsedDocument], list[dict[str, Any]]]:
    """Build documents from raw articles, skipping malformed entries instead of failing."""

    documents: list[ParsedDocument] = []
    skipped: list[dict[str, Any]] = []

    for article in articles:
        document = build_document(article, corpus_source, corpus_date)
        if document is None:
            skipped.append({"num": article.get("num"), "id": article.get("id")})
        else:
            documents.append(document)

    return documents, skipped


def load_and_build_documents(config: AppConfig) -> tuple[list[ParsedDocument], list[dict[str, Any]]]:
    articles = load_raw_articles(config.paths.raw_data_dir)
    return parse_documents(articles, config.corpus.source, config.corpus.date)


def parse_corpus(
    config: AppConfig | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> tuple[Path, list[ParsedDocument], list[dict[str, Any]]]:
    """Parse the raw corpus into documents and persist them to processed_data_dir."""

    active_config = config or load_config()
    documents, skipped = load_and_build_documents(active_config)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "skipped_count": len(skipped),
        "documents": [asdict(document) for document in documents],
    }

    output_path = active_config.paths.processed_data_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, documents, skipped


def print_quality_sample(
    documents: list[ParsedDocument],
    sample_size: int = 10,
    seed: int | None = None,
) -> None:
    """Print a random sample of documents for manual quality review."""

    import random

    rng = random.Random(seed)
    sample = rng.sample(documents, min(sample_size, len(documents)))

    for document in sample:
        preview = document.text[:300]
        if len(document.text) > 300:
            preview += "..."
        print(f"--- {document.id} ---")
        print(f"metadata: {document.metadata}")
        print(f"text: {preview}")
        print()


if __name__ == "__main__":
    saved_path, parsed_documents, skipped_articles = parse_corpus()
    print(f"Documents saved to {saved_path}")
    print(f"{len(parsed_documents)} documents produced, {len(skipped_articles)} articles skipped.")
    print()
    print_quality_sample(parsed_documents)
