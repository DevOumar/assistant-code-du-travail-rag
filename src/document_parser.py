"""Normalizes the raw Legifrance corpus into documents ready for chunking.

This module reads the raw article tree produced by the `corpus-loader` branch
(data/raw/code_du_travail_raw.json), cleans the HTML content, and builds the
id/text/metadata documents expected by `chunking.py`. It does not chunk or
index anything; that stays the responsibility of `chunking.py` and later
branches.
"""

from __future__ import annotations

import html
import re


_PARAGRAPH_BREAK_TAGS = re.compile(r"</(p|div)\s*>", re.IGNORECASE)
_LINE_BREAK_TAGS = re.compile(r"<br\s*/?>|</tr\s*>", re.IGNORECASE)
_CELL_BREAK_TAGS = re.compile(r"</(td|th)\s*>", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class DocumentParserError(Exception):
    """Base error for document parsing failures."""


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
