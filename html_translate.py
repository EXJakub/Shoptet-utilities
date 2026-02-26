from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup, NavigableString

HTML_TAG_RE = re.compile(r"<[a-zA-Z!/][^>]*>")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}")
SKU_RE = re.compile(r"\b(?:SKU|EAN|IP)\s*[:#-]?\s*[A-Z0-9-]{3,}\b|\b\d{8,14}\b")
UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|kg|g|W|kW|V|A|mAh|°C|%)\b", re.IGNORECASE)


@dataclass(slots=True)
class TranslationOptions:
    mode: str = "AUTO"  # AUTO|FORCE_HTML|FORCE_TEXT
    skip_urls: bool = True
    skip_emails: bool = True
    skip_codes: bool = True
    skip_units: bool = True


def is_html(text: str) -> bool:
    return bool(HTML_TAG_RE.search(text))


def should_skip_text(text: str, options: TranslationOptions) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if options.skip_urls and URL_RE.search(text):
        return True
    if options.skip_emails and EMAIL_RE.search(text):
        return True
    if options.skip_codes and SKU_RE.search(text):
        return True
    if options.skip_units and UNIT_RE.search(text):
        return True
    return False


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        chunk = text[cursor : cursor + max_chars]
        if cursor + max_chars < len(text):
            split_at = max(chunk.rfind(". "), chunk.rfind(" "))
            if split_at > 0:
                chunk = chunk[: split_at + 1]
        parts.append(chunk)
        cursor += len(chunk)
    return parts


def translate_plain_text(text: str, translate_fn: Callable[[list[str]], list[str]], options: TranslationOptions, max_chars: int) -> str:
    if should_skip_text(text, options):
        return text
    segments = split_long_text(text, max_chars=max_chars)
    translated = translate_fn(segments)
    return "".join(translated)


def translate_html_text_nodes(
    html: str,
    translate_fn: Callable[[list[str]], list[str]],
    options: TranslationOptions,
    max_chars: int,
) -> str:
    # lxml parser is selected for robust, tolerant parsing of broken e-commerce HTML while keeping attrs accessible.
    soup = BeautifulSoup(html, "lxml")
    text_nodes: list[NavigableString] = []
    originals: list[str] = []

    for node in soup.find_all(string=True):
        parent_name = node.parent.name if node.parent else ""
        if parent_name in {"script", "style"}:
            continue
        value = str(node)
        if should_skip_text(value, options):
            continue
        text_nodes.append(node)
        originals.append(value)

    translated_values: list[str] = []
    for text in originals:
        translated_values.append(translate_plain_text(text, translate_fn, options=options, max_chars=max_chars))

    for node, translated in zip(text_nodes, translated_values):
        node.replace_with(translated)

    body = soup.body
    if body:
        return "".join(str(x) for x in body.contents)
    return str(soup)


def translate_cell(
    value: str,
    translate_fn: Callable[[list[str]], list[str]],
    options: TranslationOptions,
    max_chars: int,
) -> str:
    if value is None:
        return ""
    if options.mode == "FORCE_TEXT":
        return translate_plain_text(value, translate_fn, options, max_chars=max_chars)
    if options.mode == "FORCE_HTML" or (options.mode == "AUTO" and is_html(value)):
        return translate_html_text_nodes(value, translate_fn, options, max_chars=max_chars)
    return translate_plain_text(value, translate_fn, options, max_chars=max_chars)
