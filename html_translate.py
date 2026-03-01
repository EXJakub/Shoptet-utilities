from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from bs4 import BeautifulSoup, NavigableString

HTML_TAG_RE = re.compile(r"<[a-zA-Z!/][^>]*>")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}")
SKU_RE = re.compile(r"\b(?:SKU|EAN|IP)\s*[:#-]?\s*[A-Z0-9-]{3,}\b|\b\d{8,14}\b")
UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|kg|g|W|kW|V|A|mAh|°C|%)\b", re.IGNORECASE)
PROTECTED_TOKEN_RE = re.compile(r"__KEEP_\d+__")


@dataclass(slots=True)
class TranslationOptions:
    mode: str = "AUTO"  # AUTO|FORCE_HTML|FORCE_TEXT
    skip_urls: bool = True
    skip_emails: bool = True
    skip_codes: bool = True
    skip_units: bool = True


@dataclass(slots=True)
class _HtmlNodePlan:
    node: NavigableString
    segment_count: int


@dataclass(slots=True)
class TranslationPlan:
    mode: str  # plain|html|skip
    original: str
    segments: list[str]
    restorers: list[dict[str, str]]
    soup: Any | None = None
    html_nodes: list[_HtmlNodePlan] | None = None


def is_html(text: str) -> bool:
    return bool(HTML_TAG_RE.search(text))


def should_skip_text(text: str, options: TranslationOptions) -> bool:
    return not text.strip()


def _protect_text(text: str, options: TranslationOptions) -> tuple[str, dict[str, str]]:
    patterns: list[re.Pattern[str]] = []
    if options.skip_urls:
        patterns.append(URL_RE)
    if options.skip_emails:
        patterns.append(EMAIL_RE)
    if options.skip_codes:
        patterns.append(SKU_RE)
    if options.skip_units:
        patterns.append(UNIT_RE)

    protected = text
    replacements: dict[str, str] = {}
    token_id = 0

    for pattern in patterns:
        def replacer(match: re.Match[str]) -> str:
            nonlocal token_id
            token = f"__KEEP_{token_id}__"
            token_id += 1
            replacements[token] = match.group(0)
            return token

        protected = pattern.sub(replacer, protected)

    return protected, replacements


def _restore_text(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, original in replacements.items():
        restored = restored.replace(token, original)
    return restored


def _has_translatable_content(text: str) -> bool:
    without_tokens = PROTECTED_TOKEN_RE.sub(" ", text)
    return bool(re.search(r"[A-Za-zÀ-ž]", without_tokens))


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


def build_translation_plan(value: str | None, options: TranslationOptions, max_chars: int) -> TranslationPlan:
    if value is None:
        return TranslationPlan(mode="skip", original="", segments=[], restorers=[])

    text = str(value)
    as_html = options.mode == "FORCE_HTML" or (options.mode == "AUTO" and is_html(text))

    if not as_html:
        if should_skip_text(text, options):
            return TranslationPlan(mode="skip", original=text, segments=[], restorers=[])

        protected, replacements = _protect_text(text, options)
        if not _has_translatable_content(protected):
            return TranslationPlan(mode="skip", original=text, segments=[], restorers=[])

        raw_segments = split_long_text(protected, max_chars=max_chars)
        restorers = [replacements for _ in raw_segments]
        return TranslationPlan(mode="plain", original=text, segments=raw_segments, restorers=restorers)

    soup = BeautifulSoup(text, "lxml")
    html_nodes: list[_HtmlNodePlan] = []
    segments: list[str] = []
    restorers: list[dict[str, str]] = []

    for node in soup.find_all(string=True):
        parent_name = node.parent.name if node.parent else ""
        if parent_name in {"script", "style"}:
            continue
        node_text = str(node)
        if should_skip_text(node_text, options):
            continue

        protected, replacements = _protect_text(node_text, options)
        if not _has_translatable_content(protected):
            continue

        node_segments = split_long_text(protected, max_chars=max_chars)
        segments.extend(node_segments)
        restorers.extend([replacements for _ in node_segments])
        html_nodes.append(_HtmlNodePlan(node=node, segment_count=len(node_segments)))

    if not segments:
        return TranslationPlan(mode="skip", original=text, segments=[], restorers=[])

    return TranslationPlan(mode="html", original=text, segments=segments, restorers=restorers, soup=soup, html_nodes=html_nodes)


def render_translation_plan(plan: TranslationPlan, translated_segments: list[str]) -> str:
    if plan.mode == "skip":
        return plan.original

    restored_segments = [
        _restore_text(segment, repl)
        for segment, repl in zip(translated_segments, plan.restorers)
    ]

    if plan.mode == "plain":
        return "".join(restored_segments)

    if plan.mode != "html" or plan.soup is None or plan.html_nodes is None:
        raise RuntimeError("Invalid translation plan.")

    cursor = 0
    for node_plan in plan.html_nodes:
        translated = "".join(restored_segments[cursor : cursor + node_plan.segment_count])
        cursor += node_plan.segment_count
        node_plan.node.replace_with(translated)

    body = plan.soup.body
    if body:
        return "".join(str(x) for x in body.contents)
    return str(plan.soup)


def translate_plain_text(text: str, translate_fn: Callable[[list[str]], list[str]], options: TranslationOptions, max_chars: int) -> str:
    plan = build_translation_plan(text, options, max_chars=max_chars)
    if plan.mode == "skip":
        return plan.original
    translated = translate_fn(plan.segments)
    return render_translation_plan(plan, translated)


def translate_html_text_nodes(
    html: str,
    translate_fn: Callable[[list[str]], list[str]],
    options: TranslationOptions,
    max_chars: int,
) -> str:
    forced_options = TranslationOptions(
        mode="FORCE_HTML",
        skip_urls=options.skip_urls,
        skip_emails=options.skip_emails,
        skip_codes=options.skip_codes,
        skip_units=options.skip_units,
    )
    plan = build_translation_plan(html, forced_options, max_chars=max_chars)
    if plan.mode == "skip":
        return plan.original
    translated = translate_fn(plan.segments)
    return render_translation_plan(plan, translated)


def translate_cell(
    value: str,
    translate_fn: Callable[[list[str]], list[str]],
    options: TranslationOptions,
    max_chars: int,
) -> str:
    plan = build_translation_plan(value, options, max_chars=max_chars)
    if plan.mode == "skip":
        return plan.original
    translated = translate_fn(plan.segments)
    return render_translation_plan(plan, translated)
