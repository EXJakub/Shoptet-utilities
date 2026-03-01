from __future__ import annotations

import re
from dataclasses import dataclass

LETTER_RE = re.compile(r"[A-Za-zÀ-ž]")
WORD_RE = re.compile(r"[A-Za-zÀ-ž]+")
WS_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CODE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9._/-]*\d[A-Z0-9._/-]*$", re.IGNORECASE)
UNIT_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:%|mm|cm|m|km|g|kg|ml|l|v|w|hz|khz|mhz)$", re.IGNORECASE)

CZECH_HINTS = {
    "který",
    "která",
    "které",
    "protože",
    "zboží",
    "objednávka",
    "pouze",
    "nebo",
    "výhodný",
}

SLOVAK_HINTS = {
    "ktorý",
    "ktorá",
    "ktoré",
    "pretože",
    "tovar",
    "objednávka",
    "iba",
    "alebo",
    "výhodný",
    "môže",
}

CZECH_DIACRITICS = set("ěščřžýáíéťďňů")
SLOVAK_DIACRITICS = set("áäčďéíĺľňóôŕšťúýž")


@dataclass(slots=True)
class QualityCheckResult:
    ok: bool
    code: str
    message: str


def _normalize(text: str) -> str:
    return WS_RE.sub(" ", text.strip()).lower()


def _alpha_len(text: str) -> int:
    return sum(1 for ch in text if LETTER_RE.match(ch))


def _strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def _word_tokens(text: str) -> list[str]:
    html_free = _strip_html(text)
    return WORD_RE.findall(html_free.lower())


def _jaccard_overlap(source_tokens: list[str], translated_tokens: list[str]) -> float:
    src_set = set(source_tokens)
    tr_set = set(translated_tokens)
    if not src_set or not tr_set:
        return 0.0
    return len(src_set & tr_set) / len(src_set | tr_set)


def _ordered_overlap_ratio(source_tokens: list[str], translated_tokens: list[str]) -> float:
    if not source_tokens or not translated_tokens:
        return 0.0

    common_len = min(len(source_tokens), len(translated_tokens))
    same_position = sum(1 for i in range(common_len) if source_tokens[i] == translated_tokens[i])
    return same_position / common_len


def _is_technical_or_boilerplate_only(text: str) -> bool:
    text_no_html = _strip_html(text)
    words = _word_tokens(text_no_html)
    if not words:
        return True

    long_words = [w for w in words if len(w) >= 3]
    if len(long_words) >= 3:
        return False

    raw_tokens = [t for t in re.split(r"\s+", text_no_html.strip()) if t]
    if not raw_tokens:
        return True

    technical_like = 0
    for tok in raw_tokens:
        if URL_RE.search(tok) or CODE_TOKEN_RE.match(tok) or UNIT_TOKEN_RE.match(tok):
            technical_like += 1
            continue
        alpha_only = WORD_RE.fullmatch(tok)
        if alpha_only and len(tok) >= 3:
            continue
        technical_like += 1

    return technical_like / len(raw_tokens) >= 0.75


def assess_translation_quality(source: str, translated: str, source_lang: str, target_lang: str) -> QualityCheckResult:
    if not source.strip() or not translated.strip():
        return QualityCheckResult(ok=False, code="empty_translation", message="Translation output is empty.")

    if source_lang.lower() != "cs" or target_lang.lower() != "sk":
        return QualityCheckResult(ok=True, code="ok", message="Quality gate applies only to CS→SK.")

    src_norm = _normalize(source)
    tr_norm = _normalize(translated)

    if src_norm == tr_norm and _alpha_len(source) >= 10:
        return QualityCheckResult(ok=False, code="unchanged_text", message="Long text is unchanged in CS→SK translation.")

    src_alpha_len = _alpha_len(source)
    if src_alpha_len >= 30 and not (
        _is_technical_or_boilerplate_only(source) and _is_technical_or_boilerplate_only(translated)
    ):
        source_tokens = _word_tokens(source)
        translated_tokens = _word_tokens(translated)
        overlap = max(
            _jaccard_overlap(source_tokens, translated_tokens),
            _ordered_overlap_ratio(source_tokens, translated_tokens),
        )
        if overlap > 0.75:
            return QualityCheckResult(
                ok=False,
                code="high_source_overlap",
                message="Translation has unusually high overlap with source text.",
            )

    words = set(WORD_RE.findall(tr_norm))
    has_czech = bool(words.intersection(CZECH_HINTS))
    has_slovak = bool(words.intersection(SLOVAK_HINTS))
    if has_czech and not has_slovak:
        return QualityCheckResult(ok=False, code="czech_wording", message="Translated text still looks Czech.")

    tr_chars = set(tr_norm)
    if tr_chars.intersection(CZECH_DIACRITICS) and not tr_chars.intersection(SLOVAK_DIACRITICS):
        return QualityCheckResult(ok=False, code="czech_diacritics", message="Translated text uses only Czech-like diacritics.")

    return QualityCheckResult(ok=True, code="ok", message="ok")
