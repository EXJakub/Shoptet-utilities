from __future__ import annotations

import re
from dataclasses import dataclass

LETTER_RE = re.compile(r"[A-Za-zÀ-ž]")
WS_RE = re.compile(r"\s+")

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


def quality_tier_for_complexity(complexity: float) -> str:
    return "strict" if complexity >= 0.6 else "fast"


def assess_translation_quality(
    source: str,
    translated: str,
    source_lang: str,
    target_lang: str,
    risk_tier: str = "strict",
) -> QualityCheckResult:
    if not source.strip() or not translated.strip():
        return QualityCheckResult(ok=False, code="empty_translation", message="Translation output is empty.")

    if source_lang.lower() != "cs" or target_lang.lower() != "sk":
        return QualityCheckResult(ok=True, code="ok", message="Quality gate applies only to CS→SK.")

    src_norm = _normalize(source)
    tr_norm = _normalize(translated)

    unchanged_min = 6 if risk_tier == "strict" else 12
    if src_norm == tr_norm and _alpha_len(source) >= unchanged_min:
        return QualityCheckResult(ok=False, code="unchanged_text", message="Long text is unchanged in CS→SK translation.")

    if risk_tier == "fast":
        return QualityCheckResult(ok=True, code="ok", message="ok")

    words = set(re.findall(r"[A-Za-zÀ-ž]+", tr_norm))
    has_czech = bool(words.intersection(CZECH_HINTS))
    has_slovak = bool(words.intersection(SLOVAK_HINTS))
    if has_czech and not has_slovak:
        return QualityCheckResult(ok=False, code="czech_wording", message="Translated text still looks Czech.")

    tr_chars = set(tr_norm)
    if tr_chars.intersection(CZECH_DIACRITICS) and not tr_chars.intersection(SLOVAK_DIACRITICS):
        return QualityCheckResult(ok=False, code="czech_diacritics", message="Translated text uses only Czech-like diacritics.")

    return QualityCheckResult(ok=True, code="ok", message="ok")
