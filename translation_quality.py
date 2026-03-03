from __future__ import annotations

import re
from dataclasses import dataclass

LETTER_RE = re.compile(r"[A-Za-zÀ-ž]")
WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+$")
EMAIL_RE = re.compile(r"^[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}$")
CODE_RE = re.compile(r"^(?:SKU|EAN|ID|IP)?[\s:#-]*[A-Z0-9-]{3,}$")
UNIT_RE = re.compile(r"^\d+(?:[.,]\d+)?\s?(?:mm|cm|m|kg|g|W|kW|V|A|mAh|°C|%)$", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"__KEEP_\d+__")

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

CZ_SK_LEXEME_RULES: tuple[tuple[re.Pattern[str], re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bzp[eě]tn(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)?\b", re.IGNORECASE),
        re.compile(r"\bsp[aä]tn(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)?\b", re.IGNORECASE),
        "zpětný->spätný",
    ),
    (
        re.compile(r"\bzved[aá]k(?:y|u|em|ům|ama|ách)?\b", re.IGNORECASE),
        re.compile(r"\bzdvih[aá]k(?:y|u|om|mi|och)?\b", re.IGNORECASE),
        "zvedák->zdvihák",
    ),
    (
        re.compile(r"\b(?:čern|cern)(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        re.compile(r"\bčiern(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        "černý->čierny",
    ),
    (
        re.compile(r"\b(?:šed|sed)(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        re.compile(r"\bsiv(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        "šedý->sivý",
    ),
    (
        re.compile(r"\b(?:bíl|bil)(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        re.compile(r"\bbiel(?:[ýyáaéeíi]|[ýy]ch|[ýy]mi)\b", re.IGNORECASE),
        "bílý->biely",
    ),
)


@dataclass(slots=True)
class QualityCheckResult:
    ok: bool
    code: str
    message: str


def _normalize(text: str) -> str:
    return WS_RE.sub(" ", text.strip()).lower()


def _alpha_len(text: str) -> int:
    return sum(1 for ch in text if LETTER_RE.match(ch))


def _residual_czech_lexemes(source: str, translated: str) -> list[str]:
    source_plain = PLACEHOLDER_RE.sub(" ", source)
    translated_plain = PLACEHOLDER_RE.sub(" ", translated)
    hits: list[str] = []
    for cz_re, sk_re, label in CZ_SK_LEXEME_RULES:
        if cz_re.search(source_plain) and cz_re.search(translated_plain) and not sk_re.search(translated_plain):
            hits.append(label)
    return hits


def is_legit_unchanged(source: str, translated: str) -> bool:
    src_norm = _normalize(source)
    tr_norm = _normalize(translated)
    if not src_norm or src_norm != tr_norm:
        return False

    plain = PLACEHOLDER_RE.sub(" ", source).strip()
    alpha_len = _alpha_len(plain)
    if alpha_len == 0:
        return True

    if URL_RE.fullmatch(plain) or EMAIL_RE.fullmatch(plain):
        return True
    if UNIT_RE.fullmatch(plain):
        return True
    if CODE_RE.fullmatch(plain) and any(ch.isdigit() for ch in plain):
        return True

    words = re.findall(r"[A-Za-zÀ-ž0-9-]+", plain)
    if words and len(words) <= 2 and alpha_len <= 18:
        # Short brand-like labels often legitimately stay unchanged in CZ/SK.
        if all(w.isupper() or w[:1].isupper() or any(ch.isdigit() for ch in w) for w in words):
            return True
    if words and len(words) <= 2 and 3 <= alpha_len <= 12:
        # CZ/SK share many short technical and catalog words (e.g. "adaptér", "black").
        # Keep the rule conservative: no Czech-specific hints and no punctuation-heavy strings.
        token_set = {w.lower() for w in words}
        if not token_set.intersection(CZECH_HINTS) and plain.replace(" ", "").isalnum():
            return True
    return False


def quality_tier_for_segment(complexity: float, segment_type: str = "plain") -> str:
    # HTML and very complex segments are riskier; keep stricter guardrails there.
    if segment_type == "html" and complexity >= 0.45:
        return "strict"
    return "strict" if complexity >= 0.6 else "fast"


def quality_tier_for_complexity(complexity: float) -> str:
    return quality_tier_for_segment(complexity, segment_type="plain")


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
    if src_norm == tr_norm and _alpha_len(source) >= unchanged_min and not is_legit_unchanged(source, translated):
        return QualityCheckResult(ok=False, code="unchanged_text", message="Long text is unchanged in CS→SK translation.")

    residual_lexemes = _residual_czech_lexemes(source, translated)
    if residual_lexemes:
        detail = ", ".join(residual_lexemes[:3])
        return QualityCheckResult(
            ok=False,
            code="czech_lexeme_residual",
            message=f"Translated text still contains Czech lexemes with Slovak equivalents ({detail}).",
        )

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
