from translation_quality import assess_translation_quality, is_legit_unchanged


def test_cs_sk_detects_unchanged_long_text() -> None:
    out = assess_translation_quality("Toto je dlouhý český popis produktu", "Toto je dlouhý český popis produktu", "cs", "sk")

    assert out.ok is False
    assert out.code == "unchanged_text"


def test_cs_sk_accepts_slovak_wording() -> None:
    out = assess_translation_quality("Tento produkt je pouze online", "Tento produkt je iba online", "cs", "sk")

    assert out.ok is True


def test_quality_gate_is_noop_for_other_pairs() -> None:
    out = assess_translation_quality("Hello", "Hello", "en", "de")

    assert out.ok is True
    assert out.code == "ok"


def test_legit_unchanged_url_is_allowed() -> None:
    out = assess_translation_quality("https://example.com/product?id=123", "https://example.com/product?id=123", "cs", "sk")
    assert out.ok is True
    assert is_legit_unchanged("https://example.com/product?id=123", "https://example.com/product?id=123") is True


def test_long_natural_language_unchanged_is_not_legit() -> None:
    text = "Tento produkt je pouze skladem na centrálním skladu"
    assert is_legit_unchanged(text, text) is False


def test_cs_sk_detects_residual_czech_lexeme_in_short_text() -> None:
    out = assess_translation_quality("Zpětný ventil PVC-U", "Zpětný ventil pre bazén PVC-U", "cs", "sk", risk_tier="fast")
    assert out.ok is False
    assert out.code == "czech_lexeme_residual"


def test_cs_sk_detects_residual_czech_lexeme_in_long_text() -> None:
    source = "Černý kryt je určen pro bazén a zpětný ventil."
    translated = "Černý kryt je určený pre bazén a spätný ventil."
    out = assess_translation_quality(source, translated, "cs", "sk", risk_tier="strict")
    assert out.ok is False
    assert out.code == "czech_lexeme_residual"


def test_cs_sk_accepts_lexeme_when_slovak_variant_used() -> None:
    out = assess_translation_quality("Zvedák termokrytu", "Zdvihák termokrytu", "cs", "sk", risk_tier="fast")
    assert out.ok is True
