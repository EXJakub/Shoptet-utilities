from translation_quality import assess_translation_quality


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
