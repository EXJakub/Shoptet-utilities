from translation_quality import assess_translation_quality


def test_cs_sk_detects_unchanged_long_text() -> None:
    out = assess_translation_quality("Toto je dlouhý český popis produktu", "Toto je dlouhý český popis produktu", "cs", "sk")

    assert out.ok is False
    assert out.code == "unchanged_text"


def test_cs_sk_detects_partially_translated_with_high_overlap() -> None:
    out = assess_translation_quality(
        "Tento produkt je vhodný pro zahradu a obsahuje kvalitní materiál s dlouhou životností",
        "Tento produkt je vhodný pre záhradu a obsahuje kvalitní materiál s dlouhou životností",
        "cs",
        "sk",
    )

    assert out.ok is False
    assert out.code == "high_source_overlap"


def test_cs_sk_accepts_slovak_wording() -> None:
    out = assess_translation_quality(
        "Tento produkt je pouze online a může být doručen zítra",
        "Tento produkt je iba online a môže byť doručený zajtra",
        "cs",
        "sk",
    )

    assert out.ok is True


def test_quality_gate_is_noop_for_other_pairs() -> None:
    out = assess_translation_quality("Hello", "Hello", "en", "de")

    assert out.ok is True
    assert out.code == "ok"
