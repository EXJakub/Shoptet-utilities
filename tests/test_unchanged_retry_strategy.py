from translation_quality import assess_translation_quality, is_legit_unchanged
from translator.openai_provider import OpenAIProvider


def _provider_stub() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.use_batch_api = True
    provider.model = "gpt-4o-mini"
    return provider


def test_legit_unchanged_identifier_passes_quality_gate() -> None:
    out = assess_translation_quality("SKU: ABC-12345", "SKU: ABC-12345", "cs", "sk", risk_tier="strict")
    assert is_legit_unchanged("SKU: ABC-12345", "SKU: ABC-12345") is True
    assert out.ok is True


def test_non_legit_unchanged_sentence_fails_quality_gate() -> None:
    text = "Tento produkt je pouze online a skladem."
    out = assess_translation_quality(text, text, "cs", "sk", risk_tier="strict")
    assert out.ok is False
    assert out.code == "unchanged_text"


def test_strict_retry_batch_prompt_contains_anti_unchanged_instruction() -> None:
    provider = _provider_stub()
    prompt = provider._batch_system_prompt(strict_change=True)
    assert "must not be an unchanged copy" in prompt
