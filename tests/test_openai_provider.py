from translator.openai_provider import OpenAIProvider, _coerce_batch_response


def _provider_stub() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.use_batch_api = True
    provider.model = "gpt-4o-mini"
    provider.max_parallel_requests = 4
    return provider


def test_coerce_batch_response_accepts_wrapped_dict() -> None:
    raw = '{"translations": ["a", "b"]}'

    parsed = _coerce_batch_response(raw, expected_len=2)

    assert parsed == ["a", "b"]


def test_coerce_batch_response_accepts_markdown_fence() -> None:
    raw = '```json\n["a", "b"]\n```'

    parsed = _coerce_batch_response(raw, expected_len=2)

    assert parsed == ["a", "b"]


def test_translate_texts_falls_back_when_batch_shape_invalid(monkeypatch) -> None:
    provider = _provider_stub()

    monkeypatch.setattr(provider, "_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("batch_invalid_shape: x")))

    async def fake_parallel(texts, source_lang, target_lang):
        return [f"tr:{t}" for t in texts]

    monkeypatch.setattr(provider, "_translate_parallel_async", fake_parallel)

    translated = provider.translate_texts(["one", "two"], "cs", "sk")

    assert translated == ["tr:one", "tr:two"]


def test_translate_texts_raises_non_batch_runtime_error(monkeypatch) -> None:
    provider = _provider_stub()
    monkeypatch.setattr(provider, "_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("other_error")))

    async def fake_parallel(texts, source_lang, target_lang):
        return ["never"]

    monkeypatch.setattr(provider, "_translate_parallel_async", fake_parallel)

    # non batch_* runtime errors should not be swallowed
    try:
        provider.translate_texts(["one"], "cs", "sk")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "other_error"
