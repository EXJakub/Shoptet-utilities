from collections import deque

from translator.openai_provider import OpenAIProvider, _coerce_batch_response


def _provider_stub() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.use_batch_api = True
    provider.model = "gpt-4o-mini"
    provider.max_parallel_requests = 4
    provider._http_requests_total = 0
    provider._translate_calls_total = 0
    provider._events = deque(maxlen=200)
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
    metrics = provider.get_metrics_snapshot()
    assert metrics["fallback_count"] == 1


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


def test_get_metrics_snapshot_exposes_events(monkeypatch) -> None:
    provider = _provider_stub()

    monkeypatch.setattr(provider, "_batch", lambda *args, **kwargs: ["a"])
    out = provider.translate_texts(["x"], "cs", "sk")

    assert out == ["a"]
    metrics = provider.get_metrics_snapshot()
    assert metrics["translate_calls_total"] == 1
    assert metrics["events_count"] == 1
    assert metrics["events"][0]["mode"] == "batch"


def test_translate_text_chunks_parallel_batch_preserves_order(monkeypatch) -> None:
    provider = _provider_stub()
    provider.use_batch_api = True

    async def fake_batch_chunks(chunks, source_lang, target_lang):
        return [[f"tr:{v}" for v in chunk] for chunk in chunks]

    monkeypatch.setattr(provider, "_translate_batch_chunks_async", fake_batch_chunks)

    out = provider.translate_text_chunks([["a", "b"], ["c"]], "cs", "sk")

    assert out == [["tr:a", "tr:b"], ["tr:c"]]
    metrics = provider.get_metrics_snapshot()
    assert metrics["events"][0]["mode"] == "batch_parallel"


def test_translate_text_chunks_fallbacks_to_per_chunk(monkeypatch) -> None:
    provider = _provider_stub()
    provider.use_batch_api = True

    async def failing_batch_chunks(chunks, source_lang, target_lang):
        raise RuntimeError("batch_invalid_shape: x")

    monkeypatch.setattr(provider, "_translate_batch_chunks_async", failing_batch_chunks)
    monkeypatch.setattr(provider, "translate_texts", lambda texts, source_lang, target_lang: [f"fallback:{t}" for t in texts])

    out = provider.translate_text_chunks([["a", "b"], ["c"]], "cs", "sk")

    assert out == [["fallback:a", "fallback:b"], ["fallback:c"]]


def test_batch_payload_contains_strict_array_contract() -> None:
    provider = _provider_stub()

    payload = provider._batch_payload(["a"], "cs", "sk")

    system_msg = payload[0]["content"]
    assert "exactly the same number of items" in system_msg
    assert "No wrapper objects" in system_msg
    assert "__KEEP_0__" in system_msg
