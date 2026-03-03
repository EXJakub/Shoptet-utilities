from collections import deque

from translator.openai_provider import OpenAIProvider, _coerce_batch_response


def _provider_stub() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.use_batch_api = True
    provider.model = "gpt-4o-mini"
    provider.max_parallel_requests = 4
    provider.partial_recovery_max_attempts = 1
    provider._http_requests_total = 0
    provider._translate_calls_total = 0
    provider._batch_invalid_shape_count = 0
    provider._partial_recovery_count = 0
    provider._isolated_item_fallback_count = 0
    provider._events = deque(maxlen=200)
    return provider


def test_coerce_batch_response_accepts_wrapped_dict() -> None:
    raw = '{"translations": [{"id": "i0", "translated": "a"}, {"id": "i1", "translated": "b"}]}'

    parsed = _coerce_batch_response(raw, expected_order=["i0", "i1"])

    assert parsed == {"i0": "a", "i1": "b"}


def test_coerce_batch_response_accepts_markdown_fence() -> None:
    raw = '```json\n[{"id":"i0","translated":"a"},{"id":"i1","translated":"b"}]\n```'

    parsed = _coerce_batch_response(raw, expected_order=["i0", "i1"])

    assert parsed == {"i0": "a", "i1": "b"}


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
    assert "created_at" in metrics["events"][0]


def test_translate_text_chunks_parallel_batch_preserves_order(monkeypatch) -> None:
    provider = _provider_stub()
    provider.use_batch_api = True

    async def fake_batch_chunks(chunks, source_lang, target_lang, strict_change=False):
        return [[f"tr:{v}" for v in chunk] for chunk in chunks]

    monkeypatch.setattr(provider, "_translate_batch_chunks_async", fake_batch_chunks)

    out = provider.translate_text_chunks([["a", "b"], ["c"]], "cs", "sk")

    assert out == [["tr:a", "tr:b"], ["tr:c"]]
    metrics = provider.get_metrics_snapshot()
    assert metrics["events"][0]["mode"] == "batch_parallel"


def test_translate_text_chunks_fallbacks_to_per_chunk(monkeypatch) -> None:
    provider = _provider_stub()
    provider.use_batch_api = True

    async def failing_batch_chunks(chunks, source_lang, target_lang, strict_change=False):
        raise RuntimeError("batch_invalid_shape: x")

    monkeypatch.setattr(provider, "_translate_batch_chunks_async", failing_batch_chunks)
    monkeypatch.setattr(
        provider,
        "translate_texts",
        lambda texts, source_lang, target_lang, strict_change=False: [f"fallback:{t}" for t in texts],
    )

    out = provider.translate_text_chunks([["a", "b"], ["c"]], "cs", "sk")

    assert out == [["fallback:a", "fallback:b"], ["fallback:c"]]


def test_batch_payload_contains_strict_array_contract() -> None:
    provider = _provider_stub()

    payload = provider._batch_payload([], "cs", "sk")

    system_msg = payload[0]["content"]
    assert "shape {id,translated}" in system_msg
    assert "IDs must be copied exactly" in system_msg
    assert "__KEEP_0__" in system_msg


def test_strict_change_prompt_mentions_no_unchanged_copy() -> None:
    provider = _provider_stub()
    payload = provider._one_payload("Tento produkt je pouze online", "cs", "sk", strict_change=True)
    system_msg = payload[0]["content"]
    assert "do not return an unchanged copy" in system_msg


def test_translate_items_with_recovery_resolves_missing_ids(monkeypatch) -> None:
    provider = _provider_stub()

    class _Item:
        def __init__(self, item_id: str, text: str):
            self.id = item_id
            self.text = text

    calls = {"count": 0}

    async def fake_once(items, source_lang, target_lang, strict_change=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return {items[0].id: f"tr:{items[0].text}"}
        return {item.id: f"tr:{item.text}" for item in items}

    monkeypatch.setattr(provider, "_translate_items_batch_once", fake_once)

    out = __import__("asyncio").run(
        provider._translate_items_with_recovery([_Item("i0", "a"), _Item("i1", "b")], "cs", "sk")
    )
    assert out == {"i0": "tr:a", "i1": "tr:b"}
    assert provider._partial_recovery_count >= 1
