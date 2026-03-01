from types import SimpleNamespace

import pandas as pd

import app
from csv_io import CsvFormat


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _ProviderStub:
    name = "stub"
    model = "m1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def translate_texts(self, texts, source_lang, target_lang):
        self.calls.append(list(texts))
        return [f"new:{text}" for text in texts]


class _CacheStub:
    def __init__(self):
        self._store = {"A": "BAD_CACHE"}
        self._data = {"stub": True}

    def get(self, segment, source_lang, target_lang, provider_name, model):
        return self._store.get(segment)

    def set(self, segment, value, source_lang, target_lang, provider_name, model):
        self._store[segment] = value

    def save(self):
        return None


def test_invalid_cache_hit_is_rejected_when_revalidation_enabled(monkeypatch) -> None:
    provider = _ProviderStub()
    monkeypatch.setattr(app, "_get_provider_for_job", lambda settings: provider)
    monkeypatch.setattr(app, "TranslationCache", _CacheStub)
    monkeypatch.setattr(app, "build_translation_plan", lambda source, options, max_chars: SimpleNamespace(segments=[source]))
    monkeypatch.setattr(app, "render_translation_plan", lambda plan, translated_segments: translated_segments[0])
    monkeypatch.setattr(app, "validate_hrefs", lambda source, translated: (True, "ok"))
    monkeypatch.setattr(app, "validate_structure", lambda source, translated: (True, "ok"))
    monkeypatch.setattr(app, "_autotune_chunk_target", lambda provider: None)

    def fake_assess_translation_quality(source, translated, source_lang, target_lang):
        if translated == "BAD_CACHE":
            return SimpleNamespace(ok=False, code="unchanged_text", message="cache invalid")
        return SimpleNamespace(ok=True, code="ok", message="ok")

    monkeypatch.setattr(app, "assess_translation_quality", fake_assess_translation_quality)

    app.st.session_state = _SessionState(
        job_active=True,
        job_settings={
            "translation_options": {
                "mode": "AUTO",
                "skip_urls": True,
                "skip_emails": True,
                "skip_codes": True,
                "skip_units": True,
            },
            "glossary": {},
            "source_lang": "cs",
            "target_lang": "sk",
            "keep_unsafe": False,
            "max_chars": 1000,
            "per_run_cells": 10,
            "provider_name": "OpenAI",
            "model": "x",
            "use_batch_api": False,
            "max_parallel_requests": 1,
            "revalidate_cache_hits_quality_gate": True,
        },
        job_tasks=[(0, "col")],
        job_cursor=0,
        job_translated_count=0,
        job_error_count=0,
        job_df_out=pd.DataFrame({"col": ["A"]}),
        job_report=[],
        job_quality_report=[],
        job_perf={
            "segments_total": 0,
            "cache_misses_total": 0,
            "unique_misses_total": 0,
            "chunks_sent": 0,
            "chunk_target_chars": 12000,
            "last_events": [],
            "http_requests_total": 0,
            "fallback_count": 0,
            "error_count": 0,
            "success_count": 0,
            "latency_ms_total": 0,
            "input_chars_total": 0,
            "input_items_total": 0,
            "quality_retry_count": 0,
            "quality_fail_count": 0,
        },
        job_chunk_target_chars=12000,
        job_status="running",
    )

    source_df = pd.DataFrame({"col": ["A"]})
    fmt = CsvFormat(encoding="utf-8", delimiter=",")

    app._process_batch(source_df, fmt)

    assert provider.calls == [["A"]]
    assert app.st.session_state.job_df_out.at[0, "col"] == "new:A"
    assert any(item.get("action") == "cache_rejected" for item in app.st.session_state.job_quality_report)
