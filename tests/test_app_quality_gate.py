from __future__ import annotations

from io import BytesIO

import pandas as pd

import app
from csv_io import CsvFormat


class _FakeProvider:
    name = "OpenAI"
    model = "fake-model"

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return list(texts)

    def translate_text_chunks(self, chunks: list[list[str]], source_lang: str, target_lang: str) -> list[list[str]]:
        return [list(chunk) for chunk in chunks]

    def get_metrics_snapshot(self) -> dict[str, object]:
        return {"events": []}


class _FakeCache:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, text: str, source_lang: str, target_lang: str, provider: str, model: str) -> str | None:
        return self._data.get(text)

    def set(self, text: str, translated: str, source_lang: str, target_lang: str, provider: str, model: str) -> None:
        self._data[text] = translated

    def save(self) -> None:
        return None


class _SessionState(dict):
    def __getattr__(self, item: str) -> object:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


class _FakeSt:
    def __init__(self) -> None:
        self.session_state = _SessionState()


def test_quality_fail_is_recorded_in_translation_report_and_increments_error_count(monkeypatch) -> None:
    fake_st = _FakeSt()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "TranslationCache", _FakeCache)

    provider = _FakeProvider()
    source_df = pd.DataFrame({"description": ["který produkt je pouze skladem"]})

    settings = {
        "provider_name": "OpenAI",
        "model": "fake-model",
        "source_lang": "cs",
        "target_lang": "sk",
        "max_chars": 500,
        "use_batch_api": False,
        "max_parallel_requests": 1,
        "keep_unsafe": False,
        "glossary": {},
        "per_run_cells": 1,
        "translation_options": {
            "mode": "AUTO",
            "skip_urls": True,
            "skip_emails": True,
            "skip_codes": True,
            "skip_units": True,
        },
    }

    fake_st.session_state.update(
        {
            "job_active": True,
            "job_status": "running",
            "job_df_out": source_df.copy(),
            "job_report": [],
            "job_quality_report": [],
            "job_tasks": [(0, "description")],
            "job_cursor": 0,
            "job_translated_count": 0,
            "job_error_count": 0,
                "job_settings": settings,
                "job_perf": {
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
            "job_chunk_target_chars": 12000,
            "provider_signature": app._provider_signature(settings),
            "provider_instance": provider,
        }
    )

    app._process_batch(source_df, CsvFormat(encoding="utf-8", delimiter=";", quotechar='"'))

    assert fake_st.session_state["job_error_count"] == 1

    report_df = pd.read_csv(BytesIO(fake_st.session_state["job_report_csv"]))
    assert "quality_gate_failed" in set(report_df["error_type"])

    quality_df = pd.read_csv(BytesIO(fake_st.session_state["job_quality_csv"]))
    assert {"row_index", "column"}.issubset(set(quality_df.columns))
