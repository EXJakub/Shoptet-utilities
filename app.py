from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from io import BytesIO
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cache import TranslationCache
from csv_io import CsvFormat, read_csv_from_upload
from html_translate import HTML_TAG_RE, TranslationOptions, build_translation_plan, render_translation_plan
from job_artifacts import should_checkpoint, write_job_artifacts
from reporting import ReportRecord, make_record, report_to_dataframe
from runtime_config import load_runtime_config, shoptet_from_session
from shoptet_api import ShoptetClient, ShoptetConfig, sync_translated_to_sk
from telemetry import build_run_envelope, build_telemetry_exporter, evaluate_alerts, evaluate_run_gate
from translation_quality import assess_translation_quality, is_legit_unchanged, quality_tier_for_segment
from translator.base import TranslationProvider
from translator.openai_provider import OpenAIProvider
from validators import validate_hrefs, validate_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
RUNTIME_CONFIG = load_runtime_config()
TELEMETRY_EXPORTER = build_telemetry_exporter(RUNTIME_CONFIG.telemetry_backend, RUNTIME_CONFIG.telemetry_jsonl_path)


QUALITY_REPORT_COLUMNS = ["source_hash", "issue", "action", "message", "row_index", "column"]


def get_provider(name: str, model: str, use_batch_api: bool, max_parallel_requests: int) -> TranslationProvider:
    if name == "OpenAI":
        api_key = RUNTIME_CONFIG.openai_api_key
        if not api_key:
            raise ValueError("Chybí OPENAI_API_KEY v .env.")
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            use_batch_api=use_batch_api,
            max_parallel_requests=max_parallel_requests,
            partial_recovery_max_attempts=RUNTIME_CONFIG.batch_partial_recovery_max_attempts,
        )
    raise ValueError("Zvolený provider není implementovaný.")





def _quality_report_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=QUALITY_REPORT_COLUMNS)


def _increment_perf_counter(counter_name: str, key: str) -> None:
    perf = st.session_state.setdefault("job_perf", {})
    counters = perf.setdefault(counter_name, {})
    counters[key] = int(counters.get(key, 0)) + 1


def _append_job_report(record: ReportRecord) -> None:
    st.session_state.job_report.append(record)
    _increment_perf_counter("translation_error_type_counts", record.error_type)


def _append_job_quality_record(record: dict[str, Any]) -> None:
    st.session_state.job_quality_report.append(record)
    issue = str(record.get("issue", "unknown"))
    action = str(record.get("action", "unknown"))
    _increment_perf_counter("quality_issue_counts", issue)
    _increment_perf_counter("quality_action_counts", action)


def _build_run_summary(source_df: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    quality_df = _quality_report_to_dataframe(st.session_state.get("job_quality_report", []))
    report_df = report_to_dataframe(st.session_state.get("job_report", []))
    perf = st.session_state.get("job_perf", {})

    quality_counts = dict(perf.get("quality_issue_counts") or {})
    if not quality_counts and not quality_df.empty:
        quality_counts = quality_df["issue"].value_counts().to_dict()

    report_counts = dict(perf.get("translation_error_type_counts") or {})
    if not report_counts and not report_df.empty:
        report_counts = report_df["error_type"].value_counts().to_dict()

    return {
        "run": {
            "run_id": st.session_state.get("job_run_id"),
            "gate_passed": bool(st.session_state.get("job_run_gate_passed", False)),
            "gate_reasons": list(st.session_state.get("job_run_gate_reasons", [])),
            "publish_blocked_reason": st.session_state.get("job_publish_blocked_reason", ""),
        },
        "source_mode": settings.get("source_mode"),
        "provider": {
            "name": settings.get("provider_name"),
            "model": settings.get("model"),
            "use_batch_api": bool(settings.get("use_batch_api")),
        },
        "languages": {
            "source": settings.get("source_lang"),
            "target": settings.get("target_lang"),
        },
        "input": {
            "row_count": int(len(source_df)),
            "translate_columns": list(settings.get("translate_columns", [])),
        },
        "job": {
            "status": st.session_state.get("job_status"),
            "translated_cells": int(st.session_state.get("job_translated_count", 0)),
            "error_count": int(st.session_state.get("job_error_count", 0)),
        },
        "translation_report": {
            "row_count": int(len(report_df)),
            "status_counts": report_counts,
        },
        "quality_report": {
            "row_count": int(len(quality_df)),
            "issue_counts": quality_counts,
        },
        "performance": st.session_state.get("job_perf", {}),
        "alerts": st.session_state.get("job_alerts", []),
    }


def _build_validation_bundle_zip() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("translated.csv", st.session_state.get("job_translated_csv", b""))
        archive.writestr("translation_report.csv", st.session_state.get("job_report_csv", b""))
        archive.writestr("quality_report.csv", st.session_state.get("job_quality_csv", b""))
        archive.writestr("translation_cache.json", st.session_state.get("job_cache_json", b""))
        archive.writestr("validation_summary.json", st.session_state.get("job_validation_summary_json", b""))
    return buffer.getvalue()


def _finalize_stopped_job(source_df: pd.DataFrame, fmt: CsvFormat) -> None:
    settings: dict[str, Any] = st.session_state.get("job_settings", {})
    if not settings or "job_df_out" not in st.session_state:
        return
    cache = TranslationCache()
    write_job_artifacts(
        state=st.session_state,
        fmt=fmt,
        cache=cache,
        source_df=source_df,
        settings=settings,
        quality_report_to_dataframe=_quality_report_to_dataframe,
        build_run_summary=_build_run_summary,
        build_validation_bundle_zip=_build_validation_bundle_zip,
        include_validation_bundle=True,
    )
    _emit_run_telemetry("stopped")
    if not bool(st.session_state.get("job_run_gate_passed", False)):
        st.session_state["job_publish_blocked_reason"] = ", ".join(
            st.session_state.get("job_run_gate_reasons", [])
        ) or "run_gate_failed"

def _provider_signature(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (
        settings["provider_name"],
        settings["model"],
        bool(settings["use_batch_api"]),
        int(settings["max_parallel_requests"]),
    )


def _get_provider_for_job(settings: dict[str, Any]) -> TranslationProvider:
    signature = _provider_signature(settings)
    cached_signature = st.session_state.get("provider_signature")
    provider = st.session_state.get("provider_instance")
    if provider is not None and cached_signature == signature:
        return provider

    provider = get_provider(*signature)
    st.session_state.provider_signature = signature
    st.session_state.provider_instance = provider
    return provider


def _chunk_by_char_budget(texts: list[str], target_chars: int = 12000, max_items: int = 128) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        text_len = max(1, len(text))
        if current and (len(current) >= max_items or current_chars + text_len > target_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_len
    if current:
        chunks.append(current)
    return chunks


def _chunk_by_complexity_budget(
    items: list[tuple[str, float]],
    target_chars: int = 12000,
    max_items: int = 128,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_weighted_chars = 0.0
    for text, complexity in items:
        # High-complexity segments consume more budget to avoid unstable huge batches.
        weighted = max(1.0, len(text) * (1.0 + complexity))
        if current and (len(current) >= max_items or current_weighted_chars + weighted > target_chars):
            chunks.append(current)
            current = []
            current_weighted_chars = 0.0
        current.append(text)
        current_weighted_chars += weighted
    if current:
        chunks.append(current)
    return chunks



def _safe_provider_metrics(provider: TranslationProvider) -> dict[str, Any]:
    getter = getattr(provider, "get_metrics_snapshot", None)
    if callable(getter):
        return getter()
    return {}


def _translate_chunks_with_policy(
    provider: TranslationProvider,
    chunks: list[list[str]],
    source_lang: str,
    target_lang: str,
    *,
    strict_change: bool = False,
) -> list[list[str]]:
    if not chunks:
        return []
    translate_chunks = getattr(provider, "translate_text_chunks", None)
    if callable(translate_chunks):
        if strict_change:
            try:
                return translate_chunks(chunks, source_lang, target_lang, strict_change=True)
            except TypeError:
                # Provider without policy support: use default behavior.
                return translate_chunks(chunks, source_lang, target_lang)
        return translate_chunks(chunks, source_lang, target_lang)

    if strict_change:
        translated: list[list[str]] = []
        for chunk in chunks:
            translate_texts = getattr(provider, "translate_texts")
            try:
                translated.append(translate_texts(chunk, source_lang, target_lang, strict_change=True))
            except TypeError:
                translated.append(translate_texts(chunk, source_lang, target_lang))
        return translated
    return [provider.translate_texts(chunk, source_lang, target_lang) for chunk in chunks]


def _gate_thresholds() -> dict[str, float]:
    return {
        "quality_fail_ratio": float(RUNTIME_CONFIG.quality_gate_max_fail_ratio),
        "provider_error_ratio": float(RUNTIME_CONFIG.quality_gate_max_provider_error_ratio),
        "batch_shape_error_rate": float(RUNTIME_CONFIG.quality_gate_max_batch_shape_error_rate),
        "p95_latency_ms": float(RUNTIME_CONFIG.quality_gate_max_p95_latency_ms),
        "quality_fail_ratio_min_samples": float(RUNTIME_CONFIG.quality_gate_min_quality_samples),
        "provider_error_ratio_min_samples": 1.0,
        "batch_shape_error_rate_min_samples": float(RUNTIME_CONFIG.quality_gate_min_batch_shape_samples),
        "p95_latency_ms_min_samples": float(RUNTIME_CONFIG.quality_gate_min_latency_samples),
    }


def _p95_latency(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    lat = sorted(max(0, int(e.get("latency_ms", 0))) for e in events)
    if not lat:
        return 0.0
    idx = max(0, int(len(lat) * 0.95) - 1)
    return float(lat[idx])


def _build_quality_diagnostics(perf: dict[str, Any]) -> dict[str, Any]:
    issue_counts = dict(perf.get("quality_issue_counts") or {})
    action_counts = dict(perf.get("quality_action_counts") or {})
    rows = st.session_state.get("job_quality_report", [])
    column_counts: dict[str, int] = {}
    unchanged_column_counts: dict[str, int] = {}
    for row in rows:
        col = str(row.get("column", "") or "").strip()
        if not col:
            continue
        column_counts[col] = column_counts.get(col, 0) + 1
        if str(row.get("issue", "")) == "unchanged_text":
            unchanged_column_counts[col] = unchanged_column_counts.get(col, 0) + 1
    top_columns = sorted(column_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    top_unchanged_columns = sorted(unchanged_column_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "quality_issue_counts": issue_counts,
        "quality_action_counts": action_counts,
        "top_quality_columns": [{"column": col, "count": count} for col, count in top_columns],
        "top_unchanged_columns": [{"column": col, "count": count} for col, count in top_unchanged_columns],
        "quality_retry_count": int(perf.get("quality_retry_count", 0)),
        "quality_fail_count": int(perf.get("quality_fail_count", 0)),
        "quality_sample_size": int(perf.get("quality_segments_total", 0)),
        "cell_reject_count": int(perf.get("cell_reject_count", 0)),
        "unchanged_retry_attempts": int(perf.get("unchanged_retry_attempts", 0)),
        "unchanged_retry_fixed": int(perf.get("unchanged_retry_fixed", 0)),
        "unchanged_retry_failed": int(perf.get("unchanged_retry_failed", 0)),
        "legit_unchanged_count": int(perf.get("legit_unchanged_count", 0)),
    }


def _calc_run_metrics() -> dict[str, float]:
    perf: dict[str, Any] = st.session_state.get("job_perf", {})
    translated_cells = max(1, int(st.session_state.get("job_translated_count", 0)))
    quality_samples = max(0, int(perf.get("quality_segments_total", 0)))
    quality_fail = int(perf.get("quality_fail_count", 0))
    cell_reject_count = int(perf.get("cell_reject_count", 0))
    provider_errors = int(perf.get("error_count", 0))
    recent_events = perf.get("last_events", [])
    run_events = perf.get("run_events", [])
    recent_p95_latency = _p95_latency(recent_events)
    run_p95_latency = _p95_latency(run_events)
    chunks_sent = max(0, int(perf.get("chunks_sent", 0)))
    quality_fail_ratio = float(quality_fail / quality_samples) if quality_samples > 0 else 0.0
    return {
        "quality_fail_ratio": quality_fail_ratio,
        "provider_error_ratio": provider_errors / translated_cells,
        "batch_shape_error_rate": float(perf.get("batch_shape_error_rate", 0.0)),
        "p95_latency_ms": run_p95_latency,
        "recent_p95_latency_ms": recent_p95_latency,
        "run_p95_latency_ms": run_p95_latency,
        "effective_tpm_estimate": float(perf.get("effective_tpm_estimate", 0)),
        "cells_per_min": float(
            int(st.session_state.get("job_translated_count", 0))
            / max(
                0.1,
                (pd.Timestamp.utcnow() - st.session_state.get("job_started_at", pd.Timestamp.utcnow())).total_seconds() / 60.0,
            )
        ),
        "cache_hit_rate": float(
            1.0
            - (
                int(perf.get("cache_misses_total", 0))
                / max(1, int(perf.get("segments_total", 0)))
            )
        ),
        "quality_sample_size": float(quality_samples),
        "provider_error_sample_size": float(translated_cells),
        "latency_sample_size": float(len(run_events)),
        "batch_shape_sample_size": float(chunks_sent),
        "cell_reject_ratio": float(cell_reject_count / max(1, translated_cells)),
        "cell_reject_sample_size": float(translated_cells),
    }


def _emit_run_telemetry(phase: str) -> None:
    metrics = _calc_run_metrics()
    history = st.session_state.setdefault("job_quality_ratio_history", [])
    history.append(
        {
            "ratio": float(metrics.get("quality_fail_ratio", 0.0)),
            "sample_size": int(metrics.get("quality_sample_size", 0.0)),
        }
    )
    history[:] = history[-10:]
    thresholds = _gate_thresholds()
    alerts = evaluate_alerts(metrics, thresholds)
    gate_passed, gate_reasons = evaluate_run_gate(metrics, thresholds)
    st.session_state["job_alerts"] = [a.message for a in alerts]
    st.session_state["job_run_gate_passed"] = gate_passed
    st.session_state["job_run_gate_reasons"] = gate_reasons

    envelope = build_run_envelope(
        run_id=str(st.session_state.get("job_run_id", "")),
        phase=phase,
        status=str(st.session_state.get("job_status", "")),
        metrics=metrics,
        alerts=alerts,
        gate_passed=gate_passed,
        gate_reasons=gate_reasons,
        diagnostics=_build_quality_diagnostics(st.session_state.get("job_perf", {})),
    )
    TELEMETRY_EXPORTER.emit(envelope)


def _should_revalidate_cache_hit(segment: str, complexity: float) -> bool:
    if complexity >= 0.6:
        return True
    sample_rate = float(RUNTIME_CONFIG.cache_revalidation_low_risk_sample_rate)
    if sample_rate <= 0:
        return False
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    sample = (int(digest[:8], 16) % 10000) / 10000.0
    return sample <= sample_rate


def _update_job_perf_metrics(
    provider: TranslationProvider,
    total_segments: int,
    cache_misses: int,
    unique_misses: int,
) -> None:
    perf = st.session_state.setdefault(
        "job_perf",
        {
            "segments_total": 0,
            "cache_misses_total": 0,
            "unique_misses_total": 0,
            "chunks_sent": 0,
            "chunk_target_chars": st.session_state.get("job_chunk_target_chars", 12000),
            "last_events": [],
            "http_requests_total": 0,
            "fallback_count": 0,
            "error_count": 0,
            "success_count": 0,
            "latency_ms_total": 0,
            "input_chars_total": 0,
            "input_items_total": 0,
            "batch_invalid_shape_count": 0,
            "partial_recovery_count": 0,
            "isolated_item_fallback_count": 0,
            "effective_tpm_estimate": 0,
            "batch_shape_error_rate": 0.0,
            "adaptive_parallel_current": st.session_state.get("job_adaptive_parallel_current", 1),
            "quality_segments_total": 0,
            "cell_reject_count": 0,
            "unchanged_retry_attempts": 0,
            "unchanged_retry_fixed": 0,
            "unchanged_retry_failed": 0,
            "legit_unchanged_count": 0,
            "run_events": [],
        },
    )
    perf.setdefault("segments_total", 0)
    perf.setdefault("cache_misses_total", 0)
    perf.setdefault("unique_misses_total", 0)
    perf.setdefault("chunks_sent", 0)
    perf.setdefault("batch_invalid_shape_count", 0)
    perf.setdefault("partial_recovery_count", 0)
    perf.setdefault("isolated_item_fallback_count", 0)
    perf.setdefault("effective_tpm_estimate", 0)
    perf.setdefault("batch_shape_error_rate", 0.0)
    perf.setdefault("adaptive_parallel_current", st.session_state.get("job_adaptive_parallel_current", 1))
    perf.setdefault("quality_segments_total", 0)
    perf.setdefault("cell_reject_count", 0)
    perf.setdefault("unchanged_retry_attempts", 0)
    perf.setdefault("unchanged_retry_fixed", 0)
    perf.setdefault("unchanged_retry_failed", 0)
    perf.setdefault("legit_unchanged_count", 0)
    perf.setdefault("quality_action_counts", {})
    perf.setdefault("run_events", [])
    perf["segments_total"] += total_segments
    perf["cache_misses_total"] += cache_misses
    perf["unique_misses_total"] += unique_misses

    metrics = _safe_provider_metrics(provider)
    if metrics:
        perf["http_requests_total"] = metrics.get("http_requests_total", 0)
        perf["fallback_count"] = metrics.get("fallback_count", 0)
        perf["error_count"] = metrics.get("error_count", 0)
        perf["success_count"] = metrics.get("success_count", 0)
        perf["latency_ms_total"] = metrics.get("total_latency_ms", 0)
        perf["input_chars_total"] = metrics.get("total_input_chars", 0)
        perf["input_items_total"] = metrics.get("total_input_items", 0)
        perf["batch_invalid_shape_count"] = metrics.get("batch_invalid_shape_count", 0)
        perf["partial_recovery_count"] = metrics.get("partial_recovery_count", 0)
        perf["isolated_item_fallback_count"] = metrics.get("isolated_item_fallback_count", 0)
        perf["last_events"] = metrics.get("events", [])[-20:]
        perf["run_events"] = metrics.get("events", [])
        elapsed_minutes = max(0.1, (pd.Timestamp.utcnow() - st.session_state.get("job_started_at", pd.Timestamp.utcnow())).total_seconds() / 60.0)
        token_estimate = int(metrics.get("total_input_chars", 0) / 4)
        perf["effective_tpm_estimate"] = int(token_estimate / elapsed_minutes)
        chunks_sent = max(1, int(perf.get("chunks_sent", 0)))
        perf["batch_shape_error_rate"] = float(perf.get("batch_invalid_shape_count", 0)) / chunks_sent


def _autotune_chunk_target(provider: TranslationProvider) -> None:
    metrics = _safe_provider_metrics(provider)
    events: list[dict[str, Any]] = metrics.get("events", [])
    if not events:
        return

    recent = events[-5:]
    avg_latency = sum(int(e.get("latency_ms", 0)) for e in recent) / len(recent)
    recent_errors = sum(1 for e in recent if not e.get("success", True))
    fallback_used = any(bool(e.get("fallback_used", False)) for e in recent)

    target = int(st.session_state.get("job_chunk_target_chars", 12000))
    perf = st.session_state.setdefault("job_perf", {})
    cooldown = int(RUNTIME_CONFIG.batch_autotune_cooldown_batches)
    batch_idx = int(st.session_state.get("job_batch_index", 0))
    last_batch = int(perf.get("autotune_last_chunk_batch", -999))
    if batch_idx - last_batch <= cooldown:
        return

    downshift_threshold = int(RUNTIME_CONFIG.batch_chunk_downshift_avg_latency_ms)
    upshift_threshold = int(RUNTIME_CONFIG.batch_chunk_upshift_avg_latency_ms)

    if recent_errors > 0 or fallback_used or avg_latency > downshift_threshold:
        target = max(6000, int(target * 0.9))
    elif avg_latency < upshift_threshold and recent_errors == 0:
        target = min(20000, int(target * 1.05))
    else:
        return

    st.session_state.job_chunk_target_chars = target
    perf["chunk_target_chars"] = target
    perf["autotune_last_chunk_batch"] = batch_idx


def _autotune_parallelism(provider: TranslationProvider) -> None:
    metrics = _safe_provider_metrics(provider)
    events: list[dict[str, Any]] = metrics.get("events", [])
    if not events:
        return

    recent = events[-8:]
    p95_latency = sorted(int(e.get("latency_ms", 0)) for e in recent)[max(0, int(len(recent) * 0.95) - 1)]
    recent_errors = sum(1 for e in recent if not e.get("success", True))
    recent_fallbacks = sum(1 for e in recent if bool(e.get("fallback_used", False)))
    settings = st.session_state.get("job_settings", {})
    current = int(st.session_state.get("job_adaptive_parallel_current", settings.get("max_parallel_requests", 1)))

    min_parallel = int(RUNTIME_CONFIG.batch_min_parallel)
    max_parallel = int(min(RUNTIME_CONFIG.batch_max_parallel, settings.get("max_parallel_requests", 1)))
    perf = st.session_state.setdefault("job_perf", {})
    cooldown = int(RUNTIME_CONFIG.batch_autotune_cooldown_batches)
    batch_idx = int(st.session_state.get("job_batch_index", 0))
    last_batch = int(perf.get("autotune_last_parallel_batch", -999))
    if batch_idx - last_batch <= cooldown:
        return

    downshift_threshold = int(RUNTIME_CONFIG.batch_parallel_downshift_p95_ms)
    upshift_threshold = int(RUNTIME_CONFIG.batch_parallel_upshift_p95_ms)
    quality_threshold = float(RUNTIME_CONFIG.quality_gate_max_fail_ratio)
    quality_min_samples = int(RUNTIME_CONFIG.quality_gate_min_quality_samples)
    quality_history = st.session_state.get("job_quality_ratio_history", [])
    recent_quality = quality_history[-3:]
    quality_downshift = (
        len(recent_quality) == 3
        and all(int(item.get("sample_size", 0)) >= quality_min_samples for item in recent_quality)
        and all(float(item.get("ratio", 0.0)) >= quality_threshold for item in recent_quality)
    )
    quality_upshift_ok = (
        len(recent_quality) == 3
        and all(int(item.get("sample_size", 0)) >= quality_min_samples for item in recent_quality)
        and all(float(item.get("ratio", 0.0)) <= (quality_threshold * 0.5) for item in recent_quality)
    )

    if recent_errors > 0 or recent_fallbacks > 0 or p95_latency > downshift_threshold or quality_downshift:
        current = max(min_parallel, max(1, current // 2))
    elif p95_latency < upshift_threshold and recent_errors == 0 and recent_fallbacks == 0 and quality_upshift_ok:
        current = min(max_parallel, current + 1)
    else:
        return
    st.session_state["job_adaptive_parallel_current"] = current
    perf["adaptive_parallel_current"] = current
    perf["autotune_last_parallel_batch"] = batch_idx


def _apply_safe_mode_guard() -> None:
    history: list[float] = st.session_state.setdefault("job_fallback_rate_history", [])
    perf: dict[str, Any] = st.session_state.get("job_perf", {})
    shape_errors = int(perf.get("batch_invalid_shape_count", 0))
    chunks_sent = int(perf.get("chunks_sent", 0))
    fallback_rate = (shape_errors / chunks_sent) if chunks_sent > 0 else 0.0
    history.append(fallback_rate)
    history[:] = history[-6:]
    if len(history) < 3:
        return
    recent = history[-3:]
    threshold = float(RUNTIME_CONFIG.batch_safe_mode_fallback_threshold)
    if all(rate >= threshold for rate in recent):
        st.session_state["job_adaptive_parallel_current"] = max(
            int(RUNTIME_CONFIG.batch_min_parallel),
            int(st.session_state.get("job_adaptive_parallel_current", 1)) // 2,
        )
        st.session_state["job_chunk_target_chars"] = max(
            6000,
            int(st.session_state.get("job_chunk_target_chars", 12000) * 0.85),
        )
        st.session_state["job_safe_mode_reason"] = (
            f"high_batch_shape_error_rate ({recent[-1]*100:.1f}% >= {threshold*100:.1f}%)"
        )


def _render_performance_panel() -> None:
    perf: dict[str, Any] = st.session_state.get("job_perf", {})
    if not perf:
        return

    st.subheader("Performance")
    segments_total = int(perf.get("segments_total", 0))
    misses_total = int(perf.get("cache_misses_total", 0))
    unique_misses = int(perf.get("unique_misses_total", 0))
    hit_rate = 0.0
    if segments_total:
        hit_rate = 1 - (misses_total / segments_total)

    cols = st.columns(4)
    cols[0].metric("Cache hit rate", f"{hit_rate*100:.1f}%")
    cols[1].metric("HTTP requests", str(perf.get("http_requests_total", 0)))
    cols[2].metric("Fallbacks", str(perf.get("fallback_count", 0)))
    cols[3].metric("Chunk target chars", str(perf.get("chunk_target_chars", 0)))

    cols2 = st.columns(4)
    cols2[0].metric("Segments", str(segments_total))
    cols2[1].metric("Misses", str(misses_total))
    cols2[2].metric("Unique misses", str(unique_misses))
    cols2[3].metric("Provider errors", str(perf.get("error_count", 0)))

    cols3 = st.columns(4)
    cols3[0].metric("Estimated TPM", str(perf.get("effective_tpm_estimate", 0)))
    cols3[1].metric("Batch shape error rate", f"{float(perf.get('batch_shape_error_rate', 0.0))*100:.1f}%")
    cols3[2].metric("Adaptive parallel", str(perf.get("adaptive_parallel_current", 0)))
    cols3[3].metric("Partial recoveries", str(perf.get("partial_recovery_count", 0)))

    run_p95 = _p95_latency(perf.get("run_events", []))
    recent_p95 = _p95_latency(perf.get("last_events", []))
    cols4 = st.columns(3)
    cols4[0].metric("Run P95 latency (ms)", f"{run_p95:.0f}")
    cols4[1].metric("Recent P95 latency (ms)", f"{recent_p95:.0f}")
    cols4[2].metric("Quality sample size", str(perf.get("quality_segments_total", 0)))

    translated_cells = max(1, int(st.session_state.get("job_translated_count", 0)))
    cell_reject_ratio = int(perf.get("cell_reject_count", 0)) / translated_cells
    cols5 = st.columns(3)
    cols5[0].metric("Cell reject ratio", f"{cell_reject_ratio*100:.1f}%")
    cols5[1].metric("Unchanged retry fixed", str(perf.get("unchanged_retry_fixed", 0)))
    cols5[2].metric("Legit unchanged", str(perf.get("legit_unchanged_count", 0)))

    events = perf.get("last_events", [])
    if events:
        st.caption("Recent OpenAI calls")
        st.dataframe(pd.DataFrame(events), width="stretch")
    safe_reason = st.session_state.get("job_safe_mode_reason", "")
    if safe_reason:
        st.warning(f"Safe mode active: {safe_reason}")
    alerts = st.session_state.get("job_alerts", [])
    if alerts:
        st.error("Alerts: " + " | ".join(alerts))



def _rebalance_chunks_for_parallelism(chunks: list[list[str]], desired_chunks: int) -> list[list[str]]:
    if desired_chunks <= 1 or len(chunks) >= desired_chunks:
        return chunks

    flat: list[str] = [item for chunk in chunks for item in chunk]
    if len(flat) <= 1:
        return chunks

    desired = min(desired_chunks, len(flat))
    base = len(flat) // desired
    extra = len(flat) % desired
    out: list[list[str]] = []
    cursor = 0
    for i in range(desired):
        size = base + (1 if i < extra else 0)
        out.append(flat[cursor:cursor + size])
        cursor += size
    return out

def column_stats(df: pd.DataFrame, column: str) -> tuple[int, int, int]:
    series = df[column].fillna("").astype(str)
    non_empty = int((series.str.strip() != "").sum())
    html_count = int(series.str.contains(HTML_TAG_RE, regex=True).sum())
    href_count = int(series.str.contains("href=", case=False, regex=False).sum())
    return non_empty, html_count, href_count


def _job_reset() -> None:
    for key in [
        "job_active",
        "job_status",
        "job_df_out",
        "job_report",
        "job_tasks",
        "job_cursor",
        "job_translated_count",
        "job_error_count",
        "job_settings",
        "job_translated_csv",
        "job_report_csv",
        "job_cache_json",
        "job_quality_report",
        "job_quality_csv",
        "provider_signature",
        "provider_instance",
        "job_perf",
        "job_chunk_target_chars",
        "job_started_at",
        "job_validation_summary_json",
        "job_validation_bundle_zip",
        "job_batch_index",
        "job_checkpoint_every_batches",
        "job_adaptive_parallel_current",
        "job_fallback_rate_history",
        "job_run_id",
        "job_run_gate_passed",
        "job_run_gate_reasons",
        "job_publish_blocked_reason",
        "job_alerts",
        "job_safe_mode_reason",
        "job_quality_ratio_history",
    ]:
        st.session_state.pop(key, None)


def _build_tasks(df: pd.DataFrame, translate_columns: list[str], skip_columns: list[str]) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    for row_idx in range(len(df)):
        for col in translate_columns:
            if col in skip_columns:
                continue
            source = str(df.at[row_idx, col] or "")
            if source.strip():
                tasks.append((row_idx, col))
    return tasks


def _ensure_job(df: pd.DataFrame, settings: dict[str, Any], translate_columns: list[str], skip_columns: list[str]) -> None:
    st.session_state.job_active = True
    st.session_state.job_status = "running"
    st.session_state.job_run_id = str(uuid4())
    st.session_state.job_df_out = df.copy()
    st.session_state.job_report = []
    st.session_state.job_quality_report = []
    st.session_state.job_tasks = _build_tasks(df, translate_columns, skip_columns)
    st.session_state.job_cursor = 0
    st.session_state.job_translated_count = 0
    st.session_state.job_error_count = 0
    st.session_state.job_settings = settings
    st.session_state.job_chunk_target_chars = 12000
    st.session_state.job_started_at = pd.Timestamp.utcnow()
    st.session_state.job_perf = {
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
        "quality_segments_total": 0,
        "cell_reject_count": 0,
        "unchanged_retry_attempts": 0,
        "unchanged_retry_fixed": 0,
        "unchanged_retry_failed": 0,
        "legit_unchanged_count": 0,
        "batch_invalid_shape_count": 0,
        "partial_recovery_count": 0,
        "isolated_item_fallback_count": 0,
        "effective_tpm_estimate": 0,
        "batch_shape_error_rate": 0.0,
        "adaptive_parallel_current": st.session_state.get("job_adaptive_parallel_current", 0),
        "translation_error_type_counts": {},
        "quality_issue_counts": {},
        "quality_action_counts": {},
        "run_events": [],
        "autotune_last_parallel_batch": -999,
        "autotune_last_chunk_batch": -999,
    }
    st.session_state.job_validation_summary_json = b""
    st.session_state.job_validation_bundle_zip = b""
    st.session_state.job_batch_index = 0
    st.session_state.job_checkpoint_every_batches = RUNTIME_CONFIG.checkpoint_every_batches
    st.session_state.job_adaptive_parallel_current = max(
        RUNTIME_CONFIG.batch_min_parallel,
        min(RUNTIME_CONFIG.batch_max_parallel, int(settings.get("max_parallel_requests", 1))),
    )
    st.session_state.job_fallback_rate_history = []
    st.session_state.job_run_gate_passed = False
    st.session_state.job_run_gate_reasons = []
    st.session_state.job_publish_blocked_reason = ""
    st.session_state.job_alerts = []
    st.session_state.job_safe_mode_reason = ""
    st.session_state.job_quality_ratio_history = []


def _process_batch(source_df: pd.DataFrame, fmt: CsvFormat) -> None:
    if not st.session_state.get("job_active"):
        return

    settings: dict[str, Any] = st.session_state.job_settings
    provider = _get_provider_for_job(settings)
    adaptive_parallel = int(st.session_state.get("job_adaptive_parallel_current", settings.get("max_parallel_requests", 1)))
    if hasattr(provider, "max_parallel_requests"):
        provider.max_parallel_requests = max(1, adaptive_parallel)
    options = TranslationOptions(**settings["translation_options"])
    glossary: dict[str, str] = settings["glossary"]
    source_lang = settings["source_lang"]
    target_lang = settings["target_lang"]
    keep_unsafe = settings["keep_unsafe"]
    max_chars = int(settings["max_chars"])
    per_run_cells = int(settings["per_run_cells"])
    revalidate_cache_hits_quality_gate = bool(settings.get("revalidate_cache_hits_quality_gate", False))

    cache = TranslationCache()
    tasks: list[tuple[int, str]] = st.session_state.job_tasks
    end_cursor = min(st.session_state.job_cursor + per_run_cells, len(tasks))

    task_records: list[dict[str, Any]] = []
    all_segments: list[tuple[str, float, str]] = []
    for idx in range(st.session_state.job_cursor, end_cursor):
        row_idx, col = tasks[idx]
        source = str(source_df.at[row_idx, col] or "")
        plan = build_translation_plan(source, options, max_chars=max_chars)
        task_records.append({"row_idx": row_idx, "col": col, "source": source, "plan": plan})
        plan_mode = getattr(plan, "mode", "plain")
        segment_meta = getattr(plan, "segment_meta", None) or [{"segment_type": plan_mode, "complexity": 0.2} for _ in plan.segments]
        for segment, meta in zip(plan.segments, segment_meta):
            all_segments.append(
                (
                    segment,
                    float(meta.get("complexity", 0.2)),
                    str(meta.get("segment_type", plan_mode) or plan_mode),
                )
            )

    unique_misses: list[tuple[str, float, str]] = []
    unique_seen: dict[str, tuple[float, str]] = {}
    cache_miss_count = 0
    for segment, complexity, segment_type in all_segments:
        cached = cache.get(segment, source_lang, target_lang, provider.name, provider.model)
        if cached is not None:
            should_revalidate = revalidate_cache_hits_quality_gate or _should_revalidate_cache_hit(segment, complexity)
            if should_revalidate:
                cached_quality = assess_translation_quality(
                    segment,
                    cached,
                    source_lang,
                    target_lang,
                    risk_tier=quality_tier_for_segment(complexity, segment_type),
                )
                if not cached_quality.ok:
                    _append_job_quality_record(
                        {
                            "source_hash": hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16],
                            "issue": cached_quality.code,
                            "action": "cache_rejected",
                            "message": cached_quality.message,
                        }
                    )
                    cache_miss_count += 1
                    prev = unique_seen.get(segment)
                    if prev is None or complexity > prev[0]:
                        unique_seen[segment] = (complexity, segment_type)
            continue
        cache_miss_count += 1
        prev = unique_seen.get(segment)
        if prev is None or complexity > prev[0]:
            unique_seen[segment] = (complexity, segment_type)
    unique_misses = [(segment, complexity, segment_type) for segment, (complexity, segment_type) in unique_seen.items()]

    segment_meta_map = {segment: {"complexity": complexity, "segment_type": segment_type} for segment, complexity, segment_type in unique_misses}
    if unique_misses:
        st.session_state.job_perf["quality_segments_total"] += len(unique_misses)
        target_chars = int(st.session_state.get("job_chunk_target_chars", 12000))
        batch_max_items = 24 if bool(settings.get("use_batch_api")) else 96
        miss_chunks = _chunk_by_complexity_budget(
            [(segment, complexity) for segment, complexity, _ in unique_misses],
            target_chars=target_chars,
            max_items=batch_max_items,
        )

        if bool(settings.get("use_batch_api")):
            max_parallel = max(1, int(st.session_state.get("job_adaptive_parallel_current", settings.get("max_parallel_requests", 1))))
            soft_desired = max(1, len(unique_misses) // 10)
            desired_chunks = min(max_parallel, soft_desired)
            miss_chunks = _rebalance_chunks_for_parallelism(miss_chunks, desired_chunks=desired_chunks)

        st.session_state.job_perf["chunks_sent"] += len(miss_chunks)
        translated_chunks = _translate_chunks_with_policy(
            provider,
            miss_chunks,
            source_lang,
            target_lang,
            strict_change=False,
        )

        retry_candidates: list[tuple[str, str, str, float, str]] = []
        for miss_chunk, translated in zip(miss_chunks, translated_chunks):
            for orig, tr in zip(miss_chunk, translated):
                fixed = glossary.get(tr, tr)
                segment_meta = segment_meta_map.get(orig, {"complexity": 0.2, "segment_type": "plain"})
                complexity = float(segment_meta.get("complexity", 0.2))
                segment_type = str(segment_meta.get("segment_type", "plain"))
                quality = assess_translation_quality(
                    orig,
                    fixed,
                    source_lang,
                    target_lang,
                    risk_tier=quality_tier_for_segment(complexity, segment_type),
                )
                if quality.ok:
                    if is_legit_unchanged(orig, fixed):
                        st.session_state.job_perf["legit_unchanged_count"] += 1
                    cache.set(orig, fixed, source_lang, target_lang, provider.name, provider.model)
                    continue

                st.session_state.job_perf["quality_retry_count"] += 1
                retry_candidates.append((orig, quality.code, quality.message, complexity, segment_type))

        if retry_candidates:
            def _retry_candidates_group(
                candidates: list[tuple[str, str, str, float, str]],
                *,
                strict_change: bool,
            ) -> dict[str, str]:
                if not candidates:
                    return {}
                retry_texts = [(orig, complexity) for orig, _, _, complexity, _ in candidates]
                retry_max_items = 24 if bool(settings.get("use_batch_api")) else 64
                retry_target_chars = max(4000, target_chars // 2)
                retry_chunks = _chunk_by_complexity_budget(
                    retry_texts,
                    target_chars=retry_target_chars,
                    max_items=retry_max_items,
                )
                if bool(settings.get("use_batch_api")):
                    max_parallel = max(1, int(settings.get("max_parallel_requests", 1)))
                    soft_desired = max(1, len(retry_texts) // 12)
                    desired_chunks = min(max_parallel, soft_desired)
                    retry_chunks = _rebalance_chunks_for_parallelism(retry_chunks, desired_chunks=desired_chunks)

                st.session_state.job_perf["chunks_sent"] += len(retry_chunks)
                retry_translated_chunks = _translate_chunks_with_policy(
                    provider,
                    retry_chunks,
                    source_lang,
                    target_lang,
                    strict_change=strict_change,
                )
                retry_results: dict[str, str] = {}
                for retry_chunk, retry_translated in zip(retry_chunks, retry_translated_chunks):
                    for orig, tr in zip(retry_chunk, retry_translated):
                        retry_results[orig] = glossary.get(tr, tr)
                return retry_results

            unchanged_candidates = [c for c in retry_candidates if c[1] == "unchanged_text"]
            normal_candidates = [c for c in retry_candidates if c[1] != "unchanged_text"]
            st.session_state.job_perf["unchanged_retry_attempts"] += len(unchanged_candidates)

            retry_results: dict[str, str] = {}
            retry_results.update(_retry_candidates_group(normal_candidates, strict_change=False))
            retry_results.update(_retry_candidates_group(unchanged_candidates, strict_change=True))

            for orig, initial_issue, initial_message, complexity, segment_type in retry_candidates:
                retry_fixed = retry_results.get(orig, orig)
                retry_quality = assess_translation_quality(
                    orig,
                    retry_fixed,
                    source_lang,
                    target_lang,
                    risk_tier=quality_tier_for_segment(complexity, segment_type),
                )
                if retry_quality.ok:
                    if is_legit_unchanged(orig, retry_fixed):
                        st.session_state.job_perf["legit_unchanged_count"] += 1
                    cache.set(orig, retry_fixed, source_lang, target_lang, provider.name, provider.model)
                    _append_job_quality_record(
                        {
                            "source_hash": hashlib.sha256(orig.encode("utf-8")).hexdigest()[:16],
                            "issue": initial_issue,
                            "action": "retry_fixed",
                            "message": initial_message,
                        }
                    )
                    if initial_issue == "unchanged_text":
                        st.session_state.job_perf["unchanged_retry_fixed"] += 1
                else:
                    st.session_state.job_perf["quality_fail_count"] += 1
                    _append_job_quality_record(
                        {
                            "source_hash": hashlib.sha256(orig.encode("utf-8")).hexdigest()[:16],
                            "issue": retry_quality.code,
                            "action": "not_cached",
                            "message": retry_quality.message,
                        }
                    )
                    if initial_issue == "unchanged_text":
                        st.session_state.job_perf["unchanged_retry_failed"] += 1

    _update_job_perf_metrics(
        provider=provider,
        total_segments=len(all_segments),
        cache_misses=cache_miss_count,
        unique_misses=len(unique_misses),
    )
    _autotune_chunk_target(provider)
    _autotune_parallelism(provider)
    _apply_safe_mode_guard()
    _emit_run_telemetry("checkpoint")

    for record in task_records:
        row_idx = record["row_idx"]
        col = record["col"]
        source = record["source"]
        plan = record["plan"]
        try:
            translated_segments = [
                cache.get(segment, source_lang, target_lang, provider.name, provider.model) or segment for segment in plan.segments
            ]
            translated_value = render_translation_plan(plan, translated_segments)
            meta = getattr(plan, "segment_meta", None) or []
            avg_complexity = (sum(float(m.get("complexity", 0.2)) for m in meta) / len(meta)) if meta else 0.2
            segment_type = "html" if getattr(plan, "mode", "plain") == "html" else "plain"
            quality = assess_translation_quality(
                source,
                translated_value,
                source_lang,
                target_lang,
                risk_tier=quality_tier_for_segment(avg_complexity, segment_type),
            )
            if not quality.ok:
                st.session_state.job_error_count += 1
                st.session_state.job_perf["cell_reject_count"] += 1
                _append_job_report(make_record(row_idx, col, "quality_gate_failed", quality.message, source))
                _append_job_quality_record(
                    {
                        "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
                        "issue": quality.code,
                        "action": "cell_rejected",
                        "message": quality.message,
                        "row_index": row_idx,
                        "column": col,
                    }
                )
                if not keep_unsafe:
                    translated_value = source

            href_ok, href_msg = validate_hrefs(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
            struct_ok, struct_msg = validate_structure(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
            if not (href_ok and struct_ok):
                st.session_state.job_error_count += 1
                _append_job_report(
                    make_record(
                        row_idx,
                        col,
                        "href_mismatch" if not href_ok else "structure_mismatch",
                        f"{href_msg}; {struct_msg}",
                        source,
                    )
                )
                if not keep_unsafe:
                    translated_value = source
            st.session_state.job_df_out.at[row_idx, col] = translated_value
            st.session_state.job_translated_count += 1
        except Exception as exc:
            logger.exception("Translation failure row=%s col=%s", row_idx, col)
            st.session_state.job_error_count += 1
            _append_job_report(make_record(row_idx, col, "api_error", str(exc), source))
            st.session_state.job_df_out.at[row_idx, col] = source

    st.session_state.job_cursor = end_cursor
    st.session_state.job_batch_index = int(st.session_state.get("job_batch_index", 0)) + 1
    cache.save()
    if st.session_state.job_cursor >= len(tasks):
        st.session_state.job_status = "completed"
    total_batches = (len(tasks) + per_run_cells - 1) // per_run_cells if per_run_cells > 0 else 0
    should_write_artifacts = should_checkpoint(
        batch_index=int(st.session_state.get("job_batch_index", 1)),
        total_batches=total_batches,
        checkpoint_every_batches=int(st.session_state.get("job_checkpoint_every_batches", 1)),
        force=st.session_state.job_status == "completed",
    )
    if should_write_artifacts:
        write_job_artifacts(
            state=st.session_state,
            fmt=fmt,
            cache=cache,
            source_df=source_df,
            settings=settings,
            quality_report_to_dataframe=_quality_report_to_dataframe,
            build_run_summary=_build_run_summary,
            build_validation_bundle_zip=_build_validation_bundle_zip,
            include_validation_bundle=st.session_state.job_status == "completed",
        )
    if st.session_state.job_status == "completed":
        _emit_run_telemetry("completed")
        if not bool(st.session_state.get("job_run_gate_passed", False)):
            st.session_state["job_publish_blocked_reason"] = ", ".join(
                st.session_state.get("job_run_gate_reasons", [])
            ) or "run_gate_failed"


def _build_shoptet_client(prefix: str) -> ShoptetClient:
    runtime = shoptet_from_session(st.session_state, prefix)
    return ShoptetClient(
        ShoptetConfig(
            base_url=runtime.base_url,
            token=runtime.token,
            products_endpoint=runtime.products_endpoint,
            ean_field=runtime.ean_field,
            id_field=runtime.id_field,
            page_size=runtime.page_size,
        )
    )


def main() -> None:
    st.set_page_config(page_title="CSV/API překladač (HTML-safe)", layout="wide")
    st.title("Překladač CZ → SK (CSV i Shoptet API)")

    source_mode = st.radio("Zdroj dat", ["CSV", "Shoptet API"], horizontal=True)

    input_df = pd.DataFrame()
    fmt = CsvFormat(encoding="utf-8-sig", delimiter=";", quotechar='"')

    if source_mode == "CSV":
        uploaded = st.file_uploader("Nahraj CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.csv_file_bytes = uploaded.getvalue()

        file_bytes = st.session_state.get("csv_file_bytes")
        if not file_bytes:
            return

        _, detected_fmt = read_csv_from_upload(file_bytes)
        st.subheader("Detekce vstupu")
        enc = st.selectbox("Encoding", [detected_fmt.encoding, "utf-8-sig", "utf-8"], index=0)
        delim = st.selectbox("Delimiter", [detected_fmt.delimiter, ";", ",", "\t"], index=0)
        quote = st.selectbox("Quotechar", ['"', "'"], index=0)
        input_df, _ = read_csv_from_upload(file_bytes, encoding=enc, delimiter=delim, quotechar=quote)
        fmt = CsvFormat(encoding=enc, delimiter=delim, quotechar=quote)
    else:
        st.subheader("Shoptet API konfigurace")
        st.session_state.setdefault("cz_base_url", os.getenv("CZ_SHOPTET_BASE_URL", ""))
        st.session_state.setdefault("cz_token", os.getenv("CZ_SHOPTET_API_TOKEN", ""))
        st.session_state.setdefault("sk_base_url", os.getenv("SK_SHOPTET_BASE_URL", ""))
        st.session_state.setdefault("sk_token", os.getenv("SK_SHOPTET_API_TOKEN", ""))
        st.session_state.setdefault("cz_products_endpoint", "/api/products")
        st.session_state.setdefault("sk_products_endpoint", "/api/products")
        st.session_state.setdefault("api_ean_field", "ean")
        st.session_state.setdefault("api_id_field", "id")
        st.session_state.setdefault("api_page_size", 100)

        st.text_input("CZ Base URL", key="cz_base_url")
        st.text_input("CZ API token", key="cz_token", type="password")
        st.text_input("CZ Products endpoint", key="cz_products_endpoint")
        st.text_input("SK Base URL", key="sk_base_url")
        st.text_input("SK API token", key="sk_token", type="password")
        st.text_input("SK Products endpoint", key="sk_products_endpoint")
        st.text_input("EAN field", key="api_ean_field")
        st.text_input("SK ID field", key="api_id_field")
        st.number_input("Page size", min_value=10, max_value=500, step=10, key="api_page_size")

        if st.button("Načíst CZ produkty z API"):
            try:
                cz_client = _build_shoptet_client("cz")
                input_df = cz_client.fetch_products_df()
                if input_df.empty:
                    st.warning("CZ API vrátilo prázdná data.")
                st.session_state.api_loaded_df = input_df
                st.success(f"Načteno {len(input_df)} produktů z CZ API")
            except Exception as exc:
                st.error(f"Načtení CZ API selhalo: {exc}")
                return

        if "api_loaded_df" not in st.session_state:
            return
        input_df = st.session_state.api_loaded_df.copy()

    st.write(f"Řádků: {len(input_df)} | Sloupců: {len(input_df.columns)}")
    st.dataframe(input_df.head(20), width="stretch")

    columns = list(input_df.columns)
    st.subheader("Výběr sloupců")
    translate_columns = st.multiselect("Sloupce k překladu", options=columns)
    skip_columns = st.multiselect("Sloupce k vynechání", options=columns)

    if translate_columns:
        stats = []
        for col in translate_columns:
            non_empty, html_count, href_count = column_stats(input_df, col)
            stats.append({"column": col, "non_empty": non_empty, "html_cells": html_count, "href_cells": href_count})
        st.dataframe(pd.DataFrame(stats), width="stretch")

    st.subheader("Nastavení překladu")
    provider_name = st.selectbox("Provider", ["OpenAI"])
    model = st.text_input("Model", value=RUNTIME_CONFIG.openai_model)
    source_lang = st.text_input("Source language", value="cs")
    target_lang = st.text_input("Target language", value="sk")
    max_chars = st.number_input("Max chars per request", min_value=100, max_value=8000, value=1200, step=100)
    per_run_cells = st.number_input("Počet buněk na jeden běh", min_value=1, max_value=500, value=120, step=1)
    html_mode = st.selectbox("HTML režim", ["AUTO", "FORCE_HTML", "FORCE_TEXT"])
    use_batch_api = st.checkbox("Použít batch API", value=True)
    max_parallel_requests = st.number_input("Max paralelních OpenAI requestů (batch i non-batch)", min_value=1, max_value=64, value=16)

    skip_urls = st.checkbox("URL v textu neměnit", value=True)
    skip_emails = st.checkbox("E-maily neměnit", value=True)
    skip_codes = st.checkbox("SKU/EAN/kódy neměnit", value=True)
    skip_units = st.checkbox("Jednotky neměnit", value=True)
    keep_unsafe = st.checkbox("Keep translated anyway (unsafe)", value=False)
    revalidate_cache_hits_quality_gate = st.checkbox("Revalidovat cache hity quality gate", value=True)
    glossary_json = st.text_area("Glossary JSON", value='{}')

    col_start, col_pause, col_resume, col_stop = st.columns(4)
    start_clicked = col_start.button("Start / Restart", type="primary")
    pause_clicked = col_pause.button("Pause")
    resume_clicked = col_resume.button("Resume")
    stop_clicked = col_stop.button("Stop")

    if pause_clicked and st.session_state.get("job_active"):
        st.session_state.job_status = "paused"
    if resume_clicked and st.session_state.get("job_active"):
        st.session_state.job_status = "running"
        st.rerun()
    if stop_clicked and st.session_state.get("job_active"):
        st.session_state.job_status = "stopped"
        try:
            _finalize_stopped_job(input_df, fmt)
        except Exception as exc:  # pragma: no cover - defensive UI safeguard
            st.error(f"Stop finalize selhal: {exc}")
        st.session_state.job_active = False

    if start_clicked:
        if not translate_columns:
            st.error("Vyber aspoň jeden sloupec k překladu.")
            return
        try:
            glossary = json.loads(glossary_json)
            if not isinstance(glossary, dict):
                raise ValueError
            _ = get_provider(provider_name, model, use_batch_api=use_batch_api, max_parallel_requests=int(max_parallel_requests))
        except Exception as exc:
            st.error(str(exc))
            return

        settings = {
            "provider_name": provider_name,
            "model": model,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "max_chars": int(max_chars),
            "use_batch_api": use_batch_api,
            "max_parallel_requests": int(max_parallel_requests),
            "keep_unsafe": keep_unsafe,
            "revalidate_cache_hits_quality_gate": revalidate_cache_hits_quality_gate,
            "glossary": glossary,
            "per_run_cells": int(per_run_cells),
            "translation_options": {
                "mode": html_mode,
                "skip_urls": skip_urls,
                "skip_emails": skip_emails,
                "skip_codes": skip_codes,
                "skip_units": skip_units,
            },
            "source_mode": source_mode,
            "translate_columns": translate_columns,
        }
        _ensure_job(input_df, settings, translate_columns, skip_columns)

    if st.session_state.get("job_active"):
        _render_performance_panel()
        total = len(st.session_state.job_tasks)
        done = st.session_state.job_cursor
        st.progress((done / total) if total else 1.0)
        st.info(f"Stav: {st.session_state.job_status} | Hotovo: {done}/{total} | Přeloženo: {st.session_state.job_translated_count} | Chyb: {st.session_state.job_error_count}")

        if st.session_state.job_status == "running":
            _process_batch(input_df, fmt)
            if st.session_state.job_status != "completed":
                st.rerun()

    if st.session_state.get("job_status") in {"completed", "stopped"}:
        _render_performance_panel()
        if st.session_state.get("job_status") == "completed":
            st.success("Překlad dokončen")
        else:
            done = int(st.session_state.get("job_cursor", 0))
            total = len(st.session_state.get("job_tasks", []))
            translated = int(st.session_state.get("job_translated_count", 0))
            errors = int(st.session_state.get("job_error_count", 0))
            st.warning(
                "Běh byl zastaven. Exporty byly vygenerovány z aktuálního stavu. "
                f"Hotovo: {done}/{total} | Přeloženo: {translated} | Chyb: {errors}"
            )
        gate_passed = bool(st.session_state.get("job_run_gate_passed", False))
        gate_reasons = st.session_state.get("job_run_gate_reasons", [])
        if gate_passed:
            st.success("Run gate: PASS")
        else:
            st.error("Run gate: FAIL" + (f" ({', '.join(gate_reasons)})" if gate_reasons else ""))
        st.download_button("Stáhnout translated.csv", data=BytesIO(st.session_state.job_translated_csv), file_name="translated.csv", mime="text/csv")
        st.download_button("Stáhnout translation_report.csv", data=BytesIO(st.session_state.job_report_csv), file_name="translation_report.csv", mime="text/csv")
        st.download_button("Stáhnout translation_cache.json", data=BytesIO(st.session_state.job_cache_json), file_name="translation_cache.json", mime="application/json")
        st.download_button("Stáhnout quality_report.csv", data=BytesIO(st.session_state.job_quality_csv), file_name="quality_report.csv", mime="text/csv")
        st.download_button("Stáhnout validation_summary.json", data=BytesIO(st.session_state.job_validation_summary_json), file_name="validation_summary.json", mime="application/json")
        st.download_button("Stáhnout validation_bundle.zip", data=BytesIO(st.session_state.job_validation_bundle_zip), file_name="validation_bundle.zip", mime="application/zip")

        if st.session_state.job_settings.get("source_mode") == "Shoptet API":
            st.subheader("Update SK produktů podle EAN")
            if not gate_passed:
                st.warning(
                    "Publish blocked: run-level quality gate failed. "
                    + (f"Důvody: {', '.join(gate_reasons)}" if gate_reasons else "")
                )
            if st.button("Update SK verzi (PATCH podle EAN)"):
                try:
                    if not gate_passed:
                        st.error("SK update zablokován run-level quality gate pravidly.")
                        return
                    sk_client = _build_shoptet_client("sk")
                    sync_errors: list[dict[str, Any]] = []
                    updated, missing = sync_translated_to_sk(
                        sk_client=sk_client,
                        translated_df=st.session_state.job_df_out,
                        columns_to_sync=st.session_state.job_settings["translate_columns"],
                        ean_field=st.session_state["api_ean_field"],
                        report_errors=sync_errors,
                        max_error_ratio=RUNTIME_CONFIG.sync_max_error_ratio,
                    )
                    sk_metrics = sk_client.get_metrics_snapshot()
                    st.success(f"SK update hotov: updated={updated}, missing_ean={missing}")
                    sync_error_ratio = float(sk_metrics.get("error_ratio", 0.0))
                    if sync_error_ratio > float(RUNTIME_CONFIG.sync_max_error_ratio):
                        st.error(
                            f"SK sync error ratio too high ({sync_error_ratio:.2%}); "
                            f"threshold {RUNTIME_CONFIG.sync_max_error_ratio:.2%}"
                        )
                    if sync_errors:
                        st.dataframe(pd.DataFrame(sync_errors), width="stretch")
                except Exception as exc:
                    st.error(f"SK update selhal: {exc}")


if __name__ == "__main__":
    main()
