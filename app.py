from __future__ import annotations

import hashlib
import json
import logging
import os
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cache import TranslationCache
from csv_io import CsvFormat, dataframe_to_csv_bytes, read_csv_from_upload
from html_translate import HTML_TAG_RE, TranslationOptions, build_translation_plan, render_translation_plan
from reporting import ReportRecord, make_record, report_to_dataframe
from shoptet_api import ShoptetClient, ShoptetConfig, sync_translated_to_sk
from translation_quality import assess_translation_quality
from translator.base import TranslationProvider
from translator.openai_provider import OpenAIProvider
from validators import validate_hrefs, validate_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


QUALITY_REPORT_COLUMNS = ["source_hash", "issue", "action", "message", "row_index", "column"]


def get_provider(name: str, model: str, use_batch_api: bool, max_parallel_requests: int) -> TranslationProvider:
    if name == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Chybí OPENAI_API_KEY v .env.")
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            use_batch_api=use_batch_api,
            max_parallel_requests=max_parallel_requests,
        )
    raise ValueError("Zvolený provider není implementovaný.")





def _quality_report_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=QUALITY_REPORT_COLUMNS)

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



def _safe_provider_metrics(provider: TranslationProvider) -> dict[str, Any]:
    getter = getattr(provider, "get_metrics_snapshot", None)
    if callable(getter):
        return getter()
    return {}


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
        },
    )
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
        perf["last_events"] = metrics.get("events", [])[-20:]


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
    if recent_errors > 0 or fallback_used or avg_latency > 7500:
        target = max(6000, int(target * 0.9))
    elif avg_latency < 3000 and recent_errors == 0:
        target = min(20000, int(target * 1.05))

    st.session_state.job_chunk_target_chars = target
    if "job_perf" in st.session_state:
        st.session_state.job_perf["chunk_target_chars"] = target


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

    events = perf.get("last_events", [])
    if events:
        st.caption("Recent OpenAI calls")
        st.dataframe(pd.DataFrame(events), width="stretch")



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
    }


def _process_batch(source_df: pd.DataFrame, fmt: CsvFormat) -> None:
    if not st.session_state.get("job_active"):
        return

    settings: dict[str, Any] = st.session_state.job_settings
    provider = _get_provider_for_job(settings)
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
    all_segments: list[str] = []
    for idx in range(st.session_state.job_cursor, end_cursor):
        row_idx, col = tasks[idx]
        source = str(source_df.at[row_idx, col] or "")
        plan = build_translation_plan(source, options, max_chars=max_chars)
        task_records.append({"row_idx": row_idx, "col": col, "source": source, "plan": plan})
        all_segments.extend(plan.segments)

    unique_misses: list[str] = []
    unique_seen: set[str] = set()
    cache_miss_count = 0
    for segment in all_segments:
        cached = cache.get(segment, source_lang, target_lang, provider.name, provider.model)
        if cached is not None:
            if revalidate_cache_hits_quality_gate:
                cached_quality = assess_translation_quality(segment, cached, source_lang, target_lang)
                if not cached_quality.ok:
                    st.session_state.job_quality_report.append(
                        {
                            "source_hash": hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16],
                            "issue": cached_quality.code,
                            "action": "cache_rejected",
                            "message": cached_quality.message,
                        }
                    )
                    cache_miss_count += 1
                    if segment not in unique_seen:
                        unique_seen.add(segment)
                        unique_misses.append(segment)
            continue
        cache_miss_count += 1
        if segment not in unique_seen:
            unique_seen.add(segment)
            unique_misses.append(segment)

    if unique_misses:
        target_chars = int(st.session_state.get("job_chunk_target_chars", 12000))
        batch_max_items = 24 if bool(settings.get("use_batch_api")) else 96
        miss_chunks = _chunk_by_char_budget(unique_misses, target_chars=target_chars, max_items=batch_max_items)

        if bool(settings.get("use_batch_api")):
            max_parallel = max(1, int(settings.get("max_parallel_requests", 1)))
            soft_desired = max(1, len(unique_misses) // 12)
            desired_chunks = min(max_parallel, soft_desired)
            miss_chunks = _rebalance_chunks_for_parallelism(miss_chunks, desired_chunks=desired_chunks)

        st.session_state.job_perf["chunks_sent"] += len(miss_chunks)
        translate_chunks = getattr(provider, "translate_text_chunks", None)
        if callable(translate_chunks):
            translated_chunks = translate_chunks(miss_chunks, source_lang, target_lang)
        else:
            translated_chunks = [provider.translate_texts(chunk, source_lang, target_lang) for chunk in miss_chunks]

        retry_candidates: list[tuple[str, str, str]] = []
        for miss_chunk, translated in zip(miss_chunks, translated_chunks):
            for orig, tr in zip(miss_chunk, translated):
                fixed = glossary.get(tr, tr)
                quality = assess_translation_quality(orig, fixed, source_lang, target_lang)
                if quality.ok:
                    cache.set(orig, fixed, source_lang, target_lang, provider.name, provider.model)
                    continue

                st.session_state.job_perf["quality_retry_count"] += 1
                retry_candidates.append((orig, quality.code, quality.message))

        if retry_candidates:
            retry_texts = [orig for orig, _, _ in retry_candidates]
            retry_max_items = 24 if bool(settings.get("use_batch_api")) else 64
            retry_target_chars = max(4000, target_chars // 2)
            retry_chunks = _chunk_by_char_budget(retry_texts, target_chars=retry_target_chars, max_items=retry_max_items)

            if bool(settings.get("use_batch_api")):
                max_parallel = max(1, int(settings.get("max_parallel_requests", 1)))
                soft_desired = max(1, len(retry_texts) // 12)
                desired_chunks = min(max_parallel, soft_desired)
                retry_chunks = _rebalance_chunks_for_parallelism(retry_chunks, desired_chunks=desired_chunks)

            st.session_state.job_perf["chunks_sent"] += len(retry_chunks)
            if callable(translate_chunks):
                retry_translated_chunks = translate_chunks(retry_chunks, source_lang, target_lang)
            else:
                retry_translated_chunks = [provider.translate_texts(chunk, source_lang, target_lang) for chunk in retry_chunks]

            retry_results: dict[str, str] = {}
            for retry_chunk, retry_translated in zip(retry_chunks, retry_translated_chunks):
                for orig, tr in zip(retry_chunk, retry_translated):
                    retry_results[orig] = glossary.get(tr, tr)

            for orig, initial_issue, initial_message in retry_candidates:
                retry_fixed = retry_results.get(orig, orig)
                retry_quality = assess_translation_quality(orig, retry_fixed, source_lang, target_lang)
                if retry_quality.ok:
                    cache.set(orig, retry_fixed, source_lang, target_lang, provider.name, provider.model)
                    st.session_state.job_quality_report.append(
                        {
                            "source_hash": hashlib.sha256(orig.encode("utf-8")).hexdigest()[:16],
                            "issue": initial_issue,
                            "action": "retry_fixed",
                            "message": initial_message,
                        }
                    )
                else:
                    st.session_state.job_perf["quality_fail_count"] += 1
                    st.session_state.job_quality_report.append(
                        {
                            "source_hash": hashlib.sha256(orig.encode("utf-8")).hexdigest()[:16],
                            "issue": retry_quality.code,
                            "action": "not_cached",
                            "message": retry_quality.message,
                        }
                    )

    _update_job_perf_metrics(
        provider=provider,
        total_segments=len(all_segments),
        cache_misses=cache_miss_count,
        unique_misses=len(unique_misses),
    )
    _autotune_chunk_target(provider)

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
            quality = assess_translation_quality(source, translated_value, source_lang, target_lang)
            if not quality.ok:
                st.session_state.job_error_count += 1
                st.session_state.job_report.append(
                    make_record(row_idx, col, "quality_gate_failed", quality.message, source)
                )
                st.session_state.job_quality_report.append(
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
                st.session_state.job_report.append(
                    make_record(row_idx, col, "href_mismatch" if not href_ok else "structure_mismatch", f"{href_msg}; {struct_msg}", source)
                )
                if not keep_unsafe:
                    translated_value = source
            st.session_state.job_df_out.at[row_idx, col] = translated_value
            st.session_state.job_translated_count += 1
        except Exception as exc:
            logger.exception("Translation failure row=%s col=%s", row_idx, col)
            st.session_state.job_error_count += 1
            st.session_state.job_report.append(make_record(row_idx, col, "api_error", str(exc), source))
            st.session_state.job_df_out.at[row_idx, col] = source

    st.session_state.job_cursor = end_cursor
    cache.save()
    if st.session_state.job_cursor >= len(tasks):
        st.session_state.job_status = "completed"

    st.session_state.job_translated_csv = dataframe_to_csv_bytes(st.session_state.job_df_out, fmt)
    report_df = report_to_dataframe(st.session_state.job_report)
    st.session_state.job_report_csv = report_df.to_csv(index=False).encode(fmt.encoding)
    quality_df = _quality_report_to_dataframe(st.session_state.job_quality_report)
    st.session_state.job_quality_csv = quality_df.to_csv(index=False).encode(fmt.encoding)
    st.session_state.job_cache_json = json.dumps(cache._data, ensure_ascii=False, indent=2).encode("utf-8")


def _build_shoptet_client(prefix: str) -> ShoptetClient:
    return ShoptetClient(
        ShoptetConfig(
            base_url=st.session_state[f"{prefix}_base_url"],
            token=st.session_state[f"{prefix}_token"],
            products_endpoint=st.session_state[f"{prefix}_products_endpoint"],
            ean_field=st.session_state["api_ean_field"],
            id_field=st.session_state["api_id_field"],
            page_size=int(st.session_state["api_page_size"]),
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
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
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
    revalidate_cache_hits_quality_gate = st.checkbox("Revalidovat cache hity quality gate", value=False)
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

    if st.session_state.get("job_status") == "completed":
        _render_performance_panel()
        st.success("Překlad dokončen")
        st.download_button("Stáhnout translated.csv", data=BytesIO(st.session_state.job_translated_csv), file_name="translated.csv", mime="text/csv")
        st.download_button("Stáhnout translation_report.csv", data=BytesIO(st.session_state.job_report_csv), file_name="translation_report.csv", mime="text/csv")
        st.download_button("Stáhnout translation_cache.json", data=BytesIO(st.session_state.job_cache_json), file_name="translation_cache.json", mime="application/json")
        st.download_button("Stáhnout quality_report.csv", data=BytesIO(st.session_state.job_quality_csv), file_name="quality_report.csv", mime="text/csv")

        if st.session_state.job_settings.get("source_mode") == "Shoptet API":
            st.subheader("Update SK produktů podle EAN")
            if st.button("Update SK verzi (PATCH podle EAN)"):
                try:
                    sk_client = _build_shoptet_client("sk")
                    sync_errors: list[dict[str, Any]] = []
                    updated, missing = sync_translated_to_sk(
                        sk_client=sk_client,
                        translated_df=st.session_state.job_df_out,
                        columns_to_sync=st.session_state.job_settings["translate_columns"],
                        ean_field=st.session_state["api_ean_field"],
                        report_errors=sync_errors,
                    )
                    st.success(f"SK update hotov: updated={updated}, missing_ean={missing}")
                    if sync_errors:
                        st.dataframe(pd.DataFrame(sync_errors), width="stretch")
                except Exception as exc:
                    st.error(f"SK update selhal: {exc}")


if __name__ == "__main__":
    main()
