from __future__ import annotations

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
from html_translate import HTML_TAG_RE, TranslationOptions, translate_cell
from reporting import ReportRecord, make_record, report_to_dataframe
from translator.base import TranslationProvider
from translator.openai_provider import OpenAIProvider
from validators import validate_hrefs, validate_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_provider(name: str, model: str, use_batch_api: bool) -> TranslationProvider:
    if name == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Chybí OPENAI_API_KEY v .env.")
        return OpenAIProvider(api_key=api_key, model=model, use_batch_api=use_batch_api)
    raise ValueError("Zvolený provider není implementovaný.")


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
        "job_skipped_count",
        "job_error_count",
        "job_settings",
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
    st.session_state.job_tasks = _build_tasks(df, translate_columns, skip_columns)
    st.session_state.job_cursor = 0
    st.session_state.job_translated_count = 0
    st.session_state.job_skipped_count = 0
    st.session_state.job_error_count = 0
    st.session_state.job_settings = settings


def _process_batch(
    source_df: pd.DataFrame,
    fmt: CsvFormat,
) -> None:
    if not st.session_state.get("job_active"):
        return

    settings: dict[str, Any] = st.session_state.job_settings
    provider = get_provider(settings["provider_name"], settings["model"], settings["use_batch_api"])
    options = TranslationOptions(**settings["translation_options"])
    glossary: dict[str, str] = settings["glossary"]
    source_lang = settings["source_lang"]
    target_lang = settings["target_lang"]
    keep_unsafe = settings["keep_unsafe"]
    max_chars = int(settings["max_chars"])
    per_run_cells = int(settings["per_run_cells"])

    cache = TranslationCache()

    def translate_with_cache(chunks: list[str]) -> list[str]:
        results: list[str] = []
        misses: list[str] = []
        miss_positions: list[int] = []
        for i, ch in enumerate(chunks):
            cached = cache.get(ch, source_lang, target_lang, provider.name, provider.model)
            if cached is not None:
                results.append(cached)
            else:
                results.append("")
                misses.append(ch)
                miss_positions.append(i)
        if misses:
            translated = provider.translate_texts(misses, source_lang, target_lang)
            for pos, orig, tr in zip(miss_positions, misses, translated):
                fixed = glossary.get(tr, tr)
                results[pos] = fixed
                cache.set(orig, fixed, source_lang, target_lang, provider.name, provider.model)
        return results

    tasks: list[tuple[int, str]] = st.session_state.job_tasks
    end_cursor = min(st.session_state.job_cursor + per_run_cells, len(tasks))

    for idx in range(st.session_state.job_cursor, end_cursor):
        row_idx, col = tasks[idx]
        source = str(source_df.at[row_idx, col] or "")

        try:
            translated_value = translate_cell(source, translate_with_cache, options, max_chars=max_chars)
            href_ok, href_msg = validate_hrefs(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
            struct_ok, struct_msg = validate_structure(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
            if not (href_ok and struct_ok):
                st.session_state.job_error_count += 1
                st.session_state.job_report.append(
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
            st.session_state.job_report.append(make_record(row_idx, col, "api_error", str(exc), source))
            st.session_state.job_df_out.at[row_idx, col] = source

    st.session_state.job_cursor = end_cursor
    cache.save()

    if st.session_state.job_cursor >= len(tasks):
        st.session_state.job_status = "completed"

    translated_bytes = dataframe_to_csv_bytes(st.session_state.job_df_out, fmt)
    report_df = report_to_dataframe(st.session_state.job_report)
    report_bytes = report_df.to_csv(index=False).encode(fmt.encoding)

    st.session_state.job_translated_csv = translated_bytes
    st.session_state.job_report_csv = report_bytes
    st.session_state.job_cache_json = json.dumps(cache._data, ensure_ascii=False, indent=2).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="CSV překladač (HTML-safe)", layout="wide")
    st.title("CSV překladač CZ → SK (ochrana HTML a href)")

    uploaded = st.file_uploader("Nahraj CSV", type=["csv"])
    if not uploaded:
        _job_reset()
        return

    file_bytes = uploaded.getvalue()
    _, detected_fmt = read_csv_from_upload(file_bytes)

    st.subheader("Detekce vstupu")
    enc = st.selectbox("Encoding", [detected_fmt.encoding, "utf-8-sig", "utf-8"], index=0)
    delim = st.selectbox("Delimiter", [detected_fmt.delimiter, ";", ",", "\t"], index=0)
    quote = st.selectbox("Quotechar", ['"', "'"], index=0)

    df, fmt = read_csv_from_upload(file_bytes, encoding=enc, delimiter=delim, quotechar=quote)
    fmt = CsvFormat(encoding=enc, delimiter=delim, quotechar=quote)

    st.write(f"Řádků: {len(df)} | Sloupců: {len(df.columns)}")
    st.dataframe(df.head(20), use_container_width=True)

    columns = list(df.columns)
    st.subheader("Výběr sloupců")
    translate_columns = st.multiselect("Sloupce k překladu", options=columns)
    skip_columns = st.multiselect("Sloupce k vynechání", options=columns)

    if translate_columns:
        stat_rows = []
        for col in translate_columns:
            non_empty, html_count, href_count = column_stats(df, col)
            stat_rows.append({"column": col, "non_empty": non_empty, "html_cells": html_count, "href_cells": href_count})
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

    st.subheader("Nastavení překladu")
    provider_name = st.selectbox("Provider", ["OpenAI"])
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    source_lang = st.text_input("Source language", value="cs")
    target_lang = st.text_input("Target language", value="sk")
    max_chars = st.number_input("Max chars per request", min_value=100, max_value=8000, value=1200, step=100)
    per_run_cells = st.number_input("Počet buněk na jeden běh (kvůli pause/resume)", min_value=1, max_value=200, value=15, step=1)
    html_mode = st.selectbox("HTML režim", ["AUTO", "FORCE_HTML", "FORCE_TEXT"])
    use_batch_api = st.checkbox("Použít batch API (hromadné volání)", value=False)

    skip_urls = st.checkbox("URL v textu neměnit", value=True)
    skip_emails = st.checkbox("E-maily neměnit", value=True)
    skip_codes = st.checkbox("SKU/EAN/kódy neměnit", value=True)
    skip_units = st.checkbox("Jednotky neměnit", value=True)

    keep_unsafe = st.checkbox("Keep translated anyway (unsafe)", value=False)

    glossary_json = st.text_area("Glossary JSON (volitelně)", value='{}', help='Např. {"Akce":"Akcia"}')

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
        except ValueError:
            st.error("Glossary musí být validní JSON objekt.")
            return

        try:
            _ = get_provider(provider_name, model, use_batch_api=use_batch_api)
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
            "keep_unsafe": keep_unsafe,
            "glossary": glossary,
            "per_run_cells": int(per_run_cells),
            "translation_options": {
                "mode": html_mode,
                "skip_urls": skip_urls,
                "skip_emails": skip_emails,
                "skip_codes": skip_codes,
                "skip_units": skip_units,
            },
        }
        _ensure_job(df, settings, translate_columns, skip_columns)

    if st.session_state.get("job_active"):
        total = len(st.session_state.job_tasks)
        done = st.session_state.job_cursor
        progress = (done / total) if total else 1.0
        st.progress(progress)
        st.info(
            f"Stav: {st.session_state.job_status} | Hotovo: {done}/{total} | "
            f"Přeloženo: {st.session_state.job_translated_count} | Chyb: {st.session_state.job_error_count}"
        )

        if st.session_state.job_status == "running":
            _process_batch(df, fmt)
            if st.session_state.job_status == "completed":
                st.success("Překlad dokončen")
            else:
                st.rerun()

    if st.session_state.get("job_status") == "completed":
        st.download_button(
            "Stáhnout translated.csv",
            data=BytesIO(st.session_state.job_translated_csv),
            file_name="translated.csv",
            mime="text/csv",
        )
        st.download_button(
            "Stáhnout translation_report.csv",
            data=BytesIO(st.session_state.job_report_csv),
            file_name="translation_report.csv",
            mime="text/csv",
        )
        st.download_button(
            "Stáhnout translation_cache.json",
            data=BytesIO(st.session_state.job_cache_json),
            file_name="translation_cache.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
