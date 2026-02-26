from __future__ import annotations

import json
import logging
import os
from io import BytesIO

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


def get_provider(name: str, model: str) -> TranslationProvider:
    if name == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Chybí OPENAI_API_KEY v .env.")
        return OpenAIProvider(api_key=api_key, model=model)
    raise ValueError("Zvolený provider není implementovaný.")


def column_stats(df: pd.DataFrame, column: str) -> tuple[int, int, int]:
    series = df[column].fillna("").astype(str)
    non_empty = int((series.str.strip() != "").sum())
    html_count = int(series.str.contains(HTML_TAG_RE, regex=True).sum())
    href_count = int(series.str.contains("href=", case=False, regex=False).sum())
    return non_empty, html_count, href_count


def main() -> None:
    st.set_page_config(page_title="CSV překladač (HTML-safe)", layout="wide")
    st.title("CSV překladač CZ → SK (ochrana HTML a href)")

    uploaded = st.file_uploader("Nahraj CSV", type=["csv"])
    if not uploaded:
        return

    file_bytes = uploaded.getvalue()
    detected_df, detected_fmt = read_csv_from_upload(file_bytes)

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
    html_mode = st.selectbox("HTML režim", ["AUTO", "FORCE_HTML", "FORCE_TEXT"])

    skip_urls = st.checkbox("URL v textu neměnit", value=True)
    skip_emails = st.checkbox("E-maily neměnit", value=True)
    skip_codes = st.checkbox("SKU/EAN/kódy neměnit", value=True)
    skip_units = st.checkbox("Jednotky neměnit", value=True)

    keep_unsafe = st.checkbox("Keep translated anyway (unsafe)", value=False)

    glossary_json = st.text_area(
        "Glossary JSON (volitelně)",
        value='{}',
        help='Např. {"Akce":"Akcia"}',
    )

    if st.button("Translate", type="primary"):
        if not translate_columns:
            st.error("Vyber aspoň jeden sloupec k překladu.")
            return

        options = TranslationOptions(
            mode=html_mode,
            skip_urls=skip_urls,
            skip_emails=skip_emails,
            skip_codes=skip_codes,
            skip_units=skip_units,
        )

        try:
            glossary = json.loads(glossary_json)
            if not isinstance(glossary, dict):
                raise ValueError
        except ValueError:
            st.error("Glossary musí být validní JSON objekt.")
            return

        try:
            provider = get_provider(provider_name, model)
        except Exception as exc:
            st.error(str(exc))
            return

        cache = TranslationCache()
        out_df = df.copy()
        report: list[ReportRecord] = []

        total_cells = len(df) * len(translate_columns)
        progress = st.progress(0)
        status = st.empty()

        translated_count = 0
        skipped_count = 0
        error_count = 0
        done = 0

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

        for row_idx in range(len(df)):
            for col in translate_columns:
                done += 1
                progress.progress(done / total_cells)

                if col in skip_columns:
                    skipped_count += 1
                    continue

                source = str(df.at[row_idx, col] or "")
                if not source.strip():
                    skipped_count += 1
                    continue

                try:
                    translated_value = translate_cell(source, translate_with_cache, options, max_chars=int(max_chars))
                    href_ok, href_msg = validate_hrefs(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
                    struct_ok, struct_msg = validate_structure(source, translated_value) if HTML_TAG_RE.search(source) else (True, "ok")
                    if not (href_ok and struct_ok):
                        error_count += 1
                        report.append(make_record(row_idx, col, "href_mismatch" if not href_ok else "structure_mismatch", f"{href_msg}; {struct_msg}", source))
                        if not keep_unsafe:
                            translated_value = source
                    out_df.at[row_idx, col] = translated_value
                    translated_count += 1
                except Exception as exc:
                    logger.exception("Translation failure row=%s col=%s", row_idx, col)
                    error_count += 1
                    report.append(make_record(row_idx, col, "api_error", str(exc), source))
                    out_df.at[row_idx, col] = source

                status.info(f"Přeloženo: {translated_count} | Přeskočeno: {skipped_count} | Chyb: {error_count}")

        cache.save()
        translated_bytes = dataframe_to_csv_bytes(out_df, fmt)
        report_df = report_to_dataframe(report)
        report_bytes = report_df.to_csv(index=False).encode(fmt.encoding)

        st.success("Hotovo")
        st.download_button("Stáhnout translated.csv", data=BytesIO(translated_bytes), file_name="translated.csv", mime="text/csv")
        st.download_button("Stáhnout translation_report.csv", data=BytesIO(report_bytes), file_name="translation_report.csv", mime="text/csv")
        st.download_button("Stáhnout translation_cache.json", data=BytesIO(json.dumps(cache._data, ensure_ascii=False, indent=2).encode("utf-8")), file_name="translation_cache.json", mime="application/json")


if __name__ == "__main__":
    main()
