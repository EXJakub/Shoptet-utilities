from __future__ import annotations

import json
from typing import Any, Callable

from cache import TranslationCache
from csv_io import CsvFormat, dataframe_to_csv_bytes
from reporting import report_to_dataframe


def should_checkpoint(
    *,
    batch_index: int,
    total_batches: int,
    checkpoint_every_batches: int,
    force: bool = False,
) -> bool:
    if force:
        return True
    if total_batches <= 0:
        return True
    if batch_index >= total_batches:
        return True
    return batch_index % checkpoint_every_batches == 0


def write_job_artifacts(
    *,
    state: dict[str, Any],
    fmt: CsvFormat,
    cache: TranslationCache,
    source_df,
    settings: dict[str, Any],
    quality_report_to_dataframe: Callable[[list[dict[str, Any]]], Any],
    build_run_summary: Callable[[Any, dict[str, Any]], dict[str, Any]],
    build_validation_bundle_zip: Callable[[], bytes],
    include_validation_bundle: bool,
) -> None:
    state["job_translated_csv"] = dataframe_to_csv_bytes(state["job_df_out"], fmt)
    report_df = report_to_dataframe(state["job_report"])
    state["job_report_csv"] = report_df.to_csv(index=False).encode(fmt.encoding)
    quality_df = quality_report_to_dataframe(state["job_quality_report"])
    state["job_quality_csv"] = quality_df.to_csv(index=False).encode(fmt.encoding)
    state["job_cache_json"] = cache.export_json_bytes()

    if include_validation_bundle:
        summary = build_run_summary(source_df, settings)
        state["job_validation_summary_json"] = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        state["job_validation_bundle_zip"] = build_validation_bundle_zip()
