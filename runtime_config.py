from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    openai_api_key: str
    openai_model: str
    checkpoint_every_batches: int
    batch_min_parallel: int
    batch_max_parallel: int
    batch_safe_mode_fallback_threshold: float
    batch_partial_recovery_max_attempts: int
    quality_gate_max_fail_ratio: float
    quality_gate_max_provider_error_ratio: float
    quality_gate_max_batch_shape_error_rate: float
    quality_gate_max_p95_latency_ms: int
    cache_revalidation_low_risk_sample_rate: float
    telemetry_backend: str
    telemetry_jsonl_path: str
    sync_max_error_ratio: float


@dataclass(slots=True)
class ShoptetRuntimeConfig:
    base_url: str
    token: str
    products_endpoint: str
    ean_field: str
    id_field: str
    page_size: int


def load_runtime_config() -> RuntimeConfig:
    checkpoint_raw = os.getenv("TRANSLATION_CHECKPOINT_EVERY_BATCHES", "5")
    try:
        checkpoint_every_batches = max(1, int(checkpoint_raw))
    except ValueError:
        checkpoint_every_batches = 5
    min_parallel_raw = os.getenv("BATCH_MIN_PARALLEL", "4")
    max_parallel_raw = os.getenv("BATCH_MAX_PARALLEL", "64")
    safe_threshold_raw = os.getenv("BATCH_SAFE_MODE_FALLBACK_THRESHOLD", "0.25")
    partial_attempts_raw = os.getenv("BATCH_PARTIAL_RECOVERY_MAX_ATTEMPTS", "2")
    quality_fail_ratio_raw = os.getenv("QUALITY_GATE_MAX_FAIL_RATIO", "0.03")
    provider_error_ratio_raw = os.getenv("QUALITY_GATE_MAX_PROVIDER_ERROR_RATIO", "0.02")
    batch_shape_rate_raw = os.getenv("QUALITY_GATE_MAX_BATCH_SHAPE_ERROR_RATE", "0.15")
    p95_latency_raw = os.getenv("QUALITY_GATE_MAX_P95_LATENCY_MS", "12000")
    cache_sample_raw = os.getenv("CACHE_REVALIDATION_LOW_RISK_SAMPLE_RATE", "0.10")
    telemetry_backend = os.getenv("TELEMETRY_BACKEND", "noop").strip().lower() or "noop"
    telemetry_jsonl_path = os.getenv("TELEMETRY_JSONL_PATH", "telemetry/runs.jsonl")
    sync_max_error_ratio_raw = os.getenv("SYNC_MAX_ERROR_RATIO", "0.20")

    try:
        min_parallel = max(1, int(min_parallel_raw))
    except ValueError:
        min_parallel = 4
    try:
        max_parallel = max(min_parallel, int(max_parallel_raw))
    except ValueError:
        max_parallel = 64
    try:
        safe_threshold = float(safe_threshold_raw)
    except ValueError:
        safe_threshold = 0.25
    safe_threshold = min(max(safe_threshold, 0.05), 1.0)
    try:
        partial_attempts = max(0, int(partial_attempts_raw))
    except ValueError:
        partial_attempts = 2
    try:
        quality_fail_ratio = float(quality_fail_ratio_raw)
    except ValueError:
        quality_fail_ratio = 0.03
    try:
        provider_error_ratio = float(provider_error_ratio_raw)
    except ValueError:
        provider_error_ratio = 0.02
    try:
        batch_shape_error_rate = float(batch_shape_rate_raw)
    except ValueError:
        batch_shape_error_rate = 0.15
    try:
        p95_latency_ms = int(p95_latency_raw)
    except ValueError:
        p95_latency_ms = 12000
    try:
        cache_sample_rate = float(cache_sample_raw)
    except ValueError:
        cache_sample_rate = 0.10
    try:
        sync_max_error_ratio = float(sync_max_error_ratio_raw)
    except ValueError:
        sync_max_error_ratio = 0.20

    quality_fail_ratio = min(max(quality_fail_ratio, 0.0), 1.0)
    provider_error_ratio = min(max(provider_error_ratio, 0.0), 1.0)
    batch_shape_error_rate = min(max(batch_shape_error_rate, 0.0), 1.0)
    cache_sample_rate = min(max(cache_sample_rate, 0.0), 1.0)
    sync_max_error_ratio = min(max(sync_max_error_ratio, 0.01), 1.0)
    p95_latency_ms = max(1000, p95_latency_ms)

    return RuntimeConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        checkpoint_every_batches=checkpoint_every_batches,
        batch_min_parallel=min_parallel,
        batch_max_parallel=max_parallel,
        batch_safe_mode_fallback_threshold=safe_threshold,
        batch_partial_recovery_max_attempts=partial_attempts,
        quality_gate_max_fail_ratio=quality_fail_ratio,
        quality_gate_max_provider_error_ratio=provider_error_ratio,
        quality_gate_max_batch_shape_error_rate=batch_shape_error_rate,
        quality_gate_max_p95_latency_ms=p95_latency_ms,
        cache_revalidation_low_risk_sample_rate=cache_sample_rate,
        telemetry_backend=telemetry_backend if telemetry_backend in {"noop", "jsonl"} else "noop",
        telemetry_jsonl_path=telemetry_jsonl_path,
        sync_max_error_ratio=sync_max_error_ratio,
    )


def shoptet_from_session(session: dict[str, Any], prefix: str) -> ShoptetRuntimeConfig:
    return ShoptetRuntimeConfig(
        base_url=str(session[f"{prefix}_base_url"]),
        token=str(session[f"{prefix}_token"]),
        products_endpoint=str(session[f"{prefix}_products_endpoint"]),
        ean_field=str(session["api_ean_field"]),
        id_field=str(session["api_id_field"]),
        page_size=int(session["api_page_size"]),
    )
