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

    return RuntimeConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        checkpoint_every_batches=checkpoint_every_batches,
        batch_min_parallel=min_parallel,
        batch_max_parallel=max_parallel,
        batch_safe_mode_fallback_threshold=safe_threshold,
        batch_partial_recovery_max_attempts=partial_attempts,
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
