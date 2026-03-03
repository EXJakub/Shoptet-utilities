from __future__ import annotations

from runtime_config import load_runtime_config


def test_runtime_config_uses_safe_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TRANSLATION_CHECKPOINT_EVERY_BATCHES", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    cfg = load_runtime_config()

    assert cfg.checkpoint_every_batches == 5
    assert cfg.openai_api_key == ""
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.batch_min_parallel == 4
    assert cfg.batch_max_parallel == 64
    assert cfg.batch_safe_mode_fallback_threshold == 0.25
    assert cfg.batch_partial_recovery_max_attempts == 2
    assert cfg.quality_gate_min_quality_samples == 25
    assert cfg.quality_gate_min_latency_samples == 8
    assert cfg.quality_gate_min_batch_shape_samples == 10
    assert cfg.batch_autotune_cooldown_batches == 2
    assert cfg.batch_parallel_downshift_p95_ms == 8500
    assert cfg.batch_parallel_upshift_p95_ms == 3500
    assert cfg.max_unchanged_retries_per_batch == 40


def test_runtime_config_clamps_invalid_checkpoint_value(monkeypatch) -> None:
    monkeypatch.setenv("TRANSLATION_CHECKPOINT_EVERY_BATCHES", "not-a-number")
    monkeypatch.setenv("BATCH_SAFE_MODE_FALLBACK_THRESHOLD", "9.0")
    cfg = load_runtime_config()
    assert cfg.checkpoint_every_batches == 5
    assert cfg.batch_safe_mode_fallback_threshold == 1.0


def test_runtime_config_clamps_new_autotune_and_sampling_settings(monkeypatch) -> None:
    monkeypatch.setenv("QUALITY_GATE_MIN_QUALITY_SAMPLES", "-10")
    monkeypatch.setenv("QUALITY_GATE_MIN_LATENCY_SAMPLES", "0")
    monkeypatch.setenv("QUALITY_GATE_MIN_BATCH_SHAPE_SAMPLES", "not-a-number")
    monkeypatch.setenv("BATCH_AUTOTUNE_COOLDOWN_BATCHES", "-1")
    monkeypatch.setenv("BATCH_PARALLEL_DOWNSHIFT_P95_MS", "500")
    monkeypatch.setenv("BATCH_PARALLEL_UPSHIFT_P95_MS", "9000")
    monkeypatch.setenv("BATCH_CHUNK_DOWNSHIFT_AVG_LATENCY_MS", "800")
    monkeypatch.setenv("BATCH_CHUNK_UPSHIFT_AVG_LATENCY_MS", "12000")
    monkeypatch.setenv("BATCH_MAX_UNCHANGED_RETRIES_PER_BATCH", "-10")

    cfg = load_runtime_config()

    assert cfg.quality_gate_min_quality_samples == 1
    assert cfg.quality_gate_min_latency_samples == 1
    assert cfg.quality_gate_min_batch_shape_samples == 10
    assert cfg.batch_autotune_cooldown_batches == 0
    assert cfg.batch_parallel_downshift_p95_ms == 1000
    assert cfg.batch_parallel_upshift_p95_ms == 1000
    assert cfg.batch_chunk_downshift_avg_latency_ms == 1000
    assert cfg.batch_chunk_upshift_avg_latency_ms == 1000
    assert cfg.max_unchanged_retries_per_batch == 0
