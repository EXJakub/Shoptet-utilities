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


def test_runtime_config_clamps_invalid_checkpoint_value(monkeypatch) -> None:
    monkeypatch.setenv("TRANSLATION_CHECKPOINT_EVERY_BATCHES", "not-a-number")
    monkeypatch.setenv("BATCH_SAFE_MODE_FALLBACK_THRESHOLD", "9.0")
    cfg = load_runtime_config()
    assert cfg.checkpoint_every_batches == 5
    assert cfg.batch_safe_mode_fallback_threshold == 1.0
