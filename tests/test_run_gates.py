from __future__ import annotations

from telemetry import evaluate_run_gate


def test_run_gate_fails_when_quality_ratio_exceeded() -> None:
    passed, reasons = evaluate_run_gate(
        {
            "quality_fail_ratio": 0.08,
            "provider_error_ratio": 0.0,
            "batch_shape_error_rate": 0.0,
            "p95_latency_ms": 2000,
        },
        {
            "quality_fail_ratio": 0.03,
            "provider_error_ratio": 0.02,
            "batch_shape_error_rate": 0.15,
            "p95_latency_ms": 12000,
        },
    )
    assert passed is False
    assert "quality_fail_ratio" in reasons


def test_run_gate_passes_within_thresholds() -> None:
    passed, reasons = evaluate_run_gate(
        {
            "quality_fail_ratio": 0.01,
            "provider_error_ratio": 0.01,
            "batch_shape_error_rate": 0.02,
            "p95_latency_ms": 1500,
        },
        {
            "quality_fail_ratio": 0.03,
            "provider_error_ratio": 0.02,
            "batch_shape_error_rate": 0.15,
            "p95_latency_ms": 12000,
        },
    )
    assert passed is True
    assert reasons == []
