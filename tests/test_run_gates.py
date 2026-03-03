from __future__ import annotations

from telemetry import evaluate_run_gate


def test_run_gate_fails_when_quality_ratio_exceeded() -> None:
    passed, reasons = evaluate_run_gate(
        {
            "quality_fail_ratio": 0.08,
            "provider_error_ratio": 0.0,
            "batch_shape_error_rate": 0.0,
            "p95_latency_ms": 2000,
            "quality_sample_size": 120,
            "provider_error_sample_size": 120,
            "latency_sample_size": 30,
            "batch_shape_sample_size": 20,
        },
        {
            "quality_fail_ratio": 0.03,
            "provider_error_ratio": 0.02,
            "batch_shape_error_rate": 0.15,
            "p95_latency_ms": 12000,
            "quality_fail_ratio_min_samples": 25,
            "provider_error_ratio_min_samples": 1,
            "batch_shape_error_rate_min_samples": 10,
            "p95_latency_ms_min_samples": 8,
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
            "quality_sample_size": 120,
            "provider_error_sample_size": 120,
            "latency_sample_size": 30,
            "batch_shape_sample_size": 20,
        },
        {
            "quality_fail_ratio": 0.03,
            "provider_error_ratio": 0.02,
            "batch_shape_error_rate": 0.15,
            "p95_latency_ms": 12000,
            "quality_fail_ratio_min_samples": 25,
            "provider_error_ratio_min_samples": 1,
            "batch_shape_error_rate_min_samples": 10,
            "p95_latency_ms_min_samples": 8,
        },
    )
    assert passed is True
    assert reasons == []


def test_run_gate_ignores_quality_ratio_when_sample_too_small() -> None:
    passed, reasons = evaluate_run_gate(
        {
            "quality_fail_ratio": 0.5,
            "provider_error_ratio": 0.0,
            "batch_shape_error_rate": 0.0,
            "p95_latency_ms": 2000,
            "quality_sample_size": 5,
            "provider_error_sample_size": 5,
            "latency_sample_size": 5,
            "batch_shape_sample_size": 5,
        },
        {
            "quality_fail_ratio": 0.03,
            "provider_error_ratio": 0.02,
            "batch_shape_error_rate": 0.15,
            "p95_latency_ms": 12000,
            "quality_fail_ratio_min_samples": 25,
            "provider_error_ratio_min_samples": 1,
            "batch_shape_error_rate_min_samples": 10,
            "p95_latency_ms_min_samples": 8,
        },
    )
    assert passed is True
    assert reasons == []
