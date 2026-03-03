from __future__ import annotations

from types import SimpleNamespace

import app


def test_stuck_zone_guardrails_maintain_progress_signal(monkeypatch) -> None:
    class _FakeProvider:
        def get_metrics_snapshot(self):
            # Simulate problematic zone with shape errors but still flowing events.
            events = [{"latency_ms": 8500, "success": True, "fallback_used": True} for _ in range(8)]
            return {
                "http_requests_total": 500,
                "fallback_count": 120,
                "error_count": 0,
                "success_count": 500,
                "total_latency_ms": 100000,
                "total_input_chars": 4_000_000,
                "total_input_items": 20_000,
                "batch_invalid_shape_count": 40,
                "partial_recovery_count": 30,
                "isolated_item_fallback_count": 15,
                "events": events,
            }

    app.st.session_state = {
        "job_perf": {"chunks_sent": 100},
        "job_started_at": app.pd.Timestamp.utcnow() - app.pd.Timedelta(minutes=2),
        "job_chunk_target_chars": 12000,
        "job_adaptive_parallel_current": 16,
        "job_fallback_rate_history": [],
        "job_settings": {"max_parallel_requests": 32},
    }
    app.RUNTIME_CONFIG.batch_min_parallel = 4
    app.RUNTIME_CONFIG.batch_max_parallel = 64
    app.RUNTIME_CONFIG.batch_safe_mode_fallback_threshold = 0.25
    app.RUNTIME_CONFIG.batch_autotune_cooldown_batches = 0
    app.RUNTIME_CONFIG.batch_parallel_downshift_p95_ms = 8500
    app.RUNTIME_CONFIG.batch_parallel_upshift_p95_ms = 3000

    provider = _FakeProvider()
    app._update_job_perf_metrics(provider, total_segments=1000, cache_misses=100, unique_misses=80)
    app._autotune_parallelism(provider)
    app._apply_safe_mode_guard()
    app._apply_safe_mode_guard()
    app._apply_safe_mode_guard()

    assert app.st.session_state["job_perf"]["effective_tpm_estimate"] > 0
    assert app.st.session_state["job_adaptive_parallel_current"] <= 16


def test_run_metrics_use_quality_sample_denominator_and_split_latency() -> None:
    app.st.session_state = {
        "job_translated_count": 10,
        "job_started_at": app.pd.Timestamp.utcnow() - app.pd.Timedelta(minutes=1),
        "job_perf": {
            "quality_fail_count": 4,
            "quality_segments_total": 40,
            "cell_reject_count": 2,
            "error_count": 1,
            "batch_shape_error_rate": 0.1,
            "effective_tpm_estimate": 1200,
            "cache_misses_total": 20,
            "segments_total": 100,
            "chunks_sent": 25,
            "last_events": [{"latency_ms": 1000}, {"latency_ms": 2000}, {"latency_ms": 9000}],
            "run_events": [{"latency_ms": 1000}, {"latency_ms": 2000}, {"latency_ms": 9000}, {"latency_ms": 18000}],
        },
    }
    metrics = app._calc_run_metrics()
    assert metrics["quality_fail_ratio"] == 0.1
    assert metrics["recent_p95_latency_ms"] == 2000.0
    assert metrics["run_p95_latency_ms"] == 9000.0
    assert metrics["p95_latency_ms"] == metrics["run_p95_latency_ms"]
    assert metrics["cell_reject_ratio"] == 0.2
