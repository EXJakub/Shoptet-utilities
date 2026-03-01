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

    provider = _FakeProvider()
    app._update_job_perf_metrics(provider, total_segments=1000, cache_misses=100, unique_misses=80)
    app._autotune_parallelism(provider)
    app._apply_safe_mode_guard()
    app._apply_safe_mode_guard()
    app._apply_safe_mode_guard()

    assert app.st.session_state["job_perf"]["effective_tpm_estimate"] > 0
    assert app.st.session_state["job_adaptive_parallel_current"] <= 16
