from __future__ import annotations

from telemetry import AlertRule, build_run_envelope


def test_telemetry_envelope_contains_required_fields() -> None:
    envelope = build_run_envelope(
        run_id="run-1",
        phase="checkpoint",
        status="running",
        metrics={"effective_tpm_estimate": 1234},
        alerts=[AlertRule(name="x", value=1.0, threshold=0.5, severity="high", message="alert")],
        gate_passed=False,
        gate_reasons=["quality_fail_ratio"],
    )

    assert envelope["run_id"] == "run-1"
    assert envelope["phase"] == "checkpoint"
    assert envelope["status"] == "running"
    assert "timestamp" in envelope
    assert envelope["metrics"]["effective_tpm_estimate"] == 1234
    assert envelope["alerts"][0]["name"] == "x"
    assert envelope["run_gate"]["passed"] is False
    assert envelope["run_gate"]["reasons"] == ["quality_fail_ratio"]
