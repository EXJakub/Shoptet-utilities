from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AlertRule:
    name: str
    value: float
    threshold: float
    severity: str
    message: str


class TelemetryExporter:
    def emit(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class NoOpTelemetryExporter(TelemetryExporter):
    def emit(self, payload: dict[str, Any]) -> None:
        return None


class JsonlTelemetryExporter(TelemetryExporter):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_telemetry_exporter(backend: str, jsonl_path: str) -> TelemetryExporter:
    if backend == "jsonl":
        return JsonlTelemetryExporter(jsonl_path)
    return NoOpTelemetryExporter()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_size(metrics: dict[str, float], name: str) -> int:
    if name == "quality_fail_ratio":
        return int(metrics.get("quality_sample_size", 0))
    if name == "p95_latency_ms":
        return int(metrics.get("latency_sample_size", 0))
    if name == "batch_shape_error_rate":
        return int(metrics.get("batch_shape_sample_size", 0))
    if name == "provider_error_ratio":
        return int(metrics.get("provider_error_sample_size", 0))
    return 0


def _min_required_samples(thresholds: dict[str, float], name: str) -> int:
    key = f"{name}_min_samples"
    return int(thresholds.get(key, 1))


def _passes_sample_gate(metrics: dict[str, float], thresholds: dict[str, float], name: str) -> bool:
    return _sample_size(metrics, name) >= _min_required_samples(thresholds, name)


def evaluate_alerts(metrics: dict[str, float], thresholds: dict[str, float]) -> list[AlertRule]:
    alerts: list[AlertRule] = []
    if _passes_sample_gate(metrics, thresholds, "quality_fail_ratio") and metrics.get("quality_fail_ratio", 0.0) > thresholds["quality_fail_ratio"]:
        alerts.append(
            AlertRule(
                name="quality_fail_ratio",
                value=float(metrics.get("quality_fail_ratio", 0.0)),
                threshold=float(thresholds["quality_fail_ratio"]),
                severity="high",
                message="Quality fail ratio exceeded threshold.",
            )
        )
    if _passes_sample_gate(metrics, thresholds, "provider_error_ratio") and metrics.get("provider_error_ratio", 0.0) > thresholds["provider_error_ratio"]:
        alerts.append(
            AlertRule(
                name="provider_error_ratio",
                value=float(metrics.get("provider_error_ratio", 0.0)),
                threshold=float(thresholds["provider_error_ratio"]),
                severity="high",
                message="Provider error ratio exceeded threshold.",
            )
        )
    if _passes_sample_gate(metrics, thresholds, "batch_shape_error_rate") and metrics.get("batch_shape_error_rate", 0.0) > thresholds["batch_shape_error_rate"]:
        alerts.append(
            AlertRule(
                name="batch_shape_error_rate",
                value=float(metrics.get("batch_shape_error_rate", 0.0)),
                threshold=float(thresholds["batch_shape_error_rate"]),
                severity="medium",
                message="Batch shape error rate exceeded threshold.",
            )
        )
    if _passes_sample_gate(metrics, thresholds, "p95_latency_ms") and metrics.get("p95_latency_ms", 0.0) > thresholds["p95_latency_ms"]:
        alerts.append(
            AlertRule(
                name="p95_latency_ms",
                value=float(metrics.get("p95_latency_ms", 0.0)),
                threshold=float(thresholds["p95_latency_ms"]),
                severity="medium",
                message="P95 latency exceeded threshold.",
            )
        )
    return alerts


def evaluate_run_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _passes_sample_gate(metrics, thresholds, "quality_fail_ratio") and metrics.get("quality_fail_ratio", 0.0) > thresholds["quality_fail_ratio"]:
        reasons.append("quality_fail_ratio")
    if _passes_sample_gate(metrics, thresholds, "provider_error_ratio") and metrics.get("provider_error_ratio", 0.0) > thresholds["provider_error_ratio"]:
        reasons.append("provider_error_ratio")
    if _passes_sample_gate(metrics, thresholds, "batch_shape_error_rate") and metrics.get("batch_shape_error_rate", 0.0) > thresholds["batch_shape_error_rate"]:
        reasons.append("batch_shape_error_rate")
    if _passes_sample_gate(metrics, thresholds, "p95_latency_ms") and metrics.get("p95_latency_ms", 0.0) > thresholds["p95_latency_ms"]:
        reasons.append("p95_latency_ms")
    return (len(reasons) == 0, reasons)


def build_run_envelope(
    *,
    run_id: str,
    phase: str,
    status: str,
    metrics: dict[str, Any],
    alerts: list[AlertRule],
    gate_passed: bool,
    gate_reasons: list[str],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "timestamp": utc_now_iso(),
        "run_id": run_id,
        "phase": phase,
        "status": status,
        "metrics": metrics,
        "alerts": [asdict(a) for a in alerts],
        "run_gate": {"passed": gate_passed, "reasons": gate_reasons},
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload
