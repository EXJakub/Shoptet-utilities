import json
import zipfile
from io import BytesIO

import pandas as pd

import app
from reporting import make_record


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def test_build_run_summary_contains_expected_counts() -> None:
    app.st.session_state = _SessionState(
        job_quality_report=[{"issue": "unchanged_text"}, {"issue": "unchanged_text"}, {"issue": "empty_translation"}],
        job_report=[
            make_record(0, "name", "quality_gate_failed", "bad", "src"),
            make_record(1, "name", "api_error", "boom", "src"),
        ],
        job_status="completed",
        job_translated_count=42,
        job_error_count=2,
        job_perf={
            "chunks_sent": 3,
            "translation_error_type_counts": {"quality_gate_failed": 1, "api_error": 1},
            "quality_issue_counts": {"unchanged_text": 2, "empty_translation": 1},
        },
    )

    source_df = pd.DataFrame({"name": ["a", "b"]})
    settings = {
        "source_mode": "CSV",
        "provider_name": "OpenAI",
        "model": "gpt-5-mini",
        "use_batch_api": False,
        "source_lang": "cs",
        "target_lang": "sk",
        "translate_columns": ["name"],
    }

    summary = app._build_run_summary(source_df, settings)

    assert summary["input"]["row_count"] == 2
    assert summary["job"]["translated_cells"] == 42
    assert summary["translation_report"]["status_counts"]["quality_gate_failed"] == 1
    assert summary["quality_report"]["issue_counts"]["unchanged_text"] == 2


def test_build_validation_bundle_zip_contains_all_artifacts() -> None:
    app.st.session_state = _SessionState(
        job_translated_csv=b"t",
        job_report_csv=b"r",
        job_quality_csv=b"q",
        job_cache_json=b"c",
        job_validation_summary_json=json.dumps({"ok": True}).encode("utf-8"),
    )

    bundle = app._build_validation_bundle_zip()

    with zipfile.ZipFile(BytesIO(bundle), "r") as archive:
        names = sorted(archive.namelist())
        assert names == [
            "quality_report.csv",
            "translated.csv",
            "translation_cache.json",
            "translation_report.csv",
            "validation_summary.json",
        ]
        assert archive.read("translated.csv") == b"t"
        assert archive.read("translation_report.csv") == b"r"
        assert archive.read("quality_report.csv") == b"q"
        assert archive.read("translation_cache.json") == b"c"


def test_build_run_summary_prefers_incremental_counters_over_reports() -> None:
    app.st.session_state = _SessionState(
        job_quality_report=[{"issue": "unchanged_text"}],
        job_report=[make_record(0, "name", "api_error", "boom", "src")],
        job_status="completed",
        job_translated_count=1,
        job_error_count=1,
        job_perf={
            "translation_error_type_counts": {"quality_gate_failed": 5},
            "quality_issue_counts": {"empty_translation": 3},
        },
    )

    source_df = pd.DataFrame({"name": ["a"]})
    settings = {
        "source_mode": "CSV",
        "provider_name": "OpenAI",
        "model": "gpt-5-mini",
        "use_batch_api": False,
        "source_lang": "cs",
        "target_lang": "sk",
        "translate_columns": ["name"],
    }

    summary = app._build_run_summary(source_df, settings)

    assert summary["translation_report"]["status_counts"] == {"quality_gate_failed": 5}
    assert summary["quality_report"]["issue_counts"] == {"empty_translation": 3}
