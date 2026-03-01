from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import pandas as pd


REPORT_COLUMNS = ["row_index", "column", "error_type", "message", "source_length", "source_hash"]


@dataclass(slots=True)
class ReportRecord:
    row_index: int
    column: str
    error_type: str
    message: str
    source_length: int
    source_hash: str


def make_record(row_index: int, column: str, error_type: str, message: str, source: str) -> ReportRecord:
    return ReportRecord(
        row_index=row_index,
        column=column,
        error_type=error_type,
        message=message,
        source_length=len(source),
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
    )


def report_to_dataframe(records: list[ReportRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records], columns=REPORT_COLUMNS)
