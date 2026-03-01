from app import QUALITY_REPORT_COLUMNS, _quality_report_to_dataframe


def test_empty_quality_report_csv_contains_header_only() -> None:
    quality_df = _quality_report_to_dataframe([])

    assert list(quality_df.columns) == QUALITY_REPORT_COLUMNS
    assert quality_df.empty

    csv_output = quality_df.to_csv(index=False)

    assert csv_output == ",".join(QUALITY_REPORT_COLUMNS) + "\n"
