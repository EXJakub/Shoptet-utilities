from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class CsvFormat:
    encoding: str
    delimiter: str
    quotechar: str = '"'


def detect_encoding(file_bytes: bytes, preferred: Iterable[str] = ("utf-8-sig", "utf-8")) -> str:
    for enc in preferred:
        try:
            file_bytes.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def detect_delimiter(decoded_text: str, preferred: str = ";") -> str:
    sample = "\n".join(decoded_text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t,")
        if preferred in sample and preferred in ";,\t," and sample.count(preferred) >= sample.count(dialect.delimiter):
            return preferred
        return dialect.delimiter
    except csv.Error:
        return preferred


def read_csv_from_upload(file_bytes: bytes, encoding: str | None = None, delimiter: str | None = None, quotechar: str = '"') -> tuple[pd.DataFrame, CsvFormat]:
    chosen_encoding = encoding or detect_encoding(file_bytes)
    decoded = file_bytes.decode(chosen_encoding, errors="replace")
    chosen_delimiter = delimiter or detect_delimiter(decoded)
    df = pd.read_csv(StringIO(decoded), sep=chosen_delimiter, quotechar=quotechar, dtype=str, keep_default_na=False)
    return df, CsvFormat(encoding=chosen_encoding, delimiter=chosen_delimiter, quotechar=quotechar)


def dataframe_to_csv_bytes(df: pd.DataFrame, fmt: CsvFormat) -> bytes:
    payload = df.to_csv(index=False, sep=fmt.delimiter, quotechar=fmt.quotechar, quoting=csv.QUOTE_MINIMAL)
    return payload.encode(fmt.encoding)
