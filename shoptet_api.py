from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def _report_row_index(idx: Any) -> int | str:
    """Return a JSON-serializable row identifier without raising on non-numeric index values."""
    try:
        return int(idx)
    except (TypeError, ValueError):
        return str(idx)


@dataclass(slots=True)
class ShoptetConfig:
    base_url: str
    token: str
    products_endpoint: str = "/api/products"
    ean_field: str = "ean"
    id_field: str = "id"
    page_param: str = "page"
    page_size_param: str = "limit"
    page_size: int = 100


class ShoptetApiError(RuntimeError):
    """Base error raised for Shoptet integration failures."""


class ShoptetRetryableError(ShoptetApiError):
    """Raised for transient failures that are likely recoverable."""


class ShoptetClient:
    def __init__(self, config: ShoptetConfig, timeout_s: int = 30, max_retries: int = 3, backoff_factor: float = 0.5) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = self._build_session()
        self._request_count = 0
        self._error_count = 0
        self._retryable_error_count = 0
        self._total_latency_ms = 0

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "PATCH"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _url(self, endpoint: str) -> str:
        base_url = self._normalized_base_url()
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _normalized_base_url(self) -> str:
        base_url = self.config.base_url.strip()
        if not base_url:
            raise ValueError("Shoptet base URL nesmí být prázdné.")

        parsed = urlparse(base_url)
        if parsed.scheme:
            return base_url
        return f"https://{base_url.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _extract_items(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in ("data", "items", "results", "products"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
                if isinstance(value, dict):
                    for nested in ("items", "results", "products"):
                        nested_value = value.get(nested)
                        if isinstance(nested_value, list):
                            return [x for x in nested_value if isinstance(x, dict)]
        return []

    def _fetch_products_page(self, page: int, include_pagination: bool) -> requests.Response:
        params: dict[str, Any] | None = None
        if include_pagination:
            params = {self.config.page_param: page, self.config.page_size_param: self.config.page_size}
            return self._request("GET", self.config.products_endpoint, params=params, allow_status_codes={400})
        return self._request("GET", self.config.products_endpoint, params=params)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        allow_status_codes: set[int] | None = None,
    ) -> requests.Response:
        started = time.perf_counter()
        try:
            response = self.session.request(
                method,
                self._url(endpoint),
                headers=self._headers(),
                params=params,
                json=json_payload,
                timeout=self.timeout_s,
            )
        except requests.Timeout as exc:
            self._request_count += 1
            self._error_count += 1
            self._retryable_error_count += 1
            self._total_latency_ms += int((time.perf_counter() - started) * 1000)
            raise ShoptetRetryableError(f"Shoptet {method} timeout: {exc}") from exc
        except requests.ConnectionError as exc:
            self._request_count += 1
            self._error_count += 1
            self._retryable_error_count += 1
            self._total_latency_ms += int((time.perf_counter() - started) * 1000)
            raise ShoptetRetryableError(f"Shoptet {method} connection error: {exc}") from exc
        except requests.RequestException as exc:
            self._request_count += 1
            self._error_count += 1
            self._total_latency_ms += int((time.perf_counter() - started) * 1000)
            raise ShoptetApiError(f"Shoptet {method} request failed: {exc}") from exc

        self._request_count += 1
        self._total_latency_ms += int((time.perf_counter() - started) * 1000)
        allow_status_codes = allow_status_codes or set()
        if response.status_code in allow_status_codes:
            return response
        if response.status_code in {429, 500, 502, 503, 504}:
            self._error_count += 1
            self._retryable_error_count += 1
            raise ShoptetRetryableError(f"Shoptet {method} transient HTTP {response.status_code}")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            self._error_count += 1
            raise ShoptetApiError(f"Shoptet {method} failed with HTTP {response.status_code}") from exc
        return response

    def get_metrics_snapshot(self) -> dict[str, float]:
        error_ratio = self._error_count / max(1, self._request_count)
        avg_latency_ms = self._total_latency_ms / max(1, self._request_count)
        return {
            "request_count": float(self._request_count),
            "error_count": float(self._error_count),
            "retryable_error_count": float(self._retryable_error_count),
            "error_ratio": float(error_ratio),
            "avg_latency_ms": float(avg_latency_ms),
        }

    def fetch_products(self, max_pages: int = 100) -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = []
        use_pagination = True

        for page in range(1, max_pages + 1):
            resp = self._fetch_products_page(page=page, include_pagination=use_pagination)
            if page == 1 and use_pagination and resp.status_code == 400:
                logger.warning(
                    "Shoptet API rejected pagination params (%s/%s). Retrying first page without params.",
                    self.config.page_param,
                    self.config.page_size_param,
                )
                use_pagination = False
                resp = self._fetch_products_page(page=page, include_pagination=False)

            payload = resp.json()
            items = self._extract_items(payload)
            if not items:
                break

            products.extend(items)
            if not use_pagination:
                break
            if len(items) < self.config.page_size:
                break
        return products

    def fetch_products_df(self, max_pages: int = 100) -> pd.DataFrame:
        items = self.fetch_products(max_pages=max_pages)
        if not items:
            return pd.DataFrame()
        return pd.json_normalize(items)

    def update_product(self, product_id: str, payload: dict[str, Any]) -> None:
        endpoint = f"{self.config.products_endpoint.rstrip('/')}/{product_id}"
        self._request("PATCH", endpoint, json_payload=payload)


def sync_translated_to_sk(
    sk_client: ShoptetClient,
    translated_df: pd.DataFrame,
    columns_to_sync: list[str],
    ean_field: str,
    report_errors: list[dict[str, Any]],
    max_error_ratio: float = 1.0,
) -> tuple[int, int]:
    sk_df = sk_client.fetch_products_df()
    if sk_df.empty:
        raise RuntimeError("SK API nevrátilo žádné produkty.")

    if ean_field not in translated_df.columns or ean_field not in sk_df.columns:
        raise RuntimeError(f"EAN field '{ean_field}' musí existovat v CZ i SK datech.")
    if sk_client.config.id_field not in sk_df.columns:
        raise RuntimeError(f"ID field '{sk_client.config.id_field}' neexistuje ve SK datech.")

    sk_index: dict[str, str] = {}
    for row in sk_df.itertuples(index=False):
        row_map = row._asdict()
        ean_value = str(row_map.get(ean_field, "")).strip()
        if not ean_value:
            continue
        sk_index[ean_value] = str(row_map.get(sk_client.config.id_field, "")).strip()

    updated = 0
    missing = 0
    for row in translated_df.itertuples(index=True):
        row_map = row._asdict()
        idx = row_map.get("Index")
        ean = str(row_map.get(ean_field, "")).strip()
        if not ean:
            continue
        product_id = sk_index.get(ean)
        if not product_id:
            missing += 1
            report_errors.append(
                {
                    "row_index": _report_row_index(idx),
                    "column": ean_field,
                    "error_type": "ean_not_found_on_sk",
                    "message": f"EAN {ean} nebyl nalezen na SK",
                }
            )
            continue

        payload = {col: row_map.get(col, "") for col in columns_to_sync if col in translated_df.columns}
        if not payload:
            continue
        try:
            sk_client.update_product(product_id, payload)
            updated += 1
        except ShoptetApiError as exc:
            report_errors.append(
                {
                    "row_index": _report_row_index(idx),
                    "column": ean_field,
                    "error_type": "sk_update_error",
                    "message": f"EAN {ean}: {exc}",
                }
            )
        metrics = sk_client.get_metrics_snapshot()
        if float(metrics.get("error_ratio", 0.0)) > max_error_ratio:
            report_errors.append(
                {
                    "row_index": _report_row_index(idx),
                    "column": ean_field,
                    "error_type": "sync_blocked_high_error_ratio",
                    "message": f"Sync halted: error ratio {metrics['error_ratio']:.2%} exceeded {max_error_ratio:.2%}",
                }
            )
            break
    return updated, missing
