from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

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


class ShoptetClient:
    def __init__(self, config: ShoptetConfig, timeout_s: int = 30) -> None:
        self.config = config
        self.timeout_s = timeout_s

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
        resp = requests.get(self._url(self.config.products_endpoint), headers=self._headers(), params=params, timeout=self.timeout_s)
        return resp

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

            resp.raise_for_status()
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
        resp = requests.patch(self._url(endpoint), headers=self._headers(), json=payload, timeout=self.timeout_s)
        resp.raise_for_status()


def sync_translated_to_sk(
    sk_client: ShoptetClient,
    translated_df: pd.DataFrame,
    columns_to_sync: list[str],
    ean_field: str,
    report_errors: list[dict[str, Any]],
) -> tuple[int, int]:
    sk_df = sk_client.fetch_products_df()
    if sk_df.empty:
        raise RuntimeError("SK API nevrátilo žádné produkty.")

    if ean_field not in translated_df.columns or ean_field not in sk_df.columns:
        raise RuntimeError(f"EAN field '{ean_field}' musí existovat v CZ i SK datech.")
    if sk_client.config.id_field not in sk_df.columns:
        raise RuntimeError(f"ID field '{sk_client.config.id_field}' neexistuje ve SK datech.")

    sk_index = {
        str(row[ean_field]).strip(): str(row[sk_client.config.id_field]).strip()
        for _, row in sk_df.iterrows()
        if str(row.get(ean_field, "")).strip()
    }

    updated = 0
    missing = 0
    for idx, row in translated_df.iterrows():
        ean = str(row.get(ean_field, "")).strip()
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

        payload = {col: row.get(col, "") for col in columns_to_sync if col in translated_df.columns}
        try:
            sk_client.update_product(product_id, payload)
            updated += 1
        except Exception as exc:
            report_errors.append(
                {
                    "row_index": _report_row_index(idx),
                    "column": ean_field,
                    "error_type": "sk_update_error",
                    "message": f"EAN {ean}: {exc}",
                }
            )
    return updated, missing
