import pandas as pd
import pytest
import requests

from shoptet_api import ShoptetApiError, ShoptetClient, ShoptetConfig, sync_translated_to_sk


class DummyClient:
    def __init__(self, sk_df: pd.DataFrame, fail_product_id: str | None = None):
        self.config = ShoptetConfig(base_url="https://example.com", token="x", id_field="id")
        self._sk_df = sk_df
        self._fail_product_id = fail_product_id
        self._requests = 0
        self._errors = 0

    def fetch_products_df(self, max_pages: int = 100) -> pd.DataFrame:
        return self._sk_df

    def update_product(self, product_id: str, payload: dict[str, str]) -> None:
        self._requests += 1
        if self._fail_product_id and product_id == self._fail_product_id:
            self._errors += 1
            raise ShoptetApiError("update failed")

    def get_metrics_snapshot(self) -> dict[str, float]:
        ratio = self._errors / max(1, self._requests)
        return {"request_count": float(self._requests), "error_count": float(self._errors), "error_ratio": float(ratio)}


def test_sync_translated_to_sk_reports_string_row_index_for_missing_ean() -> None:
    translated_df = pd.DataFrame(
        [{"ean": "111", "name": "Produkt"}],
        index=["product-A"],
    )
    sk_df = pd.DataFrame([{"id": "42", "ean": "222"}])

    report_errors: list[dict[str, object]] = []
    updated, missing = sync_translated_to_sk(
        sk_client=DummyClient(sk_df),
        translated_df=translated_df,
        columns_to_sync=["name"],
        ean_field="ean",
        report_errors=report_errors,
    )

    assert updated == 0
    assert missing == 1
    assert report_errors[0]["row_index"] == "product-A"


def test_sync_translated_to_sk_reports_string_row_index_for_update_error() -> None:
    translated_df = pd.DataFrame(
        [{"ean": "111", "name": "Produkt"}],
        index=["product-B"],
    )
    sk_df = pd.DataFrame([{"id": "42", "ean": "111"}])

    report_errors: list[dict[str, object]] = []
    updated, missing = sync_translated_to_sk(
        sk_client=DummyClient(sk_df, fail_product_id="42"),
        translated_df=translated_df,
        columns_to_sync=["name"],
        ean_field="ean",
        report_errors=report_errors,
    )

    assert updated == 0
    assert missing == 0
    assert report_errors[0]["row_index"] == "product-B"
    assert report_errors[0]["error_type"] == "sk_update_error"


def test_shoptet_client_normalizes_base_url_without_scheme() -> None:
    client = ShoptetClient(ShoptetConfig(base_url="myshop.cz", token="x"))

    assert client._url("/api/products") == "https://myshop.cz/api/products"


def test_shoptet_client_rejects_empty_base_url() -> None:
    client = ShoptetClient(ShoptetConfig(base_url="   ", token="x"))

    with pytest.raises(ValueError, match="nesmí být prázdné"):
        client._url("/api/products")


class _FakeResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> object:
        return self._payload


def test_fetch_products_fallbacks_without_pagination_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ShoptetClient(ShoptetConfig(base_url="https://example.com", token="x"))
    calls: list[dict[str, object]] = []

    def fake_request(method: str, url: str, headers: dict[str, str], params: dict[str, object] | None, json, timeout: int) -> _FakeResponse:
        calls.append({"method": method, "url": url, "params": params})
        if params is not None:
            return _FakeResponse(400, {"message": "invalid query params"})
        return _FakeResponse(200, [{"id": 1, "ean": "123"}])

    monkeypatch.setattr(client.session, "request", fake_request)

    products = client.fetch_products()

    assert products == [{"id": 1, "ean": "123"}]
    assert calls[0]["params"] == {"page": 1, "limit": 100}
    assert calls[1]["params"] is None


def test_fetch_products_raises_after_fallback_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ShoptetClient(ShoptetConfig(base_url="https://example.com", token="x"))

    def fake_request(method: str, url: str, headers: dict[str, str], params: dict[str, object] | None, json, timeout: int) -> _FakeResponse:
        return _FakeResponse(400, {"message": "bad request"})

    monkeypatch.setattr(client.session, "request", fake_request)

    with pytest.raises(ShoptetApiError):
        client.fetch_products()
