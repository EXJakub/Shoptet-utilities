import pandas as pd

from shoptet_api import ShoptetConfig, sync_translated_to_sk


class DummyClient:
    def __init__(self, sk_df: pd.DataFrame, fail_product_id: str | None = None):
        self.config = ShoptetConfig(base_url="https://example.com", token="x", id_field="id")
        self._sk_df = sk_df
        self._fail_product_id = fail_product_id

    def fetch_products_df(self, max_pages: int = 100) -> pd.DataFrame:
        return self._sk_df

    def update_product(self, product_id: str, payload: dict[str, str]) -> None:
        if self._fail_product_id and product_id == self._fail_product_id:
            raise RuntimeError("update failed")


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
