from __future__ import annotations

from translator.base import TranslationProvider


class DeepLProvider(TranslationProvider):
    name = "deepl"

    def __init__(self, api_key: str, model: str = "deepl-default") -> None:
        self.api_key = api_key
        self.model = model

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        raise NotImplementedError("DeepL provider is optional and not implemented in this MVP.")
