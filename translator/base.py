from __future__ import annotations

from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        raise NotImplementedError
