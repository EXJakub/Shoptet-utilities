from __future__ import annotations

import hashlib
import json
from pathlib import Path


class TranslationCache:
    def __init__(self, path: str = "translation_cache.json") -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {}
        self.load()

    def _key(self, text: str, source_lang: str, target_lang: str, provider: str, model: str) -> str:
        raw = f"{text}|{source_lang}|{target_lang}|{provider}|{model}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, text: str, source_lang: str, target_lang: str, provider: str, model: str) -> str | None:
        return self._data.get(self._key(text, source_lang, target_lang, provider, model))

    def set(self, text: str, translated: str, source_lang: str, target_lang: str, provider: str, model: str) -> None:
        self._data[self._key(text, source_lang, target_lang, provider, model)] = translated

    def load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
