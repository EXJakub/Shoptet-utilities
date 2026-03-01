from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None


class TranslationCache:
    def __init__(self, path: str = "translation_cache.json") -> None:
        self.path = Path(path)
        self._lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._data: dict[str, str] = {}
        self.load()

    def _key(self, text: str, source_lang: str, target_lang: str, provider: str, model: str) -> str:
        raw = f"{text}|{source_lang}|{target_lang}|{provider}|{model}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, text: str, source_lang: str, target_lang: str, provider: str, model: str) -> str | None:
        return self._data.get(self._key(text, source_lang, target_lang, provider, model))

    def set(self, text: str, translated: str, source_lang: str, target_lang: str, provider: str, model: str) -> None:
        self._data[self._key(text, source_lang, target_lang, provider, model)] = translated

    @contextmanager
    def _file_lock(self) -> object:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_handle
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def export_json_bytes(self) -> bytes:
        return json.dumps(self._data, ensure_ascii=False, indent=2).encode("utf-8")

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return

        with self._file_lock():
            if not self.path.exists():
                self._data = {}
                return
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                corrupt_path = self.path.with_suffix(f"{self.path.suffix}.corrupt-{int(time())}")
                os.replace(self.path, corrupt_path)
                self._data = {}

    def save(self) -> None:
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_name = tmp.name
            os.replace(temp_name, self.path)
