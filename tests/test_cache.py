from __future__ import annotations

from pathlib import Path

from cache import TranslationCache


def test_cache_load_recovers_from_corrupt_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "translation_cache.json"
    cache_path.write_text("{not-valid-json", encoding="utf-8")

    cache = TranslationCache(path=str(cache_path))

    assert cache.get("x", "cs", "sk", "openai", "m") is None
    corrupt_files = list(tmp_path.glob("translation_cache.json.corrupt-*"))
    assert len(corrupt_files) == 1
    assert not cache_path.exists()


def test_cache_save_and_export_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "translation_cache.json"
    cache = TranslationCache(path=str(cache_path))

    cache.set("hello", "ahoj", "en", "cs", "openai", "gpt")
    cache.save()

    reloaded = TranslationCache(path=str(cache_path))
    assert reloaded.get("hello", "en", "cs", "openai", "gpt") == "ahoj"
    assert b"ahoj" in reloaded.export_json_bytes()
