from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from translator.base import TranslationProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(TranslationProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        use_batch_api: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.use_batch_api = use_batch_api

    def _retry_call(self, input_payload: list[dict[str, str]]) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=input_payload,
                    temperature=0,
                )
                return (resp.output_text or "").strip()
            except Exception as exc:  # API/library level
                last_err = exc
                delay = 2**attempt
                logger.warning("OpenAI retry %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(delay)
        raise RuntimeError(f"retry_exhausted: {last_err}")

    def _one(self, text: str, source_lang: str, target_lang: str) -> str:
        system_prompt = (
            "You are a precise translation engine. "
            "Translate only natural language from source to target. "
            "Do not add markup. Preserve URLs, emails, codes, product numbers and units exactly. "
            "Return only translated text."
        )
        user_prompt = f"source_lang={source_lang}\ntarget_lang={target_lang}\ntext:\n{text}"
        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._retry_call(payload)

    def _batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []
        system_prompt = (
            "You are a precise translation engine. Translate each input item from source language to target language. "
            "Do not add markup or extra commentary. Preserve URLs, emails, product codes, identifiers, and units exactly. "
            "Return ONLY valid JSON array of translated strings in the same order and same length as input."
        )
        user_payload = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "texts": texts,
        }
        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        raw = self._retry_call(payload)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"batch_parse_error: {exc}") from exc
        if not isinstance(parsed, list) or len(parsed) != len(texts):
            raise RuntimeError("batch_invalid_shape: response must be a list with same length as input")
        return [str(item) for item in parsed]

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if self.use_batch_api:
            return self._batch(texts, source_lang, target_lang)
        return [self._one(text, source_lang, target_lang) for text in texts]
