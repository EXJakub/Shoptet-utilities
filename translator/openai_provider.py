from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from openai import AsyncOpenAI, OpenAI

from translator.base import TranslationProvider

logger = logging.getLogger(__name__)


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _coerce_batch_response(raw: str, expected_len: int) -> list[str]:
    candidate = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"batch_parse_error: {exc}") from exc

    if isinstance(parsed, dict):
        for key in ("translations", "items", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break

    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise RuntimeError("batch_invalid_shape: response must be a list with same length as input")

    return [str(item) for item in parsed]


class OpenAIProvider(TranslationProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        use_batch_api: bool = False,
        max_parallel_requests: int = 8,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.use_batch_api = use_batch_api
        self.max_parallel_requests = max(1, max_parallel_requests)

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
            except Exception as exc:
                last_err = exc
                delay = 2**attempt
                logger.warning("OpenAI retry %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(delay)
        raise RuntimeError(f"retry_exhausted: {last_err}")

    async def _retry_call_async(self, input_payload: list[dict[str, str]]) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.async_client.responses.create(
                    model=self.model,
                    input=input_payload,
                    temperature=0,
                )
                return (resp.output_text or "").strip()
            except Exception as exc:
                last_err = exc
                delay = 2**attempt
                logger.warning("OpenAI async retry %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                await asyncio.sleep(delay)
        raise RuntimeError(f"retry_exhausted: {last_err}")

    def _one_payload(self, text: str, source_lang: str, target_lang: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are a precise translation engine. "
            "Translate only natural language from source to target. "
            "Do not add markup. Preserve URLs, emails, codes, product numbers and units exactly. "
            "Return only translated text."
        )
        user_prompt = f"source_lang={source_lang}\ntarget_lang={target_lang}\ntext:\n{text}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _one_async(self, text: str, source_lang: str, target_lang: str, sem: asyncio.Semaphore) -> str:
        payload = self._one_payload(text, source_lang, target_lang)
        async with sem:
            return await self._retry_call_async(payload)

    async def _translate_parallel_async(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        sem = asyncio.Semaphore(self.max_parallel_requests)
        tasks = [self._one_async(text, source_lang, target_lang, sem) for text in texts]
        return await asyncio.gather(*tasks)

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
        return _coerce_batch_response(raw, expected_len=len(texts))

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []

        if self.use_batch_api:
            try:
                return self._batch(texts, source_lang, target_lang)
            except RuntimeError as exc:
                if str(exc).startswith("batch_"):
                    logger.warning("Batch translation failed (%s), falling back to parallel requests.", exc)
                else:
                    raise

        return asyncio.run(self._translate_parallel_async(texts, source_lang, target_lang))
