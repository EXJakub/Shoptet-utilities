from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAI

from translator.base import TranslationProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAICallEvent:
    mode: str
    input_count: int
    input_chars: int
    latency_ms: int
    success: bool
    error_type: str | None = None
    fallback_used: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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

        self._http_requests_total = 0
        self._translate_calls_total = 0
        self._events: deque[OpenAICallEvent] = deque(maxlen=200)

    def _record_event(self, event: OpenAICallEvent) -> None:
        self._events.append(event)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        events = list(self._events)
        success_count = sum(1 for e in events if e.success)
        error_count = len(events) - success_count
        fallback_count = sum(1 for e in events if e.fallback_used)
        total_chars = sum(e.input_chars for e in events)
        total_items = sum(e.input_count for e in events)
        total_latency_ms = sum(e.latency_ms for e in events)
        return {
            "http_requests_total": self._http_requests_total,
            "translate_calls_total": self._translate_calls_total,
            "events_count": len(events),
            "success_count": success_count,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "total_input_chars": total_chars,
            "total_input_items": total_items,
            "total_latency_ms": total_latency_ms,
            "events": [asdict(e) for e in events],
        }

    def _retry_call(self, input_payload: list[dict[str, str]]) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._http_requests_total += 1
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
                self._http_requests_total += 1
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
            "Do not add markup. Preserve URLs, emails, codes, product numbers, units, and placeholders like __KEEP_0__ exactly. "
            "Return only translated text."
        )
        user_prompt = f"source_lang={source_lang}\ntarget_lang={target_lang}\ntext:\n{text}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _batch_system_prompt(self) -> str:
        return (
            "Translate each item from source_lang to target_lang. "
            "Input is a JSON array of strings. "
            "For each item: preserve meaning and tone; do not add markup, wrappers, explanations, or commentary. "
            "Preserve URLs, emails, product codes, unique identifiers, units, and placeholders like __KEEP_0__ exactly. "
            "If part of an item is non-translatable, keep that part unchanged and translate the rest. "
            "Output requirements: return ONLY a valid JSON array of strings, with exactly the same number of items as input, "
            "in the same order. No markdown fences. No wrapper objects."
        )

    def _batch_payload(self, texts: list[str], source_lang: str, target_lang: str) -> list[dict[str, str]]:
        user_payload = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "texts": texts,
        }
        return [
            {"role": "system", "content": self._batch_system_prompt()},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    async def _one_async(self, text: str, source_lang: str, target_lang: str, sem: asyncio.Semaphore) -> str:
        payload = self._one_payload(text, source_lang, target_lang)
        async with sem:
            return await self._retry_call_async(payload)

    async def _translate_parallel_async(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        sem = asyncio.Semaphore(self.max_parallel_requests)
        tasks = [self._one_async(text, source_lang, target_lang, sem) for text in texts]
        return await asyncio.gather(*tasks)


    async def _translate_batch_chunks_async(
        self,
        text_chunks: list[list[str]],
        source_lang: str,
        target_lang: str,
    ) -> list[list[str]]:
        sem = asyncio.Semaphore(self.max_parallel_requests)

        async def one_chunk(chunk: list[str]) -> list[str]:
            async with sem:
                if not chunk:
                    return []
                payload = self._batch_payload(chunk, source_lang, target_lang)
                raw = await self._retry_call_async(payload)
                return _coerce_batch_response(raw, expected_len=len(chunk))

        tasks = [one_chunk(chunk) for chunk in text_chunks]
        return await asyncio.gather(*tasks)

    def translate_text_chunks(self, text_chunks: list[list[str]], source_lang: str, target_lang: str) -> list[list[str]]:
        if not text_chunks:
            return []

        if not self.use_batch_api:
            return [self.translate_texts(chunk, source_lang, target_lang) for chunk in text_chunks]

        total_count = sum(len(chunk) for chunk in text_chunks)
        total_chars = sum(len(item) for chunk in text_chunks for item in chunk)
        self._translate_calls_total += len(text_chunks)
        started = time.perf_counter()
        try:
            translated_chunks = asyncio.run(self._translate_batch_chunks_async(text_chunks, source_lang, target_lang))
            self._record_event(
                OpenAICallEvent(
                    mode="batch_parallel",
                    input_count=total_count,
                    input_chars=total_chars,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    success=True,
                )
            )
            return translated_chunks
        except RuntimeError as exc:
            if str(exc).startswith("batch_"):
                self._record_event(
                    OpenAICallEvent(
                        mode="batch_parallel",
                        input_count=total_count,
                        input_chars=total_chars,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        success=False,
                        error_type=str(exc).split(":", 1)[0],
                        fallback_used=True,
                    )
                )
                logger.warning("Parallel batch translation failed (%s), falling back to per-chunk translate_texts.", exc)
                return [self.translate_texts(chunk, source_lang, target_lang) for chunk in text_chunks]
            raise

    def _batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []
        payload = self._batch_payload(texts, source_lang, target_lang)
        raw = self._retry_call(payload)
        return _coerce_batch_response(raw, expected_len=len(texts))

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []

        self._translate_calls_total += 1
        input_count = len(texts)
        input_chars = sum(len(t) for t in texts)

        if self.use_batch_api:
            started = time.perf_counter()
            try:
                translated = self._batch(texts, source_lang, target_lang)
                self._record_event(
                    OpenAICallEvent(
                        mode="batch",
                        input_count=input_count,
                        input_chars=input_chars,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        success=True,
                    )
                )
                return translated
            except RuntimeError as exc:
                if str(exc).startswith("batch_"):
                    self._record_event(
                        OpenAICallEvent(
                            mode="batch",
                            input_count=input_count,
                            input_chars=input_chars,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            success=False,
                            error_type=str(exc).split(":", 1)[0],
                            fallback_used=True,
                        )
                    )
                    logger.warning("Batch translation failed (%s), falling back to parallel requests.", exc)
                else:
                    self._record_event(
                        OpenAICallEvent(
                            mode="batch",
                            input_count=input_count,
                            input_chars=input_chars,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            success=False,
                            error_type=type(exc).__name__,
                        )
                    )
                    raise

        started = time.perf_counter()
        try:
            translated = asyncio.run(self._translate_parallel_async(texts, source_lang, target_lang))
            self._record_event(
                OpenAICallEvent(
                    mode="parallel",
                    input_count=input_count,
                    input_chars=input_chars,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    success=True,
                )
            )
            return translated
        except Exception as exc:
            self._record_event(
                OpenAICallEvent(
                    mode="parallel",
                    input_count=input_count,
                    input_chars=input_chars,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    success=False,
                    error_type=type(exc).__name__,
                )
            )
            raise
