from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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


@dataclass(slots=True)
class BatchItem:
    id: str
    text: str


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _coerce_batch_response(raw: str, expected_order: list[str]) -> dict[str, str]:
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

    if not isinstance(parsed, list):
        raise RuntimeError("batch_invalid_shape: response must be a list")
    if not parsed:
        return {}
    if all(isinstance(item, str) for item in parsed):
        if len(parsed) != len(expected_order):
            raise RuntimeError("batch_invalid_shape: list[str] length mismatch")
        return {item_id: str(text) for item_id, text in zip(expected_order, parsed)}

    out: dict[str, str] = {}
    expected_ids = set(expected_order)
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id", "")).strip()
        if not item_id or item_id not in expected_ids:
            continue
        translated = row.get("translated", row.get("text"))
        if translated is None:
            continue
        out[item_id] = str(translated)
    if not out:
        raise RuntimeError("batch_invalid_shape: no valid id->translated rows")
    return out


class OpenAIProvider(TranslationProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        use_batch_api: bool = False,
        max_parallel_requests: int = 8,
        partial_recovery_max_attempts: int = 2,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.use_batch_api = use_batch_api
        self.max_parallel_requests = max(1, max_parallel_requests)
        self.partial_recovery_max_attempts = max(0, partial_recovery_max_attempts)

        self._http_requests_total = 0
        self._translate_calls_total = 0
        self._batch_invalid_shape_count = 0
        self._partial_recovery_count = 0
        self._isolated_item_fallback_count = 0
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
            "batch_invalid_shape_count": self._batch_invalid_shape_count,
            "partial_recovery_count": self._partial_recovery_count,
            "isolated_item_fallback_count": self._isolated_item_fallback_count,
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
                resp = self.client.responses.create(model=self.model, input=input_payload, temperature=0)
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
                resp = await self.async_client.responses.create(model=self.model, input=input_payload, temperature=0)
                return (resp.output_text or "").strip()
            except Exception as exc:
                last_err = exc
                delay = 2**attempt
                logger.warning("OpenAI async retry %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                await asyncio.sleep(delay)
        raise RuntimeError(f"retry_exhausted: {last_err}")

    def _one_payload(self, text: str, source_lang: str, target_lang: str, strict_change: bool = False) -> list[dict[str, str]]:
        system_prompt = (
            "You are a precise translation engine. "
            "Translate only natural language from source to target. "
            "Do not add markup. Preserve URLs, emails, codes, product numbers, units, and placeholders like __KEEP_0__ exactly. "
            "For closely related languages (for example Czech->Slovak), still translate lexical words; "
            "do not keep Czech-only wording when a Slovak equivalent exists. "
            "Return only translated text."
        )
        if strict_change:
            system_prompt += (
                " If source_lang and target_lang are different and the input contains translatable natural language, "
                "do not return an unchanged copy. Keep non-translatable tokens exactly, but translate surrounding language."
            )
        user_prompt = f"source_lang={source_lang}\ntarget_lang={target_lang}\ntext:\n{text}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _batch_system_prompt(self, strict_change: bool = False) -> str:
        prompt = (
            "Translate each item from source_lang to target_lang. "
            "Input is a JSON array of objects in shape {id,text}. "
            "For each item: preserve meaning and tone; do not add markup, wrappers, explanations, or commentary. "
            "Preserve URLs, emails, product codes, unique identifiers, units, and placeholders like __KEEP_0__ exactly. "
            "If part of an item is non-translatable, keep that part unchanged and translate the rest. "
            "For closely related languages (for example Czech->Slovak), still translate lexical words and avoid Czech-only wording "
            "when a Slovak equivalent exists. "
            "Output requirements: return ONLY a valid JSON array of objects with shape {id,translated}. "
            "IDs must be copied exactly from input. No markdown fences. No wrapper objects."
        )
        if strict_change:
            prompt += (
                " For each item containing translatable natural language where source_lang != target_lang, "
                "the translated field must not be an unchanged copy of the input text."
            )
        return prompt

    def _batch_payload(
        self,
        items: list[BatchItem],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> list[dict[str, str]]:
        user_payload = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "items": [{"id": item.id, "text": item.text} for item in items],
        }
        return [
            {"role": "system", "content": self._batch_system_prompt(strict_change=strict_change)},
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

    async def _translate_items_batch_once(
        self,
        items: list[BatchItem],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> dict[str, str]:
        payload = self._batch_payload(items, source_lang, target_lang, strict_change=strict_change)
        raw = await self._retry_call_async(payload)
        return _coerce_batch_response(raw, expected_order=[item.id for item in items])

    async def _translate_item_with_isolation(
        self,
        item: BatchItem,
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> str:
        self._isolated_item_fallback_count += 1
        payload = self._one_payload(item.text, source_lang, target_lang, strict_change=strict_change)
        return await self._retry_call_async(payload)

    async def _translate_items_with_recovery(
        self,
        items: list[BatchItem],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> dict[str, str]:
        pending = list(items)
        resolved: dict[str, str] = {}
        attempts = 0
        while pending and attempts <= self.partial_recovery_max_attempts:
            attempts += 1
            try:
                partial = await self._translate_items_batch_once(
                    pending,
                    source_lang,
                    target_lang,
                    strict_change=strict_change,
                )
            except RuntimeError as exc:
                if not str(exc).startswith("batch_"):
                    raise
                self._batch_invalid_shape_count += 1
                partial = {}
            if partial:
                resolved.update(partial)
                unresolved = [item for item in pending if item.id not in partial]
                if unresolved:
                    self._partial_recovery_count += 1
                pending = unresolved
            else:
                break

        if not pending:
            return resolved

        if len(pending) == 1:
            item = pending[0]
            resolved[item.id] = await self._translate_item_with_isolation(
                item,
                source_lang,
                target_lang,
                strict_change=strict_change,
            )
            return resolved

        # Granular fallback: recursively split only problematic subset.
        mid = len(pending) // 2
        left = await self._translate_items_with_recovery(
            pending[:mid],
            source_lang,
            target_lang,
            strict_change=strict_change,
        )
        right = await self._translate_items_with_recovery(
            pending[mid:],
            source_lang,
            target_lang,
            strict_change=strict_change,
        )
        resolved.update(left)
        resolved.update(right)
        return resolved

    async def _translate_batch_chunks_async(
        self,
        text_chunks: list[list[str]],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> list[list[str]]:
        sem = asyncio.Semaphore(self.max_parallel_requests)

        async def one_chunk(chunk: list[str], chunk_idx: int) -> list[str]:
            async with sem:
                if not chunk:
                    return []
                items = [BatchItem(id=f"c{chunk_idx}_i{i}", text=text) for i, text in enumerate(chunk)]
                resolved = await self._translate_items_with_recovery(
                    items,
                    source_lang,
                    target_lang,
                    strict_change=strict_change,
                )
                return [resolved[item.id] for item in items]

        tasks = [one_chunk(chunk, idx) for idx, chunk in enumerate(text_chunks)]
        return await asyncio.gather(*tasks)

    def translate_text_chunks(
        self,
        text_chunks: list[list[str]],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> list[list[str]]:
        if not text_chunks:
            return []
        if not self.use_batch_api:
            return [self.translate_texts(chunk, source_lang, target_lang) for chunk in text_chunks]

        total_count = sum(len(chunk) for chunk in text_chunks)
        total_chars = sum(len(item) for chunk in text_chunks for item in chunk)
        self._translate_calls_total += len(text_chunks)
        baseline_recoveries = self._partial_recovery_count
        baseline_isolations = self._isolated_item_fallback_count
        started = time.perf_counter()
        try:
            translated_chunks = asyncio.run(
                self._translate_batch_chunks_async(
                    text_chunks,
                    source_lang,
                    target_lang,
                    strict_change=strict_change,
                )
            )
            self._record_event(
                OpenAICallEvent(
                    mode="batch_parallel",
                    input_count=total_count,
                    input_chars=total_chars,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    success=True,
                    fallback_used=(
                        self._partial_recovery_count > baseline_recoveries
                        or self._isolated_item_fallback_count > baseline_isolations
                    ),
                )
            )
            return translated_chunks
        except RuntimeError as exc:
            if str(exc).startswith("batch_"):
                self._batch_invalid_shape_count += 1
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
                return [
                    self.translate_texts(chunk, source_lang, target_lang, strict_change=strict_change)
                    for chunk in text_chunks
                ]
            raise

    def _batch(self, texts: list[str], source_lang: str, target_lang: str, strict_change: bool = False) -> list[str]:
        if not texts:
            return []
        items = [BatchItem(id=f"i{i}", text=text) for i, text in enumerate(texts)]
        payload = self._batch_payload(items, source_lang, target_lang, strict_change=strict_change)
        raw = self._retry_call(payload)
        mapping = _coerce_batch_response(raw, expected_order=[item.id for item in items])
        missing = [item for item in items if item.id not in mapping]
        if missing:
            self._partial_recovery_count += 1
            for item in missing:
                mapping[item.id] = self._retry_call(
                    self._one_payload(item.text, source_lang, target_lang, strict_change=strict_change)
                )
                self._isolated_item_fallback_count += 1
        return [mapping[item.id] for item in items]

    def translate_texts(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        strict_change: bool = False,
    ) -> list[str]:
        if not texts:
            return []

        self._translate_calls_total += 1
        input_count = len(texts)
        input_chars = sum(len(t) for t in texts)

        if self.use_batch_api:
            baseline_recoveries = self._partial_recovery_count
            baseline_isolations = self._isolated_item_fallback_count
            started = time.perf_counter()
            try:
                translated = self._batch(texts, source_lang, target_lang, strict_change=strict_change)
                self._record_event(
                    OpenAICallEvent(
                        mode="batch",
                        input_count=input_count,
                        input_chars=input_chars,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        success=True,
                        fallback_used=(
                            self._partial_recovery_count > baseline_recoveries
                            or self._isolated_item_fallback_count > baseline_isolations
                        ),
                    )
                )
                return translated
            except RuntimeError as exc:
                if str(exc).startswith("batch_"):
                    self._batch_invalid_shape_count += 1
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
