from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

from translator.base import TranslationProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(TranslationProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _one(self, text: str, source_lang: str, target_lang: str) -> str:
        system_prompt = (
            "You are a precise translation engine. "
            "Translate only natural language from source to target. "
            "Do not add markup. Preserve URLs, emails, codes, product numbers and units exactly. "
            "Return only translated text."
        )
        user_prompt = f"source_lang={source_lang}\ntarget_lang={target_lang}\ntext:\n{text}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                return (resp.output_text or "").strip()
            except Exception as exc:  # API/library level
                last_err = exc
                delay = 2**attempt
                logger.warning("OpenAI retry %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(delay)
        raise RuntimeError(f"retry_exhausted: {last_err}")

    def translate_texts(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [self._one(text, source_lang, target_lang) for text in texts]
