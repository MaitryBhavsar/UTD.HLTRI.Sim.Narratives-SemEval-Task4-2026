from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential


class MinimumDelay:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, float(delay_seconds))
        self._start: Optional[float] = None

    def __enter__(self) -> None:
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is None:
            return
        elapsed = time.time() - self._start
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    temperature: float
    max_tokens: int
    seed: Optional[int] = None


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        min_delay_seconds: float = 0.0,
    ) -> None:
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_to_use:
            raise RuntimeError("OPENAI_API_KEY is not set (required for this baseline).")
        self._client = OpenAI(api_key=api_key_to_use, base_url=base_url)
        self._max_retries = max_retries
        self._min_delay_seconds = float(min_delay_seconds)

    @retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(3))
    def generate(
        self,
        *,
        system_message: str,
        user_message: str,
        request: ChatRequest,
    ) -> str:
        with MinimumDelay(self._min_delay_seconds):
            try:
                kwargs = {
                    "model": request.model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                }
                if request.seed is not None:
                    kwargs["seed"] = int(request.seed)
                resp = self._client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except BadRequestError as e:
                # If the model refuses or response is filtered, surface the error.
                raise RuntimeError(f"OpenAI BadRequestError: {e}") from e

