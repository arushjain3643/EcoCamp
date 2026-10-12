"""Thin adapters over the official OpenAI and Google Gen AI Python SDKs.

* SDKs are imported lazily, only when a call is made, so the engine imports and
  runs offline without them installed.
* API keys are read from environment variables at call time. They are never
  stored in configuration, logged, or included in error details.
* Every failure is normalised to ``LLMError`` subclasses carrying only a short
  code or the SDK exception *class name*; exception messages are dropped
  because provider errors can echo parts of credentials.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

try:  # Python 3.8+ has Protocol in typing
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]

from .config import LLMConfig, LLMProvider


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """The LLM cannot be used (``code``: ``missing_api_key`` or ``sdk_unavailable``)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class LLMTimeoutError(LLMError):
    """The request timed out."""


class LLMProviderError(LLMError):
    """The provider/SDK failed (connection, auth, rate limit, server error...)."""


class LLMClient(Protocol):
    """Anything with ``generate_json``; tests inject fakes of this shape."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        ...


def _classify(exc: BaseException) -> LLMError:
    """Map an SDK exception to an ``LLMError`` without keeping its message."""
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "timeout" in name.lower() or "timedout" in name.lower():
        return LLMTimeoutError(name)
    return LLMProviderError(name)


class OpenAIClient:
    """OpenAI Chat Completions in JSON mode (official ``openai`` SDK)."""

    def __init__(self, model: str, api_key: str, timeout_seconds: float) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMUnavailableError("sdk_unavailable") from None
        try:
            client = OpenAI(api_key=self._api_key, timeout=self._timeout, max_retries=0)
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _classify(exc) from None
        if not isinstance(content, str):
            raise LLMProviderError("EmptyResponse")
        return content


class GeminiClient:
    """Gemini via the official ``google-genai`` SDK, JSON response mode."""

    def __init__(self, model: str, api_key: str, timeout_seconds: float) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise LLMUnavailableError("sdk_unavailable") from None
        try:
            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self._timeout * 1000)),
            )
            response = client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            content = response.text
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _classify(exc) from None
        if not isinstance(content, str):
            raise LLMProviderError("EmptyResponse")
        return content


def create_llm_client(
    config: LLMConfig, environ: Optional[Mapping[str, str]] = None
) -> LLMClient:
    """Build a client for ``config``, reading the API key from the environment.

    Raises:
        LLMUnavailableError: ``missing_api_key`` if the key variable is unset or empty.
    """
    env = os.environ if environ is None else environ
    key = (env.get(config.resolved_api_key_env_var) or "").strip()
    if not key:
        raise LLMUnavailableError("missing_api_key")
    if config.model is None:
        raise ValueError("llm.model must be set to create an LLM client")
    if config.provider is LLMProvider.OPENAI:
        return OpenAIClient(config.model, key, config.timeout_seconds)
    return GeminiClient(config.model, key, config.timeout_seconds)
