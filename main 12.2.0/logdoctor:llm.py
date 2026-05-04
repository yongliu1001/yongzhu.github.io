from __future__ import annotations

import json
import os
import time
from typing import Any

from .config import Config
from .utils import setup_logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMBase:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._logger = setup_logger(__name__ + ".LLM")

    def call(
        self, msgs: list[dict], fmt: type, model: str | None = None
    ) -> tuple[Any | None, str]:
        model = model or self.config.model
        for attempt in range(self.config.llm_retries):
            try:
                result = self._call_impl(msgs, fmt, model)
                if result is not None:
                    return result, ""
            except Exception as e:
                self._logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
                time.sleep(1)
        fallback = self.config.fallback_model
        if fallback and fallback != model:
            try:
                result = self._call_impl(msgs, fmt, fallback)
                return result, ""
            except Exception as e:
                self._logger.warning("Fallback failed: %s", e)
        return None, "LLM failed after retries"

    def call_raw(
        self, msgs: list[dict], model: str | None = None
    ) -> tuple[str | None, str]:
        model = model or self.config.model
        for _ in range(self.config.llm_retries):
            try:
                text = self._call_raw_impl(msgs, model)
                if text:
                    return text, ""
            except Exception as e:
                self._logger.warning("Raw call attempt failed: %s", e)
        return None, "LLM raw call failed"

    def _call_impl(self, msgs: list[dict], fmt: type, model: str) -> Any:
        raise NotImplementedError

    def _call_raw_impl(self, msgs: list[dict], model: str) -> str:
        raise NotImplementedError


class OpenAILLM(LLMBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package is required")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _call_impl(self, msgs, fmt, model):
        r = self.client.beta.chat.completions.parse(
            model=model,
            messages=msgs,
            response_format=fmt,
            timeout=self.config.llm_timeout,
        )
        return r.choices[0].message.parsed

    def _call_raw_impl(self, msgs, model):
        r = self.client.chat.completions.create(
            model=model,
            messages=msgs,
            timeout=self.config.llm_timeout,
        )
        return r.choices[0].message.content


class OllamaLLM(LLMBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package is required")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = OpenAI(base_url=f"{host}/v1", api_key="ollama")

    def _call_impl(self, msgs, fmt, model):
        try:
            r = self.client.chat.completions.create(
                model=model,
                messages=msgs,
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=self.config.llm_timeout,
            )
            data = json.loads(r.choices[0].message.content)
            return fmt.model_validate(data)
        except Exception:
            raw_text = self._call_raw_impl(msgs, model)
            if raw_text:
                try:
                    data = json.loads(raw_text)
                    return fmt.model_validate(data)
                except Exception:
                    raise
            raise

    def _call_raw_impl(self, msgs, model):
        r = self.client.chat.completions.create(
            model=model,
            messages=msgs,
            timeout=self.config.llm_timeout,
        )
        return r.choices[0].message.content


def create_llm(config: Config) -> LLMBase:
    if os.getenv("OLLAMA_HOST"):
        return OllamaLLM(config)
    return OpenAILLM(config)