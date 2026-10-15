"""Shared DeepEval judge backed by the project's configured LLM provider."""

import os
from typing import Any

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

from src.llm_factory import get_llm, get_llm_config

load_dotenv()

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER") or get_llm_config()["provider"]
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or get_llm_config(
    provider=JUDGE_PROVIDER
)["model"]


class LLMJudge(DeepEvalBaseLLM):
    """Adapt the configured chat model for DeepEval's metric judges."""

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        config = get_llm_config(
            provider=provider or JUDGE_PROVIDER,
            model=model_name or JUDGE_MODEL,
            temperature=0,
        )
        self.provider = config["provider"]
        self.model_name = config["model"]
        self._llm: Any = None

    def load_model(self):
        if self._llm is None:
            self._llm = get_llm(
                provider=self.provider,
                model=self.model_name,
                temperature=0,
            )
        return self._llm

    @property
    def llm(self):
        return self.load_model()

    @llm.setter
    def llm(self, model: Any) -> None:
        self._llm = model

    def get_model_name(self) -> str:
        return self.model_name

    @staticmethod
    def _normalize_response_content(content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "content", "output_text"):
                        if key in item:
                            parts.append(str(item[key]))
                            break
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        if isinstance(content, dict):
            for key in ("text", "content", "output_text"):
                if key in content:
                    return str(content[key])
        return str(content)

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return self._normalize_response_content(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return self._normalize_response_content(response.content)


DEFAULT_JUDGE = LLMJudge()
