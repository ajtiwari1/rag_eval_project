"""Evaluate the generator in isolation using known-good context chunks."""

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_PROVIDER", "openai").strip().lower() in {"ollama", "local"}:
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from evals.harness import load_goldens, print_summary, summarize_by_metric
from evals.saving_data import save_results
from src.generator import build_rag_generator
from src.llm_factory import get_llm, get_llm_config

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_PROVIDER = get_llm_config()["provider"]
JUDGE_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("JUDGE_MODEL")
    or get_llm_config(provider=JUDGE_PROVIDER)["model"]
)
THRESHOLD = 0.7


class AnswerGenerator(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class LLMJudge(DeepEvalBaseLLM):
    """Adapt the configured chat model for DeepEval's metric judges."""

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        self.provider = provider or JUDGE_PROVIDER
        self.model_name = model_name or JUDGE_MODEL
        self.llm = get_llm(
            provider=self.provider,
            model=self.model_name,
            temperature=0,
        )

    def load_model(self):
        return self.llm

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


def run(
    generator: AnswerGenerator | None = None,
    judge: DeepEvalBaseLLM | None = None,
    golden_path: str = GOLDEN_PATH,
):
    """Evaluate generated answers against each golden context, without retrieval."""
    goldens = load_goldens(golden_path)
    answer_generator = generator if generator is not None else build_rag_generator()
    test_cases = []

    for golden in goldens:
        context = golden["ideal_context"]
        if not isinstance(context, list) or not all(
            isinstance(chunk, str) for chunk in context
        ):
            golden_id = golden.get("id", golden.get("query", "<unknown>"))
            raise ValueError(
                f"Golden {golden_id!r} must have ideal_context as a list of strings."
            )

        answer = answer_generator.generate(golden["query"], "\n\n".join(context))
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=answer,
                retrieval_context=context,
            )
        )

    metric_judge = judge if judge is not None else LLMJudge()
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=metric_judge,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=metric_judge,
            include_reason=True,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
            throttle_value=1,
        ),
        hyperparameters={
            "generator": "RAGGenerator",
            "judge_model": JUDGE_MODEL,
            "golden_set": golden_path,
            "evaluation_mode": "golden_context_isolated",
        },
    )
    save_results(result=result, filename_prefix="generator")
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())
