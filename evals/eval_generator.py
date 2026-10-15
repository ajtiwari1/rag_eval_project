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
from evals.llm_judge import DEFAULT_JUDGE, JUDGE_MODEL

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
THRESHOLD = 0.7


class AnswerGenerator(Protocol):
    def generate(self, query: str, context: str) -> str: ...


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

    metric_judge = judge if judge is not None else DEFAULT_JUDGE
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
