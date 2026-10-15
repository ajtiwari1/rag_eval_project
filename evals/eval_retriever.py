# eval_retriever.py

import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_PROVIDER", "openai").strip().lower() in {"ollama", "local"}:
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.evaluate.configs import AsyncConfig

from evals.llm_judge import JUDGE_MODEL, LLMJudge
from src.reranker import RerankingRetriever
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)
from evals.saving_data import save_results

GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = 0.7


# 2. Run evaluation
def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    for g in goldens:
        retrieved = retriever.invoke(g["query"])

        retrieval_context = [
            doc.page_content for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output=(
                    "(generator not evaluated in this run)"
                ),
            )
        )

    judge = LLMJudge()
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
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
        "retriever": "reranker",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)
    save_results(result=result)

    return summarize_by_metric(result)


# 3. Standalone execution
def run_local():
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())