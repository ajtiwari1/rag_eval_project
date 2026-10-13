# eval_retriever.py

import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.evaluate.configs import AsyncConfig

from src.llm_factory import get_llm, get_llm_config
from src.reranker import RerankingRetriever
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)
from evals.saving_data import save_results

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
JUDGE_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("JUDGE_MODEL")
    or get_llm_config(provider=JUDGE_PROVIDER)["model"]
)
THRESHOLD = 0.7


# 1. Custom DeepEval judge using the configured provider
class LLMJudge(DeepEvalBaseLLM):

    def __init__(self, provider=None, model_name=None):
        self.provider = provider or JUDGE_PROVIDER
        self.model_name = model_name or JUDGE_MODEL
        self.llm = get_llm(
            provider=self.provider,
            model=self.model_name,
            temperature=0,
        )

    def load_model(self):
        return self.llm

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content


GroqJudge = LLMJudge


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

    judge = GroqJudge()

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