from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from evals.llm_judge import DEFAULT_JUDGE
from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfullness_dataset.json"   # reuse the queries
THRESHOLD = 0.7


def run(rag, judge: DeepEvalBaseLLM | None = None):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    total_queries = len(goldens)
    for index, g in enumerate(goldens, start=1):
        print(f"[RAG {index}/{total_queries}] Running query...", flush=True)
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate
        print(f"[RAG {index}/{total_queries}] Complete", flush=True)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metric_judge = judge if judge is not None else DEFAULT_JUDGE
    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=metric_judge,
            include_reason=True,
        ),
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

    # 4. EVALUATE
    print(
        f"[EVAL] Scoring {len(test_cases)} case(s) with {len(metrics)} metric(s)...",
        flush=True,
    )
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
            throttle_value=1,
        ),
        display_config=DisplayConfig(show_indicator=True),
    )
    print("[EVAL] Complete", flush=True)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
