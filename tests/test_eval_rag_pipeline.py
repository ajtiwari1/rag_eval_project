from evals import eval_rag_pipeline
from evals import llm_judge


class StubRag:
    def __init__(self):
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return {
            "answer": "Generated answer.",
            "context": ["Retrieved context."],
        }


def test_run_uses_injected_judge_and_reports_progress(monkeypatch, capsys):
    judge = object()
    captured = {}
    monkeypatch.setattr(
        eval_rag_pipeline,
        "load_goldens",
        lambda _: [{"query": "What is RAG?"}],
    )
    for metric_name in (
        "ContextualRelevancyMetric",
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
    ):
        monkeypatch.setattr(
            eval_rag_pipeline,
            metric_name,
            lambda **kwargs: kwargs,
        )
    monkeypatch.setattr(
        eval_rag_pipeline,
        "evaluate",
        lambda **kwargs: captured.update(kwargs) or "evaluation-result",
    )
    monkeypatch.setattr(
        eval_rag_pipeline,
        "summarize_by_metric",
        lambda result: {"evaluation": result},
    )

    rag = StubRag()
    summary = eval_rag_pipeline.run(rag, judge=judge)

    output = capsys.readouterr().out
    assert rag.queries == ["What is RAG?"]
    assert [metric["model"] for metric in captured["metrics"]] == [judge] * 3
    assert captured["test_cases"][0].retrieval_context == ["Retrieved context."]
    assert captured["async_config"].run_async is False
    assert captured["display_config"].show_indicator is True
    assert "[RAG 1/1] Running query..." in output
    assert "[RAG 1/1] Complete" in output
    assert "[EVAL] Scoring 1 case(s) with 3 metric(s)..." in output
    assert "[EVAL] Complete" in output
    assert summary == {"evaluation": "evaluation-result"}


def test_llm_judge_builds_model_from_selected_provider(monkeypatch):
    captured = {}
    llm = object()
    monkeypatch.setattr(
        llm_judge,
        "get_llm",
        lambda **kwargs: captured.update(kwargs) or llm,
    )

    judge = llm_judge.LLMJudge(provider="gemini", model_name="gemini-test")

    assert judge.load_model() is llm
    assert captured == {
        "provider": "gemini",
        "model": "gemini-test",
        "temperature": 0,
    }
