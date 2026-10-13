import pytest

from evals import eval_generator


class StubGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, context):
        self.calls.append((query, context))
        return "Generated answer."


def test_run_evaluates_answers_against_supplied_golden_context(monkeypatch):
    goldens = [
        {
            "id": "faithfulness-1",
            "query": "What is online evaluation?",
            "ideal_context": ["First chunk.", "Second chunk."],
        }
    ]
    generator = StubGenerator()
    captured = {}

    monkeypatch.setattr(eval_generator, "load_goldens", lambda _: goldens)
    monkeypatch.setattr(eval_generator, "FaithfulnessMetric", lambda **kwargs: kwargs)
    monkeypatch.setattr(eval_generator, "AnswerRelevancyMetric", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        eval_generator,
        "evaluate",
        lambda **options: captured.update(options) or "evaluation-result",
    )
    monkeypatch.setattr(eval_generator, "save_results", lambda **_: None)
    monkeypatch.setattr(
        eval_generator,
        "summarize_by_metric",
        lambda result: {"evaluation": result},
    )

    summary = eval_generator.run(
        generator=generator,
        judge=object(),
        golden_path="goldens/test_faithfulness.json",
    )

    assert generator.calls == [
        ("What is online evaluation?", "First chunk.\n\nSecond chunk.")
    ]
    test_case = captured["test_cases"][0]
    assert test_case.input == "What is online evaluation?"
    assert test_case.actual_output == "Generated answer."
    assert test_case.retrieval_context == ["First chunk.", "Second chunk."]
    assert test_case.expected_output is None
    assert summary == {"evaluation": "evaluation-result"}


def test_run_rejects_invalid_ideal_context(monkeypatch):
    monkeypatch.setattr(
        eval_generator,
        "load_goldens",
        lambda _: [{"id": "bad-context", "query": "Question?", "ideal_context": "not chunks"}],
    )

    with pytest.raises(ValueError, match="ideal_context as a list of strings"):
        eval_generator.run(generator=StubGenerator())
