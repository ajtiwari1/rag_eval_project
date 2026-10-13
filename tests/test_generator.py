import pytest

from src.generator import RAGGenerator


class StubLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_generator_uses_supplied_context_and_returns_answer():
    llm = StubLLM(type("Response", (), {"content": "It checks changes. [Session 8]"})())

    generator = RAGGenerator(llm=llm)

    assert generator.generate(
        "What is regression testing?",
        "[Source: 8]\nRegression testing checks changes.",
    ) == (
        "It checks changes. [Session 8]"
    )
    rendered_prompt = llm.prompts[0].to_string()
    assert "<COURSE_CONTEXT>" in rendered_prompt
    assert "[Source: 8]" in rendered_prompt
    assert "Regression testing checks changes." in rendered_prompt
    assert "What is regression testing?" in rendered_prompt
    assert "I don't have enough information in the course material to answer that." in (
        rendered_prompt
    )


def test_generator_handles_empty_context_and_content_blocks():
    generator = RAGGenerator(
        llm=StubLLM(type("Response", (), {"content": [{"text": "I do not know."}]})()),
    )

    assert generator.invoke("Unknown question", "") == "I do not know."
    assert "<COURSE_CONTEXT>\n\n</COURSE_CONTEXT>" in (
        generator.llm.prompts[0].to_string()
    )


def test_generator_rejects_blank_query():
    generator = RAGGenerator(llm=StubLLM("unused"))

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate("  ", "some context")
