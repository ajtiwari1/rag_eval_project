import pytest
from langchain_core.documents import Document

from src.generator import RAGGenerator


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class StubLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_generator_uses_retrieved_context_and_returns_answer():
    retriever = StubRetriever(
        [Document(page_content="Regression testing checks changes.", metadata={"session": "8"})]
    )
    llm = StubLLM(type("Response", (), {"content": "It checks changes. [Session 8]"})())

    generator = RAGGenerator(retriever=retriever, llm=llm)

    assert generator.generate("What is regression testing?") == (
        "It checks changes. [Session 8]"
    )
    assert retriever.queries == ["What is regression testing?"]
    assert "[Source: 8]" in llm.prompts[0]
    assert "Regression testing checks changes." in llm.prompts[0]
    assert "What is regression testing?" in llm.prompts[0]


def test_generator_handles_no_retrieved_documents_and_content_blocks():
    generator = RAGGenerator(
        retriever=StubRetriever([]),
        llm=StubLLM(type("Response", (), {"content": [{"text": "I do not know."}]})()),
    )

    assert generator.invoke("Unknown question") == "I do not know."
    assert "No relevant context was retrieved." in generator.llm.prompts[0]


def test_generator_rejects_blank_query():
    generator = RAGGenerator(retriever=StubRetriever([]), llm=StubLLM("unused"))

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate("  ")
