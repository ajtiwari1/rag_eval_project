from langchain_core.documents import Document

from src.rag_pipeline import RagPipeline


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class StubGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, query, context):
        self.calls.append((query, context))
        return self.answer


def test_pipeline_returns_query_context_and_answer():
    retriever = StubRetriever(
        [
            Document(page_content="First context chunk."),
            Document(page_content="Second context chunk."),
        ]
    )
    generator = StubGenerator("Generated answer.")
    pipeline = RagPipeline(retriever=retriever, generator=generator)

    result = pipeline.invoke("What is being evaluated?")

    assert retriever.queries == ["What is being evaluated?"]
    assert generator.calls == [
        (
            "What is being evaluated?",
            "First context chunk.\n\nSecond context chunk.",
        )
    ]
    assert result == {
        "query": "What is being evaluated?",
        "context": ["First context chunk.", "Second context chunk."],
        "answer": "Generated answer.",
    }


def test_pipeline_handles_no_retrieved_documents():
    retriever = StubRetriever([])
    generator = StubGenerator("I don't have enough information.")
    pipeline = RagPipeline(retriever=retriever, generator=generator)

    result = pipeline.invoke("Unanswered question")

    assert generator.calls == [("Unanswered question", "")]
    assert result["context"] == []
    assert result["answer"] == "I don't have enough information."
