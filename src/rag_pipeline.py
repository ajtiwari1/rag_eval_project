"""Retrieve, rerank, and generate answers for a query."""

from collections.abc import Sequence
from typing import Protocol, TypedDict

from langchain_core.documents import Document
from langsmith import traceable

from src.generator import build_rag_generator
from src.reranker import RerankingRetriever


class Retriever(Protocol):
    def invoke(self, query: str) -> Sequence[Document]: ...


class Generator(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class RagPipelineResult(TypedDict):
    query: str
    context: list[str]
    answer: str


class RagPipeline:
    """Connect retrieval and reranking to context-grounded generation."""

    def __init__(
        self,
        fetch_k: int = 10,
        top_k: int = 5,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.retriever = (
            retriever
            if retriever is not None
            else RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
        )
        self.generator = (
            generator if generator is not None else build_rag_generator()
        )

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> RagPipelineResult:
        """Return the query, retrieved context chunks, and generated answer."""
        docs = self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]
        answer = self.generator.generate(query, "\n\n".join(context))
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("Why do we need golden datasets?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for index, chunk in enumerate(result["context"]):
        print(f"  [{index}] {chunk[:120]}...")
