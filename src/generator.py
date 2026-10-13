"""Generate answers from retrieved context using the configured chat model."""

import argparse
from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document

from src.llm_factory import get_llm
from src.retriever import build_retriever


class Retriever(Protocol):
    def invoke(self, query: str) -> Sequence[Document]: ...


class LanguageModel(Protocol):
    def invoke(self, prompt: str) -> object: ...


class RAGGenerator:
    """Retrieve relevant documents and use them to answer a query."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LanguageModel | None = None,
    ) -> None:
        self.retriever = retriever if retriever is not None else build_retriever()
        self.llm = llm if llm is not None else get_llm()

    @staticmethod
    def _format_context(documents: Sequence[Document]) -> str:
        if not documents:
            return "No relevant context was retrieved."

        parts = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("session") or document.metadata.get(
                "source", f"Document {index}"
            )
            parts.append(f"[Source: {source}]\n{document.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _response_text(response: object) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = next(
                        (
                            item[key]
                            for key in ("text", "content", "output_text")
                            if key in item
                        ),
                        None,
                    )
                    if text is None:
                        raise TypeError("LLM response contains an unsupported content block.")
                    parts.append(str(text))
                else:
                    raise TypeError("LLM response contains an unsupported content block.")
            return "".join(parts).strip()
        raise TypeError("LLM response content must be text or a list of text blocks.")

    def generate(self, query: str) -> str:
        """Return an answer grounded in documents retrieved for ``query``."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        documents = self.retriever.invoke(query)
        context = self._format_context(documents)
        prompt = (
            "Answer the question using only the context below. If the context "
            "does not contain the answer, say you do not know. Cite a source "
            "label from the context when useful.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return self._response_text(self.llm.invoke(prompt))

    def invoke(self, query: str) -> str:
        """Provide the standard LangChain-style invocation interface."""
        return self.generate(query)


def build_rag_generator(
    retriever: Retriever | None = None,
    llm: LanguageModel | None = None,
) -> RAGGenerator:
    """Build a RAG answer generator with the project's default components."""
    return RAGGenerator(retriever=retriever, llm=llm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an answer with the RAG pipeline.")
    parser.add_argument("query", help="talk about regr ession testing")
    args = parser.parse_args()

    print(build_rag_generator().generate(args.query))


if __name__ == "__main__":
    main()
