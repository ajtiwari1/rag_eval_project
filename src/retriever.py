
import os
import re
import glob

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_store"

# Free local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# 1. LOAD: Read transcripts and remove VTT timestamps
def load_transcripts():
    docs = []

    for path in glob.glob(os.path.join(DATA_DIR, "*.vtt")):
        lines = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if (
                    not line
                    or line == "WEBVTT"
                    or "-->" in line
                    or line.isdigit()
                ):
                    continue

                lines.append(line)

        if not lines:
            continue

        text = " ".join(lines)

        match = re.search(r"Session[ _]*(\d+)", os.path.basename(path))
        session = match.group(1) if match else os.path.basename(path)

        docs.append(
            Document(
                page_content=text,
                metadata={"session": session},
            )
        )

    return docs


def load_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

    docs = load_transcripts()

    if not docs:
        raise ValueError(
            f"No .vtt transcripts found in '{DATA_DIR}'."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )


# 3. BUILD RETRIEVER
def build_retriever():
    return load_store().as_retriever(
        search_kwargs={"k": 5}
    )


# 4. TEST
if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke(
        "What is regression testing?"
    )

    for result in results:
        print(
            f"[Session {result.metadata['session']}] "
            f"{result.page_content[:150]}...\n"
        )
