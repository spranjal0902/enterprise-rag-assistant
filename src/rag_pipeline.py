"""The RAG pipeline: build the index, retrieve, and answer with GPT-4.

This is the core of the assistant. It wires together four pieces:
  1. Embeddings  - turn text chunks into vectors (OpenAI text-embedding-3-small).
  2. Vector store - Chroma, persisted to disk, for similarity search.
  3. Retriever    - fetch the top-k most relevant chunks for a question.
  4. LLM chain    - feed those chunks to GPT-4 with a grounded prompt.

Models are instantiated lazily (inside functions), so importing this module does
NOT require an API key. The key is only needed when you actually build the index
or answer a question.
"""

import time
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src import config, ingest


# --- Prompt ----------------------------------------------------------------
# The prompt is deliberately strict: answer only from the retrieved context and
# admit when the answer is not present. This is what keeps a RAG system honest
# and is the single biggest lever on answer quality.
_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question using "
    "ONLY the context provided below. If the answer is not contained in the "
    "context, say you don't have that information rather than guessing. Be concise "
    "and cite the source file names you used.\n\n"
    "Context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", "{question}")]
)


# --- Model factories (lazy) ------------------------------------------------
def _get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL, temperature=config.TEMPERATURE)


# --- Index build / load ----------------------------------------------------
def build_index(persist: bool = True):
    """Build the Chroma vector store from the sample documents.

    Loads + cleans + chunks the corpus, embeds the chunks, and writes the index
    to disk so subsequent runs can load it without re-embedding.
    """
    from langchain_chroma import Chroma

    chunks = ingest.load_and_chunk()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=str(config.VECTORSTORE_DIR) if persist else None,
    )
    return vectorstore, len(chunks)


def load_index():
    """Load a previously built Chroma index from disk."""
    from langchain_chroma import Chroma

    if not config.VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "No index found. Build it first with `python scripts/build_index.py`."
        )
    return Chroma(
        persist_directory=str(config.VECTORSTORE_DIR),
        embedding_function=_get_embeddings(),
    )


# --- Retrieval + answering -------------------------------------------------
def _format_docs(docs: List[Document]) -> str:
    """Render retrieved chunks into a single context string with sources."""
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def answer_question(question: str, vectorstore=None) -> Tuple[str, List[Document], float]:
    """Answer a question with the RAG pipeline.

    Returns (answer_text, retrieved_documents, latency_seconds).
    Retrieval is done explicitly (rather than hidden in a chain) so the UI can
    show sources and so we can measure latency precisely.
    """
    if vectorstore is None:
        vectorstore = load_index()

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    start = time.perf_counter()
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    chain = _PROMPT | _get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    latency = time.perf_counter() - start

    return answer, docs, latency
