"""Synthetic Q&A generation for evaluation.

Evaluating a RAG system needs a question set with reference answers. Hand-writing
these is slow and does not scale to a large corpus. Instead we use GPT-4 to read
each document chunk and generate grounded question-answer pairs from it. The
document text is the ground truth, so the generated answer is a trustworthy
reference to grade the live system against later.

This is the "synthetic data workflow": we manufacture a labelled evaluation set
directly from the source documents.
"""

import json
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src import config, ingest
from dotenv import load_dotenv
load_dotenv()


_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You create evaluation data for a question-answering system. Given a "
            "passage from an internal company document, write {n} clear, factual "
            "questions that a real employee might ask, each answerable SOLELY from "
            "the passage, together with a correct, concise answer grounded in the "
            "passage. Do not invent facts beyond the passage. Respond as a JSON "
            "list of objects with keys 'question' and 'answer' and nothing else.",
        ),
        ("human", "Passage (from {source}):\n\n{passage}"),
    ]
)


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0.3)


def generate_qa(n_per_chunk: int = 2, max_chunks: int = 8) -> List[Dict]:
    """Generate synthetic Q&A pairs from the corpus chunks.

    Args:
        n_per_chunk: questions to generate per chunk.
        max_chunks: cap on chunks used, to keep token cost predictable.

    Returns a list of {question, answer, source} dicts and writes them to
    config.SYNTHETIC_QA_PATH.
    """
    chunks = ingest.load_and_chunk()[:max_chunks]
    llm = _get_llm()
    parser = JsonOutputParser()
    chain = _GEN_PROMPT | llm | parser

    dataset: List[Dict] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        try:
            pairs = chain.invoke(
                {"n": n_per_chunk, "source": source, "passage": chunk.page_content}
            )
        except Exception as exc:  # one bad chunk should not kill the run
            print(f"  ! skipped a chunk from {source}: {exc}")
            continue
        for pair in pairs:
            if "question" in pair and "answer" in pair:
                dataset.append(
                    {
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "source": source,
                    }
                )

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.SYNTHETIC_QA_PATH.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(dataset)} synthetic Q&A pairs to {config.SYNTHETIC_QA_PATH}")
    return dataset


if __name__ == "__main__":
    generate_qa()
