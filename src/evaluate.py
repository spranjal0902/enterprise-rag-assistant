"""Evaluation harness: response quality, latency, and contextual consistency.

These are exactly the three dimensions the project set out to measure:

  - Quality:     an LLM-as-judge grades the system's answer against the synthetic
                 reference answer on a 1-5 scale (correctness + groundedness).
  - Latency:     wall-clock seconds per answer, reported as mean and p95.
  - Consistency: the same question is asked twice; a judge decides whether the two
                 answers agree. A robust RAG system should be stable across runs.

The harness reads the synthetic Q&A set, runs each question through the live RAG
pipeline, scores it, and writes an aggregate report.
"""

import json
import statistics
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src import config, rag_pipeline
from dotenv import load_dotenv
load_dotenv()


# --- Judges ----------------------------------------------------------------
_QUALITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Compare the SYSTEM ANSWER to the "
            "REFERENCE ANSWER for the given QUESTION. Score 1-5 where 5 means fully "
            "correct and grounded, 3 means partially correct, and 1 means wrong or "
            "unsupported. Respond as JSON with keys 'score' (integer) and 'reason' "
            "(one short sentence).",
        ),
        (
            "human",
            "QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\n"
            "SYSTEM ANSWER: {answer}",
        ),
    ]
)

_CONSISTENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether two answers to the same question are consistent in "
            "their factual content (ignore wording). Respond as JSON with keys "
            "'consistent' (true/false) and 'reason' (one short sentence).",
        ),
        ("human", "QUESTION: {question}\n\nANSWER A: {a}\n\nANSWER B: {b}"),
    ]
)


def _get_judge():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0.0)


# --- Evaluation ------------------------------------------------------------
def evaluate(limit: int = None) -> Dict:
    """Run the full evaluation and return an aggregate report dict."""
    if not config.SYNTHETIC_QA_PATH.exists():
        raise FileNotFoundError(
            "No synthetic Q&A set found. Generate it first with "
            "`python -m src.synthetic_qa`."
        )

    qa_set: List[Dict] = json.loads(
        config.SYNTHETIC_QA_PATH.read_text(encoding="utf-8")
    )
    if limit:
        qa_set = qa_set[:limit]

    vectorstore = rag_pipeline.load_index()
    judge = _get_judge()
    quality_chain = _QUALITY_PROMPT | judge | JsonOutputParser()
    consistency_chain = _CONSISTENCY_PROMPT | judge | JsonOutputParser()

    per_item: List[Dict] = []
    for i, item in enumerate(qa_set, 1):
        question, reference = item["question"], item["answer"]

        # First answer (also used for quality + latency).
        answer1, _docs, latency = rag_pipeline.answer_question(question, vectorstore)
        # Second answer for the consistency check.
        answer2, _d2, _l2 = rag_pipeline.answer_question(question, vectorstore)

        quality = quality_chain.invoke(
            {"question": question, "reference": reference, "answer": answer1}
        )
        consistency = consistency_chain.invoke(
            {"question": question, "a": answer1, "b": answer2}
        )

        per_item.append(
            {
                "question": question,
                "source": item.get("source"),
                "quality_score": quality.get("score"),
                "quality_reason": quality.get("reason"),
                "latency_seconds": round(latency, 3),
                "consistent": consistency.get("consistent"),
            }
        )
        print(
            f"[{i}/{len(qa_set)}] quality={quality.get('score')} "
            f"latency={latency:.2f}s consistent={consistency.get('consistent')}"
        )

    report = _aggregate(per_item)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_RESULTS_PATH.write_text(
        json.dumps({"summary": report, "items": per_item}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n=== Summary ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nFull results written to {config.EVAL_RESULTS_PATH}")
    return {"summary": report, "items": per_item}


def _aggregate(per_item: List[Dict]) -> Dict:
    """Compute aggregate metrics across all evaluated items."""
    scores = [x["quality_score"] for x in per_item if isinstance(x["quality_score"], (int, float))]
    latencies = [x["latency_seconds"] for x in per_item]
    consistent_flags = [bool(x["consistent"]) for x in per_item]

    def p95(values):
        if not values:
            return None
        s = sorted(values)
        idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return round(s[idx], 3)

    return {
        "n_questions": len(per_item),
        "avg_quality_1to5": round(statistics.mean(scores), 2) if scores else None,
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "p95_latency_seconds": p95(latencies),
        "consistency_rate": round(sum(consistent_flags) / len(consistent_flags), 2)
        if consistent_flags
        else None,
    }


if __name__ == "__main__":
    evaluate()
