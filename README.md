# Enterprise RAG Assistant

A retrieval-augmented generation (RAG) assistant for enterprise knowledge
retrieval, built with **LangChain** and **GPT-4**. Ask natural-language questions
and get answers grounded in your internal documents, with the source passages
shown alongside each answer.

> **About this repository.** This is a clean, public **reference implementation**
> of the architecture. It runs on a small sample document set so it is easy to
> clone and try. The pipeline is identical to one designed to scale to a large
> enterprise corpus (tens of thousands of documents) — only the corpus size and
> the embedding/vector-store backing differ at scale.

---

## What it does

1. **Ingests and cleans** enterprise documents — normalising whitespace, stripping
   boilerplate (page markers, confidentiality stamps, rule lines) so retrieval
   keys on content, not formatting noise.
2. **Indexes** the cleaned text as vector embeddings in a Chroma vector store.
3. **Answers questions** by retrieving the most relevant passages and passing them
   to GPT-4 under a strict, grounded prompt (answer only from context; admit when
   the answer is not present; cite sources).
4. **Generates a synthetic Q&A evaluation set** from the documents using GPT-4, so
   the system can be measured without hand-labelling.
5. **Evaluates** the assistant on three dimensions: **response quality**
   (LLM-as-judge, 1–5), **latency**, and **contextual consistency** (same question
   asked twice).

---

## Architecture

```
 documents ─▶ clean ─▶ chunk ─▶ embed ─▶ Chroma vector store
                                              │
                              question ─▶ retrieve top-k
                                              │
                              context + question ─▶ GPT-4 ─▶ grounded answer + sources

 Evaluation:  documents ─▶ GPT-4 ─▶ synthetic Q&A set ─▶ run through pipeline
                                                          ─▶ judge quality
                                                          ─▶ measure latency
                                                          ─▶ check consistency
```

---

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd enterprise-rag-assistant

# 2. (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env        # then edit .env and paste your key
```

The project uses GPT-4 and OpenAI embeddings, so an `OPENAI_API_KEY` is required
to build the index and run the assistant.

---

## Usage

```bash
# 1. Build the vector index from the sample documents
python scripts/build_index.py

# 2. Launch the assistant UI
streamlit run app.py
```

To run the evaluation loop:

```bash
# Generate a synthetic Q&A set from the documents
python -m src.synthetic_qa

# Evaluate quality, latency, and consistency on that set
python -m src.evaluate
```

Evaluation results are written to `data/eval/eval_results.json` and the summary is
shown in the **Evaluation** tab of the app.

---

## Project structure

```
enterprise-rag-assistant/
├── app.py                     # Streamlit UI (Assistant + Evaluation tabs)
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/           # sample enterprise corpus (handbook, policies, FAQ)
├── scripts/
│   └── build_index.py         # build the Chroma index
└── src/
    ├── config.py              # all tunable settings in one place
    ├── ingest.py              # load -> clean -> chunk (data-cleaning workflow)
    ├── rag_pipeline.py        # embeddings, vector store, retriever, GPT-4 chain
    ├── synthetic_qa.py        # synthetic Q&A generation for evaluation
    └── evaluate.py            # quality / latency / consistency harness
```

---

## Design notes

- **Grounded prompting.** The system prompt forbids answering outside the retrieved
  context and asks the model to cite source files. This is the main lever against
  hallucination in a RAG system.
- **Synthetic evaluation data.** Reference answers are generated directly from the
  source passages, so the documents themselves act as ground truth. This makes
  evaluation scale to large corpora where hand-labelling is impractical.
- **Three evaluation axes.** Quality, latency, and consistency are tracked together
  because a useful assistant has to be correct, fast, and stable — improving one at
  the expense of another is easy to miss without measuring all three.
- **Scaling.** For a large corpus the same pipeline swaps the local Chroma store for
  a managed/distributed vector database and batches the embedding step; the
  retrieval, prompting, and evaluation code is unchanged.

---

## Requirements

See `requirements.txt`. Core stack: `langchain`, `langchain-openai`,
`langchain-community`, `langchain-chroma`, `chromadb`, `streamlit`,
`python-dotenv`.
