"""Central configuration for the Enterprise RAG Assistant.

Keeping all tunable settings in one place makes the pipeline easy to reason about
and easy to explain. Nothing here makes a network call, so this module is safe to
import anywhere.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "chroma_index"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SYNTHETIC_QA_PATH = EVAL_DIR / "synthetic_qa.json"
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.json"

# --- Models (LangChain + GPT-4, per project spec) --------------------------
# The chat model used for answering and for LLM-as-judge evaluation.
CHAT_MODEL = "gpt-4o-mini"
# Embedding model used to vectorise document chunks for retrieval.
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Chunking --------------------------------------------------------------
# Documents are split into overlapping chunks so retrieval can return focused,
# relevant passages rather than whole documents.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval -------------------------------------------------------------
# Number of chunks returned per query and fed to the model as context.
TOP_K = 4

# --- Generation ------------------------------------------------------------
# Low temperature keeps answers grounded and repeatable, which also makes the
# consistency evaluation meaningful.
TEMPERATURE = 0.0
