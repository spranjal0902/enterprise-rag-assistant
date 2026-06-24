"""Build the vector index from the sample documents.

Usage:
    python scripts/build_index.py

Requires OPENAI_API_KEY to be set (see .env.example).
"""

import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src import rag_pipeline  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    print("Building index from sample documents...")
    _vectorstore, n_chunks = rag_pipeline.build_index(persist=True)
    print(f"Done. Indexed {n_chunks} chunks.")
