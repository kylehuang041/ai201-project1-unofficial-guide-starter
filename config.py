"""Central configuration for The Unofficial Guide RAG system."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
CHROMA_DIR = ROOT / "chroma_db"
PARENTS_PATH = ROOT / "chroma_db" / "parents.json"

# --- Embedding / vector store ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 truncates input at 256 word-piece tokens. Child chunks are
# sized to stay under this so the whole chunk actually gets embedded.
EMBED_MAX_TOKENS = 256
COLLECTION_NAME = "unofficial_guide"

# --- Chunking (parent-child) ---
# Children are the *indexed* unit: small, precise, and sized to the embedder
# window. Parents are whole sections, returned to the LLM for richer context.
CHILD_TARGET_TOKENS = 220   # leave headroom under the 256 embedder limit
CHILD_OVERLAP_TOKENS = 40   # ~1-2 sentences of overlap to keep facts retrievable
PARENT_MAX_TOKENS = 1200    # cap so a giant section can't blow up the LLM context

# --- Retrieval ---
TOP_K = 3                   # final parents handed to the LLM
CANDIDATE_K = 20            # candidates pulled from each retriever before fusion
RRF_K = 60                  # reciprocal-rank-fusion constant
MIN_SEMANTIC_SIMILARITY = 0.55  # drop semantic hits below this cosine similarity

# --- LLM (Groq) ---
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_HISTORY_TURNS = 4       # conversational-memory window (Q/A pairs)

# spaCy model used for sentence segmentation + header heuristics.
SPACY_MODEL = "en_core_web_sm"
