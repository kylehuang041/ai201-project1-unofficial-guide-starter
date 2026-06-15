"""Build the vector index from documents/ (PDF + Markdown/Text).

Usage:
    python -m src.build_index                 # build with pdfplumber (default)
    python -m src.build_index --extractor spacy-layout
    python -m src.build_index --inspect       # print sample chunks, don't embed
"""
from __future__ import annotations

import argparse
import random

from config import DOCUMENTS_DIR
from src.chunking import chunk_corpus


def inspect(chunks, n=5):
    children = [c for c in chunks if c.kind == "child"]
    parents = [c for c in chunks if c.kind == "parent"]
    toks = [c.metadata["n_tokens"] for c in children]
    print(f"\nDocuments chunked. parents={len(parents)} children={len(children)}")
    print(
        f"child tokens: min={min(toks)} max={max(toks)} "
        f"avg={sum(toks) / len(toks):.0f}"
    )
    print("\n" + "=" * 70)
    print(f"{n} RANDOM CHILD CHUNKS")
    print("=" * 70)
    for c in random.sample(children, min(n, len(children))):
        m = c.metadata
        print(f"\n[{m['source']}] {m['section_path']}  (page {m['page']}, "
              f"{m['n_tokens']} tok)")
        print("-" * 70)
        print(c.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="pdfplumber",
                    choices=["pdfplumber", "spacy-layout"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"Chunking corpus with extractor={args.extractor} ...")
    chunks = chunk_corpus(DOCUMENTS_DIR, extractor=args.extractor)

    if args.inspect:
        inspect(chunks)
        return

    from src.vectorstore import build_index

    stats = build_index(chunks)
    print(f"Indexed {stats['children']} children / {stats['parents']} parents "
          f"into ChromaDB.")
    inspect(chunks)


if __name__ == "__main__":
    main()
