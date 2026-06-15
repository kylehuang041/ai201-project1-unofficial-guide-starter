"""Evaluation harness.

Runs the 5 test questions end-to-end, judges retrieval (did the gold section
make the top-k?), captures the grounded answer, and compares the three search
modes (hybrid / semantic / bm25) so the hybrid-search stretch feature can be
reported against semantic-only.

    python -m src.evaluate            # full run, writes eval_results.md
    python -m src.evaluate --no-llm   # retrieval-only (skips Groq calls)
"""
from __future__ import annotations

from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    __package__ = "src"

import argparse

from .rag import ask
from .retrieval import get_retriever

# Each gold answer is verifiable against a specific section of a specific guide.
TEST_QUESTIONS = [
    {
        "q": "Which shortest-path algorithm handles graphs with negative edge "
             "weights, and what special structure can it detect?",
        "expected": "Bellman-Ford; it is used when edges may be negative and it "
                    "detects negative cycles.",
        "gold_source": "Graphs_Study_Guide.pdf",
        "gold_section": "Bellman-Ford",
    },
    {
        "q": "In the greedy interval scheduling pattern, what should you sort the "
             "intervals by before selecting them?",
        "expected": "Sort by end time (finish earliest) and choose an interval "
                    "if it doesn't overlap the last chosen one.",
        "gold_source": "Greedy_Algorithms_Study_Guide.pdf",
        "gold_section": "Interval Scheduling",
    },
    {
        "q": "What are the four key components of the universal backtracking "
             "template?",
        "expected": "State, Choices, Constraints, and Goal.",
        "gold_source": "Backtracking_Study_Guide.pdf",
        "gold_section": "Backtracking Template",
    },
    {
        "q": "In the lowest common ancestor function, what is returned when both "
             "the left and right recursive calls return a non-null node?",
        "expected": "The current node (root) is returned as the LCA.",
        "gold_source": "Trees_Study_Guide.pdf",
        "gold_section": "Lowest Common Ancestor",
    },
    {
        # Designed failure: the guides describe Dijkstra's min-heap idea but
        # never state its time complexity, so a correct, grounded system must
        # refuse. This surfaces a corpus-coverage limitation.
        "q": "What is the time complexity of Dijkstra's algorithm when "
             "implemented with a min-heap?",
        "expected": "Not stated in the documents -> the system should say it "
                    "doesn't have enough information (true answer is "
                    "O((V+E) log V)).",
        "gold_source": "Graphs_Study_Guide.pdf",
        "gold_section": "Dijkstra",
    },
]

MODES = ["hybrid", "semantic", "bm25"]


def _hit(results, gold_source, gold_section) -> bool:
    for r in results:
        m = r.metadata
        if m.get("source") == gold_source and gold_section.lower() in (
            m.get("section_path", "").lower()
        ):
            return True
    return False


def run(use_llm: bool = True) -> str:
    retriever = get_retriever()
    lines = ["# Evaluation Results\n"]

    # --- retrieval-mode comparison -------------------------------------
    lines.append("## Retrieval mode comparison (gold section in top-3?)\n")
    lines.append("| # | Question | hybrid | semantic | bm25 |")
    lines.append("|---|----------|--------|----------|------|")
    mode_hits = {m: 0 for m in MODES}
    for i, t in enumerate(TEST_QUESTIONS, 1):
        cells = []
        for mode in MODES:
            res = retriever.search(t["q"], mode=mode)
            hit = _hit(res, t["gold_source"], t["gold_section"])
            mode_hits[mode] += hit
            cells.append("hit" if hit else "miss")
        lines.append(f"| {i} | {t['q']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("**Gold-section hit rate:** " + ", ".join(
        f"{m}={mode_hits[m]}/{len(TEST_QUESTIONS)}" for m in MODES) + "\n")

    # --- full end-to-end run (hybrid) ----------------------------------
    lines.append("## End-to-end results (hybrid mode)\n")
    for i, t in enumerate(TEST_QUESTIONS, 1):
        res = retriever.search(t["q"], mode="hybrid")
        retrieved = [f"{r.metadata['source']} :: {r.metadata['section_path']}"
                     for r in res]
        hit = _hit(res, t["gold_source"], t["gold_section"])
        answer = ask(t["q"])["answer"] if use_llm else "(LLM skipped)"

        lines.append(f"### Q{i}. {t['q']}")
        lines.append(f"- **Expected:** {t['expected']}")
        lines.append(f"- **Retrieved chunks:**")
        for rc in retrieved:
            lines.append(f"    - {rc}")
        lines.append(f"- **Retrieval quality:** "
                     f"{'Relevant (gold section retrieved)' if hit else 'Off-target (gold section missed)'}")
        lines.append(f"- **System response:** {answer}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    report = run(use_llm=not args.no_llm)
    out = Path(__file__).resolve().parents[1] / "eval_results.md"
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
