# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Data Structures and Algorithms

Well these aren't hard to find knowledge base, but a popular topic for technical interview prep. 

---

## Documents


| #   | Source  | Description                                                          | URL or location                                         |
| --- | ------- | -------------------------------------------------------------------- | ------------------------------------------------------- |
| 1   | Copilot | advanced dynamic programming of definitions, examples, and code.     | documents/Advanced_Dynamic_Programming_Study_Guide.pdf  |
| 2   | Copilot | arrays of definitions, examples, and code.                           | documents/Arrays_Study_Guide.pdf                        |
| 3   | Copilot | backtracking of definitions, examples, and code.                     | documents/Backtracking_Study_Guide.pdf                  |
| 4   | Copilot | dynamic programming of definitions, examples, and code.              | documents/Dynamic_Programming_Study_Guide.pdf           |
| 5   | Copilot | graphs of definitions, examples, and code.                           | documents/Graphs_Study_Guide.pdf                        |
| 6   | Copilot | greedy algorithms of definitions, examples, and code.                | documents/Greedy_Algorithms_Study_Guide.pdf             |
| 7   | Copilot | queues of definitions, examples, and code.                           | documents/Queues_Study_Guide.pdf                        |
| 8   | Copilot | stacks of definitions, examples, and code.                           | documents/Stacks_Study_Guide.pdf                        |
| 9   | Copilot | trees of definitions, examples, and code.                            | documents/Trees_Study_Guide.pdf                         |
| 10  | Copilot | two pointers and sliding windows of definitions, examples, and code. | documents/Two_Pointers_&_Sliding_Window_Study_Guide.pdf |


---

## Chunking Strategy

Parent-child (Recursion)

**Chunk size (child): ≤220 tokens** — *revised down from 512, see note below*

**Overlap: ~40 tokens**

**Parent chunk: whole section (≤1200 tokens, not embedded)**

**Reasoning: Documents contains compact but dense blocks of code and math, you want to ensure that a single chunk contains the entire concept. There are subsections roughly 60-100 words in length.** Parent-child allows headers to be the metadata of the paragraphs that'll be indexed and chunked. Children (small, precise) are what we embed and search; their **parent section** is what we hand to the LLM so it sees the full concept + code.

> **Update during implementation (required by spec):** The original plan said
> 512-token chunks. Measurement showed `all-MiniLM-L6-v2` truncates at **256
> tokens**, so a 512-token chunk would be silently cut before embedding. Child
> chunks were therefore capped at ~220 tokens (headroom under 256) so the whole
> child is embedded; parents stay large because they are only read by the LLM.
> Full rationale + measurements in `report.md` §2.

---

## Retrieval Approach

**Embedding model: sentence-transformers (all-MiniLM-L6-v2)**

**Top-k: 3** (parent sections; pulled from 20 fused child candidates)

**Retrieval mode: hybrid — BM25 keyword + semantic, fused with Reciprocal Rank Fusion**

**Semantic cutoff:** cosine similarity ≥ **0.55** (applied only to semantic-only mode)

**Mode behavior:** `semantic` uses only semantic hits above the cutoff, `bm25` uses
only keyword hits, and `hybrid` combines both (no fallback between single modes).

**Production tradeoff reflection:** `all-MiniLM-L6-v2` is fast, free, and local,
but it caps at 256 tokens and is English-only. If cost weren't a constraint I'd
weigh a longer-context / higher-accuracy model (e.g. `bge-large` or an API model)
against added latency and cost, plus multilingual coverage if the audience needed
it. For this English, short-section corpus, MiniLM is the right default.

---

## Evaluation Plan


| #   | Question                                                                                         | Expected answer                                                                    |
| --- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 1   | Which shortest-path algorithm handles graphs with negative edge weights, and what can it detect? | Bellman-Ford; used when edges may be negative, detects negative cycles.            |
| 2   | In the greedy interval scheduling pattern, what should you sort intervals by?                    | Sort by end time (finish earliest), then choose an interval if it doesn't overlap. |
| 3   | What are the four key components of the universal backtracking template?                         | N/A                                                                                |
| 4   | In the LCA function, what is returned when both left and right recursive calls return a node?    | The current node (root).                                                           |
| 5   | What is the time complexity of Dijkstra's algorithm with a min-heap?                             | N/A                                                                                |


---

## Anticipated Challenges

1. **Broken word spacing on extraction** — pdfplumber's default `x_tolerance`
  glues words together on these PDFs (`Arraysarethefoundation…`). Mitigated with
   `x_tolerance=1`.
2. **Embedder truncation** — sections can exceed the 256-token embedder window,
  silently dropping text. Mitigated by sizing children under the window
   (parent-child split).
3. **Header detection misses** — headings starting with a digit (`3.1 1D DP`)
  broke the first regex and merged sections. Caught via chunk inspection and fixed.
4. **Keyword-exact queries** — semantic search can miss queries that hinge on a
  literal phrase ("four key components"). Mitigated with hybrid BM25 + semantic.
5. **Corpus coverage gaps** — the guides omit some facts (e.g. Big-O of Dijkstra);
  the system must refuse rather than hallucinate (see eval Q5).

---

## Stretch Features (added after the core build — spec updated before each)

- **Hybrid search** — BM25 + semantic fused with RRF; compared against
semantic-only in `eval_results.md` (hybrid 5/5 vs semantic 4/5).
- **Metadata filtering** — filter retrieval by document/topic; exposed in the UI.
- **Conversational memory** — multi-turn; follow-ups are rewritten to standalone
queries before retrieval.
- **Chunking/extraction comparison** — pdfplumber vs spacy-layout, benchmarked in
`report.md` §1.

---

## Architecture

```
                 ┌──────────────┐
   documents/*.pdf │  Ingestion   │  pdfplumber (x_tolerance=1) + clean
                 │ extractors.py│  [alt: spacy-layout]
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Chunking    │  spaCy sentences → parent (section)
                 │ chunking.py  │  + child (≤220 tok) + metadata
                 └──────┬───────┘
                        ▼
              ┌──────────────────┐
              │ Embed + Store    │  all-MiniLM-L6-v2 → ChromaDB
              │ vectorstore.py   │  (children embedded; parents on disk)
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Retrieval       │  BM25 + semantic → RRF fusion
              │ retrieval.py     │  metadata filter → expand child→parent
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Generation      │  Groq llama-3.3-70b (grounded) +
              │ rag.py           │  conversational memory + citations
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Interface       │  Gradio web UI (app.py)
              └──────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:** Prompt the AI with this Documents +
Chunking sections and the diagram to implement `extractors.py` (pdfplumber +
spacy-layout returning labelled blocks) and `chunking.py` (parent-child split on
spaCy sentence boundaries with header metadata). Verify by inspecting sample chunks.

**Milestone 4 — Embedding and retrieval:** Give the AI the Retrieval Approach
section to implement `vectorstore.py` (ChromaDB + MiniLM) and `retrieval.py`
(hybrid BM25 + semantic with RRF and metadata filtering). Validate gold-section
hit rate per mode.

**Milestone 5 — Generation and interface:** Provide the grounding requirement and
output format to implement `rag.py` (Groq, programmatic source attribution,
conversational memory) and `app.py` (Gradio). Verify grounding holds on an
out-of-scope query.