# RAG Support Chatbot — Local RAG Prototype

A local Retrieval-Augmented Generation chatbot for customer support, greenlit for production on Azure AI Foundry.

---

## What This Is

A B2B logistics provider was running a SaaS-based customer support chatbot costing $200/month for 25,000 messages. The bot handled routine queries — parcel tracking, duty payments, ID verification — but required manual KB maintenance and gave no visibility into conversation analytics. The brief was to build a local alternative and prove it could match or exceed quality at a fraction of the cost.

This prototype replaces the SaaS chatbot with a fully local RAG pipeline. It loads a structured knowledge base of markdown files, builds per-topic FAISS vector indexes combined with BM25 keyword retrieval, passes candidates through a cross-encoder reranker, and generates answers with Claude Haiku via the Anthropic API. The Streamlit UI adds tool-calling (live parcel tracking, human escalation), per-turn sentiment detection, CSAT star ratings, and structured JSON logging for all management KPIs. An evaluation harness measures retrieval accuracy and answer quality against 33 labelled test questions.

The prototype was demoed to technical management and approved for a production build on Azure AI Foundry. The production version moves the inference workload inside the organisation's Azure tenant to meet data residency requirements, with Claude Haiku replaced by Azure-hosted GPT-4o mini for most queries and GPT-4o for the more complex ones.

---

## Results

| Metric | Result |
|---|---|
| Retrieval accuracy (Hit@5) | **87.9%** (29/33 questions) |
| Mean Reciprocal Rank | **0.81** |
| Response time (streaming) | **~2.5 seconds** to first token |
| Cost per session | **~$0.000154** (Claude Haiku + prompt caching) |
| Cost saving vs SaaS baseline | **99%** ($1.60/month vs $200/month) |

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval  (local, zero marginal cost)              │
│                                                             │
│  BM25 (keyword)  ──┐                                        │
│                    ├──► EnsembleRetriever (RRF fusion) ──►  │
│  FAISS (semantic) ─┘    40% BM25 / 60% FAISS               │
│        one index per topic, BGE-small-en-v1.5 embeddings    │
│                                                             │
│  top-10 candidates ──► CrossEncoder reranker ──► top-5      │
│                        ms-marco-MiniLM-L-6-v2               │
└─────────────────────────────────────────────────────────────┘
     │  5 ranked chunks
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude Haiku API  (pay-per-token, ~$0.000154/session)      │
│  Temperature 0.0 · Max 1024 tokens · Streaming enabled      │
│  Tool calling: parcel tracking + human escalation           │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
  Streamed answer + structured log entry
```

**Why hybrid retrieval?** Pure semantic search mis-ranks queries containing proper nouns — country codes (KSA, UAE), currency codes (SAR, JOD), carrier names (Aramex, DHL), and brand names (ASOS). BM25 handles exact token matches for these cases; FAISS handles paraphrases and intent variants. Reciprocal rank fusion merges both ranked lists into a single deduplicated candidate set weighted 40/60.

**Why a cross-encoder reranker?** Bi-encoder embeddings (used by FAISS) score query and document independently — fast but less accurate for nuanced relevance. The cross-encoder sees both query and document together, yielding a more accurate relevance score at the cost of extra CPU time. Adding the reranker lifted Hit@5 by approximately 8 percentage points in internal testing.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` (HuggingFace) | 384-dim, fast CPU inference, strong on retrieval tasks |
| Vector store | FAISS (CPU) | Local, no server, persists to disk, fast similarity search |
| Keyword retrieval | BM25 (rank-bm25) | Handles exact tokens, proper nouns, currency codes |
| Retrieval fusion | LangChain `EnsembleRetriever` | Reciprocal rank fusion, configurable weights |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~85MB, strong ranking accuracy, runs on CPU |
| LLM | Claude Haiku 4.5 (Anthropic) | Cheapest capable model; deterministic at temp=0.0 |
| LLM framework | LangChain | Chain composition, prompt templates, message history |
| Memory | `ChatMessageHistory` + sliding window | 10-turn context window, in-process, no DB needed |
| UI | Streamlit | Fast to build, interactive, streaming-compatible |
| Logging | JSON Lines (`turns.jsonl`, `sessions.jsonl`) | Append-only, queryable, supports all management KPIs |
| Evaluation | Custom harness (Hit@5, MRR, LLM-as-judge) | Measures retrieval and answer quality independently |

---

## Project Structure

```
RAG-main/
├── app.py                    # Streamlit chat UI — the demo interface
├── requirements.txt          # Loose dependency spec
├── requirements.lock         # Pinned versions for reproducibility
├── .env                      # API keys — gitignored, never committed
│
├── src/
│   ├── config.py             # Central constants (models, weights, paths)
│   ├── loader.py             # KB markdown → LangChain Documents
│   ├── embedder.py           # Documents → FAISS indexes (build once)
│   ├── llm.py                # Claude Haiku singleton (streaming + non-streaming)
│   ├── prompts.py            # System prompts v1 (no memory) and v2 (with memory)
│   ├── retriever.py          # BM25 + FAISS ensemble + CrossEncoder reranker
│   ├── chain.py              # RAG chain assembly (single-topic + full hybrid)
│   ├── logger.py             # Structured analytics logging (turns + sessions)
│   ├── tools.py              # LangChain tools: tracking lookup + escalation
│   └── memory.py             # Conversation history with sliding window
│
├── eval/
│   ├── questions.json        # 33 test questions with expected item_ids
│   ├── run_eval.py           # Hit@5 + MRR retrieval evaluation
│   ├── judge_prompts.py      # LLM-as-judge prompts (answer quality scoring)
│   ├── compare.py            # Management KPI dashboard from sessions.jsonl
│   ├── cost_report.py        # Cost comparison vs SaaS baseline
│   └── results/              # Eval output CSVs — gitignored (generated)
│
├── diagrams/                 # Architecture diagrams (.drawio source files)
│
├── kb/                       # Knowledge base markdown files — gitignored (business content)
├── indexes/                  # FAISS indexes built from KB — gitignored (generated)
└── logs/                     # Conversation logs — gitignored (may contain query data)
```

---

## Key Design Decisions

- **KB chunked at `## Item_NNN` boundaries, not character count.** Each KB file is authored with explicit item headings. Splitting at these boundaries preserves semantic coherence — each chunk is a complete answer unit, not an arbitrary fragment mid-sentence.

- **Separate FAISS index per topic, merged at query time.** Building one index per topic (wismo, how_to_pay, id_verification, etc.) prevents lower-relevance topics from diluting the embedding space of high-relevance ones. The merger happens in-memory at startup; once built, the retriever is a process-level singleton.

- **BM25 weight 0.4 / FAISS weight 0.6.** Tested weights from 0.2/0.8 to 0.5/0.5. The 40/60 split was the sweet spot: enough BM25 signal to surface exact token matches (country codes, carrier names) without overwhelming semantic ranking on intent-based queries.

- **`K_INITIAL_RERANK=10`, `K_FINAL=5`.** The cross-encoder is the latency bottleneck. Reducing from 20 to 10 candidates halved reranker time (~1.2s saving) with no measurable Hit@5 regression — the top-10 from BM25+FAISS fusion already contained the correct item in 87.9% of cases.

- **`temperature=0.0`.** Customer support answers must be deterministic and grounded. Hallucination risk rises with temperature. At 0.0, the same question gets the same answer on every run, which is also important for the eval harness to be reproducible.

- **Singleton pattern for embedder, retriever, and LLM.** Streamlit reruns the entire script on every user interaction. Without singletons, the 130MB embedding model and 85MB reranker would be reloaded on every message. Module-level `_instance` globals ensure each model loads once per process lifetime.

---

## Evaluation Framework

The eval harness in `eval/` measures two things independently: whether the *right chunk was retrieved*, and whether the *answer is correct*.

**Retrieval eval** (`eval/run_eval.py`): 33 questions spanning all 9 knowledge topics. Each question has a labelled expected `item_id`. The retriever runs each query, and two metrics are computed:
- **Hit@5**: did the correct item appear anywhere in the top 5 retrieved chunks?
- **MRR**: mean reciprocal rank of the first correct hit (penalises correct-but-ranked-low results)

**Per-topic breakdown:**

| Topic | Questions | Hit@5 | Notes |
|---|---|---|---|
| WISMO (shipment tracking) | 5 | 5/5 (100%) | — |
| Duty thresholds by country | 4 | 4/4 (100%) | — |
| How to pay duties | 4 | 4/4 (100%) | — |
| Order held pending payment | 4 | 3/4 (75%) | HELD-002: topic bleed with refusing_payment |
| Refusing payment / returns | 4 | 4/4 (100%) | — |
| ID verification | 3 | 3/3 (100%) | — |
| What are duties and taxes | 3 | 3/3 (100%) | — |
| Common / contact info | 3 | 1/3 (33%) | COM-001, COM-003: generic queries, weak topic signal |
| Damaged goods | 3 | 2/3 (67%) | DAM-001: "parcel" token pulls toward WISMO |
| **Total** | **33** | **29/33 (87.9%)** | MRR = 0.81 |

**4 failures documented:**
1. **HELD-002** — "How many days to pay before parcel is returned?" — `refusing_payment` items discuss return timelines more explicitly; the correct `order_held_pending_payment/Item_004` was outranked.
2. **COM-001** — "Who decides how much duty I pay?" — The `common` topic is a small catch-all; its items overlap semantically with `what_are_duties_and_taxes` and `refusing_payment`.
3. **COM-003** — "I have a problem, I need to speak to someone at APG" — Generic intent with no topic signal; all retrievers matched domain content over contact info.
4. **DAM-001** — "Everything inside is completely smashed" — BM25 matched "parcel" strongly to WISMO content; `damaged_goods` is a small topic with limited vocabulary overlap.

**Answer quality eval** (`eval/judge_prompts.py`): LLM-as-judge scoring each answer on three dimensions (factual accuracy, groundedness in retrieved context, appropriate escalation) on a 0–2 scale each (0–6 total).

---

## What's Not In This Repo

| Excluded | Reason |
|---|---|
| `kb/` — knowledge base markdown files | Business content, not for public distribution |
| `indexes/` — FAISS vector indexes | Generated from KB; rebuild with `python -m src.embedder` |
| `logs/` — conversation logs | May contain customer query data |
| `.env` — environment file | Contains API keys |

**To recreate from scratch:** bring your own KB markdown files (each with `## Item_NNN: Title` headings), run `python -m src.embedder` to build the indexes, add your `ANTHROPIC_API_KEY` to `.env`, then `streamlit run app.py`.

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/omarovski-27/RAG.git
cd RAG

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# 5. Add KB markdown files to kb/
# Each file must contain ## Item_NNN: Title headings

# 6. Build FAISS indexes
python -m src.embedder

# 7. Run the app
streamlit run app.py
```

---

## Production Path

This prototype was the basis for a production build being implemented on Azure AI Foundry inside the organisation's Azure tenant. The production version addresses data residency and security requirements by moving all inference inside the Azure boundary, replacing the Anthropic API with an Azure-hosted model endpoint, and connecting to the live carrier API for real parcel tracking. The evaluation framework, logging schema, and KB structure from this prototype are carried forward unchanged.