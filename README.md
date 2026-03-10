# ragmax

**State-of-the-art Retrieval-Augmented Generation for Python.**

Modular, async-first, production-grade RAG library covering the full pipeline — from document ingestion to evaluated answer generation. Built on top of the techniques and patterns described in *Mastering RAG (2026)*.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **8 chunking strategies** — character, recursive, sentence, semantic, context-enriched (Anthropic's approach), late chunking (Jina AI), agentic, and multi-vector
- **4 embedding providers** — OpenAI, Google Gemini Embedding 2 (multimodal + Matryoshka), Anthropic/Voyage, and local SentenceTransformers
- **2 vector stores** — ChromaDB and Qdrant with HNSW, filtering, and hybrid search support
- **4 retrieval strategies** — dense, hybrid (BM25 + dense with RRF), multi-query, and HyDE
- **3 reranking backends** — cross-encoder, Cohere Rerank, and LLM-based listwise reranking
- **3 LLM providers** — OpenAI, Anthropic Claude, and Google Gemini
- **4 guardrails** — PII detection/redaction, prompt injection detection, hallucination checking, and relevance validation
- **Query processing** — query rewriting (conversational context resolution) and query decomposition
- **Agentic RAG** — iterative retrieval-reasoning agent with multi-hop support
- **Evaluation suite** — recall@k, NDCG@k, MRR, precision@k, chunk attribution, chunk utilization, and context adherence
- **Full observability** — built-in tracing with nested spans and timing
- **Protocol-based** — every component is a `Protocol`, making it trivial to swap implementations
- **Async-first** — built on `async/await` for high-throughput production workloads

---

## Quick Start

### Installation

```bash
pip install ragmax

# Install with your preferred providers
pip install ragmax[openai,chroma]           # OpenAI + ChromaDB
pip install ragmax[anthropic,qdrant]        # Anthropic + Qdrant
pip install ragmax[google,chroma]           # Gemini + ChromaDB
pip install ragmax[all]                     # Everything
```

### One-liner API

```python
from ragmax import RAG

rag = RAG.from_config(llm="openai", store="chroma")

# Ingest documents
await rag.ingest("./docs/")

# Query
result = await rag.query("What is our refund policy?")
print(result.answer)
```

### Builder API

For more control, use the fluent builder:

```python
from ragmax import RAGBuilder

rag = (RAGBuilder()
    .with_llm("anthropic", model="claude-sonnet-4-6")
    .with_embedder("google", model="gemini-embedding-2-preview", dimensions=768)
    .with_store("qdrant", url="localhost:6333", collection="docs")
    .with_chunker("semantic", threshold=0.5, max_sentences=15)
    .with_reranker("cross_encoder", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    .with_input_guardrails(["pii", "injection"])
    .with_output_guardrails(["hallucination"])
    .with_hybrid_search(alpha=0.7)
    .build())

await rag.ingest(["./contracts/", "./policies/", "./faq.md"])

result = await rag.query("Can I get a refund after 30 days?")
print(result.answer)
print(f"Used {len(result.chunks_used)} chunks")
```

### Streaming

```python
async for token in rag.query_stream("Summarize our privacy policy"):
    print(token, end="", flush=True)
```

---

## Architecture

```
Query → Input Guardrails → Query Rewriting → Retrieval → Reranking → Generation → Output Guardrails → Result
                                                  ↑
                              Documents → Parsing → Chunking → Embedding → Vector Store
```

ragmax implements the full enterprise RAG architecture with pluggable components at every stage.

### Package Structure

```
src/ragmax/
├── core/           # Protocols, models, config, exceptions, tracing
├── parsers/        # Text, PDF, DOCX, HTML + auto-selecting registry
├── chunking/       # 8 strategies: character → agentic
├── embeddings/     # OpenAI, Gemini Embedding 2, Voyage, SentenceTransformers
├── stores/         # ChromaDB, Qdrant
├── retrieval/      # Dense, hybrid, multi-query, HyDE
├── reranking/      # Cross-encoder, Cohere, LLM-based
├── generation/     # OpenAI, Anthropic, Google + prompt templates
├── guardrails/     # PII, injection, hallucination, relevance
├── query/          # Query rewriting, decomposition
├── agentic/        # Iterative retrieval-reasoning agent
├── evaluation/     # Retrieval & generation metrics
└── pipeline.py     # RAGPipeline, RAG, RAGBuilder
```

---

## Components

### Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `character` | Fixed-size character splits | Simple, fast baseline |
| `recursive` | Recursive splitting by separators (`\n\n`, `\n`, `. `, ` `) | General-purpose (default) |
| `sentence` | Sentence-boundary aware grouping | Preserving sentence integrity |
| `semantic` | Embedding similarity breakpoints | Semantically coherent chunks |
| `context_enriched` | LLM-generated context prepended to each chunk | Highest retrieval accuracy |
| `late` | Embed full document, then chunk with pooled embeddings | Preserving long-range context |
| `agentic` | LLM determines optimal boundaries by topic | Best quality, highest cost |

```python
from ragmax.chunking import create_chunker

# Semantic chunking with custom threshold
chunker = create_chunker(strategy="semantic", extra={"threshold": 0.5})
chunks = chunker.chunk(document)
```

### Embedding Providers

| Provider | Models | Matryoshka | Multimodal |
|----------|--------|------------|------------|
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large` | Yes | No |
| Google | `gemini-embedding-2-preview`, `gemini-embedding-001` | Yes (3072/1536/768) | Yes (text, images, video, audio) |
| Anthropic/Voyage | `voyage-3`, `voyage-3-large`, `voyage-3-lite` | No | No |
| SentenceTransformers | `all-MiniLM-L6-v2` + any HF model | Varies | No |

```python
from ragmax.embeddings import create_embedder

# Gemini Embedding 2 with Matryoshka dimensionality reduction
embedder = create_embedder(
    provider="google",
    model="gemini-embedding-2-preview",
    dimensions=768,  # Reduced from 3072 for efficiency
)
```

### Retrieval Strategies

| Strategy | Description | Addresses Failure Mode |
|----------|-------------|----------------------|
| `dense` | Standard embed-and-search | Baseline retrieval |
| `hybrid` | Dense + BM25 with Reciprocal Rank Fusion | Missed Top-Ranked Docs |
| `multi_query` | LLM generates query variants for broader recall | Missed Top-Ranked Docs |
| `hyde` | Embed a hypothetical answer instead of the query | Query-document mismatch |

```python
# Enable hybrid search via the builder
rag = (RAGBuilder()
    .with_hybrid_search(alpha=0.7)  # 70% dense, 30% BM25
    .with_llm("openai")
    .with_store("chroma")
    .build())
```

### Rerankers

| Reranker | Latency | Quality | Cost |
|----------|---------|---------|------|
| `cross_encoder` | 200-400ms | High | Free (local) |
| `cohere` | 100-200ms | Very High | API cost |
| `llm` | 4-6s | Highest | API cost |
| `none` | 0ms | N/A | Free |

### Guardrails

**Input guardrails** (run before retrieval):
- `pii` — Detect and optionally redact emails, phone numbers, SSNs, credit cards
- `injection` — Detect prompt injection patterns

**Output guardrails** (run after generation):
- `hallucination` — Check if the response is grounded in retrieved chunks
- `relevance` — Verify retrieved chunks are relevant to the query

```python
rag = (RAGBuilder()
    .with_input_guardrails(["pii", "injection"])
    .with_output_guardrails(["hallucination", "relevance"])
    .with_llm("anthropic")
    .with_store("chroma")
    .build())
```

---

## Agentic RAG

For complex multi-hop questions, use the agentic RAG mode where the LLM iteratively decides when to search, refine, or answer:

```python
from ragmax.agentic import RAGAgent

agent = RAGAgent(
    retriever=retriever,
    llm=llm,
    max_steps=5,
    top_k=5,
)

result = await agent.query(
    "Compare the performance characteristics of HNSW vs IVF indexing "
    "and recommend which to use for a 10M document collection"
)
print(result.answer)
print(f"Completed in {result.metadata['agent_steps']} retrieval steps")
```

---

## Evaluation

Evaluate your RAG pipeline against ground-truth datasets using metrics from the 8 failure modes:

```python
from ragmax.evaluation import RAGEvaluator, EvalSample

evaluator = RAGEvaluator(k=10)

samples = [
    EvalSample(
        query="What is the refund policy?",
        relevant_ids=["chunk_42", "chunk_43"],
    ),
]

# After running each query through your pipeline:
result = await evaluator.evaluate_sample(
    retrieved_ids=["chunk_42", "chunk_10", "chunk_43"],
    response="Our refund policy allows returns within 30 days...",
    chunks=["Policy text chunk 1...", "Policy text chunk 2..."],
    sample=samples[0],
)

print(result.metrics)
# {'recall@k': 1.0, 'precision@k': 0.67, 'ndcg@k': 0.86, 'mrr': 1.0,
#  'chunk_attribution': 1.0, 'chunk_utilization': 0.45, 'context_adherence': 0.82}
```

### Available Metrics

| Metric | Category | Failure Mode Addressed |
|--------|----------|----------------------|
| Recall@k | Retrieval | Missing Content, Missed Top-Ranked |
| Precision@k | Retrieval | Not in Context |
| NDCG@k | Retrieval | Ranking quality |
| MRR | Retrieval | First relevant result position |
| Chunk Attribution | Generation | Not Extracted |
| Chunk Utilization | Generation | Chunk targeting quality |
| Context Adherence | Generation | Wrong Format, Incorrect Specificity |

---

## Extending ragmax

Every component is defined as a `Protocol`, so you can plug in custom implementations:

```python
from ragmax.core.protocols import Embedder

class MyCustomEmbedder:
    """Any class matching the Embedder protocol works."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Your custom embedding logic
        ...

    @property
    def dimension(self) -> int:
        return 768

# Use it directly in the pipeline
from ragmax import RAGPipeline

pipeline = RAGPipeline(
    embedder=MyCustomEmbedder(),
    store=my_store,
    llm=my_llm,
    chunker=my_chunker,
)
```

---

## Optional Dependencies

ragmax uses optional dependency groups to keep the base install lightweight:

| Extra | Packages | When You Need It |
|-------|----------|-----------------|
| `openai` | openai | OpenAI embeddings & LLM |
| `anthropic` | anthropic | Claude LLM |
| `google` | google-genai | Gemini embeddings & LLM |
| `qdrant` | qdrant-client | Qdrant vector store |
| `chroma` | chromadb | ChromaDB vector store |
| `rerankers` | sentence-transformers | Cross-encoder reranking & local embeddings |
| `cohere` | cohere | Cohere Rerank API |
| `parsers` | pdfplumber, python-docx, beautifulsoup4 | PDF, DOCX, HTML parsing |
| `bm25` | rank-bm25 | Hybrid search (BM25 component) |
| `dev` | pytest, pytest-asyncio, ruff | Development & testing |
| `all` | Everything above | Full installation |

---

## License

MIT
