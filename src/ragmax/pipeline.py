"""Core RAG pipeline orchestrator and high-level APIs.

Provides three layers of abstraction:

1. **RAGPipeline** — full control, explicit component wiring
2. **RAG** — one-liner setup via ``RAG.from_config(...)``
3. **RAGBuilder** — fluent builder API for intermediate customisation
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from ragmax.core.config import (
    ChunkerConfig,
    EmbedderConfig,
    GuardrailsConfig,
    LLMConfig,
    RAGConfig,
    RerankerConfig,
    RetrievalConfig,
    StoreConfig,
)
from ragmax.core.exceptions import GuardrailTriggered, RagmaxError
from ragmax.core.models import (
    Chunk,
    Document,
    GenerationResult,
    QueryContext,
    SearchResult,
)
from ragmax.core.protocols import (
    Chunker,
    Embedder,
    InputGuardrail,
    LLMProvider,
    OutputGuardrail,
    Reranker,
    Retriever,
    VectorStore,
)
from ragmax.core.tracing import TracingContext
from ragmax.generation.prompts import build_rag_messages

logger = logging.getLogger("ragmax")


class RAGPipeline:
    """Full-control RAG pipeline with explicit component wiring.

    Orchestrates the complete flow:
    query → guardrails → rewrite → retrieve → rerank → generate → guardrails → result
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMProvider,
        chunker: Chunker,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        input_guardrails: list[InputGuardrail] | None = None,
        output_guardrails: list[OutputGuardrail] | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunker = chunker
        self._retriever = retriever
        self.reranker = reranker
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    @property
    def retriever(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        from ragmax.retrieval.dense import DenseRetriever

        return DenseRetriever(self.embedder, self.store)

    # ── Ingestion ──────────────────────────────────────────────

    async def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and store a list of documents. Returns chunk count."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Batch embed
        from ragmax.core.utils import batched

        total_stored = 0
        for batch in batched(all_chunks, 64):
            texts = [c.content for c in batch]
            embeddings = await self.embedder.embed(texts)
            await self.store.upsert(batch, embeddings)
            total_stored += len(batch)

        logger.info("Ingested %d chunks from %d documents", total_stored, len(documents))
        return total_stored

    async def ingest(self, source: str | list[str]) -> int:
        """Ingest files or directories. Returns total chunks stored."""
        from ragmax.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        paths = source if isinstance(source, list) else [source]

        documents: list[Document] = []
        for path in paths:
            if os.path.isdir(path):
                documents.extend(registry.parse_directory(path))
            elif os.path.isfile(path):
                documents.append(registry.parse(path))
            else:
                logger.warning("Path not found: %s", path)

        return await self.ingest_documents(documents)

    # ── Query ──────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
        trace: bool = True,
    ) -> GenerationResult:
        """Run the full RAG pipeline and return a GenerationResult."""
        tracer = TracingContext(enabled=trace)
        context = QueryContext(
            query=question,
            original_query=question,
            conversation_history=conversation_history or [],
            filters=filters,
        )

        # 1. Input guardrails
        with tracer.span("input_guardrails"):
            for guard in self.input_guardrails:
                result = await guard.check(context.query, context)
                if not result.passed:
                    raise GuardrailTriggered(result.guardrail_name, result.message)
                if result.modified_text:
                    context.query = result.modified_text

        # 2. Retrieve
        with tracer.span("retrieve", top_k=self.retrieval_top_k):
            results = await self.retriever.retrieve(
                context.query, top_k=self.retrieval_top_k, filters=filters
            )

        # 3. Rerank
        if self.reranker and results:
            with tracer.span("rerank", top_k=self.rerank_top_k):
                reranked = await self.reranker.rerank(
                    context.query, results, top_k=self.rerank_top_k
                )
                chunks = [r.chunk for r in reranked]
        else:
            chunks = [r.chunk for r in results[: self.rerank_top_k]]

        # 4. Generate
        with tracer.span("generate"):
            context_texts = [c.content for c in chunks]
            messages = build_rag_messages(
                query=context.query,
                context_chunks=context_texts,
                system_prompt=self.system_prompt,
                user_template=self.user_template,
            )
            answer = await self.llm.generate(messages)

        # 5. Output guardrails
        with tracer.span("output_guardrails"):
            for guard in self.output_guardrails:
                result = await guard.check(answer, context, chunks)
                if not result.passed:
                    raise GuardrailTriggered(result.guardrail_name, result.message)

        return GenerationResult(
            answer=answer,
            chunks_used=chunks,
            metadata={"query": context.query, "original_query": context.original_query},
            trace=tracer.build() if trace else None,
        )

    async def query_stream(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the answer token by token."""
        context = QueryContext(
            query=question,
            original_query=question,
            conversation_history=conversation_history or [],
            filters=filters,
        )

        # Input guardrails
        for guard in self.input_guardrails:
            result = await guard.check(context.query, context)
            if not result.passed:
                raise GuardrailTriggered(result.guardrail_name, result.message)
            if result.modified_text:
                context.query = result.modified_text

        # Retrieve + rerank
        results = await self.retriever.retrieve(
            context.query, top_k=self.retrieval_top_k, filters=filters
        )
        if self.reranker and results:
            reranked = await self.reranker.rerank(
                context.query, results, top_k=self.rerank_top_k
            )
            chunks = [r.chunk for r in reranked]
        else:
            chunks = [r.chunk for r in results[: self.rerank_top_k]]

        # Stream generation
        context_texts = [c.content for c in chunks]
        messages = build_rag_messages(
            query=context.query,
            context_chunks=context_texts,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
        )
        async for token in self.llm.generate_stream(messages):
            yield token


# ── Convenience APIs ───────────────────────────────────────────


_PROVIDER_SHORTCUTS = {
    "openai": ("openai", "gpt-4o-mini"),
    "anthropic": ("anthropic", "claude-sonnet-4-20250514"),
    "google": ("google", "gemini-2.5-flash"),
}

_STORE_SHORTCUTS = {
    "chroma": "chroma",
    "qdrant": "qdrant",
}


class RAG:
    """Minimal one-liner RAG interface.

    Usage::

        rag = RAG.from_config(llm="openai", store="chroma")
        await rag.ingest("./docs/")
        result = await rag.query("What is our refund policy?")
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_config(
        cls,
        llm: str = "openai",
        store: str = "chroma",
        embedder: str = "openai",
        chunker: str = "recursive",
        reranker: str = "none",
        **kwargs: Any,
    ) -> RAG:
        """Create a fully configured RAG instance from simple string identifiers."""
        from ragmax.chunking.factory import create_chunker
        from ragmax.embeddings.factory import create_embedder
        from ragmax.generation.factory import create_llm
        from ragmax.reranking.factory import create_reranker
        from ragmax.stores.factory import create_store

        # Parse LLM shortcut: "openai:gpt-4o" or just "openai"
        llm_provider, llm_model = _parse_shortcut(llm, _PROVIDER_SHORTCUTS)
        emb_provider, emb_model = _parse_shortcut(embedder, _PROVIDER_SHORTCUTS, is_embedder=True)

        llm_instance = create_llm(LLMConfig(provider=llm_provider, model=llm_model))
        embedder_instance = create_embedder(EmbedderConfig(provider=emb_provider, model=emb_model))
        store_instance = create_store(StoreConfig(provider=store, **kwargs))
        chunker_instance = create_chunker(ChunkerConfig(strategy=chunker))
        reranker_instance = create_reranker(RerankerConfig(provider=reranker), llm=llm_instance)

        pipeline = RAGPipeline(
            embedder=embedder_instance,
            store=store_instance,
            llm=llm_instance,
            chunker=chunker_instance,
            reranker=reranker_instance,
        )
        return cls(pipeline)

    async def ingest(self, source: str | list[str]) -> int:
        return await self._pipeline.ingest(source)

    async def ingest_documents(self, documents: list[Document]) -> int:
        return await self._pipeline.ingest_documents(documents)

    async def query(self, question: str, **kwargs: Any) -> GenerationResult:
        return await self._pipeline.query(question, **kwargs)

    async def query_stream(self, question: str, **kwargs: Any) -> AsyncIterator[str]:
        async for token in self._pipeline.query_stream(question, **kwargs):
            yield token


class RAGBuilder:
    """Fluent builder API for intermediate customisation.

    Usage::

        rag = (RAGBuilder()
            .with_llm("anthropic", model="claude-sonnet-4-20250514")
            .with_embedder("openai", model="text-embedding-3-small", dimensions=512)
            .with_store("qdrant", url="localhost:6333", collection="docs")
            .with_chunker("semantic", threshold=0.5)
            .with_reranker("cross_encoder")
            .with_input_guardrails(["pii", "injection"])
            .with_output_guardrails(["hallucination"])
            .with_hybrid_search(alpha=0.7)
            .build())
    """

    def __init__(self) -> None:
        self._config = RAGConfig()
        self._system_prompt: str | None = None
        self._user_template: str | None = None
        self._retrieval_top_k: int = 10
        self._rerank_top_k: int = 5

    def with_llm(self, provider: str, model: str | None = None, **kwargs: Any) -> RAGBuilder:
        prov, default_model = _parse_shortcut(provider, _PROVIDER_SHORTCUTS)
        self._config.llm = LLMConfig(
            provider=prov,
            model=model or default_model,
            extra=kwargs,
        )
        return self

    def with_embedder(
        self, provider: str, model: str | None = None, dimensions: int | None = None, **kwargs: Any
    ) -> RAGBuilder:
        prov, default_model = _parse_shortcut(provider, _PROVIDER_SHORTCUTS, is_embedder=True)
        self._config.embedder = EmbedderConfig(
            provider=prov,
            model=model or default_model,
            dimensions=dimensions,
            extra=kwargs,
        )
        return self

    def with_store(self, provider: str, **kwargs: Any) -> RAGBuilder:
        self._config.store = StoreConfig(provider=provider, **kwargs)
        return self

    def with_chunker(self, strategy: str, **kwargs: Any) -> RAGBuilder:
        self._config.chunker = ChunkerConfig(strategy=strategy, extra=kwargs)
        return self

    def with_reranker(self, provider: str, model: str | None = None, **kwargs: Any) -> RAGBuilder:
        self._config.reranker = RerankerConfig(provider=provider, model=model, extra=kwargs)
        return self

    def with_input_guardrails(self, names: list[str]) -> RAGBuilder:
        self._config.guardrails.input_guardrails = names
        return self

    def with_output_guardrails(self, names: list[str]) -> RAGBuilder:
        self._config.guardrails.output_guardrails = names
        return self

    def with_hybrid_search(self, alpha: float = 0.7) -> RAGBuilder:
        self._config.retrieval.hybrid = True
        self._config.retrieval.hybrid_alpha = alpha
        return self

    def with_multi_query(self) -> RAGBuilder:
        self._config.retrieval.multi_query = True
        return self

    def with_hyde(self) -> RAGBuilder:
        self._config.retrieval.hyde = True
        return self

    def with_system_prompt(self, prompt: str) -> RAGBuilder:
        self._system_prompt = prompt
        return self

    def with_user_template(self, template: str) -> RAGBuilder:
        self._user_template = template
        return self

    def with_top_k(self, retrieval: int = 10, rerank: int = 5) -> RAGBuilder:
        self._retrieval_top_k = retrieval
        self._rerank_top_k = rerank
        return self

    def build(self) -> RAG:
        """Build and return a configured RAG instance."""
        from ragmax.chunking.factory import create_chunker
        from ragmax.embeddings.factory import create_embedder
        from ragmax.generation.factory import create_llm
        from ragmax.guardrails.factory import create_input_guardrails, create_output_guardrails
        from ragmax.reranking.factory import create_reranker
        from ragmax.retrieval.factory import create_retriever
        from ragmax.stores.factory import create_store

        llm = create_llm(self._config.llm)
        embedder = create_embedder(self._config.embedder)
        store = create_store(self._config.store)
        chunker = create_chunker(self._config.chunker)
        reranker = create_reranker(self._config.reranker, llm=llm)
        retriever = create_retriever(self._config.retrieval, embedder, store, llm)

        input_guards = create_input_guardrails(self._config.guardrails.input_guardrails)
        output_guards = create_output_guardrails(self._config.guardrails.output_guardrails)

        pipeline = RAGPipeline(
            embedder=embedder,
            store=store,
            llm=llm,
            chunker=chunker,
            retriever=retriever,
            reranker=reranker,
            input_guardrails=input_guards,
            output_guardrails=output_guards,
            system_prompt=self._system_prompt,
            user_template=self._user_template,
            retrieval_top_k=self._retrieval_top_k,
            rerank_top_k=self._rerank_top_k,
        )
        return RAG(pipeline)


def _parse_shortcut(
    value: str,
    shortcuts: dict[str, tuple[str, str]],
    is_embedder: bool = False,
) -> tuple[str, str]:
    """Parse 'provider:model' or just 'provider'."""
    if ":" in value:
        provider, model = value.split(":", 1)
        return provider, model

    if value in shortcuts:
        provider, model = shortcuts[value]
        if is_embedder:
            # Use embedding-specific defaults
            emb_defaults = {
                "openai": "text-embedding-3-small",
                "google": "gemini-embedding-2-preview",
                "anthropic": "voyage-3",
            }
            return provider, emb_defaults.get(provider, model)
        return provider, model

    return value, value
