#!/usr/bin/env python3
"""
Interactive RAG over an Obsidian vault.

Usage:
    python examples/obsidian_rag.py --vault /path/to/vault [options]

Options:
    --vault     Path to your Obsidian vault root (required)
    --folders   Comma-separated folders to ingest, e.g. "Meetings,Essays"
                Defaults to: Meetings,Essays,Daily Notes
    --llm       LLM provider: openai | anthropic | google  (default: openai)
    --model     LLM model name  (default: provider default)
    --embedder  Embedder: openai | google | anthropic      (default: openai)
    --store     Path for persistent ChromaDB storage       (default: ./.ragmax_vault_db)
    --top-k     Chunks to retrieve per query               (default: 8)
    --rerank    Enable cross-encoder reranking             (default: off)
    --hybrid    Enable hybrid (BM25 + dense) search        (default: off)
    --force     Force re-ingestion even if DB already has data

Example:
    # Quick test on meetings + essays, using OpenAI
    python examples/obsidian_rag.py \\
        --vault "/path/to/vault" \\
        --folders "Meetings,Essays" \\
        --llm openai

    # Full setup with Anthropic, hybrid search, reranking
    python examples/obsidian_rag.py \\
        --vault "/path/to/vault" \\
        --llm anthropic \\
        --embedder openai \\
        --hybrid --rerank
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# ── Ensure ragmax is importable from the repo root ────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── ANSI colours ──────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    MAGENTA= "\033[35m"
    RED    = "\033[31m"

def bold(s: str) -> str: return f"{C.BOLD}{s}{C.RESET}"
def dim(s: str)  -> str: return f"{C.DIM}{s}{C.RESET}"
def cyan(s: str) -> str: return f"{C.CYAN}{s}{C.RESET}"
def green(s: str)-> str: return f"{C.GREEN}{s}{C.RESET}"
def yellow(s: str)->str: return f"{C.YELLOW}{s}{C.RESET}"
def blue(s: str) -> str: return f"{C.BLUE}{s}{C.RESET}"
def red(s: str)  -> str: return f"{C.RED}{s}{C.RESET}"


# ── Helpers ───────────────────────────────────────────────────

def banner() -> None:
    print(f"""
{bold(cyan('╔══════════════════════════════════╗'))}
{bold(cyan('║'))}  {bold('ragmax')} × Obsidian Vault RAG  {bold(cyan('║'))}
{bold(cyan('╚══════════════════════════════════╝'))}
""")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vault",    required=True, help="Path to Obsidian vault root")
    p.add_argument("--folders",  default="Meetings,Essays,Daily Notes",
                   help="Comma-separated folders to ingest")
    p.add_argument("--llm",      default="openai", choices=["openai", "anthropic", "google"],
                   help="LLM provider (default: openai)")
    p.add_argument("--model",    default=None,
                   help=(
                       "Model override. Defaults: "
                       "openai→gpt-4o-mini, "
                       "anthropic→claude-sonnet-4-20250514 (also: claude-haiku-4-20250514, claude-opus-4-20250514), "
                       "google→gemini-2.5-flash (also: gemini-2.5-pro)"
                   ))
    p.add_argument("--embedder", default="google", choices=["openai", "google", "anthropic", "sentence_transformers"])
    p.add_argument("--store",    default=".ragmax_vault_db")
    p.add_argument("--top-k",   default=8, type=int)
    p.add_argument("--rerank",  action="store_true")
    p.add_argument("--hybrid",  action="store_true")
    p.add_argument("--force",   action="store_true", help="Re-ingest even if DB already has data")
    return p.parse_args()


def collect_markdown_files(vault: str, folders: list[str]) -> list[str]:
    """Find all .md files in the specified vault folders."""
    files: list[str] = []
    for folder in folders:
        folder_path = os.path.join(vault, folder)
        if not os.path.isdir(folder_path):
            print(yellow(f"  ⚠  Folder not found, skipping: {folder}"))
            continue
        for root, _, fnames in os.walk(folder_path):
            for fname in sorted(fnames):
                if fname.endswith(".md") and not fname.startswith("."):
                    files.append(os.path.join(root, fname))
    return files


def format_source(path: str, vault: str, meta: dict) -> str:
    """Format a chunk source for display."""
    rel = os.path.relpath(path, vault) if path.startswith(vault) else path
    parts = [cyan(rel)]

    date = meta.get("date")
    if date:
        parts.append(dim(f"({date})"))

    participants = meta.get("participants")
    if participants and isinstance(participants, list):
        people = ", ".join(str(p) for p in participants[:3])
        if len(participants) > 3:
            people += f" +{len(participants)-3}"
        parts.append(dim(f"[{people}]"))

    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()
    banner()

    vault = os.path.expanduser(args.vault)
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]

    print(f"{bold('Vault:')}    {vault}")
    print(f"{bold('Folders:')}  {', '.join(folders)}")
    print(f"{bold('LLM:')}      {args.llm}")
    print(f"{bold('Embedder:')} {args.embedder}")
    print(f"{bold('Hybrid:')}   {'yes' if args.hybrid else 'no'}")
    print(f"{bold('Rerank:')}   {'yes' if args.rerank else 'no'}")
    print()

    # ── Build the pipeline ────────────────────────────────────
    from ragmax.chunking.factory import create_chunker
    from ragmax.core.config import ChunkerConfig, EmbedderConfig, LLMConfig, RerankerConfig
    from ragmax.embeddings.factory import create_embedder
    from ragmax.generation.factory import create_llm
    from ragmax.parsers.obsidian import ObsidianParser
    from ragmax.pipeline import RAGPipeline
    from ragmax.reranking.factory import create_reranker
    from ragmax.retrieval.hybrid import HybridRetriever, BM25Index
    from ragmax.retrieval.dense import DenseRetriever
    from ragmax.stores.chroma import ChromaStore

    print(f"{'─'*42}")
    print(bold("Setting up components…"))

    # Embedder
    emb_model = {
        "openai": "text-embedding-3-small",
        "google": "gemini-embedding-2-preview",
        "anthropic": "voyage-3",
        "sentence_transformers": "all-MiniLM-L6-v2",
    }[args.embedder]
    emb_dims = {"google": 768}.get(args.embedder)  # Matryoshka for Gemini

    embedder = create_embedder(EmbedderConfig(
        provider=args.embedder,
        model=emb_model,
        dimensions=emb_dims,
    ))
    print(f"  ✓ Embedder: {args.embedder} / {emb_model}"
          + (f" (dim={emb_dims})" if emb_dims else ""))

    # LLM
    llm_defaults = {
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
        "google":    "gemini-2.5-flash",
    }
    llm_model = args.model or llm_defaults[args.llm]
    llm = create_llm(LLMConfig(provider=args.llm, model=llm_model))
    print(f"  ✓ LLM: {args.llm} / {llm_model}")

    # Store
    store = ChromaStore(collection="obsidian_vault", path=args.store)
    existing = await store.count()
    print(f"  ✓ ChromaDB at {args.store!r} ({existing:,} existing chunks)")

    # Chunker  — recursive is great for markdown, 600 chars ≈ 150 tokens
    chunker = create_chunker(ChunkerConfig(strategy="recursive", chunk_size=600, chunk_overlap=80))
    print(f"  ✓ Chunker: recursive (size=600, overlap=80)")

    # BM25 index for hybrid
    bm25 = BM25Index()

    # Retriever
    if args.hybrid:
        retriever = HybridRetriever(embedder=embedder, store=store, bm25=bm25, alpha=0.7)
        print(f"  ✓ Retriever: hybrid (dense α=0.7 + BM25)")
    else:
        retriever = DenseRetriever(embedder=embedder, store=store)
        print(f"  ✓ Retriever: dense")

    # Reranker
    if args.rerank:
        reranker = create_reranker(RerankerConfig(
            provider="cross_encoder",
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=5,
        ))
        print(f"  ✓ Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    else:
        reranker = None

    pipeline = RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        chunker=chunker,
        retriever=retriever,
        reranker=reranker,
        retrieval_top_k=args.top_k,
        rerank_top_k=5,
        system_prompt=(
            "You are a personal knowledge assistant with access to Obsidian vault notes. "
            "Answer questions based strictly on the retrieved notes. "
            "When referencing specific meetings or documents, mention who was involved and when. "
            "If information is not in the notes, say so clearly."
        ),
    )

    # ── Ingestion ─────────────────────────────────────────────
    print()
    if existing > 0 and not args.force:
        print(f"{bold('Ingestion:')} skipping — {existing:,} chunks already in DB")
        print(dim("  (use --force to re-ingest)"))
    else:
        files = collect_markdown_files(vault, folders)
        print(f"{bold('Ingestion:')} found {len(files):,} markdown files")

        if not files:
            print(red("  No files found. Check --vault path and --folders."))
            sys.exit(1)

        parser = ObsidianParser()
        total_chunks = 0
        skipped = 0

        t0 = time.time()
        for i, filepath in enumerate(files, 1):
            try:
                doc = parser.parse(filepath)
                # Skip mostly-empty docs (templates, stubs)
                if len(doc.content.strip()) < 50:
                    skipped += 1
                    continue

                # Add to BM25 index
                chunks = chunker.chunk(doc)
                if chunks:
                    bm25.add(chunks)
                    # Batch embed + store
                    texts = [c.content for c in chunks]
                    embeddings = await embedder.embed(texts)
                    await store.upsert(chunks, embeddings)
                    total_chunks += len(chunks)

            except Exception as e:
                print(yellow(f"  ⚠  {os.path.basename(filepath)}: {e}"))

            # Progress every 50 files
            if i % 50 == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(files) - i) / rate if rate > 0 else 0
                bar_width = 30
                filled = int(bar_width * i / len(files))
                bar = "█" * filled + "░" * (bar_width - filled)
                print(
                    f"\r  [{bar}] {i:>4}/{len(files)} files  "
                    f"{total_chunks:,} chunks  "
                    f"ETA {eta:.0f}s   ",
                    end="",
                    flush=True,
                )

        elapsed = time.time() - t0
        print(f"\n  ✓ Ingested {total_chunks:,} chunks from "
              f"{len(files) - skipped:,} notes ({skipped} skipped) "
              f"in {elapsed:.1f}s")

    # ── Interactive REPL ──────────────────────────────────────
    print()
    print(f"{'─'*42}")
    print(bold("Ready! Start asking questions about your vault."))
    print(dim("  Commands: 'chunks' to toggle chunk display, 'q' to quit"))
    print(f"{'─'*42}")
    print()

    show_chunks = False

    while True:
        try:
            query = input(f"{bold(green('You'))} › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("q", "quit", "exit", "bye"):
            print("Bye!")
            break

        if query.lower() == "chunks":
            show_chunks = not show_chunks
            print(dim(f"  Chunk display: {'ON' if show_chunks else 'OFF'}"))
            continue

        t0 = time.time()
        try:
            print(f"{dim('Thinking…')}", end="\r")
            result = await pipeline.query(query, trace=True)
            elapsed = time.time() - t0

            # ── Print answer
            print(f"\n{bold(blue('ragmax'))} › {result.answer}\n")

            # ── Source attribution
            if result.chunks_used:
                print(dim(f"  ─── Sources ({len(result.chunks_used)} chunks, {elapsed:.2f}s) ───"))
                seen_sources: dict[str, int] = {}
                for chunk in result.chunks_used:
                    src = chunk.source or ""
                    seen_sources[src] = seen_sources.get(src, 0) + 1

                for src, count in seen_sources.items():
                    # Find chunk metadata for this source
                    chunk_meta = next(
                        (c.metadata for c in result.chunks_used if c.source == src), {}
                    )
                    label = format_source(src, vault, chunk_meta)
                    print(f"  {dim('·')} {label}  {dim(f'({count} chunk{\"s\" if count>1 else \"\"})')}")

                # Optionally show chunk content
                if show_chunks:
                    print()
                    for i, chunk in enumerate(result.chunks_used, 1):
                        print(dim(f"  ── Chunk {i} ──"))
                        preview = chunk.content[:300].replace("\n", " ")
                        if len(chunk.content) > 300:
                            preview += "…"
                        print(dim(f"  {preview}"))

                # Trace timing if available
                if result.trace:
                    spans = {s.name: s.duration_ms for s in result.trace.spans if s.duration_ms}
                    timing = "  ".join(f"{k}: {v:.0f}ms" for k, v in spans.items())
                    if timing:
                        print(dim(f"\n  ⏱ {timing}"))

            print()

        except Exception as e:
            print(red(f"\n  Error: {e}\n"))


if __name__ == "__main__":
    asyncio.run(main())
