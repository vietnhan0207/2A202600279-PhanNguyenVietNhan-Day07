"""
test_phase2.py — Chunking Strategy Evaluation
==============================================
Loads the VU_HT02 regulation document, chunks it with every available strategy,
indexes each set of chunks into a separate EmbeddingStore, then runs the group's
5 benchmark queries (agreed in REPORT.md §6) and reports chunk statistics +
retrieval quality for every strategy.

Benchmark queries (group-agreed):
  Q1  Thời gian tối thiểu thực hiện luận văn thạc sĩ là bao lâu?
      Gold: Ít nhất 06 tháng
  Q2  Điểm luận văn bao nhiêu thì được xếp loại đạt?
      Gold: Lớn hơn hoặc bằng 5,5 điểm
  Q3  Học viên thi hộ hoặc nhờ thi hộ bị xử lý kỷ luật thế nào?
      Gold: Đình chỉ 1 năm lần đầu, buộc thôi học lần 2
  Q4  Số tín chỉ được công nhận và chuyển đổi tối đa là bao nhiêu?
      Gold: Không vượt quá 30 tín chỉ
  Q5  Hội đồng đánh giá luận văn cần có ít nhất bao nhiêu thành viên?
      Gold: Ít nhất 05 thành viên

Usage
-----
    python test_phase2.py                 # MockEmbedder (no API key needed)
    python test_phase2.py --local         # sentence-transformers (local GPU/CPU)
    python test_phase2.py --openai        # OpenAI text-embedding-3-small
    python test_phase2.py --top-k 5       # change retrieval top-k (default 3)
    python test_phase2.py --chunk-size 600  # override chunk_size for size-based chunkers
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.chunking import (
    FixedSizeChunker,
    ParagraphMergingChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from src.embeddings import MockEmbedder, LocalEmbedder, OpenAIEmbedder
from src.models import Document
from src.store import EmbeddingStore

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunking strategy evaluation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Use LocalEmbedder (sentence-transformers)")
    group.add_argument("--openai", action="store_true", help="Use OpenAIEmbedder")
    parser.add_argument("--top-k", type=int, default=3, metavar="K", help="Retrieval top-k (default 3)")
    parser.add_argument("--chunk-size", type=int, default=800, metavar="N", help="chunk_size for size-based chunkers (default 800)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Benchmark queries — group-agreed (REPORT.md §6)
# keywords are used for keyword-hit scoring: a result counts as a hit when
# at least one retrieved chunk contains at least one keyword from the list.
# ---------------------------------------------------------------------------

QUERIES: list[dict] = [
    {
        # Q1 — Gold answer: ít nhất 06 tháng
        "question": "Thời gian tối thiểu thực hiện luận văn thạc sĩ là bao lâu?",
        "keywords": ["06 tháng", "6 tháng", "tối thiểu", "thực hiện luận văn"],
        "gold": "Ít nhất 06 tháng",
    },
    {
        # Q2 — Gold answer: lớn hơn hoặc bằng 5,5 điểm
        "question": "Điểm luận văn bao nhiêu thì được xếp loại đạt?",
        "keywords": ["5,5", "5.5", "xếp loại đạt", "điểm đạt"],
        "gold": "Lớn hơn hoặc bằng 5,5 điểm",
    },
    {
        # Q3 — Gold answer: đình chỉ 1 năm lần đầu, buộc thôi học lần 2
        "question": "Học viên thi hộ hoặc nhờ thi hộ bị xử lý kỷ luật thế nào?",
        "keywords": ["thi hộ", "đình chỉ", "buộc thôi học", "kỷ luật"],
        "gold": "Đình chỉ 1 năm lần đầu, buộc thôi học lần 2",
    },
    {
        # Q4 — Gold answer: không vượt quá 30 tín chỉ
        "question": "Số tín chỉ được công nhận và chuyển đổi tối đa là bao nhiêu?",
        "keywords": ["30 tín chỉ", "công nhận", "chuyển đổi", "không vượt quá"],
        "gold": "Không vượt quá 30 tín chỉ",
    },
    {
        # Q5 — Gold answer: ít nhất 05 thành viên
        "question": "Hội đồng đánh giá luận văn cần có ít nhất bao nhiêu thành viên?",
        "keywords": ["05 thành viên", "5 thành viên", "hội đồng", "thành viên"],
        "gold": "Ít nhất 05 thành viên",
    },
]

# ---------------------------------------------------------------------------
# Chunk statistics helpers
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?…]\s*$")


def chunk_stats(chunks: list[str]) -> dict:
    """Compute descriptive statistics for a list of chunks."""
    if not chunks:
        return {"n": 0, "avg": 0.0, "min": 0, "max": 0, "std": 0.0, "boundary_pct": 0.0}

    lengths = [len(c) for c in chunks]
    n = len(lengths)
    avg = statistics.mean(lengths)
    std = statistics.stdev(lengths) if n > 1 else 0.0
    boundary_count = sum(1 for c in chunks if _SENTENCE_END.search(c))
    return {
        "n": n,
        "avg": avg,
        "min": min(lengths),
        "max": max(lengths),
        "std": std,
        "boundary_pct": boundary_count / n * 100,
    }

# ---------------------------------------------------------------------------
# Retrieval evaluation helpers
# ---------------------------------------------------------------------------

def keyword_hit(results: list[dict], keywords: list[str]) -> bool:
    """Return True if any top-k result contains at least one expected keyword."""
    for r in results:
        content_lower = r["content"].lower()
        if any(kw.lower() in content_lower for kw in keywords):
            return True
    return False


def evaluate_retrieval(store: EmbeddingStore, queries: list[dict], top_k: int) -> dict:
    """Run all queries and return aggregate retrieval metrics."""
    top1_scores: list[float] = []
    avg_topk_scores: list[float] = []
    hits: list[bool] = []
    query_details: list[dict] = []

    for q in queries:
        t0 = time.perf_counter()
        results = store.search(q["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        top1 = results[0]["score"] if results else 0.0
        avg_score = statistics.mean(r["score"] for r in results) if results else 0.0
        hit = keyword_hit(results, q["keywords"])

        top1_scores.append(top1)
        avg_topk_scores.append(avg_score)
        hits.append(hit)

        query_details.append({
            "question": q["question"],
            "top1_score": top1,
            "avg_score": avg_score,
            "hit": hit,
            "latency_ms": latency_ms,
            "top1_content": results[0]["content"][:120] if results else "",
        })

    return {
        "avg_top1_score": statistics.mean(top1_scores),
        "avg_topk_score": statistics.mean(avg_topk_scores),
        "keyword_hit_rate": sum(hits) / len(hits) * 100,
        "details": query_details,
    }

# ---------------------------------------------------------------------------
# Build store from chunks
# ---------------------------------------------------------------------------

def build_store(chunks: list[str], embedder, collection_name: str) -> tuple[EmbeddingStore, float]:
    """Embed and index chunks; return (store, index_time_ms)."""
    docs = [Document(id=f"{collection_name}_{i}", content=c, metadata={}) for i, c in enumerate(chunks)]
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    t0 = time.perf_counter()
    store.add_documents(docs)
    index_ms = (time.perf_counter() - t0) * 1000
    return store, index_ms

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

COL_W = 22

def _hr(widths: list[int], char: str = "─", cross: str = "┼") -> str:
    return "├" + cross.join(char * w for w in widths) + "┤"


def _row(cells: list[str], widths: list[int]) -> str:
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(cell.ljust(w)[:w])
    return "│" + "│".join(parts) + "│"


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2
              for i, h in enumerate(headers)]
    total = sum(widths) + len(widths) + 1

    print(f"\n{'━' * total}")
    print(f" {title}")
    print(f"{'━' * total}")
    print("┌" + "┬".join("─" * w for w in widths) + "┐")
    print(_row([f" {h}" for h in headers], widths))
    print("├" + "┼".join("─" * w for w in widths) + "┤")
    for r in rows:
        print(_row([f" {c}" for c in r], widths))
    print("└" + "┴".join("─" * w for w in widths) + "┘")


def fmt(val, fmt_str: str = ".2f") -> str:
    if isinstance(val, float):
        return format(val, fmt_str)
    return str(val)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    chunk_size = args.chunk_size
    top_k = args.top_k

    # -- Embedder selection ---------------------------------------------------
    print("\n[1/4] Selecting embedder...")
    if args.openai:
        embedder = OpenAIEmbedder()
        embedder_name = f"OpenAI ({embedder.model_name})"
    elif args.local:
        embedder = LocalEmbedder()
        embedder_name = f"Local ({embedder.model_name})"
    else:
        embedder = MockEmbedder()
        embedder_name = "Mock (deterministic hash)"

    print(f"      Embedder : {embedder_name}")
    print(f"      top-k    : {top_k}")
    print(f"      chunk-size parameter: {chunk_size}")

    # -- Load document --------------------------------------------------------
    print("\n[2/4] Loading document...")
    candidates = [
        Path("VU_HT02.VN_Quy-che-dao-tao-Trinh-do-Thac-sy_20.12.2022.md"),
        Path("data/VU_HT02.VN_Quy-che-dao-tao-Trinh-do-Thac-sy_20.12.2022.md"),
    ]
    data_file = next((p for p in candidates if p.exists()), None)
    if data_file is None:
        sys.exit("ERROR: VU_HT02 markdown file not found. Run from the project root.")

    text = data_file.read_text(encoding="utf-8")
    print(f"      File     : {data_file}")
    print(f"      Size     : {len(text):,} chars")

    # -- Define strategies ----------------------------------------------------
    strategies: list[tuple[str, object]] = [
        (
            "FixedSize",
            FixedSizeChunker(chunk_size=chunk_size, overlap=80),
        ),
        (
            "Sentence",
            SentenceChunker(max_sentences_per_chunk=5),
        ),
        (
            "Recursive",
            RecursiveChunker(
                separators=["\n**Điều", "\n\n", "\n", ". "],
                chunk_size=chunk_size,
            ),
        ),
        (
            "ParagraphMerging",
            ParagraphMergingChunker(target_size=chunk_size, overlap_sentences=1),
        ),
        (
            "Semantic",
            SemanticChunker(
                embed_fn=embedder,
                breakpoint_threshold=0.80,
                min_chunk_size=150,
            ),
        ),
    ]

    # -- Chunk + index + evaluate --------------------------------------------
    print(f"\n[3/4] Chunking, indexing, and evaluating {len(strategies)} strategies...")

    results: list[dict] = []
    per_query_details: dict[str, list[dict]] = {}

    for name, chunker in strategies:
        print(f"\n  → {name}", end="", flush=True)

        # Chunk
        t0 = time.perf_counter()
        chunks = chunker.chunk(text)
        chunk_ms = (time.perf_counter() - t0) * 1000
        print(f"  ({len(chunks)} chunks)", end="", flush=True)

        # Index
        store, index_ms = build_store(chunks, embedder, collection_name=f"eval_{name.lower()}")
        print(f"  indexed in {index_ms:.0f}ms", end="", flush=True)

        # Evaluate retrieval
        ret = evaluate_retrieval(store, QUERIES, top_k)
        per_query_details[name] = ret["details"]
        print(f"  hit={ret['keyword_hit_rate']:.0f}%")

        stats = chunk_stats(chunks)
        results.append({
            "name": name,
            **stats,
            "chunk_ms": chunk_ms,
            "index_ms": index_ms,
            **{f"ret_{k}": v for k, v in ret.items() if k != "details"},
        })

    # -- Print summary tables ------------------------------------------------
    print("\n[4/4] Results\n")

    # Table 1: Chunk statistics
    print_table(
        title="CHUNK STATISTICS",
        headers=["Strategy", "Count", "Avg len", "Min", "Max", "Std Dev", "Boundary%", "Chunk ms"],
        rows=[
            [
                r["name"],
                str(r["n"]),
                fmt(r["avg"], ".0f"),
                str(r["min"]),
                str(r["max"]),
                fmt(r["std"], ".0f"),
                fmt(r["boundary_pct"], ".1f") + "%",
                fmt(r["chunk_ms"], ".1f"),
            ]
            for r in results
        ],
    )

    # Table 2: Retrieval quality
    print_table(
        title=f"RETRIEVAL QUALITY  (top-k={top_k}, embedder={embedder_name})",
        headers=["Strategy", "Avg Top-1 score", f"Avg Top-{top_k} score", "Keyword hit%", "Index ms"],
        rows=[
            [
                r["name"],
                fmt(r["ret_avg_top1_score"], ".4f"),
                fmt(r["ret_avg_topk_score"], ".4f"),
                fmt(r["ret_keyword_hit_rate"], ".1f") + "%",
                fmt(r["index_ms"], ".0f"),
            ]
            for r in results
        ],
    )

    # Table 3: Per-query breakdown (top-1 score per strategy)
    q_labels = [f"Q{i+1}" for i in range(len(QUERIES))]
    per_q_rows = []
    for r in results:
        details = per_query_details[r["name"]]
        row = [r["name"]] + [fmt(d["top1_score"], ".3f") + ("✓" if d["hit"] else "✗") for d in details]
        per_q_rows.append(row)

    print_table(
        title="PER-QUERY TOP-1 SCORE  (score + ✓/✗ keyword hit)",
        headers=["Strategy"] + q_labels,
        rows=per_q_rows,
    )

    # Print query legend with gold answers
    print("\n  Query legend (group benchmark — REPORT.md §6):")
    for i, q in enumerate(QUERIES):
        print(f"    Q{i+1}: {q['question']}")
        print(f"         Gold: {q['gold']}")

    # -- Per-strategy top-1 content preview (all 5 benchmark queries) --------
    for qi, q in enumerate(QUERIES):
        print("\n" + "━" * 72)
        print(f" TOP-1 RETRIEVED CONTENT  Q{qi+1}: {q['question']}")
        print(f" Gold answer: {q['gold']}")
        print("━" * 72)
        for name, _ in strategies:
            detail = per_query_details[name][qi]
            print(f"\n  [{name}]  score={detail['top1_score']:.4f}  hit={'✓' if detail['hit'] else '✗'}")
            print(f"  {detail['top1_content'].strip()!r}")

    # -- Recommendations ------------------------------------------------------
    print("\n" + "━" * 72)
    print(" STRATEGY NOTES")
    print("━" * 72)
    notes = {
        "FixedSize":        "Fast, uniform chunks. Poor semantic coherence — cuts mid-sentence/idea.",
        "Sentence":         "Grammatically clean boundaries. Ignores topic shifts; chunk size uneven.",
        "Recursive":        "Respects structure (Điều > paragraph > sentence). Good baseline for legal docs.",
        "ParagraphMerging": "Preserves paragraph units. Most uniform non-trivial size. ✓ Recommended for structured text.",
        "Semantic":         "Topic-aware boundaries. Best retrieval coherence with a real embedder. ✓ Best for RAG.",
    }
    for name, note in notes.items():
        print(f"  {name:<18} {note}")

    print()


if __name__ == "__main__":
    main()
