"""Benchmark dense / BM25 / hybrid retrieval on a split (default: dev)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import get_settings
from .data import is_hit, load_questions, load_strategy_chunks, pack_context
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever


def evaluate_method(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int,
    budget: int,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        if not q.get("answerable", True):
            continue
        start = time.perf_counter()
        filters = q.get("filter")
        # Keep unfiltered ranking for fair Hit@k; also store filtered results if filter exists.
        results = retriever.search(q["question"], filters=None, top_k=max(top_k, 3))
        filtered = None
        if filters:
            filtered = retriever.search(q["question"], filters=filters, top_k=max(top_k, 3))
        used = filtered if filtered is not None else results
        context, ids = pack_context(used, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        top3 = results[:3]
        marked = [{**c, "hit": is_hit(c, q["gold"])} for c in top3]
        used_marked = [{**c, "hit": is_hit(c, q["gold"])} for c in used[:top_k]]
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": name,
            "top_k": top_k,
            "budget_chars": budget,
            "filter": filters,
            "gold": q["gold"],
            "results": marked,
            "filtered_results": used_marked if filters else None,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in marked),
            "context_hit": any(c["hit"] and c["id"] in ids for c in used_marked),
        })
    return traces


def summarize(traces: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(traces)
    hit1 = sum(1 for t in traces if t["hit1"])
    hit3 = sum(1 for t in traces if t["hit3"])
    ctx = sum(1 for t in traces if t["context_hit"])
    avg_ms = sum(t["elapsed_ms"] for t in traces) / n if n else 0.0
    return {
        "strategy": name,
        "n_answerable": n,
        "hit1": f"{hit1}/{n}",
        "hit1_rate": hit1 / n if n else 0.0,
        "hit3": f"{hit3}/{n}",
        "hit3_rate": hit3 / n if n else 0.0,
        "context_evidence": f"{ctx}/{n}",
        "context_evidence_rate": ctx / n if n else 0.0,
        "avg_elapsed_ms": round(avg_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh 3 phương pháp retrieval")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_strategy_chunks(settings.chunks_json, settings.strategy)
    questions = [q for q in load_questions(settings.questions_json) if q["split"] == args.split]

    dense = DenseRetriever(str(settings.chroma_path), settings.chroma_collection, settings.embedding_model)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25, candidate_k=max(10, args.top_k))

    methods = [
        ("dense", dense),
        ("bm25", bm25),
        ("hybrid_rrf", hybrid),
    ]

    out_dir = args.output or (settings.root / "starter" / "results" / f"retrieval-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_traces: dict[str, list] = {}
    for name, retriever in methods:
        print(f"Running {name} on {args.split}...")
        traces = evaluate_method(name, retriever, questions, top_k=args.top_k, budget=args.budget)
        summary = summarize(traces, name)
        summaries.append(summary)
        all_traces[name] = traces
        (out_dir / f"traces_{name}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(summary)

    payload = {
        "split": args.split,
        "strategy_chunks": settings.strategy,
        "collection": settings.chroma_collection,
        "embedding_model": settings.embedding_model,
        "top_k": args.top_k,
        "budget": args.budget,
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
