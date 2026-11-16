"""Run the real OpenAI agent on a question split and write traces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agent_loop import run_agent
from .config import get_settings
from .data import is_hit, load_questions, load_strategy_chunks
from .openai_controller import OpenAIController
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RetrieverAdapter


def build_retriever(name: str, settings, chunks):
    dense = DenseRetriever(str(settings.chroma_path), settings.chroma_collection, settings.embedding_model)
    bm25 = BM25Retriever(chunks)
    if name == "dense":
        return RetrieverAdapter(dense)
    if name == "bm25":
        return RetrieverAdapter(bm25)
    if name == "hybrid_rrf":
        return RetrieverAdapter(HybridRetriever(dense, bm25, candidate_k=10))
    raise ValueError(f"Unknown retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy agent thật trên benchmark")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--retriever", default="hybrid_rrf", choices=("dense", "bm25", "hybrid_rrf"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu (0 = tất cả)")
    parser.add_argument("--ids", nargs="*", default=None, help="Chỉ chạy các question id")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_strategy_chunks(settings.chunks_json, settings.strategy)
    questions = [q for q in load_questions(settings.questions_json) if q["split"] == args.split]
    if args.ids:
        wanted = set(args.ids)
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    controller = OpenAIController(
        settings.openai_api_key,
        settings.openai_model,
        base_url=settings.openai_base_url,
    )
    print(f"Model={settings.openai_model} base_url={settings.openai_base_url or 'default'}")
    retriever = build_retriever(args.retriever, settings, chunks)

    out_dir = args.output or (
        settings.root / "starter" / "results" / f"agent-{args.retriever}-{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']} ...", flush=True)
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "answerable": bool(q.get("answerable", True)),
            "question": q["question"],
        })
        started = time.perf_counter()
        try:
            result = run_agent(
                q["question"],
                controller,
                retriever,
                filters=q.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            status = result["status"]
        except Exception as exc:  # noqa: BLE001 - keep error traces for report
            result = {
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [{"event": "error", "message": str(exc)}],
                "usage": {},
            }
            status = "error"
            print(f"  ERROR: {exc}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        context_ids: list[str] = []
        for ev in result.get("events") or []:
            if ev.get("event") == "search":
                context_ids = list(ev.get("context_ids") or [])

        evidence_hit = None
        if q.get("answerable", True) and q.get("gold"):
            # Did any retrieved chunk in final context hit gold?
            hit_ids = set()
            for ev in result.get("events") or []:
                if ev.get("event") != "search":
                    continue
                for r in ev.get("results") or []:
                    if is_hit(r, q["gold"]):
                        hit_ids.add(r["id"])
            evidence_hit = bool(hit_ids & set(context_ids))

        row = {
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": bool(q.get("answerable", True)),
            "retriever": args.retriever,
            "status": status,
            "answer": result.get("answer"),
            "citations": result.get("citations"),
            "events": result.get("events"),
            "usage": result.get("usage"),
            "elapsed_ms": elapsed_ms,
            "context_ids": context_ids,
            "context_evidence_hit": evidence_hit,
            "model": settings.openai_model,
        }
        traces.append(row)
        print(f"  -> {status} ({elapsed_ms:.0f} ms)")

    summary = {
        "split": args.split,
        "retriever": args.retriever,
        "model": settings.openai_model,
        "n": len(traces),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstain": sum(1 for t in traces if t["status"] == "abstain"),
        "error": sum(1 for t in traces if t["status"] == "error"),
        "tool_use_rate": (
            sum(1 for t in traces if any(e.get("event") == "search" for e in (t.get("events") or [])))
            / max(1, sum(1 for t in traces if t["status"] != "error"))
        ),
        "avg_elapsed_ms": round(sum(t["elapsed_ms"] for t in traces) / max(1, len(traces)), 2),
        "total_tokens": sum(int((t.get("usage") or {}).get("total_tokens") or 0) for t in traces),
    }

    (out_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
