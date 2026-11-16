"""Load Day 6 chunks/questions into Day 7-friendly records."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = text.replace("đ", "d").replace("Ð", "d")
    return re.sub(r"\s+", " ", text).strip()


def load_strategy_chunks(path: Path, strategy: str = "A") -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get(strategy) or raw.get(strategy.upper()) or raw.get(strategy.lower())
        if not isinstance(items, list):
            raise ValueError(f"local_chunks.json thiếu strategy {strategy}")
    elif isinstance(raw, list):
        items = [c for c in raw if str((c.get("metadata") or {}).get("strategy", "")).upper() == strategy.upper()
                 or str(c.get("chunk_id", "")).startswith(f"{strategy}_")]
        if not items:
            items = raw
    else:
        raise ValueError("local_chunks.json không hợp lệ")

    chunks: list[dict[str, Any]] = []
    for c in items:
        chunk_id = c.get("id") or c.get("chunk_id")
        meta = dict(c.get("metadata") or {})
        meta.setdefault("section", c.get("section"))
        meta.setdefault("document_type", c.get("document_type"))
        meta.setdefault("strategy", strategy)
        chunks.append({
            "id": chunk_id,
            "text": c.get("text") or "",
            "source_id": c.get("source_id") or c.get("document_id"),
            "source": c.get("source") or "",
            "page": c.get("page"),
            "metadata": meta,
            "embedding": c.get("embedding"),
        })
    if not chunks:
        raise ValueError(f"Không có chunk cho strategy {strategy}")
    return chunks


def load_questions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("questions.json phải là list hoặc object có key questions")

    out: list[dict[str, Any]] = []
    for q in items:
        answerable = q.get("answerable", True) is not False
        row: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": answerable,
            "type": q.get("type"),
        }
        if "filter" in q:
            row["filter"] = q["filter"]
        if answerable:
            gold = q.get("gold")
            if gold is None:
                gold = [{
                    "source_id": q.get("gold_source_id"),
                    "page": q.get("gold_page"),
                    "evidence": q.get("evidence_quote") or "",
                }]
            row["gold"] = gold
            row["expected_answer"] = q.get("expected_answer") or ""
        else:
            row["expected_behavior"] = q.get("expected_behavior", "search_and_abstain")
            row["absence_reason"] = q.get("absence_reason", "")
        out.append(row)
    return out


def evidence_in_text(evidence: str, text: str) -> bool:
    if not evidence or not text:
        return False
    if evidence in text:
        return True
    ev = normalize_text(evidence)
    tx = normalize_text(text)
    if not ev:
        return False
    if ev in tx:
        return True
    # OCR soft match: first ~48 chars of evidence after normalizing
    probe = ev[:48]
    return bool(probe) and probe in tx


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    for g in gold:
        if chunk.get("source_id") != g.get("source_id"):
            continue
        if chunk.get("page") != g.get("page"):
            continue
        if evidence_in_text(g.get("evidence") or "", chunk.get("text") or ""):
            return True
    return False


def pack_context(results: list[dict[str, Any]], budget: int = 2400) -> tuple[str, list[str]]:
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source_id')} | page {c.get('page')}]\n{c.get('text')}"
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids
