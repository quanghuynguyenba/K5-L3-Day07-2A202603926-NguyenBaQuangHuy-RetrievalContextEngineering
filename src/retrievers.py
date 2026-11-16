"""Three retrieval methods: dense (Chroma), BM25 lexical, hybrid RRF."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from .embeddings import embed_query


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "").casefold()
    return re.findall(r"\w+", text)


def _meta_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map user filters to Chroma where. Never use gold labels."""
    if not filters:
        return None
    where: dict[str, Any] = {}
    for key in ("section", "document_type", "page", "document_id", "strategy"):
        if key in filters and filters[key] is not None:
            where[key] = filters[key]
    if "page" in where and not isinstance(where["page"], int):
        try:
            where["page"] = int(where["page"])
        except (TypeError, ValueError):
            del where["page"]
    if not where:
        return None
    if len(where) == 1:
        k, v = next(iter(where.items()))
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in where.items()]}


def _parse_chunk_id(chunk_id: str) -> tuple[str | None, int | None]:
    # e.g. A_13nd.signed_p1_c0
    m = re.match(r"^[A-Za-z]_(.+)_p(\d+)_c\d+$", chunk_id or "")
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _normalize_chunk(chunk_id: str, text: str, meta: dict[str, Any] | None,
                     score: float, strategy: str) -> dict[str, Any]:
    meta = dict(meta or {})
    parsed_doc, parsed_page = _parse_chunk_id(chunk_id)
    page = meta.get("page", parsed_page)
    if isinstance(page, str) and page.isdigit():
        page = int(page)
    source_id = meta.get("document_id") or meta.get("source_id") or parsed_doc or ""
    return {
        "id": chunk_id,
        "text": text or "",
        "source_id": source_id,
        "source": meta.get("source") or meta.get("filename") or source_id,
        "page": page,
        "metadata": meta,
        "score": float(score),
        "strategy": strategy,
    }


class DenseRetriever:
    """Dense retrieval against Day 6 Chroma collection (cosine)."""

    def __init__(self, chroma_path: str, collection: str, embedding_model: str):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_collection(collection)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        emb = embed_query(query, self.embedding_model)
        kwargs: dict[str, Any] = {
            "query_embeddings": [emb],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        where = _meta_where(filters)
        if where:
            kwargs["where"] = where
        try:
            res = self.collection.query(**kwargs)
        except Exception:
            # Fallback without filter if metadata keys mismatch
            if where:
                res = self.collection.query(
                    query_embeddings=[emb], n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                raise
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity-like score
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(_normalize_chunk(cid, doc or "", meta, score, "dense"))
        return out[:top_k]


class BM25Retriever:
    """Lexical BM25 over strategy-A chunk texts."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.tokens = [_tokenize(c["text"]) for c in chunks]
        self.index = BM25Okapi(self.tokens)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        scores = self.index.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i]["id"]))
        out: list[dict[str, Any]] = []
        for i in ranked:
            c = self.chunks[i]
            if filters:
                meta = c.get("metadata") or {}
                ok = True
                for k, v in filters.items():
                    if k == "page" and c.get("page") != v and meta.get("page") != v:
                        ok = False
                        break
                    if k in ("section", "document_type") and meta.get(k) != v:
                        ok = False
                        break
                    if k == "document_id" and c.get("source_id") != v:
                        ok = False
                        break
                if not ok:
                    continue
            if scores[i] <= 0:
                continue
            out.append({
                "id": c["id"],
                "text": c["text"],
                "source_id": c["source_id"],
                "source": c["source"],
                "page": c["page"],
                "metadata": c.get("metadata") or {},
                "score": float(scores[i]),
                "strategy": "bm25",
            })
            if len(out) >= top_k:
                break
        return out


def rrf_fuse(result_lists: list[list[dict[str, Any]]], *, top_k: int = 3,
             k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            cid = item["id"]
            scores[cid] += 1.0 / (k + rank)
            prev = best.get(cid)
            if prev is None or rank < prev.get("_rank", math.inf):
                row = dict(item)
                row["_rank"] = rank
                best[cid] = row
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for cid, score in ordered[:top_k]:
        row = dict(best[cid])
        row.pop("_rank", None)
        row["score"] = float(score)
        row["strategy"] = "hybrid_rrf"
        out.append(row)
    return out


class HybridRetriever:
    """Dense + BM25 fused with Reciprocal Rank Fusion."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, *, candidate_k: int = 10):
        self.dense = dense
        self.bm25 = bm25
        self.candidate_k = candidate_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        ck = max(self.candidate_k, top_k)
        dense_hits = self.dense.search(query, filters=filters, top_k=ck)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=ck)
        return rrf_fuse([dense_hits, bm25_hits], top_k=top_k)


class RetrieverAdapter:
    """Adapter matching starter.agentic.Retriever protocol."""

    def __init__(self, impl: Any):
        self.impl = impl

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return self.impl.search(query, filters=filters, top_k=top_k)
