"""OpenAI / Ollama tool-calling controller for the Day 7 agent loop."""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

SYSTEM = """Bạn là trợ lý trả lời CHỈ dựa trên context đã retrieve (văn bản OCR tiếng Việt không dấu).
Quy tắc bắt buộc:
1) Chưa có context → gọi tool search; tham số query KHÔNG được rỗng (dùng lại câu hỏi nếu cần).
2) Có context liên quan → gọi tool answer. Trong answer PHẢI chèn [chunk-id] lấy từ Context IDs, và citations = danh sách các id đó.
3) Ví dụ answer hợp lệ: "Du lieu ca nhan la thong tin... [A_13nd.signed_p2_c0]" với citations=["A_13nd.signed_p2_c0"].
4) Nếu câu hỏi đòi số liệu cụ thể (mức phạt, email, tên công ty, thời hạn hoàn tiền, địa chỉ...) mà context KHÔNG nêu đúng chi tiết đó → PHẢI abstain: answer="Không đủ bằng chứng để trả lời.", citations=[]. Không suy diễn chung chung.
5) Không bịa số liệu ngoài context. Không trả lời bằng kiến thức bên ngoài."""

LOCAL_JSON_SYSTEM = """Bạn trả lời câu hỏi CHỈ từ CONTEXT (OCR tiếng Việt).
Trả về ĐÚNG 1 JSON object, không markdown:
{"action":"answer","answer":"...","citations":["chunk-id"]}
hoặc abstain:
{"action":"answer","answer":"Không đủ bằng chứng để trả lời.","citations":[]}
hoặc cần tìm thêm:
{"action":"search","query":"..."}

Quy tắc:
- Nếu CONTEXT có đoạn trả lời được câu hỏi: viết câu trả lời ngắn, chèn [chunk-id] vào answer, citations chứa đúng id đó.
- Chỉ abstain khi CONTEXT không chứa đủ chi tiết (đặc biệt số liệu/email/tên công ty/thời hạn cụ thể).
- Không bịa thông tin ngoài CONTEXT."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Tìm chunk liên quan trong vector/lexical store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu truy vấn tiếng Việt để retrieval.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Trả lời cuối cùng kèm citations, hoặc abstain nếu không đủ bằng chứng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách chunk-id hỗ trợ câu trả lời; [] nếu abstain.",
                    },
                },
                "required": ["answer", "citations"],
                "additionalProperties": False,
            },
        },
    },
]


class OpenAIController:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        kwargs: dict[str, Any] = {"api_key": api_key or "ollama"}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.base_url = base_url
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _track_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_usage = {
                "prompt_tokens": (usage.prompt_tokens or 0),
                "completion_tokens": (usage.completion_tokens or 0),
                "total_tokens": (usage.total_tokens or 0),
            }

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # Small local models: first turn always search; later turns use JSON completion.
        if self.base_url:
            if not history:
                return {"action": "search", "query": question.strip()}
            return self._decide_local_json(question, history)

        user_parts = [f"Câu hỏi: {question}"]
        if not history:
            user_parts.append("Chưa có context. Hãy search trước.")
        else:
            ctx = history[-1].get("context") or ""
            ids = history[-1].get("context_ids") or []
            queries = history[-1].get("queries") or []
            user_parts.append(f"Các query đã search: {queries}")
            user_parts.append(f"Context IDs: {ids}")
            user_parts.append("Context hiện có:\n" + (ctx if ctx.strip() else "(trống)"))
            user_parts.append(
                "Nếu đủ bằng chứng hãy gọi answer với citations; nếu chưa đủ hãy search query mới; "
                "nếu đã search đủ mà không có bằng chứng hãy answer abstain."
            )

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            temperature=0,
        )
        self._track_usage(resp)

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Tool arguments không phải JSON: {exc}") from exc
            return self._normalize_action(name, args, question=question, history=history)

        return self._parse_content_fallback(msg.content or "", question, history)

    def _decide_local_json(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        ctx = history[-1].get("context") or ""
        ids = list(history[-1].get("context_ids") or [])
        searches = history[-1].get("queries") or []
        # Keep only top-2 chunks for small local models
        if ids:
            parts = [p for p in ctx.split("\n\n") if p.strip()]
            keep = []
            kept_ids = []
            for cid in ids[:2]:
                block = next((p for p in parts if p.startswith(f"[{cid}")), None)
                if block:
                    keep.append(block)
                    kept_ids.append(cid)
            if keep:
                ctx = "\n\n".join(keep)
                ids = kept_ids

        fewshot = (
            "Ví dụ JSON khi có bằng chứng:\n"
            '{"action":"answer","answer":"Dt lieu ca nhan la thong tin... [A_13nd.signed_p2_c0]",'
            '"citations":["A_13nd.signed_p2_c0"]}\n'
            "Ví dụ JSON khi thiếu bằng chứng:\n"
            '{"action":"answer","answer":"Không đủ bằng chứng để trả lời.","citations":[]}\n'
        )
        user = (
            f"{fewshot}\n"
            f"Câu hỏi: {question}\n"
            f"Context IDs hợp lệ: {ids}\n"
            f"CONTEXT:\n{ctx if ctx.strip() else '(trống)'}\n\n"
            "Hãy đọc CONTEXT. Nếu có câu trả lời, trích ý chính + [chunk-id]. Chỉ trả JSON."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LOCAL_JSON_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        self._track_usage(resp)
        content = (resp.choices[0].message.content or "").strip()
        parsed = self._parse_content_fallback(content, question, history)
        # Avoid endless search loops on local models after first retrieval.
        if parsed.get("action") == "search" and len(searches) >= 1:
            return {
                "action": "answer",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
            }
        return parsed

    def _normalize_action(
        self,
        name: str,
        args: dict[str, Any],
        *,
        question: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        history = history or []
        if name == "search":
            query = (args.get("query") or "").strip() or question.strip()
            if not query:
                raise ValueError("search query rỗng")
            return {"action": "search", "query": query}
        if name == "answer":
            answer = args.get("answer") if isinstance(args.get("answer"), str) else ""
            citations = args.get("citations") if isinstance(args.get("citations"), list) else []
            citations = [c for c in citations if isinstance(c, str)]
            if answer and not citations:
                allowed = set()
                if history:
                    allowed.update(history[-1].get("context_ids") or [])
                found = re.findall(r"\[([^\[\]]+)\]", answer)
                citations = [c for c in found if (not allowed or c in allowed)]
            low = answer.casefold()
            abstain_markers = (
                "không đủ bằng chứng",
                "khong du bang chung",
                "không có trong context",
                "khong co trong context",
                "not in context",
                "không nêu",
                "khong neu",
            )
            if any(m in low for m in abstain_markers):
                return {
                    "action": "answer",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                }
            # Agent loop requires literal [chunk-id] markers inside the answer text.
            for cid in citations:
                marker = f"[{cid}]"
                if marker not in answer:
                    answer = (answer.rstrip() + f" {marker}").strip()
            return {"action": "answer", "answer": answer, "citations": citations}
        raise ValueError(f"Tool không hỗ trợ: {name}")

    def _parse_content_fallback(
        self, content: str, question: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        # Extract first JSON object if mixed with prose
        if "{" in text and "}" in text:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                if payload.get("action") in ("search", "answer"):
                    if payload["action"] == "search":
                        return self._normalize_action(
                            "search", payload, question=question, history=history
                        )
                    return self._normalize_action(
                        "answer", payload, question=question, history=history
                    )
                if "query" in payload and "answer" not in payload:
                    return self._normalize_action(
                        "search", payload, question=question, history=history
                    )
                if "answer" in payload:
                    return self._normalize_action(
                        "answer", payload, question=question, history=history
                    )
        except json.JSONDecodeError:
            pass

        if not history:
            return {"action": "search", "query": question}
        if "không đủ bằng chứng" in text.casefold():
            return {
                "action": "answer",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
            }
        allowed = set(history[-1].get("context_ids") or []) if history else set()
        found = [c for c in re.findall(r"\[([^\[\]]+)\]", text) if c in allowed]
        if text and found:
            return {"action": "answer", "answer": text, "citations": found}
        return {
            "action": "answer",
            "answer": text or "Không đủ bằng chứng để trả lời.",
            "citations": [],
        }
