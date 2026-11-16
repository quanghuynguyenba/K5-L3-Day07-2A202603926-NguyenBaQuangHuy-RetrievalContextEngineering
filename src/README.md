# Day 7 implementation

Kết nối Day 6 (Chroma strategy **A**, `intfloat/multilingual-e5-small`) với agent OpenAI tool-calling.

## Phương pháp retrieval

| Tên | Mô tả |
|---|---|
| `dense` | Embedding query (prefix `query:`) → Chroma `chunks_A` (cosine) |
| `bm25` | BM25 lexical trên text chunk strategy A |
| `hybrid_rrf` | Dense + BM25, gộp bằng Reciprocal Rank Fusion |

## Chạy

Từ thư mục gốc repo, dùng Python trong `.venv`:

```bash
# So sánh 3 phương pháp trên dev
.\.venv\Scripts\python.exe -m src.run_retrieval --split dev

# Agent thật (hybrid) — smoke 2 câu
.\.venv\Scripts\python.exe -m src.run_agent_benchmark --split dev --retriever hybrid_rrf --ids q01 q21

# Agent full dev / test
.\.venv\Scripts\python.exe -m src.run_agent_benchmark --split dev --retriever hybrid_rrf
.\.venv\Scripts\python.exe -m src.run_agent_benchmark --split test --retriever hybrid_rrf
```

Cấu hình nằm trong `.env` (không commit). Kết quả ghi vào `starter/results/`.

**Lưu ý:** Gold OCR có thể không khớp substring tuyệt đối; Hit@k dùng khớp mềm (source+page + evidence chuẩn hóa).

## Kết quả retrieval (dev, strategy A)

| Method | Hit@1 | Evidence Hit@3 | Context Evidence |
|---|---|---|---|
| dense | 3/15 | 3/15 | 5/15 |
| bm25 | 10/15 | 13/15 | 14/15 |
| hybrid_rrf | 5/15 | 7/15 | 12/15 |

Trên corpus OCR, **BM25** mạnh hơn dense; agent nên ưu tiên `bm25` hoặc cấu hình cải thiện (ví dụ tăng `candidate_k` / ưu tiên lexical).

## Local LLM (Ollama)

`.env` mặc định trỏ Ollama:

```
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
```

Cài Ollama rồi pull model:

```powershell
ollama pull qwen2.5:7b
.\.venv\Scripts\python.exe -m src.run_agent_benchmark --split dev --retriever bm25
.\.venv\Scripts\python.exe -m src.run_agent_benchmark --split test --retriever bm25
```

Để dùng lại OpenAI cloud: xóa `OPENAI_BASE_URL`, đặt `OPENAI_API_KEY` và `OPENAI_MODEL=gpt-4o-mini`.
