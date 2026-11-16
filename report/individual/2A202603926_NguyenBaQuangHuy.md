# Báo cáo Day 7

**Họ tên:** Nguyễn Bá Quang Huy  
**MSSV:** 2A202603926  
**Lớp / đề tài:** Day 7 — Agentic RAG (Retrieval & Context Engineering)

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
  - Nguồn: `13nd.signed.pdf` — Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân (văn bản công khai; dùng cho mục đích học tập).
  - Nội dung hợp lệ: **39 trang**; OCR text lưu trong Day 6.
  - Chunk strategy chính: **A** — **145** chunks (avg ~435.8 ký tự). (B/C = 96 chunks, không dùng làm primary.)
  - File: `day6_datas/local_chunks.json`, `day6_datas/chroma.sqlite3`, `day6_datas/questions.json`.
- Vector store, embedding model, phiên bản và số chiều:
  - Vector store: **Chroma** persistent (`day6_datas`), collection `chunks_A`, metric cosine.
  - Embedding: `intfloat/multilingual-e5-small`, **384** chiều, load qua `sentence-transformers` (query dùng prefix `query:`).
- Agent model, cách chạy và phân chia dev/test:
  - Model agent: **Ollama** `qwen2.5:7b` qua OpenAI-compatible API (`OPENAI_BASE_URL=http://127.0.0.1:11434/v1`).
  - Retriever đã chọn: **BM25** trên chunk strategy A (top-k=5, context budget=2400 ký tự).
  - Benchmark: **24 câu** — Dev 15 có đáp án + 2 thiếu evidence; Test 5 + 2.
  - Lệnh chính:
    ```powershell
    .\.venv\Scripts\python.exe -m src.run_retrieval --split dev
    .\.venv\Scripts\python.exe -m src.run_retrieval --split test
    .\.venv\Scripts\python.exe -m src.run_agent_benchmark --split dev --retriever bm25 --output starter/results/agent-bm25-dev-7b
    .\.venv\Scripts\python.exe -m src.run_agent_benchmark --split test --retriever bm25 --output starter/results/agent-bm25-test-7b
    ```
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
  - Có đáp án: có `gold` (source_id, page, evidence_quote) đối chiếu chunk OCR; Hit dùng khớp mềm (source+page + evidence chuẩn hóa) vì OCR không luôn khớp substring tuyệt đối.
  - Thiếu evidence: `answerable=false`, `expected_behavior=search_and_abstain` (câu hỏi gần chủ đề nhưng không có số liệu/chi tiết trong corpus).
  - Agent: grounded answer 0/1/2; citation phải nằm trong context và hỗ trợ nội dung; abstention đúng khi từ chối câu thiếu evidence.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (dense) | Dense Chroma A + E5, top-k=5 (báo top-3), budget 2400 | Baseline bắt buộc Day 6 | dev | 3/15 (0.20) | 3/15 (0.20) | 5/15 (0.33) | ~845 ms/câu |
| Phương pháp 2 (bm25) | BM25 lexical, top-k=5, budget 2400 | OCR kém semantic → lexical khớp từ khóa tốt hơn | **dev** | **10/15 (0.67)** | **13/15 (0.87)** | **14/15 (0.93)** | ~0.6 ms/câu |
| Phương pháp 3 (hybrid_rrf) | Dense + BM25, RRF k=60, candidate_k=10 | Fusion cải thiện dense yếu | dev | 5/15 (0.33) | 7/15 (0.47) | 12/15 (0.80) | ~22 ms/câu |
| Cải thiện 1 | BM25 + tăng top-k=8, budget=3200 | Nhiều candidate hơn → context hit tăng | dev | 10/15 (0.67) | 13/15 (0.87) | 14/15 (0.93) | ~0.7 ms/câu |
| Cải thiện 2 | Hybrid RRF với context budget rộng hơn (top-k=8) | Dense vẫn kéo thứ hạng xuống | dev | 5/15 (0.33) | 7/15 (0.47) | 13/15 (0.87) | ~35 ms/câu |
| **Cấu hình đã chọn (BM25)** | BM25 top-k=5, budget 2400, không gold-filter | Chốt theo Hit@3 + Context Evidence + chi phí thấp | **test** | **3/5 (0.60)** | **4/5 (0.80)** | **4/5 (0.80)** | ~1.1 ms/câu |
| Baseline dense | Dense top-k=5, budget 2400 | Đối chiếu test sau khi chốt | test | 1/5 (0.20) | 3/5 (0.60) | 5/5 (1.00) | ~1322 ms/câu |

**Hai lỗi / hiện tượng trên dev và cách xử lý:**

1. **Dense Hit@3 rất thấp (3/15)** dù score cosine cao và “phẳng” (~0.85–0.87). Nguyên nhân: corpus OCR + E5 semantic kém phân biệt trên văn bản lỗi chính tả. Thử hybrid RRF cải thiện Context Evidence (0.33→0.80) nhưng Hit@3 vẫn kém BM25. **Chọn BM25** làm primary.
2. **Gold evidence OCR không khớp nguyên văn** (chỉ ~7/20 exact substring). Đổi metric hit sang khớp mềm (normalize + probe) để đánh giá công bằng; ghi rõ trong README/`src/data.py`.

**Lý do chọn cấu hình:** BM25 thắng rõ trên Hit@1/Hit@3/Context Evidence trên dev, cực nhanh, ổn định trên test (Hit@3 4/5). Dense giữ để đáp ứng yêu cầu Day 6 và làm baseline đối chiếu. Tăng top-k không cải thiện Hit@3 (cutoff cố định top-3) nên giữ top-k=5 cho agent.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - Tool: `search` (BM25) + quyết định `answer`/`abstain` (local JSON path cho Ollama).
  - Lượt 1: luôn search bằng câu hỏi gốc (tránh tool-call rỗng trên model local).
  - Context: gom candidate, `pack_context` theo budget 2400; khi gọi model local chỉ đưa **top-2** chunk vào prompt để giảm nhiễu.
  - Dừng: answer hợp lệ (có `[chunk-id]` trong text + citations ∈ context), abstain, hoặc hết `max_searches=4`.
  - Lỗi kết nối ghi `status=error` (không mock kết quả).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| BM25 + qwen2.5:7b / **dev** (`agent_metrics`) | 17/17 | ~1 search/câu | 0:2, 1:4, 2:9 /15 · **mean=1.47** | **12/15** | 15/15 answered | **1/2** | **1/15** (q14) |
| BM25 + qwen2.5:7b / **test** (`agent_metrics`) | 7/7 | ~1 search/câu; ~7.9 s; 10646 tokens | 0:1, 1:0, 2:4 /5 · **mean=1.60** | **4/7** | 6/6 answered | **1/2** | **0/5** |

Nhãn thủ công: `starter/results/agent-bm25-*-7b/reviews.json`. Tổng hợp:

```powershell
.\.venv\Scripts\python.exe starter/agent_metrics.py --cases starter/results/agent-bm25-dev-7b/cases.json --traces starter/results/agent-bm25-dev-7b/traces.json --reviews starter/results/agent-bm25-dev-7b/reviews.json --split dev --output starter/results/agent-bm25-dev-7b/metrics_summary.json
.\.venv\Scripts\python.exe starter/agent_metrics.py --cases starter/results/agent-bm25-test-7b/cases.json --traces starter/results/agent-bm25-test-7b/traces.json --reviews starter/results/agent-bm25-test-7b/reviews.json --split test --output starter/results/agent-bm25-test-7b/metrics_summary.json
```

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q01 (dev) | Có đáp án | search → top có `A_13nd.signed_p2_c0` (Định nghĩa DLCN) | Trả lời định nghĩa + `[A_13nd.signed_p2_c0]` | **2** — đúng ý, có citation hỗ trợ |
| q21 (dev) | Thiếu evidence (mức phạt) | search → chunk không có mức phạt tiền | Vẫn “answered” kèm citation không hỗ trợ số liệu | **Abstain sai** — hallucination / không abstain |
| q22 (dev) | Thiếu evidence (công ty ABC Soft) | search 1 lần | “Không đủ bằng chứng…” | **Abstain đúng** |
| q14 (dev) | Có đáp án (khan cấp/tính mạng) | search có evidence hit | Abstain | **Từ chối nhầm** — context đủ nhưng model local thận trọng quá |
| q16 (test) | Có đáp án | search → p23 | Cục An ninh mạng… `[A_13nd.signed_p23_c0]` | **2** |
| q17 (test) | Có đáp án | search lệch (p5) | Trả lời chung chung, không nêu 3 nhóm biện pháp | **0** — sai/thiếu so với gold p22 |
| q23 (test) | Thiếu evidence (hoàn tiền TMĐT) | search | Abstain | **Abstain đúng** |
| q24 (test) | Thiếu evidence (email Cục) | search | Answered: nói không có email nhưng vẫn gắn citation | **Abstain sai** (nên từ chối sạch) |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Retrieval: `starter/results/retrieval-dev/`, `starter/results/retrieval-test/`, `starter/results/retrieval-dev-topk8/`
  - Agent: `starter/results/agent-bm25-dev-7b/`, `starter/results/agent-bm25-test-7b/` (`traces.json`, `summary.json`, `cases.json`, `reviews.json`, `metrics_summary.json`)
  - Code: `src/` (`retrievers.py`, `openai_controller.py`, `run_retrieval.py`, `run_agent_benchmark.py`)
  - Chi phí API cloud: **0** (Ollama local). Thời gian agent test ~7.9 s/câu; tổng ~10646 tokens (local, không tính tiền).
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Trước: dense yếu trên OCR; agent `qwen2.5:3b` dao động mạnh (over-abstain hoặc hallucinate).
  - Sau: chọn BM25 + `qwen2.5:7b` + JSON local path + tự gắn `[chunk-id]` vào answer → retrieval và trả lời ổn định hơn trên câu có đáp án.
  - **Đánh đổi:** BM25 nhanh/rẻ nhưng kém semantic đa nghĩa; model local 7b không cần API key nhưng abstention kém hơn model cloud; hybrid không vượt BM25 trên corpus này.
  - **Giới hạn:** 1 PDF ~39 trang OCR; chưa làm advanced 100 trang; chưa chấm tay toàn bộ citation support từng câu bằng `agent_metrics.py` (đã chấm mẫu trong bảng trên); abstention vẫn là điểm yếu.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện trong lần nộp này (corpus hiện ~39 trang < 100 trang; chưa index bảng/ảnh/biểu đồ riêng và chưa đạt ngưỡng 90% test theo ADVANCED).
