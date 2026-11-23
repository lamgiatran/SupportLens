# Docs-QA sinh câu trả lời có nguồn (gói B — Leader)

Generator chạy local bằng Ollama (`qwen2.5:1.5b`). API chỉ trả câu trả lời
(`decision="answer"`) khi câu trả lời trích ít nhất một nguồn và **mọi** nguồn
trích đều là đoạn văn có trong prompt. Các trường hợp còn lại trả về đoạn văn
đã tìm (`retrieved`) hoặc chuyển người (`handoff`), kèm lý do.

| Kết quả generator | decision | reason |
|---|---|---|
| Có nguồn, mọi nguồn hợp lệ | `answer` | `generated_with_verified_citations` |
| Model nói không đủ thông tin | `handoff` | `generator_insufficient_evidence` |
| Không trích nguồn / nguồn bịa / rỗng | `retrieved` | `generator_<status>_fallback_to_evidence` |
| Quá deadline / lỗi Ollama | `retrieved` | `generator_timeout…` / `generator_error…` |
| Gate retrieval từ chối | `handoff` | `retrieval_below_dev_threshold` (không gọi model) |

## 1. Chuẩn bị một lần

1. Cài Ollama cho Windows: https://ollama.com/download, mở app một lần.
2. Tải model đã pin trong `configs/default.json`:

```powershell
ollama pull qwen2.5:1.5b
```

3. Tải tokenizer Qwen để đếm token prompt (không tải weights):

```powershell
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py --include-generator
```

API **không tự pull model**. Thiếu model thì `/health` báo
`docs_generator.state = "failed"`, `error_type = "LookupError"`, và Docs-QA vẫn
trả đoạn văn như trước.

## 2. Chạy API có generator

Đặt biến môi trường như RUN_DENSE.md, thêm:

```powershell
$env:SUPPORTLENS_GENERATOR = 'ollama'
# tuỳ chọn:
# $env:SUPPORTLENS_GENERATOR_DEADLINE = '30'          # giây, mặc định 30
# $env:SUPPORTLENS_OLLAMA_URL = 'http://127.0.0.1:11434'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --port 8000
```

Startup kiểm tra model có trong `ollama list` rồi sinh một câu warm-up để nạp
model vào RAM. Xem `/health` → `docs_generator_ready`, `docs_generator.warmup_ms`.

Cache: câu trả lời hợp lệ (`answer`, `refused`) được giữ trong RAM theo
(model + cấu hình + prompt), tối đa 256 mục. Response ghi `cache_status`
`miss`/`hit`. Câu trả lời lỗi, quá giờ hoặc thiếu nguồn không được cache.

Deadline: kiểm tra theo wall-clock ở mỗi token và bằng read timeout của HTTP.
Quá deadline thì bỏ phần chữ đã sinh, trả `retrieved` với lý do timeout.

## 3. Endpoint stream cho UI (giao Tuấn)

`POST /v1/assist/stream`, body giống `/v1/assist`. Trả `application/x-ndjson`,
mỗi dòng một JSON. Lỗi trước khi stream bắt đầu (retriever chưa nạp, Banking
chưa sẵn sàng…) vẫn là HTTP 503 như `/v1/assist`.

| event | Trường | Ý nghĩa |
|---|---|---|
| `start` | `request_id`, `mode`, `evidence[]`, `retrieve_ms`, `generating` | Hiện nguồn ngay. `generating=false` nghĩa là sẽ không có `delta` |
| `delta` | `text` | Ghép nối tiếp để hiện chữ dần |
| `done` | `response` (đúng `AssistResponse`) | **Nguồn sự thật cuối cùng**: dùng `response.decision/answer/citations/reason` |
| `error` | `request_id`, `code`, `error_type` | Lỗi server giữa chừng; dừng stream |

Quy tắc cho UI:
- Nếu `done.response.decision != "answer"`, **xoá phần chữ đã stream** và hiện
  đoạn văn/chuyển người theo `reason`. Chữ đã stream có thể chứa nguồn bịa.
- Huỷ = đóng kết nối. Server dừng gọi Ollama ở token kế tiếp.
- Banking qua endpoint này chỉ có `start` rồi `done`.

Ví dụ (rút gọn):

```text
{"event": "start", "request_id": "6f…", "mode": "docs_qa", "evidence": [{"evidence_id": "c1…", …}], "retrieve_ms": 4.2, "generating": true}
{"event": "delta", "text": "You can renew "}
{"event": "delta", "text": "online [S1]."}
{"event": "done", "response": {"decision": "answer", "answer": "You can renew online [S1].", "citations": ["c1…"], "cache_status": "miss", "latency": {"ttft_ms": 850.1, "generate_ms": 2410.7, …}, …}}
```

Thử bằng PowerShell:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/v1/assist/stream -H "Content-Type: application/json" `
  -d '{\"mode\":\"docs_qa\",\"question\":\"How do I renew my license online?\",\"retriever\":\"R1\"}'
```

## 4. Đo nhanh trên dev

```powershell
.\.venv\Scripts\python.exe scripts/eval_docs_generation.py --limit 50
```

Lấy 50 câu dev theo thứ tự hash (rải đều qua các hội thoại), chạy R1 → gate dev
→ generator, không dùng cache. Ghi:
- `runs/docs_qa/generation_dev.json`: tỉ lệ từng status, tỉ lệ `answer`, tỉ lệ
  câu trả lời trích đúng đoạn gold (strict qrels), TTFT/latency p50–p95.
- `runs/docs_qa/generation_predictions.dev.jsonl`: từng câu, để review nguồn bằng mắt.

"Trích đúng đoạn gold" là proxy cho grounding, không phải độ đúng của câu trả
lời. Không chạy script này trên held-out trước khi freeze cấu hình.

## Giới hạn hiện tại

- Prompt chỉ dùng câu hỏi hiện tại (retrieval vẫn dùng history). Câu hỏi phụ
  thuộc ngữ cảnh hội thoại có thể bị trả lời thiếu.
- Tối đa 3 đoạn văn đầu, tổng 600 token evidence.
- Cache nằm trong RAM, mất khi khởi động lại API.
