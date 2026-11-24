# Docs-QA sinh câu trả lời có nguồn (gói B — Leader)

Generator chạy local bằng Ollama: `qwen2.5:3b`, prompt v2 (chọn trên dev, xem mục 5). API chỉ trả câu trả lời
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
ollama pull qwen2.5:3b
```

3. Tải tokenizer Qwen để đếm token prompt (không tải weights; Qwen2.5 1.5B và 3B dùng chung tokenizer):

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

Mỗi lần gọi Ollama thật ghi một bản JSONL vào `runs/generation_cache/`. Có thể
đổi thư mục bằng `SUPPORTLENS_GENERATION_DIR`. Khóa gồm digest model, cấu hình
và tokenizer, prompt, câu hỏi, lịch sử, phiên bản retrieval cùng ID, thứ tự và
hash nội dung của từng đoạn đã chọn. Mặc định câu hỏi và lịch sử chỉ góp hash.
`SUPPORTLENS_LOG_QUESTIONS=1` cho phép lưu cả đầu vào và prompt để rà soát.

| `cache_status` | Điều đã xảy ra |
|---|---|
| `miss` | Ollama vừa sinh cho yêu cầu này. Nhật ký lưu cả lần lỗi, bị hủy và chữ sinh dở. |
| `hit` | Lấy kết quả đã hoàn tất từ RAM. |
| `replay` | Đọc kết quả đã hoàn tất từ đĩa, kể cả sau khi khởi động lại. |

Chỉ `answer` và `refused` được dùng lại. Báo TTFT và thời gian sinh live trên
các lượt `miss`. Hai loại đọc cache có TTFT bằng `null`. Thư mục nhật ký ở máy local.

Deadline tính từ lúc bắt đầu tạo prompt, bao gồm lúc chờ kết nối, chờ byte đầu
và khoảng im lặng giữa hai token. Hết giờ, API đóng luồng Ollama và trả bằng
chứng với quyết định `retrieved`. Chữ sinh dở chỉ dùng để điều tra lỗi.

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
- Huỷ bằng cách đóng kết nối. Server hủy lượt đọc Ollama và ghi trạng thái `cancelled` cho lượt đang sinh.
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
# so sánh cấu hình khác trên cùng mẫu:
.\.venv\Scripts\python.exe scripts/eval_docs_generation.py --limit 50 --model qwen2.5:1.5b --prompt-policy v1
```

Script chọn 50 câu dev theo hash ID và ghi hash của danh sách này vào metrics.
Cả ba cấu hình dùng cùng câu hỏi và cùng gate R1. Mỗi lượt qua gate gọi Ollama
thật. Phép đo loại warm-up và lượt phát lại. Hai file raw cần có trong
`data/raw/multidoc2dial/` là `multidoc2dial_doc.json` và
`multidoc2dial_dial_validation.json`. Script kiểm tra hash và chỉ lấy chú thích
của các lượt thuộc mẫu dev.

- `generation_dev_*_metrics.json` ghi số đếm, mẫu số, latency và hash file dự đoán.
- `generation_predictions_*.dev.jsonl` ghi ID đoạn được chọn, ID thực sự vào prompt, nguồn model trích và quyết định cuối.
- `generation_journal/` nằm cạnh file metrics và lưu prompt, output, số token và độ dài phần hiện được của từng đoạn cho bộ dev công khai.

## 5. Kết quả dev trên máy local của Lâm (24/11/2026)

Gate chuyển người 12/50 câu. Trong 38 lượt gửi sang Ollama, 18 lượt có đoạn
gold trong top-3, nhưng chỉ **16/38** lượt giữ được ít nhất một nửa span gold
trong phần văn bản đã đóng vào prompt 600 token. Bộ chấm đối chiếu offset trên
văn bản nguồn và loại phần tiêu đề. Một trích dẫn đạt tiêu chí khi chính nguồn
model trích còn chứa đủ span đó.

| Model và prompt | Xuất câu trả lời có citation | Trả lời khi thấy gold | Trong số đó, trích nguồn còn đủ gold | Từ chối | Thiếu citation | Thời gian sinh p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 1.5B v1 | 3/38 | 2/16 | 2/2 | 1 | 34 | 491 / 675 ms |
| Qwen 1.5B v2 | 34/38 | 14/16 | 10/14 | 1 | 3 | 269 / 687 ms |
| **Qwen 3B v2** | **29/38** | **15/16** | **11/15** | **8** | **1** | **434 / 912 ms** |

Giữ **3B v2** cho demo. Model trích nguồn còn đủ gold trong 11 trên 29 câu
trả lời và chuyển người 8 lần sau bước retrieval. Bảng tính trên một lần chạy
của từng cấu hình ở cùng mẫu dev. Với 3B, TTFT p50/p95 là 65/77 ms, đo ở
generator đã warm-up. Các lần sinh lại có thể cho câu khác. Dùng hash predictions
trong file metrics để nhận diện lần chạy.

Các tỉ lệ trên đo mức giao với chú thích nguồn. Độ đúng của từng phát biểu cần
người đọc kiểm tra. Generator chỉ thấy câu hỏi hiện tại và tối đa ba đoạn trong
ngân sách 600 token. Lịch sử hội thoại được dùng ở bước truy xuất.

[Metrics 1.5B v1](../runs/docs_qa/generation_dev_qwen2.5-1.5b_v1_n128_metrics.json) · [Metrics 1.5B v2](../runs/docs_qa/generation_dev_qwen2.5-1.5b_v2_n128_metrics.json) · [Metrics 3B v2](../runs/docs_qa/generation_dev_qwen2.5-3b_v2_n128_metrics.json)
