# SupportLens — v0.4.1

Bản nền cho nhóm 4 người: Docs-QA trên MultiDoc2Dial và Banking triage Handoff-50.
Hướng dẫn chạy: [RUN_DENSE.md](docs/RUN_DENSE.md).
Việc giao cho từng người: [TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md).
Phân định code được xuất bản: [OWNERSHIP_REVIEW.md](docs/OWNERSHIP_REVIEW.md).

Phần Leader đã có: chunks/qrels strict–lenient, dialogue dev/held-out split,
history hai tin nhắn, R1 BM25 / R2 frozen MiniLM / R3 RRF, Docs gate coverage,
audit context và adapter Docs vào API. Chỉ đánh giá retrieval trên dev.

Hưng chưa được làm thay Course-77. Cầu vẫn tự làm audit/calibration/head/H1–H4.
UI giữ scaffold v0.3 để Tuấn tự mở rộng retriever/history, streaming và Banking.
Generator phục vụ, cache generation, deadline tổng và fine-tuning R4/R5 còn pending.

## Chạy bản nền

Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\python.exe -m pytest -q
```

ZIP runtime kèm chunks/tokenizer/corpus vectors; GitHub chỉ giữ source, split IDs,
metrics và manifest. Nếu clone GitHub, cần tải/tạo lại runtime assets trước API
theo RUN_DENSE.md. Một test tokenizer thật sẽ skip nếu chưa có tokenizer asset.

API Docs nhận R1/R2/R3 và history; giao diện scaffold hiện chỉ gửi câu hỏi đơn tới R1.
Tuấn nhận contract ổn định để tự nối các lựa chọn đó. Banking trả 503 cho đến khi
Cầu giao pipeline thật và Tuấn nạp vào serving. Trạng thái retrieved là tìm thấy
evidence, chưa phải câu trả lời do LLM sinh.

## Kết quả và giới hạn

Đọc [VERIFICATION_v0.4.md](docs/VERIFICATION_v0.4.md): hit@3 strict R1/R2/R3 =
0.4668 / 0.4062 / 0.4803 trên 2.213 query dev. CI cho chênh lệch R3–R1 chứa 0.
Giữ R1 mặc định. Coverage gate 70% vẫn có retrieval risk cao; không phải bảo đảm
risk 10% hoặc độ đúng của câu trả lời. Top-3 còn có thể bị cắt khi pack prompt.

Các script tái lập ở scripts/; JSON kết quả ở runs/docs_qa/. Held-out retrieval,
Banking training, generation và UI TTFT chưa được đo. Không dùng kết quả kiểm
tra nhanh vector làm metric báo cáo/CV.

Kế hoạch gốc được giữ ở [plan-v1.0.md](docs/plan-v1.0.md). Các ticket owner là
phân công, không phải bằng chứng người đó đã hoàn thành scaffold.
[AI_USAGE.md](docs/AI_USAGE.md) ghi rõ phần trợ lý chuẩn bị.

Repo riêng tư: https://github.com/lamgiatran/SupportLens.
