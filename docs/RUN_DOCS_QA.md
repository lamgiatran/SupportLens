# Docs-QA — chạy phần đã được đo

V0.4: xem [RUN_DENSE.md](RUN_DENSE.md) để chạy R1/R2/R3 với gate. Các bước BM25
bên dưới vẫn dùng được; `score_method` hiện là `bm25_top1_per_term_v1`.

ZIP đầy đủ đã kèm chunks, qrels, split IDs và tokenizer nhỏ. Giải nén và mở
terminal tại `SupportLens`; không cần cài torch để chạy BM25 passage.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\python.exe -m pytest -q
$env:SUPPORTLENS_DOCS_BUNDLE = 'data/processed/docs_qa'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Terminal thứ hai, tại cùng thư mục:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

API trả evidence `unit=chunk` và `score_method=bm25_top1_per_term_v1`; chưa tạo câu trả lời
bằng LLM. Request có history dùng cùng query builder current-first với offline.
Biến `SUPPORTLENS_DOCS_BUNDLE` ưu tiên hơn cấu hình BM25 document cũ.

Đánh giá dev (script không có chế độ mở held-out):

```powershell
.\.venv\Scripts\python.exe scripts/eval_docs_bm25.py
```

Kết quả ở `runs/docs_qa/dev_metrics.json`: hit@3, hit@5, tỷ lệ toàn bộ gold chunks được
tìm trong top-5 và MRR@10, báo riêng strict/lenient. Đọc đúng định nghĩa từng metric;
đây là dev với segmentation riêng, không phải số test hoặc tái lập paper gốc.
Latency chỉ là truy xuất query đã chuẩn bị trên index ấm, không phải độ trễ UI/LLM.

Tạo lại từ raw đã tải khi cần:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset multidoc2dial
.\.venv\Scripts\python.exe scripts/build_docs_bundle.py
.\.venv\Scripts\python.exe scripts/smoke_docs_api.py
```

Lệnh smoke cần raw validation để đối chiếu history thật; không cần chạy nó chỉ
để mở demo. Nếu khôi phục Git bằng bundle, copy runtime assets từ thư mục đã
giải nén sang working repo; Git không chứa chunks lớn, tokenizer hay `.npz`.

Quy tắc đã cố định: chunk tối đa 220 WordPiece gồm title, overlap 32 token,
encoder/query limit 256; strict phủ >=50% một span, lenient có giao nhau; nhiều
reference lấy hợp các positives. Dedup chỉ loại chồng lấp >=50% đoạn ngắn hơn
trong cùng tài liệu; giữ các đoạn bổ sung nhau. Không sửa text gốc hay dấu link.

Public validation được chia theo dialogue, stratified domain, seed 42: dev 329
dialogues, held-out 332. Qrels được dựng/audit cho các split; retrieval chỉ chạy
dev ở giai đoạn này. Xem `qrels_audit.jsonl` để review 20 ca; unknown/mismatch/
coverage issues có file riêng, không tự xóa query để làm đẹp metric.

Citation dùng `[S1]`, `[S1, S2]` (cũng nhận `[S1, 2]`). Dấu số trần `[2]`, `[4]`
là ký hiệu trong nguồn, không tự đổi thành citation. Nếu nguồn đã chứa `[S1]`,
prompt dùng namespace ngắn khác như `[SS1]`; parser đọc đúng namespace đó.
Trả lời không có alias hợp lệ phải được tính là thiếu citation; việc alias hợp lệ
chưa chứng minh nguồn hỗ trợ nội dung. Gate coverage đã có; generation/deadline vẫn làm tiếp.

Nguồn: https://doc2dial.github.io/multidoc2dial/ và
https://huggingface.co/datasets/IBM/multidoc2dial.
