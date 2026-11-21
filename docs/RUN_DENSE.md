# Chạy Docs-QA v0.4.1

Mở terminal ở thư mục `SupportLens`. Gói đầy đủ có chunks/qrels, tokenizer MiniLM,
cache corpus R2 và cache Banking. Weights encoder và LLM không nằm trong ZIP.
Không cần raw để chạy API từ ZIP; raw dùng khi tái tạo/audit dữ liệu.
Nếu clone GitHub, cài môi trường ở bước 1 trước rồi tạo runtime assets bên dưới.

App demo chạy local: API/UI, retrieval, Banking và Ollama ở máy demo. Colab với
GPU mục tiêu A100 dành cho train/fine-tune và chuẩn bị artifact; sau đó chuyển
checkpoint/config/index về máy local. Notebook/run fine-tune R4 chưa có trong
bản hiện tại. Không cần duy trì phiên Colab để app trả lời người dùng.

## 1. Cài và kiểm tra

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\python.exe -m pytest -q
```

Nếu clone GitHub và chưa có runtime assets, chạy sau khi đã cài môi trường:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset multidoc2dial
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py
.\.venv\Scripts\python.exe scripts/build_docs_bundle.py
```

Test tokenizer thật có thể skip trước khi tải tokenizer; ghi rõ lý do skip.

Chỉ dùng BM25 thì bỏ qua bước 2. Chạy R2/R3 cần cài thêm ML và tải model một lần:

## 2. Chuẩn bị encoder R2/R3

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe scripts/build_docs_dense.py --allow-download
```

Lệnh này tải/kiểm tra model đã pin và warm-up. Nếu `corpus.npz` đã có và đúng
manifest, nó tái dùng cache; không encode lại. Máy thiếu weights sẽ nhận lỗi
startup nếu chạy API trước bước này. Không tải model trong request người dùng.

R2 dùng cosine của vector MiniLM đã chuẩn hóa L2. R3 trộn thứ hạng BM25 và R2
bằng RRF, mỗi nhánh lấy 50 ứng viên rồi khử overlap. Thay history làm đổi query
vectors, version phục vụ và gate; có thể giữ corpus vectors khi chunks/encoder
không đổi. Thay encoder hoặc chunks phải dựng cache mới, không sửa manifest để
ép một cache khác được nạp.

## 3. Chạy API và UI

```powershell
$env:SUPPORTLENS_DOCS_BUNDLE = 'data/processed/docs_qa'
$env:SUPPORTLENS_DOCS_GATES = 'artifacts/docs_gates'
# Chỉ đặt dòng dưới nếu đã chuẩn bị encoder ở bước 2:
$env:SUPPORTLENS_DOCS_DENSE = 'data/processed/docs_dense'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Nếu chỉ chạy R1, xóa biến `SUPPORTLENS_DOCS_DENSE` trong terminal đó trước khi
khởi động (`Remove-Item Env:SUPPORTLENS_DOCS_DENSE -ErrorAction SilentlyContinue`).
Không đặt đồng thời biến gate đơn `SUPPORTLENS_DOCS_GATE` và thư mục gates.

Terminal thứ hai:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

UI trên nhánh `feat/tuan-ui-mock-stream` có hai chế độ, bộ chọn retriever và
history; xem [hướng dẫn 3 demo local](RUN_UI_DEMOS.md) để review. Chỉ chọn R2/R3
khi encoder/cache đã được nạp. Cũng có thể mở http://127.0.0.1:8000/docs,
POST /v1/assist với retriever R2 hoặc R3. API lấy câu hiện tại trước, rồi tối đa hai tin nhắn quá khứ
mới nhất. Trạng thái retrieved chỉ là tìm evidence, chưa sinh câu trả lời bằng LLM.

`GET /health` báo methods, encoder ready và thời gian load/warm-up riêng.
Banking mặc định trả 503 khi chưa cấu hình pipeline. Cách bật mock serving để
test T-01 và thay bằng pipeline thật của Cầu: [RUN_BANKING_SERVING.md](RUN_BANKING_SERVING.md).

## 4. Tái lập phép đo (Leader)

Tải raw bằng `scripts/download_data.py --dataset multidoc2dial` nếu máy chưa có.
Bundle và raw đều được kiểm tra hash. Các script dưới chỉ có entry point dev:

```powershell
.\.venv\Scripts\python.exe scripts/ablate_history.py
.\.venv\Scripts\python.exe scripts/gate_scores.py runs/docs_qa/history_predictions.jsonl --k 5 --out runs/docs_qa/history_gate5_metrics.json
.\.venv\Scripts\python.exe scripts/eval_docs_retrievers.py
.\.venv\Scripts\python.exe scripts/gate_scores.py runs/docs_qa/retrieval_predictions.dev.jsonl --k 3 --out runs/docs_qa/retrieval_gate3_metrics.json
.\.venv\Scripts\python.exe scripts/fit_docs_gate.py runs/docs_qa/retrieval_predictions.dev.jsonl --method R1 --target-coverage 0.7 --out artifacts/docs_gates/R1.json
.\.venv\Scripts\python.exe scripts/fit_docs_gate.py runs/docs_qa/retrieval_predictions.dev.jsonl --method R2 --target-coverage 0.7 --out artifacts/docs_gates/R2.json
.\.venv\Scripts\python.exe scripts/fit_docs_gate.py runs/docs_qa/retrieval_predictions.dev.jsonl --method R3 --target-coverage 0.7 --out artifacts/docs_gates/R3.json
.\.venv\Scripts\python.exe scripts/smoke_docs_retrievers.py
```

Tùy chọn audit context, không gọi LLM:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py --include-generator
.\.venv\Scripts\python.exe scripts/audit_docs_context.py
```

`runs/docs_qa/` giữ metrics/audit và full risk–coverage curves. Predictions và
query vectors lớn được sinh khi chạy, không commit. Gate CLI từ chối smoke subset,
nhãn không khớp qrels, dữ liệu ngoài dev hoặc lẫn score/artifact.

Coverage 70% là cấu hình thử nghiệm. Risk đo được vẫn cao; đọc
[VERIFICATION_v0.4.md](VERIFICATION_v0.4.md) trước khi mô tả chất lượng. Chưa chạy
retrieval held-out, fine-tuning R4/R5, generation, cache generation hay TTFT LLM.

Repo GitHub hiện public: https://github.com/lamgiatran/SupportLens. Chia việc dùng
[TEAM_HANDOFF.md](TEAM_HANDOFF.md) và [bảng công việc A–D](../issues/README.md).
