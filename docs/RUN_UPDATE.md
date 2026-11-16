# Chạy bản cập nhật v0.2

Mở terminal tại thư mục `SupportLens` sau giải nén. Đây là hướng dẫn chạy code,
không phải một lịch làm việc mới. Source/bundle không chứa model đã train.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Máy đã có tokenizer thật chạy được test đó; máy chưa tải sẽ có một test skip.
UI dùng extra `ui`; Hưng dùng `classical`; Cầu dùng `banking`.
Tránh `--system-site-packages` để lỗi thư viện hệ thống không lọt vào môi trường.

## Chuẩn bị vector cho Cầu

ZIP v0.2 đã có `data/processed/handoff50/embeddings.npz` và manifest.
Cầu có thể dùng ngay cache với numpy/sklearn để fit head. Các lệnh dưới đây
chỉ cần khi muốn tự tạo lại cache; dùng cùng `banking.py` cho cả cache và API.

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset banking77_oos
.\.venv\Scripts\python.exe scripts/build_banking_embeddings.py --audit-only
.\.venv\Scripts\python.exe -m pip install -e ".[banking]"
.\.venv\Scripts\python.exe scripts/build_banking_embeddings.py --device cpu --batch-size 64
```

Lệnh audit chỉ ghi split IDs và `runs/handoff50/data_audit.json`; không sinh vector.
Lệnh cuối mới sinh `data/processed/handoff50/embeddings.npz` cùng manifest/hash.
Nó tải frozen encoder đã ghim revision; không fine-tune, không học trên 77 nhãn.
Cache và live API gọi cùng `BankingEncoder.encode()` trong `banking.py`.
Nếu gặp lỗi quyền ghi cache của Hugging Face, đặt `HF_HOME` vào thư mục được
phép ghi trước khi chạy; xem cấu hình chính thức tại
https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables.

```python
import numpy as np
with np.load("data/processed/handoff50/embeddings.npz", allow_pickle=False) as z:
    fit = z["fit_eligible"]  # chỉ ID train và input hợp lệ
    X_train, y_train = z["embeddings"][fit], z["label"][fit]
```

Cầu chọn C/ngưỡng bằng dev, xuất head + encoder fingerprint; viết wrapper trả
`BankingPrediction` và nạp vào `BankingPipeline`. Tuấn truyền pipeline vào
`create_app(banking=pipeline)`. Chưa có loader/head thì server mặc định vẫn trả
503 ở Banking; đây không phải model đã train sẵn.

## Smoke và log

```powershell
.\.venv\Scripts\python.exe scripts/smoke.py artifacts/handoff50/draft
$env:SUPPORTLENS_LOG = 'runs/api_requests.jsonl'
$env:SUPPORTLENS_LOG_QUESTIONS = '1'
```

Chỉ bật log câu hỏi khi đang dùng dữ liệu công khai. Mặc định không lưu câu hỏi;
middleware vẫn ghi status 200/4xx/5xx và request ID. Log vận hành không vào Git.

`manifest.json`: thêm `"smoke_entrypoint": "supportlens.handoff50:smoke_predict"`
sau khi Cầu tạo hàm đó. Mỗi dòng samples có `input` và `expected` (dict kỳ vọng).
Không có model/hàm thì smoke fail rõ ràng; không dùng output giả để báo cáo.

Qrels strict/lenient, score/gate Docs-QA, deadline tổng của generator, mô hình
50 nhãn và nối generation thật vẫn cần làm tiếp. `timeout=45` hiện chỉ là timeout
I/O của probe, không được công bố là deadline tổng.
