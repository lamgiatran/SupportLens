# T-01 — nạp/warm-up Banking và thay mock bằng pipeline của Cầu

Phần serving có thể kiểm tra ngay bằng model giả lập, không cần weights hoặc
head từ Cầu. Factory tạo pipeline một lần trong lifespan; API nạp backend,
chạy một dự đoán H1 để warm-up cả encoder/head, rồi mới báo Banking ready.
Request người dùng chỉ dùng lại pipeline đã sẵn sàng.

## 1. Cài môi trường và chạy test

Mở PowerShell tại `SupportLens` (máy hiện tại:
`D:\261\NLP\BTL\SupportLens`). Dùng Python >= 3.11. Nếu chưa có `.venv`:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\python.exe -m pytest tests/test_banking_serving.py tests/test_banking_comparison.py tests/test_banking_and_handoff.py -q
```

Không cần cài torch/sentence-transformers cho test hoặc mock. Test kiểm tra
khởi tạo đồng thời, encoder chỉ nạp một lần, warm-up trước ready, startup lỗi,
thiếu model, không retry trong request, log lỗi có request ID, và đối chiếu
standalone/API. Test sử dụng socket local cho FastAPI/Streamlit; nếu môi trường
sandbox chặn socket, chạy trong terminal local bình thường.

## 2. Chạy API với model giả lập

**Terminal 1:**

```powershell
$env:SUPPORTLENS_BANKING_MODE = 'mock'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Nếu biến Docs-QA từ phiên trước trỏ tới assets không còn tồn tại, xóa các biến
`SUPPORTLENS_DOCS_BUNDLE`, `SUPPORTLENS_CORPUS`, `SUPPORTLENS_DOCS_DENSE`,
`SUPPORTLENS_DOCS_GATE`, `SUPPORTLENS_DOCS_GATES` trong terminal này trước khi
chạy. Banking mock không cần dữ liệu Docs-QA. Muốn chạy cả Docs-QA, giữ cấu hình
R1 theo [RUN_UI_DEMOS.md](RUN_UI_DEMOS.md).

**Terminal 2 — kiểm tra:**

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:8000/ready/banking | ConvertTo-Json -Depth 6
$body = @{
    mode = 'banking_triage'
    model_scope = 'handoff50'
    question = 'My card payment was declined.'
    score_method = 'H1'
    history = @()
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/assist -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

Kết quả cần thấy:

- `/health`: `banking_model_ready: true`; `banking_serving.state: "ready"`,
  `is_mock: true` và các thời gian `load_ms`, `warmup_ms`, `cold_start_ms`.
- `/ready/banking`: HTTP 200 sau khi warm-up thành công.
- `/v1/assist`: artifact `MOCK-banking-serving-v1`, nhãn `MOCK_INTENT`,
  `decision: "handoff"`, reason có `mock_development_only`. Đây là dữ liệu giả
  để thử serving; không phải kết quả nhận diện intent.
- Gửi lại request: các thời gian startup giữ nguyên; latency trong response
  chỉ đo xử lý request sau ready, không gộp cold start.

Khi chạy bằng Uvicorn mặc định, terminal API cũng ghi một dòng `Banking startup:`
chứa `load_ms`, `warmup_ms`, `cold_start_ms`. Logger startup dùng mức log của
Uvicorn; nếu chủ động đặt `--log-level warning`, dòng INFO này sẽ được ẩn.

Muốn xem trên UI, trong terminal 2 chạy:

```powershell
$env:SUPPORTLENS_API = 'http://127.0.0.1:8000'
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Mở [UI](http://localhost:8501), chọn **Yêu cầu ngân hàng → API thật → H1**.
Sidebar cảnh báo API đang dùng MOCK. **API thật** ở đây nghĩa là UI có gửi HTTP
đến API; backend được cấu hình rõ ràng là model giả lập. Tùy chọn **Mock stream ·
Demo** của T-02 là mô phỏng giao diện riêng, không gọi API hoặc runtime này.

## 3. Kiểm tra thiếu model và startup lỗi

Dừng API bằng `Ctrl+C`, sau đó chạy với Banking bị tắt:

```powershell
$env:SUPPORTLENS_BANKING_MODE = 'disabled'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

`banking_model_ready` phải false; `POST /v1/assist` cho Banking và
`GET /ready/banking` trả 503, mã `handoff50_artifact_missing`. Mặc định khi không
đặt `SUPPORTLENS_BANKING_MODE` cũng là `disabled`; không tự chuyển sang mock.

Khi cấu hình `real` mà thiếu weights/factory, load lỗi hoặc warm-up lỗi, API
vẫn chạy để phục vụ Docs-QA và cho phép xem lỗi. Banking có state `failed`,
`error_type`, readiness 503 và request trả mã `banking_startup_failed`.
Log startup có traceback; log request có `request_id`, `error_code`,
`error_type`, `banking_state` và HTTP 503. Request ID cũng nằm trong header
`X-Request-ID`. Log JSONL mặc định: `runs/api_requests.jsonl`.

Sửa cấu hình/assets rồi **khởi động lại API**. Request không tự tải weights
hoặc thử khởi tạo lại. `/health` là thông tin/liveness chung, vẫn HTTP 200 khi
Banking lỗi; dùng `/ready/banking` để kiểm tra readiness riêng của Banking.

## 4. Khi Cầu bàn giao: cần nhận những gì?

Nhận pipeline **câu hỏi → encoder → head → nhãn/score/quyết định**, gồm:

- Code và dependencies, hàm load, cách gọi `predict()` độc lập.
- Weights encoder đã tải local, classifier/head, mapping 50 nhãn, cấu hình
  encoder đúng fingerprint và threshold H1/version đã chọn.
- Ba câu mẫu cùng output dự kiến, ghi rõ phiên bản artifact/ngưỡng.

Chỉ có file weights hoặc embeddings chưa đủ để nối API. Tuấn không viết lại
encoder/classifier hoặc tự chọn ngưỡng; dùng đúng pipeline Cầu bàn giao.

## 5. Nối factory thật — giữ nguyên API/UI và lifecycle

Đặt artifact ở một thư mục local, ví dụ `artifacts/handoff50/h1/`. Cài thêm
dependencies Banking và các dependencies Cầu yêu cầu:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[banking]"
```

Factory là một hàm được gọi **một lần lúc startup**, với chữ ký:

```python
def load_pipeline(*, artifact_dir, local_files_only):
    ...
    return pipeline
```

`artifact_dir` là `pathlib.Path`. Runtime luôn truyền `local_files_only=True`.
Factory phải dùng assets đã chuẩn bị trên máy; không tự tải weights.
Pipeline trả về cần:

- `predict(question, *, method="H1")` trả đối tượng có các thuộc tính
  `predicted_intent`, `score`, `decision`, `reason`, `artifact_version`, đúng
  `BankingPrediction` trong `src/supportlens/banking.py`.
- `decision` là `accepted` hoặc `handoff`; score hữu hạn và phiên bản thật.
- Có thể cung cấp `load()` để nạp encoder/head trước warm-up. Nếu không có,
  factory phải nạp sẵn weights; dự đoán warm-up vẫn chạy trước ready.

`BankingPipeline` hiện có đã cung cấp `load()`. Khi dùng encoder thật, tạo
`BankingEncoder(..., local_files_only=True)`; serving từ chối backend chưa nạp
nếu nó còn cho phép download. Lock của encoder bảo vệ cả đường lazy dùng ở
nơi khác; phần tạo cache vẫn giữ chính sách download riêng.

Nếu Cầu đã cung cấp factory đúng chữ ký, dùng thẳng `module:function` của Cầu.
Nếu chữ ký khác, tạo adapter nhỏ, ví dụ `src/supportlens/cau_adapter.py`:

```python
# Minh họa: thay tên import và tham số bằng API Cầu thực sự bàn giao.
from cau_pipeline import load_pipeline as load_cau_pipeline

def load_pipeline(*, artifact_dir, local_files_only=True):
    return load_cau_pipeline(
        artifact_dir=artifact_dir,
        local_files_only=local_files_only,
    )
```

Đoạn trên là mẫu adapter, chưa phải loader cho một định dạng model đã biết.
Nếu Cầu trả dict thay vì `BankingPrediction`, adapter chuyển đúng các trường
được bàn giao sang `BankingPrediction`; giữ nguyên nhãn/score/ngưỡng.

**Dừng API mock**, đặt cấu hình thật rồi chạy lại:

```powershell
$env:SUPPORTLENS_BANKING_MODE = 'real'
$env:SUPPORTLENS_BANKING_ARTIFACTS = 'artifacts/handoff50/h1'
# Dùng entry point thật của Cầu hoặc adapter bạn vừa tạo:
$env:SUPPORTLENS_BANKING_FACTORY = 'supportlens.cau_adapter:load_pipeline'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Kiểm tra `/health` và `/ready/banking`: ready true, `is_mock: false`, startup
không lỗi. Gửi một request H1 và kiểm tra `artifact_version` đúng bản Cầu giao,
không còn tiền tố `MOCK-`. Trong UI bấm **Kiểm tra lại kết nối** để bỏ cache health.
Sai factory hoặc thiếu assets sẽ báo lỗi/503; không có cơ chế fallback sang mock.

## 6. Đối chiếu ba câu mẫu giữa predict và API

Lưu **đúng ba câu Cầu cung cấp** vào JSONL, mỗi dòng có dạng:

```json
{"input": {"question": "Câu mẫu thực tế từ Cầu"}}
```

Ví dụ lưu tại `artifacts/handoff50/h1/smoke_samples.jsonl`. Có thể dùng file
bàn giao sẵn nếu các dòng cũng có `input.question`; script chỉ dùng câu hỏi,
không dùng trường `expected` có thể đi kèm.

Giữ API thật chạy. Mở terminal khác, đặt lại **cùng ba biến Banking** ở bước 5
(biến PowerShell không tự truyền giữa hai terminal đã mở), rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts/check_banking_api.py --samples artifacts/handoff50/h1/smoke_samples.jsonl --api http://127.0.0.1:8000 --method H1
```

Script nạp pipeline thật độc lập, warm-up, rồi so với API trên từng câu:
`predicted_intent`, `score`, `decision`, `reason`, `artifact_version`,
`score_method`. Score dùng tolerance 1e-6; không so request ID/latency.
API mock, model chưa ready hoặc khác phiên bản đều không được tính pass.
Kết quả in JSON, `status: "pass"` khi cả ba khớp; exit code 1 nếu có lỗi/lệch.
Đối chiếu thêm với output dự kiến Cầu gửi để xác nhận đúng bản bàn giao.

Nếu lệch, kiểm tra cùng weights, mapping, encoder revision/fingerprint,
preprocessing, H1 và threshold; không chỉnh output API để ép khớp.

## 7. Phạm vi nghiệm thu hiện tại

T-01 serving được kiểm chứng bằng double offline: lifecycle/load lock,
readiness/503, timing và request log. Chưa có kết quả với model H1 thật của Cầu;
chốt phần tích hợp sau khi thực hiện bước 4–6. Một model được nạp một lần
**trong mỗi process**; nếu chạy nhiều worker thì mỗi worker có bản model riêng.
Lệnh demo dùng một worker và không bật `--reload`.
