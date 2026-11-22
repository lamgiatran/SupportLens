# Chạy local và review 3 demo UI (T-02)

Hướng dẫn cho nhánh `feat/tuan-ui-mock-stream`. Mở các terminal PowerShell tại
thư mục `SupportLens` (máy hiện tại: `D:\261\NLP\BTL\SupportLens`). Dùng Python
3.11 trở lên. Ba demo gồm Docs-QA thật, Banking thiếu model và mock stream.

Docs-QA hiện trả các đoạn evidence, chưa sinh câu trả lời bằng LLM. Banking
chưa được nạp pipeline thật trong entry point hiện tại. Mock chỉ phục vụ review
giao diện; không phải kết quả phân loại hay stream từ generator thật.

## 1. Chuẩn bị môi trường

Nếu chưa có `.venv`, tạo một lần:

```powershell
python -m venv .venv
```

Cài dependencies (không cần activate môi trường):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
```

Demo Docs-QA dùng R1/BM25, không cần torch, encoder Banking hoặc Ollama. Cần
bundle tại `data/processed/docs_qa` và tokenizer tại
`data/tokenizers/minilm/tokenizer.json`. Nếu chưa có, tải/tạo một lần:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset multidoc2dial
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py
.\.venv\Scripts\python.exe scripts/build_docs_bundle.py
```

Nếu chỉ review mock stream (demo 3), bỏ qua phần chuẩn bị dữ liệu và chạy UI;
không cần khởi động API. Sidebar có thể báo chưa kết nối API, mock vẫn chạy.

## 2. Khởi động API và UI

**Terminal 1 — API R1:** xóa cấu hình dense/gate có thể còn từ phiên trước để
demo không phụ thuộc weights R2/R3 hoặc ngưỡng đã fit.

```powershell
$env:SUPPORTLENS_DOCS_BUNDLE = 'data/processed/docs_qa'
$env:SUPPORTLENS_DOCS_TOKENIZER = 'data/tokenizers/minilm/tokenizer.json'
$env:SUPPORTLENS_BANKING_MODE = 'disabled'
Remove-Item Env:SUPPORTLENS_DOCS_DENSE -ErrorAction SilentlyContinue
Remove-Item Env:SUPPORTLENS_DOCS_GATE -ErrorAction SilentlyContinue
Remove-Item Env:SUPPORTLENS_DOCS_GATES -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Chờ log `Application startup complete`. Giữ terminal này chạy.

**Terminal 2 — UI:**

```powershell
$env:SUPPORTLENS_API = 'http://127.0.0.1:8000'
.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Mở [UI local](http://127.0.0.1:8501). API có
[Swagger](http://127.0.0.1:8000/docs) và [health](http://127.0.0.1:8000/health).
Với cấu hình trên, health cần có `docs_methods: ["R1"]`,
`docs_retrieval_ready: true`, `banking_model_ready: false` và
`docs_generator_ready: false`.

## Demo 1 — Docs-QA gọi API thật, history và reset

1. Chọn **Hỏi tài liệu** ở sidebar, **API thật** ở phần cách nhận phản hồi.
2. Chọn **R1 — BM25 · Từ khóa**. Giữ **Gửi ngữ cảnh hội thoại trước đó** bật.
3. Nhập `My insurance ended so what should I do?`, bấm **Gửi câu hỏi**.
4. Xem **Tài liệu tham khảo**, mở một đoạn nguồn và **Chi tiết phản hồi API**.
5. Nhập tiếp `What should I do next?`, gửi và mở **Request đã gửi**: `history`
   cần chứa câu hỏi đầu. Các đoạn truy xuất không được ghi thành câu trả lời
   assistant trong history.
6. Bấm **＋ Hội thoại mới**. Câu hỏi, kết quả và lịch sử của chế độ này phải
   được xóa. Gửi lại một câu hỏi thì `history` bắt đầu từ danh sách rỗng.

**Kết quả mong đợi:** request thành công; JSON có `mode: "docs_qa"`,
`decision: "retrieved"`, evidence với title/text/ID và `answer: null`.
UI thông báo đã tìm tài liệu, chức năng tạo câu trả lời chưa được kết nối.
Không dùng demo này để khẳng định độ đúng của câu trả lời LLM.

R2/R3 sẽ có nhãn chưa nạp và nút gửi bị vô hiệu khi chọn chúng. Muốn review
R2/R3 thật, chuẩn bị encoder/cache theo [RUN_DENSE.md](RUN_DENSE.md).

## Demo 2 — Banking qua API thật trả 503 khi thiếu model

1. Từ demo 1, chuyển sang **Yêu cầu ngân hàng**; giữ **API thật**.
2. Chọn phương pháp **H1**, nhập `My card payment was declined.` rồi gửi.
3. Mở **Chi tiết lỗi** và **Request đã gửi**.

**Kết quả mong đợi:** sidebar báo Banking chưa nạp model; request có
`mode: "banking_triage"`, `model_scope: "handoff50"`, `score_method: "H1"`
và `history: []`. API trả **HTTP 503**, mã lỗi
`handoff50_artifact_missing`. UI hiển thị lỗi, không hiện nhãn dự đoán giả hay
evidence/kết quả Docs-QA của demo trước.

Có thể đối chiếu trực tiếp bằng Swagger: gọi `POST /v1/assist` với body:

```json
{
  "mode": "banking_triage",
  "model_scope": "handoff50",
  "question": "My card payment was declined.",
  "score_method": "H1",
  "history": []
}
```

Response body mong đợi:

```json
{
  "detail": {
    "code": "handoff50_artifact_missing",
    "message": "Train and load the separate 50-label model first."
  }
}
```

Ca này kiểm tra đường báo thiếu model. Để thử nạp một lần/warm-up bằng mock
và sau đó nối pipeline H1 thật từ Cầu, xem [hướng dẫn T-01](RUN_BANKING_SERVING.md).

## Demo 3 — Mock stream hoàn thành, lỗi và hủy

Chọn **Hỏi tài liệu** hoặc **Yêu cầu ngân hàng**, rồi chọn **Mock stream · Demo**.
Phải thấy cảnh báo **DEMO GIẢ LẬP**. Nhập câu hỏi bất kỳ, chẳng hạn
`My card payment was declined.`. Thử lần lượt ba thao tác sau:

| Thao tác | Kết quả mong đợi |
|---|---|
| Chọn **Hoàn thành**, bấm **Chạy mock stream** và chờ hết | Văn bản xuất hiện dần, thanh tiến trình đầy, báo **Mock stream đã hoàn thành.**, sự kiện `done`. |
| Chọn **Lỗi giữa chừng**, bấm chạy và chờ | Dừng khoảng giữa nội dung, báo **Lỗi giả lập giữa chừng**, ghi rõ nội dung chưa hoàn tất, sự kiện `error`. |
| Chọn **Hoàn thành**, bấm chạy rồi bấm **Hủy stream** khi đang chạy | Văn bản ngừng tăng, báo **Đã hủy mock stream**, ghi rõ nội dung chưa hoàn tất, sự kiện `cancelled`. |

Mỗi lượt đều có thể chạy lại sau khi kết thúc. Bấm **＋ Hội thoại mới** để xóa
câu hỏi và mock hiện tại; đổi chế độ làm việc cũng dừng và xóa mock đang chạy.
Mock không thêm nội dung mẫu vào history gửi cho API. Các sự kiện ở bảng là
quy ước demo local, chưa phải contract generator thật được bàn giao.

## Kiểm tra tự động và ghi kết quả review

Chạy các test UI/client/mock hiện có, không cần mạng, model hoặc API đang chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ui_workflow.py -q
```

Test dùng API với fixture và Streamlit AppTest; vẫn cần chạy demo 1 thủ công
để xác nhận bundle local và kết nối giữa hai tiến trình. Khi gửi Leader/Cầu,
kèm commit đang review (`git rev-parse --short HEAD`) và ghi actual/pass/fail:

| Ca | Bằng chứng nên ghi |
|---|---|
| Demo 1 | Request/response Docs-QA, evidence, history trước/sau reset. |
| Demo 2 | Request Banking, HTTP 503 và mã `handoff50_artifact_missing`. |
| Demo 3 | Ảnh hoặc video ngắn cho `done`, `error`, `cancelled`. |

Nếu không kết nối được API, kiểm tra terminal 1 và địa chỉ `SUPPORTLENS_API`,
rồi bấm **Kiểm tra lại kết nối**. Nếu API báo thiếu bundle/tokenizer, hoàn
thành bước chuẩn bị dữ liệu trước khi chạy lại. Dừng API/UI bằng `Ctrl+C`
trong từng terminal.
