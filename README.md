# SupportLens

SupportLens là dự án NLP xây dựng ứng dụng hỗ trợ khách hàng với hai chế độ:
tra cứu tài liệu có nguồn và phân luồng yêu cầu ngân hàng theo độ tin cậy.
Dự án kết hợp thực nghiệm mô hình với API, giao diện và kiểm tra toàn luồng.

## Mục tiêu sản phẩm

| Chế độ | Luồng sử dụng hướng tới | Cách đánh giá |
|---|---|---|
| Docs-QA | Câu hỏi và lịch sử → tìm bằng chứng → trả lời có trích dẫn hoặc yêu cầu hỗ trợ thêm | Truy xuất trên MultiDoc2Dial; kiểm tra bằng chứng trong context và citation |
| Banking | Yêu cầu → dự đoán intent → chấp nhận hoặc chuyển người theo ngưỡng | Handoff-50: độ chính xác, risk–coverage và khả năng từ chối yêu cầu ngoài phạm vi |

Hai chế độ dùng chung API, giao diện và request log. Người dùng chọn chế độ;
classifier ngân hàng không được dùng để lọc câu hỏi tài liệu.

## Thiết kế

- **Retrieval:** BM25, dense MiniLM và hybrid RRF trên cùng corpus/chunks.
- **Banking:** một pipeline encoding và dự đoán, dùng frozen encoder cùng Logistic Regression 50 nhãn.
- **Ứng dụng:** FastAPI và Streamlit; mô hình và cấu hình được nạp theo phiên bản.
- **Generation:** hướng phát triển dùng Ollama, trả lời theo bằng chứng và lưu kết quả để đánh giá lại.
- **Thực nghiệm môn học:** BoW/TF-IDF với Logistic Regression và Linear SVM trên BANKING77 77 nhãn.

## Phát triển cùng nhóm

| Thành viên | Module phụ trách | Đầu ra chính |
|---|---|---|
| Leader | Docs-QA và thực nghiệm retriever | Pipeline tài liệu, generator/cache, đánh giá; fine-tune retriever khi luồng lõi ổn |
| Hưng | Dữ liệu và baseline Course-77 | Pipeline preprocessing/train/evaluate, bảng so sánh và phân tích lỗi |
| Cầu | Banking Handoff-50 | Pipeline encode/predict, calibration, confidence và cấu hình chuyển người |
| Tuấn | Serving và giao diện | Ghép module thật vào API/UI, streaming, xử lý lỗi và kiểm tra toàn luồng |

[Gói công việc của từng thành viên](docs/TEAM_HANDOFF.md) ghi đầu vào, hướng làm,
đầu ra và người sử dụng kết quả. [Bảng công việc](issues/README.md) là nơi theo dõi
tiến độ. Mỗi thành viên phát triển phần mình và gửi commit/PR cùng cách kiểm tra.

## Chạy và tái lập

Python 3.11 trở lên. Cài môi trường phát triển:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
```

Tiếp tục theo [hướng dẫn chạy Docs-QA](docs/RUN_DENSE.md) để chuẩn bị dữ liệu,
chọn cấu hình và khởi động API/UI. Weights và các cache lớn được tải hoặc tạo
theo hướng dẫn; split IDs, cấu hình và metrics nhỏ được quản lý trong Git.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Tiến độ và kết quả

Dự án đang phát triển. Nhánh Docs-QA đã có truy xuất R1/R2/R3 và đánh giá dev;
các gói Banking, ứng dụng và generation được hoàn thiện theo bảng công việc.

Kết quả dev hiện tại: strict hit@3 của BM25 / dense / hybrid là
**0.4668 / 0.4062 / 0.4803** trên 2.213 query. Khoảng tin cậy cho chênh lệch
hybrid–BM25 chứa 0. Đọc [thiết lập và kết quả thực nghiệm](docs/VERIFICATION_v0.4.md)
trước khi so sánh hoặc sử dụng số liệu. Đây là số đo retrieval trên dev.

Xem thêm [thiết kế và protocol](docs/plan-v1.0.md), cùng
[ghi nhận công cụ hỗ trợ](docs/AI_USAGE.md).
