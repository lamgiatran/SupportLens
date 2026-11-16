# Giao việc từ bản nền hiện tại

Mỗi bạn có khoảng 1 giờ/ngày. Gộp việc thành chặng 2–3 phiên; chuyển chặng khi
có đầu ra chạy được, không phải vì đã sang một ngày mới. Kế hoạch 15 ngày gốc
vẫn giữ nguyên. Code nền do trợ lý chuẩn bị không tính là đóng góp đã làm của
người được ghi tên owner.

## Ai sở hữu phần nào?

| Người | Nhận gì để bắt đầu? | Việc tự làm và đầu ra | Bàn giao ai, để làm bước nào? |
|---|---|---|---|
| Leader | Raw MultiDoc2Dial và corpus đã chốt | Chunks/qrels, R1/R2/R3, Docs gate, generator/cache và thực nghiệm retriever; adapter Docs vào API | Giao Tuấn API cùng ví dụ request/response để bạn ấy làm UI và streaming; nhận kiểm tra qrels từ Hưng trước khi dựng train pairs R4 |
| Hưng | BANKING77 Course-77, split/config; file audit 20 qrels | Tự làm preprocessing, BoW/TF-IDF + LR/SVM, ablation và bảng metrics/confusion; review H-01 bằng mắt | Giao code + metrics cho Leader ghép báo cáo. Giao nhận xét span/chunk cho Leader để sửa dữ liệu train retriever nếu cần |
| Cầu | Cache Banking cùng thứ tự sample IDs, raw data, C-01/C-02 | Tự audit trùng, tạo calibration; chọn C, fit LR 50 nhãn, H1–H4, ngưỡng dev và phân tích OOS; xuất pipeline predict() | Giao Tuấn bản H1 đầu tiên chạy được để thay mock bằng model thật; giao H1–H4 và gate cuối để Tuấn kiểm tra accept/handoff trước demo. Giao metrics cho Leader ghép báo cáo |
| Tuấn | API Docs đã có R1/R2/R3; scaffold UI chỉ có câu hỏi đơn/R1; contract Banking | Tự làm UI chọn retriever/history, lưu hội thoại, nối stream, adapter và startup Banking, xử lý lỗi/hủy; smoke toàn luồng + hướng dẫn chạy | Giao Leader/Cầu commit và kết quả smoke để xác nhận module đã ghép đúng; lỗi retrieval/generator gửi Leader, lỗi nhãn/confidence Banking gửi Cầu |

## Chặng bắt đầu: 2–3 phiên có thể làm song song

- **Cầu:** làm [C-01](../issues/C-01-DUPLICATES.md), rồi
  [C-02](../issues/C-02-CALIBRATION.md). Đầu ra là audit, quy tắc xử lý, mask
  fit/calibration và script tái tạo. Giao Leader review trước khi fit head.
- **Hưng:** bắt đầu preprocessing + các baseline Course-77. Trong một phiên,
  làm [H-01](../issues/H-01-QRELS-REVIEW.md); không cần chờ Cầu hay Tuấn.
- **Tuấn:** dùng API Docs theo [RUN_DENSE.md](RUN_DENSE.md), tự thêm chọn
  Keywords/R1, Semantic/R2, Hybrid/R3 và history vào UI. Giữ mock Banking trong
  test để làm tiếp khi Cầu chưa giao head; chạy thật phải báo 503 nếu thiếu head.
- **Leader:** chuẩn bị generator/cache/deadline và tiếp nhận hai audit. Không
  làm thay C-01/C-02 hoặc baseline Course-77.

## Bản predict đầu tiên của Cầu dùng để làm gì?

Sau khi dữ liệu và C đã chốt, Cầu giao bản LR + H1 có metadata/encoder fingerprint
và ngưỡng dev đầu tiên, kèm vài input/output smoke. Tuấn dùng đúng predict()
đó trong endpoint Banking để thử câu hỏi thật, nhãn, latency, lỗi tải model và
hiển thị accept/handoff. Đây là bước bỏ mock trong chặng tích hợp.

Cầu tiếp tục T/calibration, H2–H4 và phân tích lỗi. Khi giao gate/head cuối, Tuấn
chỉ thay artifact/config đã version, rồi chạy lại smoke. Tuấn không viết một
encoder hoặc cách tính confidence khác với Cầu. Startup/lock Banking vẫn là
[T-01](../issues/T-01-WARMUP.md); warm-up encoder Docs đã được Leader chuẩn bị.

## Khi nào gọi là bàn giao xong?

Owner commit code + config/metrics nhỏ, ghi một lệnh smoke và kết quả trong issue/PR.
Người nhận chạy pass là **đã nhận**; API/UI dùng được module thật là **đã ghép**.
Không cần thêm README/manifest viết tay và bảng trạng thái cho mỗi lần gửi.

Vectors, indexes, weights và raw data không vào Git; dùng gói runtime/link một lần.
Giữ sample IDs và các cờ loại khỏi fit. Tám câu Banking rỗng vẫn phải được đếm;
vector zero là sentinel, không phải embedding dùng để train hoặc tìm láng giềng.
