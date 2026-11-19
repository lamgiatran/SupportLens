# SupportLens Bảng công việc

Làm theo [phân công hiện hành](../docs/TEAM_HANDOFF.md): A → B → C → D gom
khoảng 2–3 phiên làm mỗi gói. Chuyển gói khi đạt đầu ra/điều kiện nghiệm thu.
Bảng dưới đây ghi kế hoạch 15 ngày. Mục bàn giao sau bảng ghi tiến độ đã kiểm tra và việc từng người có thể bắt đầu.

| Người | A Chuẩn bị | B Bản đầu để ghép | C Hoàn thiện trên dev | D Đánh giá và bàn giao |
|---|---|---|---|---|
| Hưng | Audit/splits, E1, H-01 | E2/E3 + ablation | So sánh, phân tích lỗi | Test/report/reproduce |
| Cầu | C-01/C-02, split/mask | LR/H1 + pipeline nhận câu hỏi giao Tuấn | H2–H4, score/ngưỡng dev giao Tuấn | Test, OOS analysis, report |
| Tuấn | UI/contract, Docs API thật | Nối Banking, warm-up, lỗi | Nối stream, state, chuyển người | 40 smoke, latency, hướng dẫn |
| Leader | Nhận audit; retrieval đã có | Generator/citation/cache/deadline | Nghiệm thu; cân nhắc R4/R5 | Held-out, số đo và báo cáo |

## Bàn giao sau chặng A ngày 20/11/2026

Lúc kiểm tra, `main` ở `77517df`. Hưng và Cầu có PR đang mở. Tuấn đã đẩy nhánh UI. Các mốc sau là đề xuất nội bộ theo giờ Việt Nam, với quỹ khoảng một giờ mỗi người mỗi ngày.

**1. Hưng → Lâm · Sang B**

CSV khớp 20 ca H-01 đã giao, đủ nhận xét. Chạy lại E1 trên 1.999 câu dev cho Accuracy 0,870935 và Macro-F1 0,870725. [PR #1](https://github.com/lamgiatran/SupportLens/pull/1)

Trước khi ghép PR, giữ cho lệnh test của nhóm chạy được khi máy chỉ cài `[dev]`. Khi bật lemmatization mà thiếu WordNet, báo rõ để người chạy cài dữ liệu. Trên cùng split E1, so E2 (TF-IDF + LR), E3 (TF-IDF + Linear SVM) và thử bật/tắt lemmatization với một cấu hình cố định. **Trước 23:59 ngày 22/11**, giao Lâm bảng Accuracy/Macro-F1 trên dev và lệnh tái lập để ghép báo cáo.

**2. Cầu → Lâm, sau đó → Tuấn · Hoàn tất A trước khi fit**

Cầu đã kiểm tra raw và tìm đúng cặp mâu thuẫn nhãn. Đối chiếu theo tokenizer MiniLM và cache Handoff-50 cho thấy PR khớp 6/8 cặp trùng khác split. Hai cặp có `id_oos/train` thuộc phần dữ liệu ngoài cache. [PR #2](https://github.com/lamgiatran/SupportLens/pull/2)

Trong **20/11**, sửa C-01 để tìm trùng từ token ID thật của các hàng hợp lệ. Kiểm tra thêm `id/train/5823 ↔ id/test/1994` và `id/valid/1091 ↔ id/test/1415`. Giữ ID gốc, đánh dấu bản sao train/dev để loại khi fit hoặc chọn cấu hình/ngưỡng. **Trước 23:59 ngày 21/11**, giao Lâm split/mask C-02 và lệnh kiểm tra. Tách khoảng 10% ID train hợp lệ làm calibration (seed 42), giữ nhóm trùng cùng tập, đủ 50 nhãn trong fit và đúng thứ tự vector. Khi Lâm duyệt, bắt đầu LR/H1 rồi giao Tuấn model, hàm dự đoán và 3 ví dụ.

**3. Tuấn → Cầu và Lâm · Bắt đầu T-01 bằng model giả lập**

Nhánh UI qua 75 test. UI gọi API local và hiện 5 đoạn MultiDoc2Dial. Mock stream đã có hoàn thành, lỗi và hủy. [Commit UI](https://github.com/lamgiatran/SupportLens/commit/203d0cbcc50131b91a0bf3e3ae9b650437515c09)

Trong **20/11**, mở PR từ `feat/tuan-ui-mock-stream`, ghi lệnh chạy và 3 ca minh họa. Nhờ Cầu xác nhận dạng phản hồi H1. **Trước 23:59 ngày 22/11**, giao Lâm test T-01 cho thấy model chỉ nạp một lần và API báo 503 khi thiếu model. Khi nhận H1 thật, chạy cùng 3 câu qua hàm của Cầu và qua API, rồi gửi kết quả đối chiếu cho Cầu và Lâm.

**4. Lâm** nhận CSV H-01, review PR của Hưng và Tuấn, kiểm tra split của Cầu. Song song, làm generator Docs-QA chạy local với citation, cache và giới hạn thời gian sinh. Giao Tuấn ví dụ sự kiện stream để nối ở chặng C.

## Ticket chi tiết dùng ngay

- [Nhóm clone và chạy test](ONBOARDING.md) — mỗi bạn làm trong phiên đầu tiên.
- [C-01 Kiểm tra trùng lặp](C-01-DUPLICATES.md) — Cầu.
- [C-02 Tách calibration](C-02-CALIBRATION.md) — Cầu, sau C-01.
- [H-01 Kiểm tra qrels](H-01-QRELS-REVIEW.md) — Hưng, làm sớm song song baseline.
- [H-02 Baseline E1](H-02-COURSE77-E1.md) — Hưng.
- [T-02 UI hai chế độ và mock stream](T-02-UI-MOCK-STREAM.md) — Tuấn.
- [T-01 Nạp và warm-up Banking](T-01-WARMUP.md) — Tuấn, có thể test mock trước khi nhận pipeline.

Các ticket D01/D02/D03 lưu làm lịch sử chuẩn bị; không dùng chúng để giao việc
song song với bảng này. File ticket chưa tự tạo GitHub issue. Khi mở issue/PR,
liên kết gói tương ứng và ghi trạng thái thực tế.

Một lần giao: commit/PR, cách kiểm tra và ba ví dụ. Người nhận báo pass hoặc lỗi
tái hiện; owner sửa module mình. Số đo
cuối lấy từ phiên bản freeze sau chọn cấu hình bằng dev.
