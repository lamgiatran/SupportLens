# SupportLens Bảng công việc

Làm theo [phân công hiện hành](../docs/TEAM_HANDOFF.md): A → B → C → D gom
khoảng 2–3 phiên làm mỗi gói. Chuyển gói khi đạt đầu ra/điều kiện nghiệm thu.
Bảng này là kế hoạch giao việc, chưa xác nhận thành viên đã thực hiện.

| Người | A Chuẩn bị | B Bản đầu để ghép | C Hoàn thiện trên dev | D Đánh giá và bàn giao |
|---|---|---|---|---|
| Hưng | Audit/splits, E1, H-01 | E2/E3 + ablation | So sánh, phân tích lỗi | Test/report/reproduce |
| Cầu | C-01/C-02, split/mask | LR/H1 + pipeline nhận câu hỏi giao Tuấn | H2–H4, score/ngưỡng dev giao Tuấn | Test, OOS analysis, report |
| Tuấn | UI/contract, Docs API thật | Nối Banking, warm-up, lỗi | Nối stream, state, chuyển người | 40 smoke, latency, hướng dẫn |
| Leader | Nhận audit; retrieval đã có | Generator/citation/cache/deadline | Nghiệm thu; cân nhắc R4/R5 | Held-out, số đo và báo cáo |

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
tái hiện; owner sửa module mình. Đổi phạm vi tại mốc ngày 2, 5 hoặc 8. Số đo
cuối lấy từ phiên bản freeze sau chọn cấu hình bằng dev.
