# Kết quả kiểm tra v0.2 — 16/11/2026

- Baseline trước sửa: 22 test pass, có tokenizer MiniLM thật trên máy.
- Bộ regression sau sửa: 40 test pass, gồm query, UI decoder, log 200/422/503/500,
  history, citation aliases, cache/live alignment, split guards và smoke fail/pass.
- Kiểm tra nguồn Banking thật: 12.221 mẫu, 50 nhãn ID, 5.903 ID train hợp lệ.
  Có 8 câu rỗng; giữ nguyên IDs và cờ hợp lệ, không âm thầm bỏ mẫu khỏi nguồn.
- Đã sinh cache MiniLM thật trên CPU: ma trận 12.221 × 384 float32, trong đó
  12.213 hàng có embedding; 8 hàng invalid dùng zero sentinel có cờ rõ ràng.
  ZIP có cache; Git chỉ giữ split IDs, manifest/hash và kết quả audit.
- Dùng môi trường Python sạch và cache trong thư mục được phép ghi để khắc phục
  các lỗi thư viện/cache của môi trường cũ. Phiên bản thư viện ở embedding manifest.
- Unit test cache/live dùng test double để kiểm tra logic. Audit bổ sung với
  model MiniLM thật đối chiếu 7 mẫu đại diện cho 7 nhóm split; kết quả chi tiết
  ở `runs/handoff50/cache_audit.json`. Đây không phải benchmark chất lượng model.
- Giải nén bản source để chạy lại: 39 passed, 1 skipped; test skip cần tokenizer
  tải riêng. Trên thư mục làm việc có tokenizer thật: 40 passed.
- Chưa train LR, chưa có kết quả H1–H4, chưa phục vụ generation đầy đủ. Banking
  mặc định vẫn 503 đến khi Cầu cung cấp head/loader thật.
- Qrels strict/lenient, retrieval score/gate và deadline tổng vẫn chưa sửa.
  Kiểm tra alias chỉ xác nhận ID; không chứng minh nguồn suy ra câu trả lời.

Đã chuẩn bị code để chạy lại trên môi trường Python sạch theo `RUN_UPDATE.md`.
Khung code và bản vá v0.2 có hỗ trợ của trợ lý AI; đóng góp của từng owner là
phần họ thực sự kiểm tra, mở rộng và hoàn thiện tiếp.
Source được sửa trong repo local; không tạo remote, không push hay gửi tin nhóm.
