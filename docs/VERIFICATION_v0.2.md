# Kết quả kiểm tra v0.2 — 16/11/2026

- Baseline trước sửa: 22 test pass, có tokenizer MiniLM thật trên máy.
- Bộ regression sau sửa: 40 test pass, gồm query, UI decoder, log 200/422/503/500,
  history, citation aliases, cache/live alignment, split guards và smoke fail/pass.
- Kiểm tra nguồn Banking thật: 12.221 mẫu, 50 nhãn ID, 5.903 ID train hợp lệ.
  Có 8 câu rỗng; giữ nguyên IDs và cờ hợp lệ, không âm thầm bỏ mẫu khỏi nguồn.
- Chưa sinh được vector MiniLM thật trong phiên này. Môi trường cũ kế thừa thư
  viện hệ thống: import Sentence Transformers gặp Windows native access violation
  ở PyArrow; nhánh Transformers trực tiếp còn gặp proxy thiếu `socksio`.
  Không tạo vector giả và không công bố thời gian encode/model-quality từ mock.
- Test cache/live dùng encoder giả xác định được, chỉ kiểm tra alignment và cùng
  preprocessing; đó không phải một phép đánh giá embedding MiniLM thật.
- Chưa train LR, chưa có kết quả H1–H4, chưa phục vụ generation đầy đủ. Banking
  mặc định vẫn 503 đến khi Cầu cung cấp head/loader thật.
- Qrels strict/lenient, retrieval score/gate và deadline tổng vẫn chưa sửa.
  Kiểm tra alias chỉ xác nhận ID; không chứng minh nguồn suy ra câu trả lời.

Đã chuẩn bị code để chạy lại trên môi trường Python sạch theo `RUN_UPDATE.md`.
Source được sửa trong repo local; không tạo remote, không push hay gửi tin nhóm.
