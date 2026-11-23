# Ghi nhận trợ lý AI

## Gói B Docs-QA generator — 24/11/2026

Claude (Anthropic) hỗ trợ viết adapter generator Ollama (kiểm tra nguồn trích,
cache, deadline, huỷ stream), nối vào API (`decision="answer"`, endpoint
`/v1/assist/stream`), test với Ollama giả lập, script đo dev và hướng dẫn
RUN_GENERATION.md. Lâm chọn phạm vi, review code và chạy với model thật; số đo
generation chỉ lấy từ lần chạy thật trên máy demo, không có số nào được điền sẵn.

## T-01 Banking serving và hướng dẫn demo — 21/11/2026

Codex hỗ trợ triển khai lifecycle nạp/warm-up Banking, khóa khởi tạo encoder,
readiness/503, timing/log, model giả lập có nhãn rõ ràng, test offline và script
đối chiếu standalone/API. Codex cũng hỗ trợ viết hướng dẫn chạy ba demo UI và
chuyển từ mock sang pipeline thật của Cầu. Chưa dùng model H1 thật để nghiệm thu;
model, encoder/head, ngưỡng và kết quả thực nghiệm vẫn do Cầu bàn giao.
