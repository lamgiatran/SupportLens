# Tuấn phối hợp Cầu — warm-up và nạp model một lần

Nhận: pipeline Banking có head thật từ Cầu. Có thể viết test bằng mock trước.
Mục tiêu là hoàn thiện serving, không thay cách train/calibrate của Cầu.

1. Warm-up encoder/head trong lifespan khi API khởi động, trước khi báo ready.
   Chốt nơi chứa weights để request người dùng không tự bắt đầu tải model.
2. Thêm khóa cho khởi tạo lazy trong `BankingEncoder` nếu vẫn giữ đường gọi đó;
   hai caller cùng lúc chỉ nạp một model, không chỉ dựa vào GIL.
3. Ghi riêng load/warm-up/cold-start; request latency sau ready báo riêng.
   Nếu startup thất bại, báo readiness/503 rõ ràng và log lỗi có request ID khi có request.
4. Test hai caller đồng thời chỉ gọi loader một lần; test startup lỗi và ready.

Giao Leader: commit + kết quả smoke, cập nhật cách khởi động. Đây là phần việc
serving trước demo Banking, không phải việc chặn Leader làm chunks/qrels.
