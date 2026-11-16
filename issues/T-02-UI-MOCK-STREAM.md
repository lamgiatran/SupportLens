# Tuấn làm UI hai chế độ và kiểm tra stream bằng mock

Owner: Phan Trần Quốc Tuấn. Dự kiến 2–3 phiên, làm song song khi Cầu train.

Nhận: contracts.py, api.py, ui/app.py; Docs API R1/R2/R3 và ví dụ request/response
từ Leader. Không cần chờ head Banking hoặc Ollama để bắt đầu giao diện/test.

Việc làm:
- Thêm lựa chọn Docs-QA/Banking, retriever và history; gọi Docs API thật và hiện
  evidence cùng trạng thái phản hồi.
- Thống nhất ví dụ Banking với Cầu. Dùng dữ liệu mẫu cho phát triển/test khi
  chờ pipeline; đường gọi thật vẫn phải báo thiếu model đúng contract.
- Chuẩn bị mock stream nhỏ trong test/dev có ghi rõ mock; thử token đến dần,
  kết thúc, lỗi và hủy. Thống nhất event/response cuối với Leader trước khi nối
  generator thật, không tự coi probe hiện có là endpoint generation.
- UI phản hồi được khi lỗi không phải JSON, không giữ câu trả lời cũ khi reset
  hội thoại hoặc khi một request mới bị lỗi.

Đầu ra: UI hai chế độ, ví dụ contract và kiểm tra các trạng thái stream.
Gửi Leader/Cầu commit, cách chạy và ba ca minh họa.

Xong khi: Docs chạy bằng API thật; mock được phân biệt rõ trong dev/test;
UI xử lý đủ stream thành công/lỗi/hủy. Khi Cầu giao pipeline text-to-prediction,
Tuấn thay mock bằng pipeline đó; không viết encoder/classifier riêng.

Việc nạp/warm-up Banking thật tiếp tục theo T-01. Deadline/cache bên generator
thuộc Leader; Tuấn giữ phần serving/UI và kiểm tra tín hiệu khi tích hợp.
