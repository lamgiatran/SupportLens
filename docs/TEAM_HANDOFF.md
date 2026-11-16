# Bàn giao gọn cho bản v0.2

Giữ các chặng 1–3 / 4–6 / 7–9 / 10–12 / 13–15 trong file phân công.
Chỉ thay các điểm phối hợp sau; không cần viết lại kế hoạch.

| Người | Sở hữu và làm tiếp | Giao ai, để làm gì? |
|---|---|---|
| Hưng | Nhánh Course-77, baseline và phân tích lỗi | Commit code + metrics cho Leader ghép báo cáo; không giao head 77 nhãn cho Banking |
| Cầu | Toàn bộ `banking.py`: encode, head LR 50 nhãn, H1–H4 và ngưỡng dev | Giao Tuấn một pipeline có `predict(question, method=...)`; Tuấn dùng để nối câu hỏi thật ở chặng 2, hoàn thiện quyết định ở chặng 3 |
| Tuấn | Nạp pipeline, API/UI, log và kiểm tra toàn luồng | Trả commit + kết quả smoke cho Cầu/Leader xác nhận việc ghép; không viết lại encoder |
| Leader | Docs-QA và nghiệm thu; có thể chạy cache giúp Cầu | Gửi vector nếu đã tạo được; Cầu cũng tự chạy được cùng script, không cần chờ một encoder khác từ Leader |

**Bàn giao = commit + một lệnh smoke + kết quả.** Commit split IDs, config,
metrics và model nhỏ. Vectors/index/weights lớn dùng link; ghi một lần trong
issue/PR nơi lấy và nơi đặt. Không cần README mới, BOARD và tin nhắn dài cho mỗi lần.

Đã gửi = commit/link có đủ code và file. Đã nhận = người nhận chạy smoke pass.
Đã ghép = API/UI dùng đúng gói và đã thử toàn luồng. Chỉ cập nhật một issue/PR.

`scripts/smoke.py <gói>` đọc `manifest.json` với `smoke_entrypoint` và
`smoke_samples.jsonl`. Owner cung cấp hàm wrapper nhận `input`, `package_dir`
và trả dict; wrapper nạp đúng model của gói. Không ghi “pass” cho gói chưa có.
Script chỉ chạy callback trong repo/gói do nhóm tin cậy; không dùng pickle tải lạ.

ZIP đã kèm cache thật, giữ 12.221 ID nguồn. Có 8 input rỗng: dùng `valid_input` và
`fit_eligible`; vector zero tại hàng invalid là sentinel, không phải embedding.
Không fit/score/tìm láng giềng trên những hàng đó; giữ chúng trong thống kê nguồn,
báo riêng từ chối input rỗng khi đánh giá. API từ chối câu rỗng bằng 422.
