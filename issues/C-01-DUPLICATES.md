# Cầu — kiểm tra trùng lặp trước khi fit

Mục tiêu: phân biệt câu trùng, tokenized-input trùng và lỗi đọc file nguồn;
chốt cách xử lý trước khi train/calibrate. Dự kiến một phiên 1 giờ; nếu chưa
chốt được nguyên nhân thì ghi blocker và chưa fit, không xóa âm thầm.

Nhận: cache + `splits.json` v0.2, raw `banking77_oos`, encoder/tokenizer đã ghim.
Leader không cần tạo thêm vector cho task này.

1. Mở raw quanh train/3996–3997, 4017–4018; id_oos/valid/288–290,
   309–310; id_oos/test/417–419. In `repr`, mã Unicode và số dòng LF thực.
   So sánh cách đọc `splitlines()` với dòng vật lý; chưa kết luận nguồn sai.
2. Audit trùng giữa train/dev/test theo raw text, text chuẩn hóa và chuỗi token
   thực sự đi vào encoder. Bỏ zero sentinel khỏi việc tìm vector giống nhau.
   Vector bằng nhau là tín hiệu kiểm tra, chưa tự chứng minh hai câu raw giống nhau.
3. Xác minh cặp `id/train/2128` và `id/test/721` cùng các cặp reviewer đã nêu;
   ghi nhãn gốc, nguyên nhân chuẩn hóa/truncation nếu có và vị trí nguồn.
4. Chốt quy tắc trước fit. Phương án đề xuất để Cầu xem xét: giữ test nguyên gốc;
   loại bản sao từ train nếu trùng dev/test, từ dev nếu trùng test; giữ ID và cờ
   loại khỏi fit/selection, không xóa hàng hay sửa nhãn test theo dự đoán.
   Ca mâu thuẫn nhãn cần báo riêng; không tự coi nhãn nào là đúng.

Giao Leader/Cầu: một commit gồm `runs/handoff50/duplicate_audit.json`, danh sách
cặp cần review và quy tắc ngắn trong issue này. Nếu thay mask, giữ nguyên thứ tự
ID và vector, cập nhật manifest/hash. Không dùng số LR kiểm tra nhanh làm metric.

Xong khi: đối chiếu được 8 cặp được báo (hoặc giải thích khác biệt), giữ đủ IDs,
có quy tắc tái lập. Kết quả mở đường cho C-02; chưa phải một run benchmark.
