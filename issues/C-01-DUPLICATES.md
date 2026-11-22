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

### Quy tắc chốt trước fit (Action Rule - Cau):
1. Giữ nguyên tập test và dev (in-scope và OOS) để đảm bảo tính khách quan của benchmark; không xóa dòng vật lý khỏi file raw hoặc cache để giữ ánh xạ 1-1 `sample_id` và thứ tự vector.
2. Loại bỏ hoàn toàn khỏi `fit_eligible` (tổng 6 mẫu trong `id/train`, còn **5,899 / 5,905** mẫu hợp lệ trước khi tách calibration):
   - 2 dòng rỗng do lỗi định dạng nguồn: `id/train/3996`, `id/train/4017`.
   - 3 mẫu rò rỉ nguyên văn / token-ID MiniLM từ train sang dev/test: `id/train/3997` (trùng `id/valid/1017`), `id/train/4018` (trùng `id/test/1353`), `id/train/5823` (trùng token-ID MiniLM với `id/test/1994` do double space).
   - 1 mẫu trùng token-ID MiniLM & mâu thuẫn nhãn: `id/train/2128` (trùng token-ID với `id/test/721` nhưng nhãn trái ngược).
3. Đánh dấu loại khỏi `dev_selection` (khi chọn siêu tham số $C$ và ngưỡng trên dev, giữ nguyên test):
   - `id/valid/1091` (trùng token-ID MiniLM với `id/test/1415` do khoảng trắng trước `?`).
   - `id_oos/valid/211`, `id_oos/valid/288`, `id_oos/valid/310` (trùng với `id_oos/test/419, 595, 638`).
4. Lưu ý cho C-02: Nhóm trùng token-ID nội bộ trong `id/train` (`id/train/3389` và `id/train/3401`) phải được giữ cùng một phân vùng (`fit` hoặc `calibration`).

Xong khi: đối chiếu được 8 cặp được báo (hoặc giải thích khác biệt), giữ đủ IDs,
có quy tắc tái lập. Kết quả mở đường cho C-02; chưa phải một run benchmark.
