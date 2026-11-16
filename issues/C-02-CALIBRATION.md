# Cầu — tách calibration sau khi chốt C-01

Nhận: quyết định dữ liệu từ C-01 và cache có thứ tự ID cố định.
Task này không cần encode lại câu; chỉ tạo partition/mask và kiểm tra chúng.
Cập nhật cả script sinh lại cache/splits để chạy lại không ghi đè mất các mask.

Việc làm:
- Tách khoảng 10% ID train còn hợp lệ làm calibration, stratified theo nhãn,
  seed 42. Nếu còn nhóm câu/tokenized-input trùng nhau, giữ cả nhóm trong một
  partition để tránh fit/calibration thấy cùng đầu vào.
- Thêm cờ `calibration` vào từng sample trong `splits.json` và mảng bool trong
  `.npz`. `fit_eligible` phải loại calibration và các mẫu loại theo C-01.
  Không đưa dev, test, ID-OOS hay OOD vào fit/calibration/reference bank.
- Reference bank H3 chỉ lấy các hàng fit thực sự. Giữ đủ 50 nhãn trong fit;
  công bố số mẫu từng nhãn và xử lý trường hợp group-stratification khó đạt đúng 10%.
- Thứ tự: fit các head chỉ trên fit split, chọn C bằng dev → cố định head
  → fit T bằng NLL trên ID calibration
  → chọn confidence/threshold trên dev → freeze trước test.

Giao Tuấn và Leader: commit code, split IDs/mask, kiểm tra seed/disjointness;
cache đã cập nhật dùng link nếu lớn. Cập nhật SHA-256 `.npz` trong manifest.
Encoder fingerprint giữ nguyên nếu encoder và preprocessing không đổi; phiên bản
head/gate phải đổi khi dữ liệu fit/calibration thay đổi.

Xong khi: cùng seed sinh cùng IDs, fit/calibration/dev/test không giao nhau,
OOS không vào bank, một hàng đổi mask không làm lệch vector/nhãn. Tuấn dùng
head có đúng metadata này; Leader chỉ review invariant, không làm thay việc fit.
