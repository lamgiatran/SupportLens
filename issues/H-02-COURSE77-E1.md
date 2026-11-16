# Hưng dựng baseline E1 cho Course-77

Owner: Nguyễn Tấn Hưng. Dự kiến 2–3 phiên, tính cả chuẩn bị và ghi kết quả.

Nhận: BANKING77 gốc và protocol Course-77 trong plan. Nhánh này dùng 77 nhãn
cho thực nghiệm môn học; không thay model 50 nhãn trong ứng dụng Banking.

Việc làm:
- Audit nhãn, câu rỗng/trùng; giữ official test và tạo train-fit/dev theo nhãn,
  seed 42. Ghi split IDs và cách xử lý trùng trước khi fit.
- Viết preprocessing và E1 = BoW + Logistic Regression. Ưu tiên sklearn
  Pipeline để vectorizer và classifier được lưu/load cùng nhau.
- Chỉ fit preprocessing có tham số học/vocab/model trên train-fit; dùng dev
  để chọn cấu hình. Ghi cách tokenization/lemmatization nếu bật.
- Xuất Accuracy, Macro-F1, confusion và vài ca sai trên dev.

Đầu ra: code train/evaluate, split IDs, pipeline load lại được và bảng dev
trong runs/course77/. Vị trí model nhỏ: artifacts/course77/.

Bàn giao Leader: commit, một cách chạy và ba câu input/output. Leader dùng
để ghép bảng thực nghiệm, rồi Hưng mở rộng E2/E3 và ablation trên cùng split.

Xong khi: người khác load/predict được và Hưng giải thích được quy tắc split,
vì sao vocab chỉ fit train, cùng ít nhất một lỗi quan sát từ dữ liệu thật.
