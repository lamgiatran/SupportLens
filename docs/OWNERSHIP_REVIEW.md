# Rà soát phần được push — v0.4.1

Đối chiếu với nền v0.3, trước khi xuất bản phần Leader. Scaffold có sẵn là công
cụ bắt đầu chung, không được dùng để ghi rằng Hưng/Cầu/Tuấn đã hoàn thành việc.

| Phạm vi | Kết luận kiểm tra | Quyết định trong nhánh xuất bản |
|---|---|---|
| Leader: data/query/chunks/qrels, R1/R2/R3, Docs generator/prompt, đánh giá | Thuộc phần retrieval và tích hợp của Leader trong plan | Giữ code và số đo dev; generator serving/cache/deadline, R4/R5 còn pending |
| Cầu: banking.py, banking_data.py, cache builder, LR/calibration/H1–H4 | Ba file Banking giữ nguyên v0.3; chưa fit head hay làm C-01/C-02 | Giữ nguyên. Gate Docs nằm riêng trong docs_gate.py; evaluation.py trở lại đúng nền v0.3 để không đổi contract Gate của Banking |
| Hưng: preprocessing và baseline Course-77 | Chưa triển khai/train E1–E3 thay Hưng | Giao Hưng tự thực hiện và giao code/metrics thật; H-01 vẫn là review bằng mắt của Hưng |
| Tuấn: API/UI/streaming và Banking serving | V0.4 đã làm trước bộ chọn retriever/history trong UI | Bỏ phần UI mới khỏi nhánh xuất bản, giữ scaffold UI v0.3. Tuấn tự làm chọn R1/R2/R3, history/state, stream và ghép Banking |
| API Docs adapter | Cần để Leader kiểm tra cùng policy/model ở offline và API | Giữ registry R1/R2/R3, load corpus và warm-up encoder Docs; nhánh Banking của endpoint không đổi. Tuấn vẫn sở hữu hoàn thiện serving và nghiệm thu UI/toàn luồng |

Không xóa bản nháp để làm mất lịch sử. Nhánh xuất bản được dựng từ v0.3, lấy
chọn lọc các thay đổi thuộc Leader. Bản đầy đủ v0.4 được giữ cục bộ; commit UI
đã làm trước không là tổ tiên của nhánh xuất bản và không được push theo main.
Git bundle của bản mới chỉ chứa dòng lịch sử release và tag đi tới release đó.

Đầu ra tiếp theo cho từng bạn nằm ở [TEAM_HANDOFF.md](TEAM_HANDOFF.md). Không giao
Tuấn lặp lại warm-up Docs đã có; T-01 dành cho encoder/head Banking do Cầu giao.
Không tính code AI, kết quả smoke của Leader hoặc tên trong ticket thành đóng
góp cá nhân của người nhận task.

Kiểm tra máy: docs/verification/ownership_audit.json. Các file runtime, weights,
embeddings và request logs không thuộc payload Git. Quyền push phần Leader đã
được người dùng xác nhận; kết quả push chỉ được công bố sau khi kiểm tra remote.
