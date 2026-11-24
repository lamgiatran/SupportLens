# Docs-QA · Kiểm tra nguồn cho từng phát biểu

## Thiết kế

Toàn bộ thí nghiệm dùng dev MultiDoc2Dial. Giữ R1, gate dev, Qwen 2.5 3B,
temperature 0, giới hạn 128 token đầu ra và tối đa ba đoạn có tổng ngân sách
600 token. Các phương pháp được so theo cặp trên cùng câu và cùng đoạn nguồn.

1. Đọc 20 câu đầu tiên có quyết định trả lời trong 50 dự đoán đã lưu của v2.
   Đối chiếu từng ý với phần nguồn hiện được. Mẫu này chỉ phục vụ tìm lỗi.
2. Trên 24 hội thoại dev khác, chạy v2 và prompt yêu cầu mỗi dòng có nhãn,
   trích dẫn nguyên văn và phát biểu. Chỉ nhận khi mọi dòng hợp lệ, thân trích
   khớp chính nguồn (chuẩn hóa khoảng trắng) và có ít nhất bốn từ.
3. Sau khi xem lỗi định dạng ở bước 2, khóa một phương án dự phòng: Qwen chọn
   một ID câu bằng JSON, 0 để từ chối. Server tra bảng ID, lấy nguyên văn cùng
   offset và nhãn nguồn. Phương án này chỉ có thể trả một câu trích xuất.
   Đánh giá trên 32 hội thoại dev mới, mỗi hội thoại một câu. Loại toàn bộ
   hội thoại từ bộ 50 câu cũ và 24 hội thoại đã dùng để xem lỗi.

Mỗi phương pháp báo (a) số trả lời / toàn bộ câu, (b) số trả lời có tất cả ý
được nguồn trích hỗ trợ / số trả lời, và (c) số câu **vừa bám nguồn vừa trực
tiếp giải đáp yêu cầu** / toàn bộ câu. Mẫu số bằng 0 ghi N/A. Câu trích
khớp nguyên văn vẫn có thể lạc đề, thiếu điều kiện hoặc phụ thuộc một tiêu đề
bị loại khỏi câu. Giữ v2 trong app trừ khi phương án mới tăng (c) và giữ được
coverage của v2. Đây là quyết định cho bản demo dựa trên mẫu nhỏ.

Nhãn ở đây do Codex đối chiếu, chưa có người thứ hai chấm độc lập. Với v2,
nhãn ở cuối một cụm câu liên tiếp được đọc là nguồn cho các ý đứng trước nó
kể từ nhãn gần nhất. Một ý có nhiều nhãn đạt nếu ít nhất một nguồn được trích
hỗ trợ nó. Xét độ đúng theo tài liệu hiển thị, không xác minh chính sách ngoài đời.

## Kết quả ngày 24/11/2026

Trong 20 câu trả lời v2 đã lưu, 9 câu có bằng chứng cho mọi ý được nêu. Ví dụ
ở câu hỏi về phí REAL ID, câu trả lời trích S1 cho phát biểu về phí dù phần
phí xuất hiện ở S2. Ở câu về hai trường, đoạn trích nói về chuyển quyền lợi
cho người thân. Đoạn đó không xác lập khả năng học hoặc thanh toán cho hai
trường cùng lúc. Đây là 20 câu đã được xuất ra, nên mẫu này không đo coverage.

Trong vòng thử 24 câu, gate chuyển người 10 lần. Với 14 câu còn lại, cả 14
output của prompt yêu cầu chép trích dẫn đều bị loại vì định dạng. Model đôi
khi chép một đoạn rồi điền PASS vào trường phát biểu. Baseline trả 8/24,
trong đó có 4 câu vừa bám nguồn vừa giải đáp yêu cầu. Không thể tính độ chính
xác trên câu đã xuất của phương án không xuất câu nào.

| Trên 32 hội thoại dev mới | v2 hiện tại | Chọn một câu bằng ID |
|---|---:|---:|
| Qua gate retrieval | 20/32 | 20/32 |
| Xuất câu trả lời | 16/32 | 13/32 |
| Mọi ý được nguồn trích hỗ trợ, trong các câu đã xuất | 7/16 | 13/13 |
| Vừa bám nguồn vừa giải đáp yêu cầu, trên toàn bộ mẫu | 6/32 | 3/32 |

Với cách chọn ID, server lấy nguyên văn từ nguồn nên 13/13 phản ánh cơ chế
xuất câu. Đọc theo câu hỏi cho thấy chỉ 3/13 câu đó giải đáp trực tiếp.
Ví dụ, người dùng xác nhận đang gặp khó khăn trả nợ. v2 đưa ra hành động có
nguồn là liên hệ bên quản lý khoản vay. Câu được selector chọn chỉ nhắc lại
rằng người nói đang gặp khó khăn. Một ca có lợi cho selector là câu hỏi về
cách làm báo cáo khuyết tật qua điện thoại. v2 đưa thêm số thứ hai sau chỗ
nguồn bị cắt. Selector lấy đúng câu hướng dẫn có số thứ nhất.

**Quyết định:** tiếp tục dùng v2 cho demo local. Hai phương án thử đều chưa đạt
tiêu chí đã đặt. Các tỉ lệ là số đếm trên dev với một lượt chấm bằng AI, chưa
có ước lượng trên held-out hoặc đồng thuận giữa người chấm. Người rà soát có
thể chấm khác ở câu trả lời ngắn hoặc câu hỏi phụ thuộc hội thoại. Phương án
chọn ID cũng có thể lấy tiêu đề hoặc mảnh câu khi văn bản có viết tắt như
U.S.; chỉ đối chiếu vị trí ký tự chưa giải quyết được lỗi ngữ cảnh.

## Kiểm tra lại

[Bản ghi và nhãn từng ý](../runs/docs_qa/claim_review_audit.json) lưu câu hỏi,
quyết định của từng cách, câu trả lời v2, nhận xét cho từng ý, ID đoạn, độ dài
phần thực sự hiện trong prompt và hash. Với selector, dùng `start:end` trên
chuỗi Unicode của đoạn đó để lấy đúng trích dẫn. Chạy lệnh đầu từ gốc repo
để tính lại bảng mà không gọi model. Lệnh `--show` chỉ cần `chunks.jsonl`
để khôi phục và kiểm tra hash từng nguồn của ca REAL ID. Hai lệnh cuối cần
bundle, tokenizer và Ollama 3B local như [hướng dẫn generator](RUN_GENERATION.md).

```powershell
python scripts/score_claim_review.py
python scripts/score_claim_review.py --show b023e676deaf2ba9bbcb694f7f0f1930:1
python scripts/eval_claim_citations.py --method quote --out runs/docs_qa/claim_rerun_1
python scripts/eval_claim_citations.py --method selector --out runs/docs_qa/selector_rerun_1
```

Mỗi lần chạy dùng thư mục mới, ghi danh sách ID trước khi gọi các lượt so
sánh, rồi lưu output thô cùng nguồn hiện được. Các response mới cần được chấm
lại nếu khác bản đã ghi. Mã thí nghiệm chỉ được gọi từ dòng lệnh.
