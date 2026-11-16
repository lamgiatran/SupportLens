# SupportLens Phân công thực hiện

Nhóm xây dựng một ứng dụng NLP có hai chế độ: hỏi tài liệu có nguồn trên
MultiDoc2Dial và phân luồng ngân hàng Handoff-50. Hưng làm thêm các baseline
Course-77 đáp ứng yêu cầu môn học. Hai chế độ dùng chung API/UI và được đo riêng
trên đúng dữ liệu của từng tác vụ.

## Bắt đầu từ phần hiện có

Docs-QA đã có parser, chunks/qrels, BM25/dense/hybrid và số đo dev. Cầu sở hữu
toàn bộ encoding và dự đoán Banking; Tuấn nhận một pipeline dùng được, không
chờ Leader viết encoder riêng. Các phần có sẵn được chạy kiểm tra rồi mở rộng;
không làm lại để khớp lịch chuẩn bị cũ.

Mỗi bạn có khoảng 15 giờ chủ động: bốn gói khoảng 3 giờ, thêm 1 giờ ghép bài/tập
giải thích và 2 giờ dự phòng. Một gói tương ứng 2–3 phiên làm, có thể sang phiên
sau nếu đầu ra chưa đạt. Giờ máy tự train ghi riêng; chuẩn bị và sửa lỗi tính
giờ làm. Leader tạm dự trù 3 giờ/ngày theo plan; nếu ít hơn thì giảm phạm vi.

| Gói | Mục đích | Mốc phối hợp trong khung 15 ngày |
|---|---|---|
| A | Chuẩn bị phần riêng, thống nhất ví dụ input/output | 2–3 phiên đầu; mốc ngày 2 kiểm tra UI → API → evidence |
| B | Có bản đầu chạy được để ghép | 2–3 phiên tiếp; mốc ngày 5 thử module thật qua API |
| C | Hoàn thiện hành vi và chọn cấu hình bằng dev | 2–3 phiên tiếp; mốc ngày 8 kiểm tra hai chế độ và quyết định giữ/cắt |
| D | Đo bản chốt, viết kết quả và chạy lại | Khóa tính năng trước ngày 13; dành ngày 13–15 sửa lỗi và bảo vệ |

Mốc kiểm tra là điểm phối hợp, không phải yêu cầu mỗi ngày làm một task mới.
Công việc đã đạt đầu ra chuyển thẳng sang kiểm tra/tích hợp.

## Các phần ghép vào nhau thế nào

| Người giao → người nhận | Giao gì và khi nào | Người nhận dùng để làm gì |
|---|---|---|
| Leader → Tuấn | API Docs R1/R2/R3 và ví dụ request/response; dùng ngay ở A | Dựng màn tài liệu, gửi câu hỏi/history và hiện evidence |
| Cầu → Tuấn | Pipeline Banking nhận **câu hỏi**, trả nhãn/score; cuối B | Thay dữ liệu mẫu bằng dự đoán thật; kiểm tra UI → API → model |
| Cầu → Tuấn | Pipeline/ngưỡng đã chọn trên dev; cuối C | Hoàn thiện accept/handoff, dùng cùng phiên bản để đo và demo |
| Hưng → Leader | Review 20 ca qrels; sớm trong A hoặc B | Kiểm tra bằng chứng trước train pairs retriever và đánh giá cuối |
| Hưng → Leader | Baseline, metrics và phân tích lỗi; B đến D | Ghép phần thực nghiệm môn học và giải thích lựa chọn mô hình |
| Leader → Tuấn | Generator có stream, cấu hình cache/deadline; C | Hiện câu trả lời dần, nguồn và trạng thái lỗi/hoàn thành |
| Tuấn → Cầu và Leader | Commit tích hợp, ví dụ chạy thật, lỗi tái hiện; B đến D | Kiểm tra API khớp module độc lập và sửa đúng nơi phát sinh lỗi |

Banking bản đầu dùng để **kiểm tra kết nối sớm**. Cầu vẫn tiếp tục so sánh score
và chọn ngưỡng sau đó. Tuấn làm UI/contract bằng dữ liệu mẫu trong lúc chờ, đến
B thay bằng pipeline thật. Hưng bắt đầu độc lập với hai bạn.

## Hưng Sở hữu baseline Course 77

**Mục đích:** xác định preprocessing, biểu diễn và classifier ảnh hưởng phân
loại intent thế nào; cung cấp phần thực nghiệm môn học.

**Nhận:** BANKING77 gốc 77 nhãn, protocol Course-77 và script tải dữ liệu.
H-01 nhận qrels audit/chunks từ Leader.

| Gói | Việc tự làm và hướng đi | Đầu ra và điều kiện xong |
|---|---|---|
| A | Audit; chia fit/dev theo nhãn, giữ official test; chuẩn hóa/tokenize/lemmatize; E1 = BoW + LR. Làm H-01 sớm. | Split IDs, E1 chạy được; CSV nhận xét 20 qrels giao Leader. Giữ cùng split giữa các baseline. |
| B | E2 = TF-IDF + LR; E3 = TF-IDF + Linear SVM; một ablation preprocessing. Dùng sklearn Pipeline để vectorizer chỉ fit train. | Code train/evaluate, bảng dev Accuracy/Macro-F1 và cấu hình. Mỗi so sánh chỉ rõ thứ đã đổi. |
| C | Chọn cấu hình bằng dev, xem confusion và 10 ca sai, giải thích bằng ví dụ thật. | Pipeline load lại được, bảng dev và phân tích lỗi giao Leader. E4 chỉ mở khi phần lõi xong. |
| D | Freeze theo dev rồi đánh giá official test; hoàn thiện báo cáo và lệnh tái lập. | Metrics cuối, confusion matrix, phần báo cáo, một lệnh chạy và ví dụ load/predict. |

**Nơi làm dự kiến:** bổ sung module Course-77 và script train/evaluate;
kết quả ở `runs/course77/`, pipeline nhỏ ở `artifacts/course77/`.
**Giao Leader** để ghép báo cáo. [H-01](../issues/H-01-QRELS-REVIEW.md) được dùng
kiểm tra dữ liệu retrieval. Hưng cần giải thích vì sao vocab chỉ fit train và
vì sao preprocessing tốt/kém hơn trên những ca đã xem.

## Cầu Sở hữu toàn bộ Banking Handoff 50

**Mục đích:** nhận câu hỏi mới, dự đoán 50 intent hỗ trợ và quyết định có đủ
tin cậy để xử lý hay cần chuyển người.

**Nhận:** raw BANKING77-OOS, `banking.py`, `banking_data.py`, cache embeddings
có sample IDs/fingerprint. Cầu quản lý cùng một `encode()` cho cache và câu mới.

| Gói | Việc tự làm và hướng đi | Đầu ra và điều kiện xong |
|---|---|---|
| A | C-01 audit trùng/rỗng; C-02 tách calibration. Chốt quy tắc loại khỏi fit/selection, giữ IDs và kiểm tra disjointness. | Audit, split/mask và script tái tạo giao Leader review; thay đổi có số mẫu/lý do. |
| B | Fit LR 50 nhãn, chọn C bằng dev rồi cố định head. Bọc encode → predict; bản đầu H1 và ngưỡng dev ban đầu. | **Giao Tuấn:** model, label mapping/config, hàm nhận câu hỏi; cách load và 3 ví dụ input/output. Tuấn nối API bằng câu hỏi thật. |
| C | Fit T của H2 trên calibration; H3 kNN dùng train-fit bank; H4 energy. So risk–coverage, chọn score/ngưỡng bằng dev. | **Giao Tuấn:** pipeline/ngưỡng đã version và hai ví dụ accept/handoff. Giao Leader đường cong/bảng dev. |
| D | Freeze trước test; đánh giá cuối, ma trận ID-OOS bị hút vào intent nào và reweight OOS 5/10/20%; viết báo cáo. | Metrics/predictions, phân tích lỗi và cấu hình đúng bản Tuấn phục vụ. Kết luận dựa trên số đo thật. |

**Nơi làm:** mở rộng module Banking hiện có; `runs/handoff50/` và
`artifacts/handoff50/`. H2 fit T sau chọn C; H3 bank không có calibration,
dev, test hay OOS. Chi tiết: [C-01](../issues/C-01-DUPLICATES.md),
[C-02](../issues/C-02-CALIBRATION.md).

**Cầu → Tuấn là pipeline hoàn chỉnh**, không chỉ một file vector. Tuấn không
viết encoder khác hoặc tự chọn confidence/ngưỡng. Cầu giải thích được câu hỏi
→ vector → logits → score → quyết định và cách tránh leakage.

## Tuấn Sở hữu ứng dụng và tích hợp

**Mục đích:** người dùng thao tác được cả hai chế độ; API/UI dùng đúng module,
hiển thị đúng trạng thái và xử lý được lỗi khi chạy.

**Nhận:** `contracts.py`, `api.py`, `ui/app.py`, API Docs và ví dụ từ Leader.
Nhận Banking bản đầu ở B, bản chốt ở C.

| Gói | Việc tự làm và hướng đi | Đầu ra và điều kiện xong |
|---|---|---|
| A | Chạy nền; UI hai chế độ, lựa chọn retriever/history; thống nhất request/response với Cầu/Leader. Banking dùng dữ liệu mẫu trong test. | UI gọi Docs thật và hiện evidence; ví dụ contract Banking được Cầu xác nhận. |
| B | Thử pipeline Banking độc lập trước rồi nối endpoint/UI. Warm-up/nạp một lần, xử lý lỗi JSON và không phải JSON. | Cùng câu hỏi cho kết quả độc lập/API khớp; giao Cầu/Leader ví dụ/lỗi tái hiện. Request đầu không tự tải weights. |
| C | Nối stream generator của Leader; hội thoại/reset, nguồn, chuyển người, lỗi/huỷ stream. Nạp ngưỡng Banking chốt. | Hai chế độ dùng module thật; thử thành công/chuyển người/lỗi. Leader cung cấp deadline generator; Tuấn xử lý phía serving/UI. |
| D | Cùng nhóm chốt/chạy 40 smoke; thử môi trường mới; đo latency, tách cold start; viết cách chạy. | Case ID và expected/actual, hướng dẫn chạy lại, phần báo cáo triển khai. |

**Nơi làm:** API/UI hiện có. Mock dùng cho phát triển/test khi chờ module.
[T-01](../issues/T-01-WARMUP.md) hướng dẫn warm-up Banking. Tuấn giải thích được
lifecycle nạp model, đường request và lỗi stream. Docker/CI/SQLite không thuộc
phần bắt buộc 15 giờ này.

## Leader Sở hữu Docs QA và nghiệm thu

Dùng tiếp retrieval đã có; làm generator, citation, cache và deadline tổng,
đưa đúng cấu hình được đo vào API để giao Tuấn. Nhận qrels review từ Hưng trước
train pairs nâng cao. Review/hỗ trợ gỡ lỗi nhưng owner tiếp tục sửa phần mình.

Mục nâng cao ưu tiên: fine-tune dense retriever từ public-train query–grounding
passage bằng MultipleNegativesRankingLoss. Smoke run quyết định giờ máy. So
R4 với R2; nếu dùng hybrid, thêm R5 = BM25 + R4 so với R3. Giữ query/chunks/splits
cố định, không dùng dev/held-out dialogue để train. Chỉ mở khi luồng lõi ổn và
còn giờ chủ động. Thời gian máy tự train không trừ giờ làm của Leader.

## Bàn giao và nghiệm thu

Một lần giao gồm **commit/PR + một cách kiểm tra + 3 ví dụ input/output**.
Người nhận chạy khớp ví dụ rồi gọi qua API, owner cùng đối chiếu kết quả.
Không cần viết README/manifest mới cho mỗi lần gửi. Vector/index lớn dùng link
chung; code, split IDs, metrics và model nhỏ hợp lý được quản lý trong Git.

Lỗi nhãn/encoding/confidence Banking → Cầu; preprocessing/baseline → Hưng;
API/UI/stream → Tuấn; retrieval/generator/qrels → Leader. Mỗi bạn gửi thêm vài
ca lỗi và 5–10 dòng giải thích lựa chọn, rồi tự trình bày module. Công cụ hỗ trợ
được ghi riêng trong AI_USAGE theo quy định môn học.

Xong khi hai chế độ chạy bằng module thật, demo khớp phiên bản đã đánh giá,
có câu trả lời/nguồn hoặc lý do chuyển người, người khác chạy lại được và
nhóm giải thích được luồng. Đây là bằng chứng cho portfolio fresher; một dự án
không tự quyết định cấp bậc nghề nghiệp.
