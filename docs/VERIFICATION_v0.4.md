# Xác minh thực nghiệm v0.4 — 16/11/2026

Bản xuất bản v0.4.1 giữ nguyên query/model/số đo dưới đây, tách gate Docs khỏi
Gate Banking và giữ UI v0.3 cho Tuấn tự mở rộng. Xem OWNERSHIP_REVIEW.md.

Đã chạy trên dữ liệu thật, chỉ đánh giá dev. Không chạy retrieval trên held-out.
Corpus giữ nguyên 488 tài liệu, 2.837 chunks; dev có 2.213 queries / 329 dialogues.
Các số dưới là strict relevance (>=50% của ít nhất một span gold), không phải
độ đúng câu trả lời. Metrics lenient cũng có trong JSON.

## History và confidence

| Query | hit@3 | hit@5 | MRR@10 | Token trung bình |
|---|---:|---:|---:|---:|
| Chỉ câu hiện tại | 0.3498 | 0.4121 | 0.3028 | 16.27 |
| Thêm 1 tin nhắn | 0.4433 | 0.5224 | 0.3792 | 34.20 |
| Thêm 2 tin nhắn | 0.4668 | 0.5377 | 0.3978 | 46.80 |
| Toàn bộ, cắt ở 256 token | 0.4618 | 0.5364 | 0.3882 | 118.75 |

Một tin nhắn là một message user hoặc assistant, không phải một cặp hỏi–đáp.
Chênh MRR hai tin nhắn trừ toàn bộ: +0.00960, dialogue-cluster bootstrap 95% CI
[-0.00465, +0.02362], 2.000 lần, seed 42. Không kết luận khác biệt chất lượng
có ý nghĩa; chọn hai tin nhắn vì query ngắn hơn, không có truncation dev và gate
tốt hơn. Phiên bản R1 mới: `R1-chunk-b3ba5da19294491c`.

Top-1 BM25 thô với full history có AUROC-success@5 = 0.45394; tương quan Pearson
với số từ query = 0.92120. Điểm top1/nterms với hai tin nhắn đạt 0.70367 ở top-5
và 0.70289 ở top-3. Số từ là token regex `\w+`, có tính lặp, không phải WordPiece.
Nguồn tái lập: `history_metrics.json`, `history_gate3_metrics.json`,
`history_gate5_metrics.json` và scripts cùng tên trong repo.

## R1/R2/R3

| Cấu hình | hit@3 | hit@5 | MRR@10 | Gold còn được hỗ trợ sau packing |
|---|---:|---:|---:|---:|
| R1 BM25 | 0.4668 | 0.5377 | 0.3978 | 0.4523 |
| R2 MiniLM frozen | 0.4062 | 0.4858 | 0.3367 | 0.3972 |
| R3 BM25 + R2, RRF | 0.4803 | 0.5513 | 0.3993 | 0.4618 |

R2 thua BM25 trên dev này. R3 tăng hit@3 0.01356 so với R1 nhưng CI95
[-0.00404, +0.03136] chứa 0; MRR delta +0.00151, CI95 [-0.01035, +0.01452].
Giữ R1 làm mặc định; R2/R3 có thể chọn để thử nghiệm. Chưa có fine-tuning R4/R5.

Packing dùng đúng tokenizer Qwen, tối đa ba evidence / 600 token. Audit character
offset đo phần evidence thật sự còn nhìn thấy; title không được tính là hỗ trợ
span. R1/R2/R3 mất hỗ trợ gold ở 32/20/41 query sau packing; không có prompt
overflow. Đây là hỗ trợ >=50% của một span, chưa đo bao phủ mọi reference hoặc
mọi claim, chưa chạy hay chấm LLM. `context_audit.json` lưu chi tiết.

## Gate coverage

| Method | Score dùng quyết định | Coverage dev | Risk gold không có trong top-3 |
|---|---|---:|---:|
| R1 | Top-1 BM25 / số từ query | 0.7004 | 0.4413 |
| R2 | Cosine top-1 | 0.7004 | 0.5606 |
| R3 | Cosine lớn nhất trong ba kết quả hybrid thực tế | 0.7004 | 0.4865 |

Coverage mục tiêu đặt 0.7; chấp nhận trọn nhóm score bằng nhau. Ngưỡng đặt giữa
hai nhóm kề nhau nếu có để tránh sát một mẫu dev do sai số float. Chọn coverage
không bảo đảm risk 10%. Full curve nằm trong file gate và score metrics; API
kiểm tra method, artifact, tên score và k trước khi dùng. Ngưỡng raw cũ bị từ chối.

Cosine gate của hybrid có AUROC-success@3 = 0.58410, yếu hơn R1 chuẩn hóa 0.70289.
RRF raw có AUROC 0.69349 trong phép đo này dù nhiều ties; không suy ra rằng mọi
score RRF đều vô dụng. Cấu hình phục vụ hiện vẫn dùng cosine như đã định và báo
đúng giới hạn. Gate chưa được kiểm định trên câu ngoài corpus/unanswerable.

## Code và serving

- 64 test pass: thêm dev-only guards, history, sparse BM25 bằng chính xác công
  thức cũ (kể cả IDF âm và từ lặp), cache identity, AUROC ties, coverage, concurrency,
  startup, score binding và kiểm tra phần evidence bị cắt.
- 78 request API qua factory môi trường thật, 26 query dev thuộc bốn domain,
  gồm hai phía sát gate. Top-5 và quyết định khớp offline toàn bộ. Sai số score
  lớn nhất R2/R3 1.20e-7; R1 bằng 0. Đây là smoke test, không phải benchmark.
- Encoder Docs nạp/warm-up trước ready, dùng chung cho R2/R3 và có lock khởi tạo.
  Thời gian này đo riêng; weights phải chuẩn bị trước khi chạy API.
- Corpus vectors 2.837 x 384 float32 đã chuẩn hóa L2. Cache ID
  `4810cff123c22b2fdcdf3bc0ca61b63c8837ac89da7a400f478e285e4b059ff2` không chứa
  policy history. Thay history không bắt buộc encode lại corpus.
- Thời gian lookup offline loại trừ encode query. Smoke API có encode từng query
  thật; các thời gian là trên máy phát triển dùng chung, không phải p95/TTFT UI.

Nguồn: `retrieval_metrics.json`, `retrieval_gate3_metrics.json`,
`retrievers_api_audit.json`, `context_audit.json`, `artifacts/docs_gates/`.

## Còn phải làm và tình trạng GitHub

Giữ C-01/C-02 cho Cầu tự audit/trích calibration rồi fit Banking; T-01 Banking
warm-up vẫn thuộc Tuấn phối hợp Cầu. Hưng review qrels trước train pairs R4.
Generator streaming, generation cache, deadline tổng và đo TTFT vẫn chưa làm.
