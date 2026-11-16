# Docs-QA v0.3 — kết quả thực thi 16/11/2026

Bộ regression hiện tại: **51 passed**, gồm tokenizer thật, qrels strict/lenient,
numeric-link/citation-list, API/offline history parity và bảo toàn checksum qua
Git checkout trên Windows (LF cố định cho JSON/JSONL).

Corpus thật: 488 tài liệu, 2.837 chunks; tối đa 220 WordPiece gồm title.
Public train có 3.474 dialogues/22.806 query; public validation tách theo domain
và dialogue ID thành dev 329/2.213 query và held-out 332/2.214 query, seed 42.
Các tập dialogue không giao nhau. Chưa chạy retrieval trên held-out.

Tất cả query có positive strict và lenient; không có reference bị unknown,
offset sai, text mismatch hoặc thiếu mức phủ strict trong các reference được
map. Mỗi query có thể có nhiều reference; qrels là hợp các positives, không
phải chứng minh một chunk trả lời được mọi ý. Đã lưu 20 ca audit, gồm nhiều
reference và span qua biên, để Hưng review ngữ nghĩa/định nghĩa relevance.

R1 dev trên cùng corpus/query/dedup policy được API dùng:

| Metric | Strict | Lenient |
|---|---:|---:|
| Hit@5 (ít nhất một positive) | 0,5364 | 0,5409 |
| Gold-chunk recall@5 (macro) | 0,5052 | 0,4989 |
| MRR@10 | 0,3882 | 0,3937 |

Số trên chỉ là dev với segmentation riêng; không phải số test, độ đúng câu trả
lời hay tái lập paper gốc. Latency trong JSON chỉ là truy xuất query đã chuẩn bị
trên index ấm, đo ở máy phát triển dùng chung; chưa đo UI/LLM/end-to-end.

Smoke API: 20 lượt từ raw validation, 5 mỗi domain, có 4 ca cắt history. Query
và top-5 evidence của API khớp offline cả 20 ca. API trả chunk, chưa sinh answer.
Alias `[S1]`/danh sách đã có regression; dấu link số của raw text được giữ nguyên.
Nếu nguồn đã có namespace S, prompt chọn namespace khác và parser dùng theo.

`splits.json` Banking giảm 2.964.013 → 2.059.522 byte, một sample mỗi dòng.
Đã kiểm tra nội dung JSON và SHA-256 embedding không đổi. Chưa xử lý duplicate
hay tách calibration: đó là C-01/C-02 để Cầu thực hiện trước fit; warm-up là T-01
của Tuấn phối hợp Cầu. Không dùng accuracy LR kiểm tra nhanh làm kết quả.

Dense/hybrid, gate score Docs-QA, generation/cache và deadline tổng còn làm tiếp.
Code/API kiểm thử tự động không thay review thủ công qrels hay citation entailment.
