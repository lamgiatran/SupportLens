# Hưng — review mẫu qrels sau baseline

Nhận từ Leader: `data/processed/docs_qa/qrels_audit.jsonl` (20 ca),
`chunks.jsonl` và quy tắc strict/lenient trong `docs/RUN_DOCS_QA.md`.

Đọc gold excerpt, offset và chunk ID; tìm chunk trong `chunks.jsonl`, kiểm tra
span qua biên, nhiều reference và mức phủ. Không sửa raw text/dấu link hay dùng
kết quả held-out để tinh chỉnh retriever.

Giao Leader một commit `runs/docs_qa/qrels_review.csv`: query ID, span/chunk ID,
nhận xét đúng/cần xem lại và lý do ngắn. Leader phân xử case cần sửa trước dense.
Đây là review dữ liệu/định nghĩa relevance, không phải chấm câu trả lời LLM.
