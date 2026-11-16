# Việc thực hiện và bàn giao

Ưu tiên hiện tại theo chặng và đầu ra: [TEAM_HANDOFF.md](../docs/TEAM_HANDOFF.md).
Bảng ngày 1–3 bên dưới là các ticket gốc, không phải lịch giao việc mới.

Plan đã khóa ở docs/plan-v1.0.md. Bảng này là danh sách thực hiện, không phải một bản plan mới.
Các file là ticket cục bộ để copy lên GitHub sau khi nhóm có remote; chưa có GitHub issue IDs.
Owner là phân công dự kiến; code scaffold chưa phải bằng chứng đóng góp của owner.

| Ngày | Leader | Hưng | Cầu | Tuấn |
|---|---|---|---|---|
| 1 | [D01-L](D01-L.md) | [D01-H](D01-H.md) | [D01-C](D01-C.md) | [D01-T](D01-T.md) |
| 2 | [D02-L](D02-L.md) | [D02-H](D02-H.md) | [D02-C](D02-C.md) | [D02-T](D02-T.md) |
| 3 | [D03-L](D03-L.md) | [D03-H](D03-H.md) | [D03-C](D03-C.md) | [D03-T](D03-T.md) |

## Năm guardrail và việc sau ngày 3

| Việc | Owner / ticket | Cách kiểm tra |
|---|---|---|
| Current question trước reverse history | Leader / D03-L | Test history dài, log truncation trên query thật |
| Chunk chung vừa tokenizer encoder | Leader / D03-L | Title + body ≤ 220 tokens; encoder limit 256 |
| Docs-QA có gate retrieval-risk | Leader đã làm v0.4; Cầu review code dùng chung | Coverage 70% trên dev; error = gold absent top-3; báo risk thật |
| num_ctx tường minh, token audit, TTFT | Leader + Tuấn / D01-L, D03-T | raw probe, token count mismatch, 600-token top-3 profile |
| R4/R5 và đo thời gian bằng smoke run | Leader / backlog bên dưới | Không mặc định phải chạy đêm hoặc mất vài giờ |

## Backlog sau khi luồng lõi ổn

- R2/R4 dùng cùng query builder, chunks, split IDs. Train pairs chỉ public train; audit positive conflicts cho MNRL.
- Đo throughput từ smoke run rồi quyết định GPU schedule. Có thể chạy trong một phiên; thời gian chuẩn bị/đánh giá vẫn là giờ công.
- R4 so R2 để đo fine-tuning dense; nếu phục vụ hybrid, thêm R5 = BM25 + R4 và so R5 với R3 = BM25 + R2, giữ RRF/corpus/chunk/query cố định.
- Sau đổi checkpoint, re-encode corpus và đổi index/cache hash; giữ encoder Banking riêng.
- Docs gate chọn trên cùng phương pháp phục vụ; không coi top-5 có gold là bảo đảm câu trả lời đúng. Bổ sung case thiếu bằng chứng có nhãn riêng để kiểm tra từ chối.
- Gate ngày 2/5/8 mới quyết định đổi phạm vi bằng số đo; sửa bug và ghi log vẫn làm ngay. Giữ ngày 13–15 cho bàn giao.
- Ngày 14 mỗi người giải thích module của một người khác. Commit sau mỗi phần chạy được, kèm lệnh chạy/artifact manifest để người khác tiếp quản.

## Nhịp một phiên

5 phút đọc issue → 45 phút thực hiện → 10 phút commit/log/blocker. Leader có khoảng ba phiên/ngày theo giả định quỹ 90 giờ. Giờ máy chạy không giám sát ghi riêng.
