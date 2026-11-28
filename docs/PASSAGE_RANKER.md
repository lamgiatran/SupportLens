# Docs-QA · Xếp hạng đoạn văn cho hội thoại

R7 tìm được đoạn chứa bằng chứng trong ba kết quả đầu cho **1.264/2.213** câu dev. R3 đạt **1.063/2.213** trên cùng 329 hội thoại. Nhóm có thể chọn R7 để demo local. Kết quả về chất lượng câu trả lời còn cần được một người khác chấm độc lập.

## 1. Lâm đã thay đổi gì

Ở câu hỏi nối tiếp, câu trước giúp xác định thủ tục đang nói đến, còn câu hiện tại cho biết người dùng đang hỏi về phí, điều kiện hay cách thực hiện. R7 dùng BM25 để tìm đoạn theo bốn cách đọc hội thoại: toàn bộ query đã đóng gói, câu hiện tại, lượt trước và phần lịch sử còn lại trong ngân sách hai lượt. Mỗi cách lấy tối đa 36 đoạn có điểm dương. Hệ thống thêm các đoạn ngay trước và sau trong cùng tài liệu, rồi dùng 48 cây gradient boosting để sắp lại. Các đặc trưng gồm mức khớp của đoạn, mức khớp tốt nhất trong tài liệu, điểm của đoạn liền kề và độ dài lượt.

Mỗi hội thoại public train đóng góp tối đa một lượt, chọn bằng hash của query ID. Trong 3.474 lượt được chọn, 3.190 có cả đoạn đúng và đoạn sai trong tập ứng viên. 284 lượt còn lại được ghi riêng. Mô hình học trên 613.595 cặp, gồm 3.686 cặp dương theo qrels strict. Chỉ có nhãn train đi vào quá trình fit. Khi phục vụ một câu mới, hệ thống chỉ dùng câu hỏi, lịch sử đã có và tài liệu công khai. Model đóng gói thành JSON khoảng 30 KB. Hai lần huấn luyện cùng cấu hình đã cho file giống nhau tới từng byte.

Hướng thử dựa trên các nghiên cứu về xếp hạng nhiều cách biểu diễn câu hỏi hội thoại và reranking ở MultiDoc2Dial. R7 dùng các đặc trưng từ vựng và cây nông để có thể chạy trên CPU của máy demo. [Conversational Search, EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.354/) · [UGent-T2K, DialDoc 2022](https://aclanthology.org/2022.dialdoc-1.12/)

## 2. Đo retrieval trên toàn bộ dev

| Chỉ số | R1 · BM25 | R3 · BM25 + dense | R7 · xếp hạng lại |
|---|---:|---:|---:|
| Hit@3 strict | 46,68% | 48,03% | **57,12%** |
| Hit@5 strict | 53,77% | 55,13% | **64,53%** |
| Còn thấy bằng chứng sau khi đóng prompt 600 token | 45,23% | 46,18% | **55,04%** |
| MRR@10 | 0,398 | 0,399 | **0,480** |

Hit@3 tính một lượt thành công khi một trong ba đoạn đầu chứa ít nhất 50% của một span gold. Cột bằng chứng nhìn thấy áp dụng cùng tiêu chí lên phần văn bản thật sự tới được model. R7 tăng **9,08 điểm phần trăm** Hit@3 so với R3. Khoảng tin cậy bootstrap 95% theo hội thoại là **+6,92 đến +11,15 điểm**. Ở DMV, SSA, Student Aid và VA, mức tăng lần lượt khoảng 10,6, 3,2, 13,8 và 10,1 điểm. Đây là so sánh trên dev, nơi nhóm phát triển phương pháp. Tập held-out vẫn dành cho lần đánh giá cấu hình đã khóa.

Mỗi retriever có ngưỡng dev riêng hướng tới xử lý 70% câu. Trong số lượt được nhận, R7 thiếu gold ở top‑3 trong **34,30%**, R3 trong **48,65%**. Riêng bước R7 xếp và lấy mười đoạn có p95 **5,90 ms** trên máy chạy thử. [Số liệu theo miền và khoảng tin cậy](../runs/docs_qa/passage_ranker_metrics.json) · [Ngưỡng của R7](../artifacts/docs_gates/R7.json) · [Bản ghi huấn luyện](../artifacts/docs_ranker/training_metrics.json)

## 3. Đo câu trả lời qua API local

Chạy hai lượt cho mỗi câu qua FastAPI với Ollama Qwen2.5 3B thật, mỗi lượt dùng gate riêng của retriever. Hai bên có cùng prompt v2, kiểm tra ví dụ bị chép từ prompt, tối đa ba đoạn, 600 token bằng chứng và 128 token đầu ra. Không dùng lại câu trả lời từ cache. Chọn trước một câu từ mỗi hội thoại trong 32 hội thoại dev khác các vòng đọc lỗi trước. Thứ tự gọi hai cấu hình đổi qua lại. Phép đo sử dụng HTTP client trong tiến trình API, sau warm-up.

| Trên 32 hội thoại | R3 | R7 |
|---|---:|---:|
| Có câu trả lời / toàn bộ | 19/32 | 21/32 |
| Trích ít nhất một đoạn mà phần hiển thị có span gold / toàn bộ | 10/32 | 10/32 |
| Mọi phát biểu có nguồn hỗ trợ / số câu đã trả lời | 7/19 | 10/21 |
| Vừa trả lời đúng yêu cầu vừa có nguồn cho mọi phát biểu / toàn bộ | 5/32 | 9/32 |
| p95 xử lý một request, gồm cả lượt chuyển người | 826 ms | 821 ms |

Hai dòng về nội dung là lượt đọc của Codex với nhãn A/B đảo riêng theo câu. R7 đạt ở 6 câu mà R3 trượt và trượt ở 2 câu mà R3 đạt. Chênh lệch **+12,5 điểm**, khoảng tin cậy 95% **−3,1 đến +31,3 điểm**, kiểm định chính xác theo cặp p = 0,289. Cỡ mẫu này chưa đủ để khẳng định chất lượng trả lời tăng. Chẳng hạn, khi người dùng hỏi người khác có đăng ký xe được không, cả hai câu trả lời đều nhắc việc chính chủ phải nộp biển số. Việc nguồn có nói đến đình chỉ đăng ký chưa đủ để giải đáp câu hỏi về người thứ ba. Một lỗi khác là gán nguồn đúng chủ đề nhưng nhầm đoạn chứa chỉ dẫn. [Các đầu ra thật](../runs/docs_qa/served_answers_audit.json) · [Nhãn và lý do từng câu](../runs/docs_qa/served_review_audit.json) · [ALCE về đánh giá citation](https://aclanthology.org/2023.emnlp-main.398/)

## 4. Hai thử nghiệm ảnh hưởng tới lựa chọn

1. R6 dùng TF‑IDF tìm câu train gần giống, lấy các đoạn gold train bỏ phiếu rồi kết hợp BM25. Hit@3 chỉ đạt 38,05%. Giữ R6 làm đối chứng offline. [Kết quả R6](../runs/docs_qa/query_memory_metrics.json) · [Nghiên cứu truy hồi cặp câu hỏi, PAQ](https://aclanthology.org/2021.tacl-1.65/)
2. Prompt v3 đưa hai lượt trước vào phần làm rõ ngữ cảnh và kiểm tra giá trị số theo nguồn trích. Trong 48 hội thoại dev riêng, v3 đạt cả hai tiêu chí ở 10 câu, v2 đạt 13. Qua phân tích còn phát hiện model có thể chép nguyên ví dụ gia hạn giấy tờ và phí 25 USD vào câu trả lời về nha khoa. Chế độ `v2_checked` giữ nguyên prompt v2, kiểm tra hai mẫu ví dụ cụ thể với phần nguồn đã trích rồi mới hiển thị câu. Cơ chế này phát hiện một kiểu lỗi hẹp. Nó không xác minh mọi phát biểu. [Đầu ra của 48 ca](../runs/docs_qa/context_answers_audit.json) · [Lượt chấm](../runs/docs_qa/context_review_audit.json)

## 5. Chạy demo và tự kiểm tra

Từ gốc repo, chuẩn bị passage bundle và tokenizer theo [RUN_DENSE](RUN_DENSE.md) và tải model Ollama theo [RUN_GENERATION](RUN_GENERATION.md). R7 dùng model JSON đã huấn luyện. Các lệnh sau bật cấu hình thử trên máy local. Mở terminal thứ hai để chạy dòng Streamlit, rồi chọn **Docs-QA → API thật → R7**. Muốn so trực tiếp với R3 trên cùng máy, chuẩn bị thêm dense cache và đặt `SUPPORTLENS_DOCS_DENSE` theo RUN_DENSE trước khi chạy API.

```powershell
$env:SUPPORTLENS_DOCS_BUNDLE = 'data/processed/docs_qa'
$env:SUPPORTLENS_DOCS_GATES = 'artifacts/docs_gates'
$env:SUPPORTLENS_PASSAGE_RANKER = 'artifacts/docs_ranker/model.json'
$env:SUPPORTLENS_GENERATOR = 'ollama'
$env:SUPPORTLENS_GENERATOR_POLICY = 'v2_checked'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Ngưỡng riêng lẻ `SUPPORTLENS_DOCS_GATE` và thư mục nhiều ngưỡng `SUPPORTLENS_DOCS_GATES` là hai lựa chọn loại trừ nhau. Dùng terminal mới cho ví dụ này. Trên máy thử, lần đầu Ollama warm-up vượt mức 30 giây. Startup có hạn riêng 120 giây, mỗi request live vẫn có hạn 30 giây. Bản `v2_checked` đợi kiểm tra xong mới gửi nội dung. p95 thời gian tới câu đầu đủ điều kiện hiển thị, tính trên các lượt đã trả lời, là 822 ms với R3 và 865 ms với R7. Chưa đo thời gian vẽ lên màn hình trình duyệt.

**Chấm độc lập.** Có 40 câu trả lời để chấm từ 32 hội thoại, sáu hội thoại được cả hai cấu hình chuyển người. Với mỗi dòng CSV, đọc câu hỏi và hai lượt trước, rồi đối chiếu từng phát biểu với đúng phần nguồn mà câu trả lời trích. Điền `supported=yes/no` cho tiêu chí mọi ý có căn cứ. Điền `responsive=yes/no` cho tiêu chí giải đáp đúng việc người dùng hỏi. Ghi lý do vào `note`. Trường hợp một phát biểu có số liệu đúng ở một nguồn khác nhưng nguồn được trích không nói điều đó, đánh `supported=no`. Giữ nguyên cột mã ca, A/B và văn bản câu trả lời. Chấm xong mới so với nhãn đã lưu. Lệnh tính điểm dùng lại toàn bộ 32 ca làm mẫu số.

```powershell
python scripts/review_context_answers.py --audit runs/docs_qa/served_answers_audit.json --export review.csv
python scripts/review_context_answers.py --audit runs/docs_qa/served_answers_audit.json --score review.csv
python scripts/review_context_answers.py --audit runs/docs_qa/served_answers_audit.json --score runs/docs_qa/served_review_audit.json
```

**Tái lập thí nghiệm.** Dùng Python 3.11+, NumPy 2.4.2 và scikit-learn 1.8.0 cho lần fit đã ghi. Mỗi lần chạy chọn thư mục đầu ra mới. Lệnh đánh giá retrieval dùng predictions R1/R3 đã sinh theo RUN_DENSE và kiểm tra SHA-256 theo metrics tương ứng. Để chạy hai nhánh API, cần cả dense cache, encoder MiniLM và Ollama local.

```powershell
python scripts/train_passage_ranker.py --out runs/docs_qa/my_fit
python scripts/eval_query_memory.py --ranker runs/docs_qa/my_fit/model.json --out runs/docs_qa/my_dev
python scripts/eval_served_answers.py --model runs/docs_qa/my_fit/model.json --gate runs/docs_qa/my_dev/R7.json --out runs/docs_qa/my_served
```

Codex hỗ trợ triển khai, chạy thí nghiệm và đối chiếu đầu ra. Mọi số về độ chính xác nội dung ở đây vẫn cần lượt chấm độc lập của người thật trước khi dùng làm kết luận cuối kỳ.
