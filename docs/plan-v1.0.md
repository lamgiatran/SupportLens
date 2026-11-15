# SupportLens — Kế hoạch 15 ngày cho nhóm 4 người

Cập nhật: 15/11/2026. Leader làm nhiều hơn; ba thành viên còn lại khoảng 1 giờ/ngày. Giờ công trong kế hoạch là giờ làm chủ động; thời gian máy train/fine-tune qua đêm ghi riêng. Tài liệu gồm bản tóm tắt để giao việc và phụ lục để tra cứu.

## A. Bản tóm tắt cho nhóm

**Mục tiêu:** nộp một ứng dụng có luồng chạy hoàn chỉnh, phần thực nghiệm đáp ứng đề NLP và số đo gắn với đúng cấu hình được demo.

**Quỹ giờ:** Hưng, Cầu, Tuấn mỗi người 15 giờ trong 15 ngày. Tạm lập lịch cho leader 3 giờ/ngày = 45 giờ, tổng nhóm 90 giờ. Đây là giả định kế hoạch, chưa phải thời gian leader đã xác nhận. Bản lõi dành 13 giờ tổng cộng cho dự phòng; fine-tune retriever dùng tối đa 5 giờ chủ động từ phần dự phòng của leader. Thời gian máy chạy qua đêm không cộng vào 90 giờ này.

| Người | Trách nhiệm chính | Bàn giao |
|---|---|---|
| Leader (bạn), 45 giờ chủ động | Dữ liệu tài liệu, retrieval, generator/cache và tích hợp | Docs-QA; BM25/dense/hybrid; fine-tune retriever là mục nâng cao chính; số đo và demo |
| Hưng, 15 giờ | BANKING77, baseline và công cụ đánh giá | E1–E3; E4 dùng embeddings chung nếu kịp; metrics/plots; audit qrels; phần báo cáo |
| Cầu, 15 giờ | Handoff-50 bằng mô hình nhẹ | H1–H4, ngưỡng dev, risk–coverage; phân tích ID-OOS và độ nhạy tỷ lệ OOS |
| Tuấn, 15 giờ | Serving và giao diện | FastAPI, Streamlit, log JSONL, timeout, lệnh chạy; phần hướng dẫn và smoke test |

**Bản nộp:** Docs-QA tìm và trả lời có nguồn trực tiếp trên corpus MultiDoc2Dial; chế độ phân luồng ngân hàng dùng đúng Handoff-50 đã đo; E1–E3 cho phần học phần; 40 kịch bản kiểm tra toàn luồng. Course-77 xuất hiện ở phần thí nghiệm.

| Hạn | Điều phải chạy được |
|---|---|
| Cuối ngày 2 | UI → API → BM25 mức tài liệu trên raw text → bằng chứng; chưa cần qrels |
| Ngày 3–4 | Chunks, mapping grounding span và qrels; audit trước khi đánh giá |
| Cuối ngày 6 | E1–E3, dense/hybrid và Handoff-50 baseline có kết quả dev |
| Cuối ngày 8 | Hai chế độ chạy qua cùng UI/API; generation cache và kiểm tra citation ID hoạt động |
| Cuối ngày 12 | Hoàn tất đánh giá cuối và bản nháp báo cáo; khóa tính năng |
| Ngày 13–15 | Sửa lỗi, chạy từ môi trường mới, quay video, tập trình bày, nộp |

**Mục nâng cao chính của leader:** fine-tune dense retriever từ query–grounding passage của public train MultiDoc2Dial, chạy máy qua đêm, so R4 với R2 frozen. Gate ngày 8: luồng lõi ổn và còn đủ 5 giờ chủ động để chuẩn bị/đánh giá; nếu chưa đạt, làm ngay sau nộp. Fine-tuning classifier 50/77 nhãn, reranker, NLI, MLflow và triển khai public để sau nộp.

**Hoàn thành khi:** người khác chạy được theo README, nhập câu ra bằng chứng/câu trả lời hoặc lý do chuyển người, có số liệu thật và hiểu rõ giới hạn. Mỗi người viết phần mình từ ngày 3.

---

## B. Phụ lục — Giờ làm và cách giao việc

### B1. Ngân sách giờ

| Người | Phân bổ dự kiến | Tổng |
|---|---|---:|
| Leader | Dữ liệu/chunks/qrels và retrieval 14h; generator/cache 5h; tích hợp/review 6h; đánh giá/phân tích 5h; báo cáo/video 7h; dự phòng 8h | 45h |
| Hưng | Dữ liệu, preprocessing, E1–E3/E4 6h; metrics và audit qrels 3h; phân tích/báo cáo/review 4h; dự phòng 2h | 15h |
| Cầu | Split và mô hình 50 nhãn 4h; H1–H4/plots 4h; phân tích lỗi và báo cáo 5h; dự phòng 2h | 15h |
| Tuấn | API 4h; UI 4h; log/timeout 2h; đóng gói/hướng dẫn/kiểm tra 4h; dự phòng 1h | 15h |

Leader dự kiến làm khoảng một nửa số giờ chủ động của nhóm, đồng thời sở hữu retrieval/RAG và tích hợp. Chunks/qrels được dành riêng khoảng 4–6 giờ trong ngân sách dữ liệu/retrieval. Fine-tune retriever dự kiến cần thêm tối đa 5 giờ chủ động: chuẩn bị/audit cặp train 2h, cấu hình và smoke run 1h, kiểm tra checkpoint/re-index/so sánh 2h; lấy từ 8h dự phòng của leader. Nếu mở mục này, nhóm còn 8h dự phòng tổng cộng. Đây là ước lượng công việc, không tự chuyển thành tỷ lệ chấm điểm cá nhân.

| Loại thời gian | Tính vào giờ công chủ động? | Ví dụ |
|---|---|---|
| Thiết kế, viết code, chuẩn bị dữ liệu, cấu hình | Có | Dựng train pairs, sửa parser, chọn loss |
| Máy train/fine-tune qua đêm, không cần theo dõi liên tục | Không; ghi giờ máy/GPU riêng | Job chạy sau khi đã kiểm tra các bước đầu |
| Kiểm tra run, sửa lỗi, phục hồi, đánh giá và phân tích | Có | Xử lý OOM, re-index, đọc lỗi retrieval |

Ví dụ một đêm máy chạy 3 giờ và bạn dùng 30 phút kiểm tra thì ghi 3 giờ máy và 0,5 giờ công, không trừ 3,5 giờ khỏi quỹ chủ động. Thời gian máy vẫn chiếm lịch trước deadline; train lâu hoặc lỗi đêm có thể làm trễ mốc ngày hôm sau.

Mỗi đầu việc của ba thành viên được gói trong 45–60 phút, có input và output cụ thể. Trong một giờ: khoảng 5 phút đọc cập nhật, 45 phút thực hiện, 10 phút commit/ghi kết quả và blocker. Trao đổi bất đồng bộ nằm trong giờ này.

Một issue nên có một owner, một artifact và cách kiểm tra: chẳng hạn “chạy evaluate trên file dự đoán mẫu, xuất metrics.json”. Leader chuẩn bị contract và dữ liệu mẫu để các bạn làm độc lập; commit sau mỗi phần chạy được, cập nhật lệnh chạy và artifact manifest để Tuấn có thể khởi động nhánh tích hợp khi cần. Nếu vướng quá 15 phút, ghi lỗi để leader hỗ trợ ở phiên tích hợp; các bạn vẫn giữ trách nhiệm hoàn thiện phần được giao.

Nếu leader chỉ có 2h/ngày, tổng còn 75h: giữ E1–E3, giảm generation review xuống 20–30 câu, bỏ E4 và chuyển fine-tune retriever sang sau nộp vì vẫn cần giờ chủ động chuẩn bị/đánh giá. Nếu có 4h/ngày, tổng 105h: ưu tiên kiểm tra chạy lại và phân tích lỗi trước khi mở thêm mô hình. Các số này là kịch bản lập lịch.

### B2. Phần bắt buộc của môn

Đã đối chiếu mlAssignments_v1.1_english.pdf, v1.1, 6 trang:

- Ít nhất ba kỹ thuật preprocessing có giải thích: chuẩn hóa, tokenization, lemmatization cho nhánh ML; có ablation nhỏ về tác động.
- Ít nhất hai biểu diễn và hai mô hình: BoW/TF-IDF với Logistic Regression/Linear SVM đã đủ các mức tối thiểu này.
- Accuracy, Precision/Recall/F1, confusion matrix, so sánh và phân tích lỗi.
- Báo cáo đủ tám phần; source preprocessing/train/evaluate; bảng đóng góp.
- Demo được khuyến khích. Fine-tuning transformer không phải điều kiện bắt buộc trong đề.
- Hạn chính thức lấy trên LMS; Ngày 1–15 là lịch tương đối của nhóm, không tự suy ra ngày nộp.

## C. Sản phẩm, tài nguyên và độ trễ

### C1. Hai chế độ trong cùng ứng dụng

**Docs-QA, chế độ chính:** người dùng chọn kho MultiDoc2Dial, nhập câu và lịch sử trước đó → truy xuất → lấy bằng chứng → soạn trả lời có citation → hiện câu trả lời, bằng chứng, thời gian và phản hồi.

**Banking triage:** nhập câu → classifier 50 nhãn → acceptance score → đề xuất intent hoặc chuyển người, kèm lý do. Dùng đúng checkpoint/ngưỡng được đánh giá. Course-77 phục vụ báo cáo, không thay checkpoint của chế độ này.

Classifier ngân hàng không gate câu hỏi của MultiDoc2Dial. UI chọn chế độ tường minh; dùng chung API contracts, log, xử lý lỗi và cách hiện kết quả. Tính liền mạch nằm ở trải nghiệm và luồng vận hành chung; số đo được ghi rõ theo từng tác vụ.

Bỏ việc tự viết cả KB ngân hàng trong bản nộp. Nếu cần minh họa, tối đa 3–5 tài liệu giả lập và ghi rõ tính chất; phần này chỉ mở sau gate ngày 8, không lấy số đo MultiDoc2Dial gán sang.

Contract tối thiểu gồm request_id, mode, artifact_version, decision/reason, evidence/citations và latency_ms theo tầng. Log JSONL đủ cho bản nộp. Tuấn giữ API/UI; leader giữ generator, retrieval và cache.

### C2. Chốt generator sớm

Chốt runtime mặc định ngay ngày 1: Ollama local với qwen2.5:1.5b, bản Q4_K_M hiện được phân phối chính thức. Ghi model digest, quantization, Ollama version và cấu hình máy; tag chỉ là tên thuận tiện, digest dùng nhận diện phiên bản. Leader phụ trách generator adapter; Tuấn nối stream từ API tới UI. Không lấy Transformers fp32 CPU làm cấu hình demo mặc định. Colab dành cho train/fine-tune, không là dependency bắt buộc của inference demo.

Thử ít nhất 10 câu dev, context tối đa khoảng 1.500 token, đầu ra tối đa 128 token, decoding cố định. Bật streaming từ generator qua API tới UI, hiện evidence sớm rồi hiển thị dần nội dung trả lời. Ngày 2 chốt cấu hình và đo TTFT. Nếu phần cứng không chạy ổn, bản nộp chuyển sang trả đoạn trích có nguồn và ghi đúng là chế độ trích xuất; generator trở thành phần nâng cấp. Không gọi đoạn trích là câu trả lời sinh bởi LLM.

Mọi generation thành công và lỗi đều ghi JSONL. Khóa cache bao gồm model/revision, prompt version, câu hỏi và lịch sử, evidence IDs/content hash, decoding parameters. Lưu raw response, citations, timing gồm TTFT/tổng thời gian, trạng thái, token usage nếu runtime cung cấp. Stream bị ngắt phải lưu partial response và cờ incomplete, không cache như một câu trả lời hoàn chỉnh. Đổi metric hoặc review lại dùng output đã lưu; đổi prompt/evidence tạo entry mới.

Cache dùng để tái đánh giá; báo riêng cache hit, cache miss và chế độ replay. Demo trực tiếp nhận câu mới phải thực hiện inference thật hoặc chuyển sang fallback rõ ràng khi lỗi. Video/cache replay không thay số đo inference live.

### C3. Latency budget khởi đầu

Các mức sau là mục tiêu thiết kế, chưa phải kết quả đo hay cam kết phần cứng. Ngày 1 ghi CPU/GPU/RAM/runtime, ngày 2 chốt target theo máy demo; sau đó giữ làm điểm quy chiếu.

| Thành phần | Target p95 khi đã warm-up, batch 1, concurrency 1 |
|---|---:|
| Banking triage trọn nhánh, gồm embedding | ≤ 300 ms |
| Retrieval gồm query embedding và xếp hạng | ≤ 500 ms |
| TTFT end-to-end: gửi câu hỏi đến token nội dung đầu tiên hiện trên UI | ≤ 3 giây |
| Generation, tối đa 128 token | ≤ 20 giây |
| Docs-QA end-to-end, cache miss | ≤ 22 giây |

TTFT end-to-end gồm retrieval, xếp hàng, prefill và truyền tới UI; ghi thêm TTFT riêng của generator nếu đo được. Byte HTTP đầu tiên, metadata hay animation đang tải không được tính là token nội dung. Streaming cải thiện thời gian chờ cảm nhận nhưng không tự giảm prefill hay bảo đảm đạt 3 giây. Nếu target chưa đạt, thử giảm context/output trên dev và đo lại; công bố mức thực đạt.

Đo ít nhất 100 request triage/retrieval và 50 request generation nếu generator được giữ, dùng bộ input cố định; công bố sample size và coi p95 với ít mẫu là ước lượng sơ bộ. Báo riêng cold start, thời gian nạp model, failures và replay; không trộn cache hit vào latency model.

Chọn BM25/dense/hybrid theo chất lượng dev và budget. Hybrid không cải thiện đủ hoặc vượt budget thì phục vụ phương pháp tốt hơn thực đo. Vẫn báo cả ba cấu hình ở bảng so sánh khi đã chạy.

## D. Thực nghiệm tối thiểu và protocol

### D1. Course-77 — Hưng

| Mã | Biểu diễn | Mô hình | Ưu tiên |
|---|---|---|---|
| E1 | BoW | Logistic Regression | Bắt buộc |
| E2 | TF-IDF | Logistic Regression | Bắt buộc |
| E3 | TF-IDF | Linear SVM | Bắt buộc |
| E4 | Frozen all-MiniLM-L6-v2 | Logistic Regression | Thêm nếu hoàn thành trước ngày 5 |

E1–E3 đều có huấn luyện tham số Logistic Regression/Linear SVM; frozen encoder chỉ có nghĩa phần encoder không được cập nhật. Mục nâng cao R4 bổ sung việc fine-tune encoder của leader.

BANKING77 bản gốc: official test giữ nguyên; train chia fit/dev theo nhãn, seed 42. Vocab, preprocessing có tham số học và model chỉ fit train_fit. Audit trùng/gần trùng, lưu split IDs, label mapping, revision và cách xử lý. Báo Macro-F1, Accuracy, per-class report và confusion matrix; giải thích 10–15 ca lỗi tiêu biểu.

### D2. Handoff-50 — Cầu

Bản nhẹ dùng frozen encoder all-MiniLM-L6-v2 chưa supervised fit BANKING77, cùng một multinomial Logistic Regression 50 nhãn. Embeddings được cache; classifier 77 và 50 có artifact/split riêng. Điều này giảm việc fine-tune và giữ đúng ý nghĩa 27 intent chưa được hỗ trợ.

Dùng split BANKING77-OOS phát hành bởi tác giả. Tách một phần ID train làm calibration; phần còn lại fit classifier/reference bank. Fit scalar temperature trên ID calibration bằng NLL. Chọn k/ngưỡng/phương pháp trên dev; nếu dùng dev OOS thì ghi rõ. Test chỉ mở sau freeze. OOS train và 27 nhãn bị giữ lại không tham gia fit hay reference bank.

| Mã | Acceptance score, càng cao càng được ưu tiên chấp nhận |
|---|---|
| H1 | max softmax(z), với z là decision scores của classifier |
| H2 | max softmax(z/T_cal), T_cal fit trên calibration |
| H3 | Âm khoảng cách tới láng giềng thứ k trên embedding chuẩn hóa; bank chỉ train_fit ID |
| H4 | logsumexp(z) với nhiệt độ energy cố định 1, tương đương âm energy |

H4 là baseline inference trên scores sẵn có; không nhận là mô hình được huấn luyện bằng energy loss. Cố định metric và k của H3 bằng dev. Temperature scaling không đổi argmax; tác động lên calibration/OOS phải đo.

Giả thuyết ghi trước khi mở test: H2 có thể chỉ thay đổi ít so với H1 về risk–coverage trên LR. Đây là giả thuyết để kiểm chứng, không phải kết luận; tối ưu log-loss khi train không bảo đảm T_cal gần 1 hoặc calibration tốt trên dữ liệu mới. Cầu fit một scalar temperature theo NLL, ghi T_cal/NLL/ECE/AURC rồi dừng; dành giờ cho H3/H4 và ma trận ID-OOS. Báo cả kết quả không ủng hộ giả thuyết.

Risk–coverage tính một accepted case là sai khi nhãn ID sai hoặc câu OOS được chấp nhận. Báo coverage, accepted-error, AURC, ID coverage/error và false acceptance riêng cho ID-OOS/OOD-OOS. Khi không có ca được chấp nhận, risk là N/A. Ngưỡng vận hành lấy từ dev; test báo coverage thực đạt. Nếu làm CI, resample theo query, dùng cùng query cho hai phương pháp.

Với bản frozen encoder + solver cố định, không trình bày ba lần chạy gần như giống nhau như bằng chứng về độ ổn định fine-tuning. Cầu không gánh thêm fine-tuning classifier 50/77 nhãn trong 15 giờ; mục fine-tuning ưu tiên chuyển sang retriever của leader ở D5.

Hai phân tích ngắn bắt buộc, dùng lại file dự đoán:

- Ma trận 27 intent ID-OOS × 50 nhãn dự đoán, chuẩn hóa theo hàng; ghi top intent bị hút vào và một số cặp gần nhau trong embedding.
- Reweight tỷ lệ OOS tổng thành 5%, 10%, 20%; giữ tỷ lệ ID-OOS:OOD-OOS theo benchmark làm giả định chính, báo thêm riêng hai loại nếu kịp. Với cùng ngưỡng dev, coverage = Σw·accepted/Σw; accepted-error = Σw·accepted·wrong/Σw·accepted. Ghi rõ đây là phân tích giả định, không phải lưu lượng sản phẩm đã quan sát.

### D3. Retrieval và câu trả lời — Leader, Hưng review

R1 = BM25; R2 = dense MiniLM; R3 = hybrid RRF. Dùng cùng corpus, chunks, query IDs, candidate pool và qrels. Mặc định tìm trên toàn corpus, không cấp gold domain/doc ID cho retriever.

Ngày 1 chỉ xác nhận raw document/dialogue JSON đọc được và chọn cách chia passage; việc đối chiếu script token/structure của IBM nằm trong phiên chunks ngày 3, giới hạn tối đa một giờ. Ưu tiên cấu trúc đoạn có sẵn nếu dùng được; nếu phải viết adapter, lưu chunk-to-span mapping và config. Chỉ đặt cạnh số công bố khi cả segmentation, query/history, corpus, split và metric tương thích; cùng cách chia passage riêng lẻ chưa đủ để gọi là tái lập.

Lấy query từ lượt user và lịch sử đã có trước câu trả lời cần dự đoán. Target response và grounding của nó chỉ dùng dựng nhãn đánh giá. Không index các câu trả lời hội thoại.

Public test trong bản dữ liệu này là dummy; chia public validation theo dialogue ID thành dev và held-out evaluation, giữ tỷ lệ domain nếu có thể, lưu danh sách cố định. Corpus tài liệu được dùng chung theo thiết lập retrieval, còn hội thoại đánh giá phải tách biệt.

Ngày 2 chỉ BM25 mức document từ raw text để nối sản phẩm; chưa có qrels và chưa công bố retrieval metrics. Ngày 3–4 mới dựng chunks, giữ doc_id/character offsets và ánh xạ grounding span sang chunk để tạo qrels. Span qua biên có thể có nhiều chunk liên quan; nhiều reference giữ thành tập positives, không bỏ tùy tiện. Audit khoảng 20 ca gồm cả hai tình huống này, lưu quy tắc relevance/coverage và thống kê span không ánh xạ được. Grounding không phải danh sách đầy đủ mọi đoạn đúng. Chạy retrieval trên toàn query held-out đã chọn, báo Recall@5, MRR@10 và latency; nêu rõ cách tính relevance và số query.

Generation dùng retriever đã chọn trên dev, cùng model/prompt/config của ứng dụng. Chọn cố định 40 query held-out có phân bố domain để review; thêm BM25 đối chứng trên cùng 40 query nếu còn giờ. Đây là đánh giá nhỏ với giới hạn sample size rõ ràng.

Dành khoảng 1h trong ngân sách review/phân tích của leader cho lỗi citation: cite sai chunk, claim vượt bằng chứng, bỏ nguồn hoặc câu hỏi thiếu thông tin. Phân biệt lỗi truy xuất với lỗi sinh câu trả lời; model nhỏ có thể mắc các lỗi này, mức độ phải đo. Kết quả yếu cũng được báo cáo đầy đủ.

Bộ kiểm tra tự động chỉ xác minh citation ID tồn tại và nằm trong evidence được cấp. Người review đánh giá câu trả lời có được bằng chứng hỗ trợ, đáp ứng câu hỏi và còn claim thiếu nguồn hay không. Review chéo ít nhất 20/40 câu, giải quyết bất đồng; lưu rubric và annotation. NLI checker/agreement với người được chuyển sang v2.

### D4. Kiểm tra toàn luồng — 40 kịch bản

- 20 Docs-QA: 16 câu có bằng chứng/lịch sử và 4 ca thiếu bằng chứng được tạo có kiểm soát.
- 12 Banking triage: 6 ID, 4 ID-OOS, 2 OOD-OOS.
- 8 ca vận hành: rỗng, quá dài, timeout generator, model chưa nạp, citation sai, log lỗi, đổi mode và khởi động lại.

Các kịch bản đi qua UI/API và nhánh xử lý liên quan, có expected behavior cùng pass/fail. Báo riêng smoke test và benchmark; không gộp 40 ca này thành accuracy tổng quát. Ghi rõ generator live, extraction fallback hay replay trong từng lần chạy.

### D5. Mục nâng cao chính — Fine-tune dense retriever của leader

R4 = cùng encoder MiniLM của R2 được fine-tune bằng MultipleNegativesRankingLoss (MNRL). Dữ liệu gồm query và grounding passage của public train MultiDoc2Dial; dùng cùng cách tạo query/chunks như ứng dụng. So R4 với R2 frozen trên cùng corpus, qrels và held-out dialogue.

- Tách theo dialogue ID, kiểm tra train không giao dev/held-out; audit query trùng chính xác hoặc gần trùng giữa các split. Dev chỉ chọn cấu hình/checkpoint, held-out dùng đánh giá cuối. Không dùng gold answers/grounding dev hoặc held-out để tạo training pairs hay hard negatives.
- Grounding của target response chỉ dùng gán positive cho query từ user/history trước response đó. Lưu toàn bộ positives đã biết của query. Văn bản corpus chung được index như thiết lập retrieval; tách hội thoại không đồng nghĩa đánh giá trên tài liệu chưa từng thấy.
- MNRL coi passage của cặp khác trong batch là negative. Dùng NO_DUPLICATES, đồng thời lọc xung đột positive đã biết; tránh passage trùng/giao nhau hoặc positive khác của cùng query xuất hiện như negative. Sampler loại trùng không tự giải quyết mọi false negative; nếu phải chọn một positive/query, lưu quy tắc và phân tích giới hạn.
- Khởi đầu một cấu hình nhỏ, chẳng hạn 1 epoch, batch thực 16–32 tùy VRAM, LR 2e-5; kiểm tra 10–20 bước trước khi để máy chạy đêm. Đây là cấu hình thử, không phải cam kết tối ưu. Lưu max sequence length và bảo đảm passage positive không bị tokenizer cắt mất phần bằng chứng.
- Mỗi run lưu config, split hash, seed, train loss, checkpoint, optimizer/scheduler state nếu cần resume. Xuất checkpoint/log khỏi vùng tạm của Colab. Ghi giờ GPU, phần cứng và số cặp train riêng với giờ công.
- Sau fine-tune, encode lại cả corpus và query bằng cùng checkpoint R4; không dùng index R2 với query encoder R4. Cache/index key chứa checkpoint và corpus/chunk hash. Encoder cho Banking triage giữ artifact frozen riêng.
- Ưu tiên ba seed 42/43/44 với cấu hình đã chốt, chạy tuần tự qua đêm nếu tài nguyên cho phép; báo mean ± std. Nếu chỉ hoàn thành một seed, ghi kết quả thăm dò, không nhận đã đo độ ổn định. Bootstrap theo dialogue cho chênh lệch R4–R2 là CI do mẫu đánh giá, khác biến thiên do seed.
- Báo Recall@5, MRR@10 và latency p95 trên cùng phần cứng; có thể không cải thiện. Chọn phương pháp/seed phục vụ trước khi xem held-out; mặc định seed 42 nếu giữ R4. Đổi retriever phải cập nhật generation cache và đánh giá pipeline đúng artifact mới.

Tối ngày 8: chỉ mở full run nếu luồng lõi đã ổn và còn đủ giờ chủ động trong budget. Dự kiến đêm 8 chạy seed 42, ngày 9 kiểm tra/re-index/dev, đêm 9 chạy seed còn lại cùng cấu hình nếu kịp. Không cam kết “xong trong 1–2 giờ” trước khi đo trên số cặp và GPU thực tế. Nếu gate không đạt hoặc cần sửa quá ngân sách, fine-tune retriever là việc đầu tiên sau nộp.

## E. Lịch thực hiện theo đầu việc

| Ngày | Leader, khoảng 3h chủ động | Hưng, khoảng 1h | Cầu, khoảng 1h | Tuấn, khoảng 1h |
|---|---|---|---|---|
| 1 | Repo/contracts; đọc raw Docs; Ollama Q4 và probe TTFT | Tải/audit BANKING77 | Tải/audit split OOS | API trả response mẫu |
| 2 | BM25 document-level; ghép UI/API/evidence, chưa qrels | Preprocessing + E1 | Split/calibration manifest | UI nhập và hiện evidence |
| 3 | Chunks và offsets; kiểm tra parser/script gốc | E2 | Fit classifier 50 | API gọi artifact mẫu |
| 4 | Qrels, span qua biên/nhiều reference; audit | E3 | H1/H2/H4, ghi giả thuyết | Log, lỗi đầu vào |
| 5 | Dense và cache embeddings sau khi mapping ổn | Metrics; E4 nếu kịp | H3 và ngưỡng dev | UI evidence; nhận stream |
| 6 | Hybrid, retrieval dev; generator với evidence thật | Confusion/errors | Risk–coverage | Nối stream API tới UI |
| 7 | Cache/citation; chuẩn bị train pairs nếu kịp | Eval harness | Sensitivity từ predictions | Timeout và stream ngắt |
| 8 | Hai mode chạy; gate retriever FT, smoke 10–20 bước | Audit qrels/train-pair sample | Ma trận ID-OOS | Smoke test bản đầu |
| 9 | Kiểm tra run đêm; re-index/dev; chốt retriever/prompt | Bảng kết quả dev | Chốt handoff config | README/lệnh chạy |
| 10 | Freeze artifacts; chạy held-out; R4 chỉ khi sẵn sàng | Metrics cuối | Chạy test OOS | Kiểm tra schema/log |
| 11 | Review citation; TTFT/latency; phân tích R2/R4 | Review/viết báo cáo | Phân tích lỗi/viết | Chạy lại 40 ca |
| 12 | Tổng hợp báo cáo, khóa tính năng | Hoàn thiện phần mình | Hoàn thiện phần mình | Hướng dẫn/đóng gói |
| 13 | Môi trường mới và sửa blocker | Kiểm tra bảng/nguồn | Kiểm tra model/split | Cài theo README |
| 14 | Video; mỗi người giải thích module người khác | Giải thích Handoff của Cầu | Giải thích retrieval của leader | Giải thích baseline của Hưng |
| 15 | Kiểm tra gói nộp, nộp sớm | Rà báo cáo | Rà artifacts | Rà hướng dẫn |

Báo cáo viết dần từ ngày 3. Bảng giờ là giờ chủ động, gồm trao đổi, cấu hình run và viết tài liệu; máy train qua đêm ghi ở log riêng. Leader giải thích module serving của Tuấn trong vòng hỏi chéo ngày 14; bốn người đổi vai trong cùng phiên tập bảo vệ, không phát sinh họp ngoài quỹ giờ. Đến ngày 8 chưa có luồng ổn thì cắt model/tính năng mở rộng, dùng dự phòng để tích hợp. Ngày 13–15 chỉ sửa lỗi bàn giao.

## F. Sau nộp và phần đóng góp của leader

Leader tiếp tục sở hữu retrieval/RAG: một issue từ log → phân tích lỗi → thay đổi retriever/prompt → regression check → số đo trước/sau. Đây là phần đủ sâu để trình bày khi phỏng vấn: cách dựng qrels, chọn hybrid hay BM25, quản lý cache, giới hạn nguồn và xử lý lỗi inference.

Thứ tự nâng cấp: hoàn thành fine-tune retriever nếu chưa qua gate trước nộp; sau đó (1) deploy và log vận hành; (2) người ngoài nhóm dùng thử; (3) sửa vấn đề có bằng chứng; (4) reranker/NLI hoặc mở rộng thí nghiệm khi kết quả cho thấy cần. Ghi đúng số người và dữ liệu đã có; chưa tự chốt thời gian kết thúc sau nộp.

Bàn giao v1 gồm source/config/split IDs, model artifacts hoặc cách lấy, output JSONL, metrics/plots, bảng smoke test, báo cáo tám phần, README và video. Bảng đóng góp dựa trên artifact/PR thực tế; cả bốn người giải thích được phần mình và luồng chung.

## G. Nguồn và trạng thái

- Đề học phần: mlAssignments_v1.1_english.pdf, v1.1, 6 trang, đã đọc lại.
- [BANKING77 gốc](https://github.com/PolyAI-LDN/task-specific-datasets).
- [BANKING77-OOS, repo tác giả](https://github.com/jianguoz/Few-Shot-Intent-Detection).
- [MultiDoc2Dial data README](https://doc2dial.github.io/multidoc2dial/data_readme.html), gồm lưu ý test dummy.
- [Script passage token/structure của IBM](https://github.com/IBM/multidoc2dial/blob/main/scripts/run_data_preprocessing.sh).
- [Energy-based OOD Detection](https://arxiv.org/abs/2010.03759).
- [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).
- [Ollama qwen2.5:1.5b, bản Q4_K_M](https://ollama.com/library/qwen2.5:1.5b).
- [Ollama streaming API](https://docs.ollama.com/api/streaming).
- [Sentence Transformers — MNRL](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.MultipleNegativesRankingLoss).
- [Sentence Transformers — batch samplers](https://www.sbert.net/docs/package_reference/sentence_transformer/sampler.html).
- [Dataset card MultiDoc2Dial](https://huggingface.co/datasets/IBM/multidoc2dial): metadata và mục Licensing Information hiện ghi khác nhau. Ngày 1 đối chiếu notice kèm bản tải, lưu revision/attribution; repo dùng downloader/manifest.

Đây là kế hoạch đã điều chỉnh, chưa có code, model huấn luyện hay kết quả benchmark được tạo trong lần cập nhật này. Mọi latency, TTFT và giờ công ở trên là mục tiêu/ước lượng. Thời gian train/fine-tune không giám sát qua đêm được tách khỏi giờ chủ động, nhưng vẫn chịu mốc lịch và tài nguyên thực tế. Các nguồn dữ liệu cần được tải và audit trong giai đoạn thực hiện.
