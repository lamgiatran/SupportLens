# SupportLens — Kế hoạch 15 ngày cho nhóm 4 người

Cập nhật: 16/11/2026. Công việc được gom theo chặng A–D, thống nhất với Word phân công, [TEAM_HANDOFF.md](TEAM_HANDOFF.md) và [bảng công việc](../issues/README.md). Mỗi chặng gồm vài phiên làm, chuyển tiếp theo đầu ra đã nhận được. Các mốc ngày 2, 5 và 8 chỉ dùng kiểm tra tiến độ và điều chỉnh phạm vi.

Leader làm nhiều hơn; ba thành viên còn lại khoảng 1 giờ/ngày. Giờ công là giờ làm chủ động; thời gian máy tự train/fine-tune được ghi riêng. Phần A và E dùng điều phối, phần D giữ protocol thực nghiệm. Trạng thái code và số đo đã có xem [VERIFICATION_v0.4.md](VERIFICATION_v0.4.md).

## A. Bản tóm tắt cho nhóm

**Mục tiêu:** nộp một ứng dụng có luồng chạy hoàn chỉnh, phần thực nghiệm đáp ứng đề NLP và số đo gắn với đúng cấu hình được demo.

**Quỹ giờ:** Hưng, Cầu, Tuấn mỗi người 15 giờ trong 15 ngày: khoảng 12 giờ cho A–D, 1 giờ ghép bài/hỏi chéo và 2 giờ dự phòng. Leader tạm dự trù 3 giờ/ngày = 45 giờ, tổng nhóm 90 giờ; đây vẫn là giả định kế hoạch. Dự phòng tổng cộng 14 giờ, gồm 8 giờ của leader và 2 giờ của mỗi bạn. Fine-tune retriever chỉ dùng tối đa 5 giờ chủ động từ dự phòng của leader khi luồng lõi ổn. Giờ máy tự chạy không cộng vào quỹ chủ động.

| Người | Trách nhiệm chính | Bàn giao |
|---|---|---|
| Leader (bạn), 45 giờ chủ động | Docs-QA, generator/cache, công cụ đánh giá dùng chung và nghiệm thu | Giao Tuấn retrieval trước, generator/stream sau; nhận metrics và phân tích các nhánh để ghép báo cáo. R4/R5 là mục nâng cao |
| Hưng, 15 giờ | Course-77 và review qrels | E1–E3, một ablation preprocessing, metrics/confusion, phân tích lỗi và báo cáo; gửi leader nhận xét 20 ca qrels sớm. E4 chỉ mở khi còn giờ |
| Cầu, 15 giờ | Trọn Banking Handoff-50, từ encode đến predict | Audit/calibration, LR 50 nhãn và H1–H4; giao Tuấn pipeline nhận text ở bản đầu và cấu hình/ngưỡng ở bản chốt; giao leader phân tích OOS |
| Tuấn, 15 giờ | API/UI, nạp model, hội thoại, streaming và kiểm tra tích hợp | Ghép module của Cầu/leader; giao ứng dụng chạy local, kết quả smoke, timing và hướng dẫn chạy |

**Bản nộp:** Docs-QA tìm và trả lời có nguồn trực tiếp trên corpus MultiDoc2Dial; chế độ phân luồng ngân hàng dùng đúng Handoff-50 đã đo; E1–E3 cho phần học phần; 40 kịch bản kiểm tra toàn luồng. Course-77 xuất hiện ở phần thí nghiệm.

| Chặng | Đầu ra để chuyển tiếp | Mốc phối hợp |
|---|---|---|
| A Chuẩn bị | Input, split/rule và ví dụ request/response rõ; Tuấn gọi API Docs hiện có; Cầu/Hưng bắt đầu phần riêng | Khoảng 2–3 phiên đầu; mốc ngày 2 kiểm tra đường UI → API → evidence |
| B Ghép bản đầu | Cầu giao predict(text) thật để Tuấn nối Banking; Hưng có E2/E3; leader thử generator độc lập | Khoảng 2–3 phiên tiếp; mốc ngày 5 kiểm tra module thật qua API |
| C Hoàn thiện trên dev | Chốt score/ngưỡng và cấu hình bằng dev; nối stream, cache, nguồn và lỗi; thử hai chế độ | Khoảng 2–3 phiên tiếp; mốc ngày 8 quyết định giữ/cắt R4/R5 theo giờ còn lại |
| D Đánh giá và bàn giao | Freeze, benchmark cuối, 40 smoke, báo cáo và cách chạy lại; mỗi người giải thích được phần mình | Các phiên còn lại; khóa tính năng trước ngày 13, dành ngày 13–15 sửa lỗi và bảo vệ |

Docs-QA đã có parser, chunks/qrels và R1/R2/R3 trên dev. Dùng tiếp nền hiện có và audit phần liên quan; không lập lại các bước ngày đầu đã hoàn thành. Các chặng A–D có thể chạy song song giữa các thành viên, không đợi cả nhóm xong một chặng mới bắt đầu việc độc lập.

**Mục nâng cao chính của leader:** fine-tune dense retriever từ query–grounding passage của public train MultiDoc2Dial trên Colab, GPU mục tiêu A100. So R4 với R2; nếu phục vụ hybrid thì so thêm R5 = BM25 + R4 với R3. Smoke run quyết định giờ máy. Tại mốc ngày 8, chỉ mở mục này khi luồng lõi ổn và còn đủ 5 giờ chủ động; nếu chưa đạt, chuyển sau nộp. Fine-tuning classifier 50/77 nhãn, reranker, NLI, MLflow và triển khai public để sau nộp.

**Hoàn thành khi:** người khác chạy được theo hướng dẫn, nhập câu ra bằng chứng/câu trả lời hoặc lý do chuyển người, có số liệu thật và hiểu rõ giới hạn. Mỗi người viết phần mình từ khi có kết quả đầu tiên, giao leader ở D.

---

## B. Phụ lục — Giờ làm và cách giao việc

### B1. Ngân sách giờ

| Người | Phân bổ dự kiến | Tổng |
|---|---|---:|
| Leader | Dữ liệu/chunks/qrels và retrieval 14h; generator/cache 5h; tích hợp/review 6h; đánh giá/phân tích 5h; báo cáo/video 7h; dự phòng 8h | 45h |
| Hưng | A–B dữ liệu/E1–E3, ablation và review qrels 6h; C–D phân tích, đánh giá và báo cáo 6h; ghép bài/hỏi chéo 1h; dự phòng 2h | 15h |
| Cầu | A–B audit/calibration và pipeline LR/H1 6h; C–D H2–H4, OOS và báo cáo 6h; ghép bài/hỏi chéo 1h; dự phòng 2h | 15h |
| Tuấn | A–B UI/contract và nối Banking 6h; C–D stream, trạng thái, smoke và hướng dẫn 6h; ghép bài/hỏi chéo 1h; dự phòng 2h | 15h |

Leader dự kiến làm khoảng một nửa số giờ chủ động của nhóm, đồng thời sở hữu retrieval/RAG và nghiệm thu tích hợp. Ngân sách dữ liệu/retrieval bao gồm phần chunks/qrels đã có, không giao làm lại. Fine-tune retriever dùng tối đa 5 giờ chủ động: chuẩn bị/audit cặp train 2h, cấu hình và smoke run 1h, kiểm tra checkpoint/re-index/so sánh 2h; lấy từ 8h dự phòng của leader. Nếu mở mục này, nhóm còn 9h dự phòng tổng cộng. Đây là ước lượng công việc, không tự chuyển thành tỷ lệ chấm điểm cá nhân.

| Loại thời gian | Tính vào giờ công chủ động? | Ví dụ |
|---|---|---|
| Thiết kế, viết code, chuẩn bị dữ liệu, cấu hình | Có | Dựng train pairs, sửa parser, chọn loss |
| Máy train/fine-tune qua đêm, không cần theo dõi liên tục | Không; ghi giờ máy/GPU riêng | Job chạy sau khi đã kiểm tra các bước đầu |
| Kiểm tra run, sửa lỗi, phục hồi, đánh giá và phân tích | Có | Xử lý OOM, re-index, đọc lỗi retrieval |

Ví dụ một đêm máy chạy 3 giờ và bạn dùng 30 phút kiểm tra thì ghi 3 giờ máy và 0,5 giờ công, không trừ 3,5 giờ khỏi quỹ chủ động. Thời gian máy vẫn chiếm lịch trước deadline; train lâu hoặc lỗi đêm có thể làm trễ mốc ngày hôm sau.

Mỗi gói A–D của ba thành viên kéo dài khoảng 2–3 phiên, mỗi phiên 45–60 phút. Có thể tiếp tục cùng tác vụ qua nhiều phiên đến khi đạt đầu ra. Trong một giờ: khoảng 5 phút đọc cập nhật, 45 phút thực hiện, 10 phút commit/ghi kết quả và blocker. Trao đổi bất đồng bộ nằm trong giờ này.

Một lần bàn giao chỉ cần commit/PR, một cách kiểm tra và ba ví dụ input/output. Người nhận chạy khớp rồi thử qua API. Manifest của model/index được cập nhật khi artifact đổi; không viết thêm README/manifest cho mỗi lần gửi. Leader duy trì công cụ đánh giá dùng chung; Hưng dùng công cụ phù hợp để đo Course-77 và review qrels. Nếu vướng quá 15 phút, ghi lỗi tái hiện để leader hỗ trợ; owner tiếp tục sửa phần mình.

Nếu leader chỉ có 2h/ngày, tổng còn 75h: giữ E1–E3, giảm generation review xuống 20–30 câu, bỏ E4 và chuyển fine-tune retriever sang sau nộp vì vẫn cần giờ chủ động chuẩn bị/đánh giá. Nếu có 4h/ngày, tổng 105h: ưu tiên kiểm tra chạy lại và phân tích lỗi trước khi mở thêm mô hình. Các số này là kịch bản lập lịch.

### B2. Phần bắt buộc của môn

Đã đối chiếu mlAssignments_v1.1_english.pdf, v1.1, 6 trang:

- Ít nhất ba kỹ thuật preprocessing có giải thích: chuẩn hóa, tokenization, lemmatization cho nhánh ML; có ablation nhỏ về tác động.
- Ít nhất hai biểu diễn và hai mô hình: BoW/TF-IDF với Logistic Regression/Linear SVM đã đủ các mức tối thiểu này.
- Accuracy, Precision/Recall/F1, confusion matrix, so sánh và phân tích lỗi.
- Báo cáo đủ tám phần; source preprocessing/train/evaluate; bảng đóng góp.
- Demo được khuyến khích. Fine-tuning transformer không phải điều kiện bắt buộc trong đề.
- Hạn chính thức lấy trên LMS; khung 15 ngày và các mốc kiểm tra là lịch tương đối của nhóm, không tự suy ra ngày nộp.

## C. Sản phẩm, tài nguyên và độ trễ

### C1. Hai chế độ trong cùng ứng dụng

**Docs-QA, chế độ chính:** người dùng chọn kho MultiDoc2Dial, nhập câu và lịch sử trước đó → truy xuất → lấy bằng chứng → soạn trả lời có citation → hiện câu trả lời, bằng chứng, thời gian và phản hồi.

**Banking triage:** nhập câu → classifier 50 nhãn → acceptance score → đề xuất intent hoặc chuyển người, kèm lý do. Dùng đúng checkpoint/ngưỡng được đánh giá. Course-77 phục vụ báo cáo, không thay checkpoint của chế độ này.

Classifier ngân hàng không gate câu hỏi của MultiDoc2Dial. UI chọn chế độ tường minh; dùng chung API contracts, log, xử lý lỗi và cách hiện kết quả. Tính liền mạch nằm ở trải nghiệm và luồng vận hành chung; số đo được ghi rõ theo từng tác vụ.

Bỏ việc tự viết cả KB ngân hàng trong bản nộp. Nếu cần minh họa, tối đa 3–5 tài liệu giả lập và ghi rõ tính chất; phần này chỉ mở sau gate ngày 8, không lấy số đo MultiDoc2Dial gán sang.

Contract tối thiểu gồm request_id, mode, artifact_version, decision/reason, evidence/citations và latency_ms theo tầng. Log JSONL đủ cho bản nộp. Tuấn giữ API/UI; leader giữ generator, retrieval và cache.

### C2. Chốt generator sớm

**App và LLM lúc demo chạy local:** API/UI, retrieval, Banking và Ollama ở máy demo. Runtime dự kiến là Ollama với qwen2.5:1.5b, cấu hình lượng tử hóa Q4_K_M. Ghi model digest, quantization thực tế, Ollama version và cấu hình máy; tag không thay cho digest. Leader phụ trách generator adapter; Tuấn nối stream từ API tới UI. Chuẩn bị weights trước khi khởi động, không tải model trong request người dùng.

**Huấn luyện tách khỏi app:** Colab với GPU mục tiêu A100 dùng fine-tune retriever R4 và tạo embeddings/index khi cần. Ghi tên GPU thực tế của từng run; đây là môi trường dự kiến, chưa có notebook/run R4 được kiểm chứng. Sau huấn luyện, xuất checkpoint, cấu hình, mapping và manifest; dựng lại corpus vectors bằng đúng encoder rồi đưa artifact về máy demo. Các baseline LR/SVM có thể train bằng CPU. App không cần một phiên Colab đang chạy để trả lời người dùng.

Trong A–B, thử ít nhất 10 câu dev, context tối đa khoảng 1.500 token, đầu ra tối đa 128 token, decoding cố định. Evidence mặc định top-3/tối đa 600 token; ngân sách toàn prompt còn gồm system, query/history và output reserve. Bật streaming từ generator qua API tới UI, hiện evidence sớm rồi hiển thị dần câu trả lời. Chốt cấu hình dựa trên TTFT đo ở máy local. Nếu phần cứng không chạy ổn, bản nộp chuyển sang trả đoạn trích có nguồn và ghi đúng là chế độ trích xuất; generator trở thành phần nâng cấp.

Mọi generation thành công và lỗi đều ghi JSONL. Khóa cache bao gồm model/revision, prompt version, câu hỏi và lịch sử, evidence IDs/content hash, decoding parameters. Lưu raw response, citations, timing gồm TTFT/tổng thời gian, trạng thái, token usage nếu runtime cung cấp. Stream bị ngắt phải lưu partial response và cờ incomplete, không cache như một câu trả lời hoàn chỉnh. Đổi metric hoặc review lại dùng output đã lưu; đổi prompt/evidence tạo entry mới.

Cache dùng để tái đánh giá; báo riêng cache hit, cache miss và chế độ replay. Demo trực tiếp nhận câu mới phải thực hiện inference thật hoặc chuyển sang fallback rõ ràng khi lỗi. Video/cache replay không thay số đo inference live.

### C3. Latency budget khởi đầu

Các mức sau là mục tiêu thiết kế, chưa phải kết quả đo hay cam kết phần cứng. Ghi CPU/GPU/RAM/runtime trong A, đo probe trong A–B rồi chốt target theo máy demo local; sau đó giữ làm điểm quy chiếu.

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
| E4 | Frozen all-MiniLM-L6-v2 | Logistic Regression | Thêm nếu E1–E3 đã ổn và còn giờ |

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

Dùng tiếp parser, chunks/qrels và split đã có trong v0.4.1. Nếu cần đối chiếu cách chia passage của IBM hoặc sửa adapter, giới hạn phiên khảo sát tối đa một giờ, lưu chunk-to-span mapping và config, rồi chạy lại phần bị ảnh hưởng. Chỉ đặt cạnh số công bố khi segmentation, query/history, corpus, split và metric tương thích; cùng cách chia passage riêng lẻ chưa đủ để gọi là tái lập.

Lấy query từ lượt user và lịch sử đã có trước câu trả lời cần dự đoán. Target response và grounding của nó chỉ dùng dựng nhãn đánh giá. Không index các câu trả lời hội thoại.

Public test trong bản dữ liệu này là dummy; chia public validation theo dialogue ID thành dev và held-out evaluation, giữ tỷ lệ domain nếu có thể, lưu danh sách cố định. Corpus tài liệu được dùng chung theo thiết lập retrieval, còn hội thoại đánh giá phải tách biệt.

Tuấn dùng API retrieval hiện có để nối UI ngay ở A; leader không chờ làm lại BM25 mức tài liệu. Hưng audit khoảng 20 ca qrels trong A hoặc B, gồm span qua biên và nhiều reference; leader xử lý nhận xét trước tạo train pairs hoặc đánh giá cuối. Giữ doc_id/character offsets, tập positives và quy tắc strict/lenient đã ghi trong cấu hình. Grounding không phải danh sách đầy đủ mọi đoạn đúng. Sau freeze mới chạy held-out; báo hit@3/hit@5 strict–lenient, MRR@10 và latency, ghi rõ số query và định nghĩa metric.

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
- Khởi đầu một cấu hình nhỏ trên Colab A100, chẳng hạn 1 epoch, batch thực 16–32 tùy VRAM, LR 2e-5; kiểm tra 10–20 bước trước khi để máy tự chạy. Đây là cấu hình thử, không phải cam kết tối ưu hay runtime. Lưu max sequence length và bảo đảm passage positive không bị tokenizer cắt mất phần bằng chứng.
- Mỗi run lưu config, split hash, seed, train loss, checkpoint, optimizer/scheduler state nếu cần resume. Xuất checkpoint/log khỏi vùng tạm của Colab. Ghi giờ GPU, phần cứng và số cặp train riêng với giờ công.
- Sau fine-tune, encode lại cả corpus và query bằng cùng checkpoint R4; không dùng index R2 với query encoder R4. Cache/index key chứa checkpoint và corpus/chunk hash. Encoder cho Banking triage giữ artifact frozen riêng.
- Ưu tiên ba seed 42/43/44 với cấu hình đã chốt, chỉ chạy thêm seed khi smoke run và giờ còn lại cho phép; báo mean ± std. Nếu chỉ hoàn thành một seed, ghi kết quả thăm dò, không nhận đã đo độ ổn định. Bootstrap theo dialogue cho chênh lệch R4–R2 là CI do mẫu đánh giá, khác biến thiên do seed.
- Báo hit@3/hit@5, MRR@10 và latency p95 trên cùng phần cứng phục vụ; có thể không cải thiện. R4 so R2; R5 so R3 nếu chọn hybrid. Chọn phương pháp/seed phục vụ trước khi xem held-out; mặc định seed 42 nếu giữ R4. Đổi retriever phải cập nhật generation cache và đánh giá pipeline đúng artifact mới.

Tại mốc ngày 8 của chặng C, chỉ mở full run nếu luồng lõi ổn và còn đủ 5 giờ chủ động. Thứ tự là tạo/audit pairs → smoke run → train → re-index → so dev → quyết định giữ/cắt; không gán từng seed vào một đêm cố định. Giờ máy có thể để buổi tối nhưng thời lượng do smoke run quyết định. Nếu không đủ điều kiện hoặc vượt ngân sách, fine-tune retriever chuyển sau nộp.

## E. Công việc theo chặng A–D

Mỗi bạn làm khoảng 2–3 phiên một giờ cho một chặng. Dùng đầu ra và điều kiện nghiệm thu để chuyển tiếp; các mốc ngày chỉ giúp kiểm tra tiến độ. Chi tiết nhận gì, giao ai và ví dụ bàn giao nằm trong [TEAM_HANDOFF.md](TEAM_HANDOFF.md).

| Chặng | Leader | Hưng | Cầu | Tuấn |
|---|---|---|---|---|
| A Chuẩn bị | Giao API Docs/ví dụ đã có; chọn 20 qrels để Hưng review; probe Ollama local; review audit của Cầu | Audit/splits, preprocessing, E1; review H-01 và gửi leader nhận xét qrels | C-01/C-02; chốt split/mask và calibration trước fit | UI/contract hai chế độ; gọi Docs thật; làm mock Banking/stream để không chờ model |
| B Ghép bản đầu | Adapter generator độc lập, citation/cache/deadline; tiếp nhận qrels review | E2/E3 và ablation cùng split; giao bảng dev ban đầu | Fit LR, chọn C; giao Tuấn predict(text), model/config và ba ví dụ H1 | Thử predict độc lập rồi nối Banking vào API/UI; warm-up/nạp một lần |
| C Hoàn thiện trên dev | Chốt Docs config/gate/prompt; giao generator thật cho Tuấn; tại mốc ngày 8 cân nhắc R4/R5 | Chọn cấu hình dev, confusion và ca lỗi; pipeline load lại được | H2–H4, đường cong và ngưỡng dev; giao Tuấn bản chốt accept/handoff | Nối generator/stream và nguồn; hội thoại/reset/hủy/lỗi; thay Banking artifact đã chốt |
| D Đánh giá và bàn giao | Freeze, held-out, review citation, công cụ đánh giá dùng chung và ghép báo cáo | Test sau freeze; metrics, phân tích và phần báo cáo giao leader | Test sau freeze; ma trận ID-OOS, reweight 5/10/20%, báo cáo | 40 smoke cùng nhóm; latency/cold start, môi trường mới, hướng dẫn chạy và phần triển khai |

Hưng phụ trách kết quả Course-77 và review qrels; công cụ đánh giá dùng chung do leader duy trì từ nền hiện có. Cầu chịu trách nhiệm phương pháp/score của Banking, Tuấn kiểm tra kết quả khi đi qua API/UI. Mỗi owner viết phần báo cáo từ kết quả thật của mình.

Ở mốc ngày 2 kiểm tra UI → API → evidence; mốc ngày 5 kiểm tra Banking thật qua API; mốc ngày 8 kiểm tra hai chế độ và quyết định phạm vi nâng cao. Khóa tính năng trước ngày 13, dành ngày 13–15 chạy lại, sửa lỗi, quay video và bảo vệ. Buổi đầu dành 15 phút đi qua code/ví dụ; cuối D mỗi người giải thích module của một bạn khác trong giờ ghép bài đã dự trù.

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
- [Dataset card MultiDoc2Dial](https://huggingface.co/datasets/IBM/multidoc2dial): đối chiếu notice kèm bản tải trong bước audit, lưu revision/attribution; repo dùng downloader/manifest.

Đây là kế hoạch thực hiện từ nền v0.4.1. Retrieval R1/R2/R3 và số đo dev đã có; Banking head/calibration, Course-77, generator phục vụ thật/cache/deadline và R4/R5 vẫn là công việc được giao, chưa được coi là hoàn thành. Trạng thái và giới hạn số đo xem [VERIFICATION_v0.4.md](VERIFICATION_v0.4.md). Latency, TTFT, A100 và giờ công trong plan là mục tiêu/môi trường dự kiến, không phải kết quả đã đo. Giờ máy tự train ghi riêng; chuẩn bị, theo dõi, sửa lỗi và đánh giá vẫn tính vào giờ chủ động.
