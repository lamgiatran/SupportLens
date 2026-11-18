"""SupportLens UI: real API responses and local stream demos stay separate."""

import os

import streamlit as st

from supportlens.ui_client import fetch_health, request_assist
from supportlens.ui_mock import advance_mock, cancel_mock, start_mock


st.set_page_config(page_title="SupportLens · Không gian hỗ trợ", page_icon="📖", layout="wide")
st.markdown("""
<style>
:root { --sl-ink: #18333d; --sl-teal: #14796e; }
.stApp { background: #f5f7f6; color: #18333d; }
[data-testid="stSidebar"] { background: #eaf0ed; border-right: 1px solid #d9e3de; }
.block-container { max-width: 1160px; padding-top: 2.4rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--sl-ink); letter-spacing: -.035em; }
h1 { font-size: 2.65rem !important; }
[data-testid="stMetric"] { background: white; border: 1px solid #dce5e0;
    border-radius: 12px; padding: 14px 18px; }
[data-testid="stMetricValue"] { font-size: 1.55rem; color: var(--sl-ink); }
[data-testid="stExpander"] { background: white; border-radius: 10px; }
.sl-wordmark { font-size: 1.65rem; font-weight: 750; color: #18333d; letter-spacing: -.06em; }
.sl-kicker { color: #14796e; font-size: .74rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; margin-bottom: .5rem; }
.sl-empty { padding: 2rem; border: 1px dashed #b8cbc1; border-radius: 14px;
    color: #50675d; background: #eef3ef; margin-top: 1rem; }
.stButton > button[kind="primary"] { background: #14796e; border-color: #14796e; }
</style>
""", unsafe_allow_html=True)

API_URL = os.environ.get("SUPPORTLENS_API", "http://127.0.0.1:8000").rstrip("/")
MODES = {"docs_qa": "Hỏi tài liệu", "banking_triage": "Yêu cầu ngân hàng"}
RETRIEVERS = {"R1": "BM25 · Từ khóa", "R2": "Dense · Ngữ nghĩa", "R3": "Hybrid · Kết hợp"}


@st.cache_data(ttl=10, show_spinner=False)
def capabilities(url):
    return fetch_health(url)


def new_conversation(mode):
    st.session_state.conversations[mode] = {"history": [], "records": [], "latest": None}
    st.session_state[f"question_{mode}"] = ""
    st.session_state.mock_stream = None


def render_response(value):
    if value.decision == "retrieved":
        st.info("Đã tìm thấy tài liệu liên quan. Chức năng tạo câu trả lời chưa được kết nối.")
    elif value.decision == "handoff":
        st.warning("Yêu cầu cần được chuyển cho người hỗ trợ.")
    elif value.decision == "accepted":
        st.success("Yêu cầu đã được nhận diện và chấp nhận xử lý.")
    else:
        st.success("Đã nhận câu trả lời có nguồn.")
    if value.predicted_intent:
        st.subheader("Nhóm yêu cầu")
        st.write(value.predicted_intent)
    if value.answer:
        st.subheader("Câu trả lời")
        st.write(value.answer)
    if value.citations:
        st.caption("Nguồn trích dẫn: " + ", ".join(value.citations))
    a, b, c = st.columns(3)
    a.metric("Thời gian API", f"{value.latency.total_ms:.1f} ms")
    b.metric("Đoạn tài liệu" if value.mode == "docs_qa" else "Quyết định",
             str(len(value.evidence)) if value.mode == "docs_qa" else
             ("Chấp nhận" if value.decision == "accepted" else "Chuyển hỗ trợ"))
    method = value.artifact_version.split("-", 1)[0]
    c.metric("Phương pháp", method if method in RETRIEVERS else value.score_method or "—")
    if value.evidence:
        st.subheader("Tài liệu tham khảo")
        st.caption("Các đoạn được xếp theo mức liên quan; kết quả tìm kiếm có thể chứa đoạn chưa phù hợp.")
        for index, item in enumerate(value.evidence, 1):
            with st.expander(f"{index:02d} · {item.title}", expanded=index == 1):
                st.caption(f"Lĩnh vực: {item.domain} · Điểm truy xuất: {item.score:.3f}")
                st.write(item.text)
                st.caption(f"Mã nguồn: {item.evidence_id}")
    with st.expander("Chi tiết phản hồi API"):
        st.caption("Thông tin phục vụ đối chiếu và kiểm tra tích hợp.")
        st.json(value.model_dump())


if "conversations" not in st.session_state:
    st.session_state.conversations = {
        key: {"history": [], "records": [], "latest": None} for key in MODES
    }
if "mock_stream" not in st.session_state:
    st.session_state.mock_stream = None

with st.sidebar:
    st.markdown('<div class="sl-wordmark">SupportLens<span style="color:#14796e">.</span></div>',
                unsafe_allow_html=True)
    st.caption("Tìm nguồn. Hiểu yêu cầu. Hỗ trợ đúng lúc.")
    st.divider()
    mode = st.radio("Không gian làm việc", list(MODES), format_func=MODES.get, key="mode")
    health, health_error = capabilities(API_URL)
    st.markdown("**Kết nối dịch vụ**")
    if health is None:
        st.warning("Chưa xác định được trạng thái API")
    else:
        st.success("Đã kết nối API")
        methods = ", ".join(health["docs_methods"]) or "chưa có"
        st.caption(f"Retriever sẵn sàng: {methods}")
        st.caption("Banking: " + ("đã nạp model" if health["banking_model_ready"] else "chưa nạp model"))
    if st.button("Kiểm tra lại kết nối", use_container_width=True):
        capabilities.clear()
        st.rerun()
    st.divider()
    st.button("＋ Hội thoại mới", on_click=new_conversation, args=(mode,), use_container_width=True)
    st.caption("Xóa câu hỏi, kết quả và bản demo của chế độ đang chọn trong phiên này.")
    with st.expander("Địa chỉ API"):
        st.code(API_URL, language=None)
        st.caption("Cấu hình bằng biến SUPPORTLENS_API khi khởi động UI.")
        if health_error:
            st.text(health_error)

st.markdown('<div class="sl-kicker">Không gian hỗ trợ / ' +
            ("Docs-QA" if mode == "docs_qa" else "Banking") + '</div>', unsafe_allow_html=True)
st.title(MODES[mode])
st.write("Tra cứu tài liệu MultiDoc2Dial và xem các đoạn nguồn liên quan đến câu hỏi." if mode == "docs_qa"
         else "Nhận diện yêu cầu trong 50 nhóm ngân hàng được hỗ trợ hoặc chuyển người xử lý.")

transport = st.radio("Cách nhận phản hồi", ["API thật", "Mock stream · Demo"], horizontal=True,
                     key="transport")
context = (mode, transport)
if st.session_state.get("active_context") != context:
    st.session_state.mock_stream = None
    st.session_state.active_context = context

conversation = st.session_state.conversations[mode]
is_mock = transport != "API thật"
retriever = "R1"
score_method = "H1"
if not is_mock:
    if mode == "docs_qa":
        available = health["docs_methods"] if health else None

        def method_label(method):
            suffix = " · chưa nạp" if available is not None and method not in available else ""
            return f"{method} — {RETRIEVERS[method]}{suffix}"

        retriever = st.selectbox("Cách tìm tài liệu", list(RETRIEVERS),
                                 format_func=method_label, key="retriever")
        if available is not None and retriever not in available:
            st.warning(f"{retriever} chưa sẵn sàng. Chọn retriever đã nạp hoặc chuẩn bị encoder theo RUN_DENSE.md.")
        use_history = st.checkbox("Gửi ngữ cảnh hội thoại trước đó", value=True, key="use_history")
        st.caption("Lịch sử được giữ riêng trong phiên trình duyệt; API nhận tối đa 40 tin nhắn gần nhất.")
    else:
        score_method = st.selectbox("Phương pháp độ tin cậy", ["H1", "H2", "H3", "H4"])
        use_history = False
        st.caption("Banking phân loại câu hỏi hiện tại, không dùng lịch sử hội thoại.")
        if health is not None and not health["banking_model_ready"]:
            st.info("Model Banking chưa được nạp. Gửi yêu cầu sẽ nhận thông báo thiếu model từ API; "
                    "bạn cũng có thể chuyển sang Mock stream để thử giao diện.")
else:
    use_history = False
    st.warning("DEMO GIẢ LẬP · Không gọi API/model. Nội dung mẫu không phải câu trả lời hay dự đoán thật.")
    scenario_label = st.selectbox("Kịch bản thử stream", ["Hoàn thành", "Lỗi giữa chừng"])
    st.caption("Stream chạy dần từng phần. Dùng nút Hủy stream để thử dừng giữa chừng.")

question = st.text_area("Câu hỏi của bạn", key=f"question_{mode}", max_chars=8000, height=110,
                        placeholder="My insurance ended so what should I do?" if mode == "docs_qa"
                        else "My card payment was declined.")
running = st.session_state.mock_stream is not None and st.session_state.mock_stream.status == "running"
unavailable = (not is_mock and mode == "docs_qa" and health is not None
               and retriever not in health["docs_methods"])
submitted = st.button("Chạy mock stream" if is_mock else "Gửi câu hỏi", type="primary",
                       disabled=running or unavailable)

if submitted and not question.strip():
    st.warning("Nhập câu hỏi trước khi gửi.")

if submitted and question.strip():
    if is_mock:
        st.session_state.mock_stream = start_mock(mode, question.strip(),
                                                  "complete" if scenario_label == "Hoàn thành" else "error")
        st.rerun()  # Enable the fragment timer and disable submit while streaming.
    else:
        payload = {"mode": mode, "question": question.strip(),
                   "history": conversation["history"][-40:] if use_history else []}
        if mode == "docs_qa":
            payload.update(corpus_id="multidoc2dial", retriever=retriever)
        else:
            payload.update(model_scope="handoff50", score_method=score_method)
        conversation["latest"] = None
        with st.spinner("Đang tìm tài liệu…" if mode == "docs_qa" else "Đang phân loại yêu cầu…"):
            value, error = request_assist(API_URL, payload)
        record = {"question": question.strip(), "value": value, "error": error, "payload": payload}
        conversation["latest"] = record
        conversation["records"] = (conversation["records"] + [record])[-20:]
        if value is not None:
            conversation["history"].append({"role": "user", "content": question.strip()})
            # Retrieval fragments and mock text are never assistant answers.
            if value.answer:
                conversation["history"].append({"role": "assistant", "content": value.answer[:12000]})
            conversation["history"] = conversation["history"][-40:]

st.divider()
if is_mock:
    current = st.session_state.mock_stream

    @st.fragment(run_every=0.2 if current is not None and current.status == "running" else None)
    def stream_panel():
        stream = st.session_state.mock_stream
        if stream is None:
            st.info("Nhập câu hỏi và chạy demo để thử cách nội dung xuất hiện dần.")
            return
        if stream.status == "running":
            stream = advance_mock(stream)
            st.session_state.mock_stream = stream
            if stream.status != "running":
                st.rerun()  # Stop timer and re-enable submit on completion/error.
        st.subheader("Phản hồi giả lập")
        st.caption("MOCK · Dữ liệu mẫu trong trình duyệt, không có độ trễ hay nguồn từ model thật.")
        st.text(stream.visible + (" ▍" if stream.status == "running" else ""))
        st.progress(stream.cursor / len(stream.text))
        if stream.status == "running":
            if st.button("Hủy stream"):
                st.session_state.mock_stream = cancel_mock(stream)
                st.rerun()
        elif stream.status == "complete":
            st.success("Mock stream đã hoàn thành.")
        elif stream.status == "error":
            st.error("Lỗi giả lập giữa chừng. Nội dung trên chưa hoàn tất; bạn có thể chạy lại.")
        else:
            st.warning("Đã hủy mock stream. Nội dung trên chưa hoàn tất.")
        st.caption(f"Sự kiện demo hiện tại: {stream.event}")

    stream_panel()
else:
    latest = conversation["latest"]
    if latest is None:
        st.markdown('<div class="sl-empty"><strong>Bắt đầu từ một câu hỏi.</strong><br>'
                    'Kết quả sẽ xuất hiện ở đây sau khi bạn gửi yêu cầu.</div>', unsafe_allow_html=True)
    else:
        st.subheader("Kết quả mới nhất")
        st.write(latest["question"])
        if latest["error"]:
            st.error("Chưa nhận được kết quả. Kiểm tra kết nối hoặc trạng thái module rồi thử lại.")
            with st.expander("Chi tiết lỗi"):
                st.text(latest["error"])
        else:
            render_response(latest["value"])
        with st.expander("Request đã gửi"):
            st.json(latest["payload"])
    if len(conversation["records"]) > 1:
        with st.expander(f"Lịch sử phiên · {len(conversation['records']) - 1} yêu cầu trước"):
            for record in reversed(conversation["records"][:-1]):
                st.markdown("**Câu hỏi trước**")
                st.write(record["question"])
                if record["error"]:
                    st.caption("Yêu cầu không thành công.")
                else:
                    value = record["value"]
                    if value.answer:
                        st.write(value.answer)
                    st.caption(f"{value.decision} · {len(value.evidence)} đoạn tài liệu · "
                               f"{value.latency.total_ms:.1f} ms")
                st.divider()

st.caption("SupportLens · Docs-QA / Handoff-50 · Phiên làm việc local")
