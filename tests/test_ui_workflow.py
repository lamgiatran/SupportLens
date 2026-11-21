"""Exercise actual UI state transitions; no network/model dependency required."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from supportlens.ui_client import fetch_health, request_assist
from supportlens.ui_mock import advance_mock, cancel_mock, start_mock


def test_mock_stream_terminal_states_do_not_resume():
    stream = start_mock("docs_qa", "insurance?", "complete")
    first = advance_mock(stream)
    assert 0 < len(first.visible) < len(stream.text)
    cancelled = cancel_mock(first)
    assert cancelled.status == "cancelled" and cancelled.event == "cancelled"
    assert advance_mock(cancelled) == cancelled
    done = advance_mock(stream, len(stream.text))
    assert done.status == "complete" and done.visible == stream.text
    assert advance_mock(done) == cancel_mock(done) == done
    error = start_mock("banking_triage", "card?", "error")
    while error.status == "running":
        error = advance_mock(error)
    assert error.status == error.event == "error"
    assert 0 < error.cursor < len(error.text)
    assert advance_mock(error) == error


@pytest.mark.parametrize("body", [b"<html>Bad gateway</html>", b"[]", b'{"docs_methods": [1]}'])
def test_health_handles_non_json_and_wrong_shape(monkeypatch, body):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, content=body, request=httpx.Request("GET", "http://test/health")))
    value, error = fetch_health("http://test")
    assert value is None and error


@pytest.mark.parametrize("serving", [None, [], "unexpected", 1, {},
                                     {"is_mock": "false"}, {"is_mock": 1}])
def test_health_rejects_malformed_banking_serving(monkeypatch, serving):
    payload = {"docs_methods": ["R1"], "banking_model_ready": False, "banking_serving": serving}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json=payload, request=httpx.Request("GET", "http://test/health")))
    value, error = fetch_health("http://test")
    assert value is None and error


@pytest.mark.parametrize("serving", [None, [], "unexpected"])
def test_ui_handles_malformed_health_without_crashing(monkeypatch, serving):
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    payload = {"docs_methods": ["R1"], "banking_model_ready": False, "banking_serving": serving}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json=payload, request=httpx.Request("GET", "http://test/health")))
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "app.py"))
        app.run(timeout=10)
        assert not app.exception
        assert any("Chưa xác định được trạng thái API" in warning.value for warning in app.warning)
        # The unavailable health response must not prevent local demo use.
        app.radio(key="transport").set_value("Mock stream · Demo").run()
        app.text_area[0].input("test mock").run()
        button(app, "Chạy mock stream").click().run()
        assert not app.exception and app.session_state["mock_stream"].status == "running"
    finally:
        st.cache_data.clear()


def test_assist_transport_errors_and_html_errors(monkeypatch):
    def offline(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "post", offline)
    assert request_assist("http://test", {"mode": "docs_qa"})[0] is None
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(503, text="<html>Unavailable</html>"))
    value, error = request_assist("http://test", {"mode": "docs_qa"})
    assert value is None and "HTTP 503" in error


@pytest.fixture
def ui(monkeypatch, rows):
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from fastapi.testclient import TestClient
    from supportlens.api import create_app
    from supportlens.retrieval import BM25Documents
    from supportlens.ui_client import parse_assist_response
    import supportlens.ui_client as client

    st.cache_data.clear()
    api = TestClient(create_app(BM25Documents(rows)))
    calls = []
    monkeypatch.setattr(client, "fetch_health", lambda url: (api.get("/health").json(), None))

    def assist(url, payload):
        calls.append(payload)
        return parse_assist_response(api.post("/v1/assist", json=payload))

    monkeypatch.setattr(client, "request_assist", assist)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "app.py"))
    app.run(timeout=10)
    assert not app.exception
    yield app, calls
    st.cache_data.clear()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_docs_history_error_replacement_and_reset(ui):
    app, calls = ui
    app.text_area[0].input("uniqueaddress").run()
    button(app, "Gửi câu hỏi").click().run()
    assert not app.exception and app.metric[1].value == "3"
    app.text_area[0].input("second question").run()
    button(app, "Gửi câu hỏi").click().run()
    assert calls[-1]["history"] == [{"role": "user", "content": "uniqueaddress"}]
    assert not any(t["role"] == "assistant" for t in calls[-1]["history"])
    # Switching to Banking must not leak Docs history/evidence into its request.
    app.radio(key="mode").set_value("banking_triage").run()
    app.text_area[0].input("card?").run()
    button(app, "Gửi câu hỏi").click().run()
    assert calls[-1]["history"] == [] and app.error
    assert not app.metric and not app.exception
    app.radio(key="mode").set_value("docs_qa").run()
    button(app, "＋ Hội thoại mới").click().run()
    state = app.session_state["conversations"]["docs_qa"]
    assert state == {"history": [], "records": [], "latest": None}
    assert app.text_area[0].value == "" and not app.metric


def test_failed_request_clears_current_result(ui, monkeypatch):
    import supportlens.ui_client as client
    app, _ = ui
    app.text_area[0].input("uniqueaddress").run()
    button(app, "Gửi câu hỏi").click().run()
    assert app.metric
    monkeypatch.setattr(client, "request_assist", lambda *a: (None, "HTTP 503: <html>Unavailable</html>"))
    app.text_area[0].input("new question").run()
    button(app, "Gửi câu hỏi").click().run()
    assert app.error and not app.metric and not app.exception
    latest = app.session_state["conversations"]["docs_qa"]["latest"]
    assert latest["value"] is None and latest["question"] == "new question"


@pytest.mark.parametrize("scenario,status", [("Hoàn thành", "complete"), ("Lỗi giữa chừng", "error")])
def test_mock_ui_completion_error_cancel_and_reset(ui, scenario, status):
    app, calls = ui
    app.radio(key="transport").set_value("Mock stream · Demo").run()
    app.selectbox[0].set_value(scenario).run()
    app.text_area[0].input("test mock").run()
    button(app, "Chạy mock stream").click().run()
    assert app.session_state["mock_stream"].status == "running"
    button(app, "Hủy stream").click().run()
    assert app.session_state["mock_stream"].status == "cancelled"
    button(app, "Chạy mock stream").click().run()
    stream = app.session_state["mock_stream"]
    app.session_state["mock_stream"] = replace(stream, cursor=len(stream.text) - 1
                                                if status == "complete" else len(stream.text) // 2)
    app.run()
    assert app.session_state["mock_stream"].status == status
    assert not app.exception and not calls
    assert app.session_state["conversations"]["docs_qa"]["history"] == []
    button(app, "＋ Hội thoại mới").click().run()
    assert app.session_state["mock_stream"] is None


def test_unloaded_retriever_is_explicit_and_not_submitted(ui):
    app, calls = ui
    app.selectbox(key="retriever").set_value("R2").run()
    app.text_area[0].input("question").run()
    assert button(app, "Gửi câu hỏi").disabled
    assert any("R2 chưa sẵn sàng" in w.value for w in app.warning)
    assert not calls and not app.exception


def test_blank_input_and_workspace_change_do_not_send_or_resume_mock(ui):
    app, calls = ui
    button(app, "Gửi câu hỏi").click().run()
    assert not calls and any("Nhập câu hỏi" in w.value for w in app.warning)
    app.radio(key="transport").set_value("Mock stream · Demo").run()
    app.text_area[0].input("demo").run()
    button(app, "Chạy mock stream").click().run()
    app.radio(key="mode").set_value("banking_triage").run()
    assert app.session_state["mock_stream"] is None
    assert not calls and not app.exception
