"""Local UI demo only, without a model or network stream.

Event names are a development proposal, not the real generator contract.
"""

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class MockStream:
    text: str
    scenario: Literal["complete", "error"] = "complete"
    cursor: int = 0
    status: Literal["running", "complete", "error", "cancelled"] = "running"
    event: Literal["delta", "done", "error", "cancelled"] = "delta"

    @property
    def visible(self) -> str:
        return self.text[:self.cursor]


def start_mock(mode: str, question: str, scenario: str) -> MockStream:
    if mode not in {"docs_qa", "banking_triage"} or scenario not in {"complete", "error"}:
        raise ValueError("unsupported mock mode or scenario")
    task = "hỏi tài liệu" if mode == "docs_qa" else "phân loại yêu cầu ngân hàng"
    text = (
        f"Đây là nội dung giả lập cho màn hình {task}. "
        "Từng phần văn bản được hiển thị dần để bạn kiểm tra trải nghiệm nhận phản hồi. "
        f"Câu hỏi đang thử: “{question[:300]}”. "
        "Bản demo này không gọi model, không tìm tài liệu và không đưa ra dự đoán thật. "
        "Khi module được bàn giao, giao diện sẽ nhận nội dung từ stream thực tế. "
        "Bạn có thể bấm Hủy stream trong lúc chạy hoặc chọn kịch bản Lỗi giữa chừng "
        "để kiểm tra cách giao diện kết thúc phản hồi."
    )
    return MockStream(text=text, scenario=scenario)


def advance_mock(stream: MockStream, characters: int = 12) -> MockStream:
    if characters < 1:
        raise ValueError("characters must be positive")
    if stream.status != "running":
        return stream
    if stream.scenario == "error" and stream.cursor >= len(stream.text) // 2:
        return replace(stream, status="error", event="error")
    cursor = min(len(stream.text), stream.cursor + characters)
    done = cursor == len(stream.text)
    return replace(stream, cursor=cursor, status="complete" if done else "running",
                   event="done" if done else "delta")


def cancel_mock(stream: MockStream) -> MockStream:
    return replace(stream, status="cancelled", event="cancelled") if stream.status == "running" else stream
