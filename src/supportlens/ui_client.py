"""Decode API replies before rendering so HTML errors cannot crash the UI."""

import json
import httpx
from pydantic import ValidationError

from .contracts import AssistResponse


def fetch_health(base_url: str) -> tuple[dict | None, str | None]:
    """Read capabilities without inventing readiness when the check fails."""
    try:
        response = httpx.get(base_url.rstrip("/") + "/health", timeout=3)
        response.raise_for_status()
        value = response.json()
        if (not isinstance(value, dict)
                or not isinstance(value.get("docs_methods"), list)
                or any(not isinstance(m, str) for m in value["docs_methods"])
                or not isinstance(value.get("banking_model_ready"), bool)):
            return None, "Phản hồi /health không đúng cấu trúc SupportLens."
        if "banking_serving" in value:
            serving = value["banking_serving"]
            if not isinstance(serving, dict) or not isinstance(serving.get("is_mock"), bool):
                return None, "Phản hồi /health không đúng cấu trúc SupportLens."
        return value, None
    except (httpx.HTTPError, ValueError, UnicodeDecodeError) as error:
        return None, f"Không đọc được trạng thái API: {error}"


def request_assist(base_url: str, payload: dict) -> tuple[AssistResponse | None, str | None]:
    try:
        response = httpx.post(base_url.rstrip("/") + "/v1/assist", json=payload, timeout=30)
    except httpx.HTTPError as error:
        return None, f"Không kết nối được API: {error}"
    value, error = parse_assist_response(response)
    if value is not None and value.mode != payload["mode"]:
        return None, "API trả kết quả của chế độ khác với yêu cầu."
    return value, error


def parse_assist_response(response: httpx.Response) -> tuple[AssistResponse | None, str | None]:
    try:
        value = response.json()
    except (ValueError, UnicodeDecodeError):
        return None, f"HTTP {response.status_code}: {response.text[:1000] or 'Empty response'}"
    if response.status_code != 200:
        detail = value.get("detail", value) if isinstance(value, dict) else value
        message = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
        return None, f"HTTP {response.status_code}: {message[:1000]}"
    try:
        return AssistResponse.model_validate(value), None
    except ValidationError:
        return None, "HTTP 200: API response does not match the SupportLens contract."
