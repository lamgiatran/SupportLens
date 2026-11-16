"""Decode API replies before rendering so HTML errors cannot crash the UI."""

import json
import httpx
from pydantic import ValidationError

from .contracts import AssistResponse


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
