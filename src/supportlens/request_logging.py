"""Log every HTTP outcome without consuming the request or response stream."""

import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class RequestLogMiddleware:
    def __init__(self, app, *, path: str | Path, log_questions: bool = False):
        self.app = app
        self.path = Path(path)
        self.log_questions = log_questions

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = perf_counter()
        request_id = str(uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status, error_type = 500, None
        body = bytearray()
        body_too_large = False

        async def receive_logged():
            nonlocal body_too_large
            event = await receive()
            if self.log_questions and event["type"] == "http.request":
                chunk = event.get("body", b"")
                if len(body) + len(chunk) <= 65536 and not body_too_large:
                    body.extend(chunk)
                else:
                    body_too_large = True
                    body.clear()
            return event

        async def send_logged(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
                event = dict(event)
                event["headers"] = list(event.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii"))]
            await send(event)

        try:
            await self.app(scope, receive_logged, send_logged)
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            record = dict(state.get("response_record", {}))
            record.update(request_id=request_id, method=scope["method"],
                          path=scope["path"], status_code=status,
                          http_total_ms=(perf_counter() - started) * 1000)
            if error_type:
                record["error_type"] = error_type
            if self.log_questions:
                if body_too_large:
                    record["question_capture"] = "body_over_64_kib"
                else:
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeDecodeError):
                        payload = None
                    if isinstance(payload, dict) and isinstance(payload.get("question"), str):
                        record["question"] = payload["question"][:8000]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                logging.getLogger(__name__).exception("Could not write request log")
