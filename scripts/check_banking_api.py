"""Compare exactly three Cau samples: standalone pipeline versus the live API."""

import argparse
import json
from math import isclose
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from supportlens.banking_serving import runtime_from_env


def check_samples(pipeline, client, samples, *, method="H1"):
    if len(samples) != 3:
        raise ValueError("Provide exactly 3 JSONL samples from Cau")
    questions = [sample["input"]["question"] for sample in samples]
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("Each sample must contain a non-blank input.question")
    health_response = client.get("/health")
    health_response.raise_for_status()
    health = health_response.json()
    serving = health.get("banking_serving", {})
    if (not health.get("banking_model_ready") or serving.get("state") != "ready"
            or serving.get("is_mock") is not False):
        raise ValueError("API must be ready with a real Banking factory; mock is not accepted")
    cases = []
    for index, question in enumerate(questions, 1):
        result = pipeline.predict(question.strip(), method=method)
        expected = {key: getattr(result, key) for key in
                    ("predicted_intent", "score", "decision", "reason", "artifact_version")}
        expected["score"] = float(expected["score"])
        expected["score_method"] = method
        if expected["artifact_version"].startswith("MOCK-"):
            raise ValueError("Standalone pipeline returned a MOCK artifact")
        response = client.post("/v1/assist", json={
            "mode": "banking_triage", "model_scope": "handoff50", "question": question,
            "score_method": method, "history": []})
        response.raise_for_status()
        value = response.json()
        actual = {key: value.get(key) for key in expected}
        different = []
        for key in expected:
            if key == "score":
                matches = (isinstance(actual[key], (int, float)) and not isinstance(actual[key], bool)
                           and isclose(actual[key], expected[key], rel_tol=1e-6, abs_tol=1e-6))
            else:
                matches = actual[key] == expected[key]
            if not matches:
                different.append(key)
        cases.append({"case": index, "question": question, "status": "fail" if different else "pass",
                      "different_fields": different, "standalone": expected, "api": actual})
    return {"status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
            "method": method, "score_tolerance": 1e-6, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--method", choices=["H1", "H2", "H3", "H4"], default="H1")
    args = parser.parse_args()
    try:
        if os.environ.get("SUPPORTLENS_BANKING_MODE") != "real":
            raise ValueError("Set SUPPORTLENS_BANKING_MODE=real and use the same factory/artifacts as the API")
        samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        runtime = runtime_from_env(os.environ)
        runtime.start()
        if not runtime.ready:
            raise RuntimeError(f"Standalone startup failed: {runtime.snapshot()}")
        with httpx.Client(base_url=args.api.rstrip("/"), timeout=60) as client:
            report = check_samples(runtime.pipeline, client, samples, method=args.method)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, httpx.HTTPError) as error:
        report = {"status": "fail", "error_type": type(error).__name__, "error": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
