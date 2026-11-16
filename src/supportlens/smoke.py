"""Run repeatable handoff samples against a callable from the trusted repo."""

import importlib
import json
import math
from pathlib import Path


def check_expected(actual, expected, *, path="output", atol=1e-6):
    """Expected dictionaries are subsets; sequences keep their exact length/order."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path}: expected an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path}.{key}: missing")
            check_expected(actual[key], value, path=f"{path}.{key}", atol=atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: sequence length/type mismatch")
        for i, (left, right) in enumerate(zip(actual, expected)):
            check_expected(left, right, path=f"{path}[{i}]", atol=atol)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isclose(actual, expected, rel_tol=0, abs_tol=atol)):
            raise ValueError(f"{path}: numeric value differs")
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{path}: value/type differs")


def run_package(package: str | Path, *, entrypoint: str | None = None) -> dict:
    package = Path(package).resolve()
    if entrypoint is None:
        entrypoint = json.loads((package / "manifest.json").read_text(encoding="utf-8"))["smoke_entrypoint"]
    module_name, function_name = entrypoint.split(":", 1)
    callback = getattr(importlib.import_module(module_name), function_name)
    samples = [json.loads(line) for line in (package / "smoke_samples.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise ValueError("at least one handoff sample is required")
    for i, sample in enumerate(samples, 1):
        actual = callback(sample["input"], package_dir=package)
        check_expected(actual, sample["expected"], path=f"sample[{i}]",
                       atol=sample.get("atol", 1e-6))
    return {"status": "pass", "samples": len(samples), "entrypoint": entrypoint}
