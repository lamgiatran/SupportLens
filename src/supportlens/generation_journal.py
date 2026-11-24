"""One atomic JSONL record per live local generation, including failed attempts."""

import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time_ns
from uuid import uuid4


def identity_hash(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class GenerationJournal:
    def __init__(self, directory):
        self.directory = Path(directory)

    def save(self, key, record):
        # Keep Windows paths short. Each record carries the full key for collision checks.
        directory = self.directory / key[:12]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{time_ns()}-{uuid4().hex[:12]}.{record['result']['status']}.jsonl"
        # Readers see a complete line even if another worker is still generating.
        temporary = None
        try:
            with NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as handle:
                temporary = Path(handle.name)
                handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def replay(self, key):
        directory = self.directory / key[:12]
        paths = [path for outcome in ("answer", "refused") for path in directory.glob(f"*.{outcome}.jsonl")]
        for path in sorted(paths, reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["key"] == key and row["result"]["status"] in ("answer", "refused"):
                    return row["result"]
            except (KeyError, ValueError, OSError) as error:
                logging.getLogger(__name__).warning("Unreadable generation record %s: %s", path, error)
        return None
