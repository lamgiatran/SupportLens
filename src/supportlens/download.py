"""Bounded, atomic downloads and allowlisted ZIP extraction; no remote scripts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


MAX_BYTES = 120 * 1024 * 1024
MDD_FILES = {"multidoc2dial_doc.json", "multidoc2dial_dial_train.json",
             "multidoc2dial_dial_validation.json"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None) -> dict:
    if not url.startswith("https://"):
        raise ValueError("only HTTPS dataset sources are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > MAX_BYTES:
                    raise ValueError("download exceeds size budget")
                out.write(block)
            headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified")}
        digest = sha256(temporary)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("SHA-256 mismatch")
        temporary.replace(destination)
        return {"url": url, "sha256": digest, "size_bytes": received, "headers": headers}
    finally:
        temporary.unlink(missing_ok=True)


def extract_multidoc(archive: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            # Select expected basenames and write OUR destination, never extractall.
            name = Path(entry.filename.replace("\\", "/")).name
            if name in MDD_FILES:
                if name in selected:
                    raise ValueError("ambiguous duplicate dataset filename")
                if entry.file_size > MAX_BYTES:
                    raise ValueError("uncompressed member exceeds size budget")
                selected[name] = entry
        if set(selected) != MDD_FILES:
            raise ValueError("archive is missing required raw JSON files")
        for name, entry in selected.items():
            target = output / name
            temporary = target.with_suffix(".json.part")
            try:
                with package.open(entry) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                with temporary.open(encoding="utf-8") as data:
                    parsed = json.load(data)
                required = "doc_data" if name == "multidoc2dial_doc.json" else "dial_data"
                if required not in parsed:
                    raise ValueError(f"{name}: missing {required}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return [output / name for name in sorted(selected)]


def write_manifest(root: Path, sources: list[dict], output: Path,
                   acquisition: str = "downloaded") -> dict:
    records = []
    for source in sources:
        path = root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({**source, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    value = {"manifest_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
             "acquisition": acquisition, "files": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return value


def verify_manifest(root: Path, manifest: dict) -> None:
    for item in manifest["files"]:
        path = root / item["path"]
        if sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"manifest mismatch: {item['path']}")
