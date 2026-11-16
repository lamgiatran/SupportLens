"""Package tracked source + a portable Git bundle; never include raw data/venv."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def git(*args):
    return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-banking-cache", action="store_true",
                        help="include the verified, untracked NPZ in this handoff ZIP only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if git("status", "--porcelain").strip():
        raise ValueError("commit source changes before packaging")
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracked = [x.decode("utf-8") for x in git("ls-files", "-z").split(b"\0") if x]
    oversized = [name for name in tracked if (root / name).stat().st_size > 10 * 1024 * 1024]
    if oversized:
        raise ValueError(f"tracked files exceed the 10 MiB source-package budget: {oversized}")
    forbidden = [name for name in tracked if name.startswith(("data/raw/", "data/tokenizers/", ".venv/"))
                 or name == ".env" or Path(name).suffix in {".npz", ".npy", ".safetensors", ".gguf", ".pt", ".pth"}]
    if forbidden:
        raise ValueError(f"large/private runtime files must not be tracked in the source package: {forbidden}")
    shared_assets = []
    if args.include_banking_cache:
        cache = root / "data/processed/handoff50/embeddings.npz"
        manifest_path = cache.with_name("embedding_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        if digest != manifest["npz_sha256"]:
            raise ValueError("Banking cache bytes do not match the manifest")
        shared_assets.append({"path": cache.relative_to(root).as_posix(),
                              "sha256": digest, "bytes": cache.stat().st_size})
    bundle = destination.with_suffix(".bundle")
    subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=root, check=True)
    commit = git("rev-parse", "HEAD").decode().strip()
    tags = git("tag", "--points-at", "HEAD").decode().splitlines()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in tracked:
            archive.write(root / name, "SupportLens/" + name)
        for asset in shared_assets:
            archive.write(root / asset["path"], "SupportLens/" + asset["path"])
        archive.write(bundle, "SupportLens.history.bundle")
        archive.writestr("PACKAGE.json", json.dumps({"commit": commit, "tags": tags,
                                                    "source_files": len(tracked),
                                                    "shared_assets": shared_assets}, indent=2))
        archive.writestr("OPEN_FIRST.txt", "Source: SupportLens/README.md\n"
                          "Optional Git history/tag restore:\n"
                          "git clone SupportLens.history.bundle SupportLens-with-history\n"
                          "No remote GitHub repository has been created.\n"
                          "Raw data, tokenizers, model weights and virtualenv are excluded.\n")
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any("/.venv/" in n or "/data/raw/" in n or "/data/tokenizers/" in n for n in names)
    print(json.dumps({"archive": str(destination), "files": len(names),
                      "commit": commit, "tags": tags, "bytes": destination.stat().st_size}, ensure_ascii=True))


if __name__ == "__main__":
    main()
