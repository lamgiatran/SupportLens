"""Package tracked source + a portable Git bundle; never include raw data/venv."""

import argparse
import json
from pathlib import Path
import subprocess
import zipfile


def git(*args):
    return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if git("status", "--porcelain").strip():
        raise ValueError("commit source changes before packaging")
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = destination.with_suffix(".bundle")
    subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=root, check=True)
    tracked = [x.decode("utf-8") for x in git("ls-files", "-z").split(b"\0") if x]
    commit = git("rev-parse", "HEAD").decode().strip()
    tags = git("tag", "--points-at", "HEAD").decode().splitlines()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in tracked:
            archive.write(root / name, "SupportLens/" + name)
        archive.write(bundle, "SupportLens.history.bundle")
        archive.writestr("PACKAGE.json", json.dumps({"commit": commit, "tags": tags,
                                                    "source_files": len(tracked)}, indent=2))
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
