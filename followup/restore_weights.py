"""Restore a follow-up study's saved weights to their original paths."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


RELEASE = "https://github.com/Shengda21/zigzag/releases/download/trained-weights-v1"


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def save_verified(stream, destination, record):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            shutil.copyfileobj(stream, tmp)
            tmp.close()
            if not matches(temporary, record):
                raise ValueError(f"Digest mismatch for {destination}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def restore(manifest, archive_name, output, cache, prefix=""):
    archive = next((row for row in manifest["archives"]
                    if row["archive"] == archive_name), None)
    if archive is None:
        raise ValueError(f"Unknown archive: {archive_name}")
    files = [row for row in archive["files"] if row["path"].startswith(prefix)]
    if not files:
        raise ValueError(f"No weights found in {archive_name} for prefix {prefix!r}")

    output = output.resolve()
    verified_assets = {}
    for row in files:
        relative = PurePosixPath(row["path"])
        destination = output.joinpath(*relative.parts)
        if (relative.is_absolute() or ".." in relative.parts
                or not destination.resolve().is_relative_to(output)):
            raise ValueError(f"Invalid checkpoint path: {relative}")
        if destination.exists():
            if not matches(destination, row):
                raise ValueError(f"Existing checkpoint differs: {destination}")
            continue

        name = row["asset"]
        if Path(name).name != name:
            raise ValueError(f"Invalid asset name: {name}")
        if name not in verified_assets:
            asset = cache / name
            expected = manifest["assets"][name]
            if not matches(asset, expected):
                url = f"{RELEASE}/{urllib.parse.quote(name)}"
                with urllib.request.urlopen(url, timeout=120) as response:
                    save_verified(response, asset, expected)
            verified_assets[name] = asset

        asset = verified_assets[name]
        if "member" in row:
            with zipfile.ZipFile(asset) as package, package.open(row["member"]) as stream:
                save_verified(stream, destination, row)
        else:
            with asset.open("rb") as stream:
                save_verified(stream, destination, row)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Original archive name from the weight manifest")
    parser.add_argument("--output", required=True, type=Path, help="Experiment working root")
    parser.add_argument("--cache", type=Path, default=Path(".zigzag-weight-cache"))
    parser.add_argument("--prefix", default="", help="Restore only checkpoint paths with this prefix")
    args = parser.parse_args()
    with urllib.request.urlopen(f"{RELEASE}/weight-manifest.json", timeout=120) as response:
        manifest = json.load(response)
    count = restore(manifest, args.archive, args.output, args.cache, args.prefix)
    print(f"Verified {count} files in {args.output}")


if __name__ == "__main__":
    main()
