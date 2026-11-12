"""Download or reassemble original follow-up experiment archives."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request


RELEASE = "https://github.com/Shengda21/zigzag/releases/download/followup-raw-archives-v1"


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def asset_path(directory, name):
    if name in {"", ".", ".."} or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError(f"Invalid filename in manifest: {name}")
    return directory / name


def save_verified(destination, record, write):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        path = Path(temp.name)
        try:
            write(temp)
            temp.close()
            if not matches(path, record):
                raise ValueError(f"SHA-256 or size mismatch for {destination.name}")
            path.replace(destination)
        finally:
            temp.close()
            path.unlink(missing_ok=True)


def get_part(record, cache, local_only):
    name = record["name"]
    part = asset_path(cache, name)
    if matches(part, record):
        return part
    if local_only:
        raise ValueError(f"Missing or corrupt part: {part}")
    url = f"{RELEASE}/{urllib.parse.quote(name)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        save_verified(part, record, lambda temp: shutil.copyfileobj(response, temp))
    return part


def restore(archive, output, cache, local_only=False):
    destination = asset_path(output, archive["name"])
    if destination.exists():
        if not matches(destination, archive):
            raise ValueError(f"Existing file differs from the manifest: {destination}")
        return destination

    def write(temp):
        for part in archive["parts"]:
            with get_part(part, cache, local_only).open("rb") as stream:
                shutil.copyfileobj(stream, temp)

    save_verified(destination, archive, write)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", help="Original filename; repeat for multiple archives")
    parser.add_argument("--all", action="store_true", help="Restore every archive")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parts-dir", type=Path, help="Directory containing manually downloaded parts")
    parser.add_argument("--manifest", type=Path, help="Use a local copy of the release manifest")
    args = parser.parse_args()
    if args.all == bool(args.archive):
        parser.error("choose --all or one or more --archive options")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf8"))
    else:
        with urllib.request.urlopen(f"{RELEASE}/raw-archive-manifest.json", timeout=120) as response:
            manifest = json.load(response)
    archives = {row["name"]: row for row in manifest["archives"]}
    cache = args.parts_dir or (args.output / ".parts")
    for name in archives if args.all else dict.fromkeys(args.archive):
        if name not in archives:
            parser.error(f"unknown archive {name}; see raw-archive-manifest.json")
        path = restore(archives[name], args.output, cache, local_only=args.parts_dir is not None)
        print(f"Verified {path}", flush=True)


if __name__ == "__main__":
    main()
