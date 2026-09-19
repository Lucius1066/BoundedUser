"""Write a checksum manifest for a directory (used to certify data backups).

Usage:
    python make_backup_manifest.py <directory> [--note "text"]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", default="BACKUP_MANIFEST.json")
    args = parser.parse_args(argv)

    root = Path(args.directory)
    if not root.is_dir():
        print(f"[error] not a directory: {root}", file=sys.stderr)
        return 2

    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != args.out:
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "directory": root.as_posix(),
        "note": args.note,
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    (root / args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {root / args.out}: {len(entries)} files, {manifest['total_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
