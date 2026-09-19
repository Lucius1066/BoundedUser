"""Build the final frozen case set from an explicit selection.

Round-2 protocol: the frozen set must be exactly the cases that passed both Stage-1
numeric screening and Stage-2 manual review, on one consistent measurement basis.
This script writes that subset (and records which cases were dropped and why), so the
frozen file cannot silently drift from the selection file.

Usage:
    python freeze_cases.py --all-cases cases_frozen.json \
        --selection results/calib03_frozen8_n10/calibration_selection.json \
        --out cases_frozen_selected.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-cases", required=True, help="case file containing every candidate")
    parser.add_argument("--selection", required=True, help="calibration_selection.json with final_keep")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    all_cases = json.loads(Path(args.all_cases).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    keep = selection.get("manual_review", {}).get("final_keep")
    if not keep:
        print("[error] selection file has no manual_review.final_keep", file=sys.stderr)
        return 2

    by_id = {case["id"]: case for case in all_cases}
    missing = [case_id for case_id in keep if case_id not in by_id]
    if missing:
        print(f"[error] selected cases not found in {args.all_cases}: {missing}", file=sys.stderr)
        return 2

    frozen = [by_id[case_id] for case_id in keep]
    dropped = [case["id"] for case in all_cases if case["id"] not in set(keep)]

    out = Path(args.out)
    out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(frozen)} frozen cases")
    for case_id in keep:
        case = by_id[case_id]
        print(f"  keep   {case_id:<32} domain={case.get('domain', '-')}")
    for case_id in dropped:
        print(f"  drop   {case_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
