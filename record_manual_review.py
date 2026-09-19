"""Attach the Stage-2 manual review decisions to a calibration_selection.json file.

Keeps the audit trail in one place: numeric verdict, manual verdict, reason, and the
final keep/reject decision, so the case-selection rule is documented rather than implicit.

Usage:
    python record_manual_review.py --run-dir results/<calib run> --file decisions.json

decisions.json format:
    {"case_id": {"decision": "keep"|"reject", "reason": "..."}, ...}
    A case with no entry keeps its numeric verdict and is marked "unreviewed".
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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--file", required=True, help="JSON file of manual decisions")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    selection_path = run_dir / "calibration_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manual = json.loads(Path(args.file).read_text(encoding="utf-8"))

    kept, rejected = [], []
    for row in selection["per_case"]:
        case_id = row["case_id"]
        entry = manual.get(case_id)
        if entry is None:
            row["manual"] = {"decision": "unreviewed", "reason": "not in the manual decision file"}
        else:
            row["manual"] = {"decision": entry["decision"], "reason": entry.get("reason", "")}
        numeric_selected = bool(row.get("selected"))
        manual_keep = row["manual"]["decision"] == "keep"
        row["final"] = "keep" if (numeric_selected and manual_keep) else "reject"
        if row["final"] == "keep":
            kept.append(case_id)
        else:
            rejected.append(case_id)

    selection["manual_review"] = {
        "n_reviewed": sum(1 for r in selection["per_case"] if r["manual"]["decision"] != "unreviewed"),
        "final_keep": kept,
        "final_reject": rejected,
        "rule": "keep iff every numeric Stage-1 check passed AND Stage 2 manual review says keep",
    }
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"numeric survivors: {len(selection['selected'])}")
    print(f"final keep ({len(kept)}): {', '.join(kept)}")
    print(f"final reject ({len(rejected)}): {', '.join(rejected)}")
    print(f"wrote {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
