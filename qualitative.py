"""Qualitative extract for Stage 4: what the model actually says per condition.

Writes markdown so console code pages cannot mangle the Chinese text.

Usage: python qualitative.py --run-dir results/<run> --cases approval_31_single_approver,sorting_23_codepoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

DEFAULT_CONDITIONS = [
    "blind",
    "pretend_blind_noreq",
    "pretend_blind_neutral",
    "pretend_blind_conflict",
    "pretend_blind_support",
    "full_info_conflict",
    "full_info_support",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--out", default="QUALITATIVE.md")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    frame = pd.read_json(run_dir / "raw_results.jsonl", lines=True, dtype=False)
    frame["reason"] = frame["reason"].fillna("")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or sorted(frame.case_id.unique())

    lines = ["# Qualitative extract — what the model says, by condition", ""]
    for case_id in cases:
        lines.append(f"## case `{case_id}`")
        lines.append("")
        for condition in DEFAULT_CONDITIONS:
            sub = frame[(frame.case_id == case_id) & (frame.condition == condition)]
            if sub.empty:
                continue
            counts = sub.parsed_decision.value_counts().to_dict()
            lines.append(
                f"**{condition}** — n={len(sub)} "
                + " ".join(f"{k}={counts.get(k, 0)}" for k in ("A", "B", "C", "D"))
            )
            lines.append("")
            for decision in ("A", "B", "C", "D"):
                rows = sub[sub.parsed_decision == decision]
                if rows.empty:
                    continue
                unique = list(dict.fromkeys(rows["reason"].tolist()))
                for reason in unique[: args.per_cell]:
                    n = int((rows["reason"] == reason).sum())
                    lines.append(f"- `{decision}` ×{n}: {reason}")
            lines.append("")
    out = run_dir / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
