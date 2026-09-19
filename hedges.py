"""Hedge-leak probe: B/D answers in `pretend_blind_*` that mention the ignored requirement.

A→B movement is the leak signature the primary endpoint was designed to expose.  This
script lists the actual rows so the claim can be checked by reading, not by trusting the
distribution.  Rule-based substring only; no semantic judging.

Usage: python hedges.py --run-dir results/<run>
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

#: tokens that only exist in the requirement texts (so a match proves the model used them)
REQUIREMENT_TOKENS = {
    "database_01": ["多实例", "实例", "并发写入", "高并发", "离线", "部署依赖", "multi-instance",
                    "concurrent write", "offline", "deployment dependenc", "replaceable", "replace"],
    "config_loading_02": ["并发", "阻塞", "串行", "排查", "concurrent", "block", "serial", "debug"],
    "frontend_stack_03": ["外部客户", "登录", "权限", "鉴权", "服务端组件", "账号", "external",
                          "login", "access control", "authentication", "account"],
}
PRETEND = ("pretend_blind_conflict", "pretend_blind_support", "pretend_blind_neutral", "pretend_blind_noreq")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    frame = pd.read_json(Path(args.run_dir) / "raw_results.jsonl", lines=True, dtype=False)
    frame["reason"] = frame["reason"].fillna("")

    print("=" * 100)
    print("B/D ANSWERS UNDER AN IGNORE INSTRUCTION — the candidate leak signature (A->B / A->D)")
    print("=" * 100)
    header = f"{'case':<20}{'condition':<24}{'A':>4}{'B':>4}{'C':>4}{'D':>4}"
    print(header)
    for case_id in sorted(frame.case_id.unique()):
        for condition in ("blind",) + PRETEND:
            sub = frame[(frame.case_id == case_id) & (frame.condition == condition)]
            counts = sub.parsed_decision.value_counts()
            print(f"{case_id:<20}{condition:<24}{counts.get('A', 0):>4}{counts.get('B', 0):>4}"
                  f"{counts.get('C', 0):>4}{counts.get('D', 0):>4}")

    print()
    print("=" * 100)
    print("ROWS UNDER `pretend_blind_*` WHOSE reason MENTIONS THE IGNORED REQUIREMENT")
    print("=" * 100)
    hits = 0
    for _, row in frame[frame.condition.isin(PRETEND)].iterrows():
        tokens = REQUIREMENT_TOKENS.get(row["case_id"], [])
        matched = [t for t in tokens if t in row["reason"]]
        if matched:
            hits += 1
            print(f"\n[{hits}] {row['case_id']} / {row['condition']} / {row['parsed_decision']} "
                  f"(matched: {matched})")
            print(f"    {row['reason']}")
    if not hits:
        print("(none)")

    print()
    print("=" * 100)
    print("B AND D ANSWERS UNDER `pretend_blind_*` (verbatim, for reading)")
    print("=" * 100)
    for condition in PRETEND:
        for case_id in sorted(frame.case_id.unique()):
            sub = frame[(frame.case_id == case_id) & (frame.condition == condition)
                        & (frame.parsed_decision.isin(["B", "D"]))]
            if sub.empty:
                continue
            print(f"\n--- {case_id} / {condition}: B={int((sub.parsed_decision == 'B').sum())} "
                  f"D={int((sub.parsed_decision == 'D').sum())}")
            for reason in list(dict.fromkeys(sub["reason"].tolist()))[:6]:
                print(f"    - {reason[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
