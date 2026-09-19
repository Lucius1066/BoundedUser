"""Stage-1 instrument calibration: score candidate cases for discriminating power.

This runs on a calibration run produced with `--preset calibration`
(blind + full_info_conflict + full_info_support only -- never pretend_blind, so the
selection cannot be biased by the effect under study).

It answers two questions per candidate case:

  1. **Baseline non-degeneracy.** Is `P(A | blind)` inside a usable band, and is the
     whole A/B/C/D distribution non-degenerate for the P(A) endpoint (a case with
     P(A)=0 and P(B)=0.9 has plenty of P(A+B) headroom and none for P(A))?
  2. **Future-information sensitivity.** Does the requirement move the decision when
     its use is permitted, `dGate = P(A|FI_support) - P(A|FI_conflict)`, by enough to
     make a null result at Stage 4 informative?

Thresholds are screening heuristics, not significance tests. They are defaults and
every one of them is a CLI flag so the values used are recorded in the output.

Usage (from inside experiment/):
    python calibrate.py --run-dir results/calib01
    python calibrate.py --run-dir results/calib01 --lo 0.2 --hi 0.8 --gate 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from common import rate_stats

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

BLIND = "blind"
FI_CONFLICT = "full_info_conflict"
FI_SUPPORT = "full_info_support"

#: a case whose reason text is shared by this share of a cell carries little information
CONCENTRATION_WARN = 0.50


def load(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_json(run_dir / "raw_results.jsonl", lines=True, dtype=False)
    for column, default in (("parsed_decision", "UNPARSED"), ("reason", ""), ("condition", "")):
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)
    return frame


def score_case(frame: pd.DataFrame, case_id: str, lo: float, hi: float, gate: float) -> Dict[str, object]:
    record: Dict[str, object] = {"case_id": case_id}
    for condition, prefix in ((BLIND, "blind"), (FI_CONFLICT, "fi_conflict"), (FI_SUPPORT, "fi_support")):
        sub = frame[(frame.case_id == case_id) & (frame.condition == condition)]
        stats = rate_stats(sub["parsed_decision"].tolist())
        record[f"{prefix}_n"] = stats.n
        for label in ("A", "B", "C", "D"):
            record[f"{prefix}_p{label}"] = round(stats.p(label), 4) if stats.n_valid else None
        record[f"{prefix}_pA"] = round(stats.p_a, 4) if stats.n_valid else None
        record[f"{prefix}_pAB"] = round(stats.p_ab, 4) if stats.n_valid else None
        # answer concentration: how much of the cell is one repeated sentence
        counter = sub["reason"].value_counts()
        record[f"{prefix}_top_reason_share"] = round(float(counter.iloc[0]) / len(sub), 2) if len(sub) else None

    p_a_blind = record["blind_pA"]
    gate_value = None
    if record["fi_support_pA"] is not None and record["fi_conflict_pA"] is not None:
        gate_value = round(record["fi_support_pA"] - record["fi_conflict_pA"], 4)
    record["dGate"] = gate_value

    # --- verdicts -----------------------------------------------------------
    checks: Dict[str, bool] = {}
    if p_a_blind is None:
        checks["baseline_band"] = False
        checks["baseline_not_floor_for_A"] = False
    else:
        checks["baseline_band"] = lo < p_a_blind < hi
        # P(A) floor: no room to detect a leakage-driven *decrease* in unconditional accept
        checks["baseline_not_floor_for_A"] = not (p_a_blind <= 0.05 and (record["blind_pB"] or 0) >= 0.5)
    checks["not_ceiling"] = (p_a_blind is None) or (p_a_blind < 0.95)
    checks["gate_positive"] = gate_value is not None and gate_value > 0
    checks["gate_above_threshold"] = gate_value is not None and gate_value > gate
    checks["gate_in_both_arms"] = bool(
        record["fi_conflict_pA"] is not None
        and record["fi_support_pA"] is not None
        and (record["fi_conflict_pA"] < record["fi_support_pA"])
    )
    record["checks"] = checks
    record["selected"] = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    record["failed_checks"] = failed
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="score candidate cases for discriminating power")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lo", type=float, default=0.20, help="lower bound of the usable P(A|blind) band")
    parser.add_argument("--hi", type=float, default=0.80, help="upper bound of the usable P(A|blind) band")
    parser.add_argument("--gate", type=float, default=0.30, help="minimum dGate to keep a case")
    parser.add_argument("--out-dir", default="")
    return run(parser, argv)


def run(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> int:
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    frame = load(run_dir)
    cases = sorted(frame.case_id.unique())

    present = set(frame.condition.unique())
    needed = {BLIND, FI_CONFLICT, FI_SUPPORT}
    if not needed <= present:
        print(
            f"[error] this looks like a full run, not a calibration run. Missing conditions: "
            f"{sorted(needed - present)}\n"
            "Calibration must not include pretend_blind (it would bias case selection).",
            file=sys.stderr,
        )
        return 2
    if any(c.startswith("pretend_blind") for c in present):
        print("[warning] pretend_blind conditions are present in a calibration run.", file=sys.stderr)

    rows = [score_case(frame, case_id, args.lo, args.hi, args.gate) for case_id in cases]
    table = pd.DataFrame(rows)
    selected = table[table.selected]

    print("=" * 110)
    print(f"STAGE-1 CALIBRATION — {run_dir.name}")
    print("=" * 110)
    print(
        f"screening thresholds: P(A|blind) in ({args.lo}, {args.hi}), "
        f"dGate > {args.gate}, no ceiling, no P(A) floor, gate direction positive"
    )
    print(f"candidates: {len(table)}   selected: {len(selected)}")

    print()
    print("PER-CASE SCORES (pA = P(A), pAB = P(A+B), share = most repeated reason in the cell)")
    print("-" * 110)
    show = table[
        [
            "case_id",
            "blind_n",
            "blind_pA",
            "blind_pAB",
            "blind_pB",
            "blind_pC",
            "blind_pD",
            "fi_conflict_pA",
            "fi_support_pA",
            "dGate",
            "blind_top_reason_share",
            "selected",
        ]
    ]
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))

    print()
    print("VERDICT PER CASE")
    print("-" * 110)
    for row in rows:
        mark = "SELECT" if row["selected"] else "reject"
        detail = "" if row["selected"] else "  failed: " + ", ".join(row["failed_checks"])
        print(f"  [{mark}] {row['case_id']:<24} dGate={row['dGate']}{detail}")

    print()
    print("DIAGNOSTIC NOTES")
    print("-" * 110)
    notes: List[str] = []
    for row in rows:
        case_id = row["case_id"]
        for prefix, label in (("blind", "blind"), ("fi_conflict", "FI_conflict"), ("fi_support", "FI_support")):
            share = row.get(f"{prefix}_top_reason_share")
            if share is not None and share >= CONCENTRATION_WARN:
                notes.append(
                    f"{case_id}/{label}: {share:.0%} of samples share one sentence — the cell is close to "
                    "deterministic, so extra samples there add little"
                )
        if row["blind_pA"] is not None and row["blind_pA"] >= 0.95:
            notes.append(f"{case_id}/blind: P(A)={row['blind_pA']:.2f} — ceiling, an increase is impossible")
        if row["blind_pA"] is not None and row["blind_pA"] <= 0.05:
            notes.append(
                f"{case_id}/blind: P(A)={row['blind_pA']:.2f} — floor for the P(A) endpoint "
                f"(P(B)={row['blind_pB']}, P(C)={row['blind_pC']}, P(D)={row['blind_pD']})"
            )
        if row["dGate"] is not None and row["dGate"] <= 0:
            notes.append(
                f"{case_id}: dGate={row['dGate']} — the requirement does not move decisions even when its "
                "use is permitted; nothing can be learned about ignoring it"
            )
    if notes:
        for note in notes:
            print(f"  - {note}")
    else:
        print("  (none)")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["checks", "failed_checks"]).to_csv(out_dir / "calibration_scores.csv", index=False, encoding="utf-8")
    (out_dir / "calibration_selection.json").write_text(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "thresholds": {"lo": args.lo, "hi": args.hi, "gate": args.gate},
                "n_candidates": len(table),
                "selected": selected.case_id.tolist(),
                "rejected": table[~table.selected].case_id.tolist(),
                "per_case": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"wrote {out_dir / 'calibration_scores.csv'}")
    print(f"wrote {out_dir / 'calibration_selection.json'}")
    print()
    print("NEXT (Stage 2, manual): read the `reason` text of the selected cases before freezing anything:")
    print(f"  python postmortem.py --run-dir {run_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
