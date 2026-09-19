"""Post-run inspection of a real run: method checks + verbatim reason text.

Writes a markdown report so no console code page can mangle the Chinese reasons.
This is a reporting helper, not part of the analysis pipeline.

Usage (from inside experiment/):
    python postmortem.py --run-dir results/<run> --out POSTMORTEM_POSTMORTEM.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

HERE = Path(__file__).resolve().parent
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

LEAK_TOKENS = ("多实例", "并发写入", "离线", "部署依赖", "阻塞", "串行", "外部客户", "权限控制",
               "multi-instance", "concurrent writes", "offline", "serial", "external customer",
               "access control", "future requirement", "ignore")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    frame = pd.read_json(run_dir / "raw_results.jsonl", lines=True, dtype=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else run_dir / "INSPECTION.md"

    lines: List[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    w(f"# Post-run inspection — `{run_dir.name}`")
    w()
    w(f"- rows: {len(frame)}  |  manifest jobs: {manifest['n_jobs']}  |  "
      f"schema hash: `{manifest['request_schema']['frozen_schema_hash'][:16]}`")
    w(f"- model: {manifest['config']['model']}  temperature={manifest['config']['temperature']}  "
      f"thinking={manifest['config']['thinking']}")
    w(f"- finish_reason counts: {dict(Counter(frame['finish_reason'].fillna('NA')))}")
    w(f"- usage totals: prompt={int(frame['usage'].apply(lambda u: u.get('prompt_tokens', 0)).sum())} "
      f"completion={int(frame['usage'].apply(lambda u: u.get('completion_tokens', 0)).sum())}")

    # ---- 1. degenerate-cell / ceiling check --------------------------------
    w()
    w("## 1. Degenerate cells (a cell that is all-A or all-C cannot discriminate)")
    w()
    w("| case | condition | A | B | C | D | P(A) | P(A+B) | note |")
    w("|---|---|---|---|---|---|---|---|---|")
    per = frame.groupby(["case_id", "condition", "parsed_decision"]).size().unstack(fill_value=0)
    for (case_id, condition), row in per.iterrows():
        counts = {d: int(row.get(d, 0)) for d in ("A", "B", "C", "D")}
        total = sum(counts.values())
        pa = counts["A"] / total if total else float("nan")
        pab = (counts["A"] + counts["B"]) / total if total else float("nan")
        note = ""
        if pa >= 0.95:
            note = "**ceiling** — an increase is impossible here"
        elif pa <= 0.05 and (counts["A"] + counts["B"]) <= 2:
            note = "**floor** — a decrease is impossible here"
        w(f"| {case_id} | {condition} | {counts['A']} | {counts['B']} | {counts['C']} | {counts['D']} | "
          f"{pa:.2f} | {pab:.2f} | {note} |")

    # ---- 2. headline contrast, case by case --------------------------------
    w()
    w("## 2. Headline contrast per case (P(A) and P(A+B))")
    w()
    w("| case | P(A) conflict | P(A) support | dP(A) | P(A+B) conflict | P(A+B) support | dP(A+B) |")
    w("|---|---|---|---|---|---|---|")
    for case_id in sorted(frame["case_id"].unique()):
        vals: Dict[str, Dict[str, float]] = {}
        for condition in ("pretend_blind_conflict", "pretend_blind_support"):
            sub = frame[(frame.case_id == case_id) & (frame.condition == condition)]
            total = len(sub)
            vals[condition] = {
                "a": (sub.parsed_decision == "A").sum() / total,
                "ab": sub.parsed_decision.isin(["A", "B"]).sum() / total,
            }
        w(f"| {case_id} | {vals['pretend_blind_conflict']['a']:.2f} | {vals['pretend_blind_support']['a']:.2f} | "
          f"{vals['pretend_blind_support']['a'] - vals['pretend_blind_conflict']['a']:+.2f} | "
          f"{vals['pretend_blind_conflict']['ab']:.2f} | {vals['pretend_blind_support']['ab']:.2f} | "
          f"{vals['pretend_blind_support']['ab'] - vals['pretend_blind_conflict']['ab']:+.2f} |")

    # ---- 3. how concentrated is the answer distribution? -------------------
    w()
    w("## 3. Answer concentration (identical `reason` strings per cell)")
    w()
    w("A cell where 20/20 samples share one sentence means the sample size buys nothing there.")
    w()
    w("| case | condition | n | unique reasons | most common | share |")
    w("|---|---|---|---|---|---|")
    for (case_id, condition), group in frame.groupby(["case_id", "condition"]):
        counter = Counter(group["reason"].fillna(""))
        top, n_top = counter.most_common(1)[0]
        w(f"| {case_id} | {condition} | {len(group)} | {len(counter)} | {top[:60]} | {n_top / len(group):.0%} |")

    # ---- 4. compliance / leak scan of the reason field ---------------------
    w()
    w("## 4. Compliance scan: does `reason` mention the ignored requirement?")
    w()
    w("Rule-based substring scan only (no semantic judge). The `blind` column should be ~0 by")
    w("construction; a hit in `pretend_blind_*` is evidence of non-compliance worth reading.")
    w()
    w("| condition | rows | rows with a leak token | share |")
    w("|---|---|---|---|")
    for condition, group in frame.groupby("condition"):
        hits = group["reason"].fillna("").apply(lambda r: any(t in r for t in LEAK_TOKENS)).sum()
        w(f"| {condition} | {len(group)} | {hits} | {hits / len(group):.0%} |")
    flagged = frame[frame["reason"].fillna("").apply(lambda r: any(t in r for t in LEAK_TOKENS))]
    if len(flagged):
        w()
        w("Flagged rows (first 15):")
        w()
        for _, row in flagged.head(15).iterrows():
            w(f"- `{row['case_id']} / {row['condition']} / {row['parsed_decision']}` — {str(row['reason'])[:200]}")

    # ---- 5. verbatim reasons ----------------------------------------------
    w()
    w("## 5. Verbatim reasons by cell")
    for case_id in sorted(frame["case_id"].unique()):
        w()
        w(f"### case `{case_id}`")
        for condition in ("blind", "pretend_blind_noreq", "pretend_blind_neutral",
                          "pretend_blind_conflict", "pretend_blind_support",
                          "full_info_conflict", "full_info_support"):
            sub = frame[(frame.case_id == case_id) & (frame.condition == condition)]
            if sub.empty:
                continue
            w()
            w(f"**{condition}** (n={len(sub)})")
            w()
            for decision in ("A", "B", "C", "D"):
                rows = sub[sub.parsed_decision == decision]
                if rows.empty:
                    continue
                uniq = list(dict.fromkeys(rows["reason"].tolist()))
                shown = uniq[:3]
                more = f" …(+{len(uniq) - 3} more unique)" if len(uniq) > 3 else ""
                for reason in shown:
                    w(f"- `{decision}` ×{int((rows['reason'] == reason).sum())}: {reason[:300]}")
                if more:
                    w(f"- {more.strip()}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
