"""Aggregate the raw samples and report the pre-specified comparisons.

Primary endpoint (see DESIGN.md section 7):
    P(A | parsed decision)      -- unconditional acceptance
Secondary:
    P(A+B | parsed decision)    -- acceptance with or without reservations
Separate, never mixed in:
    P(UNPARSED), P(ERROR)       -- over ALL samples, so nothing is silently dropped

The headline comparison is `pretend_blind_support` vs `pretend_blind_conflict`:
identical system prompt, identical instruction, identical structure, only the text
of the requirement that the model was told to ignore differs.

Usage (from inside experiment/):
    python analyze.py                       # newest run in results/
    python analyze.py --run-dir results/<name>
    python analyze.py --jsonl path/to/raw_results.jsonl --out-dir reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from common import VALID_DECISIONS, ERROR, UNPARSED, rate_stats
from prompts import (
    ALL_CONDITIONS,
    ALL_LABELS,
    BLIND,
    FULL_INFO_CONFLICT,
    FULL_INFO_SUPPORT,
    PRETEND_BLIND_CONFLICT,
    PRETEND_BLIND_NEUTRAL,
    PRETEND_BLIND_NOREQ,
    PRETEND_BLIND_SUPPORT,
)

HERE = Path(__file__).resolve().parent
ENCODING = "utf-8"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

#: (baseline, compared, is_headline, what it isolates)
COMPARISONS: List[Tuple[str, str, bool, str]] = [
    (
        PRETEND_BLIND_CONFLICT,
        PRETEND_BLIND_SUPPORT,
        True,
        "HEADLINE: only the ignored requirement's content changes",
    ),
    (BLIND, PRETEND_BLIND_CONFLICT, False, "information access + order-to-ignore (conflict)"),
    (BLIND, PRETEND_BLIND_SUPPORT, False, "information access + order-to-ignore (support)"),
    (PRETEND_BLIND_NOREQ, PRETEND_BLIND_CONFLICT, False, "requirement content added to the ignore instruction"),
    (PRETEND_BLIND_NOREQ, PRETEND_BLIND_SUPPORT, False, "requirement content added to the ignore instruction"),
    (PRETEND_BLIND_NEUTRAL, PRETEND_BLIND_CONFLICT, False, "option-relevant vs option-neutral content, same instruction"),
    (PRETEND_BLIND_NEUTRAL, PRETEND_BLIND_SUPPORT, False, "option-relevant vs option-neutral content, same instruction"),
    (BLIND, PRETEND_BLIND_NOREQ, False, "meta-instruction effect (no requirement at all)"),
    (BLIND, PRETEND_BLIND_NEUTRAL, False, "meta-instruction + a future requirement exists"),
    (FULL_INFO_CONFLICT, FULL_INFO_SUPPORT, False, "INSTRUMENT GATE: same contrast with use permitted"),
    (BLIND, FULL_INFO_CONFLICT, False, "information access, use permitted (conflict)"),
    (BLIND, FULL_INFO_SUPPORT, False, "information access, use permitted (support)"),
]

#: a baseline at or above this leaves no room to detect suppression
CEILING_THRESHOLD = 0.90


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def latest_run_dir(results_dir: Path) -> Path:
    if not results_dir.is_dir():
        raise SystemExit(f"[error] results directory not found: {results_dir}")
    runs = [p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not runs:
        raise SystemExit(f"[error] no run directories inside {results_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_records(run_dir: Optional[Path], jsonl: Optional[Path]) -> Tuple[pd.DataFrame, Optional[Path]]:
    if jsonl is None:
        if run_dir is None:
            run_dir = latest_run_dir(HERE / "results")
        jsonl = run_dir / "raw_results.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"[error] no results at {jsonl}")
    frame = pd.read_json(jsonl, lines=True, dtype=False)
    if frame.empty:
        raise SystemExit(f"[error] {jsonl} contains no rows")
    for column, default in (
        ("parsed_decision", UNPARSED),
        ("condition", ""),
        ("case_id", ""),
        ("reason", ""),
        ("future_requirement_variant", ""),
    ):
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)
    if "error" not in frame.columns:
        frame["error"] = ""
    frame["error"] = frame["error"].fillna("").astype(str)
    frame["is_valid"] = frame["parsed_decision"].isin(VALID_DECISIONS)
    frame["is_error"] = frame["parsed_decision"] == ERROR
    frame["is_unparsed"] = frame["parsed_decision"] == UNPARSED
    frame["accepted"] = frame["parsed_decision"].isin(["A", "B"])
    frame["accepted_clean"] = frame["parsed_decision"] == "A"
    return frame, jsonl


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table(frame: pd.DataFrame, index: Sequence[str]) -> pd.DataFrame:
    """Counts, plus two proportion families with DIFFERENT denominators.

      * `<label>_p`     for A/B/C/D: share of parsed decisions (valid_n)
      * `<label>_p_all` for UNPARSED/ERROR: share of ALL samples (all_n)

    Dividing a failure count by the decision denominator would inflate it (it can
    even exceed 1), so the two families are kept apart on purpose.
    """
    counts = (
        frame.pivot_table(index=list(index), columns="parsed_decision", values="run_id", aggfunc="count")
        .reindex(columns=list(ALL_LABELS))
        .fillna(0)
        .astype(int)
    )
    counts = counts.reindex(columns=[c for c in ALL_LABELS if counts[c].sum() > 0])
    decision_cols = [c for c in counts.columns if c in VALID_DECISIONS]
    decision_n = counts[decision_cols].sum(axis=1)
    all_n = counts.sum(axis=1)

    out = counts.add_suffix("_n")
    for column in counts.columns:
        if column in VALID_DECISIONS:
            out[f"{column}_p"] = (
                counts[column].div(decision_n.replace(0, pd.NA)).fillna(0.0).round(4)
            )
        else:
            out[f"{column}_p_all"] = (
                counts[column].div(all_n.replace(0, pd.NA)).fillna(0.0).round(4)
            )
    out["valid_n"] = decision_n.astype(int)
    out["all_n"] = all_n.astype(int)
    return out


def condition_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """The endpoint table: P(A), P(A+B), and the failure rates, per condition."""
    rows: List[Dict[str, object]] = []
    for condition, group in frame.groupby("condition"):
        stats = rate_stats(group["parsed_decision"].tolist())
        rows.append(
            {
                "condition": condition,
                "all_n": stats.n,
                "valid_n": stats.n_valid,
                "P(A)": round(stats.p_a, 4) if stats.n_valid else float("nan"),
                "P(A+B)": round(stats.p_ab, 4) if stats.n_valid else float("nan"),
                "P(B)": round(stats.p("B"), 4) if stats.n_valid else float("nan"),
                "P(C)": round(stats.p("C"), 4) if stats.n_valid else float("nan"),
                "P(D)": round(stats.p("D"), 4) if stats.n_valid else float("nan"),
                "se_P(A)": round(stats.se_a(), 4),
                "n_unparsed": stats.n_unparsed,
                "n_error": stats.n_error,
                "P(UNPARSED|all)": round(stats.p_unparsed_of_all, 4),
                "P(ERROR|all)": round(stats.p_error_of_all, 4),
            }
        )
    out = pd.DataFrame(rows)
    order = [c for c in ALL_CONDITIONS if c in set(out["condition"])]
    return out.set_index("condition").reindex(order).reset_index()


def metrics_by_case(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (case_id, condition), group in frame.groupby(["case_id", "condition"]):
        stats = rate_stats(group["parsed_decision"].tolist())
        rows.append(
            {
                "case_id": case_id,
                "condition": condition,
                "all_n": stats.n,
                "valid_n": stats.n_valid,
                "P(A)": round(stats.p_a, 4) if stats.n_valid else float("nan"),
                "P(A+B)": round(stats.p_ab, 4) if stats.n_valid else float("nan"),
                "P(C)": round(stats.p("C"), 4) if stats.n_valid else float("nan"),
                "P(D)": round(stats.p("D"), 4) if stats.n_valid else float("nan"),
                "n_unparsed": stats.n_unparsed,
                "n_error": stats.n_error,
            }
        )
    return pd.DataFrame(rows).sort_values(["case_id", "condition"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def _metric_map(frame: pd.DataFrame, index: Sequence[str], metric: str) -> Dict[object, float]:
    """{index key -> rate} for a rate defined on valid responses only."""
    if metric == "P(A)":
        column = "accepted_clean"
    elif metric == "P(A+B)":
        column = "accepted"
    else:
        raise ValueError(metric)
    valid = frame[frame["is_valid"]]
    grouped = valid.groupby(list(index))[column].mean()
    return {key: float(value) for key, value in grouped.items()}


def comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    pooled_a = _metric_map(frame, ["condition"], "P(A)")
    pooled_ab = _metric_map(frame, ["condition"], "P(A+B)")
    per_case_a = _metric_map(frame, ["case_id", "condition"], "P(A)")
    cases = sorted(frame["case_id"].unique())

    rows: List[Dict[str, object]] = []
    for baseline, compared, headline, isolates in COMPARISONS:
        if baseline not in pooled_a or compared not in pooled_a:
            continue
        deltas = [
            per_case_a.get((case, compared), float("nan")) - per_case_a.get((case, baseline), float("nan"))
            for case in cases
        ]
        deltas = [d for d in deltas if d == d]  # drop NaN
        positives = sum(1 for d in deltas if d > 0)
        negatives = sum(1 for d in deltas if d < 0)
        zeros = sum(1 for d in deltas if d == 0)
        rows.append(
            {
                "headline": "YES" if headline else "",
                "baseline": baseline,
                "compared": compared,
                "P(A)_base": round(pooled_a[baseline], 4),
                "P(A)_cmp": round(pooled_a[compared], 4),
                "dP(A)": round(pooled_a[compared] - pooled_a[baseline], 4),
                "dP(A+B)": round(pooled_ab[compared] - pooled_ab[baseline], 4),
                "per_case_min": round(min(deltas), 4) if deltas else float("nan"),
                "per_case_max": round(max(deltas), 4) if deltas else float("nan"),
                "cases_up": f"{positives}/{len(deltas)}",
                "cases_down": f"{negatives}/{len(deltas)}",
                "cases_flat": f"{zeros}/{len(deltas)}",
                "isolates": isolates,
            }
        )
    ordered = pd.DataFrame(rows)
    if not ordered.empty:
        ordered = pd.concat(
            [ordered[ordered["headline"] == "YES"], ordered[ordered["headline"] != "YES"]],
            ignore_index=True,
        )
    return ordered


def ceiling_report(frame: pd.DataFrame) -> List[str]:
    """Warn when a baseline is so high that an upward effect cannot be seen."""
    notes: List[str] = []
    per_case = _metric_map(frame, ["case_id", "condition"], "P(A)")
    cases = sorted(frame["case_id"].unique())
    for condition in (BLIND, PRETEND_BLIND_CONFLICT):
        values = [per_case.get((case, condition)) for case in cases]
        values = [v for v in values if v is not None and v == v]
        if not values:
            continue
        mean = sum(values) / len(values)
        if mean >= CEILING_THRESHOLD:
            notes.append(
                f"{condition}: mean P(A) = {mean:.2f} across {len(values)} case(s) "
                f"(>= {CEILING_THRESHOLD:.2f}) -- at ceiling, so a *support*-direction shift "
                "cannot be detected; read the conflict direction and the paired contrast instead."
            )
    return notes


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def print_section(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def _to_string(frame: pd.DataFrame, width: int = 250) -> None:
    with pd.option_context("display.width", width):
        print(frame.to_string(index=False))


def report_manifest(run_dir: Optional[Path], frame: pd.DataFrame) -> Dict[str, object]:
    manifest: Dict[str, object] = {}
    path = (run_dir / "manifest.json") if run_dir else None
    if path is not None and path.is_file():
        manifest = json.loads(path.read_text(encoding=ENCODING))
        config = manifest.get("config", {})
        print_section("RUN")
        print(
            f"run={manifest.get('run_name')}  provider={config.get('provider')}  "
            f"model={config.get('model')}  temperature={config.get('temperature')}  "
            f"top_p={config.get('top_p')}  seed={config.get('seed')}  "
            f"thinking={config.get('thinking')}  order={manifest.get('execution_order')}"
        )
        if config.get("sampling_note"):
            print(f"sampling note : {config['sampling_note']}")
        schema = manifest.get("request_schema", {})
        if schema:
            print(
                f"request schema: {schema.get('mode')} "
                f"hash={str(schema.get('frozen_schema_hash'))[:16]} "
                f"dropped={schema.get('dropped_params')}"
            )
    print(f"rows={len(frame)}  cases={frame['case_id'].nunique()}  conditions={frame['condition'].nunique()}")
    unknown = sorted(set(frame["parsed_decision"]) - set(ALL_LABELS))
    if unknown:
        print(f"[warning] unexpected decision labels present: {unknown}")
    return manifest


def report_primaries(frame: pd.DataFrame, out_dir: Optional[Path]) -> pd.DataFrame:
    print_section("PRIMARY ENDPOINT TABLE — P(A) = unconditional acceptance (denominator = parsed decisions)")
    metrics = condition_metrics(frame)
    _to_string(metrics)
    print(
        "\nreading guide: P(A) is the primary endpoint because A->B movement ('accept, but do not\n"
        "commit too hard') is exactly the leak signature of interest, and P(A+B) hides it.\n"
        "P(UNPARSED|all) and P(ERROR|all) use ALL samples as denominator so a condition that\n"
        "fails more often stays visible instead of being absorbed into the accept rate."
    )
    if out_dir:
        metrics.to_csv(out_dir / "condition_metrics.csv", encoding=ENCODING, index=False)
    return metrics


def report_comparisons(frame: pd.DataFrame, out_dir: Optional[Path]) -> pd.DataFrame:
    print_section("COMPARISONS — headline first (pa = pretend_blind)")
    table_df = comparisons(frame)
    _to_string(table_df, width=300)
    print(
        "\nreading guide:\n"
        "  HEADLINE  pa_conflict -> pa_support : only the ignored requirement's text changes.\n"
        "            A dP(A) away from zero here means the decision moved with information the model\n"
        "            was explicitly told to ignore -- the phenomenon itself. The predicted direction\n"
        "            is dP(A) > 0 (a supporting requirement should make acceptance more likely).\n"
        "  GATE      full_info_conflict -> full_info_support : if this is ~0, the instrument\n"
        "            cannot see the information at all, so no leakage claim is interpretable.\n"
        "  CONTROLS  blind -> pa_noreq / pa_neutral : how much the ignore instruction and the\n"
        "            mere existence of a future requirement move the decision on their own.\n"
        "  cases_up / cases_down / cases_flat = per-case paired deltas counted by sign\n"
        "            (independently of which direction is 'expected')."
    )
    for note in ceiling_report(frame):
        print(f"\n[ceiling] {note}")
    if out_dir:
        table_df.to_csv(out_dir / "comparisons.csv", encoding=ENCODING, index=False)
    return table_df


def report_per_case(frame: pd.DataFrame, out_dir: Optional[Path]) -> None:
    print_section("PER-CASE ENDPOINTS")
    per_case = metrics_by_case(frame)
    _to_string(per_case)
    if out_dir:
        per_case.to_csv(out_dir / "metrics_by_case.csv", encoding=ENCODING, index=False)

    print_section("DECISION DISTRIBUTION BY CASE x CONDITION (n and proportion)")
    wide = table(frame, ["case_id", "condition"]).reset_index()
    _to_string(wide, width=300)
    if out_dir:
        wide.to_csv(out_dir / "distribution_by_case_condition.csv", encoding=ENCODING, index=False)


def report_health(frame: pd.DataFrame) -> None:
    print_section("SAMPLE HEALTH (ERROR / UNPARSED are kept, never dropped)")
    summary = (
        frame.groupby("condition")
        .agg(
            n=("run_id", "count"),
            n_error=("is_error", "sum"),
            n_unparsed=("is_unparsed", "sum"),
            n_valid=("is_valid", "sum"),
        )
        .reset_index()
    )
    summary["P(ERROR|all)"] = (summary["n_error"] / summary["n"]).round(4)
    summary["P(UNPARSED|all)"] = (summary["n_unparsed"] / summary["n"]).round(4)
    order = [c for c in ALL_CONDITIONS if c in set(summary["condition"])]
    summary["__order"] = summary["condition"].map({c: i for i, c in enumerate(order)})
    summary = summary.sort_values("__order").drop(columns="__order")
    _to_string(summary)
    if summary["n_error"].sum():
        print("\nerror kinds:")
        kinds = (
            frame[frame["is_error"]]
            .groupby(["condition", "error_kind"])
            .size()
            .rename("n")
            .reset_index()
        )
        _to_string(kinds)
        print("(these samples are excluded from decision rates but remain in P(*|all) and in the JSONL)")
    if summary["n_unparsed"].sum():
        print("\n[note] UNPARSED responses: inspect `raw_response` / `parse_note` for these rows.")
        print("       A cluster of UNPARSED in one condition is a finding (non-compliance), not noise.")

    if "reasoning_chars" in frame.columns:
        with_reasoning = int((pd.to_numeric(frame["reasoning_chars"], errors="coerce").fillna(0) > 0).sum())
        if with_reasoning:
            print(
                f"\n[warning] {with_reasoning}/{len(frame)} responses contain reasoning_content: "
                "thinking mode was active, so temperature was NOT an effective control."
            )
    if "schema_hash" in frame.columns:
        hashes = sorted({str(h) for h in frame["schema_hash"].dropna().unique() if str(h)})
        if len(hashes) > 1:
            print(f"\n[warning] {len(hashes)} distinct request-schema hashes in this run: {hashes}")
            print("          request parameters were not constant across samples.")


def report_reasons(frame: pd.DataFrame, per_cell: int, out_dir: Optional[Path]) -> None:
    print_section(f"RAW REASONS (up to {per_cell} unique per case x condition; full text in raw_results.jsonl)")
    for (case_id, condition), group in frame.groupby(["case_id", "condition"], sort=True):
        seen: List[str] = []
        for reason in group["reason"].tolist():
            if reason.strip() and reason not in seen:
                seen.append(reason)
        print(f"\n--- {case_id} / {condition} (n={len(group)}) ---")
        if not seen:
            print("  (no reason text)")
        for reason in seen[:per_cell]:
            print(f"  - {reason}")
        if len(seen) > per_cell:
            print(f"  ... {len(seen) - per_cell} more unique reasons")
    if out_dir:
        dump = (
            frame.groupby(["case_id", "condition", "parsed_decision"])["reason"]
            .apply(lambda values: "\n".join(sorted({v for v in values if v.strip()})))
            .reset_index()
        )
        dump.to_csv(out_dir / "reasons_by_cell.csv", encoding=ENCODING, index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="analyze the phenomenon-check results")
    parser.add_argument("--run-dir", default="", help="run directory (default: newest under results/)")
    parser.add_argument("--jsonl", default="", help="analyze a specific raw_results.jsonl instead")
    parser.add_argument("--out-dir", default="", help="write CSV tables here (default: the run directory)")
    parser.add_argument("--no-csv", action="store_true", help="print only, write nothing")
    parser.add_argument("--reasons", type=int, default=4, help="unique reasons to print per cell (0 = skip)")
    parser.add_argument("--quiet-reasons", action="store_true", help="alias for --reasons 0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else None
    jsonl = Path(args.jsonl) if args.jsonl else None
    frame, jsonl_path = load_records(run_dir, jsonl)
    if run_dir is None and jsonl_path is not None:
        run_dir = jsonl_path.parent

    out_dir: Optional[Path] = None
    if not args.no_csv:
        out_dir = Path(args.out_dir) if args.out_dir else run_dir
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

    report_manifest(run_dir, frame)
    report_primaries(frame, out_dir)
    report_comparisons(frame, out_dir)
    report_per_case(frame, out_dir)
    report_health(frame)
    reasons = 0 if args.quiet_reasons else args.reasons
    if reasons:
        report_reasons(frame, reasons, out_dir)

    print(
        "\ndecision codes: A = accept as is  B = accept with reservation  C = reject  D = need more info"
    )
    print("non-decisions: UNPARSED = no valid JSON answer, ERROR = API call failed after retries")
    if out_dir:
        print_section("FILES WRITTEN")
        for path in sorted(out_dir.glob("*.csv")):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
