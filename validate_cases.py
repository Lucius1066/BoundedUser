"""Static checks on a candidate-case file, BEFORE spending any API budget.

Round 1 lost two of three cases to defects that were visible in the text itself:

  * `config_loading_02` let the model reject the proposal for a reason unrelated to the
    experiment (a local-directory config loaded with the `requests` HTTP library);
  * two cases sat at ceiling/floor for the P(A) endpoint and could not discriminate.

Those are two different failure classes and only the first is checkable statically.  This
script checks the static class: structural completeness, conflict/support symmetry, an
obvious-semantic-conflict heuristic, and overlap between the requirement texts and the
current context (a requirement that restates the context is not new information).

It cannot check whether a proposal is "reasonable but not obviously right" -- that is what
Stage 1 calibration measures, and Stage 2 reads.

Usage (from inside experiment/):
    python validate_cases.py --cases cases_candidates_01.json
    python validate_cases.py --cases cases_candidates_01.json --check   # exit 1 on any error
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from common import CaseFileError, load_cases
from prompts import Case

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

#: technology nouns that make a mismatch checkable: if the context says a resource is
#: local/offline and the proposal names a network/remote mechanism (or vice versa), the
#: model can reject the proposal for a reason that has nothing to do with the experiment.
LOCAL_MARKERS = ("本地", "离线", "内网", "家里", "本机", "local", "offline", "intranet")
REMOTE_MARKERS = ("requests", "http", "云端", "云上", "远程", "上传", "轮询", "接口", "api",
                  "cloud", "remote", "upload", "polling")

#: a proposal that names no concrete mechanism gives the model nothing to accept or reject
VAGUE_PROPOSAL = ("优化", "重构", "改进", "调整", "完善", "better", "improve", "optimize")

LENGTH_TOLERANCE = 8          # chars between conflict and support requirement
MIN_REQUIREMENT_CHARS = 12


def requirement_lengths(case: Case) -> Tuple[int, int, int]:
    return (
        len(case.future_requirement_conflict),
        len(case.future_requirement_support),
        len(case.neutral_requirement),
    )


def check_case(case: Case) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for one case."""
    errors: List[str] = []
    warnings: List[str] = []

    conflict, support, neutral = requirement_lengths(case)
    gap = abs(conflict - support)
    if gap > LENGTH_TOLERANCE:
        warnings.append(
            f"conflict({conflict}) vs support({support}) differ by {gap} chars "
            f"(> {LENGTH_TOLERANCE}); length is a known confound for the content contrast"
        )
    for label, text in (
        ("conflict", case.future_requirement_conflict),
        ("support", case.future_requirement_support),
        ("neutral", case.neutral_requirement),
    ):
        if len(text) < MIN_REQUIREMENT_CHARS:
            errors.append(f"{label} requirement is shorter than {MIN_REQUIREMENT_CHARS} chars: {text!r}")
    if case.future_requirement_conflict.strip() == case.future_requirement_support.strip():
        errors.append("conflict and support requirement texts are identical")

    # requirement mirrored in the context = no new information for the model
    for label, text in (
        ("conflict", case.future_requirement_conflict),
        ("support", case.future_requirement_support),
    ):
        stem = re.sub(r"^(未来要求|未来需要|未来)", "", text).strip("，。 ")
        if len(stem) >= 6 and stem in case.current_context:
            errors.append(f"{label} requirement restates the current_context: {stem!r}")

    # local/remote mismatch, the round-1 failure mode
    ctx_local = [m for m in LOCAL_MARKERS if m in case.current_context]
    proposal_remote = [m for m in REMOTE_MARKERS if m in case.agent_proposal]
    if ctx_local and proposal_remote:
        warnings.append(
            f"context says local/offline ({', '.join(ctx_local)}) while the proposal names "
            f"remote/network machinery ({', '.join(proposal_remote)}) — the model may reject the "
            "proposal for a reason unrelated to the future requirement (round-1 failure mode)"
        )
    ctx_remote = [m for m in REMOTE_MARKERS if m in case.current_context]
    proposal_local = [m for m in LOCAL_MARKERS if m in case.agent_proposal]
    if ctx_remote and proposal_local:
        warnings.append(
            f"context names remote machinery ({', '.join(ctx_remote)}) while the proposal says "
            f"local/offline ({', '.join(proposal_local)}) — same mismatch risk"
        )

    if any(v in case.agent_proposal for v in VAGUE_PROPOSAL):
        warnings.append(
            "proposal looks vague (no concrete mechanism named); the model may answer D for "
            "lack of anything to judge"
        )
    if not case.agent_proposal.strip().endswith(("?", "？", "吗", "?")):
        warnings.append("proposal does not ask for a decision (no question form)")

    return errors, warnings


def cross_case_checks(cases: Sequence[Case]) -> List[str]:
    """Similarity warnings across candidates: near-duplicate cases waste budget."""
    warnings: List[str] = []
    for left, right in itertools.combinations(cases, 2):
        if left.domain and left.domain == right.domain:
            warnings.append(f"{left.id} and {right.id} share domain {left.domain!r}")
        # cheap containment check on the whole case text
        left_blob = f"{left.current_context}{left.agent_proposal}"
        right_blob = f"{right.current_context}{right.agent_proposal}"
        if len(left_blob) > 12 and left_blob == right_blob:
            warnings.append(f"{left.id} and {right.id} have identical context+proposal")
    return warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="static validation of candidate cases")
    parser.add_argument("--cases", default="cases_candidates_01.json")
    parser.add_argument("--check", action="store_true", help="exit 1 if any case has an error")
    args = parser.parse_args(argv)

    path = Path(args.cases)
    try:
        cases = load_cases(path)
    except CaseFileError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print("=" * 100)
    print(f"STATIC CASE VALIDATION — {path.name} ({len(cases)} candidates)")
    print("=" * 100)

    n_errors = n_warnings = 0
    for case in cases:
        errors, warnings = check_case(case)
        conflict, support, neutral = requirement_lengths(case)
        status = "ERROR" if errors else ("WARN " if warnings else "OK   ")
        print(f"\n[{status}] {case.id}  (domain={case.domain or '-'})")
        print(f"        ctx={len(case.current_context)} proposal={len(case.agent_proposal)} "
              f"conflict={conflict} support={support} neutral={neutral} gap={abs(conflict - support)}")
        for message in errors:
            print(f"        ERROR: {message}")
        for message in warnings:
            print(f"        warn : {message}")
        n_errors += len(errors)
        n_warnings += len(warnings)

    print()
    print("=" * 100)
    print(f"{n_errors} error(s), {n_warnings} warning(s) across {len(cases)} candidates")
    print("=" * 100)
    print(
        "\nWhat this script cannot check (that is Stage 1 + Stage 2):\n"
        "  * whether the proposal is 'reasonable but not obviously right' at step t\n"
        "  * whether conflict/support actually move decisions (needs P(A|blind) and dGate)\n"
        "  * whether a rejection reason comes from the intended trade-off (needs reading `reason`)\n"
        "\nNext: run Stage 1 calibration (never pretend_blind):\n"
        f"  python run_experiment.py --preset calibration --cases {path.name} --n 10 --concurrency 4\n"
        "  python calibrate.py --run-dir results/<that run>"
    )
    if args.check and n_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
