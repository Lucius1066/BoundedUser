"""Fairness audit: show exactly how the prompts differ between conditions.

Run this before spending any API budget.  It answers four questions:

  1. Is every condition except `blind` byte-identical *outside* the
     <information_access> envelope?
  2. Is `blind` really bare -- no envelope, and no mention of future
     requirements, the envelope, the instruction, or the experiment at all?
  3. Do `pretend_blind_conflict` and `pretend_blind_support` differ ONLY in the
     requirement text (same instruction, near-identical length)?
  4. Does any condition show the wrong requirement, and does `blind` leak any
     requirement text?

Usage:
    python inspect_prompts.py                       # all cases, all conditions
    python inspect_prompts.py --case database_01 --full
    python inspect_prompts.py --check               # exit 1 if the audit fails
    python inspect_prompts.py --diff pretend_blind_conflict:pretend_blind_support
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from common import CaseFileError, load_cases
from prompts import (
    ALL_CONDITIONS,
    BLIND,
    FULL_INFO_CONFLICT,
    FULL_INFO_SUPPORT,
    PRETEND_BLIND_CONFLICT,
    PRETEND_BLIND_NEUTRAL,
    PRETEND_BLIND_NOREQ,
    PRETEND_BLIND_SUPPORT,
    SYSTEM_PROMPT,
    Case,
    build_messages,
    information_block,
    variant_label,
)
from prompts import _ENVELOPE_TEMPLATE, _INSTRUCTION_BY_CONDITION, _INSTRUCTION_TEXT

HERE = Path(__file__).resolve().parent
ENVELOPE_RE = re.compile(r"<information_access>.*?</information_access>", re.DOTALL)

#: `blind` must contain none of these.  Deliberately limited to tokens that cannot
#: appear in the shared scaffolding by accident: "condition" would match the
#: current_context prose, and "blind"/"ignore" are too generic to be safe.
_BLIND_FORBIDDEN = (
    "information_access",
    "future requirement",
    "Future requirement",
    "has been revealed to you",
    "Ignore it completely",
    "you may use it",
    "blind condition",
    "experiment",
    "未来",
)

_HEADLINE_PAIR = (PRETEND_BLIND_CONFLICT, PRETEND_BLIND_SUPPORT)
_IGNORE_FAMILY = (
    PRETEND_BLIND_NOREQ,
    PRETEND_BLIND_NEUTRAL,
    PRETEND_BLIND_CONFLICT,
    PRETEND_BLIND_SUPPORT,
)


def _user_message(case: Case, condition: str) -> str:
    return build_messages(case, condition)[1].content


def _outside_envelope(case: Case, condition: str) -> str:
    return ENVELOPE_RE.sub("<information_access/>", _user_message(case, condition))


def _block(case: Case, condition: str) -> str:
    return information_block(case, condition)


def _instruction_part(case: Case, condition: str) -> str:
    """The instruction portion of the envelope, derived from the shared constants.

    Deliberately not `block.split("Future requirement:")`: a case whose requirement
    text contained that marker would defeat the check.
    """
    if condition == BLIND:
        return ""
    body = _INSTRUCTION_TEXT[_INSTRUCTION_BY_CONDITION[condition]]
    return _ENVELOPE_TEMPLATE.format(body=body)


def audit(cases: Sequence[Case], conditions: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (problems, notes).  Empty problems means the audit passed."""
    problems: List[str] = []
    notes: List[str] = []

    for case in cases:
        enveloped = [c for c in conditions if c != BLIND]

        # 1. everything outside the envelope is byte-identical
        if len(enveloped) > 1:
            reference = _outside_envelope(case, enveloped[0])
            for condition in enveloped[1:]:
                if _outside_envelope(case, condition) != reference:
                    problems.append(
                        f"{case.id}: prompt outside <information_access> differs between "
                        f"{enveloped[0]} and {condition}"
                    )

        # 2. blind is bare
        if BLIND in conditions:
            blind_text = _user_message(case, BLIND)
            if "<information_access>" in blind_text:
                problems.append(f"{case.id}/{BLIND}: still contains an <information_access> envelope")
            for token in _BLIND_FORBIDDEN:
                if token in blind_text:
                    problems.append(f"{case.id}/{BLIND}: contains forbidden token {token!r}")
            for label, text in (
                ("future_requirement_conflict", case.future_requirement_conflict),
                ("future_requirement_support", case.future_requirement_support),
                ("neutral_requirement", case.neutral_requirement),
            ):
                if text.strip() and text.strip() in blind_text:
                    problems.append(f"{case.id}/{BLIND}: {label} text appears in the blind prompt")

        # 3. system prompt + message shape
        for condition in conditions:
            messages = build_messages(case, condition)
            if messages[0].content != SYSTEM_PROMPT:
                problems.append(f"{case.id}/{condition}: system prompt is not the shared constant")
            if len(messages) != 2 or [m.role for m in messages] != ["system", "user"]:
                problems.append(f"{case.id}/{condition}: message shape differs from the shared shape")
            extra = set(messages[1].as_dict()) - {"role", "content"}
            if extra:
                problems.append(f"{case.id}/{condition}: user message carries extra fields {sorted(extra)}")

        # 4. the headline pair differs only inside the requirement slot
        if all(c in conditions for c in _HEADLINE_PAIR):
            left, right = _HEADLINE_PAIR
            if _outside_envelope(case, left) != _outside_envelope(case, right):
                problems.append(f"{case.id}: {left} and {right} differ outside the envelope")
            instruction_left = _instruction_part(case, left)
            instruction_right = _instruction_part(case, right)
            if instruction_left != instruction_right:
                problems.append(f"{case.id}: {left} and {right} do not share a byte-identical instruction")

        # 5. the ignore family shares one instruction string; noreq has an empty slot
        if all(c in conditions for c in _IGNORE_FAMILY):
            instructions = {c: _instruction_part(case, c) for c in _IGNORE_FAMILY}
            if len(set(instructions.values())) != 1:
                problems.append(
                    f"{case.id}: the ignore instruction is not identical across {list(_IGNORE_FAMILY)}"
                )
            if "Future requirement:\n" in _block(case, PRETEND_BLIND_NOREQ):
                problems.append(f"{case.id}/{PRETEND_BLIND_NOREQ}: unexpected requirement slot content")
            for condition in (PRETEND_BLIND_CONFLICT, PRETEND_BLIND_SUPPORT, PRETEND_BLIND_NEUTRAL):
                if "Future requirement:\n" not in _block(case, condition):
                    problems.append(f"{case.id}/{condition}: requirement slot is empty")
            notes.append(
                f"{case.id}/{PRETEND_BLIND_NOREQ}: keeps the shared preamble "
                "('The future requirement below has been revealed to you.') with an empty "
                "requirement slot -- intentional, see DESIGN.md 4.2"
            )

        # 6. no cross-variant leakage
        for condition in conditions:
            role = variant_label(condition)
            block = _block(case, condition)
            if role != "conflict" and case.future_requirement_conflict.strip() in block:
                problems.append(f"{case.id}/{condition}: shows the conflict requirement but its role is {role}")
            if role != "support" and case.future_requirement_support.strip() in block:
                problems.append(f"{case.id}/{condition}: shows the support requirement but its role is {role}")
    return problems, notes


def length_table(cases: Sequence[Case], conditions: Sequence[str]) -> None:
    print("=" * 100)
    print("ENVELOPE LENGTHS (chars; '-' = no envelope at all)")
    print("=" * 100)

    def short(condition: str) -> str:
        return condition.replace("pretend_blind", "pb").replace("full_info", "fi")

    print(f"{'case':<20}" + "".join(f"{short(c):>10}" for c in conditions))
    for case in cases:
        row = f"{case.id:<20}"
        for condition in conditions:
            size = len(_block(case, condition))
            row += f"{(size if size else '-'):>10}"
        print(row)
    for case in cases:
        sizes = {c: len(_block(case, c)) for c in conditions}
        for left, right in (
            (PRETEND_BLIND_CONFLICT, PRETEND_BLIND_SUPPORT),
            (FULL_INFO_CONFLICT, FULL_INFO_SUPPORT),
        ):
            if left in sizes and right in sizes:
                gap = abs(sizes[left] - sizes[right])
                flag = "OK " if gap <= 12 else "WARN"
                print(f"  {flag} {case.id}: |{short(left)} - {short(right)}| = {gap} chars")
        if BLIND in sizes:
            print(f"  --  {case.id}: blind has no envelope by construction (this is intentional)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="audit prompt fairness across conditions")
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--case", default="", help="only this case id")
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--full", action="store_true", help="print whole user messages")
    parser.add_argument(
        "--diff",
        default="",
        help="diff two conditions, e.g. pretend_blind_conflict:pretend_blind_support",
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero if the audit fails")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(Path(args.cases))
    except CaseFileError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"[error] no such case id: {args.case}", file=sys.stderr)
            return 2
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    problems, notes = audit(cases, conditions)
    print("=" * 100)
    print("STRUCTURAL AUDIT")
    print("=" * 100)
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
    else:
        enveloped = [c for c in conditions if c != BLIND]
        print(f"  OK    outside <information_access>: byte-identical across {len(enveloped)} enveloped conditions")
        print("  OK    blind: no envelope, no mention of future requirements or the experiment")
        print(f"  OK    {PRETEND_BLIND_CONFLICT} vs {PRETEND_BLIND_SUPPORT}: identical outside the requirement text")
        print(f"  OK    one shared ignore instruction across {len([c for c in _IGNORE_FAMILY if c in conditions])} conditions")
        print("  OK    system prompt and message shape identical everywhere; no cross-variant leak")
    for note in notes[:2]:
        print(f"  NOTE  {note}")
    if len(notes) > 2:
        print(f"  NOTE  ... {len(notes) - 2} more notes of the same kind")

    print()
    length_table(cases, conditions)

    if args.diff:
        left, _, right = args.diff.partition(":")
        case = cases[0]
        print()
        print("=" * 100)
        print(f"ENVELOPE DIFF  {left}  ->  {right}   (case {case.id})")
        print("=" * 100)
        for line in difflib.unified_diff(
            _block(case, left).splitlines(),
            _block(case, right).splitlines(),
            fromfile=left,
            tofile=right,
            lineterm="",
            n=1,
        ):
            print(line)

    if args.full:
        for case in cases:
            for condition in conditions:
                print()
                print("=" * 100)
                print(f"FULL USER MESSAGE  case={case.id}  condition={condition}  role={variant_label(condition)}")
                print("=" * 100)
                print(_user_message(case, condition))

    if args.check and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
