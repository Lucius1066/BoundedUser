"""Raw-response parser.  Deliberately dumb: no semantic judging, no defaults.

Rules:
  * find a JSON object that contains a "decision" key (fenced code blocks first,
    then a brace-balanced scan), taking the LAST one as the model's answer;
  * normalize the decision to an uppercase letter and validate it against the
    enum from prompts.py;
  * if anything fails, return decision="UNPARSED" and keep the raw text -- it is
    reported in the statistics instead of being silently defaulted or dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prompts import DECISION_OPTIONS

UNPARSED = "UNPARSED"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParsedResponse:
    decision: str
    reason: str = ""
    valid_json: bool = False
    extracted_from_text: bool = False
    note: str = ""
    decision_raw: str = ""


def _iter_json_objects(text: str) -> List[str]:
    """Yield brace-balanced {...} substrings, innermost-last order preserved."""
    spans: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : index + 1])
                    start = -1
    return spans


def _candidate_blobs(text: str) -> List[str]:
    fenced = _FENCE_RE.findall(text or "")
    return fenced + _iter_json_objects(text or "")


def _pick_decision(obj: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    if "decision" not in obj:
        return None
    raw = obj.get("decision")
    reason = obj.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = json.dumps(reason, ensure_ascii=False)
    if isinstance(raw, str):
        raw_text = raw.strip()
    else:
        raw_text = json.dumps(raw, ensure_ascii=False)
    return raw_text, reason.strip(), raw_text


def parse_response(text: str) -> ParsedResponse:
    """Parse one raw model reply.  Never raises."""
    if not (text or "").strip():
        return ParsedResponse(decision=UNPARSED, note="empty content")

    blobs = _candidate_blobs(text)
    json_ok = False
    for blob in reversed(blobs):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        picked = _pick_decision(obj)
        if picked is None:
            continue
        raw_decision, reason, raw_text = picked
        json_ok = True
        normalized = raw_decision.strip().upper()
        # Tolerate "A." / "A)" / "A - accept ..." style answers.
        match = re.match(r"^([ABCD])\b", normalized)
        if match:
            normalized = match.group(1)
        if normalized in DECISION_OPTIONS:
            return ParsedResponse(
                decision=normalized,
                reason=reason[:2000],
                valid_json=True,
                extracted_from_text=blob not in (text or "").strip(),
                decision_raw=raw_text,
            )
        return ParsedResponse(
            decision=UNPARSED,
            reason=reason[:2000],
            valid_json=True,
            extracted_from_text=blob not in (text or "").strip(),
            note=f"decision value not in enum: {raw_text[:80]!r}",
            decision_raw=raw_text,
        )

    return ParsedResponse(
        decision=UNPARSED,
        valid_json=json_ok,
        note="no JSON object with a 'decision' key found",
    )
