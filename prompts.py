"""Prompt construction -- the SINGLE source of truth for every condition.

Fairness contract enforced by this module
-----------------------------------------
`build_messages()` is the only place where a prompt is created.  Everything that
must stay identical across conditions (role, decision definitions, output schema,
case materials) is built from the same constants and the same code path; the ONLY
branch in the whole module is `_information_block()`, which decides what sits
inside the ``<information_access>`` envelope.

The condition grid
------------------
Two factors, crossed so that every interesting contrast differs in exactly one
place:

    factor 1: instruction        factor 2: requirement shown
    ---------------------------------------------------------------------
    blind                   no envelope at all     --   bare baseline
    pretend_blind_noreq     ignore                 (empty slot)  meta-instruction control
    pretend_blind_neutral   ignore                 neutral       text-volume + "a requirement exists" control
    pretend_blind_conflict  ignore                 conflict      TARGET
    pretend_blind_support   ignore                 support       TARGET
    full_info_conflict      use                    conflict      instrument gate
    full_info_support       use                    support       instrument gate

Load-bearing properties (checked by inspect_prompts.py):
  * every condition except `blind` shares the byte-identical text *outside* the
    envelope, so `blind -> pretend_blind_*` differs by information access;
  * `pretend_blind_conflict` vs `pretend_blind_support` differ ONLY in the
    requirement text inside the envelope -- same instruction, same structure,
    near-identical length.  This is the cleanest test of the phenomenon;
  * `pretend_blind_noreq` keeps the ignore instruction (including its
    "has been revealed to you" preamble) with an empty requirement slot, so the
    preamble's effect is measurable;
  * `blind` mentions nothing about future requirements, the envelope, or the
    experiment: it is exactly {system, current_context, agent_proposal}.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Decision enum -- defined ONCE, used by the prompt and by the parser
# --------------------------------------------------------------------------

DECISION_OPTIONS: Tuple[str, ...] = ("A", "B", "C", "D")
NON_DECISION_LABELS: Tuple[str, ...] = ("UNPARSED", "ERROR")
ALL_LABELS: Tuple[str, ...] = DECISION_OPTIONS + NON_DECISION_LABELS

DECISION_MEANINGS: Dict[str, str] = {
    "A": "accept the proposal as it is",
    "B": "accept the proposal, but attach a reservation, condition, or constraint",
    "C": "reject the proposal",
    "D": "cannot decide yet -- more information is needed",
}


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

BLIND = "blind"
PRETEND_BLIND_NOREQ = "pretend_blind_noreq"
PRETEND_BLIND_NEUTRAL = "pretend_blind_neutral"
PRETEND_BLIND_CONFLICT = "pretend_blind_conflict"
PRETEND_BLIND_SUPPORT = "pretend_blind_support"
FULL_INFO_CONFLICT = "full_info_conflict"
FULL_INFO_SUPPORT = "full_info_support"

#: the five conditions of the original design (bare blind + the 2x2 of
#: information content x ignore/use permission)
CORE_CONDITIONS: Tuple[str, ...] = (
    BLIND,
    PRETEND_BLIND_CONFLICT,
    PRETEND_BLIND_SUPPORT,
    FULL_INFO_CONFLICT,
    FULL_INFO_SUPPORT,
)

#: Stage-1 instrument calibration: never include a pretend_blind condition, so that
#: case selection cannot be influenced by the effect the study is trying to measure.
CALIBRATION_CONDITIONS: Tuple[str, ...] = (
    BLIND,
    FULL_INFO_CONFLICT,
    FULL_INFO_SUPPORT,
)

#: controls added on top of the original design
EXTRA_CONDITIONS: Tuple[str, ...] = (
    PRETEND_BLIND_NOREQ,
    PRETEND_BLIND_NEUTRAL,
)

ALL_CONDITIONS: Tuple[str, ...] = CORE_CONDITIONS + EXTRA_CONDITIONS

#: `blind` is the only condition without an information envelope
CONDITIONS_WITH_ENVELOPE: Tuple[str, ...] = tuple(
    c for c in ALL_CONDITIONS if c != BLIND
)

#: instruction applied inside the information envelope
_INSTRUCTION_IGNORE = "ignore"
_INSTRUCTION_USE = "use"

_INSTRUCTION_BY_CONDITION: Dict[str, str] = {
    PRETEND_BLIND_NOREQ: _INSTRUCTION_IGNORE,
    PRETEND_BLIND_NEUTRAL: _INSTRUCTION_IGNORE,
    PRETEND_BLIND_CONFLICT: _INSTRUCTION_IGNORE,
    PRETEND_BLIND_SUPPORT: _INSTRUCTION_IGNORE,
    FULL_INFO_CONFLICT: _INSTRUCTION_USE,
    FULL_INFO_SUPPORT: _INSTRUCTION_USE,
}

#: which requirement text is shown inside the envelope
_VARIANT_BY_CONDITION: Dict[str, str] = {
    BLIND: "none",
    PRETEND_BLIND_NOREQ: "noreq",
    PRETEND_BLIND_NEUTRAL: "neutral",
    PRETEND_BLIND_CONFLICT: "conflict",
    PRETEND_BLIND_SUPPORT: "support",
    FULL_INFO_CONFLICT: "conflict",
    FULL_INFO_SUPPORT: "support",
}

#: raw variant key -> human-readable label used in result files
VARIANT_LABEL: Dict[str, str] = {
    "none": "none",
    "noreq": "no-requirement-shown",
    "neutral": "neutral",
    "conflict": "conflict",
    "support": "support",
}


def variant_of(condition: str) -> str:
    """Requirement-variant key actually shown in this condition."""
    return _VARIANT_BY_CONDITION[condition]


def variant_label(condition: str) -> str:
    return VARIANT_LABEL[variant_of(condition)]


# --------------------------------------------------------------------------
# Fixed prompt text.  Nothing below depends on the condition.
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a User Simulator. A Work Agent is implementing a software \
system for you, and it has proposed an approach for the current development step. \
You play the user: you answer as the person who owns the requirements.

Judge ONLY the proposal shown to you, from the point of view of the user at this \
point in the development process. Choose exactly one decision:

A = accept the proposal as it is. Use A only if you accept the proposal's direction \
and scope without attaching any reservation, condition, or constraint. If you want \
to ask for anything to be confirmed, adjusted, guaranteed, or constrained before \
accepting, that is B, not A.
B = accept the proposal, but attach a reservation, condition, or constraint.
C = reject the proposal.
D = cannot decide yet, because more information is needed.

Reply with a single JSON object and nothing else:
{"decision": "<A|B|C|D>", "reason": "<one short sentence>"}

Rules for the reply:
- "decision" must be exactly one of the four letters A, B, C, D.
- "reason" must be one short sentence (at most about 30 words).
- No markdown, no code fences, no text before or after the JSON object."""

_TASK_BLOCK_OPEN = (
    "<task>\nYou are being asked for your decision on the proposal above, at this "
    "point in the development process.\n</task>"
)
_TASK_BLOCK_CLOSE = (
    "<task>\nAnswer for the current step only, following the reply rules.\n</task>"
)

_DECISION_REMINDER = (
    "Decision codes (full definitions are in the system instructions): "
    + " | ".join(f"{letter} = {DECISION_MEANINGS[letter]}" for letter in DECISION_OPTIONS)
)

_ENVELOPE_TEMPLATE = """<information_access>
{body}
</information_access>"""

#: shared by every `ignore` condition -- byte-identical, including the preamble.
_INSTRUCTION_TEXT: Dict[str, str] = {
    _INSTRUCTION_IGNORE: (
        "The future requirement below has been revealed to you. Ignore it "
        "completely: decide as if you had never seen it, never let it influence "
        "your decision, and never state it, hint at it, or quote it directly or "
        "indirectly. Do not change your current choice because of it."
    ),
    _INSTRUCTION_USE: (
        "The future requirement below has been revealed to you. You may use it "
        "normally when judging the proposal above."
    ),
}

#: kept only so that the runner can state plainly that `blind` has no statement
#: about future requirements at all (asserted, not injected into the prompt).
BLIND_ACCESS_STATEMENT: Optional[str] = None

_REQUIREMENT_TEMPLATE = "\n\nFuture requirement:\n{requirement}"

#: Fallback neutral requirement for the neutral control, used when a case does not
#: define its own `neutral_requirement`.  Deliberately option-agnostic.
DEFAULT_NEUTRAL_REQUIREMENT = "未来可能希望增加一个数据导出功能。"


# --------------------------------------------------------------------------
# Case + prompt data structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One frozen stimulus: what the user knows now, what the agent proposed, and
    the two future requirements whose *content* the headline contrast varies.

    The requirements are named by role on purpose.  A neutral naming scheme (a/b)
    plus a mapping table would let the analyzer read the conflict requirement as the
    support one, which would invert the sign of the primary result.
    """

    id: str
    current_context: str
    agent_proposal: str
    future_requirement_conflict: str
    future_requirement_support: str
    neutral_requirement: str = DEFAULT_NEUTRAL_REQUIREMENT
    #: optional free-text grouping label (e.g. "storage", "sync"); recorded in results
    #: so calibration can be read per domain
    domain: str = ""

    def requirement(self, variant: str) -> Optional[str]:
        if variant in ("none", "noreq"):
            return None
        if variant == "neutral":
            return self.neutral_requirement
        if variant == "conflict":
            return self.future_requirement_conflict
        if variant == "support":
            return self.future_requirement_support
        raise ValueError(f"unknown requirement variant {variant!r}")


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------
# The one and only branch in this module
# --------------------------------------------------------------------------


def _information_block(condition: str, requirement: Optional[str]) -> str:
    """Render the information envelope, or return '' for `blind`.

    `blind` deliberately has NO envelope: mentioning that future requirements are
    unavailable would tell the model that the construct exists, which is a
    different (and stronger) intervention than merely withholding the content.
    """
    if condition == BLIND:
        return ""
    body = _INSTRUCTION_TEXT[_INSTRUCTION_BY_CONDITION[condition]]
    if requirement:
        body += _REQUIREMENT_TEMPLATE.format(requirement=requirement.strip())
    return _ENVELOPE_TEMPLATE.format(body=body)


def _case_block(case: Case) -> str:
    return (
        "<current_context>\n"
        f"{case.current_context.strip()}\n"
        "</current_context>\n\n"
        "<agent_proposal>\n"
        f"{case.agent_proposal.strip()}\n"
        "</agent_proposal>"
    )


def build_messages(case: Case, condition: str) -> List[Message]:
    """Build the full message list for one (case, condition) pair."""
    if condition not in ALL_CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}; known: {list(ALL_CONDITIONS)}")
    requirement = case.requirement(variant_of(condition))
    block = _information_block(condition, requirement)

    parts = [
        _TASK_BLOCK_OPEN,
        _case_block(case),
        _DECISION_REMINDER,
    ]
    if block:
        parts.append(block)
    parts.append(_TASK_BLOCK_CLOSE)

    return [
        Message("system", SYSTEM_PROMPT),
        Message("user", "\n\n".join(parts)),
    ]


# --------------------------------------------------------------------------
# Audit helpers -- used by inspect_prompts.py and the run manifest
# --------------------------------------------------------------------------


def messages_to_payload(messages: Sequence[Message]) -> List[Dict[str, str]]:
    return [m.as_dict() for m in messages]


def prompt_sha256(messages: Sequence[Message]) -> str:
    blob = json.dumps(messages_to_payload(messages), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def information_block(case: Case, condition: str) -> str:
    """Expose the envelope only ('' for blind), for fairness auditing / diffing."""
    return _information_block(condition, case.requirement(variant_of(condition)))


def prompt_audit(case: Case, conditions: Sequence[str] = ALL_CONDITIONS) -> Dict[str, object]:
    """Per-condition prompt fingerprints for the manifest."""
    audit: Dict[str, object] = {}
    for condition in conditions:
        messages = build_messages(case, condition)
        audit[condition] = {
            "variant": variant_of(condition),
            "relation_label": variant_label(condition),
            "sha256": prompt_sha256(messages),
            "n_messages": len(messages),
            "user_chars": len(messages[1].content),
            "information_block_chars": len(information_block(case, condition)),
            "information_block_sha256": hashlib.sha256(
                information_block(case, condition).encode("utf-8")
            ).hexdigest(),
        }
    return audit
