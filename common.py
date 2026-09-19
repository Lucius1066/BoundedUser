"""Shared helpers used by the runner, the analyzer and the audit scripts.

Kept deliberately small.  Everything here is about *bookkeeping* (rates and
denominators, case-file loading), never about experiment logic -- that lives in
prompts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from prompts import ALL_LABELS, DECISION_OPTIONS, Case

UNPARSED = "UNPARSED"
ERROR = "ERROR"

#: labels that represent a real decision; UNPARSED/ERROR are tracked separately
VALID_DECISIONS: Tuple[str, ...] = DECISION_OPTIONS


# --------------------------------------------------------------------------
# Rates.  Denominators are explicit because mixing them up is the single easiest
# way to turn a formatting difference into a fake effect.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RateStats:
    n: int                # every sample, including ERROR/UNPARSED -- never dropped
    n_valid: int          # samples that produced a decision in {A,B,C,D}
    n_unparsed: int
    n_error: int
    counts: Mapping[str, int]

    @property
    def n_decision(self) -> int:
        """Denominator for decision rates: parsed decisions only."""
        return self.n_valid

    @property
    def p_a(self) -> float:
        """Primary endpoint: unconditional acceptance P(A | parsed decision)."""
        return self.counts.get("A", 0) / self.n_valid if self.n_valid else float("nan")

    @property
    def p_ab(self) -> float:
        """Secondary: acceptance with or without reservations."""
        if not self.n_valid:
            return float("nan")
        return (self.counts.get("A", 0) + self.counts.get("B", 0)) / self.n_valid

    def p(self, label: str) -> float:
        """Rate of one label among parsed decisions."""
        return self.counts.get(label, 0) / self.n_valid if self.n_valid else float("nan")

    def p_of_all(self, label: str) -> float:
        """Rate of one label among ALL samples -- the only correct denominator for
        ERROR/UNPARSED, whose counts are not part of the decision denominator."""
        return self.counts.get(label, 0) / self.n if self.n else float("nan")

    @property
    def p_unparsed_of_all(self) -> float:
        return self.p_of_all(UNPARSED)

    @property
    def p_error_of_all(self) -> float:
        return self.p_of_all(ERROR)

    def se_a(self) -> float:
        p = self.p_a
        if not self.n_valid or p != p:  # NaN check without importing math
            return float("nan")
        return (p * (1 - p) / self.n_valid) ** 0.5


def rate_stats(decisions: Iterable[str]) -> RateStats:
    counts: Dict[str, int] = {}
    n = n_valid = n_unparsed = n_error = 0
    for raw in decisions:
        label = str(raw)
        counts[label] = counts.get(label, 0) + 1
        n += 1
        if label in VALID_DECISIONS:
            n_valid += 1
        elif label == ERROR:
            n_error += 1
        else:
            n_unparsed += 1
    return RateStats(n=n, n_valid=n_valid, n_unparsed=n_unparsed, n_error=n_error, counts=counts)


def all_labels_present(decisions: Iterable[str]) -> List[str]:
    seen = {str(d) for d in decisions}
    return [label for label in ALL_LABELS if label in seen]


# --------------------------------------------------------------------------
# Case files
# --------------------------------------------------------------------------


class CaseFileError(ValueError):
    pass


_REQUIRED_FIELDS = (
    "id",
    "current_context",
    "agent_proposal",
    "future_requirement_conflict",
    "future_requirement_support",
)


def _read_case_file(path: Path) -> object:
    import json

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise CaseFileError(
                f"{path.name}: YAML needs PyYAML, which is not installed. "
                "Use JSON, or `pip install pyyaml`."
            ) from None
        return yaml.safe_load(text)
    raise CaseFileError(f"unsupported case file extension: {path.suffix!r}")


def load_cases(path: Path) -> List[Case]:
    """Load and validate the case file.  Fails loudly on any missing field.

    Requirements are addressed **by role only** (`future_requirement_conflict` /
    `future_requirement_support`).  There is deliberately no slot-based (a/b)
    alternative: with two naming systems in play, the analysis could silently read
    the conflict requirement as the support one, which would invert the headline
    contrast.  One schema, no ambiguity.
    """
    if not path.is_file():
        raise CaseFileError(f"case file not found: {path}")
    raw = _read_case_file(path)
    if isinstance(raw, dict) and "cases" in raw:
        raw = raw["cases"]
    if not isinstance(raw, list) or not raw:
        raise CaseFileError(f"{path.name}: expected a non-empty list of cases")

    cases: List[Case] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CaseFileError(f"{path.name}: case #{index} is not an object")
        missing = [key for key in _REQUIRED_FIELDS if not str(item.get(key, "")).strip()]
        if missing:
            extra_hint = ""
            if any(key in item for key in ("future_requirement_a", "future_requirement_b")):
                extra_hint = (
                    " (found future_requirement_a/b: rename them to "
                    "future_requirement_conflict / future_requirement_support)"
                )
            raise CaseFileError(
                f"{path.name}: case #{index} missing/empty fields: {missing}{extra_hint}"
            )
        case_id = str(item["id"]).strip()
        if case_id in seen:
            raise CaseFileError(f"{path.name}: duplicate case id {case_id!r}")
        seen.add(case_id)

        kwargs = {key: str(item[key]).strip() for key in _REQUIRED_FIELDS}
        if str(item.get("neutral_requirement", "")).strip():
            kwargs["neutral_requirement"] = str(item["neutral_requirement"]).strip()
        if str(item.get("domain", "")).strip():
            kwargs["domain"] = str(item["domain"]).strip()
        cases.append(Case(**kwargs))
    return cases
