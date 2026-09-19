"""Check that DESIGN.md's factual claims match the implementation.

This is a documentation-consistency test, not part of the experiment.  It exists
because the design document quotes exact prompt text, character counts, hashes and
condition tables that must not drift away from `prompts.py` / `cases.json`.

Usage (from inside experiment/):
    python verify_doc.py     # exit 1 if any claim in DESIGN.md is stale
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import prompts as P
from common import load_cases, rate_stats

DOC = Path("DESIGN.md")
HERE_DOC = Path(__file__).resolve().parent

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))


def _design_sections(doc: str) -> dict[str, str]:
    """{condition name -> its row in the envelope-length table} plus the table itself."""
    table = ""
    for block in re.findall(r"\| condition \|.*?\n\n", doc, re.DOTALL):
        table += block
    return {"table": table}


CONTROL_CONDITIONS = (
    "pretend_blind_noreq",
    "pretend_blind_neutral",
)


def main() -> int:
    doc = DOC.read_text(encoding="utf-8")
    cases = load_cases(Path("cases.json"))
    case = cases[0]

    # --- Appendix A.1: the system prompt must appear verbatim ---------------
    check("A.1 system prompt appears verbatim", P.SYSTEM_PROMPT in doc)

    # --- Appendix A.2: blind message body must match the generator ---------
    a2 = doc.split("### A.2")[1].split("```")[1]
    doc_body = a2.split("<information_access>")[0].strip()
    generated = P.build_messages(case, P.BLIND)[1].content.strip()
    check("A.2 blind message matches the generated prompt", doc_body == generated)
    check("A.2 blind message has no envelope", "<information_access>" not in generated)

    # --- numbers and hashes quoted in the document -------------------------
    check("system prompt is 1136 chars", len(P.SYSTEM_PROMPT) == 1136, str(len(P.SYSTEM_PROMPT)))
    check(
        "system prompt sha256 prefix 7f66fd21f0e73b8b",
        hashlib.sha256(P.SYSTEM_PROMPT.encode()).hexdigest().startswith("7f66fd21f0e73b8b"),
    )
    check(
        "blind prompt sha256 quoted in the doc",
        P.prompt_sha256(P.build_messages(case, P.BLIND))[:16]
        in doc.replace("`", ""),
        P.prompt_sha256(P.build_messages(case, P.BLIND))[:16],
    )
    check(
        "pretend_blind_conflict prompt sha256 quoted in the doc",
        P.prompt_sha256(P.build_messages(case, P.PRETEND_BLIND_CONFLICT))[:16] in doc.replace("`", ""),
        P.prompt_sha256(P.build_messages(case, P.PRETEND_BLIND_CONFLICT))[:16],
    )

    # --- condition grid -----------------------------------------------------
    check("7 conditions", len(P.ALL_CONDITIONS) == 7, str(len(P.ALL_CONDITIONS)))
    check("3 cases", len(cases) == 3, str(len(cases)))
    gaps = [
        abs(len(c.future_requirement_conflict) - len(c.future_requirement_support)) for c in cases
    ]
    check("conflict/support requirement length gap <= 2 in every case", max(gaps) <= 2, str(gaps))
    for condition in P.ALL_CONDITIONS:
        check(f"condition {condition} named in the doc", condition in doc)

    # --- the doc must carry the review fixes -------------------------------
    check("doc records that blind has no envelope", "no envelope" in doc)
    check("doc makes P(A) the primary endpoint", "P(A)" in doc and "primary endpoint" in doc)
    check("doc names the paired headline contrast", "pretend_blind_support" in doc and "pretend_blind_conflict" in doc)
    check("doc states the denominator policy", "denominator" in doc.lower())

    # --- structural invariants the doc asserts -----------------------------
    enveloped = [c for c in P.ALL_CONDITIONS if c != P.BLIND]
    outside = {
        c: re.sub(r"<information_access>.*?</information_access>", "<information_access/>",
                  P.build_messages(case, c)[1].content, flags=re.DOTALL)
        for c in P.ALL_CONDITIONS
    }
    check(
        "doc claim: enveloped conditions share identical outside-envelope text",
        len({outside[c] for c in enveloped}) == 1,
    )
    blind_text = P.build_messages(case, P.BLIND)[1].content
    check("blind message equals the shared outside-envelope text", blind_text == outside[P.BLIND])
    check("blind has no envelope", "<information_access>" not in blind_text)

    # --- numbers quoted in the document -------------------------------------
    envelope_lengths = {c: len(P.information_block(case, c)) for c in P.ALL_CONDITIONS}
    expected_lengths = {
        P.BLIND: 0,
        P.PRETEND_BLIND_CONFLICT: 358,
        P.PRETEND_BLIND_SUPPORT: 360,
        P.FULL_INFO_CONFLICT: 202,
        P.FULL_INFO_SUPPORT: 204,
        P.PRETEND_BLIND_NOREQ: 310,
        P.PRETEND_BLIND_NEUTRAL: 356,
    }
    for condition, expected in expected_lengths.items():
        check(
            f"envelope length {condition} == {expected}",
            envelope_lengths[condition] == expected,
            str(envelope_lengths[condition]),
        )
    check(
        "blind user message is 591 chars",
        len(blind_text) == 591,
        str(len(blind_text)),
    )

    # --- rate_stats bookkeeping (the shared helper both reports rely on) ----
    stats = rate_stats(["A", "A", "B", "UNPARSED", "ERROR", "C"])
    check("rate_stats: n counts every sample", stats.n == 6, str(stats.n))
    check("rate_stats: decision denominator excludes non-decisions", stats.n_valid == 4, str(stats.n_valid))
    check("rate_stats: P(A) uses the decision denominator", abs(stats.p_a - 0.5) < 1e-9, str(stats.p_a))
    check("rate_stats: P(A+B) uses the decision denominator", abs(stats.p_ab - 0.75) < 1e-9, str(stats.p_ab))
    check(
        "rate_stats: failure rates use all samples",
        abs(stats.p_unparsed_of_all - 1 / 6) < 1e-9 and abs(stats.p_error_of_all - 1 / 6) < 1e-9,
    )

    # --- analyzer's two proportion families (the v0.2 display bug) ----------
    import pandas as pd

    import analyze

    demo = pd.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(20)],
            "case_id": ["c"] * 20,
            "condition": ["x"] * 20,
            "parsed_decision": ["A"] * 10 + ["B"] * 5 + ["UNPARSED"] * 3 + ["ERROR"] * 2,
        }
    )
    panel = analyze.table(demo, ["condition"])
    row = panel.iloc[0]
    check(
        "analyzer: decision proportion uses the decision denominator",
        abs(float(row["A_p"]) - 10 / 15) < 5e-5,  # table() rounds to 4 decimals
        str(row["A_p"]),
    )
    check(
        "analyzer: failure proportion uses ALL samples as denominator",
        abs(float(row["UNPARSED_p_all"]) - 3 / 20) < 5e-5 and abs(float(row["ERROR_p_all"]) - 2 / 20) < 5e-5,
        f"{row['UNPARSED_p_all']} / {row['ERROR_p_all']}",
    )
    check(
        "analyzer: the two families would differ if the denominators were swapped",
        abs(float(row["A_p"]) - float(row["UNPARSED_p_all"])) > 0.4,
    )
    check(
        "analyzer: no failure column is expressed over the decision denominator",
        not any(c.endswith("_p") and c.split("_p")[0] in ("UNPARSED", "ERROR") for c in panel.columns),
        str(list(panel.columns)),
    )
    check(
        "analyzer: exactly one headline comparison",
        sum(1 for _, _, headline, _ in analyze.COMPARISONS if headline) == 1,
        str([c for c in analyze.COMPARISONS if c[2]]),
    )
    check(
        "analyzer: the headline comparison is the paired conflict->support contrast",
        analyze.COMPARISONS[0][0] == "pretend_blind_conflict"
        and analyze.COMPARISONS[0][1] == "pretend_blind_support"
        and analyze.COMPARISONS[0][2] is True,
    )

    # --- the frozen-schema contract the design relies on --------------------
    from client import ClientSchema

    probe = ClientSchema(mode="frozen", probed=True, dropped_params=("thinking",))
    probe.set_frozen_template({"model": "m", "temperature": 1.0, "max_tokens": 8, "stream": False})
    check("client: frozen template accepted when it matches dropped_params", probe.frozen_template is not None)
    bad = ClientSchema(mode="frozen", probed=True, dropped_params=("thinking",))
    try:
        bad.set_frozen_template({"model": "m", "thinking": {"type": "disabled"}})
        check("client: a template re-adding a dropped parameter is rejected", False)
    except ValueError:
        check("client: a template re-adding a dropped parameter is rejected", True)

    # --- the case loader refuses the ambiguous a/b schema -------------------
    from common import CaseFileError

    legacy = HERE_DOC / "_verify_doc_legacy.json"
    try:
        legacy.write_text(
            json.dumps(
                [
                    {
                        "id": "x",
                        "current_context": "c",
                        "agent_proposal": "p",
                        "future_requirement_a": "a",
                        "future_requirement_b": "b",
                    }
                ]
            ),
            encoding="utf-8",
        )
        try:
            load_cases(legacy)
            check("case loader rejects a/b-style fields", False)
        except CaseFileError:
            check("case loader rejects a/b-style fields", True)
    finally:
        if legacy.exists():
            legacy.unlink()

    for name, ok, detail in CHECKS:
        print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail and not ok else ""))
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} claims in DESIGN.md verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
