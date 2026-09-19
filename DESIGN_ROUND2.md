# Round 2 — Staged design: case calibration, freezing, and the pre-registered test

**Status:** Stage 0 (candidates) done; Stage 1 (calibration) running; thresholds below are declared
**before** looking at any Stage-1 number. This file is the round-2 protocol; `DESIGN.md` remains the
harness/protocol documentation for the measurement machinery (v0.3).

## 0. What round 1 established, and what it failed to test

Round 1 (`results/v03_deepseek-flash_n20/`, backed up in `backups/v1_first_real_run_*/`) ran the full
7-condition design on 3 cases and found:

| finding | value | meaning |
|---|---|---|
| Instrument gate | Δ_FI = **+0.617** | requirement content strongly moves decisions when its use is permitted |
| Headline | Δ_PB = **−0.067**, cases_up 1 / cases_down 2 | no evidence of the predicted contamination |
| Compliance | 1 of 240 `pretend_blind` rows mentioned the requirement content | self-blinding was followed almost verbatim |
| Instruction effect | `blind → noreq` moved P(A+B) by +0.10 pooled, +0.30 in the only discriminating case | larger than the content effect the study was built to find |
| **Case quality** | `database_01` P(A|blind) = 1.00 (ceiling); `config_loading_02` P(A|blind) = 0.00 (floor) | **2 of 3 cases could not discriminate at all** |
| **Material defect** | `config_loading_02`'s proposal named `requests` while the context said "local directory" | rejections came from an unintended semantic conflict, not the intended trade-off |

So round 1's negative headline result is **uninformative rather than reassuring**: with only one case
having decision space, and that case's conflict arm showing *higher* acceptance than its support arm,
the null could be a property of the materials rather than of the model. Round 2 fixes the materials first
and only then re-asks the question.

## 1. Stage 0 — candidate cases (DONE)

`cases_candidates_01.json`: **15 candidates**, 9 domains. Each carries
`current_context`, `agent_proposal`, `future_requirement_conflict`, `future_requirement_support`,
`neutral_requirement`, and a free-text `domain` label for per-domain reading.

Authoring rules applied:

1. The proposal must be defensible at step *t* but **not obviously right** — the context should make it a
   judgment call rather than a foregone conclusion.
2. The conflict requirement must undermine *the same resource the proposal commits to* (a database that must
   now be multi-instance, a per-device config that must now be per-project, a client-side render that must
   now page a million rows), so the causal link is direct rather than thematic.
3. The support requirement must make the same commitment look durable, and must be **plausible in the same
   domain**, not a strawman.
4. No latent semantic conflict between `current_context` and `agent_proposal` (round 1's defect).
5. conflict/support matched on phrasing, length (±8 chars enforced, actual gaps 0–7) and constraint strength.
6. `neutral_requirement` stays option-irrelevant but domain-plausible.

**Static validation** — `python validate_cases.py --cases cases_candidates_01.json --check`:
structural completeness, conflict↔support length gap, requirement-restates-context detection,
vague-proposal detection, question-form check, and a **local/remote mismatch heuristic** that fires exactly
on round 1's defect class. Current state: 0 errors, 0 warnings across 15 candidates (the one mismatch
warning on `photos_04_wifi_only` was fixed by making the context state where photos are uploaded *to*).

This script cannot check "reasonable but not obviously right" or "the requirement actually moves the
decision" — those are Stage 1 and Stage 2.

## 2. Stage 1 — stimulus calibration (RUNNING)

`--preset calibration` = **`blind`, `full_info_conflict`, `full_info_support` only**. The runner *refuses*
`pretend_blind` under this preset: case selection must not be influenced by the effect the study measures.

Design: 15 candidates × 3 conditions × **n = 10** = **450 calls** (`--n 10` rather than the 20/case in the
brief, to keep the screening cheap; the surviving cases are re-measured at n = 30–50 in Stage 4 anyway).

### 2.1 Declared screening thresholds (fixed before looking)

| check | requirement | rationale |
|---|---|---|
| `baseline_band` | `0.20 < P(A \| blind) < 0.80` | the primary endpoint needs headroom in both directions |
| `baseline_not_floor_for_A` | not (`P(A\|blind) ≤ 0.05` **and** `P(B\|blind) ≥ 0.50`) | the round-1 pattern: `P(A+B)` has headroom while `P(A)` is pinned at the floor, so a leakage-driven A→B move is invisible to the primary endpoint |
| `not_ceiling` | `P(A\|blind) < 0.95` | an increase is arithmetically impossible above this |
| `gate_positive` | `Δ_gate > 0` | the requirement must move decisions when permitted |
| `gate_above_threshold` | `Δ_gate > 0.30` | empirical discriminating-power floor, **not** a significance test |
| `gate_in_both_arms` | `P(A\|FI_support) > P(A\|FI_conflict)` | direction must be the predicted one per case, not just pooled |

A case is **SELECTED** only if every check passes. Reported alongside, but not gates: full A/B/C/D per
condition, and *answer concentration* (share of the cell sharing one sentence) as a warning that extra
samples in that cell buy little.

All thresholds are CLI flags (`--lo`, `--hi`, `--gate`) and are written into
`calibration_selection.json`, so the screening rule used is auditable rather than remembered.

## 3. Stage 2 — manual review of the survivors

For each numeric survivor, read `reason` (via `postmortem.py`) and ask:

- Does the blind rejection/reservation come from the **intended** design trade-off, or from something else
  in the text (round 1's `requests`)?
- Does `full_info_conflict` change the decision **because of** the conflict requirement, or for an unrelated
  reason that happens to co-occur?
- Same for `full_info_support`.
- Is one of the two requirements obviously stronger, more specific, or more concrete than the other?
- Does the blind cell look like a coin flip over two defensible readings, or like a single deterministic
  reading?

Target: **5–10 frozen cases**. Anything rejected here is recorded with the reason in
`calibration_selection.json` (extended by hand with a `manual` field) so the final selection criteria are
documented, not implicit.

### 3.1 Outcome of the first batch (2026-09-19)

Stage 1 selected 5 of 15; Stage 2 kept **3, with a 4th conditional**. Full detail in
`results/calib01_candidates15_n10/CALIBRATION_MEMO.md`. Summary:

- kept: `apiusers_03_inmemory` (P(A|blind)=0.70), `chatbot_07_no_memory` (0.70), `report_09_client_render` (0.40)
- rejected at Stage 2: `iot_12_local_only`, `survey_15_local_storage` — in both, the blind cell's
  reservation is about an *adjacent* concern (backup/history retention; data aggregation) rather than the
  dimension the conflict requirement manipulates, so a `pretend_blind` shift could not be attributed
- two structural lessons recorded there: (i) `P(A|FI_support) ≥ 0.70` in **all 15** candidates, so the gate
  behaves as a near step function and the support arm has no headroom by construction; (ii) 10 of 15
  candidates still landed at ceiling/floor for `P(A|blind)` even though they were authored to sit mid-range,
  which motivates a second candidate batch built on an explicit principle: *the proposal must visibly
  sacrifice something the current user cares about*, so that "accept" is a real trade rather than the
  obvious answer.

## 4. Stage 3 — freeze

Frozen at this point, and **not** revisited after seeing any `pretend_blind` result:

- case texts (`cases_frozen.json`, byte-frozen, hashed);
- prompts (`prompts.py` unchanged from v0.3 — the envelopes are already fixed constants);
- primary endpoint **`P(A | parsed decision)`**;
- primary contrast **`Δ_PB = P(A | PB_support) − P(A | PB_conflict)`**;
- instrument gate **`Δ_FI = P(A | FI_support) − P(A | FI_conflict)`** (now a manipulation check, since
  Stage 1 already calibrated on it);
- the analysis order of Stage 5.

Freezing is recorded by `prompt_audit` hashes in the run manifest plus a checksum manifest for
`cases_frozen.json` (`make_backup_manifest.py`).

## 5. Stage 4 — the real experiment

All 7 conditions on the frozen cases: `blind`, `pretend_blind_noreq`, `pretend_blind_neutral`,
`pretend_blind_conflict`, `pretend_blind_support`, `full_info_conflict`, `full_info_support`.

- Hypothesis test: **`PB_conflict ↔ PB_support`** only.
- `blind`: the natural baseline (never exposed to future information).
- `noreq` / `neutral`: framing and meta-instruction diagnostics. Per the brief, `noreq` is a **diagnostic
  control, not a clean causal estimate of instruction effect** — its slot is present but empty, which is an
  odd prompt in its own right. (The byte-identical-instruction control is `neutral`.)
- Sampling: **n = 30–50 per cell**. With 6 cases × 7 conditions × 30 = 1,260 calls.

The stage-1/4 response format, schema freezing, retry and no-silent-drop guarantees are unchanged from
`DESIGN.md` § 5.

## 6. Stage 5 — pre-registered analysis order

Reported in this order; each step can stop the interpretation.

1. **Data quality** — ERROR rate, UNPARSED rate, single schema hash, per-condition sampling parameters. If
   conditions differ systematically in API/parse behaviour, report that first and interpret nothing else.
2. **Manipulation check** — `Δ_FI > 0` pooled **and per case**.
3. **Headline test** — per-case `Δ_i`, then mean/median `Δ`, counts of positive/negative/zero, and the
   pooled distribution. Pooled number never reported alone.
4. **Full decision distribution** — P(A), P(B), P(C), P(D), looking specifically for A→B hedging, A/B→D
   hesitation, and different reservation/rejection patterns between arms.
5. **Framing controls** — `blind→noreq`, `blind→neutral`, `noreq→neutral`, `neutral→conflict/support`.
6. **Qualitative inspection** — distinguishing *explicit leakage* (naming the requirement) from *implicit
   behavioral contamination* (a hedge consistent with the requirement without revealing it). Only the second
   is the target phenomenon; both are reported separately, and neither counts as evidence on its own.

## 7. Decision logic for round 2 (fixed in advance)

Let `G = Δ_FI` and `H = Δ_PB`.

- **If `G ≫ 0` and `H ≫ 0` with consistent sign across cases and larger than the `neutral`/`noreq`
  controls** → privileged future information systematically influences the current decision despite an
  explicit ignore instruction.
- **If `G ≫ 0` but `H ≈ 0`, with most cases having adequate decision space after calibration** → under this
  model, prompt and setup, no material privileged-information contamination was observed; prompt-based
  self-blinding suppressed large behavioral effects. (This is the outcome round 1 *hinted* at but could not
  support, because two of three cases had no decision space.)
- **If the framing controls are large while the `conflict`↔`support` content contrast is small** → the more
  interesting question becomes whether **being told to ignore something** systematically changes User
  Simulator behaviour at all. That is a different phenomenon and gets its own design; round 1's
  `blind→noreq` = +0.30 (P(A+B), discriminating case only) is already suggestive of it.
- **If `G ≤ 0` per case in Stage 1** → the case is discarded rather than run; a case where the requirement
  does not move decisions when permitted cannot show anything when it is forbidden.

## 8. Harness changes made for round 2

| change | file |
|---|---|
| `--preset calibration` (3 conditions, refuses to include `pretend_blind`) | `run_experiment.py`, `prompts.CALIBRATION_CONDITIONS` |
| `--candidate-cases` alias for `--cases` | `run_experiment.py` |
| stage-1 scorer with declared thresholds, per-case verdicts, concentration diagnostics | `calibrate.py` (new) |
| static pre-flight validation of candidate cases | `validate_cases.py` (new) |
| backup checksum manifests | `make_backup_manifest.py` (new) |
| `domain` field carried through to results | `prompts.Case`, `common.load_cases` |
| `results/` and `backups/` ignored, `.env.example` shipped | `.gitignore` |
