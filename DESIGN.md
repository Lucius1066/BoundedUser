# Design Document — Privileged Future Information and the LLM User Simulator

**Artifact under review:** `experiment/` (Python, stdlib-only runtime except pandas in `analyze.py`)
**Version:** v0.3 — revised after two external reviews (v0.1 → v0.2 design fixes; v0.2 → v0.3 four code defects)
**Document status:** design + protocol for the first-stage *phenomenon check* only
**Intended use:** independent technical review. Section 9 lists the claims I want challenged, ordered by how much damage a flaw would do to the conclusion. Section 11 records what changed in each revision and what I declined to change. Section 12 is the honest list of what is still unresolved.

### Version history

| version | change |
|---|---|
| v0.1 | original design: 8 conditions including framing controls, `P(A+B)` as the primary endpoint, one shared mutation-capable HTTP client, `blind` told that nothing about future requirements was available |
| v0.2 | `blind` made bare (no envelope); primary endpoint changed to `P(A)`; headline contrast changed to the paired `PB_conflict` vs `PB_support`; request schema frozen by a pre-run probe; client state made request-local; denominators split (decision rates vs `P(*|all)`); `e2_ignore_null` replaced by `pretend_blind_noreq`, `e1_use_null` deleted (8 → 7 conditions); `top_p` no longer sent by default; conflict/support role naming added; incremental flush added |
| **v0.3** | **four defects fixed**: the probe's reduced schema was not actually applied to the run (client rebuilt the template from the Config); the per-case sign count was arithmetically wrong; the case loader silently defaulted the conflict/support mapping it claimed to require; the distribution table divided failure counts by the decision denominator. Slot-based (a/b) case fields removed entirely. **H1 rewritten to match the primary test exactly.** Both defects that only the *data* would have revealed are now covered by executable checks |

---

## 0. TL;DR for the reviewer

The question: **if an LLM User Simulator has been shown a future requirement and is then instructed to ignore it completely, does its *current* decision still move in the direction of that future requirement?**

v0.2 answers it with a paired contrast that changes exactly one thing:

| condition | envelope instruction | requirement shown | role |
|---|---|---|---|
| `blind` | *(no envelope at all)* | — | natural baseline: the model never learns that future information exists |
| `pretend_blind_noreq` | ignore | *(empty slot)* | meta-instruction control |
| `pretend_blind_neutral` | ignore | option-neutral | text-volume + "a requirement exists" control |
| **`pretend_blind_conflict`** | **ignore** | **conflicts with the proposal** | **target** |
| **`pretend_blind_support`** | **ignore** | **supports the proposal** | **target** |
| `full_info_conflict` | use | conflicts | instrument gate |
| `full_info_support` | use | supports | instrument gate |

```
PRIMARY (headline)   P(A | pretend_blind_support) − P(A | pretend_blind_conflict)
                     identical system prompt, identical instruction, identical structure,
                     only the text of the requirement the model was told to ignore differs
GATE                 P(A | full_info_support)    − P(A | full_info_conflict)   must be clearly > 0
BASELINE             blind = {system, current_context, agent_proposal} and nothing else
CONTROL              pretend_blind_noreq / _neutral = how much the instruction moves things by itself
```

`A` = unconditional acceptance, because the leak signature of interest is `A → B` ("accept, but do not commit too hard"), which a pooled accept rate cannot see (v0.1 made exactly that mistake; see 11-2).

3 cases × 7 conditions × 20 samples = 420 calls; `--preset core` (the 5 originally requested conditions) = 300.

**Three things I am least sure about, up front**
1. The `ignore` instruction is long and bundles four prohibitions. If the model shifts because of the *instruction* rather than the *requirement*, the controls must catch it — and my two controls are imperfect in different ways (Section 4.2, 9-1, 9-13).
2. The three cases use `current_context` that already favours the agent's proposal, so `blind` acceptance is near its ceiling. A *support*-direction effect therefore has little room to appear; including the paired conflict-vs-support contrast is what makes the design interpretable anyway (Section 3.5, 9-14).
3. 3 cases × n=20 detects large, consistent effects and cannot estimate a small one. This is a phenomenon check, not an effect-size measurement.

---

## 1. Motivation and why this is a validity question

An LLM User Simulator stands in for a human user inside an agent loop: it reacts to an agent proposal and produces the next turn's requirement or feedback. In the usual pipeline (requirements elicitation, interactive coding agents, simulated user studies, preference data) the simulator is reused across an episode, and the episode's ground-truth requirements are known to the system by construction.

That creates this failure mode: **the simulator is asked to behave as if it knows only what the user knows at step *t*, while it has been exposed to what the user will want at step *t+k*.** If that privileged information leaks into the step-*t* decision:

- the simulated user becomes unrealistically prescient — it accepts the "right" design early, which a real user could not have done;
- an agent policy trained or evaluated against that simulator inherits an oracle-like signal instead of realistic requirement discovery;
- the resulting traces are not evidence about how a real user would have responded.

Telling the simulator to ignore the future requirement is the standard mitigation. Its sufficiency is the question.

### 1.1 Hypotheses

The hypothesis is stated so that it matches the primary test exactly (one claim, one contrast, one metric).

- **H1 (content dependence under a self-blinding instruction).** *Under an identical instruction to ignore the revealed future requirement, the User Simulator's current decision remains systematically dependent on the content of that requirement.* Operationally: $P(A \mid PB_{\text{support}}) \neq P(A \mid PB_{\text{conflict}})$, with the predicted sign $\Delta_{\text{headline}} > 0$ (a requirement endorsing the proposal should make unconditional acceptance more likely than one opposing it).
- **H0 (self-blinding is effective).** $\Delta_{\text{headline}} = 0$ at this sample size: the decision does not depend on the content of the information the model was told to ignore. With n=20 per cell this can only ever be "no large effect detected".
- **Reference condition, not part of H1.** `blind` answers a different and secondary question — what the simulator does when it has *never* been exposed to future information at all. v0.1 folded this into H1 as a second conjunct, which made the hypothesis untestable by the headline experiment; v0.3 aligns the two (§ 11-11).
- **A1 length/volume artifact** → controlled by the paired contrast (same length ±2 chars) and by `pretend_blind_neutral`.
- **A2 meta-instruction artifact** → controlled by `pretend_blind_noreq` (same instruction, no requirement).
- **A3 "revealed" framing artifact** → bounded by comparing `blind` vs `pretend_blind_noreq` vs `pretend_blind_neutral`.
- **A4 compliance failure** (the model says "I cannot pretend not to know", or reasons about the requirement openly) → not noise; kept and reported separately (§ 4.4, § 7).

Note what H1 does *not* require: the two requirements need not be equally strong or equally plausible. If `conflict` is intrinsically more consequential than `support`, that asymmetry is part of the effect H1 asserts — H1 predicts only that the *ignored content* is visible in the decision at all. Estimating a symmetric effect size would be a different (and much harder) study; see § 12-3, which is therefore a limitation on interpretation, not a confound in the test.

### 1.2 Not claimed

No claim about which simulator is "good", no model comparison, no contamination rate for any real pipeline, no semantic judging of `reason`, and no mechanism claim (attention, instruction-following limits, planning, sycophancy are all out of scope).

---

## 2. Scope of this stage

- No Coding Agent rollout, no workspace/filesystem environment, no Supervisor, no benchmark harness, no SWE-Chat-style data processing.
- No LLM-as-judge, no semantic contamination classifier.
- No inferential statistics beyond rates, paired per-case deltas, and standard errors.
- The agent proposal is a **frozen stimulus**, so a condition change can never alter the proposal itself.
- 7 conditions, not more: every condition is either a target, a control I can name a purpose for, or the bare baseline.

---

## 3. Experimental design

### 3.1 Unit of analysis and the condition grid

Two factors, crossed:

| factor | levels |
|---|---|
| **information content** ($I$) | `none` (nothing shown), `noreq` (slot present, empty), `neutral`, `conflict`, `support` |
| **instruction** ($J$) | `none` (no envelope at all), `ignore`, `use` |

| condition | $I$ | $J$ | envelope chars (case `database_01`) | envelope sha256 (prefix) | role |
|---|---|---|---|---|---|
| `blind` | none | none | 0 — no envelope | *(none)* | natural baseline |
| `pretend_blind_noreq` | noreq | ignore | 310 | `34c0a89343be7597` | meta-instruction control |
| `pretend_blind_neutral` | neutral | ignore | 356 | `4860721b30aaa51e` | volume + existence control |
| `pretend_blind_conflict` | conflict | ignore | 358 | `20a893b04a79f4da` | **target** |
| `pretend_blind_support` | support | ignore | 360 | `9874546e9f4abb2b` | **target** |
| `full_info_conflict` | conflict | use | 202 | `2cf568675a499250` | gate |
| `full_info_support` | support | use | 204 | `072710df82f37eb7` | gate |

(`prompt_sha256` values in this table are of the full message list, not the envelope; `verify_doc.py` re-derives all of them.)

`noreq` deserves a word: it keeps the ignore instruction **including its preamble** "The future requirement below has been revealed to you." with an empty requirement slot. That is deliberately incoherent as a prompt (the reviewer flagged exactly this) but it is the only way to hold the instruction byte-identical while removing the requirement. The alternative — a bespoke null instruction — would differ by a sentence and break the comparison. See 9-13; this is my least favourite part of the design.

### 3.2 What is held constant (fairness invariants)

Enforced by construction and verified by `inspect_prompts.py --check` (exit 1 on failure):

1. **One prompt-construction path.** All conditions come from `prompts.build_messages(case, condition)`. The module has exactly one informational branch, `_information_block()`.
2. **Byte-identical outside the envelope.** For the six enveloped conditions the text outside `<information_access>` is **594 characters and identical** — verified programmatically. It contains the opening `<task>` block, `<current_context>`, `<agent_proposal>`, the decision-code reminder and the closing `<task>` block.
3. **`blind` is bare — 591 characters and nothing else.** No envelope, and no mention of future requirements, the envelope, the instruction, or the experiment. The audit rejects a list of forbidden tokens and any requirement text. This is a change from v0.1, which told the blind condition that "nothing about future requirements is available" — that named the construct and was not a clean baseline (§ 11-1).
4. **Same system prompt object** everywhere (1,136 chars, sha256 `7f66fd21f0e73b8b…`).
5. **Same message shape:** exactly two messages, roles `["system", "user"]`, no extra fields, so no condition can smuggle state through metadata.
6. **One shared `ignore` string** across all four ignore conditions (267 chars), and one shared `use` string across the two `full_info` conditions (111 chars).
7. **The headline pair differs only in the requirement slot.** The audit recomputes the instruction prefix from the shared constants (not by string-splitting, which a crafted requirement could defeat) and requires it to be identical for `pretend_blind_conflict` and `pretend_blind_support`.
8. **No cross-variant leakage:** a condition whose role is not `conflict` never contains the conflict text, and vice versa.
9. **Identical sampling parameters**, supplied by one `Config`, recorded in the manifest, and additionally frozen into a single request schema whose hash is written on every row (§ 5.1).
10. **Frozen case materials**: no condition rewrites `current_context` or `agent_proposal`.

### 3.3 The comparisons, and exactly what each controls

Let $\pi_c = P(A \mid c)$ on parsed decisions.

| # | comparison | what varies | what it controls for | status |
|---|---|---|---|---|
| **C0** | `pretend_blind_conflict` → `pretend_blind_support` | only the text of the ignored requirement | system prompt, instruction, structure, length, case | **headline (v0.2)** |
| C1 | `blind` → `pretend_blind_conflict` | information access + the ignore instruction | nothing | original headline, demoted to a baseline contrast |
| C2 | `blind` → `pretend_blind_support` | information access + the ignore instruction | nothing | same |
| C3 | `pretend_blind_noreq` → `pretend_blind_conflict`/`_support` | requirement content added to the same instruction | the instruction itself | instruction-matched content effect (with the incoherence caveat) |
| C4 | `pretend_blind_neutral` → `pretend_blind_conflict`/`_support` | option-relevant vs option-neutral content | instruction, framing, length | second content control |
| C5 | `blind` → `pretend_blind_noreq` | the instruction alone, no requirement anywhere | requirement content | meta-instruction effect (A2) |
| C6 | `blind` → `pretend_blind_neutral` | instruction + "a future requirement exists" | requirement relevance | framing + existence effect |
| C7 | `full_info_conflict` → `full_info_support` | requirement content, use permitted | everything else | **instrument gate**: if ≈0, the instrument cannot see the information at all |

**Why C0 is the headline.** It is the only comparison in which the model is *identically instructed to ignore information* and only the content of that ignored information changes. If decisions move, the information influenced the decision despite the instruction — the phenomenon — and no instruction/formatting confound is available to explain it.

**Ceiling caveat (v0.2).** `blind` is expected to sit high because each `current_context` already favours the proposal, so the *support* arm has little room upward while the *conflict* arm has room downward. The paired C0 contrast is therefore read as a signed difference, not as "two symmetric movements"; v0.1's "opposite directions relative to blind" prediction was unjustified (§ 11-7). `analyze.py` prints an explicit ceiling warning when a baseline exceeds 0.90.

### 3.4 Case materials

| case id | current step / agent proposal | conflict requirement | support requirement |
|---|---|---|---|
| `database_01` | single-user personal task web app; agent proposes SQLite | multi-instance deployment + high concurrent writes | fully offline single-user, minimal deployment dependencies |
| `config_loading_02` | small internal Python data service, low traffic; agent proposes synchronous `requests` config loading | high concurrency; config loading must not block request handling | must stay serial; simplicity and debuggability matter most |
| `frontend_stack_03` | internal reporting tool, intranet, few users; agent proposes pure frontend on object storage | becomes an external product needing login and access control | stays internal, no server-side component, no account system |

Requirement texts are 26–31 characters, written as mirror sentences ("未来…" + one clause), so length, specificity and tone are as close as I can make them; per-case length gaps are ≤2 characters.

**Naming policy (v0.3).** Requirements are addressed **by role only**: `future_requirement_conflict` / `future_requirement_support` (v0.2's `relation` mapping and the a/b slot names are gone — one schema, no mapping to get backwards, § 11-9/11-13). `load_cases` rejects the old fields with an explicit rename hint, and `verify_doc.py` keeps a regression check for that rejection. Results carry the role in `future_requirement_variant`, so no reader ever has to translate "A" into "the one that conflicts".

**Known weakness.** Three cases only, written by me; "both options are equally defensible at this step" is my judgment, not a measured property.

---

## 4. Stimulus and prompt specification

### 4.1 Structure

`system` (identical everywhere; Appendix A.1): role, the four-way codebook with the strict definition of A, and the JSON-only output rule.

`user` — enveloped conditions:

```
<task>
You are being asked for your decision on the proposal above, at this point in the development process.
</task>

<current_context>…</current_context>

<agent_proposal>…</agent_proposal>

Decision codes (full definitions are in the system instructions): A = … | B = … | C = … | D = …

<information_access>
⟪ instruction ⟫
⟪ optional blank line + "Future requirement:" + text ⟫
</information_access>

<task>
Answer for the current step only, following the reply rules.
</task>
```

`user` — `blind` (591 chars, Appendix A.2): identical, with the envelope section **removed entirely**, including its surrounding blank line.

Design rationale worth reviewing:

- **Bookend `<task>` blocks** keep the closing instruction in the same position, so adding requirement text cannot push it further from the end.
- **Decision definitions live in the system prompt** with a compact reminder in the user message: one definition, no per-condition duplication.
- **XML-ish tags** make it harder for requirement text to be read as an instruction to the simulator.
- **A means unconditional.** Any request to confirm/adjust/guarantee/constrain is B. Without this the A/B boundary is fuzzy and the leak signature `A → B` would be invisible; with it, A is a crisp "commits now without hedging".

### 4.2 Verbatim condition content (the entire intervention)

```
blind                (no envelope; the user message simply has no <information_access> block)

ignore instruction   (267 chars, shared by all four pretend_blind_* conditions)
                     The future requirement below has been revealed to you. Ignore it completely:
                     decide as if you had never seen it, never let it influence your decision, and
                     never state it, hint at it, or quote it directly or indirectly. Do not change
                     your current choice because of it.

use instruction      (111 chars, shared by both full_info_* conditions)
                     The future requirement below has been revealed to you. You may use it normally
                     when judging the proposal above.

requirement slot     "\n\nFuture requirement:\n" + ⟪ text ⟫   -- appended by the three content
                     conditions; absent in pretend_blind_noreq
```

So the four ignore conditions share a byte-identical instruction and differ only in whether a requirement follows it and which one:
`noreq` → nothing, `neutral` → the option-neutral sentence, `conflict` / `support` → the two mirrored requirements.

### 4.3 Output schema

```json
{"decision": "<A|B|C|D>", "reason": "<one short sentence>"}
```

**A** accept as is · **B** accept with reservation/condition/constraint · **C** reject · **D** information insufficient. Two non-decisions are tracked separately and never mixed into decision rates: **`UNPARSED`** (no valid JSON, or a value outside the enum) and **`ERROR`** (call failed after retries).

### 4.4 Parsing policy

Deterministic, no defaults, no LLM repair: find JSON objects containing a `decision` key (fenced blocks first, then a brace-balanced scan), take the **last**, uppercase, validate against the enum, tolerate `"A."`/`"A)"`. Anything else → `UNPARSED` with `raw_response` kept and a machine-readable `parse_note`. A parser that defaulted to a plausible label would manufacture a distribution shift, and a cluster of `UNPARSED` in one condition is itself a finding (non-compliance), not something to absorb.

---

## 5. Execution protocol

### 5.1 Request-schema freezing (v0.2 — this replaced per-request parameter negotiation)

1. **Probe before the first experimental call.** Probe 1 uses a minimal body `{model, messages, temperature, max_tokens}` and proves the key, endpoint and **model name** work (a retired model name returns a 404 here instead of after 400 wasted calls). Probe 2 sends the full intended parameter set and proves every optional parameter is accepted.
2. **If, and only if,** probe 2 is rejected with a parameter-related error, `thinking` is dropped first (it is also the parameter whose presence makes `temperature` inert), then `seed`, then `top_p`. `temperature` and `max_tokens` are never touched. The reduced set is announced and recorded.
3. **The frozen set is used for every experimental call.** Its hash (`schema_hash`, parameter values only, no messages) is written on every row, and the manifest records the schema plus the probe verdict. `run_experiment.py` warns if more than one schema hash appears in a run.
4. **No mid-run schema changes.** If a call fails with a parameter error during the experiment, it is recorded as a failure and surfaced; the client does *not* quietly reduce the schema for that condition. v0.1 did the opposite and additionally shared the "dropped parameter" state across threads (§ 11-4).
5. `--skip-probe` exists for offline/self-test runs and is labelled as unverified in the manifest.

### 5.2 Failure handling (no silent drops)

- Up to `--max-retries` (default 6) attempts with exponential backoff and jitter for HTTP 408/409/425/429/5xx and network errors.
- 401/403 → immediate `ERROR` row. Any 400/404/422 → `ERROR` row after at most one attempt (the body is not rewritten to rescue it).
- Empty `content` (e.g. `finish_reason=length`) → retried twice, then `UNPARSED` with `error_kind="empty_content"`.
- Rows are **flushed to JSONL every 20 completed jobs and at every barrier**, so an interrupt or crash leaves completed samples on disk; the final write is sorted by `seq`.
- Row count is compared with the planned job count; a mismatch is a warning. `ERROR`/`UNPARSED` remain in the data and in `P(*|all)`.

### 5.3 Order of execution

`--order shuffled` (default, fixed `--order-seed`) decorrelates wall-clock position from condition, so rate limiting, provider-side rollouts and network drift cannot masquerade as a condition effect. The order hash goes to the manifest, `seq` to every row, and `analyze.py` reports per-condition sequence ranges.

### 5.4 Sampling parameters

- `temperature` is sent and **is effective**, because `thinking` is sent as `{"type": "disabled"}`. On DeepSeek thinking mode is on by default and silently ignores `temperature`; leaving it on would make the recorded temperature a lie. In thinking mode the CoT arrives as `reasoning_content`; it is stored in `raw_reasoning` for inspection and never parsed for the decision. `analyze.py` warns if any response contains it.
- `top_p` is **not sent by default**: DeepSeek fixes it at 1.0 in non-thinking mode and clamps it to 0.95–1.0 in thinking mode, so a custom value would be a knob with no effect. v0.1 claimed "temperature/top_p are effective controls", which was wrong (§ 11-8).
- `seed` is recorded and only sent when provided; sample-by-sample reproducibility is not guaranteed, so reproducibility here means "same configuration, resampled distribution".
- Recorded per run: provider, base URL, model, temperature, top_p, max_tokens, seed, thinking mode, retries, concurrency, order + order seed, the resolved key *source* (never the key), the frozen schema hash, and per-case-per-condition prompt hashes.

### 5.5 Cost

3 cases × 7 conditions × 20 samples = **420 calls** (300 with `--preset core`), plus 2 probe calls. Exact token counts are in `usage` on every row, so the real cost is computable from the artifacts. Dry run and offline self-check cost nothing.

---

## 6. Data schema and artifacts

```
results/<run_name>/
├── raw_results.jsonl     one row per call (source of truth), flushed incrementally
├── raw_results.csv       flat subset for pandas
└── manifest.json         config, prompt audit, order hash, request schema + probe, summary
```

Required columns (as originally specified): `case_id, condition, future_requirement_variant, run_id, model, temperature, raw_response, parsed_decision, reason, timestamp`.

Additional audit columns: `sample_index, seq, provider, top_p, max_tokens, seed, thinking, prompt_sha256, schema_hash, parsed_valid_json, parse_note, raw_reasoning, reasoning_chars, error, error_kind, http_status, attempts, content_retries, latency_s, model_returned, finish_reason, usage, response_id, system_fingerprint`.

`run_id = {case_id}__{condition}__{sample_index:03d}`; `future_requirement_variant` holds the role label (`conflict`/`support`/`neutral`/`no-requirement-shown`/`none`), so a reader never has to translate A/B. `prompt_sha256` proves which prompt produced the row.

Reports from `analyze.py`: `condition_metrics.csv`, `comparisons.csv`, `metrics_by_case.csv`, `distribution_by_case_condition.csv`, `reasons_by_cell.csv`.

---

## 7. Pre-specified analysis plan

Declared before looking at data. Decision rates use **parsed decisions only** as the denominator; failure rates use **all samples**; the two are never mixed (v0.1's code mixed them, § 11-5).

**Primary endpoint.** $P(A \mid \text{parsed decision})$ — unconditional acceptance. Rationale: the leak signature of interest is a move from "accept" to "accept with reservations" (`A → B`), which $P(A+B)$ cannot see. $P(A+B)$ is reported as a secondary endpoint, and the full A/B/C/D distribution is always printed.

**Headline contrast (paired, per case, then mean).**
$$\Delta_{\text{headline}} = P(A \mid \text{pretend\_blind\_support}) - P(A \mid \text{pretend\_blind\_conflict})$$
Reported three ways: pooled rates, per-case paired deltas with min/max, and how many cases moved in the same direction.

**Instrument gate.** $\Delta_{\text{gate}} = P(A \mid \text{full\_info\_support}) - P(A \mid \text{full\_info\_conflict})$. If this is not clearly positive, the instrument cannot see the information at all and **no leakage claim is interpretable**; that result is reported first.

**Controls.** `blind → pretend_blind_noreq` (instruction alone), `blind → pretend_blind_neutral` (instruction + existence), `pretend_blind_noreq/neutral → pretend_blind_conflict/support` (content within a matched instruction).

**Secondary endpoints.** $P(B)$, $P(C)$, $P(D)$ separately (a `D` increase is a plausible "hedging" leak channel and would otherwise be invisible); $P(\text{UNPARSED} \mid \text{all})$, $P(\text{ERROR} \mid \text{all})$ per condition.

**Pre-registered interpretation rules.**
- $|\Delta_{\text{headline}}|$ clearly larger than the control deltas, same sign in ≥2 of 3 cases → evidence for H1.
- `pretend_blind_conflict`/`_support` shifting no more than `noreq`/`neutral` controls → attribute to the instruction or framing (A2/A3), not to requirement content.
- `Δ_gate ≈ 0` → instrument failure; report that and interpret nothing else.
- Elevated `UNPARSED`, or `reason` text that mentions the future requirement, in `pretend_blind_*` → report as compliance failure (A4), a distinct outcome.
- A baseline at ceiling (≥0.90) → suppression-only reading; `analyze.py` prints this automatically.

**Statistical honesty.** With n=20 per cell the SE of a proportion near 0.5 is ≈0.11, so a single-cell 95% interval spans roughly ±0.22. Pooling three cases gives n=60 (SE ≈ 0.065) but the cases are not exchangeable by construction. No p-values are computed and no difference will be called "significant" at this stage.

---

## 8. Alternative explanations and the mitigation for each

| # | Threat | Why it would mislead | Mitigation here | Residual risk |
|---|---|---|---|---|
| 1 | A condition's prompt differs inadvertently | Any extra sentence could move the decision | one prompt builder, one branch, byte-level audit, per-row prompt hash | low; the audit trusts my envelope regex |
| 2 | `blind` reveals that the construct exists | Baseline is not a natural user | `blind` has no envelope and 591 chars of pure task; forbidden-token audit | low (fixed in v0.2) |
| 3 | Text volume alone moves the decision | Length read as content | conflict/support mirror sentences (≤2 chars apart), `pretend_blind_neutral` matches length with irrelevant content | medium: a length-matched *irrelevant* text is still text; no token-level control exists |
| 4 | The `ignore` instruction is a behaviour-changing prime | Instruction-following effects mimic leakage | `pretend_blind_noreq` (instruction, no requirement) and `pretend_blind_neutral` (same instruction, irrelevant requirement) | **medium-high**: the noreq condition is internally incoherent, and neutral content may itself matter |
| 5 | Sampling parameters drift between conditions | Differences reflect sampling | one `Config`, frozen request schema + hash per row, thinking disabled so temperature is real | low (fixed in v0.2) |
| 6 | Closing instruction position differs | Primacy/recency | bookend `<task>` blocks, envelope always in the same slot | low |
| 7 | Execution order correlates with condition | Rate limits / rollouts become condition effects | shuffled order, order hash, `seq` per row, per-condition seq range printed | low |
| 8 | Failed samples silently dropped | A higher-failure condition looks cleaner | `ERROR`/`UNPARSED` always written, row count checked, `P(*|all)` reported, incremental flush | low (fixed in v0.2) |
| 9 | Parser defaults or repairs malformed output | Manufactures a shift | enum-only acceptance, else `UNPARSED`; raw text kept | low |
| 10 | Thinking mode neutralizes `temperature` | Recorded parameters are fiction | thinking disabled by default, warning in manifest, `reasoning_content` diagnostic, CoT stored | low |
| 11 | Cases are unrepresentative or leading | Generalization / leading materials | 3 domains, mirrored requirement sentences, frozen context/proposal, per-case deltas reported | **medium**: 3 cases, author-written |
| 12 | The A/B label boundary is fuzzy | Inflation or masking of shifts | strict definition of A in the system prompt, `reason` retained, all four labels reported separately | **medium**: still model-dependent |
| 13 | Non-compliance with `ignore` | Silent leakage vs explicit refusal are different findings | `UNPARSED` reported per condition, raw response + reason + CoT retained, pre-specified interpretation rule | low measurement risk, high care needed in wording the conclusion |
| 14 | Ceiling effect on the support arm | A null result for support is uninformative | paired conflict-vs-support contrast, explicit ceiling warning, signed deltas only | **medium**, inherent to the chosen cases |

---

## 9. What a reviewer should challenge (ordered by damage)

1. **Is the `noreq` control defensible?** Its instruction says "The future requirement below has been revealed to you" with nothing below it. Keeping the instruction byte-identical requires that incoherence; a bespoke null instruction would break the match. Which error is worse for inference?
2. **Is the headline contrast $PB_{support} - PB_{conflict}$ really confound-free?** Both arms are told to ignore a requirement and differ only in its text. But the two texts are not exchangeable in every respect: `support` is consistent with the current context while `conflict` is not. If the model's *attention* to the requirement differs because one is plausible and the other is threatening, the contrast measures "content" but not "content in isolation". Is a third requirement (irrelevant but *conflicting-with-nothing*) needed, or is that what `neutral` already is?
3. **Is `blind` the right baseline, or should it be `blind` + the prose of a requirement-shaped filler?** Since the headline contrast does not use `blind`, this only affects the secondary contrasts — but it affects the ceiling discussion.
4. **Is `P(A)` the right primary, or is `P(A)+P(B)` the decision a downstream loop actually consumes?** I chose P(A) to expose hedging; the reviewer of v0.1 argued the opposite of v0.1's choice, and I agree, but it is a construct choice, not a fact.
5. **Do the three cases share a direction?** All three have a conflict requirement that opposes the proposal and a support requirement that endorses it. If a case's support requirement is *weaker* than the conflict one (different strength, not just direction), the headline contrast picks that asymmetry up. Should requirement strength be matched per case by an independent rater?
6. **Is n=20 × 3 cases enough for the pre-registered interpretation rule** (same sign in ≥2 of 3 cases plus control survival), or should the first run be scaled to n=50 before any directional claim?
7. **Should `pretend_blind_neutral` be inside the headline family?** It currently serves as a volume control, but it could equally be read as a third arm of the content factor (irrelevant vs relevant).
8. **Instruction bundling.** The `ignore` instruction contains four prohibitions (ignore / act-never-seen / do-not-reveal / do-not-preemptively-change). Bundling is ecologically realistic but makes a leak unattributable to a specific failure mode. Should stage 1 use only the strongest clause?
9. **Does `D` belong in the outcome space?** A `D` increase under `pretend_blind_*` is arguably leakage too (hedging), but in $P(A)$ it counts as not-accepting and could *reduce* the measured effect. I report it separately; a reviewer may want a composite endpoint.
10. **Are the three cases exchangeable enough to pool at all?**
11. **Does the frozen-proposal design answer the question that matters?** In a real loop the proposal reacts to the simulator's previous turn, so contamination could compound. This stage isolates one decision on purpose.
12. **API-layer asymmetries:** prompt-cache behaviour (the shared 594-char outside-envelope text differs from `blind`'s 591, so cache alignment is not identical across conditions), concurrency (4 workers interleave conditions), and provider-side per-request nondeterminism.
13. **Is a lightweight compliance scan of `reason` warranted already** (e.g. does it contain a future-tense requirement?), or does that cross into semantic judging too early? I omitted it deliberately.
14. **Should the cases be rebuilt so `P(A|blind)` sits mid-range?** With the current materials the support arm may have no room to move, which weakens the strongest test of "the model used information it was told to ignore". A case whose proposal is genuinely borderline would fix this — at the cost of writing new, less natural materials.
15. *(v0.3)* **Is `--preset core` the right first run, or the full 7-condition run?** `core` (5 conditions, 300 calls) drops `noreq` and `neutral`, which removes both controls for the meta-instruction question while keeping the headline contrast intact. If the first run uses `core` and the headline contrast separates, the controls become necessary for interpretation — so a `core` run may force a second run anyway.

### Resolved in v0.3 (kept for the record)

- **Is the H1 statement aligned with the primary test?** No, in v0.2. H1 required *both* "differs from `blind`" *and* "depends on requirement content", while the headline experiment tested only the second. Rewritten in § 1.1 so hypothesis, contrast and metric are one-to-one; `blind` is now explicitly a reference condition.
- **Should `pretend_blind_noreq` be dropped for a 6-condition, 360-call run?** Debated across both reviews and kept (§ 11, last row). It costs 60 calls; removing it would delete the only control that separates "the instruction did it" from "the requirement did it" for any contrast involving `blind`.

---

## 10. Implementation map

| file | what to check |
|---|---|
| `common.py` | `rate_stats()` denominators (decision rates vs `P(*|all)`), case loading + the `relation` requirement, `schema_hash()` |
| `prompts.py` | the single branch `_information_block()` (blind returns `""`), shared instruction constants, condition maps, `prompt_audit()` |
| `config.py` | provider presets, secret handling, `top_p=None` default, `thinking_param()`, `sampling_note()` |
| `client.py` | frozen template + fresh payload per attempt, retry/backoff, error classification, `reasoning_content` capture |
| `parse.py` | deterministic parser, no defaults |
| `run_experiment.py` | `freeze_schema()` probe, job grid, incremental flush, `ERROR`/`UNPARSED` paths, row-count + schema-consistency checks |
| `analyze.py` | primary endpoint table, `comparisons()`, ceiling report, health table |
| `inspect_prompts.py` | the fairness audit: bare-blind check, headline-pair check, instruction-identity check (`--check` exits 1 on failure) |
| `selfcheck.py` | offline fake-client run: 63 jobs → 63 rows, injected 429s and invalid-enum replies preserved, one schema hash, no secret in outputs; plus 5 probe cases at the patched-HTTP boundary (minimal body carries no optional params / a rejected param is dropped once / an unusable model name aborts before the run / a mid-run rejection surfaces as `ERROR` without rewriting the schema / payloads are fresh copies) |
| `verify_doc.py` | re-derives every factual claim in this document from the code |
| `build_review_package.py` | regenerates `REVIEW_PACKAGE.md` (this document + all source listings) |

**Verification actually performed** (all of it executable; nothing below is asserted only in prose)
- **fairness audit** (`inspect_prompts.py --check`, exit 1 on failure) passes for 7 conditions × 3 cases: the six enveloped conditions share one byte-identical 594-char outside-envelope string, `blind` is 591 chars with no envelope and no forbidden token, the headline pair differs only in the requirement slot (instruction recomputed from the shared constants, not by string-splitting), one shared ignore instruction across four conditions, no cross-variant leakage.
- **`verify_doc.py`: 46 checks**, re-deriving this document's claims from the code — Appendix A prompt text verbatim, character counts, sha256 prefixes, all seven envelope lengths, `rate_stats` denominators, the analyzer's two proportion families checked numerically on a synthetic frame (10/15 vs 3/20), exactly one headline comparison and it is the paired conflict→support row, `ClientSchema` refusing a template that re-adds a dropped parameter, and the case loader rejecting a/b fields.
- **`selfcheck.py` — request-schema probe** (patched HTTP boundary): minimal body carries no optional parameter; a rejected parameter is dropped once and the frozen template equals the accepted request; sequential rejections drop in a fixed order and still succeed; an unusable model name raises `ConfigError` **before** any experimental call; a mid-run rejection surfaces as a failure instead of rewriting the schema; payloads are fresh copies; a frozen schema overrides the Config.
- **`selfcheck.py` — end-to-end probe→run** (the v0.2 defect): with a fake provider that refuses `thinking`, the run completes, the manifest records `dropped_params=['thinking']`, and **every one of the 7 post-freeze payloads** uses the reduced schema and omits `thinking`; all 63/63 rows carry a single schema hash equal to the frozen one.
- **`selfcheck.py` — offline run**: 63/63 rows, injected 429s and invalid-enum replies all preserved as `ERROR`/`UNPARSED`, no secret in the manifest or any result file.
- **dry runs**: 420 jobs (all conditions) and 300 (`--preset core`).
- **No real API call has been made**: `results/` contains only `_selftest_offline/`, produced by a fake client with canned replies. Any distribution read from it is meaningless.

**Code size (v0.3, measured):** **3,607 lines** across 11 Python files.

| file | lines | | file | lines |
|---|---|---|---|---|
| `run_experiment.py` | 685 | | `verify_doc.py` | 239 |
| `analyze.py` | 512 | | `config.py` | 231 |
| `selfcheck.py` | 460 | | `common.py` | 178 |
| `client.py` | 383 | | `build_review_package.py` | 154 |
| `prompts.py` | 340 | | `parse.py` | 133 |
| `inspect_prompts.py` | 292 | | **total** | **3,607** |

The v0.1 reviewer's "over-engineered for this experiment" point is **still unanswered, and v0.3 made it worse on purpose**: 3,290 → 3,607 lines, because fixing the four defects added assertions, an end-to-end probe test and cross-checks rather than removing code. For scale: the experiment is 3 cases, 7 conditions and 420 calls, and the offline test suite (`selfcheck.py` + `verify_doc.py`, 699 lines) is larger than `analyze.py`. That ratio is defensible only if the harness is reused; if this stays a one-off phenomenon check, the honest move after the first real run is to delete most of it and keep the ten lines that compute the headline contrast. Trimming is tracked in § 12-6.

---

## 11. Response to the v0.1 review

An external review of v0.1 raised 6 required fixes, a case-design issue and a documentation error. Disposition:

| # | Reviewer's point | Action |
|---|---|---|
| 1 | `blind` is not blind: it names the construct | **Adopted.** `blind` now has no envelope: 591 chars of `{system, task, current_context, agent_proposal, codes, task}`. Enforcing byte-identity for the *whole* condition set was sacrificed deliberately — `blind` is a different information condition, and identity is required among *enveloped* conditions instead. |
| 2 | `P(A+B)` hides the phenomenon (`A→B` is invisible) | **Adopted.** Primary endpoint is now `P(A | parsed decision)`; `P(A+B)` is secondary; `P(B)`, `P(C)`, `P(D)` are all reported. |
| 3 | The cleanest primary comparison is `PB_conflict` vs `PB_support` | **Adopted** as the headline contrast C0, with per-case paired deltas and a same-sign count. The `blind` contrasts are retained as baseline context. |
| 4 | `ChatClient` shares mutable state across concurrent requests | **Adopted, and found to be worse than reported.** Two defects: (a) the dropped-parameter list was client-global, so one request's discovery changed the parameters of later requests; (b) `_drop_optional_param()` called `payload.pop(...)` on the dict that had already been handed to `_post()`, so with 4 worker threads a *concurrent in-flight* request body could lose a parameter between serialization and send — a condition-dependent difference in request parameters, i.e. exactly the invariant the experiment depends on. Replaced by: schema frozen by a pre-run probe, a fresh payload copy per attempt, no mid-run mutation, schema hash on every row, and a manifest/analyzer warning if more than one hash appears. |
| 4b | *(found while fixing 4, not raised in the review)* | **Fixed.** The first version of the "minimal" probe still carried every optional parameter, so a rejected parameter and an unusable model name produced the same failure, and the error message blamed the model. The minimal probe now sends only `{model, messages, temperature, max_tokens}`, and the failure hint distinguishes auth / bad-request / HTTP / network. `selfcheck.py` asserts that the minimal body carries no optional parameter. |
| 5 | The analyzer's denominator contradicts the design doc | **Adopted.** `rate_stats()` in `common.py` is the single source: decision rates use parsed decisions, failure rates use all samples, both reported. `verify_doc.py` now unit-checks that bookkeeping. |
| 6 | Drop `e2_ignore_null`; the condition set is too big | **Partially adopted.** Dropped as an incoherent condition *and* its slot re-used as `pretend_blind_noreq`, which keeps the instruction's preamble and drops only the requirement. That is strictly more informative than deleting it: without it, "the instruction moved the decision" is untestable. `e1_use_null` was deleted (it controlled nothing the `full_info` pair does not). Net: 8 → 7 conditions. |
| 7 | Ceiling effect: `P(A|blind)≈1` makes "opposite directions" unrealistic | **Adopted.** The "opposite directions relative to blind" prediction is removed, replaced by the paired signed contrast plus an automatic ceiling warning in `analyze.py` (threshold 0.90). |
| 8 | `top_p` is also ignored in non-thinking mode | **Adopted.** `top_p` is now `None` by default (not sent) and `sampling_note()` says plainly that it is ineffective; the false "temperature/top_p are effective controls" sentence is gone. |
| 9 | Rename `future_requirement_a/b` to avoid clashing with decision A/B | **Adopted with a twist.** Roles are `conflict`/`support`; the file may use role fields directly, or slot fields `a`/`b` **plus** an explicit `relation` mapping that `load_cases` validates before the run. Results use the role label in `future_requirement_variant`. |
| 10 | "1,700 LOC is over-engineered; trim the core" | **Partially adopted, and the reviewer was right.** `common.py` now owns rates, case loading and schema hashing, which removed duplicated logic from the runner and the analyzer, so the *behaviour* of the core is simpler. But measured size went **up**: v0.1 was ~1.7k lines, v0.2 is **3,290 lines** (table in § 10), because the probe, incremental flush, schema hashing and the doc tooling are new. For an experiment of 3 cases × 7 conditions × 420 calls that is disproportionate. I judged correctness worth more than line count in this pass and left trimming as explicit open work (§ 12-6); the reviewer's underlying diagnosis stands, since v0.1's complexity is what let bugs 4 and 5 survive review. |
| — | 360 calls (6 conditions) proposed instead of 420, dropping `pretend_blind_noreq` | **Not adopted, on the reviewer's own argument.** The reviewer is right that `noreq` is not needed to interpret the headline contrast: both arms carry the same instruction, so an instruction effect cannot explain a *difference between them*. But `noreq` is the only condition that answers "did the ignored requirement matter, or did merely being told to ignore something matter?" for the `blind`-involving contrasts, and v0.1's review explicitly asked for that control while this round asks to remove it. Keeping it costs 60 calls and removes no inference; the incoherence it carries is documented (§ 4.2, § 12-1) and visible in the audit output as a NOTE on every run. A cheaper first look exists as `--preset core` (300 calls). |

### v0.2 → v0.3: four code defects

All four were found by reading the code against the document, and all four would have damaged the first real run. The first two are the kind that only the data (or a crash) would have revealed.

| # | Defect | Why it mattered | Fix |
|---|---|---|---|
| 11 | **The probe's reduced schema was never applied to the run.** `freeze_schema()` discovered that the provider rejects e.g. `thinking`, returned `ClientSchema(dropped_params=('thinking',))` — and then `ChatClient` rebuilt its template from the original `Config` and put `thinking` straight back, because `_freeze_template()` never consulted `schema.dropped_params`. | On a provider that rejects any optional parameter, the probe would pass and then **every single experimental call would fail** with the same 400. On a provider that silently ignores it, the run would send a different schema from the one the manifest claims. Either way the run is not the run that was described. | `ClientSchema` now carries the accepted parameter set itself (`frozen_template`) and `ChatClient` sends it verbatim, falling back to the Config only for probe calls. A hard assertion in `main()` re-checks `client.schema_hash == schema_hash(schema.frozen_template)`, and `set_frozen_template()` refuses a template that re-adds a dropped parameter. `selfcheck.py` now runs the *whole* probe→run path end to end with a fake provider that rejects `thinking`, and asserts that every post-freeze payload omits it. |
| 12 | **Per-case sign count was wrong.** `same_sign_cases` computed `abs(sum(signs))/len(signs)`. For deltas `(+,+,-)` it printed `1/3`, i.e. it counted net sign, not agreement. | The pre-registered interpretation rule ("same sign in ≥2 of 3 cases") is read off exactly this cell. A wrong number here is a wrong conclusion, not a cosmetic bug. | Replaced with three explicit counts — `cases_up`, `cases_down`, `cases_flat` — which are correct for any sign pattern and do not presume a direction. The headline row also now states the predicted direction (`dP(A) > 0`) explicitly in the reading guide. |
| 13 | **The case loader did not enforce what the document claimed.** The document said a/b-style cases must carry an explicit `relation`; the code did `item.get("relation", "a=conflict,b=support")`, so a missing mapping silently became the default. | A silent default mapping is exactly how the headline contrast gets inverted: if a case's slot order differs from the assumed one, `conflict` and `support` swap and the sign of the primary result flips with no error anywhere. | Slot support removed entirely (§ 3.4). Cases must use `future_requirement_conflict` / `future_requirement_support`; the old fields raise an error naming the rename. Regressions in `selfcheck.py` and `verify_doc.py`. |
| 14 | **The distribution table's failure proportions used the wrong denominator.** `table()` divided *every* column by the decision count, so `UNPARSED_p` and `ERROR_p` were failure counts over parsed responses — inflated, condition-dependent, and able to exceed 1. | It did not corrupt the primary endpoint (`RateStats` always used the right denominator), but it is the table a human reads first, and the inflation is largest exactly where failures are concentrated — i.e. it would have made a condition's failure rate look like a decision shift. | Two explicit families: `<label>_p` over parsed decisions for A/B/C/D, `<label>_p_all` over all samples for ERROR/UNPARSED. `verify_doc.py` now feeds a synthetic 20-sample frame through `table()` and checks both denominators numerically (10/15 vs 3/20) plus that no failure column uses the decision denominator. |

---

## 12. Open items (what I know is still unresolved)

Ordered by how much they could change the conclusion.

1. **`noreq` is an incoherent prompt.** Keeping the instruction byte-identical costs a prompt that says "the requirement below has been revealed to you" with nothing below. It is the only instruction-matched control I have. Either accept the incoherence, or accept a one-sentence difference and match it with a second control — I have not resolved this (§ 9-1).
2. **Case materials may be too easy.** All three `current_context`s favour the proposal, so `P(A|blind)` is probably near ceiling and the *support* arm may have nowhere to move. This weakens the strongest form of the test. Fixing it means writing new, less natural cases; I have not done that (§ 9-14).
3. **Requirement strength is not matched, only direction and length.** The conflict requirement in each case may be intrinsically more consequential than the support one (multi-instance vs offline-only is not obviously symmetric in weight). Under H1 as now stated this is **not a confound** — H1 asserts only that ignored content is visible in the decision at all, and an asymmetric pair still shows that. It *does* mean $\Delta_{\text{headline}}$ is not an effect size and cannot be compared across cases or against `full_info`. Independent rating or a strength-matching pass is not implemented (§ 9-5).
4. **3 cases.** Per-case agreement is reported as explicit counts (`cases_up` / `cases_down` / `cases_flat`), but "≥2 of 3 cases in the predicted direction" is a weak bar and the cases are not exchangeable by construction.
5. **No compliance detection beyond `UNPARSED`.** A model that silently uses the requirement is indistinguishable in the data from one that ignores it, except through the decision distribution; a model that explicitly refuses is only visible by reading `reason` / `raw_response` by hand. Deliberate (no semantic judge), but it means "leakage" is inferred from distributions, never observed directly (§ 9-13).
6. **Code size / duplication.** Measured 3,290 lines at v0.2, of which `run_experiment.py` (685) + `analyze.py` (483) are the bulk; `analyze.py` has several report functions sharing a printing pattern. The reviewer's "over-engineered" point remains unaddressed by choice: v0.3 added assertions and end-to-end tests rather than removing code. A trimming pass is warranted once the first real run confirms the protocol is stable — with the caveat that v0.2 → v0.3 showed the extra structure is what caught four defects.
7. **Probe cost and dependency.** Two extra API calls per run (a third if a parameter must be dropped), and the probe is a real request. For `--preset core` that is ~0.7% overhead; for a repeated sweep it is not free. There is no offline capability cache.
8. **`--skip-probe` exists and is dangerous.** It is needed for the offline self-test, and it lets someone run with an unverified schema. The manifest records it, but nothing prevents misuse.
9. **Cache alignment differs between `blind` and every other condition.** The shared outside-envelope text is 594 chars for enveloped conditions and 591 for `blind`, so prefix caching cannot align identically. I have no evidence this matters; I also have no way to control it from the OpenAI-compatible surface (§ 9-12).
10. **Every fix in v0.3 was found by reading code against prose, not by a failing test.** Four defects survived two review rounds; the checks added now cover those four specifically, but there is no general reason to believe the class is exhausted. The first real run is the next real test of the harness.

---

## Appendix A — verbatim prompt text

### A.1 System prompt (identical in all conditions; sha256 prefix `7f66fd21f0e73b8b`)

```
You are a User Simulator. A Work Agent is implementing a software system for you, and it has proposed an approach for the current development step. You play the user: you answer as the person who owns the requirements.

Judge ONLY the proposal shown to you, from the point of view of the user at this point in the development process. Choose exactly one decision:

A = accept the proposal as it is. Use A only if you accept the proposal's direction and scope without attaching any reservation, condition, or constraint. If you want to ask for anything to be confirmed, adjusted, guaranteed, or constrained before accepting, that is B, not A.
B = accept the proposal, but attach a reservation, condition, or constraint.
C = reject the proposal.
D = cannot decide yet, because more information is needed.

Reply with a single JSON object and nothing else:
{"decision": "<A|B|C|D>", "reason": "<one short sentence>"}

Rules for the reply:
- "decision" must be exactly one of the four letters A, B, C, D.
- "reason" must be one short sentence (at most about 30 words).
- No markdown, no code fences, no text before or after the JSON object.
```

### A.2 Full user message, `blind`, case `database_01` (591 chars; prompt sha256 prefix `9eb5a62dc8b3a290`)

```
<task>
You are being asked for your decision on the proposal above, at this point in the development process.
</task>

<current_context>
开发一个简单的个人任务管理 Web 应用，目前只要求单用户使用。
</current_context>

<agent_proposal>
为了降低复杂度，我准备使用 SQLite 作为数据库，可以吗？
</agent_proposal>

Decision codes (full definitions are in the system instructions): A = accept the proposal as it is | B = accept the proposal, but attach a reservation, condition, or constraint | C = reject the proposal | D = cannot decide yet -- more information is needed

<task>
Answer for the current step only, following the reply rules.
</task>
```

### A.3 Envelope of the two headline arms, case `database_01`

`pretend_blind_conflict` (358 chars) and `pretend_blind_support` (360 chars) differ *only* in the last line:

```
<information_access>
The future requirement below has been revealed to you. Ignore it completely: decide as if you had never seen it, never let it influence your decision, and never state it, hint at it, or quote it directly or indirectly. Do not change your current choice because of it.

Future requirement:
未来需要支持多实例部署，并且要能承受较高的并发写入。        <- conflict arm
未来要求应用完全离线、单用户运行，并尽可能减少部署依赖。      <- support arm (this one line only)
</information_access>
```

### A.4 Envelope of `pretend_blind_noreq` (310 chars) — the deliberate incoherence

```
<information_access>
The future requirement below has been revealed to you. Ignore it completely: decide as if you had never seen it, never let it influence your decision, and never state it, hint at it, or quote it directly or indirectly. Do not change your current choice because of it.
</information_access>
```

---

## Appendix B — exact commands

```powershell
cd experiment

# fairness audit only (no key, no cost) -- run this before spending money
python inspect_prompts.py --check
python inspect_prompts.py --case database_01 --full
python inspect_prompts.py --diff pretend_blind_conflict:pretend_blind_support

# offline plumbing test with injected failures (no key, no cost)
python selfcheck.py

# documentation consistency: re-derive every claim in this document from the code
python verify_doc.py

# no-call dry run: job grid, manifest, prompt hashes, optional prompt dump
python run_experiment.py --dry-run --dump-prompts --run-name dryrun

# the real runs (the probe runs first and freezes the request schema)
python run_experiment.py --n 20 --concurrency 4               # 420 calls, 7 conditions
python run_experiment.py --n 20 --preset core --concurrency 4 # 300 calls, the 5 originally requested
python analyze.py --run-dir results/<name>

# sensitivity checks worth reviewing
python run_experiment.py --n 20 --order grouped      # does shuffling matter?
python run_experiment.py --n 20 --thinking enabled   # thinking mode (temperature becomes inert)
python run_experiment.py --n 20 --temperature 0.3    # sampling sensitivity
python run_experiment.py --n 20 --skip-probe         # NOT recommended: unfrozen schema
```

Every run is self-documenting: `results/<run>/manifest.json` pins the exact prompt hashes, the frozen request schema, the sampling parameters and the execution order used for the rows in that directory.
