"""Offline plumbing test for the experiment harness.

Replaces the network layer with a fake client that returns canned/random model
replies, so the whole chain (schema freeze -> job grid -> records -> JSONL/CSV ->
manifest -> analyze.py) can be exercised without an API key or any API cost.

It also injects failures on purpose to prove that failed samples are recorded as
decision="ERROR" rows and never silently dropped, and it unit-checks the
request-schema probe at the HTTP boundary (patched `_post`).

Usage (from inside experiment/):
    python selfcheck.py     # writes results/_selftest_offline/, then runs analyze.py on it

The run directory is deliberately named with a leading underscore: the rows in it
are produced by a fake client (canned replies), NOT by a real model.  It exists to
prove the plumbing works and that failed samples are recorded.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import analyze
import client as C
import run_experiment as runner
from client import ChatFailure, ChatResult, schema_hash
from config import Config, ConfigError
from prompts import Message

HERE = Path(__file__).resolve().parent
RUN_NAME = "_selftest_offline"
RUN_DIR = HERE / "results" / RUN_NAME

REPLIES = [
    '{"decision": "A", "reason": "SQLite 足够简单，符合当前的单用户需求。"}',
    '```json\n{"decision": "B", "reason": "可以，但请确认后续迁移成本。"}\n```',
    'Sure! Here is my answer:\n{"decision": "C", "reason": "这个方案不适合当前需求。"}',
    '{"decision": "D", "reason": "还需要知道数据规模。"}',
    '{"decision": "E", "reason": "invalid enum value"}',  # -> UNPARSED
    "I cannot answer without more context.",  # -> UNPARSED
    "",  # -> UNPARSED (empty content)
]


class FakeClient:
    """Deterministic fake standing in for ChatClient.

    Mirrors the real client's public surface and parameter rules (`schema_hash`,
    `template`, `build_request`, `chat`), records every payload it was asked to
    send, and can be told to reject specific parameters the way a strict provider
    would -- which is what makes the probe path testable offline.

    Class attributes control behaviour:
      * `reject_params`: parameters the server refuses (raises a 400 the probe driver
        recognises as parameter-related)
      * `fail_every`: fail every Nth call after retries, to exercise the ERROR path
      * `forbidden`: parameters that must never appear in a payload (contract check)
      * `payloads`: every payload actually sent, for assertions
    """

    payloads: list = []
    reject_params: tuple = ()
    forbidden: tuple = ()
    fail_every: int = 7

    def __init__(self, config: Config, schema=None, rng=None):
        self.config = config
        self.calls = 0
        self.rng = random.Random(1234)
        self.schema = schema
        # Mirror the real client's precedence rule exactly: a frozen template wins,
        # and a frozen template is used for every call.  If it were rebuilt from the
        # Config here, this fake could not detect the very bug it exists to test.
        if schema is not None and schema.frozen_template is not None:
            self._template = dict(schema.frozen_template)
        else:
            # Same parameter rules as ChatClient._template_from_config().
            self._template = {
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stream": False,
            }
            if config.top_p is not None:
                self._template["top_p"] = config.top_p
            if config.seed is not None:
                self._template["seed"] = config.seed
            self._template.update(config.thinking_param())
            if config.reasoning_effort and config.thinking == "enabled":
                self._template["reasoning_effort"] = config.reasoning_effort
        if not FakeClient.payloads:
            FakeClient.payloads = []

    @classmethod
    def reset(cls, reject_params=(), forbidden=(), fail_every=7) -> None:
        cls.payloads = []
        cls.reject_params = tuple(reject_params)
        cls.forbidden = tuple(forbidden)
        cls.fail_every = fail_every

    @property
    def schema_hash(self) -> str:
        return schema_hash(self._template)

    @property
    def template(self):
        return dict(self._template)

    def build_request(self, messages):
        payload = dict(self._template)
        payload["messages"] = [m.as_dict() for m in messages]
        return payload

    def chat(self, messages):
        self.calls += 1
        payload = self.build_request(messages)
        leaked = [name for name in self.forbidden if name in payload]
        if leaked:
            raise AssertionError(
                f"request would send parameters the probe rejected: {leaked} "
                f"(payload keys: {sorted(payload)})"
            )
        FakeClient.payloads.append(payload)
        rejected = [name for name in self.reject_params if name in payload]
        if rejected:
            # Same *observable contract* as the real client for a parameter complaint:
            # errors never escape chat(), they come back as a ChatFailure, and the
            # error text names the parameter so the probe driver can classify it.
            return ChatFailure(
                error=(
                    f"bad-request 400: Unrecognized request argument supplied: {rejected[0]}"
                ),
                error_kind="bad_request",
                http_status=400,
                attempts=1,
                latency_s=0.0,
                request_payload=payload,
            )
        if self.fail_every and self.calls % self.fail_every == 0:
            return ChatFailure(
                error="http 429: rate limit exceeded",
                error_kind="http",
                http_status=429,
                attempts=self.config.max_retries,
                latency_s=0.01,
                request_payload=payload,
            )
        text = REPLIES[(self.calls - 1) % len(REPLIES)]
        return ChatResult(
            text=text,
            model_returned=self.config.model + "-fake",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            http_status=200,
            attempts=1,
            latency_s=0.005,
            request_payload=payload,
            response_id=f"fake-{self.calls}",
        )


CONDITIONS = 7
SAMPLES = 3
CASES = 3
EXPECTED_ROWS = CASES * CONDITIONS * SAMPLES


# --------------------------------------------------------------------------
# Request-schema probe, tested at the HTTP boundary with a patched _post
# --------------------------------------------------------------------------


def _fake_post_factory(behaviour):
    calls = []

    def fake_post(self, payload):
        calls.append(dict(payload))
        verdict = behaviour(payload, len(calls))
        if verdict is None:
            return ChatResult(text="ok", model_returned=payload["model"], http_status=200, attempts=1)
        status, body = verdict
        raise C._classify_http_error(status, json.dumps(body))

    return fake_post, calls


def check_schema_probe() -> None:
    print("\n### schema probe (patched HTTP layer) ###")
    messages = [Message("user", "Reply with the single word: ok")]
    config = runner.build_config(runner.parse_args(["--api-key", "offline-test", "--n", "1"]))
    real_post = C.ChatClient._post

    try:
        # 1. everything accepted -> frozen schema, minimal probe carries no optional params
        C.ChatClient._post, calls = _fake_post_factory(lambda payload, n: None)
        schema = runner.freeze_schema(config, quiet=True)
        assert schema.mode == "frozen" and schema.probed and not schema.dropped_params
        minimal_payload = calls[0]
        assert "thinking" not in minimal_payload and "seed" not in minimal_payload, (
            f"the minimal probe must not carry optional params: {sorted(minimal_payload)}"
        )
        assert schema.frozen_template is not None
        assert schema_hash(schema.frozen_template) == schema_hash(calls[-1]), (
            "the frozen template must be exactly what the accepted probe sent"
        )
        print("probe case 1: minimal body has no optional params; frozen template == accepted probe body")

        # 2. a rejected optional parameter is dropped once and the reduced schema is frozen
        C.ChatClient._post, calls = _fake_post_factory(
            lambda payload, n: (400, {"error": {"message": "Unrecognized request argument supplied: thinking"}})
            if "thinking" in payload
            else None
        )
        schema2 = runner.freeze_schema(config, quiet=True)
        assert schema2.dropped_params == ("thinking",), schema2.dropped_params
        assert "thinking" not in calls[-1]
        assert "thinking" not in (schema2.frozen_template or {})
        assert schema_hash(schema2.frozen_template) == schema_hash(calls[-1])
        print("probe case 2: rejected param dropped, frozen template == the accepted (reduced) request")

        # 2b. with a seed requested too, the drop order is thinking then seed
        C.ChatClient._post, calls = _fake_post_factory(
            lambda payload, n: (400, {"error": {"message": "unsupported parameter: seed"}})
            if "seed" in payload
            else (400, {"error": {"message": "Unrecognized request argument supplied: thinking"}})
            if "thinking" in payload
            else None
        )
        seeded_config = runner.build_config(
            runner.parse_args(["--api-key", "offline-test", "--n", "1", "--seed", "7", "--max-retries", "2"])
        )
        schema2b = runner.freeze_schema(seeded_config, quiet=True)
        # Drops happen in DROPPABLE_PARAMS order, one per retry, so a rejection that
        # only mentions `seed` still walks past `thinking` and `reasoning_effort`.
        assert tuple(runner.DROPPABLE_PARAMS[: len(schema2b.dropped_params)]) == schema2b.dropped_params
        assert "seed" in schema2b.dropped_params
        for name in ("thinking", "reasoning_effort", "seed"):
            assert name not in (schema2b.frozen_template or {})
        print(
            "probe case 2b: sequential rejections -> "
            f"{len(schema2b.dropped_params)} params dropped in fixed order, probe still succeeds"
        )

        # 3. a retired model name aborts the run before any experimental call
        C.ChatClient._post, calls = _fake_post_factory(
            lambda payload, n: (404, {"error": {"message": "Model Not Exist"}})
        )
        try:
            runner.freeze_schema(config, quiet=True)
            raise AssertionError("expected ConfigError for an unusable model name")
        except ConfigError as exc:
            assert "model" in str(exc).lower()
            print("probe case 3: unusable model name raises ConfigError before the run")

        # 4. a mid-run parameter rejection is reported, never silently absorbed
        C.ChatClient._post, calls = _fake_post_factory(
            lambda payload, n: (400, {"error": {"message": "unsupported parameter: seed"}})
        )
        outcome = C.ChatClient(seeded_config, schema).chat(messages)
        assert isinstance(outcome, ChatFailure) and outcome.attempts == 1
        assert "seed" in outcome.error
        print("probe case 4: mid-run rejection surfaces as an ERROR row, schema is not rewritten")

        # 5. payloads are fresh copies; mutating one cannot reach the template
        client = C.ChatClient(config, schema)
        first = client.build_request(messages)
        first["temperature"] = 99
        second = client.build_request(messages)
        assert second["temperature"] == config.temperature
        assert client.schema_hash == schema_hash(second)
        print("probe case 5: fresh payload per attempt; schema hash reflects the frozen params")

        # 6. a frozen template is used verbatim, NOT rebuilt from the Config
        reduced_config = runner.build_config(
            runner.parse_args(["--api-key", "offline-test", "--n", "1", "--seed", "7"])
        )
        frozen_client = C.ChatClient(reduced_config, schema2b)
        assert "seed" not in frozen_client.template and "thinking" not in frozen_client.template, (
            f"a frozen schema must override the Config: {sorted(frozen_client.template)}"
        )
        assert schema_hash(frozen_client.template) == schema_hash(schema2b.frozen_template)
        print("probe case 6: frozen schema overrides the Config (dropped params stay dropped)")
    finally:
        C.ChatClient._post = real_post


def check_frozen_schema_reaches_the_run() -> None:
    """End-to-end proof of the bug that made the probe useless.

    The probe discovers that the provider rejects `thinking`, freezes a reduced
    schema, and the run must then send that reduced schema on EVERY call.  Before
    this test existed, `ChatClient` rebuilt the template from the Config and put
    `thinking` straight back, so every experimental call failed.
    """
    print("\n### frozen schema actually reaches the run (end-to-end) ###")
    real_fake = runner.ChatClient
    run_dir = HERE / "results" / "_selftest_probe"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    try:
        # The provider refuses `thinking`.  main() runs its real freeze_schema()
        # path; if the frozen (reduced) schema were not what the run uses, either
        # the probe would reject every call, or the payloads would quietly carry
        # `thinking` again -- both are asserted below.
        FakeClient.reset(reject_params=("thinking",), forbidden=(), fail_every=0)
        runner.ChatClient = FakeClient  # type: ignore[assignment]
        code = runner.main(
            [
                "--run-name", "_selftest_probe",
                "--results-dir", str(HERE / "results"),
                "--api-key", "offline-test",
                "--n", "1",
                "--conditions", "blind,pretend_blind_conflict",
                "--concurrency", "1",
            ]
        )
        assert code == 0, f"run returned {code} (a probe/run schema mismatch shows up here)"

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        schema_info = manifest["request_schema"]
        assert schema_info["dropped_params"] == ["thinking"], schema_info["dropped_params"]
        assert "thinking" not in schema_info["frozen_template_keys"], schema_info["frozen_template_keys"]
        frozen_hash = schema_info["frozen_schema_hash"]

        rows = [
            json.loads(line)
            for line in (run_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert rows, "no rows were written"
        hashes = {row["schema_hash"] for row in rows}
        assert hashes == {frozen_hash}, f"rows carry {hashes}, manifest froze {frozen_hash}"

        # Probe payloads are allowed to carry `thinking` (that is how the rejection is
        # discovered); every payload sent AFTER the schema was frozen must not.
        n_probe = 2  # minimal probe + first full probe (rejected)
        post_freeze = FakeClient.payloads[n_probe:]
        assert post_freeze, "no experimental payloads were recorded"
        assert not [p for p in post_freeze if "thinking" in p], (
            f"the dropped parameter reappeared in {len(post_freeze)} post-freeze payloads"
        )
        assert {schema_hash(p) for p in post_freeze} == {frozen_hash}
        print(
            f"probe+run: {len(FakeClient.payloads)} calls total; the {len(post_freeze)} after "
            f"freezing all used the reduced schema ({frozen_hash[:16]}) and none carried 'thinking'"
        )
        print("rows and manifest agree: one hash, no dropped parameter in the frozen template")
    finally:
        runner.ChatClient = real_fake  # type: ignore[assignment]
        FakeClient.reset()
        if run_dir.exists():
            shutil.rmtree(run_dir)


def check_relation_loader_is_strict() -> None:
    """The loader must not invent a conflict/support mapping for a/b style cases."""
    print("\n### case loader rejects the old a/b schema ###")
    from common import CaseFileError, load_cases

    legacy = [
        {
            "id": "x",
            "current_context": "c",
            "agent_proposal": "p",
            "future_requirement_a": "a text",
            "future_requirement_b": "b text",
        }
    ]
    # Written next to this file on purpose: the OS temp directory is not always
    # writable under the sandbox, and a self-test must not fail for that reason.
    path = HERE / "_selftest_legacy_cases.json"
    try:
        path.write_text(json.dumps(legacy), encoding="utf-8")
        try:
            load_cases(path)
            raise AssertionError("a/b-style case should have been rejected")
        except CaseFileError as exc:
            message = str(exc)
            assert "future_requirement_conflict" in message, message
            assert "future_requirement_a/b" in message, message
            print("loader rejects a/b fields and names the required role field")
    finally:
        if path.exists():
            path.unlink()


def main() -> int:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    check_schema_probe()
    check_frozen_schema_reaches_the_run()
    check_relation_loader_is_strict()

    runner.ChatClient = FakeClient  # type: ignore[assignment]

    argv = [
        "--run-name", RUN_NAME,
        "--results-dir", str(HERE / "results"),
        "--api-key", "not-a-real-key-offline-test",
        "--n", str(SAMPLES),
        "--order", "shuffled",
        "--concurrency", "1",
        "--skip-probe",
    ]
    print("### run_experiment.py (fake client) ###")
    code = runner.main(argv)
    print(f"\nexit code: {code}")

    jsonl = RUN_DIR / "raw_results.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    print(f"\nrows written: {len(rows)} (expected {CASES} cases x {CONDITIONS} conditions x {SAMPLES} = {EXPECTED_ROWS})")
    assert len(rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(rows)}"

    required = {
        "case_id", "condition", "future_requirement_variant", "run_id", "model",
        "temperature", "raw_response", "parsed_decision", "reason", "timestamp",
    }
    missing = required - set(rows[0])
    assert not missing, f"records missing required fields: {missing}"
    print("required fields present:", sorted(required))

    labels = {}
    for row in rows:
        labels[row["parsed_decision"]] = labels.get(row["parsed_decision"], 0) + 1
    print("decision labels:", labels)
    assert labels.get("ERROR", 0) > 0, "failure injection did not produce ERROR rows"
    assert labels.get("UNPARSED", 0) > 0, "no UNPARSED rows produced"
    assert sum(labels.values()) == len(rows), "some rows lost their decision label"
    print("no sample dropped: rows == sum(labels) ==", len(rows))

    schema_hashes = {row.get("schema_hash") for row in rows}
    assert len(schema_hashes) == 1 and None not in schema_hashes, f"schema not frozen: {schema_hashes}"
    print("one frozen request-schema hash on every row:", list(schema_hashes)[0][:16])

    manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    dumped = json.dumps(manifest, ensure_ascii=False)
    assert "not-a-real-key-offline-test" not in dumped, "API key value leaked into the manifest"
    assert "api_key" not in manifest["config"], "raw api_key field leaked into the manifest config"
    assert manifest["config"]["api_key_present"] is True
    assert manifest["summary"]["schema_consistent"] is True
    for path in RUN_DIR.glob("*"):
        if path.is_file():
            assert "not-a-real-key-offline-test" not in path.read_text(encoding="utf-8", errors="replace")
    print("manifest clean of secrets; all 7 conditions audited:", len(manifest["prompt_audit"]["database_01"]))

    print("\n### analyze.py (fake results) ###")
    analyze.main(["--run-dir", str(RUN_DIR), "--reasons", "0"])
    csvs = sorted(p.name for p in RUN_DIR.glob("*.csv"))
    print("\ncsv files written:", csvs)
    for expected in ("raw_results.csv", "condition_metrics.csv", "comparisons.csv", "metrics_by_case.csv"):
        assert expected in csvs, f"missing report {expected}"
    print("\nSELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
