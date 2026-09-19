"""Run the phenomenon check: build the (case x condition x sample) grid, call the
model, and persist one JSONL row per sample.

Guarantees that matter for the experiment:
  * every sample produces exactly one row -- API failures become decision="ERROR"
    rows after retries are exhausted, and are never dropped;
  * parsing failures become decision="UNPARSED" rows with the raw text kept;
  * the request schema is frozen BEFORE the first experimental call and the frozen
    template is what every experimental call sends (verified by hash, see
    `freeze_schema`), so a parameter cannot change halfway through a run;
  * `--order shuffled` (default) decorrelates execution time from condition;
  * rows are flushed to disk at every job barrier, so an interrupt or crash still
    leaves the completed samples on disk.

Usage (from inside experiment/):
    python run_experiment.py --dry-run                 # prompts only, no API calls
    python run_experiment.py --n 20 --concurrency 4
    python run_experiment.py --preset core             # 5 conditions, 300 calls
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import json
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from client import ChatClient, ChatFailure, ClientSchema, looks_like_param_error, schema_hash
from common import ERROR, UNPARSED, CaseFileError, load_cases
from config import Config, ConfigError
from parse import ParsedResponse, parse_response
from prompts import (
    ALL_CONDITIONS,
    CALIBRATION_CONDITIONS,
    CORE_CONDITIONS,
    Case,
    Message,
    build_messages,
    messages_to_payload,
    prompt_audit,
    prompt_sha256,
    variant_label,
)

HERE = Path(__file__).resolve().parent

# Windows consoles default to a legacy code page (e.g. cp936/GBK), which turns
# Chinese case text in log lines into '?' and can even raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

#: optional parameters the probe may remove, in order, one per retry
DROPPABLE_PARAMS: Tuple[str, ...] = ("thinking", "reasoning_effort", "seed", "top_p")

CSV_COLUMNS = [
    "case_id",
    "condition",
    "future_requirement_variant",
    "run_id",
    "model",
    "temperature",
    "parsed_decision",
    "reason",
    "timestamp",
    "parsed_valid_json",
    "parse_note",
    "reasoning_chars",
    "schema_hash",
    "error_kind",
    "error",
    "attempts",
    "latency_s",
    "prompt_sha256",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM User Simulator: privileged-future-information phenomenon check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "dashscope", "custom"])
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="", help="prefer the env var / .env instead of passing it here")
    parser.add_argument("--env-file", default=str(HERE / ".env"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None, help="not sent by default: DeepSeek ignores it")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None, help="only sent when provided; recorded either way")
    parser.add_argument(
        "--thinking",
        default="disabled",
        choices=["disabled", "enabled", "omit"],
        help="DeepSeek 'thinking' param; disabled keeps temperature effective",
    )
    parser.add_argument("--reasoning-effort", default="", help="only sent when --thinking enabled")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("-n", "--n", type=int, default=20, dest="samples", help="samples per condition")
    parser.add_argument(
        "--conditions",
        default="",
        help=f"comma-separated subset of: {','.join(ALL_CONDITIONS)}",
    )
    parser.add_argument("--preset", choices=["core", "all", "calibration"], default="all")
    parser.add_argument("--candidate-cases", default="", help="case file for Stage-1 calibration (alias for --cases)")
    parser.add_argument("--order", choices=["shuffled", "grouped"], default="shuffled")
    parser.add_argument("--order-seed", type=int, default=20260101)
    parser.add_argument("--limit-cases", type=int, default=0, help="0 = all cases")
    parser.add_argument("--extra-body", default="", help='JSON merged into the request body')
    parser.add_argument("--skip-probe", action="store_true", help="do not verify/freeze the request schema (not recommended)")
    parser.add_argument("--dry-run", action="store_true", help="build+audit prompts and write the manifest, but make no API calls")
    parser.add_argument("--dump-prompts", action="store_true", help="store the exact message arrays in the run directory")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing run directory")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    elif args.preset == "core":
        conditions = list(CORE_CONDITIONS)
    elif args.preset == "calibration":
        conditions = list(CALIBRATION_CONDITIONS)
    else:
        conditions = list(ALL_CONDITIONS)
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise ConfigError(f"unknown conditions {unknown}; known: {list(ALL_CONDITIONS)}")
    if args.preset == "calibration" and any(c.startswith("pretend_blind") for c in conditions):
        raise ConfigError(
            "the calibration preset must not include pretend_blind conditions: case selection would "
            "then be influenced by the effect under study. Use --preset all for the real experiment."
        )
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    return Config(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        extra_body=extra_body,
        samples_per_condition=args.samples,
        order=args.order,
        order_seed=args.order_seed,
        conditions=tuple(conditions),
        cases_path=args.cases,
        results_dir=args.results_dir,
        run_name=args.run_name,
        limit_cases=args.limit_cases,
    )


# --------------------------------------------------------------------------
# Job grid
# --------------------------------------------------------------------------


@dataclass
class Job:
    case: Case
    condition: str
    sample_index: int
    run_id: str

    @property
    def variant_label(self) -> str:
        return variant_label(self.condition)


def build_jobs(cases: Sequence[Case], config: Config) -> List[Job]:
    jobs: List[Job] = []
    for case in cases:
        for condition in config.conditions:
            for sample_index in range(config.samples_per_condition):
                jobs.append(
                    Job(
                        case=case,
                        condition=condition,
                        sample_index=sample_index,
                        run_id=f"{case.id}__{condition}__{sample_index:03d}",
                    )
                )
    if config.order == "grouped":
        return jobs
    rng = random.Random(config.order_seed)
    rng.shuffle(jobs)
    return jobs


def execution_order_hash(jobs: Sequence[Job]) -> str:
    blob = "\n".join(f"{j.run_id}|{j.condition}" for j in jobs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Schema freezing
# --------------------------------------------------------------------------


def _probe_config(config: Config, minimal: bool, dropped: Sequence[str]) -> Config:
    """A Config for one probe attempt: optionally reduced, optionally minimal."""
    candidate = Config(**{**config.__dict__, "max_retries": 2, "extra_body": {} if minimal else config.extra_body})
    candidate.api_key = config.api_key
    for name in dropped:
        if name == "thinking":
            candidate.thinking = "omit"
        elif name == "reasoning_effort":
            candidate.reasoning_effort = ""
        elif name == "seed":
            candidate.seed = None
        elif name == "top_p":
            candidate.top_p = None
    if minimal:
        candidate.thinking = "omit"
        candidate.reasoning_effort = ""
        candidate.seed = None
        candidate.top_p = None
    return candidate


def freeze_schema(config: Config, quiet: bool = False) -> ClientSchema:
    """Verify and freeze the request parameter set before any experimental call.

    1. **Minimal probe**: a genuinely minimal body
       `{model, messages, temperature, max_tokens}`.  It proves the key, endpoint
       and *model name* work, and because it carries no optional parameter, a
       rejected optional parameter can never be misreported as a bad model name.
    2. **Full probe**: the intended parameter set.  If (and only if) it is rejected
       with a parameter-related error, one optional parameter is removed and the
       probe is retried, up to `len(DROPPABLE_PARAMS)` removals.
    3. The parameter set that the provider accepted is **frozen into the returned
       schema**, and `ChatClient` sends exactly that set for every experimental
       call -- it is never rebuilt from the Config afterwards.
    """
    def say(message: str) -> None:
        if not quiet:
            print(message)

    if config.max_retries < 1:
        raise ConfigError("--max-retries must be >= 1")

    notes: List[str] = ["probe 1 (minimal): {model, messages, temperature, max_tokens}"]

    minimal_result = ChatClient(
        _probe_config(config, minimal=True, dropped=()), ClientSchema(mode="probe"), rng=random.Random(1)
    ).chat([Message("user", "Reply with the single word: ok")])
    if isinstance(minimal_result, ChatFailure):
        hint = {
            "auth": "the API key was rejected",
            "bad_request": "the model name or the request body was rejected",
            "http": "the endpoint returned an HTTP error",
            "network": "the endpoint could not be reached",
        }.get(minimal_result.error_kind, "the request failed")
        raise ConfigError(
            f"request-schema probe failed with a minimal body ({hint}); no experimental calls "
            f"were made. kind={minimal_result.error_kind} status={minimal_result.http_status} "
            f"error={minimal_result.error}\n"
            "Check --model / --base-url / --provider and the key. A retired model name returns 404 here."
        )
    if minimal_result.model_returned and minimal_result.model_returned != config.model:
        notes.append(
            f"probe 1: server reported model={minimal_result.model_returned!r} for requested {config.model!r}"
        )
    say(f"probe minimal   : OK (returned model={minimal_result.model_returned or 'n/a'})")

    dropped: List[str] = []
    while True:
        client = ChatClient(
            _probe_config(config, minimal=False, dropped=dropped),
            ClientSchema(mode="probe"),
            rng=random.Random(2),
        )
        probe = client.chat([Message("user", "Reply with the single word: ok")])
        if not isinstance(probe, ChatFailure):
            schema = ClientSchema(
                mode="frozen",
                probed=True,
                dropped_params=tuple(dropped),
                notes=tuple(notes),
            )
            frozen_hash = schema.set_frozen_template(client.template)
            if dropped:
                notes.append(
                    "probe 2: provider rejected optional parameter(s) "
                    + ", ".join(dropped)
                    + "; the frozen schema omits them for EVERY call"
                )
                say(f"probe full      : OK after dropping {', '.join(dropped)}")
            else:
                notes.append("probe 2 (full): all optional parameters accepted")
                say("probe full      : OK (all optional parameters accepted)")
            say(f"frozen schema   : {frozen_hash[:16]} keys={sorted(client.template)}")
            return schema

        if probe.error_kind != "bad_request" or not looks_like_param_error(probe.error):
            raise ConfigError(
                "request-schema probe failed with the full parameter set and the error does not "
                f"look parameter-related, so no experimental calls were made. error={probe.error}"
            )
        if len(dropped) >= len(DROPPABLE_PARAMS):
            raise ConfigError(
                "request-schema probe still rejected after removing every optional parameter "
                f"({list(dropped)}); no experimental calls were made. error={probe.error}"
            )
        dropped.append(DROPPABLE_PARAMS[len(dropped)])
        say(f"probe full      : rejected, retrying without {dropped[-1]}")


# --------------------------------------------------------------------------
# One job -> one record
# --------------------------------------------------------------------------


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _base_record(job: Job, config: Config) -> Dict[str, object]:
    return {
        "case_id": job.case.id,
        "condition": job.condition,
        "future_requirement_variant": job.variant_label,
        "run_id": job.run_id,
        "sample_index": job.sample_index,
        "model": config.model,
        "provider": config.provider,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "thinking": config.thinking,
        "started_at": _iso_now(),
    }


def run_job(job: Job, config: Config, client: ChatClient, seq: int) -> Dict[str, object]:
    messages = build_messages(job.case, job.condition)
    record = _base_record(job, config)
    record["seq"] = seq
    record["prompt_sha256"] = prompt_sha256(messages)
    record["schema_hash"] = client.schema_hash

    # One retry pass if the transport succeeded but the model returned nothing.
    content_retries = 0
    total_attempts = 0
    while True:
        outcome = client.chat(messages)
        if isinstance(outcome, ChatFailure):
            total_attempts += outcome.attempts
            record.update(
                raw_response="",
                raw_reasoning="",
                parsed_decision=ERROR,
                reason="",
                parsed_valid_json=False,
                parse_note="",
                reasoning_chars=0,
                error=outcome.error,
                error_kind=outcome.error_kind,
                http_status=outcome.http_status,
                attempts=total_attempts,
                content_retries=content_retries,
                latency_s=round(outcome.latency_s, 3),
                timestamp=_iso_now(),
            )
            return record

        total_attempts += outcome.attempts
        if outcome.text.strip():
            parsed: ParsedResponse = parse_response(outcome.text)
            record.update(
                raw_response=outcome.text,
                raw_reasoning=outcome.reasoning_text,
                parsed_decision=parsed.decision,
                reason=parsed.reason,
                parsed_valid_json=parsed.valid_json,
                parse_note=parsed.note,
                reasoning_chars=len(outcome.reasoning_text or ""),
                error="",
                error_kind="",
                http_status=outcome.http_status,
                model_returned=outcome.model_returned,
                finish_reason=outcome.finish_reason,
                usage=outcome.usage,
                response_id=outcome.response_id,
                system_fingerprint=outcome.system_fingerprint,
                attempts=total_attempts,
                content_retries=content_retries,
                latency_s=round(outcome.latency_s, 3),
                timestamp=_iso_now(),
            )
            return record

        if content_retries >= 2:
            record.update(
                raw_response="",
                raw_reasoning=outcome.reasoning_text,
                parsed_decision=UNPARSED,
                reason="",
                parsed_valid_json=False,
                parse_note="empty content returned repeatedly",
                reasoning_chars=len(outcome.reasoning_text or ""),
                error="",
                error_kind="empty_content",
                http_status=outcome.http_status,
                model_returned=outcome.model_returned,
                finish_reason=outcome.finish_reason,
                usage=outcome.usage,
                attempts=total_attempts,
                content_retries=content_retries,
                latency_s=round(outcome.latency_s, 3),
                timestamp=_iso_now(),
            )
            return record

        content_retries += 1


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_condition: Dict[str, Dict[str, int]] = {}
    schema_hashes = set()
    for record in records:
        condition = str(record["condition"])
        decision = str(record["parsed_decision"])
        bucket = by_condition.setdefault(condition, {})
        bucket[decision] = bucket.get(decision, 0) + 1
        if record.get("schema_hash"):
            schema_hashes.add(str(record["schema_hash"]))
    total = len(records)
    errors = sum(1 for r in records if r["parsed_decision"] == ERROR)
    unparsed = sum(1 for r in records if r["parsed_decision"] == UNPARSED)
    summary = {
        "n_records": total,
        "n_error": errors,
        "n_unparsed": unparsed,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "unparsed_rate": round(unparsed / total, 4) if total else 0.0,
        "schema_hashes_seen": sorted(schema_hashes),
        "schema_consistent": len(schema_hashes) <= 1,
        "decisions_by_condition": by_condition,
    }
    if len(schema_hashes) > 1:
        summary["schema_warning"] = (
            "more than one request schema hash was recorded: request parameters were NOT "
            "constant across samples"
        )
    return summary


def resolve_run_name(args: argparse.Namespace, config: Config) -> str:
    if args.run_name:
        return args.run_name
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = config.model.replace("/", "-")
    return f"{stamp}_{config.provider}_{model}_n{config.samples_per_condition}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.candidate_cases:
        args.cases = args.candidate_cases
    try:
        config = build_config(args)
        cases = load_cases(Path(args.cases))
        if config.limit_cases:
            cases = cases[: config.limit_cases]
        if not args.dry_run:
            config.resolve_api_key(args.env_file)
    except (ConfigError, CaseFileError) as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    jobs = build_jobs(cases, config)
    run_name = resolve_run_name(args, config)
    run_dir = Path(args.results_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(f"[error] run directory already exists: {run_dir} (use --overwrite)", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "raw_results.jsonl"

    print(f"run dir        : {run_dir}")
    print(f"cases          : {len(cases)}  conditions: {len(config.conditions)}  n={config.samples_per_condition}")
    print(f"jobs           : {len(jobs)}")
    print(f"model          : {config.provider}:{config.model} @ {config.chat_url}")
    print(f"sampling       : temperature={config.temperature} top_p={config.top_p} "
          f"max_tokens={config.max_tokens} seed={config.seed} thinking={config.thinking}")
    print(f"  note         : {config.sampling_note()}")
    print(f"order          : {config.order} (seed={config.order_seed})")

    # ---- freeze the request schema BEFORE any experimental call -------------
    rng = random.Random(config.order_seed + 1)
    if args.dry_run:
        schema = ClientSchema(mode="unprobed", notes=("dry run: no probe performed",))
        client = ChatClient(config, schema, rng=rng)
        print("probe          : skipped (dry run)")
    elif args.skip_probe:
        schema = ClientSchema(mode="full", notes=("--skip-probe: schema not verified",))
        client = ChatClient(config, schema, rng=rng)
        print("probe          : SKIPPED (--skip-probe); schema is whatever was passed on the CLI")
    else:
        try:
            schema = freeze_schema(config)
        except (ConfigError, ValueError) as exc:
            print(f"[probe error] {exc}", file=sys.stderr)
            return 2
        client = ChatClient(config, schema, rng=rng)
        # Hard guarantee that the frozen template survived into the client: this is
        # the bug that let the probe's reduced schema be silently re-expanded.
        if schema.frozen_template is None:
            print("[probe error] frozen schema has no frozen_template", file=sys.stderr)
            return 2
        assert client.schema_hash == schema_hash(schema.frozen_template), (
            "the client is not using the frozen request schema"
        )

    frozen_schema_hash = client.schema_hash
    print(f"schema hash    : {frozen_schema_hash[:16]} (recorded on every row)")
    print(f"schema params  : {sorted(client.template)}")

    per_case_audit = {case.id: prompt_audit(case, config.conditions) for case in cases}
    manifest: Dict[str, object] = {
        "run_name": run_name,
        "created_at": _iso_now(),
        "python": sys.version,
        "config": config.manifest(),
        "cases_file": str(Path(args.cases).resolve()),
        "case_ids": [case.id for case in cases],
        "conditions": list(config.conditions),
        "n_jobs": len(jobs),
        "execution_order": config.order,
        "execution_order_sha256": execution_order_hash(jobs),
        "prompt_audit": per_case_audit,
        "request_schema": {**schema.as_dict(), "frozen_schema_hash": frozen_schema_hash},
        "dry_run": bool(args.dry_run),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dump_prompts:
        dump_dir = run_dir / "prompts"
        dump_dir.mkdir(exist_ok=True)
        for case in cases:
            for condition in config.conditions:
                path = dump_dir / f"{case.id}__{condition}.json"
                path.write_text(
                    json.dumps(messages_to_payload(build_messages(case, condition)), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        print(f"prompt dump    : {dump_dir}")

    if args.dry_run:
        print("\n[dry-run] no API calls made. Inspect the prompts with:")
        print(f"  python inspect_prompts.py --cases {args.cases}")
        return 0

    # ---- run ----------------------------------------------------------------
    records: List[Dict[str, object]] = []
    failures = 0
    interrupted = False

    def flush() -> None:
        """Persist what we have; called at every barrier so a crash loses nothing."""
        ordered = sorted(records, key=lambda r: int(r.get("seq", 0)))
        write_jsonl(jsonl_path, ordered)

    try:
        if config.concurrency <= 1:
            for seq, job in enumerate(jobs):
                record = run_job(job, config, client, seq)
                records.append(record)
                failures += record["parsed_decision"] == ERROR
                print(f"[{seq + 1}/{len(jobs)}] {record['run_id']} -> {record['parsed_decision']}", flush=True)
                if (seq + 1) % 20 == 0:
                    flush()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = {pool.submit(run_job, job, config, client, seq): job for seq, job in enumerate(jobs)}
                done = 0
                for future in concurrent.futures.as_completed(futures):
                    done += 1
                    try:
                        record = future.result()
                    except Exception:  # pragma: no cover - defensive
                        job = futures[future]
                        record = _base_record(job, config)
                        record.update(
                            seq=-1,
                            raw_response="",
                            parsed_decision=ERROR,
                            reason="",
                            error=traceback.format_exc()[-500:],
                            error_kind="harness_exception",
                            schema_hash=frozen_schema_hash,
                            timestamp=_iso_now(),
                        )
                    records.append(record)
                    failures += record["parsed_decision"] == ERROR
                    print(f"[{done}/{len(jobs)}] {record['run_id']} -> {record['parsed_decision']}", flush=True)
                    if done % 20 == 0:
                        flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] writing the samples collected so far ...", file=sys.stderr)

    flush()
    records.sort(key=lambda r: (str(r["case_id"]), str(r["condition"]), int(r.get("sample_index", 0))))
    write_csv(run_dir / "raw_results.csv", records)

    summary = summarize(records)
    manifest["summary"] = summary
    manifest["finished_at"] = _iso_now()
    manifest["interrupted"] = interrupted
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {len(records)} rows -> {jsonl_path}")
    print(f"wrote {len(records)} rows -> {run_dir / 'raw_results.csv'}")
    print(
        f"errors={summary['n_error']} ({summary['error_rate']:.1%})  "
        f"unparsed={summary['n_unparsed']} ({summary['unparsed_rate']:.1%})"
    )
    if not summary["schema_consistent"]:
        print(
            "[warning] the request schema changed during the run "
            f"({summary['schema_hashes_seen']}); the run is NOT internally comparable.",
            file=sys.stderr,
        )
    expected = len(jobs)
    if len(records) != expected:
        print(f"[warning] expected {expected} rows, wrote {len(records)} -- samples are missing!", file=sys.stderr)
    print(f"\nnext: python analyze.py --run-dir {run_dir}")
    if interrupted:
        return 130
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
