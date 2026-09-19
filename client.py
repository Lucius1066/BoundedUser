"""Minimal OpenAI-compatible chat client built on the standard library only.

No `openai` / `requests` dependency, so the experiment runs on a bare Python.
Every failure mode is surfaced to the caller instead of being swallowed: the
caller (run_experiment.py) records an ERROR row so no sample is ever dropped.

Concurrency and schema contract
-------------------------------
A ChatClient instance is **immutable during a run**:

  * the parameter set comes from `schema.frozen_template` when a frozen schema is
    supplied, and is rebuilt from the Config only for probe calls (before freezing).
    A reduced schema discovered by the probe is therefore *actually* used by every
    experimental call -- it cannot be silently re-expanded from the original config.
  * retries always start from that template and build a fresh payload dict, so no
    request can observe or mutate another request's parameters, including requests
    running in other threads.
  * if the server rejects an optional parameter mid-run, the client does NOT learn
    it: the error is returned and recorded, rather than making later samples differ
    from earlier ones.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from config import Config
from prompts import Message, messages_to_payload


def schema_hash(payload: Mapping[str, object]) -> str:
    """Hash of a request *shape*: keys and non-message values.

    Two requests with the same schema hash were sent with the same parameter set,
    which is what makes a cross-condition comparison legitimate.
    """
    shape = {key: value for key, value in payload.items() if key != "messages"}
    blob = json.dumps(shape, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ApiError(RuntimeError):
    """A call that failed after exhausting retries (or failed non-retryably)."""

    def __init__(self, message: str, status: Optional[int] = None, kind: str = "api"):
        super().__init__(message)
        self.status = status
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return False


class RetryableApiError(ApiError):
    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        kind: str = "api",
        permanent: bool = False,
    ):
        super().__init__(message, status=status, kind=kind)
        # permanent=True: the body itself is rejected, so re-sending it identically
        # would just burn retries.  The client never edits the schema to rescue it.
        self.permanent = permanent

    @property
    def retryable(self) -> bool:
        return not self.permanent


@dataclass
class ChatResult:
    text: str = ""
    reasoning_text: str = ""
    model_returned: str = ""
    finish_reason: str = ""
    usage: Dict[str, object] = field(default_factory=dict)
    http_status: int = 0
    latency_s: float = 0.0
    attempts: int = 0
    request_payload: Dict[str, object] = field(default_factory=dict)
    response_id: str = ""
    system_fingerprint: str = ""

    @property
    def schema_hash(self) -> str:
        return schema_hash(self.request_payload)


@dataclass
class ChatFailure:
    error: str
    error_kind: str = "api"
    http_status: int = 0
    attempts: int = 0
    latency_s: float = 0.0
    request_payload: Dict[str, object] = field(default_factory=dict)

    @property
    def schema_hash(self) -> str:
        return schema_hash(self.request_payload)


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}


def _classify_http_error(status: int, body: str) -> ApiError:
    snippet = body.strip().replace("\n", " ")[:400]
    if status in (401, 403):
        return ApiError(f"auth error {status}: {snippet}", status=status, kind="auth")
    if status in (400, 404, 422):
        # Permanent either way: the body is what it is, and the client will not
        # rewrite it mid-run.  A parameter complaint is reported verbatim so the
        # operator can decide before the next run.
        return RetryableApiError(
            f"bad-request {status}: {snippet}", status=status, kind="bad_request", permanent=True
        )
    if status in _RETRYABLE_STATUS:
        return RetryableApiError(f"http {status}: {snippet}", status=status, kind="http")
    return ApiError(f"http {status}: {snippet}", status=status, kind="http")


def looks_like_param_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "parameter",
            "unsupported",
            "unrecognized",
            "not supported",
            "unknown field",
            "invalid_request_error",
            "extra fields",
        )
    )


# --------------------------------------------------------------------------
# Request schema
# --------------------------------------------------------------------------


@dataclass
class ClientSchema:
    """The frozen request parameter set for a run, plus how it was established.

    `frozen_template` is the authoritative parameter set (no `messages` key).  When
    it is present, `ChatClient` uses it verbatim; when it is absent (probe calls
    before freezing), the client derives the template from the Config.
    """

    mode: str = "frozen"            # "frozen" | "full" | "unprobed"
    probed: bool = False
    dropped_params: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    frozen_template: Optional[Dict[str, object]] = None

    #: parameter names that must NOT appear in a frozen template
    _FORBIDDEN = ("thinking", "reasoning_effort", "seed", "top_p")

    def set_frozen_template(self, template: Mapping[str, object]) -> str:
        """Record the accepted parameter set and verify it matches `dropped_params`."""
        frozen = copy.deepcopy({k: v for k, v in template.items() if k != "messages"})
        still_present = [name for name in self._FORBIDDEN if name in frozen and name in self.dropped_params]
        if still_present:
            raise ValueError(
                "frozen template still contains parameter(s) the probe rejected: "
                f"{still_present}"
            )
        self.frozen_template = frozen
        return schema_hash(frozen)

    def as_dict(self) -> Dict[str, object]:
        data = {
            "mode": self.mode,
            "probed": self.probed,
            "dropped_params": list(self.dropped_params),
            "notes": list(self.notes),
            "frozen_template_hash": (
                schema_hash(self.frozen_template) if self.frozen_template is not None else None
            ),
            "frozen_template_keys": (
                sorted(self.frozen_template) if self.frozen_template is not None else None
            ),
        }
        return data


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class ChatClient:
    """Synchronous chat client with a frozen request schema and retrying calls."""

    def __init__(
        self,
        config: Config,
        schema: Optional[ClientSchema] = None,
        rng: Optional[random.Random] = None,
    ):
        self.config = config
        self.schema = schema or ClientSchema(mode="unprobed")
        self.rng = rng or random.Random()
        self._template = self._resolve_template()

    # -- schema -------------------------------------------------------------

    def _resolve_template(self) -> Dict[str, object]:
        """Frozen template from the schema, else derived from the Config (probes)."""
        if self.schema.frozen_template is not None:
            return copy.deepcopy(self.schema.frozen_template)
        return self._template_from_config()

    def _template_from_config(self) -> Dict[str, object]:
        cfg = self.config
        payload: Dict[str, object] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        payload.update(cfg.thinking_param())
        if cfg.reasoning_effort and cfg.thinking == "enabled":
            payload["reasoning_effort"] = cfg.reasoning_effort
        for key, value in (cfg.extra_body or {}).items():
            payload[key] = value
        return payload

    @property
    def schema_hash(self) -> str:
        """Hash of the frozen parameter set, without the messages."""
        return schema_hash(self._template)

    @property
    def template(self) -> Dict[str, object]:
        return copy.deepcopy(self._template)

    def build_request(self, messages: Sequence[Message]) -> Dict[str, object]:
        """A fresh request body.  Callers must not mutate the returned template."""
        payload = dict(self._template)
        payload["messages"] = messages_to_payload(messages)
        return payload

    # -- calls --------------------------------------------------------------

    def chat(self, messages: Sequence[Message]) -> "ChatResult | ChatFailure":
        """Call the endpoint.  Never raises for API problems: returns ChatFailure."""
        cfg = self.config
        started = time.perf_counter()
        attempts = 0
        last_error: Optional[ApiError] = None
        last_payload: Dict[str, object] = {}

        while attempts < max(1, cfg.max_retries):
            attempts += 1
            payload = self.build_request(messages)  # fresh copy per attempt
            last_payload = payload
            try:
                result = self._post(payload)
                result.attempts = attempts
                result.latency_s = time.perf_counter() - started
                return result
            except ApiError as exc:
                last_error = exc
                if not exc.retryable or attempts >= max(1, cfg.max_retries):
                    break
                self._sleep_before_retry(attempts)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = ApiError(f"{type(exc).__name__}: {exc}", kind="unexpected")
                if attempts >= max(1, cfg.max_retries):
                    break
                self._sleep_before_retry(attempts)

        assert last_error is not None
        return ChatFailure(
            error=str(last_error),
            error_kind=last_error.kind,
            http_status=last_error.status or 0,
            attempts=attempts,
            latency_s=time.perf_counter() - started,
            request_payload=last_payload,
        )

    # -- internals ----------------------------------------------------------

    def _sleep_before_retry(self, attempt: int) -> None:
        cfg = self.config
        delay = min(cfg.backoff_cap_s, cfg.backoff_base_s ** attempt)
        delay *= 0.75 + 0.5 * self.rng.random()  # jitter, avoids lock-step retries
        time.sleep(delay)

    def _post(self, payload: Dict[str, object]) -> ChatResult:
        cfg = self.config
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            cfg.chat_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
                "User-Agent": "llm-user-sim-phenomenon-check/0.3",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
                status = response.status
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise _classify_http_error(exc.code, body or str(exc)) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise RetryableApiError(f"network error: {exc}", kind="network") from None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise RetryableApiError(
                f"non-JSON response body: {raw[:300]!r}", status=status, kind="bad_response"
            ) from None

        if isinstance(parsed, dict) and parsed.get("error"):
            message = json.dumps(parsed["error"], ensure_ascii=False)[:400]
            raise RetryableApiError(f"server returned error object: {message}", kind="api_error")

        try:
            choice = parsed["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise RetryableApiError(
                f"unexpected response shape: {json.dumps(parsed, ensure_ascii=False)[:300]}",
                status=status,
                kind="bad_response",
            ) from None

        # DeepSeek returns the chain of thought in `reasoning_content` next to
        # `content`.  We never parse the decision out of it, but we keep it so a
        # human can see whether the ignored requirement shows up there.
        reasoning = ""
        if isinstance(message, dict):
            candidate = message.get("reasoning_content") or message.get("reasoning") or ""
            if isinstance(candidate, str):
                reasoning = candidate

        return ChatResult(
            text=content or "",
            reasoning_text=reasoning,
            model_returned=str(parsed.get("model", "")),
            finish_reason=str(choice.get("finish_reason", "")),
            usage=parsed.get("usage") or {},
            http_status=status,
            request_payload=payload,
            response_id=str(parsed.get("id", "")),
            system_fingerprint=str(parsed.get("system_fingerprint", "")),
        )
