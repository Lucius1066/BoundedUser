"""Provider / model / API-key configuration and experiment parameters.

This module is intentionally free of any experiment logic: it only resolves
*where* to call and *with what sampling parameters*.  Experiment logic lives in
prompts.py / run_experiment.py.

Resolution order for every setting:
    explicit CLI argument  >  real environment variable  >  .env file  >  default

The API key is never written to any result file.  Only the key *source* is
recorded in the manifest so a run stays auditable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Provider presets (OpenAI-compatible /chat/completions endpoints only)
# --------------------------------------------------------------------------

PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "dashscope": {  # Alibaba Cloud Qwen, OpenAI-compatible mode
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "custom": {
        "base_url": "",
        "model": "",
        "key_env": "OPENAI_API_KEY",
    },
}


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration (never for API failures)."""


# --------------------------------------------------------------------------
# .env loading (kept dependency-free on purpose: python-dotenv is not required)
# --------------------------------------------------------------------------


def load_dotenv(path: str | os.PathLike[str], override: bool = False) -> Dict[str, str]:
    """Parse a minimal .env file.  Returns the key/value pairs it contained.

    Supported syntax: ``KEY=value``, optional ``export`` prefix, optional single
    or double quotes, ``#`` comments on their own line.  Malformed lines are
    skipped silently rather than crashing a long run.
    """
    loaded: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return loaded
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    # --- provider -----------------------------------------------------------
    provider: str = "deepseek"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    api_key_source: str = "unset"

    # --- sampling parameters (MUST stay identical across all conditions) ----
    temperature: float = 1.0
    # None = do not send top_p.  Reason: DeepSeek fixes top_p at 1.0 in
    # non-thinking mode and clamps it to 0.95-1.0 in thinking mode, so sending a
    # custom value would record a knob that has no effect.  Set it explicitly only
    # if your provider actually honours it.
    top_p: Optional[float] = None
    max_tokens: int = 512
    seed: Optional[int] = None  # sent only if not None; recorded either way
    # On DeepSeek thinking mode is ON by default and silently ignores temperature
    # (and pins top_p).  Default here is "disabled" so that temperature is a real
    # control; --thinking enabled records a warning instead of pretending.
    thinking: str = "disabled"  # "disabled" | "enabled" | "omit"
    reasoning_effort: str = ""  # only sent when thinking == "enabled"

    # --- request behaviour --------------------------------------------------
    timeout_s: float = 60.0
    max_retries: int = 6
    backoff_base_s: float = 1.5
    backoff_cap_s: float = 30.0
    concurrency: int = 4
    extra_body: Dict[str, object] = field(default_factory=dict)

    # --- experiment shape ---------------------------------------------------
    samples_per_condition: int = 20
    order: str = "shuffled"  # "shuffled" | "grouped"
    order_seed: int = 20260101
    conditions: tuple = ()  # empty = all conditions known to prompts.py

    # --- io -----------------------------------------------------------------
    cases_path: str = "cases.json"
    results_dir: str = "results"
    run_name: str = ""
    limit_cases: int = 0  # 0 = no limit

    def __post_init__(self) -> None:
        preset = PROVIDERS.get(self.provider)
        if preset is None:
            raise ConfigError(
                f"unknown provider {self.provider!r}; known: {sorted(PROVIDERS)}"
            )
        if not self.base_url:
            self.base_url = preset["base_url"]
        if not self.model:
            self.model = preset["model"]
        if not self.base_url or not self.model:
            raise ConfigError(
                f"provider {self.provider!r} needs an explicit --base-url and --model"
            )
        self.base_url = self.base_url.rstrip("/")

    # -- key handling --------------------------------------------------------

    def resolve_api_key(self, dotenv_path: str | os.PathLike[str] | None = None) -> str:
        """Find the API key and remember where it came from (never its value)."""
        if self.api_key:
            self.api_key_source = "explicit"
            return self.api_key

        if dotenv_path is not None:
            load_dotenv(dotenv_path)

        candidates = []
        key_env = PROVIDERS.get(self.provider, {}).get("key_env", "")
        if key_env:
            candidates.append(key_env)
        # Generic fallbacks, so switching provider does not require renaming vars.
        for name in ("OPENAI_API_KEY", "LLM_API_KEY", "DSH_LLM_API_KEY"):
            if name not in candidates:
                candidates.append(name)

        for name in candidates:
            value = os.environ.get(name, "").strip()
            if value:
                self.api_key = value
                self.api_key_source = f"env:{name}"
                return self.api_key

        raise ConfigError(
            "no API key found. Set one of "
            + ", ".join(candidates)
            + " in your environment or in experiment/.env "
            + "(see .env.example). Nothing was called."
        )

    # -- request shape -------------------------------------------------------

    def thinking_param(self) -> Dict[str, object]:
        """The `thinking` value to merge into the request body ({} = omit)."""
        if self.thinking == "omit":
            return {}
        return {"thinking": {"type": self.thinking}}

    def sampling_note(self) -> str:
        """Plain statement of which sampling knobs are actually effective."""
        parts: List[str] = []
        if self.thinking == "enabled":
            parts.append(
                "WARNING: thinking mode is enabled. On DeepSeek this silently ignores "
                "temperature and only honours top_p within 0.95-1.0, so the recorded "
                "temperature is documentation only, not an effective control."
            )
        else:
            parts.append(
                "thinking mode disabled: temperature is an effective control. "
                "top_p is NOT effective on DeepSeek in either mode"
                + (" (not sent)" if self.top_p is None else f" (sent as {self.top_p} but ignored)")
                + "."
            )
        if self.seed is None:
            parts.append("seed is not sent, so sampling is not reproducible sample-by-sample.")
        return " ".join(parts)

    # -- logging -------------------------------------------------------------

    def manifest(self) -> Dict[str, object]:
        """Config snapshot for the run manifest.  Secret-free."""
        data = asdict(self)
        data.pop("api_key", None)
        data["api_key_present"] = bool(self.api_key)
        data["chat_completions_url"] = self.chat_url
        data["sampling_note"] = self.sampling_note()
        return data

    @property
    def chat_url(self) -> str:
        # Accept base URLs given with or without the /v1 suffix.
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"
