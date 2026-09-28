"""Pluggable LLM backends.

Every backend exposes generate_json(system, user, schema) and returns a dict
matching the JSON schema. Structured output is enforced server-side where the
backend supports it (Ollama `format`, Claude `output_config.format`, Groq strict `json_schema`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import httpx

from .config import settings

log = logging.getLogger(__name__)


class BackendError(RuntimeError):
    pass


def is_fatal(exc: Exception) -> bool:
    """Errors a retry cannot fix (service unreachable, bad key, quota used up): stop the job, keep the work."""
    text = str(exc)
    return any(k in text for k in ("reach", "API key", "usage limit"))


class LLMBackend(Protocol):
    name: str
    model: str

    def generate_json(self, system: str, user: str, schema: dict, **opts) -> dict: ...

    def status(self) -> dict: ...


# Jobs register a callback here so a wait for an API rate limit shows up in the chat ("waiting 12 s ...").
_waits = threading.local()


@contextmanager
def report_waits(fn):
    prev = getattr(_waits, "fn", None)
    _waits.fn = fn
    try:
        yield
    finally:
        _waits.fn = prev


def _report_wait(seconds: float) -> None:
    fn = getattr(_waits, "fn", None)
    if fn:
        try:
            fn(seconds)
        except Exception:  # noqa: BLE001 - reporting must never break a model call
            pass


# ---------------------------------------------------------------- Ollama (local)

class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or settings.ollama_model
        self.url = (url or settings.ollama_url).rstrip("/")

    def generate_json(self, system: str, user: str, schema: dict, **_opts) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": _bound_arrays(schema),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # 4096 tokens covers one paragraph's prompt + output and keeps a 7B model fully in 6 GB of VRAM
            "options": {"temperature": 0.2, "num_ctx": settings.ollama_num_ctx, "num_predict": 1500},
        }
        try:
            resp = httpx.post(f"{self.url}/api/chat", json=payload, timeout=300)
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach Ollama at {self.url}: {exc}") from exc
        if resp.status_code != 200:
            raise BackendError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if data.get("done_reason") == "length":
            raise BackendError("model output hit the length cap (it was repeating itself)")
        return _parse_json(data.get("message", {}).get("content", ""))

    def status(self) -> dict:
        try:
            resp = httpx.get(f"{self.url}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError):
            return {"available": False, "detail": f"Ollama not reachable at {self.url}", "models": []}
        has_model = any(m == self.model or m.startswith(self.model + ":") for m in models)
        detail = "ready" if has_model else f"model '{self.model}' not pulled (run: ollama pull {self.model})"
        return {"available": has_model, "detail": detail, "models": models}


# ---------------------------------------------------------------- Claude (API)

class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str | None = None, effort: str | None = None):
        self.model = model or settings.claude_model
        self.effort = effort or settings.claude_effort
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict, **_opts) -> dict:
        import anthropic

        client = self._get_client()
        try:
            # fallbacks="default": if a safety classifier declines, the API
            # re-runs the request on Anthropic's recommended fallback model.
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.AuthenticationError as exc:
            raise BackendError("Claude API key invalid or missing (set ANTHROPIC_API_KEY)") from exc
        except anthropic.RateLimitError as exc:
            raise BackendError("Claude API rate limited, try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError("Cannot reach the Claude API (network)") from exc

        if response.stop_reason == "refusal":
            raise BackendError("Claude declined this request")
        if response.stop_reason == "max_tokens":
            raise BackendError("Claude output hit max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return _parse_json(text)

    def status(self) -> dict:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return {"available": False, "detail": "anthropic package not installed"}
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return {"available": True, "detail": f"ready ({self.model})"}
        if (Path.home() / ".config" / "anthropic").is_dir():   # `ant auth login` profile, read by the SDK
            return {"available": True, "detail": f"ready via ant profile ({self.model})"}
        return {"available": False, "detail": "set ANTHROPIC_API_KEY in .env (or run `ant auth login`) to enable"}


# ---------------------------------------------------------------- OpenAI-compatible APIs (Groq by default)

def _seconds(text: str | None) -> float | None:
    """'7.66s', '1m2.5s', '120ms', '2h3m4s' or '12' -> seconds."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    total, found = 0.0, False
    for value, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", text):
        found = True
        total += float(value) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if found else None


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


class _Budget:
    """What the API last said about a model's token allowance, so a request it would refuse waits instead."""

    def __init__(self) -> None:
        self.remaining: int | None = None
        self.reset_at = 0.0
        self.limit: int | None = None
        self.lock = threading.Lock()


_budgets: dict[str, _Budget] = {}
_budgets_guard = threading.Lock()


def _budget(key: str) -> _Budget:
    with _budgets_guard:
        return _budgets.setdefault(key, _Budget())


class OpenAICompatBackend:
    """Chat-completions APIs in the OpenAI format. Configured for Groq; any compatible provider works with
    GROQ_URL / GROQ_MODEL / GROQ_API_KEY pointed at it."""

    name = "groq"
    label = "Groq"

    def __init__(self, model: str | None = None, url: str | None = None, effort: str | None = None,
                 key_env: str = "GROQ_API_KEY", fallback: str | None = None):
        self.model = model or settings.groq_model
        self.url = (url or settings.groq_url).rstrip("/")
        self.effort = effort or settings.groq_effort
        self.key_env = key_env
        self.fallback = fallback if fallback and fallback != self.model else None   # if the provider retires the model
        # strict json_schema (constrained decoding) where the model supports it; downgraded on refusal
        self.mode = "strict" if any(k in self.model for k in ("gpt-oss", "qwen3")) else "object"

    @property
    def _key(self) -> str:
        return os.environ.get(self.key_env, "").strip()

    # ---------------------------------------------------------------- rate limits
    def expected_wait(self, estimate: int) -> float:
        """Seconds a request of about `estimate` tokens would wait for this model's per-minute allowance."""
        b = _budgets.get(self.url + "|" + self.model)
        if not b:
            return 0.0
        with b.lock:
            if b.remaining is None or b.remaining >= estimate:
                return 0.0
            return max(0.0, b.reset_at - time.time())

    def _throttle(self, estimate: int) -> None:
        b = _budget(self.url + "|" + self.model)
        with b.lock:
            wait = b.reset_at - time.time() if b.remaining is not None and b.remaining < estimate else 0.0
        if wait > 0:
            wait = min(wait, settings.max_rate_wait)
            _report_wait(wait)
            time.sleep(wait + 0.25)

    def _note(self, headers: httpx.Headers) -> None:
        b = _budget(self.url + "|" + self.model)
        remaining, reset = headers.get("x-ratelimit-remaining-tokens"), _seconds(headers.get("x-ratelimit-reset-tokens"))
        limit = headers.get("x-ratelimit-limit-tokens")
        with b.lock:
            if remaining is not None and remaining.isdigit():
                b.remaining = int(remaining)
                b.reset_at = time.time() + (reset or 0)
            if limit and limit.isdigit():
                b.limit = int(limit)

    def _post(self, payload: dict, estimate: int) -> httpx.Response:
        if not self._key:
            raise BackendError(f"{self.label} API key missing (set {self.key_env} in .env)")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        server_errors = 0
        for _ in range(10):
            self._throttle(estimate)
            try:
                resp = httpx.post(f"{self.url}/chat/completions", json=payload, headers=headers, timeout=180)
            except httpx.TimeoutException:
                server_errors += 1
                if server_errors > 2:
                    raise BackendError(f"{self.label} did not answer in time") from None
                continue
            except httpx.HTTPError as exc:
                raise BackendError(f"Cannot reach {self.label} ({self.url}): {exc}") from exc
            self._note(resp.headers)
            if resp.status_code == 429:
                message = _error_message(resp)
                wait = _seconds(resp.headers.get("retry-after"))
                m = re.search(r"try again in ([\dhms.]+)", message)
                if m:
                    wait = max(wait or 0, _seconds(m.group(1)) or 0)
                wait = wait if wait is not None else 10
                if wait > settings.max_rate_wait:
                    per_day = "per day" in message or "(TPD)" in message or "(RPD)" in message
                    raise BackendError(f"{self.label} usage limit reached ({'daily' if per_day else 'rate'} limit of the "
                                       f"free tier); it frees up in about {_human(wait)}")
                log.info("%s rate limited, waiting %.1fs: %s", self.model, wait, message[:160])
                _report_wait(wait)
                time.sleep(wait + 0.5)
                continue
            if resp.status_code >= 500:
                server_errors += 1
                if server_errors > 3:
                    return resp
                time.sleep(2 * server_errors)
                continue
            return resp
        raise BackendError(f"{self.label} kept refusing the request (rate limits); try again in a minute")

    # ---------------------------------------------------------------- JSON generation
    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int = 3000,
                      effort: str | None = None, temperature: float = 0.3, **_opts) -> dict:
        estimate = int((len(system) + len(user)) / 3.3) + min(max_tokens, 1500)
        garbled_retries = 0
        for _ in range(5):
            sys_text = system
            payload: dict = {"model": self.model, "temperature": temperature, "max_completion_tokens": max_tokens}
            if "gpt-oss" in self.model:
                payload["reasoning_effort"] = effort or self.effort
            if self.mode == "strict":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "output", "strict": True, "schema": _strict_schema(schema)}}
            elif self.mode == "loose":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "output", "schema": _strip_bounds(schema)}}
            else:
                payload["response_format"] = {"type": "json_object"}
                sys_text = f"{system}\n\nReturn only a JSON object matching this JSON schema:\n{json.dumps(_strip_bounds(schema))}"
            payload["messages"] = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
            resp = self._post(payload, estimate)
            if resp.status_code in (401, 403):
                raise BackendError(f"{self.label} API key invalid or missing (set {self.key_env} in .env)")
            if resp.status_code == 413:
                raise BackendError(f"request too large for the {self.label} free tier ({_error_message(resp)[:160]})")
            if self.fallback and resp.status_code in (400, 404) and re.search(
                    r"decommission|does not exist|model_not_found|model .{0,60}not found", _error_message(resp), re.I):
                log.warning("%s is not available (%s); using %s", self.model, _error_message(resp)[:120], self.fallback)
                self.model, self.fallback = self.fallback, None
                self.mode = "strict" if any(k in self.model for k in ("gpt-oss", "qwen3")) else "object"
                estimate = int((len(system) + len(user)) / 3.3) + min(max_tokens, 1500)
                continue
            if resp.status_code == 400:
                message = _error_message(resp)
                if "json_validate_failed" in resp.text or "failed to generate" in message.lower():
                    raise BackendError(f"model output did not match the JSON schema ({message[:120]})")
                if self.mode != "object" and re.search(r"schema|response_format|json", message, re.I):
                    log.info("%s: %s mode refused (%s), downgrading", self.model, self.mode, message[:200])
                    self.mode = "loose" if self.mode == "strict" else "object"
                    continue
                raise BackendError(f"{self.label} error 400: {message[:300]}")
            if resp.status_code != 200:
                raise BackendError(f"{self.label} error {resp.status_code}: {_error_message(resp)[:300]}")
            choice = (resp.json().get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                raise BackendError("model output hit the length cap")
            data = _parse_json(_plain(_ungarble((choice.get("message") or {}).get("content") or "")))
            if _garbled(data) and garbled_retries < 2:
                # now and then a response arrives with µ ² ³ turned into control characters: ask again
                garbled_retries += 1
                log.info("%s returned garbled characters; asking again", self.model)
                continue
            return _clean_strings(data)
        raise BackendError(f"{self.label} would not accept the output format")

    def status(self) -> dict:
        if not self._key:
            return {"available": False, "detail": f"set {self.key_env} in .env to enable"}
        b = _budgets.get(self.url + "|" + self.model)
        extra = f", {b.remaining} of {b.limit} tokens/min left" if b and b.limit and b.remaining is not None else ""
        return {"available": True, "detail": f"ready ({self.model}{extra})"}


_TYPOGRAPHY = str.maketrans({"‐": "-", "‑": "-", " ": " ", " ": " ", " ": " "})
# control characters never belong in model text (tab, newline and carriage return do)
_GARBLED = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _ungarble(text: str) -> str:
    """Last resort for a response whose µ ² ³ arrived as control characters (the high bits lost):
    a control character before a unit letter was µ ("µg/m³"), \\x02 / \\x03 otherwise were ² / ³."""
    if not _GARBLED.search(text):
        return text
    text = re.sub(r"[\x02\x05](?=[gmsLlMV]\b|[gm]/|g\b|m\b)", "µ", text)
    text = text.replace("\x02", "²").replace("\x03", "³").replace("\x01", "¹")
    return _GARBLED.sub("", text)


def _garbled(data) -> bool:
    if isinstance(data, str):
        return bool(_GARBLED.search(data))
    if isinstance(data, dict):
        return any(_garbled(v) for v in data.values())
    if isinstance(data, list):
        return any(_garbled(v) for v in data)
    return False


def _clean_strings(data):
    """Every string in a parsed response, ungarbled and with plain typography."""
    if isinstance(data, str):
        return _plain(_ungarble(data))
    if isinstance(data, dict):
        return {k: _clean_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_clean_strings(v) for v in data]
    return data


def _plain(text: str) -> str:
    """Models like non-breaking hyphens and thin spaces ("5‑fold"); they break search and the fact checks."""
    return text.translate(_TYPOGRAPHY)


def _error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return str(err.get("message") or err) if isinstance(err, dict) else str(err)
    except ValueError:
        return resp.text[:400]


def _strip_bounds(schema):
    """Schema without maxItems/minItems (not every provider accepts them). An array that must stay empty
    (maxItems 0, e.g. citations for a paragraph with none) can only hold empty strings instead."""
    if isinstance(schema, list):
        return [_strip_bounds(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _strip_bounds(v) for k, v in schema.items() if k not in ("maxItems", "minItems")}
    if out.get("type") == "array" and schema.get("maxItems") == 0:
        out["items"] = {"type": "string", "enum": [""]}
    return out


def _strict_schema(schema):
    """Strict mode wants every object closed and every property required."""
    out = _strip_bounds(schema)

    def close(node):
        if isinstance(node, list):
            return [close(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = {k: close(v) for k, v in node.items()}
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        return node

    return close(out)


# ---------------------------------------------------------------- Mock (offline, deterministic)

class MockBackend:
    """Restates each claim as its own sentence. No model, fully deterministic.

    Useful for tests, demos without a GPU, and as a lower bound: zero ideation,
    minimal fluency.
    """

    name = "mock"
    model = "restate-v1"

    _CLAIM_RE = re.compile(r"^\[(C\d+)\] \((\w+)\) (.*?)(?: \{refs: ([^}]*)\})?$")

    _PASSAGE_RE = re.compile(r"^\[([^\]]+)\] \(([^)]*)\) (.*)$")

    def generate_json(self, system: str, user: str, schema: dict, **_opts) -> dict:
        props = schema.get("properties", {})
        if "reply" in props:               # the mentor: a scripted stand-in for tests and demos
            return self._mentor(user)
        if "claims" in props:              # extraction: each sentence or bullet of each passage, verbatim
            from .extract import guess_type
            blocks: list[tuple[str, list[str]]] = []
            for line in user.splitlines():
                m = self._PASSAGE_RE.match(line.strip())
                if m:
                    blocks.append((m.group(1), [m.group(3)]))
                elif blocks and line.strip() and not line.startswith("List the claims"):
                    blocks[-1][1].append(line.strip())       # passages span several lines (slides, bullets)
            claims = []
            for pid, lines in blocks:
                for piece in lines:
                    piece = re.sub(r"^(speaker notes:\s*)", "", piece.strip(" -•"), flags=re.I)
                    for sent in re.split(r"(?<=[.!?])\s+", piece):
                        sent = sent.strip(" -•").strip()
                        if len(sent.split()) >= 5:
                            claims.append({"text": sent.rstrip("."), "type": guess_type(sent).value, "passages": [pid]})
            return {"claims": claims[:25]}
        if "answer" in props:              # grounded Q&A: the first excerpt
            m = re.search(r"^\(([^)]*)\) (.+)$", user, re.M)
            return {"answer": m.group(2)[:300] if m else "", "found": bool(m)}
        sentences = []
        for line in user.splitlines():
            m = self._CLAIM_RE.match(line.strip())
            if not m:
                continue
            cid, _ctype, text, refs = m.groups()
            text = text.strip()
            if text and text[-1] not in ".!?":
                text += "."
            text = text[0].upper() + text[1:] if text else text
            ref_keys = [r.strip() for r in refs.split(",")] if refs else []
            sentences.append({"text": text, "claim_ids": [cid], "ref_keys": ref_keys, "signpost": False})
        return {"sentences": sentences}

    def _mentor(self, prompt: str) -> dict:
        m = re.search(r"^Reply to: (M\d+)", prompt, re.M)
        mid = m.group(1) if m else ""
        line = re.search(rf"^{mid} (?:researcher|event): (.*)$", prompt, re.M) if mid else None
        text = line.group(1).strip() if line else ""
        low = text.lower()
        if re.match(r"^(please )?(write|draft)\b", low):
            return {"reply": "Writing it now.", "facts": [], "options": [],
                    "actions": [{"kind": "write", "target": "", "value": ""}]}
        if "review" in low:
            if "PAPER TEXT" not in prompt:
                return {"reply": "", "facts": [], "options": [], "actions": [{"kind": "read_paper", "target": "", "value": ""}]}
            return {"reply": "Review: the paper states your results clearly.", "facts": [], "options": [], "actions": []}
        facts = []
        is_event = bool(line) and line.group(0).startswith(f"{mid} event")
        if text and not text.endswith("?") and not is_event:
            from .extract import guess_type
            for sent in re.split(r"(?<=[.!?])\s+", text):
                sent = sent.strip().rstrip(".")
                if len(sent.split()) >= 4:
                    facts.append({"text": sent, "type": guess_type(sent).value, "source": mid})
        reply = f"Noted {len(facts)} point{'s' if len(facts) != 1 else ''}. What else should the paper say?" if facts \
            else "Tell me about your work, or drop in your files."
        return {"reply": reply, "facts": facts, "actions": [], "options": []}

    def status(self) -> dict:
        return {"available": True, "detail": "always available (restates claims verbatim)"}


# ---------------------------------------------------------------- helpers

def _bound_arrays(schema: dict, max_items: int = 40) -> dict:
    """Copy of the schema with maxItems on every array. Arrays of enum values are capped at the enum size,
    which stops small models looping inside them ("C4", "C5", "C4", "C5", ...) under grammar decoding."""
    if isinstance(schema, list):
        return [_bound_arrays(s, max_items) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _bound_arrays(v, max_items) for k, v in schema.items()}
    if out.get("type") == "array" and "maxItems" not in out:
        enum = out.get("items", {}).get("enum") if isinstance(out.get("items"), dict) else None
        out["maxItems"] = len(enum) if enum else max_items
    return out


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise BackendError(f"Model did not return valid JSON: {text[:200]}")


def get_backend(name: str | None = None, model: str | None = None, role: str = "write") -> LLMBackend:
    """role: "write" (paragraphs, answers) or "read" (extracting facts from files); API backends may use a
    different model for each so their rate limits do not collide."""
    name = (name or settings.default_backend).lower()
    if name == "ollama":
        return OllamaBackend(model=model or None)
    if name == "groq":
        if model:
            return OpenAICompatBackend(model=model)
        if role == "read":
            return OpenAICompatBackend(model=settings.groq_read_model, fallback=settings.groq_fallback_model)
        return OpenAICompatBackend(model=settings.groq_model, fallback=settings.groq_fallback_model)
    if name == "claude":
        return ClaudeBackend(model=model or None)
    if name == "mock":
        return MockBackend()
    raise BackendError(f"Unknown backend '{name}'")


def is_local(name: str | None = None) -> bool:
    """One GPU: model calls to a local backend are serialised; API backends run concurrently."""
    return (name or settings.default_backend).lower() == "ollama"


def agent_backend() -> LLMBackend | None:
    """The model that runs the conversation, or None for the rule-based assistant (the small local model
    cannot hold a mentor conversation, so it only extracts and writes)."""
    mode = settings.agent.lower()
    name = settings.default_backend.lower()
    if mode == "off":
        return None
    if name == "groq":
        return OpenAICompatBackend(model=settings.groq_agent_model or settings.groq_model, effort=settings.groq_agent_effort,
                                   fallback=settings.groq_fallback_model)
    if name == "claude":
        return ClaudeBackend()
    return get_backend(name) if mode == "on" else None


def all_status() -> dict:
    return {
        "ollama": {"model": settings.ollama_model, **OllamaBackend().status()},
        "groq": {"model": settings.groq_model, **OpenAICompatBackend().status()},
        "claude": {"model": settings.claude_model, **ClaudeBackend().status()},
        "mock": {"model": MockBackend.model, **MockBackend().status()},
        "default": settings.default_backend,
        "mentor": bool(agent_backend()),
    }
