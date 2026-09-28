"""Runtime settings, read from environment variables (or a .env file at the repo root)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def save_env_value(key: str, value: str) -> None:
    """Set KEY=value in the repo's .env (replacing an existing line, commented or not) and in this process."""
    env_file = REPO_ROOT / ".env"
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    out, done = [], False
    for line in lines:
        if not done and line.lstrip("# ").strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


# Keep model downloads next to the project (C: on this machine has little free space).
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PAPERSMITH_DATA", REPO_ROOT / "data")))
    default_backend: str = os.environ.get("PAPERSMITH_BACKEND", "ollama")   # ollama | groq | claude | mock

    ollama_url: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_num_ctx: int = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

    claude_model: str = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    claude_effort: str = os.environ.get("CLAUDE_EFFORT", "medium")

    # any OpenAI-compatible API; Groq by default (GROQ_API_KEY)
    groq_url: str = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1")
    # one model per job: each has its own free-tier allowance, so the chat never waits for the writer
    groq_model: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")                # writes paragraphs
    groq_read_model: str = os.environ.get("GROQ_READ_MODEL", "openai/gpt-oss-20b")    # reads files into facts
    groq_agent_model: str = os.environ.get("GROQ_AGENT_MODEL", "openai/gpt-oss-120b") # the mentor
    groq_fallback_model: str = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")  # if one is retired
    groq_effort: str = os.environ.get("GROQ_EFFORT", "low")                            # reasoning effort for writing
    groq_agent_effort: str = os.environ.get("GROQ_AGENT_EFFORT", "medium")
    max_rate_wait: float = float(os.environ.get("MAX_RATE_WAIT", "75"))               # seconds to wait out a rate limit

    # the mentor: an LLM that runs the conversation. auto = on for API models, off for the small local model
    agent: str = os.environ.get("PAPERSMITH_AGENT", "auto")    # auto | on | off

    # hosted version: papers mirrored to Supabase, sign-in required, each person sees only their own papers
    storage: str = os.environ.get("PAPERSMITH_STORAGE", "local")          # local | supabase
    auth: str = os.environ.get("PAPERSMITH_AUTH", "none")                 # none | supabase
    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    supabase_anon_key: str = os.environ.get("SUPABASE_ANON_KEY", "")
    supabase_service_key: str = os.environ.get("SUPABASE_SERVICE_KEY", "")
    cors_origins: str = os.environ.get("CORS_ORIGINS", "http://localhost:5174,http://127.0.0.1:5174")
    # fair use of one shared model quota: per person, per day
    daily_messages: int = int(os.environ.get("DAILY_MESSAGES", "150"))
    daily_uploads: int = int(os.environ.get("DAILY_UPLOADS", "40"))
    max_papers: int = int(os.environ.get("MAX_PAPERS", "25"))
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "60"))

    @property
    def hosted(self) -> bool:
        return self.auth == "supabase"

    nli_enabled: bool = _bool("PAPERSMITH_NLI", True)
    nli_model: str = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    nli_device: str = os.environ.get("NLI_DEVICE", "auto")   # auto | cpu | cuda
    nli_backend: str = os.environ.get("NLI_BACKEND", "local")   # local (DeBERTa) | llm (API model judges; small servers)
    entail_threshold: float = float(os.environ.get("ENTAIL_THRESHOLD", "0.5"))
    contradiction_threshold: float = float(os.environ.get("CONTRADICTION_THRESHOLD", "0.5"))

    max_rewrite_attempts: int = int(os.environ.get("MAX_REWRITE_ATTEMPTS", "2"))

    @property
    def projects_dir(self) -> Path:
        path = self.data_dir / "projects"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
