"""Background jobs: reading uploads, answering chat, writing. Several kinds may run at once per project;
the same kind (and key, e.g. one file) may not run twice."""

from __future__ import annotations

import secrets
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .models import now_iso


@dataclass
class Job:
    id: str
    project_id: str
    kind: str
    key: str = ""
    status: str = "queued"        # queued | running | done | error
    progress: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


class JobConflict(RuntimeError):
    pass


def running(project_id: str, kind: str | None = None, key: str | None = None) -> list[Job]:
    return [j for j in _jobs.values() if j.project_id == project_id and j.status in ("queued", "running")
            and (kind is None or j.kind == kind) and (key is None or j.key == key)]


def start(project_id: str, kind: str, fn: Callable[[Job], None], key: str = "") -> Job:
    with _lock:
        if running(project_id, kind, key):
            raise JobConflict(f"a {kind} job is already running for this project")
        job = Job(id=secrets.token_hex(5), project_id=project_id, kind=kind, key=key)
        _jobs[job.id] = job

    def run() -> None:
        job.status = "running"
        try:
            fn(job)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished_at = now_iso()

    threading.Thread(target=run, daemon=True).start()
    return job


def get(job_id: str) -> Job:
    return _jobs[job_id]


def for_project(project_id: str) -> list[Job]:
    return sorted((j for j in _jobs.values() if j.project_id == project_id), key=lambda j: j.started_at, reverse=True)
