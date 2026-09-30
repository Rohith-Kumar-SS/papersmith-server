"""Project store: one JSON file per project under data/projects/, plus each project's files.

On a laptop that is all. In the hosted version the same files act as a fast local cache and every change
is mirrored to Supabase (see cloud.py), so papers survive a restart of the server.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import cloud
from .config import settings
from .models import Project, now_iso

_lock = threading.RLock()
_owners: dict[str, str] = {}                # project id -> owner, for the per-request access check
_members: dict[str, dict[str, str]] = {}    # project id -> {user id: "author" | "reviewer"}


def _path(project_id: str) -> Path:
    if not project_id.isalnum():
        raise ValueError("invalid project id")
    return settings.projects_dir / f"{project_id}.json"


def new_id() -> str:
    return secrets.token_hex(6)


def summary_of(data: dict) -> dict:
    """The row a paper shows in the list of papers."""
    ledger = data.get("ledger") or {}
    draft = data.get("draft") or {}
    words = sum(len(str(s.get("text", "")).split()) for p in draft.get("paragraphs", []) for s in p.get("sentences", []))
    per_page = {"ieee": 850, "acm": 800, "article": 500}.get(ledger.get("template", "ieee"), 700)
    return {
        "id": data["id"],
        "name": data.get("name", ""),
        "title": ledger.get("title", ""),
        "claims": len(ledger.get("claims", [])),
        "has_draft": bool(data.get("draft")),
        "updated_at": data.get("updated_at", ""),
        "files": len(data.get("sources", [])),
        "pages": round(words / per_page, 1),
        "stage": data.get("stage", ""),
        "owner": data.get("owner", ""),
        "members": {m["uid"]: m.get("role", "author") for m in data.get("members", []) if m.get("uid")},
    }


def _remember(project: Project) -> None:
    _owners[project.id] = project.owner
    _members[project.id] = {m.uid: m.role for m in project.members}


def save(project: Project) -> Project:
    with _lock:
        project.updated_at = now_iso()
        text = project.model_dump_json(indent=1)
        tmp = _path(project.id).with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(_path(project.id))
        _remember(project)
        if cloud.enabled():
            data = json.loads(text)
            cloud.store().put_project(project.id, project.owner, summary_of(data), text)
    return project


def load(project_id: str) -> Project:
    with _lock:
        path = _path(project_id)
        if not path.exists() and cloud.enabled():
            data = cloud.store().get_project(project_id)       # a paper from before the server restarted
            if data:
                path.write_text(json.dumps(data), encoding="utf-8")
        if not path.exists():
            raise KeyError(project_id)
        project = Project.model_validate_json(path.read_text(encoding="utf-8"))
        _remember(project)
        return project


@contextmanager
def edit(project_id: str) -> Iterator[Project]:
    """Load, mutate and save a project atomically with respect to other editors."""
    with _lock:
        project = load(project_id)
        yield project
        save(project)


def owner_of(project_id: str) -> str | None:
    """The project's owner, or None if there is no such project."""
    if project_id in _owners:
        return _owners[project_id]
    try:
        return load(project_id).owner
    except (KeyError, ValueError):
        return None


def access_of(project_id: str, uid: str) -> str | None:
    """'owner', 'author' (co-author), 'reviewer', or None when this person may not see the project."""
    owner = owner_of(project_id)
    if owner is None:
        return None
    if owner == uid:
        return "owner"
    return _members.get(project_id, {}).get(uid)


# ---------------------------------------------------------------- files (uploads, figures)

def files_dir(project_id: str, create: bool = False) -> Path:
    """A project's files on this machine."""
    if not project_id.isalnum():
        raise ValueError("invalid project id")
    path = settings.data_dir / "files" / project_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _file_path(project_id: str, rel: str) -> Path:
    if not rel or ".." in Path(rel).parts or Path(rel).is_absolute():
        raise ValueError("invalid file name")
    return files_dir(project_id) / rel


def write_file(project_id: str, rel: str, data: bytes) -> Path:
    path = _file_path(project_id, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if cloud.enabled():
        cloud.store().put_object(f"{project_id}/{rel}", data, mimetypes.guess_type(rel)[0] or "application/octet-stream")
    return path


def local_file(project_id: str, rel: str) -> Path:
    """Path of a project file, fetched back from the cloud first if this machine does not have it."""
    path = _file_path(project_id, rel)
    if not path.exists() and cloud.enabled():
        data = cloud.store().get_object(f"{project_id}/{rel}")
        if data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return path


def remove_file(project_id: str, rel: str) -> None:
    _file_path(project_id, rel).unlink(missing_ok=True)
    if cloud.enabled():
        cloud.store().delete_objects([f"{project_id}/{rel}"])


def project_files(project: Project) -> list[str]:
    """Every file a project owns, relative to its folder."""
    return [f"uploads/{s.stored_as}" for s in project.sources if s.stored_as] + \
           [f.filename for f in project.ledger.figures if f.filename]


def delete(project_id: str) -> None:
    with _lock:
        try:
            paths = project_files(load(project_id))
        except (KeyError, ValueError):
            paths = []
        _path(project_id).unlink(missing_ok=True)
        folder = files_dir(project_id)
        if folder.is_dir():
            shutil.rmtree(folder)
        _owners.pop(project_id, None)
        _members.pop(project_id, None)
        if cloud.enabled():
            cloud.store().delete_project(project_id, [f"{project_id}/{p}" for p in paths])


def list_all(owner: str | None = None) -> list[dict]:
    """Summaries of the papers, newest first; with an owner, only theirs (and in the hosted version, also
    the ones not yet loaded on this machine since it restarted)."""
    items: dict[str, dict] = {}
    if cloud.enabled() and owner:
        items = {s["id"]: s for s in cloud.store().list_projects(owner)}
    for path in sorted(settings.projects_dir.glob("*.json")):
        if not path.stem.isalnum():
            continue                        # "<id>.sources.json" holds a paper's passages, not a paper
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "id" not in data:
            continue
        s = summary_of(data)
        if owner is None or s["owner"] == owner or owner in s["members"]:
            items[s["id"]] = s              # this machine's copy is the newest
    for s in items.values():
        s["my_role"] = "owner" if owner is None or s.get("owner") == owner else s.get("members", {}).get(owner, "author")
    return sorted(items.values(), key=lambda x: x["updated_at"], reverse=True)
