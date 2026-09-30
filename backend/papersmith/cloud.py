"""Durable storage for the hosted version: papers, passages and files mirrored to Supabase.

The server keeps working from its local disk, which is fast and what every other module uses. This
module copies each change to Supabase a moment later (the latest version of a paper wins, so a burst of
saves while writing becomes one upload) and brings papers back after a restart, when the disk of a free
cloud host starts empty. Only the server talks to Supabase, with the service key; row-level security
keeps the tables closed to everyone else.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time

import httpx

from .config import settings

log = logging.getLogger(__name__)

BUCKET = "papersmith"
FLUSH_EVERY = 1.5          # seconds


def enabled() -> bool:
    return settings.storage == "supabase" and bool(settings.supabase_url and settings.supabase_service_key)


class _Supabase:
    def __init__(self) -> None:
        self.url = settings.supabase_url.rstrip("/")
        self.key = settings.supabase_service_key
        self._pending: dict[tuple[str, str], tuple] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._client = httpx.Client(timeout=60)
        threading.Thread(target=self._run, name="cloud-flush", daemon=True).start()
        atexit.register(self.flush)

    # ---------------------------------------------------------------- plumbing
    def _headers(self, **extra: str) -> dict[str, str]:
        # legacy service_role keys are JWTs and go in both headers; new "sb_secret_..." keys only in apikey
        auth = {} if self.key.startswith("sb_") else {"Authorization": f"Bearer {self.key}"}
        return {"apikey": self.key, **auth, **extra}

    def _rest(self, method: str, table: str, params: dict | None = None, body=None, prefer: str = "") -> httpx.Response:
        headers = self._headers(**({"Prefer": prefer} if prefer else {}))
        resp = self._client.request(method, f"{self.url}/rest/v1/{table}", params=params, json=body, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"Supabase {method} {table}: {resp.status_code} {resp.text[:200]}")
        return resp

    def _run(self) -> None:
        while True:
            self._wake.wait(FLUSH_EVERY)
            self._wake.clear()
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - keep the mirror alive; the next round retries
                log.exception("cloud flush failed")

    def flush(self) -> None:
        with self._lock:
            batch, self._pending = self._pending, {}
        # papers first: their passages and files belong to them
        order = {"project": 0, "passages": 1, "object": 2}
        failed = {}
        for key, payload in sorted(batch.items(), key=lambda kv: order[kv[0][0]]):
            kind = key[0]
            try:
                if kind == "project":
                    self._rest("POST", "ps_projects", body=payload[0], prefer="resolution=merge-duplicates,return=minimal")
                elif kind == "passages":
                    self._rest("POST", "ps_passages", body=payload[0], prefer="resolution=merge-duplicates,return=minimal")
                else:
                    path, data, ctype = payload
                    resp = self._client.post(f"{self.url}/storage/v1/object/{BUCKET}/{path}", content=data,
                                             headers=self._headers(**{"Content-Type": ctype, "x-upsert": "true"}))
                    if resp.status_code >= 400:
                        raise RuntimeError(f"upload {path}: {resp.status_code} {resp.text[:200]}")
            except Exception as exc:  # noqa: BLE001
                log.warning("cloud save of %s failed, will retry: %s", key, exc)
                failed[key] = payload
        if failed:
            with self._lock:
                for key, payload in failed.items():
                    self._pending.setdefault(key, payload)      # a newer version queued meanwhile wins

    def _queue(self, key: tuple[str, str], payload: tuple) -> None:
        with self._lock:
            self._pending[key] = payload
            backlog = len(self._pending)
        if backlog > 20:
            self._wake.set()

    # ---------------------------------------------------------------- papers
    def put_project(self, pid: str, owner: str, summary: dict, data_json: str) -> None:
        row = {"id": pid, "owner": owner or None, "members": list(summary.get("members", {})),
               "summary": summary, "data": json.loads(data_json),
               "updated_at": summary.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._queue(("project", pid), (row,))

    def get_project(self, pid: str) -> dict | None:
        rows = self._rest("GET", "ps_projects", params={"id": f"eq.{pid}", "select": "data"}).json()
        return rows[0]["data"] if rows else None

    def list_projects(self, owner: str) -> list[dict]:
        rows = self._rest("GET", "ps_projects", params={"or": f"(owner.eq.{owner},members.cs.{{{owner}}})",
                                                      "select": "summary", "order": "updated_at.desc"}).json()
        return [r["summary"] for r in rows if r.get("summary")]

    def delete_project(self, pid: str, paths: list[str]) -> None:
        with self._lock:
            self._pending = {k: v for k, v in self._pending.items() if k[1] != pid and not k[1].startswith(f"{pid}/")}
        self._rest("DELETE", "ps_passages", params={"project_id": f"eq.{pid}"})
        self._rest("DELETE", "ps_projects", params={"id": f"eq.{pid}"})
        if paths:
            self._client.request("DELETE", f"{self.url}/storage/v1/object/{BUCKET}", json={"prefixes": paths}, headers=self._headers())

    # ---------------------------------------------------------------- passages
    def put_passages(self, pid: str, passages: list[dict]) -> None:
        self._queue(("passages", pid), ({"project_id": pid, "data": passages},))

    def get_passages(self, pid: str) -> list[dict] | None:
        rows = self._rest("GET", "ps_passages", params={"project_id": f"eq.{pid}", "select": "data"}).json()
        return rows[0]["data"] if rows else None

    # ---------------------------------------------------------------- files
    def put_object(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._queue(("object", path), (path, data, content_type))

    def delete_objects(self, paths: list[str]) -> None:
        with self._lock:
            for p in paths:
                self._pending.pop(("object", p), None)
        try:
            self._client.request("DELETE", f"{self.url}/storage/v1/object/{BUCKET}", json={"prefixes": paths}, headers=self._headers())
        except httpx.HTTPError as exc:
            log.warning("could not delete %s: %s", paths, exc)

    def get_object(self, path: str) -> bytes | None:
        with self._lock:
            pending = self._pending.get(("object", path))
        if pending:
            return pending[1]
        resp = self._client.get(f"{self.url}/storage/v1/object/{BUCKET}/{path}", headers=self._headers())
        return resp.content if resp.status_code == 200 else None


_instance: _Supabase | None = None
_instance_lock = threading.Lock()


def store() -> _Supabase:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = _Supabase()
        return _instance


SCHEMA_SQL = """
create table if not exists public.ps_projects (
  id text primary key,
  owner uuid,
  summary jsonb not null default '{}'::jsonb,
  data jsonb not null,
  updated_at timestamptz not null default now()
);
create index if not exists ps_projects_owner_idx on public.ps_projects (owner, updated_at desc);
alter table public.ps_projects add column if not exists members uuid[] not null default '{}';
create index if not exists ps_projects_members_idx on public.ps_projects using gin (members);
create table if not exists public.ps_passages (
  project_id text primary key,
  data jsonb not null
);
alter table public.ps_projects enable row level security;
alter table public.ps_passages enable row level security;
insert into storage.buckets (id, name, public) values ('papersmith', 'papersmith', false) on conflict (id) do nothing;
"""
