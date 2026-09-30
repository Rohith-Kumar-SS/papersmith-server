"""Sign-in for the hosted version: Supabase issues the session; the server checks it and scopes every
request to the person who made it. Off (PAPERSMITH_AUTH=none) on a laptop, where there is one user.

PAPERSMITH_AUTH=dev is for local testing only: a token "dev:<id>:<name>" signs in as that made-up person,
so several people can be tried on one machine without real accounts."""

from __future__ import annotations

import contextvars
import datetime as dt
import threading
import time

import httpx

from .config import settings

current_user: contextvars.ContextVar[str] = contextvars.ContextVar("current_user", default="")
current_email: contextvars.ContextVar[str] = contextvars.ContextVar("current_email", default="")

_cache: dict[str, tuple[str, str, float]] = {}      # token -> (user id, email, expires at)
_meta: dict[str, dict] = {}                          # user id -> what they entered at sign-up
_cache_lock = threading.Lock()
CACHE_SECONDS = 300


def enabled() -> bool:
    return settings.auth in ("supabase", "dev")


def metadata(uid: str) -> dict:
    return dict(_meta.get(uid, {}))


def user_for(token: str) -> tuple[str, str] | None:
    """(user id, email) for a Supabase access token, or None when it is missing, expired or forged."""
    if not token:
        return None
    if settings.auth == "dev":
        parts = token.split(":", 2)
        if len(parts) == 3 and parts[0] == "dev" and parts[1].isalnum():
            _meta.setdefault(parts[1], {"name": parts[2]})
            return parts[1], f"{parts[1]}@dev.local"
        return None
    now = time.time()
    with _cache_lock:
        hit = _cache.get(token)
        if hit and hit[2] > now:
            return hit[0], hit[1]
    try:
        resp = httpx.get(f"{settings.supabase_url.rstrip('/')}/auth/v1/user", timeout=15,
                         headers={"apikey": settings.supabase_anon_key or settings.supabase_service_key,
                                  "Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    uid, email = str(data.get("id", "")), str(data.get("email", ""))
    if not uid:
        return None
    with _cache_lock:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[token] = (uid, email, now + CACHE_SECONDS)
        _meta[uid] = dict(data.get("user_metadata") or {})
    return uid, email


# ---------------------------------------------------------------- fair use

_usage: dict[tuple[str, str, str], int] = {}          # (user, kind, day) -> count
_usage_lock = threading.Lock()


def spend(kind: str, limit: int, amount: int = 1) -> bool:
    """Count one use of `kind` for the current user today; False when that would pass the daily limit."""
    user = current_user.get()
    if not enabled() or not user or limit <= 0:
        return True
    key = (user, kind, dt.date.today().isoformat())
    with _usage_lock:
        used = _usage.get(key, 0)
        if used + amount > limit:
            return False
        _usage[key] = used + amount
    return True
