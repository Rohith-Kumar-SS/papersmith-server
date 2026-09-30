"""What every part of the community shares: who is asking, their college, admin rights, notifications and
the model calls. Kept apart from the routers so openings, questions and college dashboards can all use it."""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time

from fastapi import HTTPException

from .. import auth
from ..config import settings
from ..llm import BackendError, get_backend
from ..models import now_iso
from . import directory
from .store import key_of, store

log = logging.getLogger("papersmith.community")

ROLES = {"ug": "UG student", "pg": "PG student", "phd": "PhD scholar", "faculty": "Faculty / Professor",
         "researcher": "Researcher", "staff": "Research office / staff"}
MENTOR_ROLES = {"faculty", "researcher", "phd"}
STUDENT_ROLES = {"ug", "pg", "phd"}


def new_id(prefix: str = "") -> str:
    return prefix + secrets.token_hex(5)


# ---------------------------------------------------------------- who is asking

def me_uid() -> str:
    uid = auth.current_user.get()
    if not auth.enabled() or not uid:
        raise HTTPException(404, "The community is part of the hosted PaperSmith.")
    return uid


def is_owner() -> bool:
    return settings.is_owner(auth.current_email.get())


def require_owner() -> str:
    uid = me_uid()
    if not is_owner():
        raise HTTPException(403, "Only the people who run this PaperSmith can open this page.")
    return uid


# ---------------------------------------------------------------- colleges

def college_from_directory(entry: dict) -> dict:
    """The graph node for a listed college, created the first time someone joins it."""
    s = store()
    inst = s.institution(entry["id"])
    if inst is None:
        s.upsert_institution(entry["name"], key=entry["id"])
        inst = s.update_institution(entry["id"], kind=entry["kind"], state=entry["state"], district=entry["district"],
                                    openalex=entry["openalex"], ror=entry["ror"], website=entry["domain"], listed=True)
    return inst or {}


def resolve_college(college_id: str | None, name: str | None) -> str:
    """Institution key for what someone picked (a directory id) or typed (an exact directory name, or else a new,
    unlisted college that the owner confirms later)."""
    if college_id:
        entry = directory.get(college_id)
        if entry:
            return college_from_directory(entry)["key"]
        if store().institution(college_id):
            return college_id                   # an unlisted college someone already created
        raise HTTPException(400, "Choose your college from the list.")
    name = " ".join((name or "").split())[:120]
    if not name:
        return ""
    for hit in directory.search(name, 3):
        if directory.norm(hit["name"]) == directory.norm(name) or directory.norm(hit["name"].split(",")[0]) == directory.norm(name):
            return college_from_directory(hit)["key"]
    key = "X-" + key_of(name).replace(" ", "-")[:60]
    s = store()
    if s.institution(key) is None:
        s.upsert_institution(name, key=key)
        s.update_institution(key, kind="X", listed=False)
    return key


NEW_KEY = re.compile(r"^(U|C|S|R|X|OA)-")


def migrate_old_colleges() -> int:
    """Colleges stored before the directory existed (keyed by their typed name) move onto their directory entry,
    or become an unlisted 'X-' college, so everyone from one college shares one community. Members keep their
    profiles and verification; admins and email domains move with them. Safe to run again."""
    s = store()
    moved = 0
    for inst in s.all_institutions():
        old = inst["key"]
        if NEW_KEY.match(old):
            continue
        entry = next((h for h in directory.search(inst["name"], 3)
                      if directory.norm(h["name"]) == directory.norm(inst["name"])
                      or directory.norm(h["name"].split(",")[0]) == directory.norm(inst["name"])), None)
        if entry:
            target = college_from_directory(entry)["key"]
        else:
            target = "X-" + key_of(inst["name"]).replace(" ", "-")[:60]
            if s.institution(target) is None:
                s.upsert_institution(inst["name"], key=target)
                s.update_institution(target, kind="X", listed=False)
        for d in inst.get("domains", []):
            s.upsert_institution(inst["name"], domain=d, key=target)
        if inst.get("admins"):
            s.update_institution(target, admins=sorted(set(admins_of(target)) | set(inst["admins"])))
        for p in s.members(old):
            s.upsert_person(p["uid"], institution=target)
            moved += 1
        s.delete_institution(old)
        forget_name(old)
        log.info("college %r moved to %s", old, target)
    return moved


_names: dict[str, tuple[float, str]] = {}


def institution_name(key: str) -> str:
    if not key:
        return ""
    hit = _names.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    inst = store().institution(key)
    name = inst["name"] if inst else ""
    _names[key] = (time.time(), name)
    return name


def forget_name(key: str) -> None:
    _names.pop(key, None)


def departments(key: str) -> list[str]:
    inst = store().institution(key) if key else None
    return (inst or {}).get("departments") or directory.DEPARTMENTS


def ensure_person(uid: str) -> dict:
    """The signed-in person's node, created on first visit from what they entered at sign-up."""
    s = store()
    person = s.get_person(uid)
    if person is not None:
        return person
    meta = auth.metadata(uid)
    fields = {"name": str(meta.get("name", "")).strip()[:80] or auth.current_email.get().split("@")[0],
              "role": meta.get("role") if meta.get("role") in ROLES else "",
              "department": str(meta.get("department", "")).strip()[:80]}
    try:
        college = resolve_college(str(meta.get("college_id", "")).strip() or None, str(meta.get("college", "")))
    except HTTPException:
        college = resolve_college(None, str(meta.get("college", "")))
    if college:
        fields["institution"] = college
    return s.upsert_person(uid, **fields)


def display_name(uid: str) -> str:
    p = store().get_person(uid)
    return (p or {}).get("name") or auth.metadata(uid).get("name") or "Researcher"


def admins_of(key: str) -> list[str]:
    inst = store().institution(key) if key else None
    return list((inst or {}).get("admins") or [])


def is_admin(uid: str, key: str) -> bool:
    return bool(key) and uid in admins_of(key)


def require_admin() -> tuple[dict, str]:
    """(me, my college key) for a college admin; the owner may act as admin of any college they belong to."""
    me = ensure_person(me_uid())
    key = me.get("institution", "")
    if not key:
        raise HTTPException(400, "Join a college first.")
    if not (is_admin(me["uid"], key) or is_owner()):
        raise HTTPException(403, "Only your college's admins can open this.")
    return me, key


def same_college(me: dict, doc: dict) -> None:
    if not doc or doc.get("institution") != me.get("institution"):
        raise HTTPException(404, "not found")


# ---------------------------------------------------------------- notifications

MAX_NOTIFICATIONS = 60


def notify(uid: str, kind: str, text: str, link: str = "", actor: str = "") -> None:
    """An in-app notification (the bell). Never fails the action that caused it."""
    if not uid or uid == actor:
        return
    try:
        store().put_doc("notification", {"id": new_id("n"), "owner": uid, "kind": kind, "text": text[:240],
                                         "link": link, "actor": actor, "at": now_iso(), "read": False})
    except Exception:  # noqa: BLE001
        log.exception("notification failed")


def notifications(uid: str) -> list[dict]:
    s = store()
    items = s.list_docs("notification", owner=uid, limit=MAX_NOTIFICATIONS + 40)
    for old in items[MAX_NOTIFICATIONS:]:
        s.delete_doc("notification", old["id"])
    return items[:MAX_NOTIFICATIONS]


def mark_read(uid: str, ids: list[str] | None = None) -> None:
    s = store()
    for n in s.list_docs("notification", owner=uid, limit=MAX_NOTIFICATIONS):
        if not n.get("read") and (ids is None or n["id"] in ids):
            s.put_doc("notification", {**n, "read": True})


# ---------------------------------------------------------------- the model

def ask(system: str, user: str, schema: dict, max_tokens: int = 1500, role: str = "read") -> dict:
    """One structured model call; a friendly ValueError when the model can't be reached."""
    try:
        return get_backend(role=role).generate_json(system, user, schema, max_tokens=max_tokens)
    except BackendError as exc:
        log.info("community model call failed: %s", exc)
        raise ValueError(f"PaperSmith's model is busy right now ({exc}). Try again in a minute.") from exc


def in_background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()
