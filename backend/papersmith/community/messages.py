"""/api/community/messages: private conversations between connected people.

One conversation per pair of people, stored as one document (the latest MAX_KEPT messages). Only the two people
in it can read it; college admins and the owner cannot. Sending needs a connection: removing someone from your
connections stops new messages from them, and the history stays readable to both.

The recipient is notified once when a conversation gets new messages (not for every message), and the page
polls for new ones while it is open.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import auth
from ..models import now_iso
from . import core, matching
from .store import store

router = APIRouter(prefix="/api/community/messages")

MAX_KEPT = 500
MAX_CHARS = 2000
DAILY_LIMIT = 500
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def thread_id(a: str, b: str) -> str:
    return "__".join(sorted((a, b)))


def _lock(tid: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(tid, threading.Lock())


def _connected(a: str, b: str) -> bool:
    return b in store().network(a)["connections"]


def _other_view(uid: str) -> dict:
    p = store().get_person(uid)
    if not p:
        return {"uid": uid, "name": "Researcher", "role_label": "", "department": "", "institution": ""}
    return matching.public_view(p, core.institution_name(p.get("institution", "")))


def _unread(t: dict, me: str) -> int:
    seen = t.get("read", {}).get(me, 0)
    return sum(1 for m in t.get("messages", []) if m["n"] > seen and m["from"] != me)


def _message_view(m: dict, me: str) -> dict:
    return {"n": m["n"], "text": m["text"], "at": m["at"], "mine": m["from"] == me}


@router.get("")
def conversations():
    """My conversations, newest first, with the last message and how many are unread."""
    me = core.me_uid()
    out = []
    for t in store().list_docs("thread", member=me, limit=200):
        other = next((u for u in t["members"] if u != me), me)
        last = t["messages"][-1] if t.get("messages") else None
        out.append({"with": _other_view(other), "last": _message_view(last, me) if last else None,
                    "unread": _unread(t, me), "at": t.get("last_at", "")})
    out.sort(key=lambda c: c["at"], reverse=True)
    return {"conversations": out, "unread": sum(c["unread"] for c in out)}


@router.get("/{uid}")
def conversation(uid: str, after: int = 0):
    """Messages with one person (only those after `after` when polling), and mark them read."""
    me = core.me_uid()
    if uid == me:
        raise HTTPException(400, "That's you.")
    connected = _connected(me, uid)
    tid = thread_id(me, uid)
    s = store()
    t = s.get_doc("thread", tid)
    if t is None and not connected:
        raise HTTPException(404, "You can message people you're connected with.")
    msgs = (t or {}).get("messages", [])
    if t and msgs and t.get("read", {}).get(me, 0) < msgs[-1]["n"]:
        with _lock(tid):
            t = s.get_doc("thread", tid) or t
            t.setdefault("read", {})[me] = t["messages"][-1]["n"]
            s.put_doc("thread", t)
            msgs = t["messages"]
    return {"with": _other_view(uid), "connected": connected,
            "messages": [_message_view(m, me) for m in msgs if m["n"] > after][-200:],
            "their_read": (t or {}).get("read", {}).get(uid, 0)}


class MessageIn(BaseModel):
    text: str


@router.post("/{uid}")
def send(uid: str, body: MessageIn):
    me = core.me_uid()
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Write a message first.")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"Messages can be up to {MAX_CHARS} characters.")
    if uid == me or not _connected(me, uid):
        raise HTTPException(403, "You can message people you're connected with.")
    if not auth.spend("message_dm", DAILY_LIMIT):
        raise HTTPException(429, "You've sent a lot of messages today. Try again tomorrow.")
    tid = thread_id(me, uid)
    s = store()
    with _lock(tid):
        t = s.get_doc("thread", tid) or {"id": tid, "members": sorted((me, uid)), "institution": "", "owner": "",
                                         "messages": [], "read": {}, "seq": 0}
        was_unread = _unread(t, uid)
        t["seq"] = t.get("seq", 0) + 1
        t["messages"].append({"n": t["seq"], "from": me, "text": text, "at": now_iso()})
        t["messages"] = t["messages"][-MAX_KEPT:]
        t.setdefault("read", {})[me] = t["seq"]           # what I send, I've seen
        t["last_at"] = now_iso()
        s.put_doc("thread", t)
    if not was_unread:                                   # one notification per batch of unread messages
        preview = text if len(text) <= 70 else text[:67] + "…"
        core.notify(uid, "message", f"{core.display_name(me)} sent you a message: “{preview}”", link=f"/messages/{me}", actor=me)
    return {"message": _message_view(t["messages"][-1], me), "their_read": t["read"].get(uid, 0)}
