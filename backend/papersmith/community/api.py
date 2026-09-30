"""/api/community: profiles, college verification, suggestions, mentors, connections and the research map.

Everything is scoped to the signed-in person's college for now (a local community inside each college)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import auth
from . import matching, profiles, verify
from .store import key_of, store

router = APIRouter(prefix="/api/community")
ROLES = set(matching.ROLES)


def me_uid() -> str:
    uid = auth.current_user.get()
    if not auth.enabled() or not uid:
        raise HTTPException(404, "The community is part of the hosted PaperSmith.")
    return uid


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
    college = str(meta.get("college", "")).strip()
    if college:
        fields["institution"] = s.upsert_institution(college[:120])["key"]
    return s.upsert_person(uid, **fields)


def _institution_name(key: str) -> str:
    inst = store().institution(key) if key else None
    return inst["name"] if inst else ""


def _own_view(p: dict) -> dict:
    return {**matching.public_view(p, _institution_name(p.get("institution", ""))),
            "published": bool(p.get("published")), "visibility": p.get("visibility", "community"),
            "institution_key": p.get("institution", ""), "email_domain": p.get("email_domain", "")}


def display_name(uid: str) -> str:
    p = store().get_person(uid)
    return (p or {}).get("name") or auth.metadata(uid).get("name") or "Researcher"


# ---------------------------------------------------------------- colleges (public: used while signing up)

@router.get("/institutions")
def institutions(q: str = ""):
    return [{"key": i["key"], "name": i["name"], "members": i.get("members", 0)} for i in store().search_institutions(q)]


# ---------------------------------------------------------------- my profile

@router.get("/me")
def get_me():
    uid = me_uid()
    p = ensure_person(uid)
    complete = bool(p.get("name") and p.get("role") and p.get("institution"))
    return {"person": _own_view(p), "complete": complete, "verify_available": verify.available()}


class ProfileIn(BaseModel):
    name: str | None = None
    role: str | None = None
    department: str | None = None
    college: str | None = None
    bio: str | None = None
    field: str | None = None
    topics: list[str] | None = None
    methods: list[str] | None = None
    needs: list[str] | None = None
    published: bool | None = None
    visibility: str | None = None


@router.put("/me")
def put_me(body: ProfileIn):
    uid = me_uid()
    current = ensure_person(uid)
    s = store()
    fields: dict = {}
    for k in ("name", "department", "bio", "field"):
        v = getattr(body, k)
        if v is not None:
            fields[k] = " ".join(v.split())[:240 if k == "bio" else 80]
    if body.role is not None:
        if body.role not in ROLES:
            raise HTTPException(400, "Choose a role.")
        fields["role"] = body.role
    if body.college is not None and body.college.strip():
        key = key_of(body.college)
        if key != current.get("institution"):
            fields["institution"] = s.upsert_institution(body.college.strip()[:120])["key"]
            fields["verified"] = False      # a new college needs its own verification
    if body.topics is not None:
        fields["topics"] = matching.as_items(body.topics, 8)
    if body.methods is not None:
        fields["methods"] = matching.as_items(body.methods, 8)
    if body.needs is not None:
        fields["needs"] = [{"text": " ".join(n.split())[:120]} for n in body.needs if n.strip()][:4]
    if body.published is not None:
        fields["published"] = body.published
    if body.visibility is not None:
        if body.visibility not in ("community", "hidden"):
            raise HTTPException(400, "visibility must be community or hidden")
        fields["visibility"] = body.visibility
    return {"person": _own_view(s.upsert_person(uid, **fields))}


@router.post("/me/draft")
def draft_me():
    uid = me_uid()
    ensure_person(uid)
    try:
        return profiles.draft(uid)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


class VerifyStart(BaseModel):
    email: str


class VerifyCode(BaseModel):
    code: str


@router.post("/verify/start")
def verify_start(body: VerifyStart):
    uid = me_uid()
    ensure_person(uid)
    try:
        college = verify.start(uid, body.email)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"sent": True, "college": college}


@router.post("/verify/confirm")
def verify_confirm(body: VerifyCode):
    uid = me_uid()
    try:
        person = verify.confirm(uid, body.code)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"person": _own_view(person)}


# ---------------------------------------------------------------- people

def _college_people(me: dict) -> list[dict]:
    return store().members(me["institution"]) if me.get("institution") else []


@router.get("/suggestions")
def suggestions():
    uid = me_uid()
    me = ensure_person(uid)
    if not me.get("institution"):
        return {"collaborators": [], "mentors": [], "reason": "no_college"}
    ranked = matching.rank(me, _college_people(me), store().network(uid), _institution_name(me["institution"]))
    ranked["reason"] = "" if (me.get("topics") or me.get("methods") or me.get("needs")) else "no_topics"
    return ranked


@router.get("/people/{uid}")
def person(uid: str):
    me = ensure_person(me_uid())
    s = store()
    p = s.get_person(uid)
    network = s.network(me["uid"])
    connected = uid in network["connections"]
    if not p or (not connected and (p.get("institution") != me.get("institution") or not matching.visible(p))):
        raise HTTPException(404, "person not found")
    return {"person": matching.public_view(p, _institution_name(p.get("institution", ""))),
            "status": matching._status(uid, network)}


class ConnectIn(BaseModel):
    to: str
    message: str = ""


@router.post("/connect")
def connect(body: ConnectIn):
    me = ensure_person(me_uid())
    other = store().get_person(body.to)
    if not other or body.to == me["uid"] or other.get("institution") != me.get("institution") or not matching.visible(other):
        raise HTTPException(404, "person not found")
    store().request(me["uid"], body.to, body.message)
    return network()


@router.post("/requests/{from_uid}/accept")
def accept(from_uid: str):
    uid = me_uid()
    if not store().accept(from_uid, uid):
        raise HTTPException(404, "request not found")
    return network()


@router.post("/requests/{from_uid}/decline")
def decline(from_uid: str):
    store().decline(from_uid, me_uid())
    return network()


@router.delete("/connections/{uid}")
def disconnect(uid: str):
    store().disconnect(me_uid(), uid)
    return network()


@router.get("/network")
def network():
    uid = me_uid()
    s = store()
    raw = s.network(uid)

    def view(other: str) -> dict:
        p = s.get_person(other)
        return matching.public_view(p, _institution_name(p.get("institution", ""))) if p else {"uid": other, "name": "Researcher"}

    return {"connections": [view(u) for u in raw["connections"]],
            "incoming": [{"person": view(r["from"]), "message": r.get("message", ""), "at": r.get("at", "")} for r in raw["incoming"]],
            "outgoing": [{"person": view(r["to"]), "message": r.get("message", ""), "at": r.get("at", "")} for r in raw["outgoing"]]}


@router.get("/graph")
def research_map():
    me = ensure_person(me_uid())
    if not me.get("institution"):
        return {"people": [], "topics": [], "links": [], "connections": [], "institution": ""}
    s = store()
    data = matching.graph(s.members(me["institution"]), s.connections_among(me["institution"]), _institution_name(me["institution"]))
    return {**data, "institution": _institution_name(me["institution"]), "me": me["uid"]}
