"""/api/community: profiles, college verification, suggestions, mentors, connections, your research map,
what PaperSmith learned about your interests, your published papers, and notifications.

Everything is scoped to the signed-in person's college for now (a local community inside each college).
Members see their own network; the whole college graph is for its admins (college.py)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import core, directory, interests, matching, profiles, publications, verify
from .questions import credits
from .store import store

router = APIRouter(prefix="/api/community")
ROLES = set(core.ROLES)

def _warm_up() -> None:
    directory.size()                      # load the college directory while the server wakes, not on the first keystroke
    try:
        core.migrate_old_colleges()
    except Exception:  # noqa: BLE001 - retried at the next start
        core.log.exception("moving old colleges onto the directory failed")


if core.settings.hosted:
    core.in_background(_warm_up)

me_uid = core.me_uid
ensure_person = core.ensure_person
display_name = core.display_name


def _own_view(p: dict) -> dict:
    key = p.get("institution", "")
    view = matching.public_view(p, core.institution_name(key))
    entry = directory.get(key) if key else None
    inst = store().institution(key) if key and not entry else None
    return {**view, "published": bool(p.get("published")), "visibility": p.get("visibility", "community"),
            "institution_key": key, "institution_place": entry["place"] if entry else "",
            "institution_listed": bool(entry) or bool((inst or {}).get("listed")), "email_domain": p.get("email_domain", ""),
            "scholar": p.get("scholar") or {}}


# ---------------------------------------------------------------- colleges (public: used while signing up)

_counts: tuple[float, dict] = (0.0, {})


def _member_counts() -> dict[str, int]:
    global _counts
    if time.time() - _counts[0] > 60:
        _counts = (time.time(), store().member_counts())
    return _counts[1]


@router.get("/institutions")
def institutions(q: str = ""):
    counts = _member_counts()
    out = [{"id": e["id"], "name": e["name"], "place": e["place"], "kind_label": e["kind_label"],
            "members": counts.get(e["id"], 0), "listed": True} for e in directory.search(q, 10, boost=counts)]
    if q.strip():
        # colleges people added that aren't in the directory yet
        for i in store().search_institutions(q, 5):
            if i["key"].startswith("X-") and not i.get("merged_into") and all(o["id"] != i["key"] for o in out):
                out.append({"id": i["key"], "name": i["name"], "place": "", "kind_label": "Added by members",
                            "members": i.get("members", 0), "listed": False})
    return out


@router.get("/institutions/{inst_id}/departments")
def institution_departments(inst_id: str):
    return core.departments(inst_id)


# ---------------------------------------------------------------- my profile

@router.get("/me")
def get_me():
    uid = me_uid()
    p = ensure_person(uid)
    key = p.get("institution", "")
    complete = bool(p.get("name") and p.get("role") and key)
    return {"person": _own_view(p), "complete": complete, "verify_available": verify.available(),
            "is_admin": core.is_admin(uid, key), "is_owner": core.is_owner(),
            "learned": interests.learned(p), "credits": credits(uid, key) if key else {"helped": 0, "topics": []}}


class ProfileIn(BaseModel):
    name: str | None = None
    role: str | None = None
    department: str | None = None
    college: str | None = None
    college_id: str | None = None
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
        if body.role != current.get("role"):
            fields["role_verified"] = False      # a new role needs the college to confirm it again
    if body.college_id or (body.college is not None and body.college.strip()):
        key = core.resolve_college(body.college_id, body.college)
        if key and key != current.get("institution"):
            fields["institution"] = key
            fields["verified"] = False          # a new college needs its own verification
            fields["role_verified"] = False
    if body.topics is not None:
        fields["topics"] = matching.as_items(body.topics, 12)
    if body.methods is not None:
        fields["methods"] = matching.as_items(body.methods, 12)
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


@router.post("/me/interests/{key}/confirm")
def confirm_interest(key: str):
    uid = me_uid()
    try:
        p = interests.confirm(uid, key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"person": _own_view(p), "learned": interests.learned(p)}


@router.post("/me/interests/{key}/hide")
def hide_interest(key: str):
    p = interests.hide(me_uid(), key)
    return {"person": _own_view(p), "learned": interests.learned(p)}


# ---------------------------------------------------------------- my published papers

@router.get("/me/scholar/search")
def scholar_search(q: str = "", orcid: str = ""):
    me = ensure_person(me_uid())
    inst = store().institution(me.get("institution", "")) if me.get("institution") else None
    name = q.strip() or me.get("name", "")
    if len(name) < 3 and not orcid:
        raise HTTPException(400, "Type your name as it appears on your papers.")
    try:
        return publications.find_authors(name, (inst or {}).get("openalex", ""), orcid)
    except publications.SourceError as exc:
        raise HTTPException(503, str(exc)) from None


class ScholarIn(BaseModel):
    author_id: str


@router.post("/me/scholar")
def claim_scholar(body: ScholarIn):
    """Show my published papers (my OpenAlex author profile). They also tell PaperSmith what I work on."""
    uid = me_uid()
    ensure_person(uid)
    aid = body.author_id.strip().rsplit("/", 1)[-1].upper()
    if not (aid.startswith("A") and aid[1:].isdigit()):
        raise HTTPException(400, "That isn't an OpenAlex author ID.")
    try:
        author, works = publications.author_works(aid)
    except publications.SourceError as exc:
        raise HTTPException(503, str(exc)) from None
    scholar = {"id": aid, "source": "OpenAlex", "name": author.get("display_name", ""), "orcid": author.get("orcid") or "",
               "works_count": author.get("works_count", 0), "cited_by": author.get("cited_by_count", 0),
               "recent": works[:12], "claimed_at": time.strftime("%Y-%m-%d")}
    p = store().upsert_person(uid, scholar=scholar)
    interests.learn_from_works(uid, works)
    p = store().get_person(uid) or p
    return {"person": _own_view(p), "learned": interests.learned(p)}


@router.delete("/me/scholar")
def remove_scholar():
    uid = me_uid()
    s = store()
    p = s.get_person(uid) or {}
    st = p.get("inferred") or {}
    for item in (st.get("items") or {}).values():
        item["sources"] = [x for x in item.get("sources", []) if x.get("type") != "publication"]
    p = s.upsert_person(uid, scholar={}, inferred=st)
    return {"person": _own_view(p), "learned": interests.learned(p)}


# ---------------------------------------------------------------- verification

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
    private = interests.private_interests(me)
    ranked = matching.rank(me, _college_people(me), store().network(uid), core.institution_name(me["institution"]),
                           private=private)
    ranked["reason"] = "" if (me.get("topics") or me.get("methods") or me.get("needs") or private) else "no_topics"
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
    if uid != me["uid"]:
        interests.explored(me["uid"], p.get("topics", []), "people you looked up", f"person:{uid}")
    return {"person": matching.public_view(p, core.institution_name(p.get("institution", ""))),
            "status": matching._status(uid, network),
            "credits": credits(uid, p.get("institution", "")) if p.get("institution") else {"helped": 0, "topics": []}}


class ConnectIn(BaseModel):
    to: str
    message: str = ""


@router.post("/connect")
def connect(body: ConnectIn):
    me = ensure_person(me_uid())
    other = store().get_person(body.to)
    if not other or body.to == me["uid"] or other.get("institution") != me.get("institution") or not matching.visible(other):
        raise HTTPException(404, "person not found")
    before = store().network(me["uid"])
    store().request(me["uid"], body.to, body.message)
    if not any(r["from"] == body.to for r in before["incoming"]):
        core.notify(body.to, "request", f"{me.get('name', 'Someone')} wants to connect" + (f": “{body.message[:80]}”" if body.message else "."),
                    link="/community/network", actor=me["uid"])
    else:
        core.notify(body.to, "connected", f"{me.get('name', 'Someone')} accepted your request.", link="/community/network", actor=me["uid"])
    return network()


@router.post("/requests/{from_uid}/accept")
def accept(from_uid: str):
    uid = me_uid()
    if not store().accept(from_uid, uid):
        raise HTTPException(404, "request not found")
    core.notify(from_uid, "connected", f"{display_name(uid)} accepted your request.", link="/community/network", actor=uid)
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
        return matching.public_view(p, core.institution_name(p.get("institution", ""))) if p else {"uid": other, "name": "Researcher"}

    return {"connections": [view(u) for u in raw["connections"]],
            "incoming": [{"person": view(r["from"]), "message": r.get("message", ""), "at": r.get("at", "")} for r in raw["incoming"]],
            "outgoing": [{"person": view(r["to"]), "message": r.get("message", ""), "at": r.get("at", "")} for r in raw["outgoing"]]}


@router.get("/graph")
def research_map():
    """My own map: me, the people I'm connected with, and the people suggested to me, on the topics we share.
    (The whole college graph is for its admins.)"""
    me = ensure_person(me_uid())
    if not me.get("institution"):
        return {"people": [], "topics": [], "links": [], "connections": [], "coauthors": [], "docs": [], "institution": "", "scope": "me"}
    s = store()
    people = {p["uid"]: p for p in _college_people(me)}
    net = s.network(me["uid"])
    ranked = matching.rank(me, list(people.values()), net, private=interests.private_interests(me), limit=8)
    keep = {me["uid"], *net["connections"], *(x["person"]["uid"] for x in ranked["mentors"][:5] + ranked["collaborators"][:6])}
    shown = [dict(p, published=True, visibility="community") if p["uid"] == me["uid"] else p
             for uid, p in people.items() if uid in keep]
    # only the edges that touch me: my map, not other people's
    edges = [(a, b) for a, b in s.connections_among(me["institution"]) if me["uid"] in (a, b)]
    data = matching.graph(shown, edges, core.institution_name(me["institution"]))
    return {**data, "institution": core.institution_name(me["institution"]), "me": me["uid"], "scope": "me"}


# ---------------------------------------------------------------- notifications

@router.get("/notifications")
def get_notifications():
    uid = me_uid()
    items = core.notifications(uid)
    return {"items": items, "unread": sum(1 for n in items if not n.get("read"))}


class ReadIn(BaseModel):
    ids: list[str] | None = None


@router.post("/notifications/read")
def read_notifications(body: ReadIn):
    uid = me_uid()
    core.mark_read(uid, body.ids)
    return get_notifications()
