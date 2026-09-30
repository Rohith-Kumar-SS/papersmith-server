"""/api/community/openings: the bridge between people who have a research problem and people who want one.

  * A professor (or anyone with a problem) posts a *project*: a problem statement, the skills it needs, what
    the work would produce. PaperSmith drafts it from a rough idea; the author edits and publishes.
  * A student posts *seeking*: looking for a mentor for their own idea, or for a problem to work on.
  * People apply (or offer to mentor). The author sees applicants ranked by fit, with the evidence behind each
    rank taken only from the applicant's public profile, published papers and answers. The model may summarise
    that evidence, never add to it.
  * Accepting someone connects you, and "Start a paper" opens a shared PaperSmith paper for the team, which is
    also recorded as an ongoing project of the college.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import storage
from ..models import Ledger, Project, ProjectMember, now_iso
from . import core, interests, matching
from .store import store

router = APIRouter(prefix="/api/community/openings")

KINDS = ["Research project", "Final-year project", "Thesis / dissertation", "Internship", "Paper collaboration",
         "Mentorship", "Funded position"]
TYPES = {"project", "seeking"}

DRAFT_SYSTEM = """You turn a researcher's rough idea into a clear research opening that students and colleagues in their college will read.

- title: at most 12 words.
- problem: 2 to 5 sentences in your own clear words (never a copy of the idea): the problem, why it matters, and what the work would involve. Use only what the idea says or directly implies; do not invent results, data, funding, deadlines or collaborators.
- topics: 2 to 5 research areas, 1 to 4 words each, in sentence case ("Cryptocurrency forensics").
- methods: up to 5 skills, techniques or tools the work needs ("Machine learning", "Python", "Graph analysis").
- prerequisites: up to 4 things the idea says an applicant must already know or have (empty if it says none).
- outcomes: up to 3 things the idea says the work should produce.
- kind: the closest kind from the list.
- slots: how many people the idea asks for (1 if it doesn't say).
- duration: how long, only if the idea says ("6 months"); otherwise "".
For a student looking for a mentor (TYPE seeking), write the problem as what they want to work on and what help they want. Return JSON."""

DRAFT_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["title", "problem", "topics", "methods", "prerequisites", "outcomes", "kind", "slots", "duration"],
                "properties": {"title": {"type": "string"}, "problem": {"type": "string"},
                               "topics": {"type": "array", "items": {"type": "string"}},
                               "methods": {"type": "array", "items": {"type": "string"}},
                               "prerequisites": {"type": "array", "items": {"type": "string"}},
                               "outcomes": {"type": "array", "items": {"type": "string"}},
                               "kind": {"type": "string", "enum": KINDS}, "slots": {"type": "integer"},
                               "duration": {"type": "string"}}}

FIT_SYSTEM = """You help a researcher choose between applicants to their research opening. Summarise how this applicant fits, in at most two sentences, using ONLY the evidence listed and their message. Name the strongest match and anything the opening needs that the evidence does not show. Never guess or add skills, grades or experience that are not listed. Return JSON."""

FIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary"],
              "properties": {"summary": {"type": "string"}}}


def _clean(text: str, n: int) -> str:
    return " ".join((text or "").split())[:n]


def _view(o: dict, me: dict, with_counts: bool = False) -> dict:
    s = store()
    owner = s.get_person(o["owner"]) or {"uid": o["owner"]}
    out = {k: o.get(k) for k in ("id", "type", "title", "problem", "prerequisites", "outcomes", "kind", "duration",
                                  "slots", "status", "created_at", "reading")}
    out.update({"topics": [t["name"] for t in o.get("topics", [])], "methods": [m["name"] for m in o.get("methods", [])],
                "owner": matching.public_view(owner, core.institution_name(o.get("institution", ""))) if owner.get("name") else
                {"uid": o["owner"], "name": "Researcher"}, "mine": o["owner"] == me["uid"]})
    apps = [a for a in s.list_docs("application", institution=o.get("institution", ""), limit=1000) if a.get("opening") == o["id"]]
    mine = next((a for a in apps if a.get("owner") == me["uid"]), None)
    out["my_application"] = {"status": mine["status"], "at": mine["created_at"]} if mine else None
    if with_counts or o["owner"] == me["uid"]:
        out["applicants"] = sum(1 for a in apps if a["status"] == "pending")
        out["accepted"] = sum(1 for a in apps if a["status"] == "accepted")
    return out


def _load(oid: str, me: dict) -> dict:
    o = store().get_doc("opening", oid)
    core.same_college(me, o)
    return o  # type: ignore[return-value]


# ---------------------------------------------------------------- browse

@router.get("")
def list_openings(type: str = "", mine: bool = False):
    me = core.ensure_person(core.me_uid())
    if not me.get("institution"):
        return {"openings": [], "for_you": []}
    s = store()
    items = s.list_docs("opening", institution=me["institution"], limit=300)
    if mine:
        return {"openings": [_view(o, me, with_counts=True) for o in items if o["owner"] == me["uid"]], "for_you": []}
    shown = [o for o in items if o.get("status", "open") == "open" and (not type or o.get("type") == type)]
    private = interests.private_interests(me)
    ranked = matching.rank_openings(me, shown, private, limit=12)
    reasons = {r["opening"]["id"]: r["reasons"] for r in ranked}
    return {"openings": [dict(_view(o, me), reasons=reasons.get(o["id"], [])) for o in shown],
            "for_you": [dict(_view(r["opening"], me), reasons=r["reasons"]) for r in ranked]}


@router.get("/applications/mine")
def my_applications():
    me = core.ensure_person(core.me_uid())
    s = store()
    out = []
    for a in s.list_docs("application", owner=me["uid"], limit=200):
        o = s.get_doc("opening", a["opening"])
        if o:
            out.append({"opening": _view(o, me), "status": a["status"], "at": a["created_at"], "message": a.get("message", "")})
    return out


@router.get("/{oid}")
def get_opening(oid: str):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        interests.explored(me["uid"], o.get("topics", []) + o.get("methods", []), "openings you looked at", f"opening:{oid}")
    return _view(o, me, with_counts=True)


# ---------------------------------------------------------------- post

class DraftIn(BaseModel):
    idea: str
    type: str = "project"


@router.post("/draft")
def draft(body: DraftIn):
    me = core.ensure_person(core.me_uid())
    idea = _clean(body.idea, 3000)
    if len(idea) < 20:
        raise HTTPException(400, "Describe the idea in a sentence or two first.")
    prompt = f"TYPE: {body.type}\nAUTHOR ROLE: {core.ROLES.get(me.get('role', ''), 'researcher')}\nKINDS: {', '.join(KINDS)}\n\nIDEA:\n{idea}"
    try:
        raw = core.ask(DRAFT_SYSTEM, prompt, DRAFT_SCHEMA, max_tokens=1400)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from None
    try:
        slots = max(1, min(20, int(raw.get("slots") or 1)))
    except (TypeError, ValueError):
        slots = 1
    out = {"title": _clean(raw.get("title", ""), 120), "problem": _clean(raw.get("problem", ""), 1500),
           "topics": [t["name"] for t in matching.as_items(matching.tidy(raw.get("topics", [])), 5)],
           "methods": [m["name"] for m in matching.as_items(matching.tidy(raw.get("methods", [])), 6)],
           "slots": slots, "duration": _clean(str(raw.get("duration") or ""), 60),
           "prerequisites": [_clean(x, 120) for x in raw.get("prerequisites", []) if str(x).strip()][:4],
           "outcomes": [_clean(x, 120) for x in raw.get("outcomes", []) if str(x).strip()][:3],
           "kind": raw.get("kind") if raw.get("kind") in KINDS else ("Mentorship" if body.type == "seeking" else "Research project")}
    out["reading"] = reading_for(me, out["topics"] + out["methods"])
    return out


def reading_for(me: dict, names: list[str]) -> list[dict]:
    """Background reading from the author's own claimed papers on the opening's topics (never invented)."""
    items = matching.as_items(names, 12)
    out = []
    for w in (me.get("scholar") or {}).get("recent", []):
        text = matching._stems(f"{w.get('title', '')} {' '.join(w.get('topics', []))}")
        if any(matching._stems(it["name"]) and matching._stems(it["name"]) <= text for it in items):
            out.append({"title": w["title"], "year": w.get("year", 0), "url": w.get("doi", "")})
    return out[:5]


class OpeningIn(BaseModel):
    type: str = "project"
    title: str
    problem: str
    topics: list[str] = []
    methods: list[str] = []
    prerequisites: list[str] = []
    outcomes: list[str] = []
    kind: str = "Research project"
    duration: str = ""
    slots: int = 1
    reading: list[dict] = []
    status: str = "open"


def _fields(body: OpeningIn) -> dict:
    if body.type not in TYPES:
        raise HTTPException(400, "type must be project or seeking")
    if not body.title.strip() or len(body.problem.strip()) < 20:
        raise HTTPException(400, "Give it a title and a short description.")
    if body.status not in ("open", "closed", "filled"):
        raise HTTPException(400, "status must be open, closed or filled")
    return {"type": body.type, "title": _clean(body.title, 120), "problem": _clean(body.problem, 2000),
            "topics": matching.as_items(body.topics, 6), "methods": matching.as_items(body.methods, 8),
            "prerequisites": [_clean(x, 120) for x in body.prerequisites if x.strip()][:6],
            "outcomes": [_clean(x, 120) for x in body.outcomes if x.strip()][:4],
            "kind": body.kind if body.kind in KINDS else "Research project", "duration": _clean(body.duration, 60),
            "slots": max(1, min(20, body.slots)), "status": body.status,
            "reading": [{"title": _clean(r.get("title", ""), 200), "year": r.get("year", 0), "url": _clean(r.get("url", ""), 300)}
                        for r in body.reading[:6] if r.get("title")]}


@router.post("")
def create(body: OpeningIn):
    me = core.ensure_person(core.me_uid())
    if not me.get("institution"):
        raise HTTPException(400, "Join your college first (Your profile).")
    s = store()
    if sum(1 for o in s.list_docs("opening", owner=me["uid"], limit=100) if o.get("status") == "open") >= 15:
        raise HTTPException(429, "You have 15 open posts. Close one to post another.")
    o = {"id": core.new_id("o"), "owner": me["uid"], "institution": me["institution"], "created_at": now_iso(), **_fields(body)}
    s.put_doc("opening", o)
    _announce(o, me)
    return _view(o, me, with_counts=True)


def _announce(o: dict, me: dict) -> None:
    """Tell the few people it fits best (not everyone: nobody wants a feed of every post)."""
    s = store()
    people = [p for p in s.members(o["institution"]) if p["uid"] != me["uid"] and matching.visible(p)]
    if o["type"] == "seeking":
        people = [p for p in people if p.get("role") in core.MENTOR_ROLES]
    topics = {t["name"] for t in o.get("topics", [])}
    scored = []
    for p in people:
        f = matching.fit(p, o)
        if f["score"] >= 0.34 or topics & set(f["covered"]):      # a shared research area is enough
            scored.append((f["score"], p))
    scored.sort(key=lambda x: -x[0])
    what = "is looking for a mentor" if o["type"] == "seeking" else "posted a research opening"
    for _, p in scored[:8]:
        core.notify(p["uid"], "opening", f"{me.get('name', 'Someone')} {what} that fits your work: “{o['title'][:80]}”",
                    link=f"/community/openings/{o['id']}", actor=me["uid"])


@router.put("/{oid}")
def update(oid: str, body: OpeningIn):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who posted it can change it.")
    o.update(_fields(body))
    store().put_doc("opening", o)
    return _view(o, me, with_counts=True)


@router.delete("/{oid}")
def delete(oid: str):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"] and not core.is_admin(me["uid"], me["institution"]):
        raise HTTPException(403, "Only the person who posted it (or a college admin) can remove it.")
    s = store()
    for a in s.list_docs("application", institution=o["institution"], limit=1000):
        if a.get("opening") == oid:
            s.delete_doc("application", a["id"])
    s.delete_doc("opening", oid)
    return {"ok": True}


# ---------------------------------------------------------------- apply

class ApplyIn(BaseModel):
    message: str = ""


@router.post("/{oid}/apply")
def apply(oid: str, body: ApplyIn):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] == me["uid"]:
        raise HTTPException(400, "This is your own post.")
    if o.get("status", "open") != "open":
        raise HTTPException(400, "This opening is no longer taking applications.")
    if not matching.visible(me):
        raise HTTPException(400, "Share your profile first (Your profile → Visibility), so they can see who you are.")
    s = store()
    aid = f"{oid}.{me['uid']}"
    existing = s.get_doc("application", aid)
    if existing and existing["status"] in ("pending", "accepted"):
        raise HTTPException(400, "You've already applied.")
    s.put_doc("application", {"id": aid, "opening": oid, "owner": me["uid"], "institution": o["institution"],
                              "message": _clean(body.message, 1200), "status": "pending", "created_at": now_iso()})
    verb = "offered to mentor you on" if o["type"] == "seeking" else "applied to"
    core.notify(o["owner"], "application", f"{me.get('name', 'Someone')} {verb} “{o['title'][:80]}”",
                link=f"/community/openings/{oid}", actor=me["uid"])
    interests.explored(me["uid"], o.get("topics", []) + o.get("methods", []), "openings you applied to", f"apply:{oid}")
    return _view(o, me)


@router.delete("/{oid}/apply")
def withdraw(oid: str):
    me = core.ensure_person(core.me_uid())
    _load(oid, me)
    s = store()
    a = s.get_doc("application", f"{oid}.{me['uid']}")
    if a and a["status"] == "pending":
        s.put_doc("application", {**a, "status": "withdrawn"})
    return {"ok": True}


def _credits(uid: str, institution: str) -> dict[str, int]:
    """Topics a person has given helpful answers on: {topic name: people helped}."""
    out: dict[str, int] = {}
    for q in store().list_docs("question", institution=institution, limit=500):
        for a in q.get("answers", []):
            if a["owner"] == uid and (a.get("helpful") or q.get("accepted") == a["id"]):
                for t in q.get("topics", []):
                    out[t["name"]] = out.get(t["name"], 0) + 1
    return out


@router.get("/{oid}/applicants")
def applicants(oid: str):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who posted it can see who applied.")
    s = store()
    net = s.network(me["uid"])
    out = []
    for a in s.list_docs("application", institution=o["institution"], limit=1000):
        if a.get("opening") != oid or a["status"] == "withdrawn":
            continue
        p = s.get_person(a["owner"])
        if not p:
            continue
        f = matching.fit(p, o, _credits(p["uid"], o["institution"]))
        if p["uid"] in net["connections"]:
            f["evidence"].append("Already connected with you")
        if p["uid"] in net["coauthors"]:
            f["evidence"].append("Has written a paper with you")
        out.append({"person": matching.public_view(p, core.institution_name(p.get("institution", ""))),
                    "message": a.get("message", ""), "status": a["status"], "at": a["created_at"],
                    "score": f["score"], "evidence": f["evidence"], "missing": f["missing"], "summary": a.get("summary", "")})
    out.sort(key=lambda x: (x["status"] != "pending", -x["score"]))
    return out


@router.post("/{oid}/applicants/{uid}/summary")
def summarise(oid: str, uid: str):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who posted it can see who applied.")
    s = store()
    a = s.get_doc("application", f"{oid}.{uid}")
    p = s.get_person(uid)
    if not a or not p:
        raise HTTPException(404, "application not found")
    f = matching.fit(p, o, _credits(uid, o["institution"]))
    prompt = (f"OPENING: {o['title']}\nNEEDS: {', '.join(t['name'] for t in o.get('topics', []) + o.get('methods', []))}\n"
              f"APPLICANT: {p.get('name')} ({core.ROLES.get(p.get('role', ''), 'researcher')}, {p.get('department', '')})\n"
              f"EVIDENCE:\n" + "\n".join(f"- {e}" for e in f["evidence"] or ["(none)"]) +
              f"\nNOT SHOWN BY THE EVIDENCE: {', '.join(f['missing']) or '(nothing)'}\nTHEIR MESSAGE: {a.get('message') or '(none)'}")
    try:
        raw = core.ask(FIT_SYSTEM, prompt, FIT_SCHEMA, max_tokens=400)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from None
    summary = _clean(raw.get("summary", ""), 400)
    s.put_doc("application", {**a, "summary": summary})
    return {"summary": summary}


class DecisionIn(BaseModel):
    accept: bool


@router.post("/{oid}/applicants/{uid}/decision")
def decide(oid: str, uid: str, body: DecisionIn):
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who posted it can decide.")
    s = store()
    a = s.get_doc("application", f"{oid}.{uid}")
    if not a or a["status"] == "withdrawn":
        raise HTTPException(404, "application not found")
    s.put_doc("application", {**a, "status": "accepted" if body.accept else "declined"})
    if body.accept:
        s.connect_now(me["uid"], uid)
        core.notify(uid, "accepted", f"{me.get('name', 'The author')} accepted you for “{o['title'][:80]}”. You're now connected.",
                    link=f"/community/openings/{oid}", actor=me["uid"])
        taken = sum(1 for x in s.list_docs("application", institution=o["institution"], limit=1000)
                    if x.get("opening") == oid and x["status"] == "accepted")
        if o["type"] == "project" and taken >= o.get("slots", 1):
            s.put_doc("opening", {**o, "status": "filled"})
    else:
        core.notify(uid, "declined", f"Your application to “{o['title'][:80]}” wasn't taken forward this time.",
                    link="/community/openings", actor=me["uid"])
    return applicants(oid)


@router.post("/{oid}/paper")
def start_paper(oid: str):
    """A shared PaperSmith paper for the opening's team, recorded as an ongoing project of the college."""
    me = core.ensure_person(core.me_uid())
    o = _load(oid, me)
    if o["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who posted it can start the paper.")
    s = store()
    team = [a["owner"] for a in s.list_docs("application", institution=o["institution"], limit=1000)
            if a.get("opening") == oid and a["status"] == "accepted"]
    if o.get("paper"):
        return {"paper": o["paper"]}
    from .. import assistant as A
    from .. import workspace

    project = Project(id=storage.new_id(), name=o["title"], ledger=Ledger(title=o["title"]), owner=me["uid"],
                      members=[ProjectMember(uid=u, name=core.display_name(u), role="author") for u in team])
    A.welcome(project, mentor=workspace.mentor_on())
    storage.save(project)
    s.add_coauthors([me["uid"], *team])
    s.put_doc("opening", {**o, "paper": project.id, "status": "filled" if o["type"] == "project" else o.get("status", "open")})
    s.put_doc("record", {"id": core.new_id("r"), "institution": o["institution"], "owner": me["uid"], "title": o["title"],
                         "summary": o["problem"][:600], "lead_uid": me["uid"], "lead_name": me.get("name", ""),
                         "members": [core.display_name(u) for u in team], "department": me.get("department", ""),
                         "status": "ongoing", "start": now_iso()[:10], "end": "", "funding": "", "topics": o.get("topics", []),
                         "source": "opening", "link": oid, "visibility": "college", "created_at": now_iso()})
    for u in team:
        core.notify(u, "paper", f"{me.get('name', 'Your mentor')} started a shared paper for “{o['title'][:80]}”.",
                    link=f"/{project.id}", actor=me["uid"])
    return {"paper": project.id}
