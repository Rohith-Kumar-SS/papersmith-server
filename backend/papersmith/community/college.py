"""/api/community/college: a college's own workspace, run by its admins. /api/community/owner: the people who run
this PaperSmith, who approve each college's first admin.

Admins see what members don't: the whole college graph, every member, analytics (what students look for and
whether any faculty covers it, departments that share topics but never talk), the college's research output
(OpenAlex, or Scopus with the college's own key) and its records of ongoing projects. Members see only their own
network and suggestions.
"""

from __future__ import annotations

import time
from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models import now_iso
from . import core, directory, matching, publications
from .store import store

router = APIRouter(prefix="/api/community")

STATUSES = ("proposed", "ongoing", "completed")


def _member_view(p: dict, inst_name: str) -> dict:
    """What an admin sees of a member: who they are always; their research only if they shared it."""
    base = {"uid": p["uid"], "name": p.get("name") or "Researcher", "role": p.get("role", ""),
            "role_label": core.ROLES.get(p.get("role", ""), ""), "department": p.get("department", ""),
            "verified": bool(p.get("verified")), "role_verified": bool(p.get("role_verified")),
            "published": matching.visible(p), "joined_at": p.get("joined_at", ""), "active_at": p.get("updated_at", "")}
    if matching.visible(p):
        base.update(topics=[t["name"] for t in p.get("topics", [])], methods=[m["name"] for m in p.get("methods", [])])
    return base


# ================================================================ becoming an admin

class AdminRequestIn(BaseModel):
    designation: str
    message: str = ""


@router.get("/college")
def my_college():
    """Any member: their college, its admins, and whether they are one (or have asked to be)."""
    me = core.ensure_person(core.me_uid())
    key = me.get("institution", "")
    inst = store().institution(key) if key else None
    if not inst:
        return {"institution": None, "is_admin": False, "is_owner": core.is_owner()}
    pending = next((r for r in store().list_docs("admin_request", institution=key, limit=100)
                    if r["owner"] == me["uid"] and r["status"] == "pending"), None)
    entry = directory.get(key)
    return {"institution": {"key": key, "name": inst["name"], "listed": bool(inst.get("listed")),
                            "place": entry["place"] if entry else "", "kind_label": directory.KINDS.get(inst.get("kind", ""), "")},
            "admins": [{"uid": u, "name": core.display_name(u)} for u in inst.get("admins", [])],
            "is_admin": core.is_admin(me["uid"], key), "is_owner": core.is_owner(), "request_pending": bool(pending)}


@router.post("/college/admin-request")
def request_admin(body: AdminRequestIn):
    me = core.ensure_person(core.me_uid())
    key = me.get("institution", "")
    if not key:
        raise HTTPException(400, "Join your college first.")
    if core.is_admin(me["uid"], key):
        raise HTTPException(400, "You're already an admin.")
    s = store()
    if any(r["owner"] == me["uid"] and r["status"] == "pending" for r in s.list_docs("admin_request", institution=key, limit=100)):
        raise HTTPException(400, "Your request is waiting for approval.")
    r = {"id": core.new_id("ar"), "owner": me["uid"], "institution": key, "designation": " ".join(body.designation.split())[:120],
         "message": body.message.strip()[:1000], "status": "pending", "at": now_iso(), "verified": bool(me.get("verified")),
         "email_domain": me.get("email_domain", "")}
    s.put_doc("admin_request", r)
    for admin in core.admins_of(key):          # a college with admins decides itself; else the owner does
        core.notify(admin, "admin_request", f"{me.get('name', 'Someone')} asked to become an admin of your college.",
                    link="/community/college/admins", actor=me["uid"])
    return {"ok": True}


def _decide(r: dict, approve: bool, by: str) -> None:
    s = store()
    s.put_doc("admin_request", {**r, "status": "approved" if approve else "declined", "decided_by": by, "decided_at": now_iso()})
    if approve:
        admins = core.admins_of(r["institution"])
        if r["owner"] not in admins:
            s.update_institution(r["institution"], admins=admins + [r["owner"]])
        core.notify(r["owner"], "admin", f"You're now an admin of {core.institution_name(r['institution'])}.",
                    link="/community/college")
    else:
        core.notify(r["owner"], "admin", "Your request to become a college admin wasn't approved.", link="/community")


# ================================================================ the owner

@router.get("/owner/overview")
def owner_overview():
    core.require_owner()
    s = store()
    colleges = sorted(s.all_institutions(), key=lambda i: -i["members"])
    requests = [r for r in s.list_docs("admin_request", limit=300) if r["status"] == "pending"]
    return {
        "requests": [{**r, "name": core.display_name(r["owner"]), "college": core.institution_name(r["institution"]),
                      "college_has_admins": bool(core.admins_of(r["institution"]))} for r in requests],
        "colleges": [{"key": i["key"], "name": i["name"], "members": i["members"], "listed": bool(i.get("listed")),
                      "admins": [{"uid": u, "name": core.display_name(u)} for u in i.get("admins", [])]} for i in colleges],
        "totals": {"colleges": len(colleges), "people": sum(i["members"] for i in colleges),
                   "unlisted": sum(1 for i in colleges if not i.get("listed"))},
    }


@router.post("/owner/requests/{rid}/{decision}")
def owner_decide(rid: str, decision: str):
    uid = core.require_owner()
    r = store().get_doc("admin_request", rid)
    if not r or r["status"] != "pending":
        raise HTTPException(404, "request not found")
    _decide(r, decision == "approve", uid)
    return owner_overview()


class MergeIn(BaseModel):
    into: str


@router.post("/owner/colleges/{key}/merge")
def owner_merge(key: str, body: MergeIn):
    """An unlisted college is really a listed one: move its members there."""
    core.require_owner()
    s = store()
    entry = directory.get(body.into)
    if not entry or not s.institution(key):
        raise HTTPException(404, "college not found")
    target = core.college_from_directory(entry)["key"]
    for p in s.members(key):
        s.upsert_person(p["uid"], institution=target, verified=False)
    old = s.institution(key) or {}
    if old.get("admins"):
        s.update_institution(target, admins=sorted(set(core.admins_of(target)) | set(old["admins"])))
    s.update_institution(key, merged_into=target, admins=[])
    core.forget_name(key)
    return owner_overview()


@router.post("/owner/colleges/{key}/approve")
def owner_approve_unlisted(key: str):
    core.require_owner()
    if not store().update_institution(key, listed=True, approved=True):
        raise HTTPException(404, "college not found")
    return owner_overview()


# ================================================================ the college's admins

@router.get("/college/requests")
def college_requests():
    _, key = core.require_admin()
    return [{**r, "name": core.display_name(r["owner"])} for r in store().list_docs("admin_request", institution=key, limit=100)
            if r["status"] == "pending"]


@router.post("/college/requests/{rid}/{decision}")
def college_decide(rid: str, decision: str):
    me, key = core.require_admin()
    r = store().get_doc("admin_request", rid)
    if not r or r["institution"] != key or r["status"] != "pending":
        raise HTTPException(404, "request not found")
    _decide(r, decision == "approve", me["uid"])
    return college_requests()


class AdminIn(BaseModel):
    uid: str


@router.post("/college/admins")
def add_admin(body: AdminIn):
    _, key = core.require_admin()
    s = store()
    p = s.get_person(body.uid)
    if not p or p.get("institution") != key:
        raise HTTPException(404, "That person isn't a member of your college.")
    admins = core.admins_of(key)
    if body.uid not in admins:
        s.update_institution(key, admins=admins + [body.uid])
        core.notify(body.uid, "admin", f"You're now an admin of {core.institution_name(key)}.", link="/community/college")
    return my_college()


@router.delete("/college/admins/{uid}")
def remove_admin(uid: str):
    me, key = core.require_admin()
    admins = core.admins_of(key)
    if uid not in admins:
        raise HTTPException(404, "not an admin")
    if admins == [uid] and not core.is_owner():
        raise HTTPException(400, "Add another admin before removing the last one.")
    store().update_institution(key, admins=[a for a in admins if a != uid])
    return my_college()


@router.get("/college/members")
def members():
    _, key = core.require_admin()
    name = core.institution_name(key)
    s = store()
    net_counts = Counter()
    for a, b in s.connections_among(key):
        net_counts[a] += 1
        net_counts[b] += 1
    admins = set(core.admins_of(key))
    out = [dict(_member_view(p, name), connections=net_counts[p["uid"]], admin=p["uid"] in admins) for p in s.members(key)]
    return sorted(out, key=lambda m: (m["role"] != "faculty", m["name"].lower()))


class RoleVerifyIn(BaseModel):
    verified: bool


@router.post("/college/members/{uid}/role-verify")
def verify_role(uid: str, body: RoleVerifyIn):
    """An admin confirms someone really is faculty / staff / a scholar here (roles are self-declared)."""
    _, key = core.require_admin()
    s = store()
    p = s.get_person(uid)
    if not p or p.get("institution") != key:
        raise HTTPException(404, "member not found")
    s.upsert_person(uid, role_verified=body.verified)
    if body.verified:
        core.notify(uid, "role", f"Your college confirmed your role ({core.ROLES.get(p.get('role', ''), 'member')}).")
    return members()


class DepartmentsIn(BaseModel):
    departments: list[str]


@router.get("/college/departments")
def get_departments():
    _, key = core.require_admin()
    return {"departments": core.departments(key), "custom": bool((store().institution(key) or {}).get("departments"))}


@router.put("/college/departments")
def put_departments(body: DepartmentsIn):
    _, key = core.require_admin()
    seen, clean = set(), []
    for d in body.departments:
        d = " ".join(d.split())[:80]
        if d and d.lower() not in seen:
            seen.add(d.lower())
            clean.append(d)
    store().update_institution(key, departments=clean[:150])
    return get_departments()


# ---------------------------------------------------------------- project records

class RecordIn(BaseModel):
    title: str
    summary: str = ""
    lead_name: str = ""
    lead_uid: str = ""
    members: list[str] = []
    department: str = ""
    status: str = "ongoing"
    start: str = ""
    end: str = ""
    funding: str = ""
    topics: list[str] = []
    visibility: str = "college"


def _record_fields(body: RecordIn) -> dict:
    if not body.title.strip():
        raise HTTPException(400, "Give the project a title.")
    if body.status not in STATUSES:
        raise HTTPException(400, "status must be proposed, ongoing or completed")
    return {"title": " ".join(body.title.split())[:200], "summary": body.summary.strip()[:2000],
            "lead_name": " ".join(body.lead_name.split())[:100], "lead_uid": body.lead_uid[:64],
            "members": [" ".join(m.split())[:100] for m in body.members if m.strip()][:30],
            "department": " ".join(body.department.split())[:80], "status": body.status, "start": body.start[:10],
            "end": body.end[:10], "funding": " ".join(body.funding.split())[:200], "topics": matching.as_items(body.topics, 8),
            "visibility": body.visibility if body.visibility in ("college", "admins") else "college"}


def _record_view(r: dict) -> dict:
    return {**{k: v for k, v in r.items() if k not in ("topics", "institution")}, "topics": [t["name"] for t in r.get("topics", [])]}


@router.get("/college/records")
def records():
    _, key = core.require_admin()
    return [_record_view(r) for r in store().list_docs("record", institution=key, limit=1000)]


@router.post("/college/records")
def add_record(body: RecordIn):
    me, key = core.require_admin()
    r = {"id": core.new_id("r"), "institution": key, "owner": me["uid"], "source": "manual", "created_at": now_iso(),
         **_record_fields(body)}
    store().put_doc("record", r)
    return _record_view(r)


@router.put("/college/records/{rid}")
def edit_record(rid: str, body: RecordIn):
    _, key = core.require_admin()
    s = store()
    r = s.get_doc("record", rid)
    if not r or r["institution"] != key:
        raise HTTPException(404, "record not found")
    r.update(_record_fields(body))
    s.put_doc("record", r)
    return _record_view(r)


@router.delete("/college/records/{rid}")
def delete_record(rid: str):
    _, key = core.require_admin()
    s = store()
    r = s.get_doc("record", rid)
    if not r or r["institution"] != key:
        raise HTTPException(404, "record not found")
    s.delete_doc("record", rid)
    return {"ok": True}


@router.get("/projects")
def college_projects():
    """Members: the college's ongoing projects that admins chose to share with the college."""
    me = core.ensure_person(core.me_uid())
    if not me.get("institution"):
        return []
    return [_record_view(r) for r in store().list_docs("record", institution=me["institution"], limit=500)
            if r.get("visibility", "college") == "college" and r.get("status") != "completed"]


# ---------------------------------------------------------------- research output

class SourcesIn(BaseModel):
    openalex: str | None = None
    scopus_afid: str | None = None
    scopus_api_key: str | None = None
    scopus_insttoken: str | None = None
    clear_scopus: bool = False


def _sources_view(inst: dict) -> dict:
    sc = inst.get("scopus") or {}
    return {"openalex": inst.get("openalex", ""), "ror": inst.get("ror", ""),
            "scopus": {"afid": sc.get("afid", ""), "has_key": bool(sc.get("api_key")), "has_insttoken": bool(sc.get("insttoken"))}}


@router.get("/college/sources")
def get_sources():
    _, key = core.require_admin()
    return _sources_view(store().institution(key) or {})


@router.put("/college/sources")
def put_sources(body: SourcesIn):
    """The college's OpenAlex ID, or its own Scopus key (never shown again after saving)."""
    _, key = core.require_admin()
    s = store()
    inst = s.institution(key) or {}
    fields: dict = {}
    if body.openalex is not None:
        oa = body.openalex.strip().rsplit("/", 1)[-1].upper()
        if oa and not (oa.startswith("I") and oa[1:].isdigit()):
            raise HTTPException(400, "An OpenAlex institution ID looks like I122964287.")
        fields["openalex"] = oa
    if body.clear_scopus:
        fields["scopus"] = {}
    elif body.scopus_afid is not None or body.scopus_api_key or body.scopus_insttoken:
        sc = dict(inst.get("scopus") or {})
        if body.scopus_afid is not None:
            afid = body.scopus_afid.strip()
            if afid and not afid.isdigit():
                raise HTTPException(400, "A Scopus affiliation ID is a number, like 60014340.")
            sc["afid"] = afid
        if body.scopus_api_key:
            sc["api_key"] = body.scopus_api_key.strip()[:100]
        if body.scopus_insttoken:
            sc["insttoken"] = body.scopus_insttoken.strip()[:200]
        fields["scopus"] = sc
    if fields:
        inst = s.update_institution(key, **fields) or inst
        s.delete_doc("pubcache", key)          # the next view fetches from the new source
    return _sources_view(inst)


@router.get("/college/publications")
def college_publications(refresh: bool = False):
    _, key = core.require_admin()
    inst = store().institution(key) or {"key": key}
    try:
        return publications.college_output(inst, refresh=refresh)
    except publications.SourceError as exc:
        raise HTTPException(503, str(exc)) from None


# ---------------------------------------------------------------- the college graph (admins only)

@router.get("/college/graph")
def college_graph():
    me, key = core.require_admin()
    s = store()
    docs = [{**o, "kind": "opening"} for o in s.list_docs("opening", institution=key, limit=200) if o.get("status") == "open"]
    docs += [{**r, "kind": "record"} for r in s.list_docs("record", institution=key, limit=200) if r.get("status") != "completed"]
    data = matching.graph(s.members(key), s.connections_among(key), core.institution_name(key),
                          coauthors=s.coauthors_among(key), docs=docs, max_topics=120)
    return {**data, "institution": core.institution_name(key), "me": me["uid"], "scope": "college"}


# ---------------------------------------------------------------- analytics

def analytics(key: str) -> dict:
    s = store()
    people = s.members(key)
    shared = [p for p in people if matching.visible(p)]
    by_role = Counter(p.get("role") or "unknown" for p in people)
    by_dept = Counter(p.get("department") or "Not given" for p in people)
    month_ago = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))
    connections = s.connections_among(key)
    degree = Counter()
    for a, b in connections:
        degree[a] += 1
        degree[b] += 1

    topic_names: dict[str, str] = {}
    supply, student_interest = Counter(), Counter()
    for p in shared:
        for t in p.get("topics", []) + p.get("methods", []):
            topic_names[t["key"]] = t["name"]
            if p.get("role") in ("faculty", "researcher"):
                supply[t["key"]] += 1
            if p.get("role") in core.STUDENT_ROLES:
                student_interest[t["key"]] += 1
    top_topics = Counter()
    for p in shared:
        top_topics.update({t["key"] for t in p.get("topics", [])})

    # what students ask for: their 'looking for' lines, their seeking posts, their questions
    demand = Counter()
    vocab = dict(topic_names)
    openings = s.list_docs("opening", institution=key, limit=500)
    questions = s.list_docs("question", institution=key, limit=500)
    for o in openings:
        for t in o.get("topics", []) + o.get("methods", []):
            vocab.setdefault(t["key"], t["name"])
    for q in questions:
        for t in q.get("topics", []) + q.get("methods", []):
            vocab.setdefault(t["key"], t["name"])
    for p in shared:
        if p.get("role") in core.STUDENT_ROLES:
            for nd in p.get("needs", []):
                demand.update(matching.need_matches(nd["text"], vocab))
    for o in openings:
        if o.get("type") == "seeking":
            demand.update(t["key"] for t in o.get("topics", []) + o.get("methods", []))
    for q in questions:
        demand.update(t["key"] for t in q.get("topics", []) + q.get("methods", []))
    unmet = [{"topic": vocab[k], "asked": n} for k, n in demand.most_common() if supply[k] == 0][:8]
    untapped = [{"topic": topic_names[k], "faculty": n} for k, n in supply.most_common() if student_interest[k] == 0 and demand[k] == 0][:8]

    # departments that share topics but have no connections between them
    dept_topics: dict[str, set[str]] = {}
    dept_of = {p["uid"]: p.get("department") or "" for p in people}
    for p in shared:
        if p.get("department"):
            dept_topics.setdefault(p["department"], set()).update(t["key"] for t in p.get("topics", []))
    linked = Counter()
    for a, b in connections:
        da, db = dept_of.get(a, ""), dept_of.get(b, "")
        if da and db and da != db:
            linked[tuple(sorted((da, db)))] += 1
    bridges = []
    depts = sorted(dept_topics)
    for i, a in enumerate(depts):
        for b in depts[i + 1:]:
            common = dept_topics[a] & dept_topics[b]
            if common and not linked[(a, b)]:
                bridges.append({"departments": [a, b], "shared": [topic_names[k] for k in sorted(common)][:3], "size": len(common)})
    bridges.sort(key=lambda x: -x["size"])

    apps = s.list_docs("application", institution=key, limit=2000)
    records_ = s.list_docs("record", institution=key, limit=1000)
    answered = sum(1 for q in questions if q.get("status") == "answered" or q.get("answers"))
    pub = s.get_doc("pubcache", key) or {}
    return {
        "members": {"total": len(people), "shared_profiles": len(shared), "verified": sum(1 for p in people if p.get("verified")),
                    "active_30d": sum(1 for p in people if (p.get("updated_at") or "") >= month_ago),
                    "by_role": [{"role": core.ROLES.get(r, "Not given"), "count": n} for r, n in by_role.most_common()],
                    "by_department": [{"department": d, "count": n} for d, n in by_dept.most_common(12)]},
        "network": {"connections": len(connections), "isolated": sum(1 for p in shared if not degree[p["uid"]]),
                    "isolated_people": [{"uid": p["uid"], "name": p.get("name", "")} for p in shared if not degree[p["uid"]]][:12]},
        "topics": [{"topic": topic_names[k], "people": n} for k, n in top_topics.most_common(15)],
        "unmet_demand": unmet, "untapped_expertise": untapped, "bridges": bridges[:6],
        "openings": {"open": sum(1 for o in openings if o.get("status") == "open" and o.get("type") == "project"),
                     "seeking": sum(1 for o in openings if o.get("status") == "open" and o.get("type") == "seeking"),
                     "applications": sum(1 for a in apps if a["status"] != "withdrawn"),
                     "accepted": sum(1 for a in apps if a["status"] == "accepted"),
                     "filled": sum(1 for o in openings if o.get("status") == "filled")},
        "questions": {"asked": len(questions), "answered": answered},
        "records": {st: sum(1 for r in records_ if r.get("status") == st) for st in STATUSES},
        "publications": {"source": pub.get("source", ""), "total": pub.get("total", 0), "by_year": pub.get("by_year", [])[-6:],
                         "topics": [t["name"] for t in pub.get("topics", [])[:6]]} if pub else None,
    }


@router.get("/college/overview")
def overview():
    _, key = core.require_admin()
    inst = store().institution(key) or {}
    return {"institution": core.institution_name(key), "analytics": analytics(key), "insights": inst.get("insights")}


def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def insights_from(a: dict) -> list[dict]:
    """Observations and next steps, each computed from one number on the dashboard (rules, not a model: every
    sentence can be traced to the figure it quotes)."""
    out: list[dict] = []
    m, net = a["members"], a["network"]
    if a["unmet_demand"]:
        top = a["unmet_demand"][:3]
        out.append({"observation": "Students are asking about " + ", ".join(f"{x['topic']} ({x['asked']})" for x in top)
                    + ", and no faculty member at the college lists these topics.",
                    "action": "Ask faculty in nearby areas to post a problem statement on them, or invite a mentor from another college."})
    for b in a["bridges"][:2]:
        out.append({"observation": f"{b['departments'][0]} and {b['departments'][1]} both work on {', '.join(b['shared'])}, "
                                   "but nobody in one is connected to anyone in the other.",
                    "action": f"Hold a joint seminar for {b['departments'][0]} and {b['departments'][1]} on {b['shared'][0]}."})
    if a["untapped_expertise"]:
        top = [x["topic"] for x in a["untapped_expertise"][:3]]
        out.append({"observation": f"Faculty expertise in {', '.join(top)} has no student interest, request or question yet.",
                    "action": "Ask those faculty to post problem statements; PaperSmith tells the students whose work fits."})
    if m["shared_profiles"] and net["isolated"] and net["isolated"] / m["shared_profiles"] >= 0.3:
        out.append({"observation": f"{_n(net['isolated'], 'member', 'members')} of the {m['shared_profiles']} who shared a profile "
                                   "have no connections yet.",
                    "action": "Point them to For you, where each suggestion says why two people should talk."})
    hidden = m["total"] - m["shared_profiles"]
    if m["total"] and hidden / m["total"] >= 0.3:
        out.append({"observation": f"{_n(hidden, 'member has', 'members have')} not shared a profile, so nobody can find them.",
                    "action": "Remind members that only topics and methods are shared; papers and results stay private."})
    o = a["openings"]
    if o["open"] + o["seeking"] and not o["applications"]:
        out.append({"observation": f"{_n(o['open'] + o['seeking'], 'open post has', 'open posts have')} no applications yet.",
                    "action": "Share the Openings page in department groups, or ask authors to add the skills they need."})
    q = a["questions"]
    if q["asked"] and q["asked"] - q["answered"] >= 2:
        out.append({"observation": f"{q['asked'] - q['answered']} of {q['asked']} questions have no answer.",
                    "action": "Encourage faculty to answer one question a week: answers are credited on their profile."})
    pub = a.get("publications")
    if pub and len(pub.get("by_year", [])) >= 3:
        years = [y for y in pub["by_year"] if y["year"] < time.gmtime().tm_year]
        if len(years) >= 2 and years[-2]["count"]:
            change = (years[-1]["count"] - years[-2]["count"]) / years[-2]["count"]
            if abs(change) >= 0.1:
                out.append({"observation": f"Papers went {'up' if change > 0 else 'down'} {abs(change):.0%} from {years[-2]['year']} "
                                           f"({years[-2]['count']}) to {years[-1]['year']} ({years[-1]['count']}), per {pub['source']}.",
                            "action": "Compare with the departments' ongoing projects under Projects to see where output is changing."})
        community = {t["topic"].lower() for t in a["topics"]}
        missing = [t for t in pub.get("topics", []) if t.lower() not in community][:3]
        if missing and a["topics"]:
            out.append({"observation": f"The college publishes on {', '.join(missing)}, but no member lists it on their profile yet.",
                        "action": "Invite the authors of those papers to join, so students can find them."})
    return out[:6]


@router.post("/college/insights")
def insights():
    _, key = core.require_admin()
    out = {"items": insights_from(analytics(key)), "at": now_iso()}
    store().update_institution(key, insights=out)
    return out
