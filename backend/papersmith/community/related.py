"""'Already being researched': when a paper's topics are known, the people, openings and ongoing projects at the
writer's college that work on the same thing, so they can collaborate instead of duplicating work.

Only what others chose to share is used: their public profiles, their openings, and project records their
college keeps. Nobody else's papers are ever read.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import auth
from . import core, matching
from .store import store

router = APIRouter(prefix="/api/projects/{pid}")


def _about(items: list[dict], wanted: list[dict]) -> list[str]:
    return [w["name"] for w in wanted if any(matching.related_keys(w["key"], it["key"]) for it in items)]


def find(doc: dict, exclude: set[str] | None = None, limit: int = 6) -> dict:
    """{topics, people, openings, projects} at the paper's college that share its topics."""
    inst = doc.get("institution", "")
    wanted = doc.get("topics", []) + doc.get("methods", [])
    empty = {"topics": [t["name"] for t in doc.get("topics", [])], "methods": [m["name"] for m in doc.get("methods", [])],
             "people": [], "openings": [], "projects": []}
    if not inst or not wanted:
        return empty
    s = store()
    exclude = (exclude or set()) | set(doc.get("people", []))
    topic_keys = {t["key"] for t in doc.get("topics", [])}
    people = []
    for p in s.members(inst):
        if p["uid"] in exclude or not matching.visible(p):
            continue
        shared = _about(p.get("topics", []) + p.get("methods", []), wanted)
        # a shared method alone ('machine learning') is weak: need a shared topic, or two shared things
        if shared and (any(matching.related_keys(t["key"], x["key"]) for t in p.get("topics", []) for x in doc.get("topics", []))
                       or len(shared) >= 2):
            people.append((len(shared), p, shared))
    people.sort(key=lambda x: -x[0])
    name = core.institution_name(inst)
    openings = [o for o in s.list_docs("opening", institution=inst, limit=200)
                if o.get("status", "open") == "open" and o.get("owner") not in exclude
                and _about(o.get("topics", []) + o.get("methods", []), wanted)]
    projects = [r for r in s.list_docs("record", institution=inst, limit=300)
                if r.get("visibility", "college") == "college" and r.get("status") != "completed"
                and any(matching.related_keys(k, t["key"]) for k in topic_keys for t in r.get("topics", []))]
    return {**empty,
            "people": [{"person": matching.public_view(p, name), "shared": shared[:3]} for _, p, shared in people[:limit]],
            "openings": [{"id": o["id"], "title": o["title"], "type": o.get("type", "project"), "owner_name": core.display_name(o["owner"]),
                          "shared": _about(o.get("topics", []) + o.get("methods", []), wanted)[:3]} for o in openings[:limit]],
            "projects": [{"id": r["id"], "title": r["title"], "lead": r.get("lead_name", ""), "department": r.get("department", ""),
                          "status": r.get("status", "ongoing"), "shared": _about(r.get("topics", []), doc.get("topics", []))[:3]}
                         for r in projects[:limit]]}


def announce(doc: dict) -> None:
    """Tell the writer once per paper when their college already has people or projects on its topics."""
    s = store()
    found = find(doc)
    n_people, n_other = len(found["people"]), len(found["openings"]) + len(found["projects"])
    if not (n_people or n_other) or doc.get("announced"):
        return
    s.put_doc("paper_topics", {**doc, "announced": True})
    bits = []
    if n_people:
        bits.append(f"{n_people} {'person' if n_people == 1 else 'people'}")
    if n_other:
        bits.append(f"{n_other} open project{'s' if n_other != 1 else ''}")
    core.notify(doc["owner"], "related",
                f"{' and '.join(bits)} at your college already work on topics in “{doc.get('title', 'your paper')[:60]}”. "
                "You could team up instead of starting alone.", link=f"/{doc['id']}/team")


@router.get("/related")
def related(pid: str):
    if not auth.enabled():
        raise HTTPException(404, "The community is part of the hosted PaperSmith.")
    doc = store().get_doc("paper_topics", pid)
    if not doc:
        return {"ready": False, "topics": [], "methods": [], "people": [], "openings": [], "projects": []}
    return {"ready": True, **find(doc, exclude={auth.current_user.get()})}
