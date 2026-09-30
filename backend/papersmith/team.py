"""Writing a paper together: co-authors, reviewers, review comments, and who contributed what.

Every fact in a paper already has a receipt (the file or chat message it came from), and every file and
message records who added it, so the contribution record is computed, never self-reported."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import auth, storage
from .community import api as community_api
from .community.store import store as community_store
from .models import ClaimType, Project, ProjectMember, ReviewComment, now_iso

router = APIRouter(prefix="/api/projects/{pid}")

TYPE_GROUP = {ClaimType.method: "methods", ClaimType.dataset: "data", ClaimType.result: "results",
              ClaimType.comparison: "results", ClaimType.background: "background", ClaimType.gap: "background",
              ClaimType.objective: "aims", ClaimType.contribution: "aims", ClaimType.interpretation: "interpretation",
              ClaimType.limitation: "limitations", ClaimType.future_work: "future work"}


def _name(uid: str) -> str:
    return community_api.display_name(uid) if uid else "You"


def contributions(project: Project) -> list[dict]:
    """Per person: how many facts of which kind came from their files and messages."""
    by_msg = {m.id: m.author for m in project.messages if m.role == "user"}
    by_file = {f.id: f.uploaded_by for f in project.sources}
    people: dict[str, dict] = {}

    def entry(uid: str) -> dict:
        uid = uid or project.owner
        return people.setdefault(uid, {"uid": uid, "facts": 0, "kinds": {}, "files": 0, "messages": 0})

    for f in project.sources:
        entry(f.uploaded_by)["files"] += 1
    for m in project.messages:
        if m.role == "user" and not m.data.get("event") and not m.attachments:
            entry(m.author)["messages"] += 1
    for c in project.ledger.claims:
        src = c.sources[0] if c.sources else ""
        if src.startswith("chat:"):
            who = by_msg.get(src[5:], "")
        else:
            who = by_file.get(src.split(".")[0], "")
        e = entry(who)
        e["facts"] += 1
        group = TYPE_GROUP.get(c.type, "other")
        e["kinds"][group] = e["kinds"].get(group, 0) + 1
    out = []
    for e in people.values():
        kinds = sorted(e["kinds"].items(), key=lambda kv: -kv[1])
        out.append({**e, "name": _name(e["uid"]), "kinds": [k for k, _ in kinds]})
    return sorted(out, key=lambda e: -e["facts"])


def statement(project: Project) -> str:
    """The author-contribution statement for the export, or '' for a single-author paper."""
    if not project.members:
        return ""
    parts = []
    for c in contributions(project):
        if c["facts"]:
            kinds = ", ".join(c["kinds"][:3])
            parts.append(f"{c['name']} contributed {c['facts']} of the paper's facts ({kinds})")
    reviewers = [m.name for m in project.members if m.role == "reviewer"]
    text = "; ".join(parts) + "." if parts else ""
    if reviewers:
        text += f" Reviewed by {', '.join(reviewers)}."
    return text.strip()


def _role(pid: str) -> str:
    if not auth.enabled():
        return "owner"                      # the laptop version has one user
    return storage.access_of(pid, auth.current_user.get()) or ""


def _load(pid: str) -> Project:
    try:
        return storage.load(pid)
    except (KeyError, ValueError):
        raise HTTPException(404, "project not found") from None


def _team(project: Project, role: str) -> dict:
    return {
        "my_role": role,
        "owner": {"uid": project.owner, "name": _name(project.owner)},
        "members": [m.model_dump() for m in project.members],
        "contributions": contributions(project),
        "reviews": [r.model_dump() for r in project.reviews],
    }


@router.get("/team")
def get_team(pid: str):
    return _team(_load(pid), _role(pid))


class MemberIn(BaseModel):
    uid: str
    role: str = "author"


@router.post("/members")
def add_member(pid: str, body: MemberIn):
    role = _role(pid)
    if role != "owner":
        raise HTTPException(403, "Only the paper's owner can add people.")
    if body.role not in ("author", "reviewer"):
        raise HTTPException(400, "role must be author or reviewer")
    me = auth.current_user.get()
    if body.uid == me:
        raise HTTPException(400, "You already own this paper.")
    if body.uid not in community_store().network(me)["connections"]:
        raise HTTPException(400, "You can add people you're connected with in the community.")
    with storage.edit(pid) as p:
        p.members = [m for m in p.members if m.uid != body.uid]
        p.members.append(ProjectMember(uid=body.uid, name=_name(body.uid), role=body.role))  # type: ignore[arg-type]
        p.history.append({"at": now_iso(), "action": "member_added", "role": body.role})
        team = _team(p, role)
    if body.role == "author":
        community_store().add_coauthors([me, body.uid])
    return team


@router.delete("/members/{uid}")
def remove_member(pid: str, uid: str):
    role = _role(pid)
    if role != "owner" and uid != auth.current_user.get():
        raise HTTPException(403, "Only the owner can remove other people.")
    with storage.edit(pid) as p:
        p.members = [m for m in p.members if m.uid != uid]
        team = _team(p, role)
    return team


class ReviewIn(BaseModel):
    target: str = "general"
    text: str


@router.post("/reviews")
def add_review(pid: str, body: ReviewIn):
    if not body.text.strip():
        raise HTTPException(400, "empty comment")
    uid = auth.current_user.get()
    with storage.edit(pid) as p:
        nums = [int(r.id[1:]) for r in p.reviews if r.id[1:].isdigit()]
        p.reviews.append(ReviewComment(id=f"R{(max(nums) if nums else 0) + 1}", author=uid, author_name=_name(uid),
                                       target=body.target.strip()[:20] or "general", text=body.text.strip()[:2000]))
        team = _team(p, _role(pid))
    return team


class ReviewPatch(BaseModel):
    resolved: bool


@router.patch("/reviews/{rid}")
def resolve_review(pid: str, rid: str, body: ReviewPatch):
    if _role(pid) not in ("owner", "author"):
        raise HTTPException(403, "Only the paper's authors can resolve comments.")
    with storage.edit(pid) as p:
        r = next((x for x in p.reviews if x.id == rid), None)
        if not r:
            raise HTTPException(404, "comment not found")
        r.resolved = body.resolved
        team = _team(p, _role(pid))
    return team


@router.delete("/reviews/{rid}")
def delete_review(pid: str, rid: str):
    uid, role = auth.current_user.get(), _role(pid)
    with storage.edit(pid) as p:
        r = next((x for x in p.reviews if x.id == rid), None)
        if not r:
            raise HTTPException(404, "comment not found")
        if role != "owner" and r.author != uid:
            raise HTTPException(403, "You can delete your own comments.")
        p.reviews = [x for x in p.reviews if x.id != rid]
        team = _team(p, role)
    return team
